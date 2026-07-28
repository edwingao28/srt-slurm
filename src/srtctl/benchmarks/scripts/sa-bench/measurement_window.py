# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Formal measurement-window records for SA-Bench.

Written by the benchmark process so a power consumer can join watts to exactly
the requests that were measured. Warmup never writes a window: the writer is
created only when the run saves a result and srtctl injected
``SRT_MEASUREMENT_WINDOW_DIR``.

Standalone by design — this module is mounted into the benchmark container and
must not import srtctl.
"""

import json
import os
import tempfile

SCHEMA_VERSION = 1
BENCHMARK_TYPE = "sa-bench"
CLOCK_SOURCE = "head_node_unix_clock"
WINDOW_DIR_ENV = "SRT_MEASUREMENT_WINDOW_DIR"
CONTAINER_LOG_DIR = "/logs"

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".{}.".format(os.path.basename(path)), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


class MeasurementWindow:
    """One formal window file for one measured concurrency point."""

    def __init__(self, path, result_path, concurrency):
        self.path = path
        self.result_path = result_path
        self.concurrency = concurrency

    @classmethod
    def create(
        cls,
        *,
        save_result,
        result_dir,
        result_filename,
        concurrency,
        window_dir=None,
        log_root=None,
    ):
        """Return a writer, or ``None`` when this run must not record a window.

        Warmup inherits the environment variable but omits ``--save-result``,
        so it is a strict no-op.
        """
        if not save_result or not result_filename:
            return None

        log_root = CONTAINER_LOG_DIR if log_root is None else log_root
        window_dir = os.environ.get(WINDOW_DIR_ENV) if window_dir is None else window_dir
        if not window_dir or not os.path.isdir(window_dir):
            return None

        result_full = os.path.join(result_dir or "", result_filename)
        result_path = os.path.relpath(os.path.abspath(result_full), os.path.abspath(log_root))
        if result_path.startswith(".."):
            return None

        stem = os.path.splitext(os.path.basename(result_filename))[0]
        return cls(os.path.join(window_dir, stem + ".json"), result_path.replace(os.sep, "/"), concurrency)

    def _write(self, status, start_unix, end_unix, duration, reason):
        _atomic_write_json(
            self.path,
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark_type": BENCHMARK_TYPE,
                "result_path": self.result_path,
                "concurrency": self.concurrency,
                "benchmark_start_time_unix": start_unix,
                "benchmark_end_time_unix": end_unix,
                "duration": duration,
                "clock_source": CLOCK_SOURCE,
                "status": status,
                "reason": reason,
            },
        )

    def mark_running(self, start_unix):
        """Record the formal start before the first measured request is scheduled."""
        self._write(STATUS_RUNNING, start_unix, None, None, None)

    def mark_completed(self, *, start_unix, end_unix, duration):
        """Publish the boundary the saved result was computed from."""
        self._write(STATUS_COMPLETED, start_unix, end_unix, duration, None)

    def mark_failed(self, *, start_unix, end_unix, duration, reason):
        """Keep a trustworthy boundary that produced no publishable result."""
        self._write(STATUS_FAILED, start_unix, end_unix, duration, reason)
