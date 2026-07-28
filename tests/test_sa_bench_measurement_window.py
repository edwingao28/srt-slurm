# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench formal measurement window and its power-coverage audit."""

import argparse
import asyncio
import contextlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch

import pytest

from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.core.power.contract import MAX_SAMPLE_GAP_SECONDS, WINDOWS_DIRNAME, Reason
from srtctl.core.power.manifest import ExpectedWindow
from srtctl.core.power.samples import SampleRow, derive_observed_devices
from srtctl.core.power.windows import convert_running_windows, validate_expected_windows
from srtctl.core.schema import (
    BenchmarkConfig,
    ModelConfig,
    ResourceConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)

SA_BENCH_DIR = Path(__file__).resolve().parents[1] / "src/srtctl/benchmarks/scripts/sa-bench"


def _benchmark_harness(tmp_path, *, provider="dcgm-power", enabled=True):
    telemetry = TelemetryConfig(
        enabled=enabled,
        provider=provider,
        default_frequency=1.0,
        storage_subdir="power",
        container_image="scraper" if provider == "scraper" else None,
        dcgm_exporter=TelemetryExporterConfig(container_image="dcgm-exporter", port=9401),
        node_exporter=TelemetryExporterConfig(container_image="node-exporter", port=9101)
        if provider == "scraper"
        else None,
    )
    harness = BenchmarkStageMixin()
    harness.config = SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/image", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb200"),
        benchmark=BenchmarkConfig(type="sa-bench", concurrencies=[4], isl=8192, osl=1024),
        telemetry=telemetry,
    )
    harness.runtime = MagicMock()
    harness.runtime.log_dir = tmp_path
    harness.runtime.container_mounts = {tmp_path: Path("/logs")}
    return harness


def _import_sa_bench_module(module_name):
    """Import a script that lives beside its siblings in the bench container."""
    sys.path.insert(0, str(SA_BENCH_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SA_BENCH_DIR))


def _load_measurement_window():
    spec = importlib.util.spec_from_file_location("sa_bench_measurement_window", SA_BENCH_DIR / "measurement_window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measurement_window = _load_measurement_window()
MeasurementWindow = measurement_window.MeasurementWindow

RESULT_STEM = "results_concurrency_4_gpus_8_ctx_4_gen_4"
RESULT_SUBDIR = "sa-bench_isl_8192_osl_1024"


@pytest.fixture
def logs(tmp_path):
    """A container ``/logs`` mount holding both results and power artifacts."""
    (tmp_path / RESULT_SUBDIR).mkdir()
    (tmp_path / "power" / WINDOWS_DIRNAME).mkdir(parents=True)
    return tmp_path


def _create(logs, *, save_result=True, window_dir=None, result_filename=f"{RESULT_STEM}.json"):
    return MeasurementWindow.create(
        save_result=save_result,
        window_dir=str(logs / "power" / WINDOWS_DIRNAME) if window_dir is None else window_dir,
        result_dir=str(logs / RESULT_SUBDIR),
        result_filename=result_filename,
        concurrency=4,
        log_root=str(logs),
    )


def _sa_bench_args(logs):
    """The exact argument shape bench.sh passes for a formal run."""
    return argparse.Namespace(
        backend="dynamo",
        model="model",
        served_model_name="model",
        tokenizer="/model",
        tokenizer_mode="auto",
        base_url=None,
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        trust_remote_code=True,
        custom_tokenizer=None,
        use_chat_template=False,
        dataset_name="random",
        dataset_path=None,
        dataset=None,
        num_prompts=4,
        random_prefix_len=0,
        random_input_len=8192,
        random_output_len=1024,
        random_range_ratio=1.0,
        random_num_workers=0,
        seed=0,
        logprobs=None,
        best_of=1,
        request_rate=float("inf"),
        burstiness=1.0,
        disable_tqdm=True,
        profile=False,
        percentile_metrics="ttft",
        metric_percentiles="99",
        ignore_eos=True,
        goodput=None,
        max_concurrency=4,
        lora_modules=None,
        slow_down_servers=None,
        slow_down_sleep_time=1.0,
        slow_down_wait_time=60.0,
        reuse_http_connections=False,
        save_result=True,
        result_dir=str(logs / RESULT_SUBDIR),
        result_filename=f"{RESULT_STEM}.json",
        metadata=None,
    )


def _write_result(logs, *, start, end, duration, stem=RESULT_STEM):
    path = logs / RESULT_SUBDIR / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "duration": duration,
                "benchmark_start_time_unix": start,
                "benchmark_end_time_unix": end,
                "completed": 40,
            }
        )
    )
    return path


def _window_json(logs, stem=RESULT_STEM):
    return json.loads((logs / "power" / WINDOWS_DIRNAME / f"{stem}.json").read_text())


def _samples(start, end, *, step=1.0, devices=(("node-a", 0, "GPU-a0"),), pad=None):
    rows = []
    seq = 0
    pad = max(2.0, step) if pad is None else pad
    timestamp = start - pad
    while timestamp <= end + pad:
        for hostname, gpu_index, uuid in devices:
            rows.append(SampleRow(timestamp, seq, hostname, gpu_index, uuid, 400.0))
        seq += 1
        timestamp += step
    return derive_observed_devices(rows)


def _validate(logs, observed, expected=(("sa-bench", 4),), errors=None):
    return validate_expected_windows(
        power_dir=logs / "power",
        result_root=logs,
        expected_windows=[ExpectedWindow(bt, c) for bt, c in expected],
        expected_device_keys={device.key for device in observed},
        observed_devices=observed,
        artifact_errors=errors if errors is not None else [],
    )


class TestWindowWriterActivation:
    def test_warmup_without_save_result_writes_nothing(self, logs):
        assert _create(logs, save_result=False) is None
        assert list((logs / "power" / WINDOWS_DIRNAME).iterdir()) == []

    def test_absent_window_dir_env_writes_nothing(self, logs):
        assert _create(logs, window_dir="") is None
        assert _create(logs, window_dir=None if False else str(logs / "nope")) is None

    def test_missing_result_filename_writes_nothing(self, logs):
        assert _create(logs, result_filename=None) is None

    def test_formal_run_creates_a_running_window_first(self, logs):
        window = _create(logs)

        window.mark_running(1785168100.0)

        payload = _window_json(logs)
        assert payload["schema_version"] == 1
        assert payload["benchmark_type"] == "sa-bench"
        assert payload["concurrency"] == 4
        assert payload["clock_source"] == "head_node_unix_clock"
        assert payload["status"] == "running"
        assert payload["benchmark_start_time_unix"] == 1785168100.0
        assert payload["benchmark_end_time_unix"] is None
        assert payload["duration"] is None
        assert payload["reason"] is None
        assert payload["result_path"] == f"{RESULT_SUBDIR}/{RESULT_STEM}.json"


class TestWindowStates:
    def test_completed_window_shape(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        payload = _window_json(logs)
        assert payload["status"] == "completed"
        assert payload["benchmark_end_time_unix"] == 1785168120.0
        assert payload["duration"] == 20.0
        assert payload["reason"] is None

    def test_failed_window_keeps_the_boundary_and_a_reason(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        window.mark_failed(
            start_unix=1785168100.0, end_unix=1785168110.0, duration=10.0, reason="RuntimeError: upstream reset"
        )

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] == 10.0
        assert payload["reason"] == "RuntimeError: upstream reset"

    def test_atomic_replacement_leaves_no_partial_file(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)
        window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        assert [p.name for p in (logs / "power" / WINDOWS_DIRNAME).iterdir()] == [f"{RESULT_STEM}.json"]

    def test_multiple_concurrencies_do_not_overwrite_each_other(self, logs):
        for concurrency, stem in ((4, RESULT_STEM), (16, "results_concurrency_16_gpus_8_ctx_4_gen_4")):
            window = MeasurementWindow.create(
                save_result=True,
                window_dir=str(logs / "power" / WINDOWS_DIRNAME),
                result_dir=str(logs / RESULT_SUBDIR),
                result_filename=f"{stem}.json",
                concurrency=concurrency,
                log_root=str(logs),
            )
            window.mark_running(1785168100.0)
            window.mark_completed(start_unix=1785168100.0, end_unix=1785168120.0, duration=20.0)

        assert len(list((logs / "power" / WINDOWS_DIRNAME).iterdir())) == 2
        assert _window_json(logs)["concurrency"] == 4

    def test_orchestrator_converts_running_windows_to_interrupted(self, logs):
        window = _create(logs)
        window.mark_running(1785168100.0)

        convert_running_windows(logs / "power" / WINDOWS_DIRNAME, reason="benchmark child terminated")

        payload = _window_json(logs)
        assert payload["status"] == "interrupted"
        assert payload["benchmark_end_time_unix"] is None
        assert payload["duration"] is None
        assert payload["reason"] == "benchmark child terminated"


class TestCoverageValidation:
    def _completed(self, logs, *, start=1000.0, end=1020.0, duration=20.0, result_duration=None):
        window = _create(logs)
        window.mark_running(start)
        window.mark_completed(start_unix=start, end_unix=end, duration=duration)
        _write_result(logs, start=start, end=end, duration=result_duration if result_duration is not None else duration)
        return start, end

    def test_bracketed_window_with_small_gaps_is_valid(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end))

        assert len(rows) == 1
        assert rows[0].power_coverage_valid is True
        assert rows[0].reason_codes == ()
        assert rows[0].window_file == f"{WINDOWS_DIRNAME}/{RESULT_STEM}.json"
        assert rows[0].per_device_max_sample_gap_seconds["node-a/GPU-a0"] == pytest.approx(1.0)

    def test_missing_expected_window_is_never_vacuously_valid(self, logs):
        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert rows[0].window_file is None
        assert Reason.MEASUREMENT_WINDOW_MISSING in rows[0].reason_codes
        assert rows[0].per_device_max_sample_gap_seconds == {}

    def test_device_without_bracketing_samples_is_invalid(self, logs):
        start, end = self._completed(logs)
        late = _samples(start + 5, end)

        rows = _validate(logs, late)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_NOT_BRACKETED in rows[0].reason_codes

    def test_gap_exactly_at_the_threshold_passes(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end, step=MAX_SAMPLE_GAP_SECONDS))

        assert rows[0].power_coverage_valid is True

    def test_gap_above_the_threshold_fails(self, logs):
        start, end = self._completed(logs)

        rows = _validate(logs, _samples(start, end, step=MAX_SAMPLE_GAP_SECONDS + 0.5))

        assert rows[0].power_coverage_valid is False
        assert Reason.SAMPLE_GAP_EXCEEDED in rows[0].reason_codes

    def test_result_timing_mismatch_is_reported(self, logs):
        start, end = self._completed(logs, result_duration=19.0)

        rows = _validate(logs, _samples(start, end))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH in rows[0].reason_codes

    def test_wall_clock_disagreeing_with_monotonic_duration_is_reported(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        window.mark_completed(start_unix=1000.0, end_unix=1200.0, duration=20.0)
        _write_result(logs, start=1000.0, end=1200.0, duration=20.0)

        rows = _validate(logs, _samples(1000.0, 1200.0, step=1.0))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH in rows[0].reason_codes

    @pytest.mark.parametrize(
        ("duration", "valid"),
        [
            (19.5, True),  # exactly max(0.5s, 1% of 20s) = 0.5s of skew
            (19.49, False),
        ],
    )
    def test_clock_tolerance_boundary(self, logs, duration, valid):
        window = _create(logs)
        window.mark_running(1000.0)
        window.mark_completed(start_unix=1000.0, end_unix=1020.0, duration=duration)
        _write_result(logs, start=1000.0, end=1020.0, duration=duration)

        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is valid
        if not valid:
            assert Reason.MEASUREMENT_WINDOW_CLOCK_MISMATCH in rows[0].reason_codes

    def test_missing_result_file_is_reported(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        window.mark_completed(start_unix=1000.0, end_unix=1020.0, duration=20.0)

        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISSING in rows[0].reason_codes

    def test_non_object_result_json_is_reported(self, logs):
        """Valid JSON that is not an object must not reach ``result.get()``."""
        start, end = self._completed(logs)
        (logs / RESULT_SUBDIR / f"{RESULT_STEM}.json").write_text("[]")

        rows = _validate(logs, _samples(start, end))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_MISMATCH in rows[0].reason_codes

    def test_uuid_change_invalidates_the_window_verdict(self, logs):
        start, end = self._completed(logs)
        observed = _samples(start, end)
        swapped = derive_observed_devices(
            [
                SampleRow(start - 1.0, 0, "node-a", 0, "GPU-a0", 400.0),
                SampleRow(start + 1.0, 1, "node-a", 0, "GPU-swapped", 400.0),
                SampleRow(end + 1.0, 2, "node-a", 0, "GPU-swapped", 400.0),
            ]
        )
        assert observed  # the happy-path fixture is otherwise identical

        rows = _validate(logs, swapped)

        assert rows[0].power_coverage_valid is False
        assert Reason.GPU_UUID_CHANGED in rows[0].reason_codes

    def test_interrupted_window_requires_a_reason(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        payload = json.loads(path.read_text())
        payload["status"] = "interrupted"  # orchestrator always records why
        path.write_text(json.dumps(payload))
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes

    def test_running_window_with_a_reason_is_malformed(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        payload = json.loads(path.read_text())
        payload["reason"] = "premature"  # nothing has gone wrong yet
        path.write_text(json.dumps(payload))
        errors = []

        _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes

    def test_running_window_is_incomplete(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)

        rows = _validate(logs, _samples(1000.0, 1020.0))

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_INCOMPLETE in rows[0].reason_codes

    def test_one_invalid_window_among_several_is_isolated(self, logs):
        self._completed(logs)
        stale_stem = "results_concurrency_16_gpus_8_ctx_4_gen_4"
        other = MeasurementWindow.create(
            save_result=True,
            window_dir=str(logs / "power" / WINDOWS_DIRNAME),
            result_dir=str(logs / RESULT_SUBDIR),
            result_filename=f"{stale_stem}.json",
            concurrency=16,
            log_root=str(logs),
        )
        other.mark_running(1000.0)

        rows = _validate(logs, _samples(1000.0, 1020.0), expected=(("sa-bench", 4), ("sa-bench", 16)))

        by_concurrency = {row.concurrency: row for row in rows}
        assert by_concurrency[4].power_coverage_valid is True
        assert by_concurrency[16].power_coverage_valid is False


class TestFormalBoundaryCapture:
    """The window brackets only formal request execution."""

    @staticmethod
    def _run_benchmark(logs, window, *, fail=False, pooled=True):
        serving = _import_sa_bench_module("benchmark_serving")
        backend_request_func = _import_sa_bench_module("backend_request_func")
        observed_status = []
        calls = []

        async def fake_request(request_func_input=None, pbar=None, **_kwargs):
            calls.append(1)
            if window is not None and Path(window.path).exists():
                observed_status.append(json.loads(Path(window.path).read_text())["status"])
            if fail and len(calls) > 1:
                raise RuntimeError("upstream reset")
            return backend_request_func.RequestFuncOutput(
                generated_text="hello",
                success=True,
                latency=0.01,
                output_tokens=4,
                ttft=0.005,
                itl=[0.001, 0.001, 0.001],
                prompt_len=8192,
                start_time=time.perf_counter(),
            )

        with patch.dict(serving.ASYNC_REQUEST_FUNCS, {"dynamo": fake_request}):
            coroutine = serving.benchmark(
                backend="dynamo",
                api_url="http://localhost:8000/v1/completions",
                base_url="http://localhost:8000",
                model_id="model",
                model_name="model",
                tokenizer=MagicMock(),
                input_requests=[("prompt", 8192, 1024, None)] * 4,
                logprobs=None,
                best_of=1,
                request_rate=float("inf"),
                burstiness=1.0,
                disable_tqdm=True,
                profile=False,
                selected_percentile_metrics=["ttft"],
                selected_percentiles=[99.0],
                ignore_eos=True,
                goodput_config_dict={},
                max_concurrency=4,
                lora_modules=None,
                request_session=MagicMock(closed=True) if (fail and pooled) else None,
                measurement_window=window,
            )
            return asyncio.run(coroutine), observed_status

    def test_formal_run_records_adjacent_boundaries(self, logs):
        window = _create(logs)

        result, observed_status = self._run_benchmark(logs, window)

        start = result["benchmark_start_time_unix"]
        end = result["benchmark_end_time_unix"]
        assert observed_status and set(observed_status) == {"running"}
        assert end > start
        assert abs((end - start) - result["duration"]) < 0.5

    def test_slow_marker_write_does_not_skew_the_window(self, logs):
        """Shared-filesystem latency must not land in the wall clock only.

        The running marker is fsynced to a networked log directory. If that
        write sat between the Unix and monotonic start captures, its latency
        would inflate ``end - start`` without inflating ``duration``.
        """

        class SlowWindow:
            def __init__(self, inner):
                self._inner = inner
                self.path = inner.path

            def mark_running(self, start_unix):
                time.sleep(0.7)
                self._inner.mark_running(start_unix)

            def mark_failed(self, **kwargs):
                self._inner.mark_failed(**kwargs)

            def record_boundary(self, **kwargs):
                self._inner.record_boundary(**kwargs)

            def fail_at_recorded_boundary(self, reason):
                return self._inner.fail_at_recorded_boundary(reason)

        result, _ = self._run_benchmark(logs, SlowWindow(_create(logs)))

        wall = result["benchmark_end_time_unix"] - result["benchmark_start_time_unix"]
        assert abs(wall - result["duration"]) < 0.5

    def test_warmup_without_a_window_writes_nothing(self, logs):
        self._run_benchmark(logs, None)

        assert list((logs / "power" / WINDOWS_DIRNAME).iterdir()) == []

    def test_settled_failure_publishes_a_failed_boundary(self, logs):
        window = _create(logs)

        with pytest.raises(RuntimeError):
            self._run_benchmark(logs, window, fail=True)

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] > 0
        assert "upstream reset" in payload["reason"]


class TestProductionResultWiring:
    """main() must publish the window from the same boundary it saved."""

    def test_saved_result_and_window_agree_exactly(self, logs, monkeypatch):
        serving = _import_sa_bench_module("benchmark_serving")
        backend_request_func = _import_sa_bench_module("backend_request_func")

        async def fake_request(request_func_input=None, pbar=None, **_kwargs):
            return backend_request_func.RequestFuncOutput(
                generated_text="hello",
                success=True,
                latency=0.01,
                output_tokens=4,
                ttft=0.005,
                itl=[0.001],
                prompt_len=8192,
                start_time=time.perf_counter(),
            )

        monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(logs / "power" / WINDOWS_DIRNAME))
        args = _sa_bench_args(logs)

        with (
            patch.dict(serving.ASYNC_REQUEST_FUNCS, {"dynamo": fake_request}),
            patch.object(sys.modules["measurement_window"], "CONTAINER_LOG_DIR", str(logs)),
            patch.object(serving, "load_tokenizer", return_value=MagicMock()),
            patch.object(serving, "sample_random_requests", return_value=[("prompt", 8192, 1024, None)] * 4),
            patch.object(serving, "save_to_pytorch_benchmark_format"),
        ):
            serving.main(args)

        result = json.loads((logs / RESULT_SUBDIR / f"{RESULT_STEM}.json").read_text())
        window = _window_json(logs)

        assert window["status"] == "completed"
        assert window["result_path"] == f"{RESULT_SUBDIR}/{RESULT_STEM}.json"
        assert window["benchmark_start_time_unix"] == result["benchmark_start_time_unix"]
        assert window["benchmark_end_time_unix"] == result["benchmark_end_time_unix"]
        assert window["duration"] == result["duration"]
        assert window["concurrency"] == result["max_concurrency"]

        rows = _validate(logs, _samples(result["benchmark_start_time_unix"], result["benchmark_end_time_unix"]))
        assert rows[0].power_coverage_valid is True

    def _run_main(self, logs, serving, fake_request, *, after=None):
        args = _sa_bench_args(logs)
        patches = [
            patch.dict(serving.ASYNC_REQUEST_FUNCS, {"dynamo": fake_request}),
            patch.object(sys.modules["measurement_window"], "CONTAINER_LOG_DIR", str(logs)),
            patch.object(serving, "load_tokenizer", return_value=MagicMock()),
            patch.object(serving, "sample_random_requests", return_value=[("prompt", 8192, 1024, None)] * 4),
            patch.object(serving, "save_to_pytorch_benchmark_format"),
        ]
        if after is not None:
            patches.append(after)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            serving.main(args)

    def _ok_request(self, backend_request_func):
        async def fake_request(request_func_input=None, pbar=None, **_kwargs):
            return backend_request_func.RequestFuncOutput(
                generated_text="hello",
                success=True,
                latency=0.01,
                output_tokens=4,
                ttft=0.005,
                itl=[0.001],
                prompt_len=8192,
                start_time=time.perf_counter(),
            )

        return fake_request

    def test_metrics_failure_publishes_the_unchanged_boundary(self, logs, monkeypatch):
        """An exception after the formal end must not lose that boundary."""
        serving = _import_sa_bench_module("benchmark_serving")
        backend_request_func = _import_sa_bench_module("backend_request_func")
        monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(logs / "power" / WINDOWS_DIRNAME))

        with pytest.raises(RuntimeError):
            self._run_main(
                logs,
                serving,
                self._ok_request(backend_request_func),
                after=patch.object(serving, "calculate_metrics", side_effect=RuntimeError("metrics blew up")),
            )

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] > 0
        assert payload["benchmark_end_time_unix"] > payload["benchmark_start_time_unix"]
        assert "metrics blew up" in payload["reason"]

    def test_result_write_failure_publishes_the_unchanged_boundary(self, logs, monkeypatch):
        serving = _import_sa_bench_module("benchmark_serving")
        backend_request_func = _import_sa_bench_module("backend_request_func")
        monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(logs / "power" / WINDOWS_DIRNAME))

        with pytest.raises(OSError):
            self._run_main(
                logs,
                serving,
                self._ok_request(backend_request_func),
                after=patch.object(serving, "save_to_pytorch_benchmark_format", side_effect=OSError("disk full")),
            )

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] > 0
        assert "disk full" in payload["reason"]

    def test_non_pooled_request_failure_publishes_a_failed_boundary(self, logs):
        """Previously only the pooled path saved a boundary; now both do."""
        window = _create(logs)

        with pytest.raises(RuntimeError):
            TestFormalBoundaryCapture._run_benchmark(logs, window, fail=True, pooled=False)

        payload = _window_json(logs)
        assert payload["status"] == "failed"
        assert payload["duration"] > 0
        assert "upstream reset" in payload["reason"]

    def test_a_failure_before_the_formal_end_leaves_the_window_running(self, logs):
        """No trustworthy end exists yet, so the orchestrator must decide."""
        window = _create(logs)
        window.mark_running(1000.0)

        assert window.fail_at_recorded_boundary("nothing to publish") is False
        assert _window_json(logs)["status"] == "running"

    def test_warmup_invocation_writes_no_window(self, logs, monkeypatch):
        serving = _import_sa_bench_module("benchmark_serving")
        backend_request_func = _import_sa_bench_module("backend_request_func")

        async def fake_request(request_func_input=None, pbar=None, **_kwargs):
            return backend_request_func.RequestFuncOutput(
                generated_text="hi", success=True, latency=0.01, output_tokens=2, prompt_len=8192
            )

        monkeypatch.setenv("SRT_MEASUREMENT_WINDOW_DIR", str(logs / "power" / WINDOWS_DIRNAME))
        args = _sa_bench_args(logs)
        args.save_result = False  # exactly what bench.sh does for warmup

        with (
            patch.dict(serving.ASYNC_REQUEST_FUNCS, {"dynamo": fake_request}),
            patch.object(sys.modules["measurement_window"], "CONTAINER_LOG_DIR", str(logs)),
            patch.object(serving, "load_tokenizer", return_value=MagicMock()),
            patch.object(serving, "sample_random_requests", return_value=[("prompt", 8192, 1024, None)] * 2),
        ):
            serving.main(args)

        assert list((logs / "power" / WINDOWS_DIRNAME).iterdir()) == []


class TestArtifactErrors:
    def test_unexpected_window_file_is_recorded(self, logs):
        stale = logs / "power" / WINDOWS_DIRNAME / "results_concurrency_99_gpus_8.json"
        stale.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/results_concurrency_99_gpus_8.json",
                    "concurrency": 99,
                    "benchmark_start_time_unix": 1000.0,
                    "benchmark_end_time_unix": 1020.0,
                    "duration": 20.0,
                    "clock_source": "head_node_unix_clock",
                    "status": "completed",
                    "reason": None,
                }
            )
        )
        errors = []

        _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert [error.path for error in errors] == [f"{WINDOWS_DIRNAME}/results_concurrency_99_gpus_8.json"]
        assert Reason.MEASUREMENT_WINDOW_UNEXPECTED in errors[0].reason_codes

    def test_malformed_window_file_is_recorded(self, logs):
        (logs / "power" / WINDOWS_DIRNAME / "broken.json").write_text("{not json")
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes
        assert rows[0].power_coverage_valid is False

    def test_duplicate_windows_for_one_key_invalidate_it(self, logs):
        self._write_duplicate(logs)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_DUPLICATE in rows[0].reason_codes
        assert len(errors) == 2

    def _write_duplicate(self, logs):
        """Two well-formed windows both claiming (sa-bench, 4)."""
        for stem in (RESULT_STEM, "results_concurrency_4_gpus_8"):
            _write_result(logs, start=1000.0, end=1020.0, duration=20.0, stem=stem)
            (logs / "power" / WINDOWS_DIRNAME / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_type": "sa-bench",
                        "result_path": f"{RESULT_SUBDIR}/{stem}.json",
                        "concurrency": 4,
                        "benchmark_start_time_unix": 1000.0,
                        "benchmark_end_time_unix": 1020.0,
                        "duration": 20.0,
                        "clock_source": "head_node_unix_clock",
                        "status": "completed",
                        "reason": None,
                    }
                )
            )

    def test_wrong_clock_source_is_malformed(self, logs):
        window = _create(logs)
        window.mark_running(1000.0)
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        payload = json.loads(path.read_text())
        payload["clock_source"] = "node_local_clock"
        path.write_text(json.dumps(payload))
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert Reason.MEASUREMENT_WINDOW_MALFORMED in errors[0].reason_codes
        assert rows[0].power_coverage_valid is False

    @pytest.mark.parametrize("result_path", ["/etc/passwd", "../escape.json", "a/../../b.json"])
    def test_unsafe_result_paths_are_rejected(self, logs, result_path):
        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": result_path,
                    "concurrency": 4,
                    "benchmark_start_time_unix": 1000.0,
                    "benchmark_end_time_unix": 1020.0,
                    "duration": 20.0,
                    "clock_source": "head_node_unix_clock",
                    "status": "completed",
                    "reason": None,
                }
            )
        )
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID in errors[0].reason_codes

    def test_benchmark_stage_injects_the_container_windows_dir(self, tmp_path):
        harness = _benchmark_harness(tmp_path, provider="dcgm-power")

        env = harness._get_measurement_window_env()

        assert env == {"SRT_MEASUREMENT_WINDOW_DIR": f"/logs/power/{WINDOWS_DIRNAME}"}

    def test_other_providers_get_no_window_dir(self, tmp_path):
        assert _benchmark_harness(tmp_path, provider="scraper")._get_measurement_window_env() == {}
        assert _benchmark_harness(tmp_path, enabled=False)._get_measurement_window_env() == {}

    def test_container_and_host_window_paths_are_the_same_directory(self, tmp_path):
        harness = _benchmark_harness(tmp_path, provider="dcgm-power")
        container_path = harness._get_measurement_window_env()["SRT_MEASUREMENT_WINDOW_DIR"]

        host_path = tmp_path / PurePosixPath(container_path).relative_to("/logs")
        host_path.mkdir(parents=True)
        (host_path / "probe.json").write_text("{}")

        assert harness.runtime.container_mounts[tmp_path] == Path("/logs")
        assert (tmp_path / "power" / WINDOWS_DIRNAME / "probe.json").exists()

    def test_bench_script_saves_results_only_for_the_formal_run(self):
        script = (SA_BENCH_DIR / "bench.sh").read_text()
        warmup, _, formal = script.partition("num_prompts=$((concurrency * NUM_PROMPTS_MULT))")

        assert "--save-result" not in warmup
        assert "--save-result --result-dir" in formal

    def test_result_path_symlinked_outside_the_root_is_rejected(self, logs, tmp_path_factory):
        """The window file itself is safe; its result_path escapes via a symlink."""
        outside = tmp_path_factory.mktemp("outside_result_root")
        (outside / f"{RESULT_STEM}.json").write_text("{}")
        (logs / RESULT_SUBDIR / "escape").symlink_to(outside, target_is_directory=True)

        path = logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/escape/{RESULT_STEM}.json",
                    "concurrency": 4,
                    "benchmark_start_time_unix": 1000.0,
                    "benchmark_end_time_unix": 1020.0,
                    "duration": 20.0,
                    "clock_source": "head_node_unix_clock",
                    "status": "completed",
                    "reason": None,
                }
            )
        )
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_RESULT_PATH_INVALID in errors[0].reason_codes

    def test_symlinked_windows_directory_is_rejected(self, logs, tmp_path_factory):
        """The whole directory moved outside and linked back must not pass."""
        outside = tmp_path_factory.mktemp("outside_windows")
        _write_result(logs, start=1000.0, end=1020.0, duration=20.0)
        (outside / f"{RESULT_STEM}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark_type": "sa-bench",
                    "result_path": f"{RESULT_SUBDIR}/{RESULT_STEM}.json",
                    "concurrency": 4,
                    "benchmark_start_time_unix": 1000.0,
                    "benchmark_end_time_unix": 1020.0,
                    "duration": 20.0,
                    "clock_source": "head_node_unix_clock",
                    "status": "completed",
                    "reason": None,
                }
            )
        )
        windows = logs / "power" / WINDOWS_DIRNAME
        windows.rmdir()
        windows.symlink_to(outside, target_is_directory=True)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert [error.path for error in errors] == [WINDOWS_DIRNAME]
        assert Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID in errors[0].reason_codes

    def test_symlinked_window_is_rejected(self, logs):
        real = logs / "outside.json"
        real.write_text("{}")
        (logs / "power" / WINDOWS_DIRNAME / f"{RESULT_STEM}.json").symlink_to(real)
        errors = []

        rows = _validate(logs, _samples(1000.0, 1020.0), errors=errors)

        assert rows[0].power_coverage_valid is False
        assert Reason.MEASUREMENT_WINDOW_ARTIFACT_PATH_INVALID in errors[0].reason_codes
