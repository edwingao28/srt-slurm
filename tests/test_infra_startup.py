# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure startup retains real subprocess ownership before readiness."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from srtctl.cli import do_sweep
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.mock import MockOptions, run_mock_sweep


@pytest.fixture
def infra(tmp_path: Path) -> do_sweep.SweepOrchestrator:
    config = MagicMock()
    config.name = "infra-test"
    config.infra.nats_max_payload_mb = None
    runtime = MagicMock()
    runtime.nodes.infra = "head"
    runtime.nodes.het_group_for.return_value = None
    runtime.log_dir = tmp_path
    runtime.container_mounts = {}
    runtime.container_image = tmp_path / "image.sqsh"
    return do_sweep.SweepOrchestrator(config=config, runtime=runtime)


@pytest.mark.parametrize("exit_code", [0, 23])
@pytest.mark.parametrize("port_ready", [False, True])
def test_infra_child_exit_is_reported_before_service_timeout(
    infra: do_sweep.SweepOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    port_ready: bool,
) -> None:
    with subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.stdin.buffer.read(1); sys.exit({exit_code})"],
        stdin=subprocess.PIPE,
    ) as child:
        monkeypatch.setattr(do_sweep, "start_srun_process", lambda **_: child)

        def unavailable(*_args: object, **_kwargs: object) -> bool:
            assert child.stdin is not None
            if child.poll() is None:
                child.stdin.write(b"x")
                child.stdin.flush()
                child.wait(timeout=5)
            return port_ready

        monkeypatch.setattr(do_sweep, "wait_for_port", unavailable)
        registry = ProcessRegistry("infra-test")
        with pytest.raises(RuntimeError, match=rf"exited with code {exit_code}.*NATS.*infra.out"):
            infra.start_head_infrastructure(registry)
        assert registry.get_process("infra_services").popen is child


def test_infra_timeout_leaves_child_owned_for_cleanup(
    infra: do_sweep.SweepOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(do_sweep, "start_srun_process", lambda **_: child)
        clock = [0.0]
        monkeypatch.setattr(do_sweep, "time", SimpleNamespace(monotonic=lambda: clock[0]))

        def unavailable(*_args: object, **_kwargs: object) -> bool:
            clock[0] += 301
            return False

        monkeypatch.setattr(do_sweep, "wait_for_port", unavailable)
        registry = ProcessRegistry("infra-test")
        with pytest.raises(RuntimeError, match="NATS failed to start"):
            infra.start_head_infrastructure(registry)
        registry.cleanup()
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_ready_infra_is_registered_once_in_complete_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "infra-ready",
                "model": {"path": "hf:fake/model", "container": "fake/image", "precision": "fp8"},
                "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
                "benchmark": {"type": "custom", "command": "echo benchmark"},
            }
        )
    )
    registered = []
    original_add = ProcessRegistry.add_process

    def add(registry: ProcessRegistry, process: ManagedProcess) -> None:
        registered.append(process.name)
        original_add(registry, process)

    monkeypatch.setattr(ProcessRegistry, "add_process", add)
    output = tmp_path / "outputs"
    assert (
        run_mock_sweep(
            config_path=cfg,
            output_dir=output,
            job_id="42046",
            options=MockOptions(child_duration_s=0.15, phase_pause_s=0.01),
        )
        == 0
    )
    assert registered.count("infra_services") == 1
    assert json.loads((output / "result.json").read_text())["status"] == "completed"
