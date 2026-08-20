# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-process VLLM_PORT assignment (rendezvous EADDRINUSE avoidance)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from srtctl.backends.vllm import VLLMProtocol
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.submit import submit_with_orchestrator
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import ModelConfig, ResourceConfig, SrtConfig
from srtctl.core.topology import Process
from srtctl.ports import (
    DYN_SYSTEM_PORT_BASE,
    KV_EVENTS_PORT_BASE,
    VLLM_PORT_BASE,
    VLLM_PORT_END,
    VLLM_PORT_STRIDE,
)


def _process(sys_port: int) -> Process:
    """A minimal vLLM worker process (nixl_port=None so no host lookup runs)."""
    return Process(
        node="node0",
        gpu_indices=frozenset({0}),
        sys_port=sys_port,
        http_port=0,
        endpoint_mode="decode",
        endpoint_index=0,
    )


def _vllm_config(worker_count: int) -> SrtConfig:
    return SrtConfig(
        name="vllm-port-preflight",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
        resources=ResourceConfig(
            gpu_type="h200",
            gpus_per_node=1,
            agg_nodes=worker_count,
            agg_workers=worker_count,
        ),
        backend=VLLMProtocol(),
    )


def test_vllm_port_is_unique_per_process_with_stride():
    """Co-located workers get distinct VLLM_PORT bases spaced by the full stride."""
    backend = VLLMProtocol()

    envs = [backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE + i)) for i in range(3)]
    ports = [int(env["VLLM_PORT"]) for env in envs]

    assert ports == [
        VLLM_PORT_BASE,
        VLLM_PORT_BASE + VLLM_PORT_STRIDE,
        VLLM_PORT_BASE + 2 * VLLM_PORT_STRIDE,
    ]
    # Distinct and a full stride apart, so per-process get_open_port() scan
    # ranges cannot overlap.
    assert len(set(ports)) == len(ports)
    assert all(ports[i + 1] - ports[i] == VLLM_PORT_STRIDE for i in range(len(ports) - 1))


def test_vllm_port_clamps_when_sys_port_below_anchor():
    """A sys_port below the anchor must not produce a negative offset."""
    backend = VLLMProtocol()

    env = backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE - 10))

    assert env["VLLM_PORT"] == str(VLLM_PORT_BASE)


def test_vllm_port_windows_fail_before_the_kv_events_range():
    backend = VLLMProtocol()
    last_process_index = (VLLM_PORT_END - VLLM_PORT_BASE + 1) // VLLM_PORT_STRIDE - 1

    env = backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE + last_process_index))

    assert int(env["VLLM_PORT"]) + VLLM_PORT_STRIDE - 1 == VLLM_PORT_END
    assert VLLM_PORT_END < KV_EVENTS_PORT_BASE
    with pytest.raises(ValueError, match="VLLM process port range exhausted"):
        backend.get_process_environment(_process(DYN_SYSTEM_PORT_BASE + last_process_index + 1))


def test_vllm_port_exhaustion_fails_before_sbatch(tmp_path: Path):
    sbatch_calls: list[list[str]] = []

    def reject_sbatch(command: list[str], **_kwargs):
        if command and command[0] == "sbatch":
            sbatch_calls.append(command)
        raise AssertionError(f"subprocess must not run during topology preflight: {command}")

    with (
        patch("srtctl.cli.submit.get_srtslurm_setting", return_value=None),
        patch("srtctl.cli.submit.validate_setup"),
        patch("srtctl.cli.submit.subprocess.run", side_effect=reject_sbatch),
        pytest.raises(ValueError, match="VLLM process port range exhausted"),
    ):
        submit_with_orchestrator(
            config_path=tmp_path / "config.yaml",
            config=_vllm_config(worker_count=161),
        )

    assert sbatch_calls == []


def test_vllm_port_exhaustion_launches_no_partial_workers(tmp_path: Path):
    config = _vllm_config(worker_count=161)
    nodes = tuple(f"worker-{index}" for index in range(161))
    runtime = RuntimeContext(
        job_id="12345",
        run_name="vllm-port-preflight",
        nodes=Nodes(head=nodes[0], bench=nodes[0], infra=nodes[0], worker=nodes),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/model"),
        container_image=Path("/container.sqsh"),
        gpus_per_node=1,
        network_interface=None,
    )
    orchestrator = SweepOrchestrator(config=config, runtime=runtime)

    with (
        patch.object(orchestrator, "start_worker") as start_worker,
        pytest.raises(ValueError, match="VLLM process port range exhausted"),
    ):
        orchestrator.start_all_workers()

    start_worker.assert_not_called()
