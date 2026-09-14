# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise make setup with real curl/tar/file and a local HTTP endpoint."""

import io
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ELF = (
    b"\x7fELF"
    + bytes([2, 1, 1, 0])
    + bytes(8)
    + struct.pack("<HHIQQQIHHHHHH", 2, 183, 1, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0)
)


@pytest.fixture
def setup_dir(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "Makefile").write_bytes((root / "Makefile").read_bytes())
    (tmp_path / "srtslurm.yaml").touch()
    (tmp_path / "configs").mkdir()
    for name in ("nats-server", "etcd", "etcdctl"):
        (tmp_path / "configs" / name).write_bytes(ELF)
    return tmp_path


def exercise_setup(path, statuses, payload=ELF, cache=False, resource="uv"):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in ("uv", "uvx") if resource == "uv" else ("etcd", "etcdctl"):
            archive_dir = "uv-aarch64-unknown-linux-gnu" if resource == "uv" else "etcd-v3.5.21-linux-arm64"
            info = tarfile.TarInfo(archive_dir + "/" + name)
            info.size = len(payload)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(payload))
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            code = statuses[min(len(hits), len(statuses) - 1)]
            hits.append(code)
            body = archive.getvalue() if code == 200 else b"temporary gateway failure"
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shims = path / "commands"
    shims.mkdir()
    (shims / "curl").write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        f"args = [os.environ['TEST_UV_URL'] if a.startswith(('https://github.com/astral-sh/uv/', 'https://github.com/etcd-io/etcd/', 'https://github.com/nats-io/nats-server/')) else a for a in sys.argv[1:]]\n"
        f"os.execv({shutil.which('curl')!r}, ['curl', *args])\n"
    )
    (shims / "curl").chmod(0o755)
    if cache:
        (shims / "uv").write_bytes(ELF)
        (shims / "uv").chmod(0o755)
    else:
        # An installed host uv must not turn a download test into a cache hit.
        (shims / "uv").write_text("#!/bin/sh\nexit 99\n")
        (shims / "uv").chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(shims) + os.pathsep + os.environ["PATH"],
        "TEST_UV_URL": f"http://127.0.0.1:{server.server_port}/uv.tar.gz",
        "TMPDIR": str(path),
    }
    try:
        result = subprocess.run(
            ["make", "setup", "ARCH=aarch64"],
            cwd=path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return result, hits


def test_retry_temporary_gateway_failure(setup_dir):
    result, hits = exercise_setup(setup_dir, [504, 200])
    assert result.returncode == 0, result.stdout + result.stderr
    assert (setup_dir / "bin/uv").read_bytes() == ELF
    assert hits == [504, 200]


def test_terminal_download_failure_is_not_installed(setup_dir):
    result, hits = exercise_setup(setup_dir, [504])
    assert result.returncode != 0
    assert "uv installed to" not in result.stdout
    assert not (setup_dir / "bin/uv").exists()
    assert len(hits) == 3
    assert not list(setup_dir.glob("tmp.*"))


def test_reuse_matching_compute_binary_without_network(setup_dir):
    result, hits = exercise_setup(setup_dir, [504], cache=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hits == []
    assert (setup_dir / "bin/uv").read_bytes() == ELF


def test_wrong_arch_archive_is_not_installed(setup_dir):
    result, _ = exercise_setup(setup_dir, [200], payload=b"#!/bin/sh\nexit 0\n")
    assert result.returncode != 0
    assert not (setup_dir / "bin/uv").exists()
    assert "uv installed to" not in result.stdout


@pytest.fixture
def etcd_setup_dir(setup_dir):
    for name in ("etcd", "etcdctl"):
        (setup_dir / "configs" / name).unlink()
    (setup_dir / "bin").mkdir()
    (setup_dir / "bin/uv").write_bytes(ELF)
    (setup_dir / "bin/uv").chmod(0o755)
    return setup_dir


def test_etcd_retry_temporary_gateway_failure(etcd_setup_dir):
    result, hits = exercise_setup(etcd_setup_dir, [504, 200], resource="etcd")
    assert result.returncode == 0, result.stdout + result.stderr
    assert hits == [504, 200]
    for name in ("etcd", "etcdctl"):
        assert (etcd_setup_dir / "configs" / name).read_bytes() == ELF


def test_etcd_terminal_failure_stops_setup(etcd_setup_dir):
    result, hits = exercise_setup(etcd_setup_dir, [504], resource="etcd")
    assert result.returncode != 0
    assert hits == [504, 504, 504]
    assert "504" in result.stderr
    assert "ETCD installed to" not in result.stdout
    assert not (etcd_setup_dir / "configs/etcd").exists()
    assert not (etcd_setup_dir / "configs/etcd-v3.5.21-linux-arm64.tar.gz").exists()


def test_nats_terminal_failure_stops_setup(setup_dir):
    (setup_dir / "configs/nats-server").unlink()
    result, hits = exercise_setup(setup_dir, [503])
    assert result.returncode != 0
    assert hits == [503, 503, 503]
    assert "503" in result.stderr
    assert "NATS installed to" not in result.stdout
    assert "--- ETCD" not in result.stdout
    assert not (setup_dir / "configs/nats-server").exists()
    assert not (setup_dir / "configs/nats-server-v2.10.28-arm64.deb").exists()
