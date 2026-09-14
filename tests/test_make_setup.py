# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real setup recipe and curl against a local release server."""

import io
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

ELF = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 + b"\x02\x00\xb7\x00" + b"\x00" * 44


@pytest.fixture
def setup_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copyfile(Path(__file__).resolve().parents[1] / "Makefile", tmp_path / "Makefile")
    (tmp_path / "configs").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "srtslurm.yaml").touch()
    for name in ("nats-server", "etcd", "etcdctl"):
        (tmp_path / "configs" / name).write_bytes(ELF)

    # Redirect only the release URL; real curl still handles HTTP and retries.
    curl = shutil.which("curl")
    assert curl is not None
    mockbin = tmp_path / "mockbin"
    mockbin.mkdir()
    wrapper = mockbin / "curl"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = [os.environ['UV_TEST_URL'] if arg.startswith('https://github.com/astral-sh/uv/') "
        "else arg for arg in sys.argv[1:]]\n"
        f"os.execv({curl!r}, [{curl!r}, *args])\n"
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{mockbin}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    state = {"status": 200, "failures": 0, "requests": 0, "body": b""}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state["requests"] += 1
            status = 504 if state["requests"] <= state["failures"] else state["status"]
            self.send_response(status)
            self.end_headers()
            self.wfile.write(state["body"] if status == 200 else b"gateway timeout")

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    monkeypatch.setenv("UV_TEST_URL", f"http://127.0.0.1:{server.server_port}/uv.tar.gz")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def archive(binary: bytes = ELF) -> bytes:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for name in ("uv", "uvx"):
            entry = tarfile.TarInfo(f"uv-aarch64-unknown-linux-gnu/{name}")
            entry.size = len(binary)
            entry.mode = 0o755
            tar.addfile(entry, io.BytesIO(binary))
    return data.getvalue()


def setup(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "setup", "ARCH=aarch64"], cwd=path, capture_output=True, text=True, timeout=30, check=False
    )


@pytest.mark.parametrize("failure", ["http", "truncated", "wrong_arch"])
def test_failed_download_preserves_existing_binaries(setup_dir: Path, release: dict, failure: str) -> None:
    old = ELF[:18] + b"\x3e\x00" + ELF[20:]
    for name in ("uv", "uvx"):
        (setup_dir / "bin" / name).write_bytes(old)
    release["status"] = 504 if failure == "http" else 200
    release["body"] = b"truncated" if failure == "truncated" else archive(old)

    result = setup(setup_dir)

    assert result.returncode != 0, result.stdout + result.stderr
    if failure == "http":
        assert release["requests"] == 4
    assert "uv installed to" not in result.stdout
    for name in ("uv", "uvx"):
        assert (setup_dir / "bin" / name).read_bytes() == old
    assert not list((setup_dir / "bin").glob(".uv-*"))


@pytest.mark.parametrize("failures", [0, 1])
def test_installs_valid_release_and_retries_504(setup_dir: Path, release: dict, failures: int) -> None:
    release.update(body=archive(), failures=failures)

    result = setup(setup_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert release["requests"] == failures + 1
    for name in ("uv", "uvx"):
        binary = setup_dir / "bin" / name
        assert binary.read_bytes() == ELF
        assert os.access(binary, os.X_OK)
    assert not list((setup_dir / "bin").glob(".uv-*"))


def test_reuses_valid_binaries_without_download(setup_dir: Path, release: dict) -> None:
    for name in ("uv", "uvx"):
        binary = setup_dir / "bin" / name
        binary.write_bytes(ELF)
        binary.chmod(0o755)

    result = setup(setup_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    assert release["requests"] == 0
