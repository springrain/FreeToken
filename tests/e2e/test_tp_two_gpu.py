"""2-GPU tensor-parallel end-to-end boot of a real server.

Boots ``ft serve --tensor-parallel-size 2 --gpu 0,1`` on a small TP-capable checkpoint
(dense bf16/fp16, deepseek-v4 MXFP4, or gpt-oss MXFP4) and checks the server reaches the
serving state and answers a deterministic completion. A rank mismatch in the sharded layouts
or a hang in the inter-rank all-reduce kills the boot or the first generation, so getting text
out the other side is the whole assertion; the per-rank slicing math itself stays in the CPU
suites (tests/dsv4/test_dsv4_tp_sharding.py, tests/moe/test_gpt_oss_tp_sharding.py).

Gated behind ``needs_weights`` and a real 2-GPU box:

  FREETOKEN_TP_TEST_MODEL    small local model dir (falls back to FREETOKEN_TEST_MODEL)
  FREETOKEN_TP_MIN_FREE_GIB  per-GPU free-memory gate (default 10)
  FREETOKEN_TP_BOOT_TIMEOUT  seconds to wait for "serving" (default 300)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.needs_weights


def _model_dir() -> Path | None:
    value = os.environ.get("FREETOKEN_TP_TEST_MODEL") or os.environ.get("FREETOKEN_TEST_MODEL")
    return Path(value).expanduser() if value else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_gib(device: int) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / (1 << 30)


def _get(base: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read())


def _post(base: str, path: str, payload: dict, timeout: float = 180.0) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _wait_until_serving(base: str, proc: subprocess.Popen, deadline: float) -> None:
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if _get(base, "/v1/cache/status")["state"] == "serving":
                return
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            pass  # not listening yet, or still loading
        time.sleep(2.0)
    raise TimeoutError("server never reached the serving state")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="2-GPU TP e2e needs CUDA")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2-GPU TP e2e needs >= 2 CUDA devices")
def test_tp_two_gpu_serves_a_deterministic_completion(tmp_path):
    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip("set FREETOKEN_TP_TEST_MODEL to a small TP-capable model directory")
    if not model_dir.is_dir():
        pytest.skip(f"model is not downloaded: {model_dir}")

    # every rank allocates on its own card, so the tightest one gates the boot
    min_free = float(os.environ.get("FREETOKEN_TP_MIN_FREE_GIB", "10"))
    for dev in range(2):
        free_gib = _free_gib(dev)
        if free_gib < min_free:
            pytest.skip(f"GPU {dev} needs ~{min_free:.0f} GiB free; only {free_gib:.2f} GiB")

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # FREETOKEN_DIST_ADDR: keep the rendezvous off the default so a second engine can coexist
    dist_addr = f"tcp://127.0.0.1:{_free_port()}"
    log = (tmp_path / "serve.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "freetoken",
            "--model-path", str(model_dir),
            "--served-model-name", "tp-test",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--tensor-parallel-size", "2",
            "--gpu", "0,1",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": "python", "FREETOKEN_DIST_ADDR": dist_addr},
    )
    try:
        boot_timeout = float(os.environ.get("FREETOKEN_TP_BOOT_TIMEOUT", "300"))
        _wait_until_serving(base, proc, time.monotonic() + boot_timeout)

        code, body = _post(
            base,
            "/v1/chat/completions",
            {
                "model": "tp-test",
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 32,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert code == 200, body
        assert body["choices"][0]["message"]["content"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
