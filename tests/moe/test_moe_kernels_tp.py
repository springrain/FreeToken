"""TP gating of the fp8-block / NVFP4 MoE kernels.

An aligned tensor-parallel size is accepted by the fp8-block triton kernel and
sizes every bank from the rank-LOCAL intermediate (``cfg.local_intermediate``);
a size that would split a quant block across ranks is rejected with the
divisibility requirement. NVFP4 kernels stay TP-gated until an expert reader
shards pieces by rank — opt-in without a sharding reader loads full pieces
into rank-local banks and crashes mid-load (glm4_moe was reachable there).

CPU-only: gate decisions, bank descriptions and pack shapes need no device.
The two sibling suites (tests/moe/test_nvfp4_backends.py,
tests/kernels/test_fp8_blockscale_moe.py) are module-level GPU-gated, so the
TP gate checks live here.
"""

from __future__ import annotations

import pytest
import torch

FP8 = torch.float8_e4m3fn


def _moe_cfg(tp_size, *, intermediate, hidden, strategy="offload"):
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(
        num_experts=2, intermediate=intermediate, hidden=hidden, top_k=2,
        tp_size=tp_size, tp_rank=0, strategy=strategy,
        activation="silu", alpha=1.0, beta=0.0, limit=None, has_bias=False,
        apply_router_weight_on_input=False, interleaved=False,
    )


# ── fp8 e4m3 experts with 128x128 block scales ─────────────────────────────


# distinct local-intermediate sizes AND a non-power-of-two tp: (4, 512) collapsed to
# the same local layout as (2, 256) and is dropped
@pytest.mark.parametrize("tp_size, intermediate", [(2, 256), (2, 512), (3, 768)])
def test_fp8_block_kernel_accepts_aligned_tp_and_sizes_banks_locally(tp_size, intermediate):
    from freetoken.kernel.aot_models import fp8_block_scale_pad
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel

    k = TritonFp8BlockMoEKernel()
    cfg = _moe_cfg(tp_size, intermediate=intermediate, hidden=128)
    assert k.unusable_reason(cfg) is None

    i = cfg.local_intermediate
    layout = k.layout(cfg)
    assert layout["gate_up"].shape == (2 * i, 128)
    assert layout["gate_up_scale"].shape == (2 * i // 128, fp8_block_scale_pad(2 * i // 128, 1))
    assert layout["down"].shape == (128, i)
    assert layout["down_scale"].shape == (1, fp8_block_scale_pad(1, i // 128))


def test_fp8_block_kernel_tp1_sizes_banks_from_full_intermediate():
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel

    k = TritonFp8BlockMoEKernel()
    cfg = _moe_cfg(1, intermediate=256, hidden=128)
    assert k.unusable_reason(cfg) is None
    assert k.layout(cfg)["gate_up"].shape == (512, 128)
    assert k.layout(cfg)["down"].shape == (128, 256)


@pytest.mark.parametrize("tp_size, intermediate", [(2, 192), (4, 256), (2, 448)])
def test_fp8_block_kernel_rejects_misaligned_tp(tp_size, intermediate):
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel

    reason = TritonFp8BlockMoEKernel().unusable_reason(_moe_cfg(tp_size, intermediate=intermediate, hidden=128))
    assert reason is not None
    assert "128*tp_size" in reason


def test_fp8_block_method_selection_fails_loudly_on_misaligned_tp():
    from freetoken.layers.quantization.method import KernelSelectionError
    from freetoken.layers.quantization.moe.fp8_block import Fp8BlockMoEMethod

    with pytest.raises(KernelSelectionError, match=r"128\*tp_size"):
        Fp8BlockMoEMethod(_moe_cfg(2, intermediate=192, hidden=128))


def test_fp8_block_method_create_weights_sizes_resident_banks_locally():
    from freetoken.layers.quantization.moe.fp8_block import Fp8BlockMoEMethod

    class Layer:
        pass

    cfg = _moe_cfg(2, intermediate=256, hidden=128, strategy="resident")
    method = Fp8BlockMoEMethod(cfg)
    layer = Layer()
    method.create_weights(layer)
    assert layer.gate_up_proj.shape == (2, 2 * 128, 128)  # [E, 2*local_i, H]
    assert layer.gate_up_scale_inv.shape == (2, 2 * 128 // 128, 128 // 128)
    assert layer.down_proj.shape == (2, 128, 128)  # [E, H, local_i]
    assert layer.down_scale_inv.shape == (2, 128 // 128, 128 // 128)


# ── NVFP4 experts (packed e2m1 + per-16 fp8 scales + global) ───────────────
#
# No NVFP4 expert reader shards pieces by rank today, so EVERY NVFP4 kernel
# stays TP-gated: opting in would load full expert pieces into rank-local
# banks and crash on the shape mismatch (the glm4_moe path reached exactly
# that). Gate, method-level rejection and the single-GPU pack wiring live here.


@pytest.mark.parametrize("tp_size, intermediate", [(2, 128), (4, 448), (2, 48), (3, 120)])
def test_nvfp4_triton_kernel_stays_tp_gated(tp_size, intermediate):
    """Aligned or not, the triton kernel rejects TP until a reader opts in."""
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    k = TritonNvfp4MoEKernel()
    assert k.unusable_reason(_moe_cfg(1, intermediate=intermediate, hidden=64)) is None
    reason = k.unusable_reason(_moe_cfg(tp_size, intermediate=intermediate, hidden=64))
    assert reason is not None
    assert "TP" in reason


def test_nvfp4_method_rejects_tp2_loudly():
    from freetoken.layers.quantization.method import KernelSelectionError
    from freetoken.layers.quantization.moe.nvfp4 import Nvfp4MoEMethod

    with pytest.raises(KernelSelectionError, match="TP"):
        Nvfp4MoEMethod(_moe_cfg(2, intermediate=128, hidden=64))


def test_nvfp4_marlin_and_b12x_stay_gated_under_tp2(monkeypatch):
    """The TP gate must be the rejection reason even on boxes lacking the kernels'
    prerequisites: monkeypatch past the environment checks (vLLM import, sm_120
    capability, flashinfer import) so the tp_ok=False reject is what fires --
    otherwise a tp_ok flip would stay invisible behind the env reasons."""
    import freetoken.kernel.backend as backend_mod
    from freetoken.layers.quantization.moe.nvfp4 import B12xNvfp4MoEKernel, MarlinNvfp4MoEKernel

    monkeypatch.setattr(backend_mod, "is_vllm_installed", lambda: True)
    monkeypatch.setattr(backend_mod, "is_flashinfer_installed", lambda: True)
    monkeypatch.setattr(backend_mod, "device_capability", lambda: (12, 0))

    cfg = _moe_cfg(2, intermediate=128, hidden=64)
    marlin_reason = MarlinNvfp4MoEKernel().unusable_reason(cfg)
    assert marlin_reason is not None and "TP" in marlin_reason
    b12x_reason = B12xNvfp4MoEKernel().unusable_reason(cfg)
    assert b12x_reason is not None and "TP" in b12x_reason


def _nvfp4_pieces(e, i, h):
    """One expert batch at the kernel's full intermediate window."""
    return dict(
        gate=torch.zeros(e, i, h // 2, dtype=torch.uint8),
        up=torch.zeros(e, i, h // 2, dtype=torch.uint8),
        gate_scale=torch.zeros(e, i, h // 16, dtype=FP8),
        up_scale=torch.zeros(e, i, h // 16, dtype=FP8),
        gate_global=torch.full((e, 1), 2.0),
        up_global=torch.full((e, 1), 3.0),
        down=torch.zeros(e, h, i // 2, dtype=torch.uint8),
        down_scale=torch.zeros(e, h, i // 16, dtype=FP8),
        down_global=torch.full((e, 1), 4.0),
    )


def test_nvfp4_triton_pack_fused_globals_follow_gate_up_rows():
    """The gate/up scalars broadcast to the first/second half of the fused 2*i
    bank rows and down fills every row; a fused_global sized off the wrong
    intermediate breaks exactly these halves."""
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    k = TritonNvfp4MoEKernel()
    cfg = _moe_cfg(1, intermediate=128, hidden=64)
    e, i = 1, 128
    out = {name: torch.zeros((e,) + spec.shape, dtype=spec.dtype) for name, spec in k.layout(cfg).items()}
    k.pack(_nvfp4_pieces(e, i, 64), cfg, out)

    assert out["gate_up"].shape == (e, 2 * i, 32)
    assert out["gate_up_global"].shape == (e, 2 * i)
    assert torch.all(out["gate_up_global"][0, :i] == 2.0)
    assert torch.all(out["gate_up_global"][0, i:] == 3.0)
    assert torch.all(out["down_global"][0] == 4.0)
