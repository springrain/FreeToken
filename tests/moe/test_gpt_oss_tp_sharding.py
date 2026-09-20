"""gpt-oss TP sharding: pure-function coverage of the rank slicing that
weights/experts go through before kernels see them (shard_gpt_oss_tensor,
local_mxfp4_intermediate_range, _source_slices, iter_expert_pieces).

CPU-only: fake tensors + a synthetic single-file safetensors checkpoint.
The invariant everywhere is REASSEMBLY == ORIGINAL (modulo the rank0-only
biases, which the TP all-reduce is supposed to count exactly once).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _shard(name, value, *, rank, ws, q=4, kv=2, hd=8, inter=64):
    from freetoken.models.gpt_oss.weight import shard_gpt_oss_tensor

    return shard_gpt_oss_tensor(
        name, value, rank=rank, world_size=ws,
        num_q_heads=q, num_kv_heads=kv, head_dim=hd, intermediate_size=inter,
    )


def _shards(name, value, ws, **kw):
    return [_shard(name, value, rank=r, ws=ws, **kw) for r in range(ws)]


# ── attention dense weights ────────────────────────────────────────────────


def test_q_proj_weight_and_bias_reassemble_over_ranks():
    w = torch.arange(4 * 8 * 16, dtype=torch.float32).reshape(4 * 8, 16)  # [q*hd, H]
    b = torch.arange(4 * 8, dtype=torch.float32)

    w0, w1 = _shards("model.layers.0.self_attn.q_proj.weight", w, 2)
    b0, b1 = _shards("model.layers.0.self_attn.q_proj.bias", b, 2)

    assert torch.equal(torch.cat([w0, w1], dim=0), w)
    assert torch.equal(torch.cat([b0, b1], dim=0), b)


def test_kv_proj_replicates_when_tp_exceeds_kv_heads():
    w = torch.arange(2 * 8 * 16, dtype=torch.float32).reshape(2 * 8, 16)  # [kv*hd, H]

    r0, r1, r2, r3 = _shards("model.layers.0.self_attn.k_proj.weight", w, 4)

    assert torch.equal(r0, r1) == torch.equal(r2, r3) == True  # noqa: E712
    assert torch.equal(torch.cat([r0, r2], dim=0), w)


def test_o_proj_weight_shards_dim1_and_bias_lives_on_rank0_only():
    w = torch.arange(16 * (4 * 8), dtype=torch.float32).reshape(16, 4 * 8)  # [H, q*hd]
    b = torch.arange(16, dtype=torch.float32) + 0.5

    w0, w1 = _shards("model.layers.0.self_attn.o_proj.weight", w, 2)
    b0, b1 = _shards("model.layers.0.self_attn.o_proj.bias", b, 2)

    assert torch.equal(torch.cat([w0, w1], dim=1), w)
    assert torch.equal(b0, b)
    assert torch.equal(b1, torch.zeros_like(b))


def test_sinks_split_per_query_head():
    s = torch.arange(4, dtype=torch.float32)

    r0, r1 = _shards("model.layers.0.self_attn.sinks", s, 2)

    assert torch.equal(torch.cat([r0, r1]), s)


def test_head_count_errors_are_explicit():
    with pytest.raises(ValueError, match="query heads"):
        _shards("model.layers.0.self_attn.q_proj.weight", torch.zeros(3, 4), 4, q=3, hd=1)
    with pytest.raises(ValueError, match="KV heads"):
        _shards("model.layers.0.self_attn.k_proj.weight", torch.zeros(3, 4), 8, q=8, kv=3, hd=1)


def test_unmatched_dense_weight_replicates():
    for r in range(2):
        out = _shard("model.layers.0.input_layernorm.weight", torch.ones(4), rank=r, ws=2)
        assert torch.equal(out, torch.ones(4))


# ── mxfp4 expert windowing ─────────────────────────────────────────────────


def test_local_mxfp4_intermediate_range_edges():
    from freetoken.models.gpt_oss.weight import local_mxfp4_intermediate_range

    assert local_mxfp4_intermediate_range(64, rank=0, world_size=1) == (0, 64, 64)

    # clean split: 2 blocks over 2 ranks
    assert local_mxfp4_intermediate_range(64, rank=0, world_size=2) == (0, 32, 32)
    assert local_mxfp4_intermediate_range(64, rank=1, world_size=2) == (32, 64, 32)

    # 3 blocks over 2 ranks: 32-block ceil window, last rank short
    assert local_mxfp4_intermediate_range(96, rank=0, world_size=2) == (0, 64, 64)
    assert local_mxfp4_intermediate_range(96, rank=1, world_size=2) == (64, 96, 64)

    # gpt-oss-120b: 2880 = 90 blocks; tp=8 -> 12-block windows, rank 7 short
    assert local_mxfp4_intermediate_range(2880, rank=0, world_size=8) == (0, 384, 384)
    assert local_mxfp4_intermediate_range(2880, rank=7, world_size=8) == (2688, 2880, 384)

    with pytest.raises(ValueError, match="divisible by 32"):
        local_mxfp4_intermediate_range(48, rank=0, world_size=1)


def test_shard_gpt_oss_expert_tensors_reassemble():
    """_shard_dim_pad slicing of {gate_up,down}_{blocks,scales,bias} cats back."""
    E, I, Hb = 2, 96, 4  # 3 blocks over tp=2 -> rank1 window [64:96]
    gu = torch.arange(E * 2 * I * Hb * 2, dtype=torch.float32).reshape(E, 2 * I, Hb, 2)
    gu_s = torch.arange(E * 2 * I * Hb, dtype=torch.float32).reshape(E, 2 * I, Hb)
    gu_b = torch.arange(E * 2 * I, dtype=torch.float32).reshape(E, 2 * I)

    r0 = _shard("model.layers.0.mlp.experts.gate_up_proj_blocks", gu, rank=0, ws=2, inter=I)
    r1 = _shard("model.layers.0.mlp.experts.gate_up_proj_blocks", gu, rank=1, ws=2, inter=I)
    # rank0 padded window is full; rank1 short window gets zero-padded to the window width
    assert torch.equal(r0, gu[:, 0:128])
    assert torch.equal(r1[:, : 2 * (I - 64)], gu[:, 128 : 2 * I])
    assert torch.equal(r1[:, 2 * (I - 64) :], torch.zeros_like(r1[:, 2 * (I - 64) :]))

    s1 = _shard("model.layers.0.mlp.experts.gate_up_proj_scales", gu_s, rank=0, ws=2, inter=I)
    assert torch.equal(s1, gu_s[:, 0:128])
    b1 = _shard("model.layers.0.mlp.experts.gate_up_proj_bias", gu_b, rank=1, ws=2, inter=I)
    assert torch.equal(b1[:, : 2 * (I - 64)], gu_b[:, 128:])

    dn_b = torch.arange(E * 16, dtype=torch.float32).reshape(E, 16)
    d0 = _shard("model.layers.0.mlp.experts.down_proj_bias", dn_b, rank=0, ws=2, inter=I)
    d1 = _shard("model.layers.0.mlp.experts.down_proj_bias", dn_b, rank=1, ws=2, inter=I)
    assert torch.equal(d0, dn_b)
    assert torch.equal(d1, torch.zeros_like(dn_b))


# ── source slices + file-backed iter_expert_pieces ─────────────────────────


def _write_fake_checkpoint(tmp_path, *, E=2, I=64, H=64):
    from safetensors.torch import save_file

    Hb, Ib = H // 32, I // 32
    tensors = {
        # gpt-oss mxfp4: blocks [E, rows, cols/32, 16], scales [E, rows, cols/32]
        "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.randint(
            0, 255, (E, 2 * I, Hb, 16), dtype=torch.uint8),
        "model.layers.0.mlp.experts.gate_up_proj_scales": torch.randint(
            0, 255, (E, 2 * I, Hb), dtype=torch.uint8),
        "model.layers.0.mlp.experts.gate_up_proj_bias": torch.randn(E, 2 * I),
        "model.layers.0.mlp.experts.down_proj_blocks": torch.randint(
            0, 255, (E, H, Ib, 16), dtype=torch.uint8),
        "model.layers.0.mlp.experts.down_proj_scales": torch.randint(
            0, 255, (E, H, Ib), dtype=torch.uint8),
        "model.layers.0.mlp.experts.down_proj_bias": torch.randn(E, H),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(H, H),  # non-expert: skipped
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def _iter_rank_pieces(monkeypatch, path, config, rank, ws):
    import freetoken.models.gpt_oss.weight as gpt_weight
    from freetoken.distributed import DistributedInfo
    from freetoken.layers.quantization import QuantKind

    monkeypatch.setattr(gpt_weight, "get_tp_info", lambda: DistributedInfo(rank=rank, size=ws))
    return list(gpt_weight.iter_expert_pieces(str(path), config, QuantKind.MXFP4))


def test_source_slices_ceil_window_and_block_alignment():
    import freetoken.models.gpt_oss.weight as gpt_weight
    from freetoken.distributed import DistributedInfo

    config = SimpleNamespace(moe_intermediate_size=96)
    slices0 = gpt_weight._source_slices(config, DistributedInfo(rank=0, size=2))
    slices1 = gpt_weight._source_slices(config, DistributedInfo(rank=1, size=2))

    # 3 blocks over 2 ranks -> window [0:64] / [64:96], gate_up rows are 2x
    assert slices0["gate_up_proj_blocks"][1][1] == slice(0, 128)
    assert slices1["gate_up_proj_blocks"][1][1] == slice(128, 192)
    # down cols are in 32-blocks of the intermediate axis
    assert slices0["down_proj_blocks"][1][2] == slice(0, 2)
    assert slices1["down_proj_blocks"][1][2] == slice(2, 3)
    # roles map to the bank contract
    assert slices0["gate_up_proj_scales"][0] == "gate_up_scale"
    assert slices0["down_proj_bias"][0] == "down_bias"


def test_iter_expert_pieces_reassemble_per_rank(monkeypatch, tmp_path):
    full = _write_fake_checkpoint(tmp_path, E=2, I=64, H=64)
    config = SimpleNamespace(
        num_layers=1, num_experts=2, moe_intermediate_size=64, moe_weight_format="mxfp4"
    )

    pieces = [_iter_rank_pieces(monkeypatch, tmp_path, config, r, 2) for r in range(2)]

    for rank_pieces in pieces:
        assert len(rank_pieces) == 1
        layer_id, e0, e1, piece = rank_pieces[0]
        assert (layer_id, e0, e1) == (0, 0, 2)
        assert len(piece) == 6

    p0, p1 = pieces[0][0][3], pieces[1][0][3]
    assert torch.equal(
        torch.cat([p0["gate_up"], p1["gate_up"]], dim=1),
        full["model.layers.0.mlp.experts.gate_up_proj_blocks"],
    )
    assert torch.equal(
        torch.cat([p0["gate_up_scale"], p1["gate_up_scale"]], dim=1),
        full["model.layers.0.mlp.experts.gate_up_proj_scales"],
    )
    assert torch.equal(
        torch.cat([p0["gate_up_bias"], p1["gate_up_bias"]], dim=1),
        full["model.layers.0.mlp.experts.gate_up_proj_bias"],
    )
    assert torch.equal(
        torch.cat([p0["down"], p1["down"]], dim=2),
        full["model.layers.0.mlp.experts.down_proj_blocks"],
    )
    assert torch.equal(
        torch.cat([p0["down_scale"], p1["down_scale"]], dim=2),
        full["model.layers.0.mlp.experts.down_proj_scales"],
    )
    # bias counted exactly once: rank0 keeps it, ranks>0 zero it
    assert torch.equal(p0["down_bias"], full["model.layers.0.mlp.experts.down_proj_bias"])
    assert torch.equal(p1["down_bias"], torch.zeros_like(p1["down_bias"]))


def test_iter_expert_pieces_rejects_non_mxfp4_kind(tmp_path):
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.gpt_oss.weight import iter_expert_pieces

    config = SimpleNamespace(
        num_layers=1, num_experts=2, moe_intermediate_size=64, moe_weight_format="mxfp4"
    )
    assert iter_expert_pieces(str(tmp_path), config, QuantKind.NONE) is None


# ── kernel TP gating: floor-sized banks must equal the padded ceil window ──
#
# The kernel sizes banks from floor(intermediate/tp) (MoEConfig.local_intermediate)
# while the sharder zero-pads each rank to ceil(blocks/tp)*32. The two only
# agree when intermediate % (32*tp) == 0, so the kernel must accept exactly
# those sizes and reject the rest loudly instead of crashing at weight load.


def _kernel_cfg(tp_size, intermediate=2880, hidden=64):
    # intermediate=2880 is the real gpt-oss expert size (90 blocks)
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(
        num_experts=2, intermediate=intermediate, hidden=hidden, top_k=2,
        tp_size=tp_size, tp_rank=0, strategy="resident",
        activation="swiglu", alpha=1.0, beta=0.0, limit=None, has_bias=True,
        apply_router_weight_on_input=False, interleaved=True,
    )


@pytest.mark.parametrize("tp_size", [2, 3, 5, 6])
def test_gptoss_mxfp4_kernel_accepts_tp_where_floor_equals_window(tp_size):
    from freetoken.layers.quantization.moe.mxfp4 import TritonGptossMxfp4MoEKernel
    from freetoken.models.gpt_oss.weight import local_mxfp4_intermediate_range

    cfg = _kernel_cfg(tp_size)
    assert TritonGptossMxfp4MoEKernel().unusable_reason(cfg) is None
    _, _, window = local_mxfp4_intermediate_range(2880, rank=0, world_size=tp_size)
    assert cfg.local_intermediate == window


@pytest.mark.parametrize("tp_size", [4, 8])
def test_gptoss_mxfp4_kernel_rejects_tp_splitting_a_32_block(tp_size):
    from freetoken.layers.quantization.moe.mxfp4 import TritonGptossMxfp4MoEKernel

    reason = TritonGptossMxfp4MoEKernel().unusable_reason(_kernel_cfg(tp_size))
    assert reason is not None
    assert "32*tp_size" in reason


def test_gptoss_mxfp4_kernel_sizes_banks_from_local_intermediate():
    from freetoken.layers.quantization.moe.mxfp4 import TritonGptossMxfp4MoEKernel

    cfg = _kernel_cfg(2)
    assert cfg.local_intermediate == 1440
    layout = TritonGptossMxfp4MoEKernel().layout(cfg)
    assert layout["gate_up"].shape == (32, 2 * 1440)
    assert layout["gate_up_bias"].shape == (2 * 1440,)
    assert layout["down"].shape == (720, 64)
