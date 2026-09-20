"""DeepSeek-V4 TP sharding: rank slicing of dense weights, MXFP4 expert pieces,
and the attention module's local geometry.

CPU-only: synthetic single-file safetensors checkpoints + tiny DeepseekV4Args. The
invariant everywhere is REASSEMBLY == ORIGINAL (cat of the rank shards); the wo_a/wo_b
split-K identity is checked numerically (rank partials sum to the full projection, with
exactly one all-reduce per row-parallel forward).
"""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.distributed.info import reset_tp_info, set_tp_info

E8M0 = torch.float8_e8m0fnu
FP8 = torch.float8_e4m3fn


def _tiny_args(**over):
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    kw = dict(
        vocab_size=131, dim=64, moe_inter_dim=256, n_layers=1, n_hash_layers=1,
        n_heads=2, q_lora_rank=64, head_dim=128, rope_head_dim=64,
        o_groups=2, o_lora_rank=128, compress_ratios=(0,),
    )
    kw.update(over)
    return DeepseekV4Args(**kw)


# ── attention local geometry ────────────────────────────────────────────────


def test_attention_local_geometry_under_tp2():
    from freetoken.models.deepseek_v4.attention import Attention

    # full-suite runs arrive here with an import-time tp(0,1) already set
    reset_tp_info()
    set_tp_info(0, 2)
    m = Attention(0, _tiny_args())

    assert m.n_heads == 2 and m.n_heads_local == 1
    assert m.n_groups == 2 and m.n_groups_local == 1
    assert m.attn_sink.shape == (1,)
    # wo_a rows are the rank's whole groups; K per group is unchanged
    assert m.wo_a.shape == (128, 128)
    # wq_b keeps its global declaration and the column-parallel class divides it
    assert m.wq_b.weight.shape == (128, 64)
    assert m.wo_b.weight.shape == (64, 128)


@pytest.mark.parametrize(
    "over, match",
    [({"n_heads": 3}, "n_heads divisible"), ({"o_groups": 3}, "o_groups divisible")],
)
def test_attention_raises_on_indivisible_sharding(over, match):
    from freetoken.models.deepseek_v4.attention import Attention

    reset_tp_info()
    set_tp_info(0, 2)
    with pytest.raises(ValueError, match=match):
        Attention(0, _tiny_args(**over))


def test_wo_split_k_partials_sum_to_full_wo(monkeypatch):
    """Each rank runs wo_a over its own group + wo_b over its input slice; the
    all-reduce sums the partials back to the full projection, once per call."""
    from freetoken.distributed.impl import DistributedCommunicator
    from freetoken.models.deepseek_v4.attention import Attention

    torch.manual_seed(0)
    calls = []

    def fake_all_reduce(self, y):
        calls.append(1)
        return y

    monkeypatch.setattr(DistributedCommunicator, "all_reduce", fake_all_reduce)

    G, OLR, K, DIM, T = 2, 128, 128, 64, 5
    full_woa = torch.randn(G * OLR, K)
    full_wob = torch.randn(DIM, G * OLR)
    o_full = torch.randn(1, T, G * K)

    ref = (
        torch.einsum("btgd,grd->btgr", o_full.view(1, T, G, K), full_woa.view(G, OLR, K))
        .flatten(2)
        @ full_wob.T
    )

    partials = []
    for rank in range(2):
        reset_tp_info()
        set_tp_info(rank, 2)
        m = Attention(0, _tiny_args())
        lo = rank * OLR
        # stay float32 (wo_a is bf16 in the engine) for an exact split-K comparison
        m.wo_a = full_woa[lo:lo + OLR]
        m.wo_b.weight.copy_(full_wob[:, lo:lo + OLR])
        o_rank = o_full[..., rank * K:(rank + 1) * K].contiguous()
        partials.append(m._wo(o_rank, 1, T))

    assert calls == [1, 1]
    torch.testing.assert_close(partials[0] + partials[1], ref, rtol=1e-4, atol=1e-5)


# ── dense iter_weights sharding ──────────────────────────────────────────────


def _fp8(shape):
    return (torch.randn(*shape) * 0.02).to(FP8)


def _e8m0(shape):
    return torch.randint(120, 134, shape, dtype=torch.uint8).view(E8M0)


CHECKPOINT_CFG = dict(
    vocab_size=131, dim=64, moe_inter_dim=256, n_layers=2, n_hash_layers=1,
    n_heads=2, q_lora_rank=64, head_dim=128, rope_head_dim=64,
    o_groups=2, o_lora_rank=128, compress_ratios=[0, 4], n_routed_experts=2,
)


def _write_dsv4_checkpoint(path, *, cfg_over=None, inter=256):
    """Single-file fake DSV4 checkpoint (config.json + safetensors + index)."""
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(CHECKPOINT_CFG)
    if cfg_over:
        cfg.update(cfg_over)
    (path / "inference").mkdir(exist_ok=True)
    (path / "inference" / "config.json").write_text(json.dumps(cfg))

    sb = inter // 128 if inter % 128 == 0 else -(-inter // 128)
    t = {
        "embed.weight": torch.randn(cfg["vocab_size"], 64),
        "norm.weight": torch.randn(64),
        "head.weight": torch.randn(cfg["vocab_size"], 64),
    }
    for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        t[nm] = torch.randn(4, 4)
    for L in range(cfg["n_layers"]):
        a = f"layers.{L}.attn"
        t[f"{a}.wq_a.weight"] = _fp8((64, 64))
        t[f"{a}.wq_a.scale"] = _e8m0((1, 1))
        t[f"{a}.q_norm.weight"] = torch.randn(64)
        t[f"{a}.wq_b.weight"] = _fp8((256, 64))
        t[f"{a}.wq_b.scale"] = _e8m0((2, 1))
        t[f"{a}.wkv.weight"] = _fp8((128, 64))
        t[f"{a}.wkv.scale"] = _e8m0((1, 1))
        t[f"{a}.kv_norm.weight"] = torch.randn(128)
        t[f"{a}.wo_a.weight"] = _fp8((256, 128))
        t[f"{a}.wo_a.scale"] = _e8m0((2, 1))
        t[f"{a}.wo_b.weight"] = _fp8((64, 256))
        t[f"{a}.wo_b.scale"] = _e8m0((1, 2))
        t[f"{a}.attn_sink"] = torch.randn(2)
        t[f"layers.{L}.attn_norm.weight"] = torch.randn(64)
        t[f"layers.{L}.ffn_norm.weight"] = torch.randn(64)
        g = f"layers.{L}.ffn.gate"
        t[f"{g}.weight"] = torch.randn(2, 64)
        if L < cfg["n_hash_layers"]:
            t[f"{g}.tid2eid"] = torch.randint(0, 2, (cfg["vocab_size"],))
        else:
            t[f"{g}.bias"] = torch.randn(2)
        for proj in ("w1", "w3"):
            t[f"layers.{L}.ffn.shared_experts.{proj}.weight"] = _fp8((inter, 64))
            t[f"layers.{L}.ffn.shared_experts.{proj}.scale"] = _e8m0((sb, 1))
        t[f"layers.{L}.ffn.shared_experts.w2.weight"] = _fp8((64, inter))
        t[f"layers.{L}.ffn.shared_experts.w2.scale"] = _e8m0((1, sb))
        for nm in (
            "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
            "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
        ):
            t[f"layers.{L}.{nm}"] = torch.randn(4, 4)
        if cfg["compress_ratios"][L]:
            c = f"{a}.compressor"
            for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                t[f"{c}.{nm}"] = torch.randn(8, 8)
            idx = f"{a}.indexer"
            t[f"{idx}.wq_b.weight"] = _fp8((64, 64))
            t[f"{idx}.wq_b.scale"] = _e8m0((1, 1))
            t[f"{idx}.weights_proj.weight"] = torch.randn(4, 4)
            for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                t[f"{idx}.compressor.{nm}"] = torch.randn(8, 8)

    from safetensors.torch import save_file

    save_file(t, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in t}})
    )
    return t


def _run_iter_weights(path, rank, ws):
    import freetoken.models.deepseek_v4.weight as dw

    reset_tp_info()
    set_tp_info(rank, ws)
    return dict(dw.iter_weights(str(path), torch.device("cpu"), include_moe_experts=False))


def test_iter_weights_dense_shards_reassemble_per_rank(tmp_path):
    full = _write_dsv4_checkpoint(tmp_path / "model")
    path = tmp_path / "model"
    r0 = _run_iter_weights(path, 0, 2)
    r1 = _run_iter_weights(path, 1, 2)
    assert r0.keys() == r1.keys()

    # vocab: div_ceil window on every rank; the last rank's short tail is
    # zero-padded (the head slices logits back to V after all-gather).
    assert r0["model.embed.weight"].shape == (66, 64)
    assert r1["model.embed.weight"].shape == (66, 64)
    for key, ckpt in (("model.embed.weight", "embed.weight"), ("model.head.weight", "head.weight")):
        joined = torch.cat([r0[key], r1[key]], dim=0)
        assert joined.shape[0] == 132
        assert torch.equal(joined[:131], full[ckpt])
        assert torch.equal(joined[131], torch.zeros(64))

    for L in range(2):
        a = f"model.layers.{L}.attn"
        # wq_b column-parallel: weight + 128-block scale both split rows
        assert torch.equal(
            torch.cat([r0[f"{a}.wq_b.weight"], r1[f"{a}.wq_b.weight"]], dim=0),
            full[f"layers.{L}.attn.wq_b.weight"],
        )
        assert torch.equal(
            torch.cat([r0[f"{a}.wq_b.weight_scale_inv"], r1[f"{a}.wq_b.weight_scale_inv"]], dim=0),
            full[f"layers.{L}.attn.wq_b.scale"],
        )
        # wo_a: bf16 dequant sliced by whole group (o_groups//tp * o_lora_rank rows)
        from freetoken.models.deepseek_v4.weight import _dequant_fp8_block

        deq = _dequant_fp8_block(full[f"layers.{L}.attn.wo_a.weight"], full[f"layers.{L}.attn.wo_a.scale"])
        assert r0[f"{a}.wo_a"].dtype == torch.bfloat16
        assert torch.equal(r0[f"{a}.wo_a"], deq[:128])
        assert torch.equal(torch.cat([r0[f"{a}.wo_a"], r1[f"{a}.wo_a"]], dim=0), deq)
        # wo_b row-parallel: weight + scale split columns
        assert torch.equal(
            torch.cat([r0[f"{a}.wo_b.weight"], r1[f"{a}.wo_b.weight"]], dim=1),
            full[f"layers.{L}.attn.wo_b.weight"],
        )
        assert torch.equal(
            torch.cat([r0[f"{a}.wo_b.weight_scale_inv"], r1[f"{a}.wo_b.weight_scale_inv"]], dim=1),
            full[f"layers.{L}.attn.wo_b.scale"],
        )
        # attn_sink splits per head
        assert torch.equal(
            torch.cat([r0[f"{a}.attn_sink"], r1[f"{a}.attn_sink"]]),
            full[f"layers.{L}.attn.attn_sink"],
        )
        # shared experts: w1/w3 split rows, w2 splits columns
        for proj in ("w1", "w3"):
            dst = f"model.layers.{L}.ffn.shared_experts.{proj}"
            src = f"layers.{L}.ffn.shared_experts.{proj}"
            assert torch.equal(torch.cat([r0[f"{dst}.weight"], r1[f"{dst}.weight"]], dim=0), full[f"{src}.weight"])
            assert torch.equal(
                torch.cat([r0[f"{dst}.weight_scale_inv"], r1[f"{dst}.weight_scale_inv"]], dim=0),
                full[f"{src}.scale"],
            )
        dst = f"model.layers.{L}.ffn.shared_experts.w2"
        src = f"layers.{L}.ffn.shared_experts.w2"
        assert torch.equal(torch.cat([r0[f"{dst}.weight"], r1[f"{dst}.weight"]], dim=1), full[f"{src}.weight"])
        assert torch.equal(
            torch.cat([r0[f"{dst}.weight_scale_inv"], r1[f"{dst}.weight_scale_inv"]], dim=1),
            full[f"{src}.scale"],
        )

        # replicated tensors are identical on every rank (fp8 scale companions included)
        replicated = [
            f"{a}.wq_a.weight", f"{a}.wq_a.weight_scale_inv", f"{a}.q_norm.weight",
            f"{a}.wkv.weight", f"{a}.wkv.weight_scale_inv", f"{a}.kv_norm.weight",
            f"model.layers.{L}.attn_norm.weight", f"model.layers.{L}.ffn_norm.weight",
            f"model.layers.{L}.ffn.gate.weight",
        ]
        replicated.append(
            f"model.layers.{L}.ffn.gate.tid2eid" if L < 1 else f"model.layers.{L}.ffn.gate.bias"
        )
        replicated += [f"model.layers.{L}.{nm}" for nm in (
            "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale")]
        if L == 1:  # the ratio-4 layer's compressor/indexer stay replicated
            replicated += [f"model.layers.{L}.attn.compressor.{nm}" for nm in (
                "ape", "wkv.weight", "wgate.weight", "norm.weight")]
            replicated += [
                f"model.layers.{L}.attn.indexer.wq_b.weight",
                f"model.layers.{L}.attn.indexer.wq_b.weight_scale_inv",
                f"model.layers.{L}.attn.indexer.weights_proj.weight",
            ]
            replicated += [f"model.layers.{L}.attn.indexer.compressor.{nm}" for nm in (
                "ape", "wkv.weight", "wgate.weight", "norm.weight")]
        for key in replicated:
            assert torch.equal(r0[key], r1[key]), key

    assert torch.equal(r0["model.norm.weight"], r1["model.norm.weight"])
    for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        assert torch.equal(r0[f"model.{nm}"], r1[f"model.{nm}"])


def test_iter_weights_rejects_o_groups_not_divisible(tmp_path):
    _write_dsv4_checkpoint(tmp_path / "model", cfg_over={"o_groups": 3})
    with pytest.raises(ValueError, match="o_groups divisible"):
        _run_iter_weights(tmp_path / "model", 0, 2)


def test_iter_weights_rejects_n_heads_not_divisible(tmp_path):
    _write_dsv4_checkpoint(tmp_path / "model", cfg_over={"n_heads": 3})
    with pytest.raises(ValueError, match="n_heads divisible"):
        _run_iter_weights(tmp_path / "model", 0, 2)


def test_iter_weights_rejects_fp8_scale_misaligned_shard(tmp_path):
    _write_dsv4_checkpoint(tmp_path / "model", cfg_over={"moe_inter_dim": 192}, inter=192)
    with pytest.raises(ValueError, match="128-block"):
        _run_iter_weights(tmp_path / "model", 0, 2)


# ── MXFP4 expert pieces sharding ─────────────────────────────────────────────


def _write_experts_checkpoint(path, *, inter=256, n_layers=1, n_experts=2):
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        vocab_size=8, dim=64, moe_inter_dim=inter, n_layers=n_layers,
        n_routed_experts=n_experts, compress_ratios=[0] * n_layers,
    )
    (path / "inference").mkdir(exist_ok=True)
    (path / "inference" / "config.json").write_text(json.dumps(cfg))

    t = {}
    for L in range(n_layers):
        for e in range(n_experts):
            base = f"layers.{L}.ffn.experts.{e}"
            t[f"{base}.w1.weight"] = torch.randint(0, 256, (inter, 32), dtype=torch.uint8)
            t[f"{base}.w1.scale"] = _e8m0((inter, 2))
            t[f"{base}.w3.weight"] = torch.randint(0, 256, (inter, 32), dtype=torch.uint8)
            t[f"{base}.w3.scale"] = _e8m0((inter, 2))
            t[f"{base}.w2.weight"] = torch.randint(0, 256, (64, inter // 2), dtype=torch.uint8)
            t[f"{base}.w2.scale"] = _e8m0((64, inter // 32))
    t["layers.0.attn.wq_a.weight"] = torch.randn(64, 64)  # non-expert decoy: skipped

    from safetensors.torch import save_file

    save_file(t, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in t}})
    )
    return t


def _iter_expert_pieces(path, rank, ws, *, parallel=False):
    import freetoken.models.deepseek_v4.weight as dw
    from freetoken.layers.quantization import QuantKind

    reset_tp_info()
    set_tp_info(rank, ws)
    return list(dw.iter_expert_pieces(str(path), None, QuantKind.MXFP4, parallel=parallel))


# the parallel prefetch reader is O_DIRECT-based (POSIX-only); no O_DIRECT -> skip
import os as _os

_ODIRECT = pytest.mark.skipif(not hasattr(_os, "O_DIRECT"), reason="O_DIRECT prefetch is POSIX-only")


@pytest.mark.parametrize("parallel", [False, pytest.param(True, marks=_ODIRECT)])
def test_iter_expert_pieces_slice_and_reassemble_per_rank(tmp_path, parallel):
    full = _write_experts_checkpoint(tmp_path / "model", inter=256)
    path = tmp_path / "model"

    pieces = [_iter_expert_pieces(path, r, 2, parallel=parallel) for r in range(2)]
    # parallel streams may complete experts out of order: key by (layer, first_expert)
    by_rank = []
    for rank_pieces in pieces:
        assert len(rank_pieces) == 2  # the decoy tensor is skipped
        assert all(b == a + 1 for (_, a, b, p) in rank_pieces)
        by_rank.append({(l, a): p for (l, a, b, p) in rank_pieces})

    for e in range(2):
        p0, p1 = by_rank[0][(0, e)], by_rank[1][(0, e)]
        base = f"layers.0.ffn.experts.{e}"
        # gate/up: rows of the packed weight and of the [I, H//32] scale
        assert p0["gate"].shape == (1, 128, 32)
        assert torch.equal(
            torch.cat([p0["gate"], p1["gate"]], dim=1), full[f"{base}.w1.weight"].unsqueeze(0)
        )
        assert torch.equal(
            torch.cat([p0["gate_scale"], p1["gate_scale"]], dim=1),
            full[f"{base}.w1.scale"].unsqueeze(0),
        )
        assert torch.equal(
            torch.cat([p0["up"], p1["up"]], dim=1), full[f"{base}.w3.weight"].unsqueeze(0)
        )
        assert torch.equal(
            torch.cat([p0["up_scale"], p1["up_scale"]], dim=1),
            full[f"{base}.w3.scale"].unsqueeze(0),
        )
        # down: packed-pair weight columns and 32-block scale columns
        assert p0["down"].shape == (1, 64, 64)
        assert p0["down_scale"].shape == (1, 64, 4)
        assert torch.equal(
            torch.cat([p0["down"], p1["down"]], dim=2), full[f"{base}.w2.weight"].unsqueeze(0)
        )
        assert torch.equal(
            torch.cat([p0["down_scale"], p1["down_scale"]], dim=2),
            full[f"{base}.w2.scale"].unsqueeze(0),
        )


def test_iter_expert_pieces_tp1_passes_tensors_through(tmp_path):
    full = _write_experts_checkpoint(tmp_path / "model", inter=256)
    pieces = _iter_expert_pieces(tmp_path / "model", 0, 1)
    assert len(pieces) == 2
    p = pieces[0][3]
    assert torch.equal(p["gate"], full["layers.0.ffn.experts.0.w1.weight"].unsqueeze(0))
    assert torch.equal(p["down"], full["layers.0.ffn.experts.0.w2.weight"].unsqueeze(0))
    assert torch.equal(p["down_scale"], full["layers.0.ffn.experts.0.w2.scale"].unsqueeze(0))


def test_iter_expert_pieces_rejects_32_misaligned_window(tmp_path):
    import freetoken.models.deepseek_v4.weight as dw
    from freetoken.layers.quantization import QuantKind

    _write_experts_checkpoint(tmp_path / "model", inter=96)
    reset_tp_info()
    set_tp_info(0, 2)
    with pytest.raises(ValueError, match=r"32\*tp_size"):
        dw.iter_expert_pieces(str(tmp_path / "model"), None, QuantKind.MXFP4)


def test_iter_expert_pieces_rejects_non_mxfp4_kind(tmp_path):
    import freetoken.models.deepseek_v4.weight as dw
    from freetoken.layers.quantization import QuantKind

    assert dw.iter_expert_pieces(str(tmp_path), None, QuantKind.NONE) is None


# ── MXFP4 kernel TP gating ──────────────────────────────────────────────────


def _moe_cfg(tp_size, intermediate=256, has_bias=False):
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(
        num_experts=2, intermediate=intermediate, hidden=64, top_k=2,
        tp_size=tp_size, tp_rank=0, strategy="offload",
        activation="silu", alpha=1.0, beta=0.0, limit=None, has_bias=has_bias,
        apply_router_weight_on_input=False, interleaved=False,
    )


def test_mxfp4_kernel_accepts_aligned_tp_and_sizes_banks_locally():
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    k = TritonMxfp4MoEKernel()
    cfg = _moe_cfg(tp_size=2, intermediate=256)
    assert k.unusable_reason(cfg) is None
    layout = k.layout(cfg)
    # banks use the LOCAL intermediate: fused gate|up rows, packed down columns
    assert layout["gate_up"].shape == (2 * 128, 32)
    assert layout["down"].shape == (64, 64)
    assert layout["down_scale"].shape == (64, 4)


def test_mxfp4_kernel_rejects_32_misaligned_tp():
    from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

    k = TritonMxfp4MoEKernel()
    assert k.unusable_reason(_moe_cfg(tp_size=1)) is None
    reason = k.unusable_reason(_moe_cfg(tp_size=2, intermediate=96))
    assert reason is not None and "32*tp_size" in reason
