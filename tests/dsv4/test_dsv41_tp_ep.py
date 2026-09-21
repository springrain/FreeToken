"""DeepSeek-V4.1 rank-local text/vision checkpoint loading."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed.info import DistributedInfo
from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.config import VisionConfig

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


@pytest.fixture(autouse=True)
def _portable_drop_page_cache(monkeypatch):
    monkeypatch.setattr(
        "freetoken.models.deepseek_v41.weight.drop_page_cache", lambda _path: None
    )


@contextmanager
def _tp(rank: int, size: int):
    from freetoken.distributed import info

    previous = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, size)
    try:
        yield
    finally:
        info._TP_INFO = previous


def _args() -> DeepseekV41Args:
    return DeepseekV41Args(
        vocab_size=131,
        dim=256,
        moe_inter_dim=256,
        n_layers=1,
        n_heads=8,
        n_routed_experts=8,
        n_activated_experts=2,
        q_lora_rank=256,
        head_dim=64,
        rope_head_dim=32,
        o_groups=8,
        o_lora_rank=64,
        compress_ratios=(2,),
        kv_source_layers=(0,),
        index_source_layers=(0,),
        index_n_heads=8,
        index_head_dim=64,
        candidate_source_layer=99,
        vision_n_layers=1,
        vision_dim=64,
        vision_n_heads=8,
        vision_inter_dim=256,
        vision_patch_size=2,
        vision_downsample_ratio=2,
        vision_max_n_token=16,
        vision_min_pixels=16,
        image_token_id=129,
    )


def _vision_config(args: DeepseekV41Args) -> VisionConfig:
    return VisionConfig(
        num_layers=args.vision_n_layers,
        hidden_size=args.vision_dim,
        num_heads=args.vision_n_heads,
        intermediate_size=args.vision_inter_dim,
        patch_size=args.vision_patch_size,
        rope_theta=args.vision_rope_theta,
        downsample_ratio=args.vision_downsample_ratio,
        max_image_tokens=args.vision_max_n_token,
        min_pixels=args.vision_min_pixels,
        max_wh_ratio=args.vision_max_wh_ratio,
        out_hidden_size=args.dim,
        image_token_id=args.image_token_id,
    )


def _fp8(shape, offset=0):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    return (((values + offset) % 31 - 15).reshape(shape) / 16).to(FP8)


def _e8m0(shape, code=127):
    return torch.full(shape, code, dtype=torch.uint8).view(E8M0)


def _bf16(shape, offset=0):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    return (((values + offset) % 37 - 18).reshape(shape) / 32).to(torch.bfloat16)


def _add_fp8_linear(tensors, name, shape, offset):
    tensors[f"{name}.weight"] = _fp8(shape, offset)
    tensors[f"{name}.scale"] = _e8m0((shape[0] // 32, shape[1] // 32))


def _write_checkpoint(path):
    from safetensors.torch import save_file

    args = _args()
    path.mkdir(parents=True)
    (path / "inference").mkdir()
    raw_config = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in vars(args).items()
        if name not in {"max_batch_size", "max_seq_len"}
    }
    (path / "inference" / "config.json").write_text(json.dumps(raw_config))

    tensors = {
        "embed.weight": _bf16((args.vocab_size, args.dim)),
        "norm.weight": _bf16((args.dim,), 1),
        "head.weight": _bf16((args.vocab_size, args.dim), 2),
    }
    layer = "layers.0"
    attn = f"{layer}.attn"
    _add_fp8_linear(tensors, f"{attn}.wq_a", (args.q_lora_rank, args.dim), 3)
    tensors[f"{attn}.q_norm.weight"] = _bf16((args.q_lora_rank,), 4)
    _add_fp8_linear(
        tensors,
        f"{attn}.wq_b",
        (args.n_heads * args.head_dim, args.q_lora_rank),
        5,
    )
    _add_fp8_linear(tensors, f"{attn}.wkv", (args.head_dim, args.dim), 6)
    tensors[f"{attn}.kv_norm.weight"] = _bf16((args.head_dim,), 7)
    group_k = args.n_heads * args.head_dim // args.o_groups
    _add_fp8_linear(
        tensors,
        f"{attn}.wo_a",
        (args.o_groups * args.o_lora_rank, group_k),
        8,
    )
    _add_fp8_linear(
        tensors,
        f"{attn}.wo_b",
        (args.dim, args.o_groups * args.o_lora_rank),
        9,
    )
    tensors[f"{attn}.attn_sink"] = torch.arange(args.n_heads, dtype=torch.float32)

    compressor = f"{attn}.compressor"
    tensors[f"{compressor}.wkv.weight"] = torch.randn(args.head_dim, args.dim)
    tensors[f"{compressor}.wgate.weight"] = torch.randn(args.head_dim, args.dim)
    tensors[f"{compressor}.norm.weight"] = _bf16((args.head_dim,), 11)
    indexer = f"{attn}.indexer"
    _add_fp8_linear(
        tensors,
        f"{indexer}.wq_b",
        (args.index_n_heads * args.index_head_dim, args.q_lora_rank),
        12,
    )
    tensors[f"{indexer}.weights_proj.weight"] = _bf16(
        (args.index_n_heads, args.dim), 13
    )
    tensors[f"{indexer}.wk.weight"] = _bf16((args.index_head_dim, args.head_dim), 14)
    tensors[f"{indexer}.k_norm.weight"] = _bf16((args.index_head_dim,), 15)

    tensors[f"{layer}.attn_norm.weight"] = _bf16((args.dim,), 16)
    tensors[f"{layer}.ffn_norm.weight"] = _bf16((args.dim,), 17)
    tensors[f"{layer}.ffn.gate.weight"] = _bf16((args.n_routed_experts, args.dim), 18)
    tensors[f"{layer}.ffn.gate.bias"] = torch.arange(
        args.n_routed_experts, dtype=torch.float32
    )
    tensors[f"{layer}.ffn.gate.bias_vl"] = (
        torch.arange(args.n_routed_experts, dtype=torch.float32) + 10
    )
    for projection, offset in (("w1", 19), ("w3", 20)):
        _add_fp8_linear(
            tensors,
            f"{layer}.ffn.shared_experts.{projection}",
            (args.moe_inter_dim, args.dim),
            offset,
        )
    _add_fp8_linear(
        tensors,
        f"{layer}.ffn.shared_experts.w2",
        (args.dim, args.moe_inter_dim),
        21,
    )
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        tensors[f"{layer}.{name}"] = torch.randn(
            (2 + args.hc_mult) * args.hc_mult, args.hc_mult * args.dim
        )
    for name in ("hc_attn_base", "hc_ffn_base"):
        tensors[f"{layer}.{name}"] = torch.randn((2 + args.hc_mult) * args.hc_mult)
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        tensors[f"{layer}.{name}"] = torch.randn(3)

    vd = args.vision_dim
    vi = args.vision_inter_dim
    patch_dim = 3 * args.vision_patch_size**2
    tensors.update(
        {
            "vision.patch_embed.proj.weight": _bf16((vd, patch_dim), 22),
            "vision.patch_embed.proj.bias": _bf16((vd,), 23),
            "vision.blocks.0.norm1.weight": _bf16((vd,), 24),
            "vision.blocks.0.attn.wqkv.weight": _bf16((3 * vd, vd), 25),
            "vision.blocks.0.attn.wqkv.bias": _bf16((3 * vd,), 26),
            "vision.blocks.0.attn.wo.weight": _bf16((vd, vd), 27),
            "vision.blocks.0.attn.wo.bias": _bf16((vd,), 28),
            "vision.blocks.0.norm2.weight": _bf16((vd,), 29),
            "vision.blocks.0.mlp.w1.weight": _bf16((2 * vi, vd), 30),
            "vision.blocks.0.mlp.w2.weight": _bf16((vd, vi), 31),
            "vision.norm.weight": _bf16((vd,), 32),
            "aligner.w1.weight": _bf16(
                (args.dim, vd * args.vision_downsample_ratio**2), 33
            ),
            "aligner.w1.bias": _bf16((args.dim,), 34),
            "aligner.w2.weight": _bf16((args.dim, args.dim), 35),
            "aligner.w2.bias": _bf16((args.dim,), 36),
            "image_start": _bf16((args.dim,), 37),
            "image_newline": _bf16((args.dim,), 38),
            "image_end": _bf16((args.dim,), 39),
        }
    )

    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}})
    )
    return tensors


def _model_config(args: DeepseekV41Args, tp_size: int):
    from freetoken.layers.quantization.configs.fp8 import Fp8BlockConfig

    quant = Fp8BlockConfig(
        {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        SimpleNamespace(expert_dtype="fp4"),
        unquantized=(
            "head",
            "vision.*",
            "aligner.*",
            "*.compressor.wkv",
            "*.compressor.wgate",
            "*.indexer.weights_proj",
            "*.indexer.wk",
        ),
    )
    return SimpleNamespace(
        dsv4_args=args,
        quant=quant,
        moe_strategy="offload",
        decode_target="gpu",
        moe_ep_size=tp_size,
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        is_multimodal=True,
        vision_config=_vision_config(args),
    )


def _rank_weights(path, rank: int, size: int):
    from freetoken.models.deepseek_v41.weight import iter_weights

    args = _args()
    with _tp(rank, size):
        return dict(
            iter_weights(
                str(path),
                torch.device("cpu"),
                include_moe_experts=False,
                tp_shard=True,
                config=_model_config(args, size),
            )
        )


def _join_grouped(ranks, key: str, groups: int):
    parts = [rank[key].chunk(groups, dim=0) for rank in ranks]
    return torch.cat(
        [
            torch.cat([rank_parts[group] for rank_parts in parts], dim=0)
            for group in range(groups)
        ],
        dim=0,
    )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_v41_text_fp8_block32_shards_reassemble(tmp_path, tp_size):
    full = _write_checkpoint(tmp_path / "model")
    ranks = [
        _rank_weights(tmp_path / "model", rank, tp_size) for rank in range(tp_size)
    ]
    attn = "model.layers.0.attn"

    for dst, src in (
        ("model.embed.weight", "embed.weight"),
        ("head.weight", "head.weight"),
    ):
        joined = torch.cat([rank[dst] for rank in ranks], dim=0)
        assert torch.equal(joined[: full[src].shape[0]], full[src])
        assert torch.count_nonzero(joined[full[src].shape[0] :]) == 0

    for dst, src, dim in (
        (f"{attn}.wq_b.weight", "layers.0.attn.wq_b.weight", 0),
        (f"{attn}.wq_b.weight_scale_inv", "layers.0.attn.wq_b.scale", 0),
        (f"{attn}.wo_b.weight", "layers.0.attn.wo_b.weight", 1),
        (f"{attn}.wo_b.weight_scale_inv", "layers.0.attn.wo_b.scale", 1),
        (
            f"{attn}.indexer.wq_b.weight",
            "layers.0.attn.indexer.wq_b.weight",
            0,
        ),
        (
            f"{attn}.indexer.wq_b.weight_scale_inv",
            "layers.0.attn.indexer.wq_b.scale",
            0,
        ),
        (
            "model.layers.0.ffn.shared_experts.w2.weight",
            "layers.0.ffn.shared_experts.w2.weight",
            1,
        ),
        (
            "model.layers.0.ffn.shared_experts.w2.weight_scale_inv",
            "layers.0.ffn.shared_experts.w2.scale",
            1,
        ),
    ):
        assert torch.equal(torch.cat([rank[dst] for rank in ranks], dim=dim), full[src])

    for projection in ("w1", "w3"):
        dst = f"model.layers.0.ffn.shared_experts.{projection}"
        src = f"layers.0.ffn.shared_experts.{projection}"
        assert torch.equal(
            torch.cat([rank[f"{dst}.weight"] for rank in ranks]),
            full[f"{src}.weight"],
        )
        assert torch.equal(
            torch.cat([rank[f"{dst}.weight_scale_inv"] for rank in ranks]),
            full[f"{src}.scale"],
        )

    from freetoken.models.deepseek_v41.weight import _dequant_fp8_block

    expected_wo_a = _dequant_fp8_block(
        full["layers.0.attn.wo_a.weight"], full["layers.0.attn.wo_a.scale"]
    )
    assert torch.equal(
        torch.cat([rank[f"{attn}.wo_a"] for rank in ranks], dim=0), expected_wo_a
    )
    assert torch.equal(
        torch.cat([rank[f"{attn}.attn_sink"] for rank in ranks]),
        full["layers.0.attn.attn_sink"],
    )

    for key in (
        f"{attn}.wq_a.weight",
        f"{attn}.wq_a.weight_scale_inv",
        f"{attn}.wkv.weight",
        f"{attn}.wkv.weight_scale_inv",
        f"{attn}.compressor.wkv_weight",
        f"{attn}.compressor.wgate_weight",
        f"{attn}.indexer.wk.weight",
    ):
        assert all(torch.equal(rank[key], ranks[0][key]) for rank in ranks[1:])


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_v41_vision_and_aligner_shards_reassemble(tmp_path, tp_size):
    full = _write_checkpoint(tmp_path / "model")
    ranks = [
        _rank_weights(tmp_path / "model", rank, tp_size) for rank in range(tp_size)
    ]

    for suffix in ("weight", "bias"):
        key = f"vision.blocks.0.attn.wqkv.{suffix}"
        assert torch.equal(_join_grouped(ranks, key, 3), full[key])
    key = "vision.blocks.0.mlp.w1.weight"
    assert torch.equal(_join_grouped(ranks, key, 2), full[key])

    for key in ("aligner.w1.weight", "aligner.w1.bias"):
        assert torch.equal(torch.cat([rank[key] for rank in ranks], dim=0), full[key])
    for key in (
        "vision.blocks.0.attn.wo.weight",
        "vision.blocks.0.mlp.w2.weight",
        "aligner.w2.weight",
    ):
        assert torch.equal(torch.cat([rank[key] for rank in ranks], dim=1), full[key])
    for key in ("vision.blocks.0.attn.wo.bias", "aligner.w2.bias"):
        assert torch.equal(sum(rank[key] for rank in ranks), full[key])
        assert all(torch.count_nonzero(rank[key]) == 0 for rank in ranks[1:])

    for key in (
        "vision.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias",
        "vision.blocks.0.norm1.weight",
        "vision.blocks.0.norm2.weight",
        "vision.norm.weight",
        "image_start",
        "image_newline",
        "image_end",
    ):
        assert all(torch.equal(rank[key], full[key]) for rank in ranks)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_v41_loader_state_matches_tp_model_and_loads_strictly(tmp_path, tp_size):
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from freetoken.models.deepseek_v41.weight import iter_weights
    from freetoken.utils.torch_utils import torch_dtype

    path = tmp_path / "model"
    _write_checkpoint(path)
    args = _args()
    config = _model_config(args, tp_size)
    with _tp(tp_size - 1, tp_size):
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            model = DeepseekV41ForCausalLM(config)
        model_state = model.state_dict()
        loaded = _materialize_loaded_weight_state_dict(
            model_state,
            iter_weights(
                str(path),
                torch.device("cpu"),
                include_moe_experts=False,
                tp_shard=True,
                config=config,
            ),
            device=torch.device("cpu"),
        )

    assert loaded.keys() == model_state.keys()
    assert {name: tensor.shape for name, tensor in loaded.items()} == {
        name: tensor.shape for name, tensor in model_state.items()
    }
    assert {name: tensor.dtype for name, tensor in loaded.items()} == {
        name: tensor.dtype for name, tensor in model_state.items()
    }
    model.load_state_dict(loaded)
    assert model.model.layers.op_list[0].ffn.experts.expert_tp_size == 1
