"""DeepSeek-V4 Flash/Pro dense TP and owner-local routed-expert loading."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.layers.quantization import QuantKind
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.weight import _dequant_fp8_block
from freetoken.moe.ownership import ExpertOwnership


@pytest.fixture(autouse=True)
def _portable_drop_page_cache(monkeypatch):
    """Windows has no posix_fadvise; page-cache eviction is irrelevant to tiny fixtures."""
    monkeypatch.setattr("freetoken.models.deepseek_v4.weight.drop_page_cache", lambda _path: None)


@contextmanager
def _tp(rank: int, size: int):
    """Temporarily replace the process-global TP geometry for a CPU unit test."""
    import freetoken.distributed.info as info

    previous = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, size)
    try:
        yield
    finally:
        info._TP_INFO = previous


def _args(**overrides) -> DeepseekV4Args:
    values = dict(
        vocab_size=131,
        dim=128,
        moe_inter_dim=1024,
        n_layers=1,
        n_hash_layers=0,
        n_heads=8,
        n_routed_experts=8,
        n_activated_experts=2,
        q_lora_rank=128,
        head_dim=128,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=128,
        index_n_heads=8,
        index_head_dim=128,
        compress_ratios=(0, 0),
    )
    values.update(overrides)
    return DeepseekV4Args(**values)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_attention_uses_rank_local_heads_and_output_groups(tp_size):
    from freetoken.models.deepseek_v4.attention import Attention

    with _tp(0, tp_size):
        attention = Attention(0, _args())

    assert attention.n_heads_local == 8 // tp_size
    assert attention.n_groups_local == 8 // tp_size
    assert attention.attn_sink.shape == (8 // tp_size,)
    assert attention.wq_b.weight.shape == (8 * 128 // tp_size, 128)
    assert attention.wo_a.shape == (8 * 128 // tp_size, 128)
    assert attention.wo_b.weight.shape == (128, 8 * 128 // tp_size)


@pytest.mark.parametrize(
    ("family", "values"),
    [
        (
            "flash",
            dict(dim=4096, moe_inter_dim=2048, n_layers=43, n_heads=64,
                 n_routed_experts=256, q_lora_rank=1024, o_groups=8),
        ),
        (
            "pro",
            dict(dim=7168, moe_inter_dim=3072, n_layers=61, n_heads=128,
                 n_routed_experts=384, q_lora_rank=1536, o_groups=16),
        ),
    ],
)
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_released_flash_and_pro_geometry_supports_tp2_tp4_tp8(family, values, tp_size):
    args = _args(**values)

    assert args.n_heads % tp_size == 0, family
    assert args.o_groups % tp_size == 0, family
    assert args.n_routed_experts % tp_size == 0, family
    assert args.moe_inter_dim // tp_size % 128 == 0, family
    assert args.o_groups * args.o_lora_rank // tp_size % 128 == 0, family


def test_parse_config_accepts_official_v4_pro_geometry(tmp_path):
    from freetoken.models.deepseek_v4.config import parse_config

    (tmp_path / "inference").mkdir()
    pro = dict(
        vocab_size=129280,
        dim=7168,
        moe_inter_dim=3072,
        n_layers=61,
        n_hash_layers=3,
        n_heads=128,
        n_routed_experts=384,
        n_shared_experts=1,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        route_scale=2.5,
        swiglu_limit=10.0,
        q_lora_rank=1536,
        head_dim=512,
        rope_head_dim=64,
        o_groups=16,
        o_lora_rank=1024,
        window_size=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        compress_ratios=[128, 128] + [4, 128] * 29 + [4],
    )
    (tmp_path / "inference" / "config.json").write_text(json.dumps(pro))

    config = parse_config(SimpleNamespace(
        _name_or_path=str(tmp_path), max_position_embeddings=1_048_576
    ))

    assert config.model_type == "deepseek_v4"
    assert config.architectures == ["DeepseekV4ForCausalLM"]
    assert (config.hidden_size, config.num_layers) == (7168, 61)
    assert (config.num_qo_heads, config.num_experts) == (128, 384)
    assert config.moe_intermediate_size == 3072
    assert config.dsv4_args.o_groups == 16
    assert config.dsv4_args.index_topk == 1024


FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _fp8(shape, offset=0):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    values = ((values + offset) % 31 - 15).reshape(shape) / 16
    return values.to(FP8)


def _e8m0(shape, code=127):
    return torch.full(shape, code, dtype=torch.uint8).view(E8M0)


def _write_dense_checkpoint(path):
    args = _args()
    path.mkdir(parents=True)
    (path / "inference").mkdir()
    config = {
        name: value
        for name, value in vars(args).items()
        if name not in {"max_batch_size", "max_seq_len"}
    }
    config["compress_ratios"] = list(args.compress_ratios)
    (path / "inference" / "config.json").write_text(json.dumps(config))

    tensors = {
        "embed.weight": torch.arange(args.vocab_size * args.dim, dtype=torch.float32).reshape(args.vocab_size, args.dim),
        "norm.weight": torch.randn(args.dim),
        "head.weight": torch.arange(args.vocab_size * args.dim, dtype=torch.float32).reshape(args.vocab_size, args.dim) + 1,
        "hc_head_fn": torch.randn(args.hc_mult, args.hc_mult * args.dim),
        "hc_head_base": torch.randn(args.hc_mult),
        "hc_head_scale": torch.randn(1),
    }
    layer = "layers.0"
    attn = f"{layer}.attn"
    tensors.update({
        f"{attn}.wq_a.weight": _fp8((128, 128), 1),
        f"{attn}.wq_a.scale": _e8m0((1, 1)),
        f"{attn}.q_norm.weight": torch.randn(128),
        f"{attn}.wq_b.weight": _fp8((1024, 128), 2),
        f"{attn}.wq_b.scale": _e8m0((8, 1)),
        f"{attn}.wkv.weight": _fp8((128, 128), 3),
        f"{attn}.wkv.scale": _e8m0((1, 1)),
        f"{attn}.kv_norm.weight": torch.randn(128),
        f"{attn}.wo_a.weight": _fp8((1024, 128), 4),
        f"{attn}.wo_a.scale": _e8m0((8, 1)),
        f"{attn}.wo_b.weight": _fp8((128, 1024), 5),
        f"{attn}.wo_b.scale": _e8m0((1, 8)),
        f"{attn}.attn_sink": torch.arange(8, dtype=torch.float32),
        f"{layer}.attn_norm.weight": torch.randn(128),
        f"{layer}.ffn_norm.weight": torch.randn(128),
        f"{layer}.ffn.gate.weight": torch.randn(8, 128),
        f"{layer}.ffn.gate.bias": torch.randn(8),
    })
    for projection, offset in (("w1", 6), ("w3", 7)):
        base = f"{layer}.ffn.shared_experts.{projection}"
        tensors[f"{base}.weight"] = _fp8((1024, 128), offset)
        tensors[f"{base}.scale"] = _e8m0((8, 1))
    base = f"{layer}.ffn.shared_experts.w2"
    tensors[f"{base}.weight"] = _fp8((128, 1024), 8)
    tensors[f"{base}.scale"] = _e8m0((1, 8))
    for name in (
        "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
        "hc_attn_scale", "hc_ffn_scale",
    ):
        if name.endswith("_fn"):
            shape = ((2 + args.hc_mult) * args.hc_mult, args.hc_mult * args.dim)
        elif name.endswith("_base"):
            shape = ((2 + args.hc_mult) * args.hc_mult,)
        else:
            shape = (3,)
        tensors[f"{layer}.{name}"] = torch.randn(*shape)

    from safetensors.torch import save_file

    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}
    }))
    return tensors


def _dense_rank(path, rank, size):
    from freetoken.models.deepseek_v4.weight import iter_weights

    with _tp(rank, size):
        return dict(iter_weights(
            str(path), torch.device("cpu"), include_moe_experts=False,
            include_non_moe=True, tp_shard=True,
        ))


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_dense_weight_shards_reassemble_for_tp2_tp4_tp8(tmp_path, tp_size):
    full = _write_dense_checkpoint(tmp_path / "model")
    ranks = [_dense_rank(tmp_path / "model", rank, tp_size) for rank in range(tp_size)]
    assert all(rank.keys() == ranks[0].keys() for rank in ranks)

    for dst, src in (("model.embed.weight", "embed.weight"), ("model.head.weight", "head.weight")):
        joined = torch.cat([rank[dst] for rank in ranks], dim=0)
        assert torch.equal(joined[:131], full[src])
        assert torch.count_nonzero(joined[131:]) == 0

    attn = "model.layers.0.attn"
    assert torch.equal(
        torch.cat([rank[f"{attn}.wq_b.weight"] for rank in ranks], dim=0),
        full["layers.0.attn.wq_b.weight"],
    )
    assert torch.equal(
        torch.cat([rank[f"{attn}.wq_b.weight_scale_inv"] for rank in ranks], dim=0),
        full["layers.0.attn.wq_b.scale"],
    )
    assert torch.equal(
        torch.cat([rank[f"{attn}.wo_b.weight"] for rank in ranks], dim=1),
        full["layers.0.attn.wo_b.weight"],
    )
    assert torch.equal(
        torch.cat([rank[f"{attn}.wo_b.weight_scale_inv"] for rank in ranks], dim=1),
        full["layers.0.attn.wo_b.scale"],
    )
    expected_wo_a = _dequant_fp8_block(
        full["layers.0.attn.wo_a.weight"], full["layers.0.attn.wo_a.scale"]
    )
    assert torch.equal(torch.cat([rank[f"{attn}.wo_a"] for rank in ranks]), expected_wo_a)
    assert torch.equal(
        torch.cat([rank[f"{attn}.attn_sink"] for rank in ranks]),
        full["layers.0.attn.attn_sink"],
    )

    for projection in ("w1", "w3"):
        dst = f"model.layers.0.ffn.shared_experts.{projection}"
        src = f"layers.0.ffn.shared_experts.{projection}"
        assert torch.equal(
            torch.cat([rank[f"{dst}.weight"] for rank in ranks], dim=0),
            full[f"{src}.weight"],
        )
        assert torch.equal(
            torch.cat([rank[f"{dst}.weight_scale_inv"] for rank in ranks], dim=0),
            full[f"{src}.scale"],
        )
    dst = "model.layers.0.ffn.shared_experts.w2"
    src = "layers.0.ffn.shared_experts.w2"
    assert torch.equal(
        torch.cat([rank[f"{dst}.weight"] for rank in ranks], dim=1),
        full[f"{src}.weight"],
    )
    assert torch.equal(
        torch.cat([rank[f"{dst}.weight_scale_inv"] for rank in ranks], dim=1),
        full[f"{src}.scale"],
    )

    for key in (
        f"{attn}.wq_a.weight", f"{attn}.wq_a.weight_scale_inv",
        f"{attn}.wkv.weight", f"{attn}.wkv.weight_scale_inv",
        "model.layers.0.ffn.gate.weight", "model.layers.0.ffn.gate.bias",
    ):
        assert all(torch.equal(rank[key], ranks[0][key]) for rank in ranks[1:]), key


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_dense_loader_shapes_match_owner_ep_model_state(tmp_path, tp_size):
    from freetoken.layers.quantization.configs.fp8 import Fp8BlockConfig
    from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM

    path = tmp_path / "model"
    _write_dense_checkpoint(path)
    quant = Fp8BlockConfig(
        {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
        SimpleNamespace(expert_dtype="fp4"),
        unquantized=("model.head",),
    )
    with _tp(0, tp_size):
        model = DeepseekV4ForCausalLM(SimpleNamespace(
            dsv4_args=_args(),
            quant=quant,
            moe_strategy="offload",
            decode_target="gpu",
            moe_ep_size=tp_size,
        ))
        loaded = _dense_rank(path, 0, tp_size)

    state = model.state_dict()
    assert loaded.keys() == state.keys()
    assert {name: tensor.shape for name, tensor in loaded.items()} == {
        name: tensor.shape for name, tensor in state.items()
    }
    assert model.model.layers.op_list[0].ffn.experts.expert_tp_size == 1


def _write_expert_checkpoint(path, *, layers=2, experts=8, hidden=64, intermediate=64):
    path.mkdir(parents=True)
    (path / "inference").mkdir()
    (path / "inference" / "config.json").write_text(json.dumps({
        "vocab_size": 32,
        "dim": hidden,
        "moe_inter_dim": intermediate,
        "n_layers": layers,
        "n_routed_experts": experts,
        "compress_ratios": [0] * (layers + 1),
    }))
    tensors = {}
    for layer in range(layers):
        for expert in range(experts):
            marker = 10 * layer + expert
            base = f"layers.{layer}.ffn.experts.{expert}"
            for projection, delta in (("w1", 1), ("w3", 2)):
                tensors[f"{base}.{projection}.weight"] = torch.full(
                    (intermediate, hidden // 2), marker + delta, dtype=torch.uint8
                )
                tensors[f"{base}.{projection}.scale"] = _e8m0(
                    (intermediate, hidden // 32), 120 + delta
                )
            tensors[f"{base}.w2.weight"] = torch.full(
                (hidden, intermediate // 2), marker + 3, dtype=torch.uint8
            )
            tensors[f"{base}.w2.scale"] = _e8m0(
                (hidden, intermediate // 32), 123
            )

    from safetensors.torch import save_file

    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}
    }))


@pytest.mark.parametrize(("tp_size", "rank"), [(2, 1), (4, 2), (8, 7)])
def test_owner_expert_reader_keeps_whole_local_experts_and_renumbers(tmp_path, tp_size, rank):
    from freetoken.models.deepseek_v4.weight import iter_expert_pieces

    path = tmp_path / "model"
    _write_expert_checkpoint(path)
    ownership = ExpertOwnership(8, tp_size, rank)
    with _tp(rank, tp_size):
        pieces = list(iter_expert_pieces(
            str(path), SimpleNamespace(num_experts=8), QuantKind.MXFP4,
            ownership=ownership,
        ))

    local_experts = 8 // tp_size
    assert len(pieces) == 2 * local_experts
    assert {(layer, local) for layer, local, end, piece in pieces if end == local + 1} == {
        (layer, local) for layer in range(2) for local in range(local_experts)
    }
    for layer, local, end, piece in pieces:
        global_expert = ownership.global_start + local
        marker = 10 * layer + global_expert
        assert end == local + 1
        assert piece["gate"].shape == (1, 64, 32)
        assert piece["down"].shape == (1, 64, 32)
        assert torch.all(piece["gate"] == marker + 1)
        assert torch.all(piece["up"] == marker + 2)
        assert torch.all(piece["down"] == marker + 3)


def test_tp_expert_reader_requires_owner_geometry(tmp_path):
    from freetoken.models.deepseek_v4.weight import iter_expert_pieces

    path = tmp_path / "model"
    _write_expert_checkpoint(path)
    with _tp(0, 4), pytest.raises(NotImplementedError, match="owner-local EP"):
        iter_expert_pieces(str(path), SimpleNamespace(num_experts=8), QuantKind.MXFP4)


def test_owner_expert_parallel_reader_filters_and_renumbers(tmp_path, monkeypatch):
    import safetensors

    from freetoken.models.deepseek_v4.weight import iter_expert_pieces

    path = tmp_path / "model"
    _write_expert_checkpoint(path)
    selected = []

    def fake_parallel(model_path, wanted, **_kwargs):
        with safetensors.safe_open(
            str(path / "model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            for name in handle.keys():
                if wanted(name):
                    selected.append(name)
                    yield name, handle.get_tensor(name)

    monkeypatch.setattr(
        "freetoken.models.weight.iter_expert_tensors_parallel", fake_parallel
    )
    ownership = ExpertOwnership(8, 4, 2)  # global experts [4, 6) -> local [0, 2)
    with _tp(2, 4):
        pieces = list(iter_expert_pieces(
            str(path), SimpleNamespace(num_experts=8), QuantKind.MXFP4,
            parallel=True, ownership=ownership,
        ))

    assert len(selected) == 2 * 2 * 6
    assert all(".experts.4." in name or ".experts.5." in name for name in selected)
    assert {(layer, local) for layer, local, _, _ in pieces} == {
        (0, 0), (0, 1), (1, 0), (1, 1)
    }


def test_owner_moe_reduces_shared_and_routed_partials_together():
    from freetoken.models.deepseek_v4.moe import MoE

    with _tp(0, 4):
        moe = MoE(0, _args(moe_inter_dim=128), expert_tp_size=1)
    moe.experts.owner_cache = object()
    calls = []
    moe.gate.forward = lambda x, ids: (
        torch.ones(x.shape[0], 1), torch.zeros(x.shape[0], 1, dtype=torch.int64)
    )
    moe.shared_experts.forward = lambda x, *, reduce: calls.append(("shared", reduce)) or torch.full_like(x, 2)
    moe.experts.routed_forward = (
        lambda x, weights, ids, *, reduce: calls.append(("routed", reduce)) or torch.full_like(x, 3)
    )
    moe.experts._maybe_all_reduce = lambda x: calls.append(("reduce", True)) or (x + 7)

    output = moe.forward(torch.zeros(2, 128), torch.zeros(2, dtype=torch.int64))

    assert calls == [("shared", False), ("routed", False), ("reduce", True)]
    assert torch.equal(output, torch.full((2, 128), 12.0))


def test_aot_table_covers_flash_and_pro_shapes():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    models = {model.name: model for model in SUPPORTED_MODELS}
    flash = models["deepseek-ai/DeepSeek-V4-Flash"]
    pro = models["deepseek-ai/DeepSeek-V4-Pro"]
    assert (flash.hidden_size, flash.moe_intermediate_size, flash.top_k) == (4096, 2048, 6)
    assert (pro.hidden_size, pro.moe_intermediate_size, pro.top_k) == (7168, 3072, 6)
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in flash.aliases
    assert "deepseek-ai/DeepSeek-V4-Pro-0813" in pro.aliases
