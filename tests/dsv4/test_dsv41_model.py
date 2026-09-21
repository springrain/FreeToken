"""DeepSeek-V4.1 model wiring: TP geometry, CSA2 sharing and state layout."""

from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed.info import DistributedInfo
from freetoken.models.deepseek_v41.args import DeepseekV41Args


@contextmanager
def _tp(rank: int, size: int):
    import freetoken.distributed.info as info

    previous = info._TP_INFO
    info._TP_INFO = DistributedInfo(rank, size)
    try:
        yield
    finally:
        info._TP_INFO = previous


def _args(**overrides):
    values = dict(
        vocab_size=128,
        dim=128,
        moe_inter_dim=128,
        n_layers=6,
        n_mtp_layers=0,
        n_heads=8,
        n_routed_experts=8,
        n_activated_experts=2,
        q_lora_rank=128,
        head_dim=32,
        rope_head_dim=16,
        o_groups=8,
        o_lora_rank=128,
        window_size=8,
        compress_ratios=(0, 0, 2, 2, 1, 1),
        kv_source_layers=(2, 4),
        index_source_layers=(2, 3, 4, 5),
        index_n_heads=8,
        index_head_dim=32,
        index_topk=8,
        candidate_source_layer=4,
        candidate_topk_blocks=4,
        candidate_block_size=2,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        dspark_block_size=0,
        dspark_target_layer_ids=(),
        vision_n_layers=0,
    )
    values.update(overrides)
    return DeepseekV41Args(**values)


def _config(args, *, tp_size, multimodal=False):
    vision = None
    if multimodal:
        from freetoken.models.deepseek_v41.config import VisionConfig

        vision = VisionConfig(
            num_layers=1,
            hidden_size=32,
            num_heads=8,
            intermediate_size=32,
            patch_size=2,
            rope_theta=10000.0,
            downsample_ratio=1,
            max_image_tokens=16,
            min_pixels=4,
            max_wh_ratio=None,
            out_hidden_size=args.dim,
            image_token_id=args.image_token_id,
        )
    return SimpleNamespace(
        dsv4_args=args,
        quant=None,
        moe_strategy="offload",
        decode_target="gpu",
        moe_ep_size=tp_size,
        is_multimodal=multimodal,
        vision_config=vision,
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
    )


def test_v41_config_and_registry_expose_text_vision_and_owner_ep(tmp_path):
    from freetoken.models.deepseek_v41.config import parse_config
    from freetoken.models.register import get_model_spec

    args = _args(vision_n_layers=2)
    (tmp_path / "inference").mkdir()
    raw = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in vars(args).items()
        if name not in {"max_batch_size", "max_seq_len"}
    }
    (tmp_path / "inference" / "config.json").write_text(json.dumps(raw))
    hf = SimpleNamespace(
        _name_or_path=str(tmp_path),
        architectures=["DeepseekV41ForCausalLM"],
        text_config=SimpleNamespace(max_position_embeddings=32768),
        vision_config=SimpleNamespace(),
    )

    config = parse_config(hf)
    spec = get_model_spec("DeepseekV41ForCausalLM")

    assert config.model_type == "deepseek_v41"
    assert config.owner_ep_expert_quants == ("ds_fp4",)
    assert config.dsv4_args.kv_source_layers == args.kv_source_layers
    assert config.dsv4_args.index_source_layers == args.index_source_layers
    assert config.vision_config.num_layers == 2
    assert spec.mm_processor.endswith(":DeepseekV41MMProcessor")
    assert spec.encoders[0].modalities == ("image",)

    hf.vision_config = None
    text = parse_config(hf)
    assert text.vision_config is None
    assert not text.dsv4_args.vision_enabled


def test_v41_ftw_fails_before_trying_to_read_engram_side_tables(monkeypatch):
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    args = _args(
        engram_layer_ids=(1,),
        engram_num_embeddings=(32,),
        engram_vocab_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_max_ngram_size=2,
    )
    with _tp(0, 1):
        model = DeepseekV41ForCausalLM(_config(args, tp_size=1))
    monkeypatch.setattr(
        "freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda _path: True
    )
    engine_config = SimpleNamespace(
        model_path="/models/v41.ftw", use_dummy_weight=False
    )

    with pytest.raises(NotImplementedError, match="Engram"):
        model.load_host_tables(engine_config)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_v41_tp_local_attention_and_owner_expert_geometry(tp_size):
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    args = _args()
    with _tp(0, tp_size):
        model = DeepseekV41ForCausalLM(_config(args, tp_size=tp_size))

    source = model.model.layers.op_list[2]
    attention = source.attn
    assert attention.n_heads_local == args.n_heads // tp_size
    assert attention.n_groups_local == args.o_groups // tp_size
    assert attention.attn_sink.shape == (args.n_heads // tp_size,)
    assert attention.wo_a.shape == (
        args.o_groups * args.o_lora_rank // tp_size,
        args.n_heads * args.head_dim // args.o_groups,
    )
    assert attention.wq_b.weight.shape == (
        args.n_heads * args.head_dim // tp_size,
        args.q_lora_rank,
    )
    assert attention.wo_b.weight.shape == (
        args.dim,
        args.o_groups * args.o_lora_rank // tp_size,
    )
    assert source.ffn.experts.expert_tp_size == 1
    assert source.attn.indexer.local_heads == args.index_n_heads // tp_size


def test_v41_tp_accepts_any_divisible_world_size():
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    args = _args(
        vocab_size=129,
        dim=96,
        moe_inter_dim=96,
        n_heads=12,
        n_routed_experts=9,
        q_lora_rank=96,
        head_dim=96,
        rope_head_dim=32,
        o_groups=6,
        o_lora_rank=96,
        index_n_heads=6,
        index_head_dim=32,
    )
    with _tp(0, 3):
        model = DeepseekV41ForCausalLM(_config(args, tp_size=3))

    source = model.model.layers.op_list[2]
    assert source.attn.n_heads_local == 4
    assert source.attn.n_groups_local == 2
    assert source.attn.indexer.local_heads == 2
    assert source.ffn.experts.expert_tp_size == 1


def test_v41_tp_rejects_only_incompatible_head_geometry():
    from freetoken.models.deepseek_v41.attention import Attention

    args = _args(n_heads=12, o_groups=6, index_n_heads=8)
    with _tp(0, 3), pytest.raises(ValueError, match="index_n_heads divisible"):
        Attention(2, args)


def test_v41_attention_sources_and_reuse_follow_csa2_groups():
    from freetoken.models.deepseek_v41.attention import Attention

    args = _args()
    with _tp(0, 1):
        layers = [Attention(index, args) for index in range(args.n_layers)]

    assert layers[0].compress_ratio == 0
    assert layers[0].compressor is layers[0].indexer is None

    assert layers[2].kv_source_layer == 2
    assert layers[2].index_source_layer == 2
    assert layers[2].compressor is not None and layers[2].indexer is not None

    assert layers[3].kv_source_layer == 2
    assert layers[3].index_source_layer == 3
    assert layers[3].compressor is None and layers[3].indexer is not None

    assert layers[4].kv_source_layer == 4
    assert layers[4].index_source_layer == 4
    assert layers[4].compressor is not None and layers[4].indexer is not None
    assert layers[5].kv_source_layer == 4
    assert layers[5].index_source_layer == 5


def test_v41_shared_attention_runtime_is_request_local():
    from freetoken.models.deepseek_v41.attention import SharedAttentionRuntime

    runtime = SharedAttentionRuntime.prefill(2)
    first = torch.tensor([[[0, 1]]], dtype=torch.int32)
    second = torch.tensor([[[3, 4]]], dtype=torch.int32)
    runtime.prefill_picks[0] = first
    runtime.prefill_picks[1] = second

    assert runtime.prefill_picks[0] is first
    assert runtime.prefill_picks[1] is second
    assert runtime.prefill_candidates == [None, None]
    decode = SharedAttentionRuntime.decode()
    assert decode.decode_picks is None and decode.decode_candidates is None


def test_v41_prefill_extension_source_publishes_once_and_consumer_reuses(
    monkeypatch,
):
    import freetoken.models.deepseek_v41.attention as attention_mod

    args = _args(index_source_layers=(2, 4), candidate_source_layer=-1)
    with _tp(0, 1):
        source = attention_mod.Attention(2, args)
        consumer = attention_mod.Attention(3, args)
    calls = []

    class _Compressor:
        def prefill(self, x, **kwargs):
            assert kwargs["start_pos"] == 5
            assert kwargs["tail_window_slot"] == 12
            calls.append("compress")
            return torch.ones(1, 1, args.head_dim), torch.tensor([4])

        def store_prefill(self, latent, starts, **kwargs):
            calls.append("store")

    class _Indexer:
        def publish_prefill_keys(self, latent, starts, **kwargs):
            calls.append("publish")

        def prefill_scores(self, x, qr, **kwargs):
            assert kwargs["start_pos"] == 5
            calls.append("score")
            return torch.tensor([[[0, 1]]], dtype=torch.int32), torch.tensor(
                [[[True, True]]]
            )

    class _Backend:
        def blocks_to_global(self, blocks, ratio, ti=None, rows=None):
            calls.append(("map", ratio, ti, blocks.data_ptr()))
            return blocks + 100

    monkeypatch.setattr(
        attention_mod,
        "get_global_ctx",
        lambda: SimpleNamespace(attn_backend=_Backend()),
    )
    source.compressor = _Compressor()
    source.indexer = _Indexer()
    runtime = attention_mod.SharedAttentionRuntime.prefill(1)
    hidden = torch.zeros(1, 2, args.dim)
    qr = torch.zeros(1, 2, args.q_lora_rank)
    slots = torch.arange(2)

    first = source._prefill_compressed(
        hidden,
        qr,
        segment_index=0,
        table_idx=7,
        start_pos=5,
        window_slots=slots,
        tail_window_slot=12,
        runtime=runtime,
    )
    second = consumer._prefill_compressed(
        hidden,
        qr,
        segment_index=0,
        table_idx=7,
        start_pos=5,
        window_slots=slots,
        tail_window_slot=12,
        runtime=runtime,
    )

    assert calls.count("score") == 1
    assert calls[:4] == ["compress", "publish", "store", "score"]
    map_calls = [call for call in calls if isinstance(call, tuple)]
    assert map_calls[0][3] == map_calls[1][3]
    assert torch.equal(first, torch.tensor([[[100, 101]]]))
    assert torch.equal(second, first)
    assert runtime.prefill_candidates[0].tolist() == [[[True, True]]]


def test_v41_indexer_decode_accepts_zero_staged_blocks(monkeypatch):
    import freetoken.models.deepseek_v41.compress as compress_mod

    args = _args(index_source_layers=(2, 4), candidate_source_layer=-1)
    with _tp(0, 1):
        indexer = compress_mod.Indexer(args, 2)

    class _Backend:
        def indexer_decode_scores(self, q, weights, valid, n_stage, ratio, source):
            assert n_stage == 0
            return torch.empty(q.shape[0], 0)

        def indexer_select_decode(self, scores, **kwargs):
            return torch.empty(scores.shape[0], 1, 0, dtype=torch.int64)

    monkeypatch.setattr(
        compress_mod,
        "get_global_ctx",
        lambda: SimpleNamespace(attn_backend=_Backend()),
    )
    indexer._queries = lambda qr, positions, decode=False: torch.zeros(
        qr.shape[0], 1, indexer.local_heads, indexer.head_dim
    )
    indexer._weights = lambda x: torch.zeros(x.shape[0], 1, indexer.local_heads)

    picks, candidate = indexer.decode_scores(
        torch.zeros(2, 1, args.dim),
        torch.zeros(2, 1, args.q_lora_rank),
        pos=torch.zeros(2, dtype=torch.int64),
        n_stage=0,
        source_layer=2,
        candidate_mask=None,
    )

    assert picks.shape == (2, 1, 0)
    assert candidate is None


def test_v41_state_dict_uses_checkpoint_facing_roots():
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    args = _args(
        engram_layer_ids=(1,),
        engram_num_embeddings=(32,),
        engram_vocab_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_max_ngram_size=2,
    )
    with _tp(0, 1):
        model = DeepseekV41ForCausalLM(_config(args, tp_size=1))
    keys = set(model.state_dict())

    assert "model.embed.weight" in keys
    assert "model.norm.weight" in keys
    assert "head.weight" in keys
    assert "model.layers.2.attn.compressor.wkv_weight" in keys
    assert "model.layers.2.attn.compressor.wgate_weight" in keys
    assert "model.layers.2.attn.indexer.wq_b.weight" in keys
    assert "model.layers.2.attn.indexer.wk.weight" in keys
    assert "model.layers.3.attn.indexer.wq_b.weight" in keys
    assert "model.layers.3.attn.indexer.wk.weight" not in keys
    assert "model.layers.1.engram.q_weight" in keys
    assert "model.layers.1.engram.k_weight" in keys
    assert "model.layers.1.engram.wkv.weight" in keys
    assert not any(key.startswith("model._") for key in keys)


def test_v41_multimodal_parameters_stay_at_checkpoint_root():
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    args = _args(vision_n_layers=1, vision_dim=32, vision_n_heads=8,
                 vision_inter_dim=32, vision_patch_size=2,
                 vision_downsample_ratio=1)
    with _tp(0, 1):
        model = DeepseekV41ForCausalLM(
            _config(args, tp_size=1, multimodal=True)
        )
    keys = set(model.state_dict())

    assert "vision.patch_embed.proj.weight" in keys
    assert "aligner.w1.weight" in keys
    assert "image_start" in keys
    assert "image_newline" in keys
    assert "image_end" in keys
    assert "model.layers.0.ffn.gate.bias_vl" in keys


def test_v41_single_pass_hc_threads_each_sublayer_pre_mix(monkeypatch):
    from freetoken.models.deepseek_v41.model import Block

    args = _args(n_layers=1, compress_ratios=(0,), kv_source_layers=(),
                 index_source_layers=(), candidate_source_layer=-1)
    with _tp(0, 1):
        block = Block(0, args, None)
    seen = []
    attn_pre = torch.full((2, args.hc_mult), 2.0)
    ffn_pre = torch.full((2, args.hc_mult), 3.0)
    posts = torch.zeros(2, args.hc_mult)
    comb = torch.zeros(2, args.hc_mult, args.hc_mult)

    monkeypatch.setattr(
        block,
        "hc_mixes",
        lambda x, weight, scale, base: (
            (attn_pre if weight is block.hc_attn_fn else ffn_pre), posts, comb
        ),
    )
    monkeypatch.setattr(
        block,
        "hc_pre",
        lambda x, pre: seen.append(pre.clone()) or x[..., 0, :],
    )
    monkeypatch.setattr(block, "hc_post", lambda x, residual, post, matrix: residual)
    block.attn_norm.forward = lambda x: x
    block.ffn_norm.forward = lambda x: x
    block.attn.forward_ragged = lambda x, segments, positions, runtime: x
    block.ffn.forward = lambda x, image_mask=None: x

    hidden = torch.zeros(1, 2, args.hc_mult, args.dim)
    initial = torch.full((1, 2, args.hc_mult), 1.0)
    _, next_mix = block.prefill_batched(
        hidden,
        initial,
        [(0, 2, 0, 0)],
        torch.arange(2),
        None,
        SimpleNamespace(),
    )

    assert torch.equal(seen[0], initial)
    assert torch.equal(seen[1], attn_pre)
    assert torch.equal(next_mix, ffn_pre)


def test_v41_engram_injection_is_suppressed_on_image_rows(monkeypatch):
    from freetoken.models.deepseek_v41.model import Transformer

    hidden = torch.zeros(1, 3, 2, 4)
    rows = torch.zeros(3, 1, 1, dtype=torch.int64)
    image_mask = torch.tensor([[False, True, False]])
    seen = {}

    def forward(x, row_ids, token_mask):
        seen["row_ids"] = row_ids
        seen["token_mask"] = token_mask
        return x + token_mask[..., None, None]

    layer = SimpleNamespace(forward=forward)

    result = Transformer._apply_engram(layer, hidden, rows, image_mask)

    assert seen["row_ids"] is rows
    assert seen["token_mask"].tolist() == [[True, False, True]]
    assert torch.equal(result[:, 0], torch.ones_like(result[:, 0]))
    assert torch.equal(result[:, 1], hidden[:, 1])
    assert torch.equal(result[:, 2], torch.ones_like(result[:, 2]))


def test_v41_args_remain_runtime_mutable_for_engine_resolution():
    args = _args()
    args.max_seq_len = 8192
    args.max_batch_size = 9
    assert (args.max_seq_len, args.max_batch_size) == (8192, 9)


def test_v41_runtime_is_forward_local_not_model_state():
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM

    with _tp(0, 1):
        model = DeepseekV41ForCausalLM(_config(_args(), tp_size=1))
    assert "runtime" not in model.model.__dict__
    assert all("runtime" not in layer.__dict__ for layer in model.model.layers.op_list)
