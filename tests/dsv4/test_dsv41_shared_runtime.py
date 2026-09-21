from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_v41_decode_queries_use_per_row_rotary(monkeypatch):
    from freetoken.models.deepseek_v41 import compress as mod

    indexer = object.__new__(mod.Indexer)
    indexer.local_heads = 2
    indexer.head_dim = 4
    indexer.rope_head_dim = 4
    indexer._freqs_cis = torch.ones(16, 2, dtype=torch.complex64)
    indexer.wq_b = SimpleNamespace(
        forward=lambda qr: torch.zeros(*qr.shape[:-1], 8, dtype=qr.dtype)
    )
    seen = {}

    def decode_rope(x, freqs):
        seen["shape"] = (x.shape, freqs.shape)
        return x

    monkeypatch.setattr(mod, "apply_rotary_emb_decode", decode_rope)
    monkeypatch.setattr(
        mod,
        "apply_rotary_emb",
        lambda *_args, **_kwargs: pytest.fail("decode must use per-row RoPE"),
    )
    monkeypatch.setattr(mod, "fp4_act_quant_inplace", lambda x, _block: x)

    qr = torch.zeros(3, 1, 6)
    got = indexer._queries(qr, torch.tensor([2, 5, 9]), decode=True)

    assert got.shape == (3, 1, 2, 4)
    assert seen["shape"] == (torch.Size([3, 1, 2, 4]), torch.Size([3, 2]))


def test_v41_candidate_mask_excludes_future_blocks_before_topk():
    from freetoken.models.deepseek_v41.compress import (
        _mask_unreachable,
        select_candidate_blocks,
    )

    # Future positions deliberately have the largest logits. They must not consume
    # either candidate block available to the first query.
    logits = torch.tensor(
        [[[5.0, 4.0, 100.0, 99.0], [5.0, 4.0, 100.0, 99.0]]]
    )
    live = torch.tensor([[[2], [4]]])
    masked = _mask_unreachable(logits, live)
    candidates = select_candidate_blocks(logits, live, topk_blocks=1, block_size=2)

    assert torch.isneginf(masked[0, 0, 2:]).all()
    assert candidates[0, 0].tolist() == [True, True, False, False]
    # The second query can see every position; its newest partial/full block is pinned.
    assert candidates[0, 1].tolist() == [False, False, True, True]


def test_v41_position_sort_is_opt_in_and_v4_order_is_unchanged():
    from freetoken.attention.dsv4_indexer import IndexerBackendMixin

    mixin = IndexerBackendMixin()
    scores = torch.tensor([[[9.0, 1.0, 2.0, 10.0]]])
    legacy = mixin.indexer_select_prefill(
        scores, start_pos=3, seqlen=1, ratio=1, topk=2, offset=0
    )
    v41 = mixin.indexer_select_prefill(
        scores,
        start_pos=3,
        seqlen=1,
        ratio=1,
        topk=2,
        offset=0,
        sort_by_position=True,
    )

    assert legacy.tolist() == [[[3, 0]]]
    assert v41.tolist() == [[[0, 3]]]


def test_v41_pool_allocates_only_source_layers_and_ratio2_state():
    from freetoken.kvcache.dsv4_cost_model import dsv4_pool_bytes, dsv4_pool_sizes
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache

    args = SimpleNamespace(
        n_layers=4,
        head_dim=8,
        index_head_dim=4,
        compress_ratios=(0, 2, 2, 1),
        kv_source_layers=(1, 3),
        engram_layer_ids=(),
    )
    sizes = dsv4_pool_sizes(2, args, 1.0, P=8, n_win_pages=2)
    assert sizes.cmp_blocks == [None, 8, None, 16]
    assert sizes.idx_blocks == [None, 8, None, 16]
    assert sizes.state_slots == [None, 4, None, None]
    assert sizes.ring_sizes == [None, 2, None, None]

    pool = DSV4PagedKVCache(
        sizes, args, torch.device("cpu"), P=8, n_scratch=2
    )
    assert pool.cmp_pool[0] is None and pool.cmp_pool[2] is None
    assert pool.cmp_pool[1].shape == (10, 8)
    assert pool.cmp_pool[3].shape == (18, 8)
    assert pool.state_ring[1].buffer.shape == (5, 16)
    assert pool.state_ring[3] is None
    assert pool.total_bytes() == dsv4_pool_bytes(sizes, args, n_scratch=2)


def test_sparse_attention_can_read_compressed_kv_from_an_earlier_source(
    monkeypatch,
):
    import freetoken.attention.dsv4_sparse as sparse_mod

    source_cmp = torch.full((4, 8), 7.0)
    pool = SimpleNamespace(
        window_pool=[torch.zeros(4, 8), torch.ones(4, 8)],
        cmp_pool=[source_cmp, None],
    )
    monkeypatch.setattr(
        sparse_mod, "get_global_ctx", lambda: SimpleNamespace(kv_cache=pool)
    )
    seen = {}

    def fake_sparse(q, window, cmp, sink, idx, n_window, scale, **kwargs):
        seen["cmp"] = cmp
        return torch.empty_like(q)

    monkeypatch.setattr(
        "freetoken.kernel.triton.dsv4.sparse_attn.sparse_attn_paged",
        fake_sparse,
    )
    backend = object.__new__(sparse_mod.DSV4SparseAttnBackend)
    q = torch.zeros(1, 1, 1, 8)
    backend.attend(
        q,
        layer_id=1,
        topk_idxs=torch.zeros(1, 1, 1, dtype=torch.int32),
        n_window=0,
        attn_sink=torch.zeros(1),
        softmax_scale=1.0,
        cmp_layer_id=0,
    )
    assert seen["cmp"] is source_cmp


def test_engram_gather_wrapper_uses_one_scale_per_32_columns(monkeypatch):
    import freetoken.kernel.triton.ple as ple

    seen = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            seen["grid"] = grid

            def launch(*args, **kwargs):
                seen["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(ple, "_engram_gather_kernel", _FakeKernel())
    ids = torch.tensor([0, 7], dtype=torch.int32)
    out = torch.empty(2, 64, dtype=torch.bfloat16)
    assert ple.engram_gather_rows(
        1,
        2,
        global_start=4,
        local_rows=8,
        embed_dim=64,
        row_ids=ids,
        out=out,
    ) is out
    assert seen["grid"] == (2,)
    assert seen["kwargs"]["GROUP"] == 32


def test_engram_dead_token_blocks_all_older_lookbacks():
    from freetoken.models.deepseek_v41.engram import DEAD, EngramHash

    state = object.__new__(EngramHash)
    state._pad_id = 77
    values = torch.tensor(
        [
            [10, DEAD, 20, 30],
            [10, 20, 30, 40],
            [10, 20, 30, 40],
            [DEAD, 20, 30, 40],
        ]
    )
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )

    got = state._stop_at_dead(values, valid)

    assert got.tolist() == [
        [10, 77, 77, 77],
        [10, 20, 30, 40],
        [10, 20, 77, 77],
        [77, 77, 77, 77],
    ]


def test_v41_ratio2_compressor_carries_across_chunk_and_decode(monkeypatch):
    from freetoken.models.deepseek_v41 import compress as mod

    page_size = 4
    carried = {}

    class _Attention:
        @staticmethod
        def _page(slot):
            return int(slot) // page_size

        def read_carry(self, _layer, _tier, slot, _ratio):
            return carried[self._page(slot)].clone()

        def write_carry(self, _layer, _tier, slot, _ratio, block):
            carried[self._page(slot)] = block.clone()

        def read_carry_blocks(self, _layer, _tier, slots, _ratio):
            return torch.stack(
                [carried[self._page(slot)] for slot in slots.tolist()]
            )

        def write_carry_blocks(self, _layer, _tier, slots, _ratio, blocks):
            for slot, block in zip(slots.tolist(), blocks):
                carried[self._page(slot)] = block.clone()

    attention = _Attention()
    monkeypatch.setattr(
        mod,
        "get_global_ctx",
        lambda: SimpleNamespace(
            attn_backend=attention,
            kv_cache=SimpleNamespace(P=page_size),
        ),
    )
    monkeypatch.setattr(
        mod,
        "gated_pool",
        lambda kv, score, dtype: (
            kv * score.softmax(dim=1)
        ).sum(dim=1, keepdim=True).to(dtype),
    )

    compressor = object.__new__(mod.Compressor)
    compressor.layer_id = 0
    compressor.ratio = 2
    compressor.head_dim = 2
    compressor.norm = SimpleNamespace(forward=lambda value: value)
    compressor._project = lambda value: (
        value.float(),
        torch.zeros_like(value, dtype=torch.float32),
    )

    first = torch.tensor([[[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]]])
    latent, starts = compressor.prefill(
        first,
        start_pos=0,
        window_slots=torch.tensor([0, 1, 2]),
        tail_window_slot=None,
    )
    assert starts.tolist() == [0]
    assert latent.tolist() == [[[1.0, 11.0]]]
    assert carried[0][0, :2].tolist() == [4.0, 14.0]

    second = torch.tensor([[[6.0, 16.0], [8.0, 18.0]]])
    latent, starts = compressor.prefill(
        second,
        start_pos=3,
        window_slots=torch.tensor([3, 4]),
        tail_window_slot=2,
    )
    assert starts.tolist() == [2]
    assert latent.tolist() == [[[5.0, 15.0]]]
    assert carried[1][0, :2].tolist() == [8.0, 18.0]

    latent, complete = compressor.decode(
        torch.tensor([[[10.0, 20.0]]]),
        pos=torch.tensor([5]),
        prev_window_slots=torch.tensor([4]),
        window_slots=torch.tensor([5]),
    )
    assert complete.tolist() == [True]
    assert latent.tolist() == [[[9.0, 19.0]]]
    assert torch.count_nonzero(carried[1][:, :2]) == 0
    assert torch.isneginf(carried[1][:, 2:]).all()
