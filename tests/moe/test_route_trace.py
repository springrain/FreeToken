"""Tests for moe/route_trace.py: capture round-trip, LRU-mirror semantics, EP slow-side.

Pure CPU, no torch/CUDA needed for the replay/LRU parts (the recorder's ``record``
takes a tensor, so it uses a tiny stub). Mirrors PLAN_TP_EP.md 6A's requirement that
the replay match flashlib lru_ensure semantics before it is trusted for EP analysis.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load route_trace.py directly by path: it is stdlib-only, so importing it without
# the freetoken.moe package __init__ (which pulls torch/transformers) keeps these
# tests runnable on a bare interpreter. Resolve it relative to THIS file so the test
# collects in any checkout (an absolute author path fails everywhere else).
_MOD = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "moe" / "route_trace.py"
_spec = importlib.util.spec_from_file_location("route_trace", _MOD)
_rt = importlib.util.module_from_spec(_spec)
sys.modules["route_trace"] = _rt  # @dataclass resolves cls.__module__ via sys.modules
_spec.loader.exec_module(_rt)
LRU = _rt.LRU
RouteTraceRecorder = _rt.RouteTraceRecorder
read_trace = _rt.read_trace
replay = _rt.replay
replay_ep = _rt.replay_ep
replay_ep2 = _rt.replay_ep2


class _FakeTensor:
    """Minimal stand-in for the raw expert-ids tensor ``record`` receives."""

    def __init__(self, ids):
        self._ids = list(ids)

    def reshape(self, _):
        return self

    def tolist(self):
        return self._ids


def test_roundtrip(tmp_path):
    import struct

    path = str(tmp_path / "tr.bin")
    rec = RouteTraceRecorder(
        path, num_experts=512, num_layers=48, cache_size=2840, top_k=10,
        model="flash", decode_target="gpu",
    )
    # emulate record() without torch: write the same binary layout by hand
    def put(phase, layer, ids):
        rec._buf += struct.Struct("<bii").pack(phase, layer, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1

    put(0, 0, [5, 300, 7])
    put(0, 1, [10, 11])
    put(1, 0, [1, 2, 3])  # prefill record
    rec.close()

    meta, records = read_trace(path)
    assert meta.num_experts == 512 and meta.cache_size == 2840
    assert meta.records == 3
    assert records[0] == (0, 0, (5, 300, 7))
    assert records[1] == (0, 1, (10, 11))
    assert records[2] == (1, 0, (1, 2, 3))


def test_lru_batch_protection():
    # C=2: [1,2] then [0,2] -> expert 2 is a protected hit, only 0 misses (v1 bug = 2)
    c = LRU(2, 512)
    c.ensure(0, [1, 2])
    before = c.miss
    c.ensure(0, [0, 2])
    assert c.miss - before == 1, f"expected 1 miss, got {c.miss - before}"


def test_lru_stack_property():
    # same access sequence, larger cache never increases miss (standard LRU property,
    # the theoretical basis for the capacity-expansion main line)
    seq = [(0, [i % 40 for i in range(s, s + 10)]) for s in range(0, 300, 3)]
    counts = []
    for C in (8, 16, 32, 64):
        c = LRU(C, 512)
        for layer, ids in seq:
            c.ensure(layer, ids)
        counts.append(c.miss)
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_lru_dedup():
    c = LRU(8, 512)
    c.ensure(0, [3, 3, 3, 7])
    assert c.miss == 2 and c.active == 2


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 6, 8, 10, 12])
def test_replay_tp1_vs_ep(world_size):
    # Each rank repeatedly touches two experts from a six-wide owner-local window
    # on two layers. A 12-slot local cache covers that rank's full working set,
    # while TP1 sees the combined 12*world_size rows with the same capacity.
    import random

    rng = random.Random(0)
    num_experts = 120
    local_num_experts = num_experts // world_size
    records = []
    for _ in range(400):
        ids = []
        for rank in range(world_size):
            lo = rank * local_num_experts
            ids.extend(lo + e for e in rng.sample(range(6), 2))
        for layer in range(2):
            records.append((0, layer, tuple(ids)))

    _, _, tp1 = replay(records, 12, num_experts)
    ep = replay_ep(records, (12,) * world_size, num_experts)
    assert ep["slow_rate"] < tp1, (
        f"EP{world_size} slow {ep['slow_rate']} should beat TP1 {tp1}"
    )
    assert len(ep["rank_miss"]) == len(ep["rank_active"]) == world_size
    assert len(set(ep["rank_active"])) == 1
    # slow-side miss >= each rank's miss (it is the per-step max), so it is the limiter
    assert ep["slow_miss"] >= max(ep["rank_miss"])
    assert tp1 > 0.3, f"expected real cache pressure, TP1 miss={tp1}"


def test_replay_ep2_is_a_compatibility_wrapper():
    records = [(0, 0, (0, 4, 1, 5)), (0, 0, (2, 6, 3, 7))]
    assert replay_ep2(records, (2, 2), 8) == replay_ep(records, (2, 2), 8)


@pytest.mark.parametrize(
    ("cache_sizes", "num_experts", "message"),
    [
        ((), 8, "at least 2 ranks"),
        ((4,), 8, "at least 2 ranks"),
        ((4, 4, 4), 10, "not divisible"),
        ((4,) * 6, 32, "not divisible"),
    ],
)
def test_replay_ep_rejects_invalid_geometry(cache_sizes, num_experts, message):
    with pytest.raises(ValueError, match=message):
        replay_ep([], cache_sizes, num_experts)


def test_overflow_flag(tmp_path):
    import struct

    path = str(tmp_path / "o.bin")
    rec = RouteTraceRecorder(
        path, num_experts=512, num_layers=1, cache_size=10, top_k=2, model="t",
        decode_target="gpu", max_records=3,
    )

    def put(ids):
        if rec._overflow:
            return
        if rec._n >= rec.max_records:
            rec._overflow = True
            return
        rec._buf += struct.Struct("<bii").pack(0, 0, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1

    for _ in range(10):
        put([1, 2])
    rec.close()
    meta, records = read_trace(path)
    assert meta.overflow is True
    assert len(records) == 3


def test_each_rank_records_to_its_own_path(tmp_path):
    """Every TP rank is configured with the SAME path and ``wb`` truncates, so sharing
    it would let the writers race on both the body and the meta sidecar and leave a
    truncated trace holding one rank's records."""
    path = str(tmp_path / "t.bin")
    rec0 = RouteTraceRecorder(
        path, num_experts=8, num_layers=1, cache_size=10, top_k=2, rank=0
    )
    rec1 = RouteTraceRecorder(
        path, num_experts=8, num_layers=1, cache_size=10, top_k=2, rank=1
    )
    assert rec0.path == path + ".rank0"
    assert rec1.path == path + ".rank1"
    assert rec0.meta_path == rec0.path + ".meta.json"
    assert rec0.path != rec1.path
    rec0.close()
    rec1.close()


def test_no_rank_keeps_the_configured_path_byte_for_byte(tmp_path):
    """Single-rank runs (and the offline replay tooling) must keep the exact path."""
    path = str(tmp_path / "t.bin")
    rec = RouteTraceRecorder(
        path, num_experts=8, num_layers=1, cache_size=10, top_k=2
    )
    assert rec.path == path
    assert rec.meta_path == path + ".meta.json"
    rec.close()


def test_two_rank_recorders_do_not_truncate_each_other(tmp_path):
    """The actual failure mode: rank 1 opening its file must not empty rank 0's."""
    import struct

    path = str(tmp_path / "t.bin")
    rec0 = RouteTraceRecorder(
        path, num_experts=8, num_layers=1, cache_size=10, top_k=2, rank=0
    )
    rec0._buf += struct.Struct("<bii").pack(0, 0, 2)
    rec0._buf += struct.pack("<2i", 1, 2)
    rec0._n += 1
    rec0.close()

    rec1 = RouteTraceRecorder(
        path, num_experts=8, num_layers=1, cache_size=10, top_k=2, rank=1
    )
    rec1.close()

    _meta0, records0 = read_trace(rec0.path)
    _meta1, records1 = read_trace(rec1.path)
    assert len(records0) == 1
    assert records1 == []
