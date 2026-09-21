"""DeepSeek-V4.1 Engram hashing and rank-sharded pinned-host tables."""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight

from .args import DeepseekV41Args

DEAD = -1


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = math.isqrt(value)
    return all(value % divisor for divisor in range(3, limit + 1, 2))


def _next_prime(start: int, seen: set[int]) -> int:
    value = start + 1
    while not _is_prime(value) or value in seen:
        value += 1
    return value


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args: DeepseekV41Args) -> "EngramLayout | None":
        if not args.engram_layer_ids:
            return None
        seen: set[int] = set()
        layers = []
        for _ in args.engram_layer_ids:
            ngrams = []
            for _ in range(args.engram_max_ngram_size - 1):
                heads = []
                current = args.engram_vocab_size - 1
                for _ in range(args.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    heads.append(current)
                ngrams.append(tuple(heads))
            layers.append(tuple(ngrams))
        return cls(
            max_ngram_size=args.engram_max_ngram_size,
            layer_ids=args.engram_layer_ids,
            num_embeddings=args.engram_num_embeddings,
            primes=tuple(layers),
            n_heads=args.engram_n_heads,
            head_dim=args.engram_head_dim,
        )


def build_compressed_token_map(tokenizer) -> tuple[torch.Tensor, int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    by_text: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = (
            backend.id_to_token(token_id)
            if "\ufffd" in text
            else (normalizer.normalize_str(text) or text)
        )
        lookup.append(by_text.setdefault(key, len(by_text)))
    return torch.tensor(lookup, dtype=torch.int64), len(by_text)


class EngramHash:
    def __init__(self, args: DeepseekV41Args, layout: EngramLayout):
        self.args = args
        self.layout = layout
        self._token_map: torch.Tensor | None = None
        self._pad_id: int | None = None
        self._multipliers: torch.Tensor | None = None
        self._primes: torch.Tensor | None = None
        self._offsets: torch.Tensor | None = None
        self._token_cache_initialized = False

    def initialize(self, model_path: str, device: torch.device) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != self.args.engram_compressed_vocab_size:
            raise ValueError(
                f"Engram compressed vocabulary is {vocab_size}, checkpoint expects "
                f"{self.args.engram_compressed_vocab_size}"
            )
        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // vocab_size) // 2)
        rows = []
        for layer_id in self.layout.layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(
                0, bound, size=(self.layout.max_ngram_size,), dtype=np.int64
            )
            rows.append(torch.tensor(values * 2 + 1))
        flat_primes = [
            [prime for ngram in layer for prime in ngram]
            for layer in self.layout.primes
        ]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat_primes]
        self._token_map = token_map.to(device)
        self._pad_id = int(token_map[self.args.engram_pad_id])
        self._multipliers = torch.stack(rows).to(device)
        self._primes = torch.tensor(self.layout.primes, dtype=torch.int64, device=device)
        self._offsets = torch.tensor(np.asarray(offsets), dtype=torch.int64, device=device)

    def _compressed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self._token_map is not None
        image = input_ids >= MM_PAD_SHIFT_VALUE
        safe = input_ids.clamp(min=0, max=self._token_map.numel() - 1).long()
        return torch.where(image, safe.new_full((), DEAD), self._token_map[safe])

    def _stop_at_dead(
        self, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Pad the dead token and every older lookback so n-grams never cross images."""
        blocked = (~valid) | (values == DEAD)
        blocked = blocked.to(torch.int32).cumsum(dim=-1).bool()
        return torch.where(blocked, values.new_full((), self._pad_id), values)

    def _prefill_windows(self, batch, width: int) -> torch.Tensor:
        pool = get_global_ctx().kv_cache
        parts = []
        for offset, length, table_idx, start in batch.attn_metadata.segments:
            positions = start + torch.arange(length, device=batch.input_ids.device)
            shifts = torch.arange(width, device=positions.device)
            source = positions[:, None] - shifts[None, :]
            valid = source >= 0
            full = pool.full_loc_map[table_idx, source.clamp_min(0)]
            values = pool.token_cache[full.long()]
            parts.append(self._stop_at_dead(values, valid))
        return torch.cat(parts, dim=0)

    def _decode_windows(self, batch, width: int) -> torch.Tensor:
        positions = batch.positions.long().view(-1)[: batch.padded_size]
        shifts = torch.arange(width, device=positions.device)
        source = positions[:, None] - shifts[None, :]
        valid = source >= 0
        snap = batch.attn_metadata.full_snapshot()
        rows = torch.arange(positions.numel(), device=positions.device)
        full = snap[rows[:, None], source.clamp_min(0)]
        values = get_global_ctx().kv_cache.token_cache[full.long()]
        return self._stop_at_dead(values, valid)

    def row_ids(self, batch) -> torch.Tensor:
        if self._token_map is None:
            raise RuntimeError("Engram hash state was not initialized")
        pool = get_global_ctx().kv_cache
        if not self._token_cache_initialized:
            if pool.token_cache is None:
                raise RuntimeError("DeepSeek-V4.1 Engram needs the DSV4 token cache")
            pool.token_cache.fill_(self._pad_id)
            self._token_cache_initialized = True
        compressed = self._compressed_input(batch.input_ids)
        pool.token_cache.index_copy_(0, batch.out_loc.long(), compressed.to(torch.int32))
        tokens = (
            self._prefill_windows(batch, self.layout.max_ngram_size)
            if batch.is_prefill
            else self._decode_windows(batch, self.layout.max_ngram_size)
        )
        assert self._multipliers is not None
        assert self._primes is not None
        assert self._offsets is not None
        products = tokens[:, None, :] * self._multipliers[None, :, :]
        rolling = products[..., 0]
        hashes = []
        for index in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(
                rolling.unsqueeze(-1) % self._primes[:, index - 1, :]
            )
        rows = torch.cat(hashes, dim=-1) + self._offsets
        return rows


class ZeroEngramTable:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*row_ids.shape, self.embed_dim),
            dtype=torch.bfloat16,
            device=row_ids.device,
        )


class ParallelPinnedEngramTable:
    def __init__(
        self,
        weight: HostBank,
        scale: HostBank,
        *,
        global_start: int,
        valid_local_rows: int,
        embed_dim: int,
        device: torch.device,
    ):
        from freetoken.kernel.pinned import device_ptr

        self.weight = weight
        self.scale = scale
        self.global_start = global_start
        self.valid_local_rows = valid_local_rows
        self.embed_dim = embed_dim
        self.device = device
        self._weight_ptr = device_ptr(weight.tensor)
        self._scale_ptr = device_ptr(scale.tensor)
        self._comm = DistributedCommunicator()

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import engram_gather_rows

        out = torch.empty(
            (row_ids.numel(), self.embed_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        engram_gather_rows(
            self._weight_ptr,
            self._scale_ptr,
            global_start=self.global_start,
            local_rows=self.valid_local_rows,
            embed_dim=self.embed_dim,
            row_ids=row_ids.reshape(-1),
            out=out,
        )
        if get_tp_info().size > 1:
            out = self._comm.all_reduce(out)
        return out.view(*row_ids.shape, self.embed_dim)


class EngramLayer(BaseOP):
    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        layout: EngramLayout,
        *,
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.layer_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.eps = args.norm_eps
        self.q_weight = torch.empty(args.hc_mult, args.dim)
        self.k_weight = torch.empty(args.hc_mult, args.dim)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = LinearReplicated(
            n_hash_cols * layout.head_dim,
            args.dim * (args.hc_mult + 1),
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self._table = None

    def attach_table(self, table) -> None:
        self._table = table

    def forward(
        self,
        x: torch.Tensor,
        row_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._table is None:
            raise RuntimeError(f"Engram table for layer {self.layer_id} was not attached")
        embed = self._table.lookup(row_ids[:, self.layer_index]).flatten(1)
        kv = self.wkv.forward(embed)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().view(-1, self.hc_mult, self.dim)
        hidden = x.float().view(-1, self.hc_mult, self.dim)
        weight = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.eps)
        rstd = rstd * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.reshape(-1, 1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(1)).to(x.dtype).view_as(x)


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def load_engram_table(
    model_path: str,
    *,
    layer_id: int,
    num_embeddings: int,
    embed_dim: int,
    device: torch.device,
    pin: bool = True,
) -> tuple[ParallelPinnedEngramTable, int]:
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    weight_name = f"layers.{layer_id}.engram.embed.weight"
    scale_name = f"layers.{layer_id}.engram.embed.scale"
    paths = {
        name: os.path.join(folder, weight_map[name])
        for name in (weight_name, scale_name)
    }
    metadata = {}
    for name, path in paths.items():
        header, base = _safetensors_header(path)
        item = header[name]
        begin, end = item["data_offsets"]
        metadata[name] = (path, base + begin, end - begin, tuple(item["shape"]), item["dtype"])
    if metadata[weight_name][3] != (num_embeddings, embed_dim):
        raise ValueError(
            f"{weight_name} has shape {metadata[weight_name][3]}, expected "
            f"{(num_embeddings, embed_dim)}"
        )
    scale_dim = embed_dim // 32
    if metadata[scale_name][3] != (num_embeddings, scale_dim):
        raise ValueError(
            f"{scale_name} has shape {metadata[scale_name][3]}, expected "
            f"{(num_embeddings, scale_dim)}"
        )
    if metadata[weight_name][4] != "F8_E4M3" or metadata[scale_name][4] != "F8_E8M0":
        raise ValueError(
            f"unsupported Engram dtypes {metadata[weight_name][4]} / {metadata[scale_name][4]}"
        )

    tp = get_tp_info()
    rows_per_rank = (num_embeddings + tp.size - 1) // tp.size
    global_start = tp.rank * rows_per_rank
    valid_rows = max(0, min(rows_per_rank, num_embeddings - global_start))
    weight = HostBank((rows_per_rank, embed_dim), torch.float8_e4m3fn)
    scale = HostBank((rows_per_rank, scale_dim), torch.float8_e8m0fnu)
    if valid_rows:
        read_range_into(
            weight.memoryview(),
            metadata[weight_name][0],
            file_offset=metadata[weight_name][1] + global_start * embed_dim,
            nbytes=valid_rows * embed_dim,
            dest_offset=0,
        )
        read_range_into(
            scale.memoryview(),
            metadata[scale_name][0],
            file_offset=metadata[scale_name][1] + global_start * scale_dim,
            nbytes=valid_rows * scale_dim,
            dest_offset=0,
        )
    if pin and torch.cuda.is_available():
        weight.pin()
        scale.pin()
    table = ParallelPinnedEngramTable(
        weight,
        scale,
        global_start=global_start,
        valid_local_rows=valid_rows,
        embed_dim=embed_dim,
        device=device,
    )
    return table, weight.nbytes + scale.nbytes


__all__ = [
    "EngramHash",
    "EngramLayer",
    "EngramLayout",
    "ParallelPinnedEngramTable",
    "ZeroEngramTable",
    "build_compressed_token_map",
    "load_engram_table",
]
