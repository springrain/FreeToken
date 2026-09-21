"""DeepSeek-V4.1 shared compressed-KV producer and distributed indexer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.kernel.triton.dsv4.compress import gated_pool
from freetoken.kernel.triton.dsv4.fp8_linear import (
    fp4_act_quant_e4m3_inplace,
    fp4_act_quant_inplace,
)
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, RMSNorm

from .args import DeepseekV41Args
from freetoken.models.deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode


class Compressor(BaseOP):
    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        *,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.ratio = args.compress_ratios[layer_id]
        if self.ratio not in (1, 2):
            raise ValueError(f"V4.1 compressor needs ratio 1 or 2, got {self.ratio}")
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.norm = RMSNorm(args.head_dim, args.norm_eps)
        dtype = torch.float32 if self.ratio > 1 else torch.bfloat16
        self.wkv_weight = torch.empty(args.head_dim, args.dim, dtype=dtype)
        self.wgate_weight = (
            torch.empty(args.head_dim, args.dim, dtype=torch.float32)
            if self.ratio > 1
            else None
        )
        self._freqs_cis: torch.Tensor | None = None
        self._device: torch.device | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, freqs_cis: torch.Tensor, device: torch.device) -> None:
        self._freqs_cis = freqs_cis
        self._device = device

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.ratio == 1:
            weight = self.wkv_weight.to(x.dtype) if self.wkv_weight.dtype != x.dtype else self.wkv_weight
            return F.linear(x, weight), None
        hidden = x.float()
        return F.linear(hidden, self.wkv_weight), F.linear(hidden, self.wgate_weight)

    def _empty_carry(self, device: torch.device) -> torch.Tensor:
        kv = torch.zeros(self.ratio, self.head_dim, dtype=torch.float32, device=device)
        score = torch.full_like(kv, float("-inf"))
        return torch.cat((kv, score), dim=-1)

    def _write_prefill_carry(
        self,
        *,
        start: int,
        end: int,
        window_slots: torch.Tensor,
        tail_kv: torch.Tensor | None,
        tail_score: torch.Tensor | None,
    ) -> None:
        if self.ratio == 1 or end <= start:
            return
        empty = self._empty_carry(window_slots.device)
        P = get_global_ctx().kv_cache.P
        for boundary in range((start // P + 1) * P, end + 1, P):
            slot = int(window_slots[boundary - 1 - start].item())
            self.attn.write_carry(
                self.layer_id, "attn", slot, self.ratio, empty
            )
        if end % P:
            block = empty.clone()
            if tail_kv is not None:
                block[0, : self.head_dim] = tail_kv.float()
                block[0, self.head_dim :] = tail_score.float()
            self.attn.write_carry(
                self.layer_id,
                "attn",
                int(window_slots[-1].item()),
                self.ratio,
                block,
            )

    def prefill(
        self,
        x: torch.Tensor,
        *,
        start_pos: int,
        window_slots: torch.Tensor,
        tail_window_slot: int | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return unrotated latents and their absolute block-start positions."""
        kv, score = self._project(x)
        n = x.shape[1]
        if self.ratio == 1:
            starts = torch.arange(start_pos, start_pos + n, device=x.device)
            return self.norm.forward(kv.to(x.dtype)), starts

        latents = []
        starts = []
        cursor = 0
        if start_pos % 2:
            if tail_window_slot is None:
                raise ValueError("ratio-2 continuation needs the previous window slot")
            carry = self.attn.read_carry(
                self.layer_id, "attn", tail_window_slot, self.ratio
            )
            pair_kv = torch.stack((carry[0, : self.head_dim], kv[0, 0].float()))
            pair_score = torch.stack((carry[0, self.head_dim :], score[0, 0]))
            latents.append(gated_pool(pair_kv[None], pair_score[None], x.dtype)[0, 0])
            starts.append(start_pos - 1)
            cursor = 1

        complete = (n - cursor) // 2
        if complete:
            pair_kv = kv[:, cursor : cursor + 2 * complete].reshape(1, complete, 2, -1)
            pair_score = score[:, cursor : cursor + 2 * complete].reshape(
                1, complete, 2, -1
            )
            reduced = (pair_kv * pair_score.softmax(dim=2)).sum(dim=2).to(x.dtype)
            latents.extend(reduced[0].unbind(0))
            starts.extend(start_pos + cursor + 2 * index for index in range(complete))
            cursor += 2 * complete

        tail_kv = kv[0, cursor] if cursor < n else None
        tail_score = score[0, cursor] if cursor < n else None
        self._write_prefill_carry(
            start=start_pos,
            end=start_pos + n,
            window_slots=window_slots,
            tail_kv=tail_kv,
            tail_score=tail_score,
        )
        if not latents:
            return None, torch.empty(0, dtype=torch.int64, device=x.device)
        latent = self.norm.forward(torch.stack(latents, dim=0).unsqueeze(0))
        return latent, torch.tensor(starts, dtype=torch.int64, device=x.device)

    def decode(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor,
        prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv, score = self._project(x.view(x.shape[0], 1, -1))
        if self.ratio == 1:
            return self.norm.forward(kv.to(x.dtype)), torch.ones_like(pos, dtype=torch.bool)

        block = self.attn.read_carry_blocks(
            self.layer_id, "attn", prev_window_slots, self.ratio
        )
        ks = block[..., : self.head_dim].clone()
        ss = block[..., self.head_dim :].clone()
        slot = (pos % self.ratio).view(-1, 1, 1)
        ks.scatter_(1, slot.expand(-1, 1, self.head_dim), kv.float())
        ss.scatter_(1, slot.expand(-1, 1, self.head_dim), score.float())
        complete = (pos + 1) % self.ratio == 0
        latent = self.norm.forward(gated_pool(ks, ss, x.dtype))
        reset = complete[:, None, None]
        ks = torch.where(reset, torch.zeros_like(ks), ks)
        ss = torch.where(reset, torch.full_like(ss, float("-inf")), ss)
        self.attn.write_carry_blocks(
            self.layer_id,
            "attn",
            window_slots,
            self.ratio,
            torch.cat((ks, ss), dim=-1),
        )
        return latent, complete

    def store_prefill(
        self,
        latent: torch.Tensor | None,
        block_starts: torch.Tensor,
        *,
        table_idx: int,
    ) -> None:
        if latent is None:
            return
        assert self._freqs_cis is not None
        values = latent.clone()
        apply_rotary_emb(
            values[..., -self.rope_head_dim :],
            self._freqs_cis.index_select(0, block_starts),
        )
        fp4_act_quant_e4m3_inplace(values, 16)
        rows = self.attn.compress_rows_of(table_idx, block_starts, self.ratio)
        self.attn.scatter_compressed(self.layer_id, "attn", rows, values[0])

    def store_decode(
        self,
        latent: torch.Tensor,
        complete: torch.Tensor,
        *,
        pos: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        assert self._freqs_cis is not None
        values = latent.clone()
        starts = (pos + 1 - self.ratio).clamp_min(0)
        apply_rotary_emb_decode(
            values[..., -self.rope_head_dim :],
            self._freqs_cis.index_select(0, starts),
        )
        fp4_act_quant_e4m3_inplace(values, 16)
        dst = self.attn.decode_compress_rows(
            rows, pos, self.ratio, self.layer_id, "attn", complete
        )
        self.attn.scatter_compressed(self.layer_id, "attn", dst, values.view(pos.numel(), -1))


def _mask_unreachable(
    logits: torch.Tensor, live: torch.Tensor | int
) -> torch.Tensor:
    columns = torch.arange(logits.shape[-1], device=logits.device).view(
        *((1,) * (logits.ndim - 1)), logits.shape[-1]
    )
    return logits.masked_fill(columns >= live, -torch.inf)


def select_candidate_blocks(
    logits: torch.Tensor,
    live: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    logits = _mask_unreachable(logits, live)
    width = logits.shape[-1]
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    last = (live - 1) // block_size
    arange = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(arange == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(BaseOP):
    def __init__(
        self,
        args: DeepseekV41Args,
        layer_id: int,
        *,
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.owns_k = layer_id in args.kv_source_layers
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.ratio = args.compress_ratios[layer_id]
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.local_heads = args.index_n_heads // get_tp_info().size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = LinearColParallelMerged(
            self.q_lora_rank,
            [self.n_heads * self.head_dim],
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = LinearColParallelMerged(
            self.dim,
            [self.n_heads],
            has_bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        if self.owns_k:
            self.wk = LinearReplicated(
                args.head_dim,
                args.index_head_dim,
                has_bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(args.index_head_dim, args.norm_eps)
        else:
            self.wk = None
            self.k_norm = None
        self._freqs_cis: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, freqs_cis: torch.Tensor) -> None:
        self._freqs_cis = freqs_cis

    def _queries(
        self, qr: torch.Tensor, positions: torch.Tensor, *, decode: bool = False
    ) -> torch.Tensor:
        assert self._freqs_cis is not None
        q = self.wq_b.forward(qr).unflatten(-1, (self.local_heads, self.head_dim))
        freqs = self._freqs_cis.index_select(0, positions)
        if decode:
            apply_rotary_emb_decode(q[..., -self.rope_head_dim :], freqs)
        else:
            apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        fp4_act_quant_inplace(q, 32)
        return q

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights_proj.forward(x) * (
            self.softmax_scale * self.n_heads**-0.5
        )

    def publish_prefill_keys(
        self,
        latent: torch.Tensor | None,
        block_starts: torch.Tensor,
        *,
        table_idx: int,
        source_layer: int,
    ) -> None:
        if not self.owns_k or latent is None:
            return
        assert self._freqs_cis is not None and self.wk is not None and self.k_norm is not None
        key = self.k_norm.forward(self.wk.forward(latent))
        apply_rotary_emb(
            key[..., -self.rope_head_dim :],
            self._freqs_cis.index_select(0, block_starts),
        )
        fp4_act_quant_inplace(key, 32)
        rows = self.attn.compress_rows_of(table_idx, block_starts, self.ratio)
        self.attn.scatter_compressed(source_layer, "idx", rows, key[0])

    def publish_decode_keys(
        self,
        latent: torch.Tensor,
        complete: torch.Tensor,
        *,
        pos: torch.Tensor,
        rows: torch.Tensor,
        source_layer: int,
    ) -> None:
        if not self.owns_k:
            return
        assert self._freqs_cis is not None and self.wk is not None and self.k_norm is not None
        key = self.k_norm.forward(self.wk.forward(latent))
        starts = (pos + 1 - self.ratio).clamp_min(0)
        apply_rotary_emb_decode(
            key[..., -self.rope_head_dim :],
            self._freqs_cis.index_select(0, starts),
        )
        fp4_act_quant_inplace(key, 32)
        dst = self.attn.decode_compress_rows(
            rows, pos, self.ratio, source_layer, "idx", complete
        )
        self.attn.scatter_compressed(source_layer, "idx", dst, key.view(pos.numel(), -1))

    def prefill_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        *,
        start_pos: int,
        table_idx: int,
        source_layer: int,
        candidate_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = x.shape[1]
        end = start_pos + n
        n_blocks = end // self.ratio
        if n_blocks == 0:
            empty = torch.empty((1, n, 0), dtype=torch.int32, device=x.device)
            return empty, candidate_mask
        positions = torch.arange(start_pos, end, device=x.device)
        q = self._queries(qr, positions)
        weights = self._weights(x)
        keys = self.attn.indexer_keys(
            table_idx, n_blocks, self.ratio, source_layer, 1
        )
        scores = self.attn.indexer_prefill_logits(q, keys, weights)
        if get_tp_info().size > 1:
            scores = self._comm.all_reduce(scores)
        live = (
            (start_pos + torch.arange(1, n + 1, device=x.device)) // self.ratio
        ).view(1, n, 1)
        # Candidate selection assumes unreachable positions already score -inf. Mask
        # them before the level-one block top-k so future blocks cannot consume the
        # limited candidate budget and displace reachable history.
        scores = _mask_unreachable(scores, live)
        if self.is_candidate_source:
            candidate_mask = select_candidate_blocks(
                scores,
                live,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            if candidate_mask is None:
                raise RuntimeError("V4.1 candidate consumer ran before its source")
            scores = scores.masked_fill(~candidate_mask, -torch.inf)
        picks = self.attn.indexer_select_prefill(
            scores,
            start_pos=start_pos,
            seqlen=n,
            ratio=self.ratio,
            topk=self.index_topk,
            offset=0,
            sort_by_position=True,
        )
        return picks.to(torch.int32), candidate_mask

    def decode_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        *,
        pos: torch.Tensor,
        n_stage: int,
        source_layer: int,
        candidate_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if n_stage == 0:
            return (
                torch.empty(
                    (pos.numel(), 1, 0), dtype=torch.int32, device=pos.device
                ),
                candidate_mask,
            )
        q = self._queries(
            qr.view(qr.shape[0], 1, -1), pos, decode=True
        ).squeeze(1)
        weights = self._weights(x).view(x.shape[0], -1)
        valid = (pos + 1) // self.ratio
        scores = self.attn.indexer_decode_scores(
            q,
            weights,
            valid,
            n_stage,
            self.ratio,
            source_layer,
        )
        if get_tp_info().size > 1:
            scores = self._comm.all_reduce(scores)
        scores = _mask_unreachable(scores, valid[:, None])
        if self.is_candidate_source:
            candidate_mask = select_candidate_blocks(
                scores, valid[:, None], self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            if candidate_mask is None:
                raise RuntimeError("V4.1 candidate consumer ran before its source")
            scores = scores.masked_fill(~candidate_mask, -torch.inf)
        picks = self.attn.indexer_select_decode(
            scores[:, None], valid=valid, topk=self.index_topk, offset=0,
            sort_by_position=True,
        )
        return picks.to(torch.int32), candidate_mask


__all__ = ["Compressor", "Indexer", "select_candidate_blocks"]
