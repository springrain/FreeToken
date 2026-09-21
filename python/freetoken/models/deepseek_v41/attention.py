"""DeepSeek-V4.1 CSA2 attention with cross-layer KV and index reuse."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    RMSNorm,
)
from freetoken.models.deepseek_v4.layers import get_window_topk_idxs
from freetoken.models.deepseek_v4.ops import (
    apply_rotary_emb,
    apply_rotary_emb_decode,
    get_freqs_cis,
)

from .args import DeepseekV41Args
from .compress import Compressor, Indexer


@dataclass
class SharedAttentionRuntime:
    """Per-forward CSA2 hand-off between source and reuse layers."""

    prefill_picks: list[torch.Tensor | None]
    prefill_candidates: list[torch.Tensor | None]
    decode_picks: torch.Tensor | None = None
    decode_candidates: torch.Tensor | None = None

    @classmethod
    def prefill(cls, num_segments: int) -> "SharedAttentionRuntime":
        return cls([None] * num_segments, [None] * num_segments)

    @classmethod
    def decode(cls) -> "SharedAttentionRuntime":
        return cls([], [])


class Attention(BaseOP):
    """Sliding-window MLA plus shared compressed KV and shared index selections."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.kv_source_layer = args.kv_source_for(layer_id)
        self.index_source_layer = args.index_source_for(layer_id)

        tp_size = get_tp_info().size
        if self.n_heads % tp_size:
            raise ValueError(
                "DeepSeek-V4.1 TP needs n_heads divisible by tp_size, "
                f"got {self.n_heads} % {tp_size}"
            )
        if self.n_groups % tp_size:
            raise ValueError(
                "DeepSeek-V4.1 TP needs o_groups divisible by tp_size, "
                f"got {self.n_groups} % {tp_size}"
            )
        if self.n_heads % self.n_groups:
            raise ValueError(
                "DeepSeek-V4.1 grouped output needs n_heads divisible by o_groups, "
                f"got {self.n_heads} % {self.n_groups}"
            )
        if layer_id in args.index_source_layers and args.index_n_heads % tp_size:
            raise ValueError(
                "DeepSeek-V4.1 TP needs index_n_heads divisible by tp_size, "
                f"got {args.index_n_heads} % {tp_size}"
            )
        self.n_heads_local = self.n_heads // tp_size
        self.n_groups_local = self.n_groups // tp_size

        if self.compress_ratio:
            if self.kv_source_layer is None or self.index_source_layer is None:
                raise ValueError(
                    f"compressed layer {layer_id} has no preceding KV/index source"
                )
            source_ratio = args.compress_ratios[self.kv_source_layer]
            index_ratio = args.compress_ratios[self.index_source_layer]
            if source_ratio != self.compress_ratio or index_ratio != self.compress_ratio:
                raise ValueError(
                    f"layer {layer_id} ratio {self.compress_ratio} does not match its "
                    f"KV/index sources ({source_ratio}, {index_ratio})"
                )

        self.attn_sink = torch.empty(self.n_heads_local, dtype=torch.float32)
        self.wq_a = LinearReplicated(
            self.dim,
            self.q_lora_rank,
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
        self.wq_b = LinearColParallelMerged(
            self.q_lora_rank,
            [self.n_heads * self.head_dim],
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wkv = LinearReplicated(
            self.dim,
            self.head_dim,
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        rows = self.n_groups_local * self.o_lora_rank
        group_k = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = torch.empty(rows, group_k, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(
            self.n_groups * self.o_lora_rank,
            self.dim,
            has_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.compressor = (
            Compressor(args, layer_id, prefix=f"{prefix}.compressor")
            if layer_id in args.kv_source_layers
            else None
        )
        self.indexer = (
            Indexer(
                args,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer",
            )
            if layer_id in args.index_source_layers
            else None
        )
        if self.compressor is not None and self.indexer is None:
            raise ValueError(
                f"KV source layer {layer_id} must also publish an index source"
            )

        if self.compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = args.compress_rope_theta
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta
        self._freqs_params = (
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    @property
    def attn(self):
        return get_global_ctx().attn_backend

    def bind(self, device: torch.device) -> None:
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)
        if self.compressor is not None:
            self.compressor.bind(self._freqs_cis, device)
        if self.indexer is not None:
            self.indexer.bind(self._freqs_cis)

    def _output(self, value: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        value = value.reshape(bsz, seqlen, self.n_groups_local, -1)
        wo_a = self.wo_a.view(self.n_groups_local, self.o_lora_rank, -1)
        value = torch.einsum("bsgd,grd->bsgr", value, wo_a).flatten(2)
        return self.wo_b.forward(value)

    def _prefill_window(
        self,
        kv: torch.Tensor,
        *,
        table_idx: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        n = kv.shape[0]
        end = start_pos + n
        slots = self.attn.window_slots_of(table_idx, start_pos, end)
        self.attn.store_window(kv, self.layer_id, slots)
        if start_pos == 0:
            columns = get_window_topk_idxs(self.window_size, 1, n, 0).to(kv.device)
            global_slots = self.attn.win_cols_to_global(columns, slots)
        else:
            lo = max(0, start_pos - self.window_size + 1)
            lookup = self.attn.window_slots_of(table_idx, lo, end)
            absolute = start_pos + torch.arange(n, device=kv.device).unsqueeze(1)
            candidates = (
                (absolute - self.window_size + 1).clamp(min=lo)
                + torch.arange(self.window_size, device=kv.device)
            )
            columns = torch.where(
                candidates > absolute,
                -1,
                candidates - lo,
            ).unsqueeze(0)
            global_slots = self.attn.win_cols_to_global(columns, lookup)
        tail = (
            int(self.attn.window_slots_of(table_idx, start_pos - 1, start_pos).item())
            if start_pos > 0 and self.compressor is not None and self.compress_ratio > 1
            else None
        )
        return global_slots, slots, tail

    def _prefill_compressed(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        *,
        segment_index: int,
        table_idx: int,
        start_pos: int,
        window_slots: torch.Tensor,
        tail_window_slot: int | None,
        runtime: SharedAttentionRuntime,
    ) -> torch.Tensor:
        latent = None
        starts = torch.empty(0, dtype=torch.int64, device=x.device)
        if self.compressor is not None:
            latent, starts = self.compressor.prefill(
                x,
                start_pos=start_pos,
                window_slots=window_slots,
                tail_window_slot=tail_window_slot,
            )
            assert self.indexer is not None
            self.indexer.publish_prefill_keys(
                latent,
                starts,
                table_idx=table_idx,
                source_layer=self.layer_id,
            )
            self.compressor.store_prefill(latent, starts, table_idx=table_idx)

        if self.indexer is not None:
            picks, candidates = self.indexer.prefill_scores(
                x,
                qr,
                start_pos=start_pos,
                table_idx=table_idx,
                source_layer=self.kv_source_layer,
                candidate_mask=runtime.prefill_candidates[segment_index],
            )
            runtime.prefill_picks[segment_index] = picks
            runtime.prefill_candidates[segment_index] = candidates
        else:
            picks = runtime.prefill_picks[segment_index]
            if picks is None:
                raise RuntimeError(
                    f"layer {self.layer_id} reused index picks before source "
                    f"layer {self.index_source_layer} published them"
                )
        return self.attn.blocks_to_global(
            picks, self.compress_ratio, ti=table_idx
        )

    def forward_ragged(
        self,
        x: torch.Tensor,
        segments,
        flat_positions: torch.Tensor,
        runtime: SharedAttentionRuntime,
    ) -> torch.Tensor:
        assert self._freqs_cis is not None
        _, total, _ = x.shape
        if len(segments) == 1:
            start = segments[0][3]
            freqs = self._freqs_cis[start : start + total]
        else:
            freqs = self._freqs_cis.index_select(0, flat_positions)

        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(
            -1, (self.n_heads_local, self.head_dim)
        )
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
        act_quant_fp8_inplace(kv, 32)

        window_parts = []
        compressed_parts = []
        max_compressed = 0
        for index, (offset, n, table_idx, start_pos) in enumerate(segments):
            window, slots, tail = self._prefill_window(
                kv[0, offset : offset + n],
                table_idx=table_idx,
                start_pos=start_pos,
            )
            window_parts.append(window)
            if self.compress_ratio:
                compressed = self._prefill_compressed(
                    x[:, offset : offset + n],
                    qr[:, offset : offset + n],
                    segment_index=index,
                    table_idx=table_idx,
                    start_pos=start_pos,
                    window_slots=slots,
                    tail_window_slot=tail,
                    runtime=runtime,
                )
                compressed_parts.append(compressed)
                max_compressed = max(max_compressed, compressed.shape[-1])

        n_window = (
            self.window_size if len(segments) > 1 else window_parts[0].shape[-1]
        )
        flat_topk = []
        for index, window in enumerate(window_parts):
            if window.shape[-1] < n_window:
                window = F.pad(window, (0, n_window - window.shape[-1]), value=-1)
            if self.compress_ratio:
                compressed = compressed_parts[index]
                if compressed.shape[-1] < max_compressed:
                    compressed = F.pad(
                        compressed,
                        (0, max_compressed - compressed.shape[-1]),
                        value=-1,
                    )
                window = torch.cat((window, compressed), dim=-1)
            flat_topk.append(window)
        topk = flat_topk[0] if len(flat_topk) == 1 else torch.cat(flat_topk, dim=1)

        value = self.attn.attend(
            q,
            self.layer_id,
            topk.int(),
            n_window,
            self.attn_sink,
            self.softmax_scale,
            has_compression=bool(self.compress_ratio),
            cmp_layer_id=self.kv_source_layer,
        )
        apply_rotary_emb(value[..., -self.rope_head_dim :], freqs, True)
        return self._output(value, 1, total)

    def decode_step(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rows: torch.Tensor,
        cmp_stage_cap: int,
        runtime: SharedAttentionRuntime,
        wctx,
    ) -> torch.Tensor:
        assert self._freqs_cis is not None
        batch = x.shape[0]
        freqs = self._freqs_cis.index_select(0, pos)
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(
            -1, (self.n_heads_local, self.head_dim)
        )
        apply_rotary_emb_decode(q[..., -self.rope_head_dim :], freqs)
        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb_decode(kv[..., -self.rope_head_dim :], freqs)
        act_quant_fp8_inplace(kv, 32)

        window_slots, prev_window_slots, window_topk = wctx
        self.attn.store_window(kv.view(batch, -1), self.layer_id, window_slots)
        compressed_topk = None
        compressed_counts = None
        if self.compress_ratio:
            if self.compressor is not None:
                latent, complete = self.compressor.decode(
                    x,
                    pos=pos,
                    prev_window_slots=prev_window_slots,
                    window_slots=window_slots,
                )
                assert self.indexer is not None
                self.indexer.publish_decode_keys(
                    latent,
                    complete,
                    pos=pos,
                    rows=rows,
                    source_layer=self.layer_id,
                )
                self.compressor.store_decode(
                    latent, complete, pos=pos, rows=rows
                )

            if self.indexer is not None:
                n_stage = (cmp_stage_cap + 1) // self.compress_ratio
                picks, candidates = self.indexer.decode_scores(
                    x,
                    qr,
                    pos=pos,
                    n_stage=n_stage,
                    source_layer=self.kv_source_layer,
                    candidate_mask=runtime.decode_candidates,
                )
                runtime.decode_picks = picks
                runtime.decode_candidates = candidates
            else:
                picks = runtime.decode_picks
                if picks is None:
                    raise RuntimeError(
                        f"layer {self.layer_id} reused index picks before source "
                        f"layer {self.index_source_layer} published them"
                    )
            compressed_topk = self.attn.blocks_to_global(
                picks, self.compress_ratio, rows=rows
            )
            live = (pos + 1) // self.compress_ratio
            compressed_counts = live.clamp(max=compressed_topk.shape[-1]).to(
                torch.int32
            ).view(batch, 1)

        topk = (
            window_topk
            if compressed_topk is None
            else torch.cat((window_topk, compressed_topk), dim=-1)
        )
        value = self.attn.attend(
            q,
            self.layer_id,
            topk.int(),
            self.window_size,
            self.attn_sink,
            self.softmax_scale,
            cmp_counts=compressed_counts,
            has_compression=bool(self.compress_ratio),
            cmp_layer_id=self.kv_source_layer,
        )
        apply_rotary_emb_decode(
            value[..., -self.rope_head_dim :], freqs, True
        )
        return self._output(value, batch, 1)


__all__ = ["Attention", "SharedAttentionRuntime"]
