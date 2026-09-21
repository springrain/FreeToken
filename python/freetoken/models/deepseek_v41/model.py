"""Engine-native DeepSeek-V4.1 Flash backbone and multimodal wrapper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids

from .args import DeepseekV41Args
from .attention import Attention, SharedAttentionRuntime
from .engram import (
    EngramHash,
    EngramLayer,
    EngramLayout,
    ZeroEngramTable,
    load_engram_table,
)
from .moe import MoE
from .vision import (
    DeepseekV41Aligner,
    DeepseekV41VisionMixin,
    DeepseekV41VisionModel,
)


def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    mix = x.new_zeros((*x.shape[:2], hc_mult), dtype=torch.float32)
    mix[..., 0] = 1.0
    return mix


class Block(BaseOP):
    """Single-pass mHC block: each sublayer publishes the next pre-mix."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        engram_layout: EngramLayout | None,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        expert_tp_size: int | None = None,
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.dim = args.dim
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.attn = Attention(
            layer_id,
            args,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn = MoE(
            layer_id,
            args,
            strategy=strategy,
            decode_target=decode_target,
            expert_tp_size=expert_tp_size,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.engram = (
            EngramLayer(
                args,
                layer_id,
                engram_layout,
                quant_config=quant_config,
                prefix=f"{prefix}.engram",
            )
            if engram_layout is not None and layer_id in engram_layout.layer_ids
            else None
        )
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * args.dim
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def hc_mixes(self, x, weight, scale, base):
        shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, weight) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes.reshape(-1, mixes.shape[-1]),
            scale,
            base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        count = shape[0] * shape[1]
        return (
            pre.view(count, self.hc_mult),
            post.view(count, self.hc_mult),
            comb.view(count, self.hc_mult, self.hc_mult),
        )

    def hc_pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        count = shape[0] * shape[1]
        return hc_pre_combine(
            x.reshape(count, self.hc_mult, self.dim).float(),
            pre.reshape(count, self.hc_mult),
            x.dtype,
        ).view(*shape[:2], self.dim)

    def hc_post(self, x, residual, post, comb):
        shape = residual.shape
        count = shape[0] * shape[1]
        return hc_post_combine(
            x.reshape(count, self.dim),
            residual.reshape(count, self.hc_mult, self.dim),
            post,
            comb,
        ).view(shape)

    def prefill_batched(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        segments,
        flat_positions: torch.Tensor,
        image_mask: torch.Tensor | None,
        runtime: SharedAttentionRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden = self.hc_pre(x, pre_mix)
        hidden = self.attn_norm.forward(hidden)
        hidden = self.attn.forward_ragged(
            hidden, segments, flat_positions, runtime
        )
        x = self.hc_post(hidden, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden = self.hc_pre(x, attn_pre)
        hidden = self.ffn_norm.forward(hidden)
        hidden = self.ffn.forward(hidden, image_mask)
        return self.hc_post(hidden, residual, ffn_post, ffn_comb), ffn_pre

    def decode_step(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        pos: torch.Tensor,
        rows: torch.Tensor,
        cmp_stage_cap: int,
        runtime: SharedAttentionRuntime,
        wctx,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden = self.hc_pre(x, pre_mix)
        hidden = self.attn_norm.forward(hidden)
        hidden = self.attn.decode_step(
            hidden, pos, rows, cmp_stage_cap, runtime, wctx
        )
        x = self.hc_post(hidden, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden = self.hc_pre(x, attn_pre)
        hidden = self.ffn_norm.forward(hidden)
        hidden = self.ffn.forward(hidden)
        return self.hc_post(hidden, residual, ffn_post, ffn_comb), ffn_pre


class Transformer(BaseOP):
    def __init__(
        self,
        args: DeepseekV41Args,
        quant_config=None,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        expert_tp_size: int | None = None,
        prefix: str = "model",
    ):
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self._engram_layout = EngramLayout.from_args(args)
        self._engram_hash = (
            EngramHash(args, self._engram_layout)
            if self._engram_layout is not None
            else None
        )
        self.layers = OPList([
            Block(
                layer_id,
                args,
                self._engram_layout,
                strategy=strategy,
                decode_target=decode_target,
                expert_tp_size=expert_tp_size,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_id}",
            )
            for layer_id in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, args.norm_eps)

    @property
    def engram_layers(self) -> list[EngramLayer]:
        return [
            layer.engram
            for layer in self.layers.op_list
            if layer.engram is not None
        ]

    @property
    def engram_hash(self) -> EngramHash | None:
        return self._engram_hash

    def bind(self, device: torch.device) -> None:
        for layer in self.layers.op_list:
            layer.attn.bind(device)

    @staticmethod
    def _image_mask(batch, token_count: int, device: torch.device):
        if batch.mm_rows is None:
            return None
        mask = torch.zeros(token_count, dtype=torch.bool, device=device)
        mask.index_fill_(0, batch.mm_rows.long(), True)
        return mask.view(1, token_count)

    def _final_hidden(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        if not self.layers.op_list:
            raise RuntimeError("DeepSeek-V4.1 needs at least one backbone layer")
        return self.norm.forward(self.layers.op_list[-1].hc_pre(x, pre_mix))

    @staticmethod
    def _apply_engram(
        layer: EngramLayer,
        hidden: torch.Tensor,
        row_ids: torch.Tensor | None,
        image_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if row_ids is None:
            raise RuntimeError("Engram layer ran without hash row ids")
        token_mask = None if image_mask is None else ~image_mask
        return layer.forward(hidden, row_ids, token_mask)

    def prefill_batched(
        self,
        input_ids: torch.Tensor,
        batch,
        row_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        hidden = embed_input_ids(self.embed, flat_ids, batch).view(
            1, -1, self.args.dim
        )
        image_mask = self._image_mask(batch, flat_ids.numel(), flat_ids.device)
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = identity_pre_mix(hidden, self.hc_mult)
        runtime = SharedAttentionRuntime.prefill(len(batch.attn_metadata.segments))
        for layer in self.layers.op_list:
            if layer.engram is not None:
                hidden = self._apply_engram(
                    layer.engram, hidden, row_ids, image_mask
                )
            hidden, pre_mix = layer.prefill_batched(
                hidden,
                pre_mix,
                batch.attn_metadata.segments,
                batch.positions.long(),
                image_mask,
                runtime,
            )
        return self._final_hidden(hidden, pre_mix)[0]

    def decode(
        self,
        input_ids: torch.Tensor,
        batch,
        row_ids: torch.Tensor | None,
        cmp_stage_cap: int,
    ) -> torch.Tensor:
        size = input_ids.shape[0]
        pos = batch.positions.long().view(-1)[:size]
        rows = torch.arange(size, device=input_ids.device)
        hidden = self.embed.forward(input_ids.reshape(-1)).view(
            size, 1, self.args.dim
        )
        hidden = hidden.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = identity_pre_mix(hidden, self.hc_mult)
        runtime = SharedAttentionRuntime.decode()
        wctx = batch.attn_metadata.window_ctx(pos, rows)
        for layer in self.layers.op_list:
            if layer.engram is not None:
                hidden = self._apply_engram(layer.engram, hidden, row_ids, None)
            hidden, pre_mix = layer.decode_step(
                hidden,
                pre_mix,
                pos,
                rows,
                cmp_stage_cap,
                runtime,
                wctx,
            )
        return self._final_hidden(hidden, pre_mix)[:, -1]


class DeepseekV41ForCausalLM(DeepseekV41VisionMixin, BaseLLMModel):
    def __init__(self, config):
        self._config = config
        self._args: DeepseekV41Args = config.dsv4_args
        expert_tp_size = 1 if getattr(config, "moe_ep_size", 1) > 1 else None
        self.model = Transformer(
            self._args,
            config.quant,
            strategy=config.moe_strategy,
            decode_target=config.decode_target,
            expert_tp_size=expert_tp_size,
            prefix="model",
        )
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=config.quant,
            prefix="head",
        )
        if config.is_multimodal:
            self.vision = DeepseekV41VisionModel(
                config.vision_config,
                quant_config=config.quant,
                prefix="vision",
            )
            self.aligner = DeepseekV41Aligner(
                config.vision_config,
                quant_config=config.quant,
                prefix="aligner",
            )
            self.image_start = torch.empty(config.hidden_size)
            self.image_newline = torch.empty(config.hidden_size)
            self.image_end = torch.empty(config.hidden_size)
        self._bound = False
        self._engram_tables = []

    def load_host_tables(self, engine_config) -> int:
        layers = self.model.engram_layers
        hash_state = self.model.engram_hash
        if not layers or hash_state is None:
            return 0
        if not engine_config.use_dummy_weight:
            from freetoken.checkpoint.ftw import is_ftw_checkpoint

            if is_ftw_checkpoint(engine_config.model_path):
                raise NotImplementedError(
                    "DeepSeek-V4.1 FTW loading is not supported yet: the two Engram "
                    "tables are safetensors side data and are not written by the FTW converter"
                )
        device = self.model.embed.weight.device
        hash_state.initialize(engine_config.model_path, device)
        if engine_config.use_dummy_weight:
            for layer in layers:
                layer.attach_table(ZeroEngramTable(self._args.engram_head_dim))
            return 0

        total = 0
        by_layer = dict(
            zip(self._args.engram_layer_ids, self._args.engram_num_embeddings)
        )
        for layer in layers:
            table, nbytes = load_engram_table(
                engine_config.model_path,
                layer_id=layer.layer_id,
                num_embeddings=by_layer[layer.layer_id],
                embed_dim=self._args.engram_head_dim,
                device=device,
            )
            layer.attach_table(table)
            self._engram_tables.append(table)
            total += nbytes
        return total

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        self.model.bind(get_global_ctx().kv_cache.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        self._bound = False
        if self.model.engram_hash is not None:
            self.model.engram_hash._token_cache_initialized = False

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        hash_state = self.model.engram_hash
        row_ids = None if hash_state is None else hash_state.row_ids(batch)
        if batch.is_prefill:
            hidden = self.model.prefill_batched(input_ids, batch, row_ids)
        else:
            size = batch.padded_size
            if torch.cuda.is_current_stream_capturing():
                cmp_stage_cap = batch.attn_metadata.stage_width - 1
            else:
                cmp_stage_cap = int(batch.positions[:size].max().item())
            hidden = self.model.decode(
                input_ids.view(size, 1), batch, row_ids, cmp_stage_cap
            )
        return self.head.forward(hidden)


__all__ = [
    "Block",
    "DeepseekV41ForCausalLM",
    "Transformer",
    "identity_pre_mix",
]
