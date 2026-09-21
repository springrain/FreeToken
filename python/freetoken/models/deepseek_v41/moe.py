"""DeepSeek-V4.1 owner-local MoE with image-specific routing bias."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.layers import BaseOP
from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer, Expert

from .args import DeepseekV41Args


class Gate(BaseOP):
    def __init__(self, args: DeepseekV41Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = torch.empty(args.n_routed_experts, args.dim)
        self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        self.bias_vl = (
            torch.empty(args.n_routed_experts, dtype=torch.float32)
            if args.vision_enabled
            else None
        )

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = bf16_linear_fp32(x, self.weight) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask[:, None], self.bias_vl[None, :], bias[None, :])
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


class MoE(BaseOP):
    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        expert_tp_size: int | None = None,
        quant_config=None,
        prefix: str = "",
    ):
        self.dim = args.dim
        self.gate = Gate(args)
        self.shared_experts = Expert(
            args.dim,
            args.moe_inter_dim,
            args.swiglu_limit,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
        )
        self.experts = DSV4OffloadMoELayer(
            layer_id,
            args,
            strategy=strategy,
            decode_target=decode_target,
            expert_tp_size=expert_tp_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = x.shape
        hidden = x.reshape(-1, self.dim)
        mask = None if image_mask is None else image_mask.reshape(-1)
        weights, indices = self.gate.forward(hidden, mask)
        owner_ep = self.experts.owner_cache is not None
        shared = self.shared_experts.forward(hidden, reduce=not owner_ep)
        routed = self.experts.routed_forward(
            hidden,
            weights.float().contiguous(),
            indices.to(torch.int32).contiguous(),
            reduce=not owner_ep,
        )
        if owner_ep:
            routed = self.experts._maybe_all_reduce(routed + shared)
            return routed.view(shape)
        return (routed + shared).view(shape)


__all__ = ["Gate", "MoE"]
