"""Engine-facing configuration for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import DeepseekV41Args, load_args


@dataclass(frozen=True)
class VisionConfig:
    num_layers: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: int | None
    out_hidden_size: int
    image_token_id: int


def parse_vision_config(hf_config: Any, args: DeepseekV41Args) -> VisionConfig | None:
    if getattr(hf_config, "vision_config", None) is None or not args.vision_enabled:
        return None
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


def parse_config(hf_config: Any) -> ModelConfig:
    model_path = getattr(hf_config, "_name_or_path", None) or getattr(
        hf_config, "name_or_path", None
    )
    if not model_path:
        raise ValueError(
            "DeepSeek-V4.1 parse_config needs the checkpoint path "
            "(hf_config._name_or_path)"
        )
    args = load_args(model_path, max_batch_size=1)
    if getattr(hf_config, "vision_config", None) is None:
        args = args.without_vision()

    text = getattr(hf_config, "text_config", hf_config)
    max_position = int(getattr(text, "max_position_embeddings", 0) or 0)
    if max_position <= 0:
        max_position = int(args.original_seq_len * args.rope_factor)
    rope = RotaryConfig(
        head_dim=args.head_dim,
        rotary_dim=args.rope_head_dim,
        max_position=max_position,
        base=args.rope_theta,
        scaling={
            "rope_type": "yarn",
            "factor": args.rope_factor,
            "beta_fast": args.beta_fast,
            "beta_slow": args.beta_slow,
            "original_max_position_embeddings": args.original_seq_len,
        },
    )

    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act="silu",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=rope,
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="deepseek_v41",
        architectures=list(
            getattr(hf_config, "architectures", ["DeepseekV41ForCausalLM"])
        ),
        moe_enabled=True,
        expert_quant="ds_fp4",
        owner_ep_expert_quants=("ds_fp4",),
        weight_block_size=(32, 32),
        n_shared_experts=args.n_shared_experts,
        shared_expert_intermediate_size=args.moe_inter_dim,
        routed_scaling_factor=args.route_scale,
        has_router_bias=True,
        swiglu_limit=args.swiglu_limit,
        attn_sm_scale=args.head_dim**-0.5,
        dsv4_args=args,
        vision_config=parse_vision_config(hf_config, args),
        image_token_id=args.image_token_id,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv41",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
    )


__all__ = ["VisionConfig", "parse_config", "parse_vision_config"]
