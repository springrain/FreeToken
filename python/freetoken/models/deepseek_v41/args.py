"""DeepSeek-V4.1-Flash model geometry from ``inference/config.json``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace
from typing import Literal

from freetoken.utils.hf import optional_hf_file


@dataclass
class DeepseekV41Args:
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal[None, "fp4"] = "fp4"

    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64

    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = ()
    kv_source_layers: tuple[int, ...] = ()
    index_source_layers: tuple[int, ...] = ()

    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    compress_rope_theta: float = 160000.0

    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092

    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936
    vision_max_wh_ratio: int | None = None
    image_token_id: int = 129264

    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    def __post_init__(self) -> None:
        tuple_fields = (
            "compress_ratios",
            "kv_source_layers",
            "index_source_layers",
            "engram_layer_ids",
            "engram_num_embeddings",
            "dspark_target_layer_ids",
        )
        for name in tuple_fields:
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        ratios = self.compress_ratios[: self.n_layers]
        if len(ratios) != self.n_layers:
            raise ValueError(
                f"DeepSeek-V4.1 needs {self.n_layers} backbone compress ratios, got {len(ratios)}"
            )
        if any(r not in (0, 1, 2) for r in ratios):
            raise ValueError(f"DeepSeek-V4.1 supports compression ratios 0/1/2, got {ratios}")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings must have equal length")

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def without_vision(self) -> "DeepseekV41Args":
        return replace(self, vision_n_layers=0)

    @staticmethod
    def _source_for(layer_id: int, sources: tuple[int, ...]) -> int | None:
        return next((source for source in reversed(sources) if source <= layer_id), None)

    def kv_source_for(self, layer_id: int) -> int | None:
        return self._source_for(layer_id, self.kv_source_layers)

    def index_source_for(self, layer_id: int) -> int | None:
        return self._source_for(layer_id, self.index_source_layers)


def _config_path(model_path: str) -> str:
    candidates = (
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "model_args.json"),
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    if not os.path.isdir(model_path):
        for filename in ("inference/config.json", "model_args.json"):
            path = optional_hf_file(model_path, filename)
            if path is not None:
                return path
    raise FileNotFoundError(
        f"No DeepSeek-V4.1 ModelArgs JSON found under {model_path} "
        "(looked for inference/config.json and model_args.json)"
    )


def load_args(model_path: str, **overrides) -> DeepseekV41Args:
    with open(_config_path(model_path), encoding="utf-8") as f:
        raw = json.load(f)
    valid = {item.name for item in fields(DeepseekV41Args)}
    kwargs = {key: value for key, value in raw.items() if key in valid}
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


__all__ = ["DeepseekV41Args", "load_args"]
