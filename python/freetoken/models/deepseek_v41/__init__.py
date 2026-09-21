"""DeepSeek-V4.1-Flash text, MoE, Engram and vision support."""

from .args import DeepseekV41Args, load_args
from .config import VisionConfig, parse_config, parse_vision_config
from .model import DeepseekV41ForCausalLM
from .vision import DeepseekV41VisionModel
from .weight import iter_expert_pieces, iter_vision_weights, iter_weights

__all__ = [
    "DeepseekV41Args",
    "DeepseekV41ForCausalLM",
    "DeepseekV41VisionModel",
    "VisionConfig",
    "iter_expert_pieces",
    "iter_vision_weights",
    "iter_weights",
    "load_args",
    "parse_config",
    "parse_vision_config",
]
