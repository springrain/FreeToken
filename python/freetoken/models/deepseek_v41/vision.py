"""DeepSeek-V4.1 vision tower and image-span embedding assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.distributed import get_tp_info
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    OPList,
)
from freetoken.models.weight_stream import BlockWeightStreamer
from freetoken.utils import div_even

if TYPE_CHECKING:
    from freetoken.layers.quantization import QuantConfig
    from freetoken.message import MMItem

    from .config import VisionConfig


def _shard_dim0(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if tensor.shape[0] % world_size:
        raise ValueError(
            f"dimension 0 ({tensor.shape[0]}) is not divisible by TP={world_size}"
        )
    return tensor.chunk(world_size, dim=0)[rank].contiguous()


def _shard_dim1(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[1] % world_size:
        raise ValueError(
            f"dimension 1 of {tuple(tensor.shape)} is not divisible by TP={world_size}"
        )
    return tensor.chunk(world_size, dim=1)[rank].contiguous()


def shard_deepseek_v41_vision_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    config: VisionConfig,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Shard one raw official vision/aligner tensor to the rank-local model shape."""
    if world_size == 1:
        return tensor
    if not 0 <= rank < world_size:
        raise ValueError(f"TP rank {rank} is outside [0, {world_size})")
    if name.endswith((".attn.wqkv.weight", ".attn.wqkv.bias")):
        rows = config.hidden_size
        if tensor.shape[0] != 3 * rows:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, expected QKV={3 * rows}"
            )
        return torch.cat(
            [_shard_dim0(part, rank, world_size) for part in tensor.split(rows, dim=0)],
            dim=0,
        ).contiguous()
    if name.endswith(".mlp.w1.weight"):
        rows = config.intermediate_size
        if tensor.shape[0] != 2 * rows:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows, expected gate|up={2 * rows}"
            )
        return torch.cat(
            [_shard_dim0(part, rank, world_size) for part in tensor.split(rows, dim=0)],
            dim=0,
        ).contiguous()
    if name in ("aligner.w1.weight", "aligner.w1.bias"):
        return _shard_dim0(tensor, rank, world_size)
    if (
        name.endswith((".attn.wo.weight", ".mlp.w2.weight"))
        or name == "aligner.w2.weight"
    ):
        return _shard_dim1(tensor, rank, world_size)
    if name.endswith(".attn.wo.bias") or name == "aligner.w2.bias":
        return tensor if rank == 0 else torch.zeros_like(tensor)
    # Patch embedding, all RMSNorm scales and learned image-span embeddings replicate.
    return tensor


class DeepseekV41VisionRMSNorm(BaseOP):
    """The official vision RMSNorm keeps its scale in fp32."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.weight = torch.empty(size, dtype=torch.float32)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self._eps)
        return (self.weight * xf).to(dtype)


def _apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


class DeepseekV41VisionPatchEmbed(BaseOP):
    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ) -> None:
        # The released vision tower is BF16, including FP8 text checkpoints.
        self.proj = LinearReplicated(
            3 * vc.patch_size**2,
            vc.hidden_size,
            has_bias=True,
            prefix=f"{prefix}.proj",
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj.forward(patches.flatten(1).to(self.proj.weight.dtype))


class DeepseekV41VisionAttention(BaseOP):
    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.num_heads = div_even(vc.num_heads, get_tp_info().size)
        self.head_dim = vc.hidden_size // vc.num_heads
        self.wqkv = LinearQKVMerged(
            vc.hidden_size,
            self.head_dim,
            vc.num_heads,
            vc.num_heads,
            has_bias=True,
            prefix=f"{prefix}.wqkv",
        )
        self.wo = LinearOProj(
            vc.hidden_size,
            vc.hidden_size,
            has_bias=True,
            prefix=f"{prefix}.wo",
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = (
            self.wqkv.forward(x).view(n, 3, self.num_heads, self.head_dim).unbind(1)
        )
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo.forward(out.transpose(0, 1).reshape(n, -1))


class DeepseekV41VisionMLP(BaseOP):
    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.w1 = LinearColParallelMerged(
            vc.hidden_size,
            [vc.intermediate_size, vc.intermediate_size],
            has_bias=False,
            prefix=f"{prefix}.w1",
        )
        self.w2 = LinearRowParallel(
            vc.intermediate_size,
            vc.hidden_size,
            has_bias=False,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1.forward(x).chunk(2, dim=-1)
        return self.w2.forward(F.silu(gate) * up)


class DeepseekV41VisionBlock(BaseOP):
    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.norm1 = DeepseekV41VisionRMSNorm(vc.hidden_size)
        self.attn = DeepseekV41VisionAttention(
            vc, quant_config=quant_config, prefix=f"{prefix}.attn"
        )
        self.norm2 = DeepseekV41VisionRMSNorm(vc.hidden_size)
        self.mlp = DeepseekV41VisionMLP(
            vc, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin)
        return x + self.mlp.forward(self.norm2.forward(x))


class DeepseekV41VisionModel(BaseOP):
    """One image's patch grid -> per-patch ViT states in ``vision_dim``."""

    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "vision",
    ) -> None:
        self.patch_embed = DeepseekV41VisionPatchEmbed(
            vc, quant_config=quant_config, prefix=f"{prefix}.patch_embed"
        )
        self.blocks = OPList(
            [
                DeepseekV41VisionBlock(
                    vc, quant_config=quant_config, prefix=f"{prefix}.blocks.{layer_id}"
                )
                for layer_id in range(vc.num_layers)
            ]
        )
        self.norm = DeepseekV41VisionRMSNorm(vc.hidden_size)
        self._vc = vc
        self._inv_freq: torch.Tensor | None = None
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        """Keep the patch embed resident and optionally stream the uniform block stack."""
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(
                self.blocks.op_list, self.patch_embed.proj.weight.device
            )
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _blocks(self):
        if self._streamer is None:
            return enumerate(self.blocks.op_list)
        return self._streamer.blocks(self.blocks.op_list)

    def _cos_sin(
        self, n_h: int, n_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = (self._vc.hidden_size // self._vc.num_heads) // 2
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self._vc.rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
        hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
        wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
        freqs = (
            torch.stack((hpos, wpos), dim=-1).reshape(-1, 2, 1).float() * self._inv_freq
        ).flatten(1)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    @torch.inference_mode()
    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        if patches.shape[0] != n_h * n_w:
            raise ValueError(
                f"vision patch rows {patches.shape[0]} do not match grid {n_h}x{n_w}"
            )
        device = self.patch_embed.proj.weight.device
        x = self.patch_embed.forward(patches.to(device))
        cos, sin = self._cos_sin(n_h, n_w, device)
        for _, block in self._blocks():
            x = block.forward(x, cos, sin)
        return self.norm.forward(x)


class DeepseekV41Aligner(BaseOP):
    """Official r x r patch unfold followed by GELU MLP into the text width."""

    def __init__(
        self,
        vc: VisionConfig,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "aligner",
    ) -> None:
        self.downsample_ratio = vc.downsample_ratio
        in_dim = vc.hidden_size * self.downsample_ratio**2
        self.w1 = LinearColParallelMerged(
            in_dim,
            [vc.out_hidden_size],
            has_bias=True,
            prefix=f"{prefix}.w1",
        )
        self.w2 = LinearRowParallel(
            vc.out_hidden_size,
            vc.out_hidden_size,
            has_bias=True,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2.forward(F.gelu(self.w1.forward(x)))


class DeepseekV41VisionMixin:
    """Engine hooks for a model owning the official root-level vision parameters."""

    vision: DeepseekV41VisionModel
    aligner: DeepseekV41Aligner
    image_start: torch.Tensor
    image_newline: torch.Tensor
    image_end: torch.Tensor

    def place_encoder_weights(self, mode: str) -> None:
        self.vision.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        from freetoken.mm.processors.deepseek_v41 import (
            IMAGE,
            IMAGE_END,
            IMAGE_NEW_LINE,
            IMAGE_START,
        )

        patch_states = self.vision.forward(item.feature, item.n_vit_h, item.n_vit_w)
        image_states = self.aligner.forward(patch_states, item.n_vit_h, item.n_vit_w)
        expected = item.n_llm_h * item.n_llm_w
        if image_states.shape[0] != expected:
            raise ValueError(
                f"aligner emitted {image_states.shape[0]} rows, expected {expected} "
                f"for {item.n_llm_h}x{item.n_llm_w}"
            )
        types = item.token_types.to(image_states.device)
        if int((types == IMAGE).sum()) != expected:
            raise ValueError("image token types do not match the aligner grid")
        span = image_states.new_empty((types.numel(), image_states.shape[1]))
        span[types == IMAGE_START] = self.image_start.to(image_states.dtype)
        span[types == IMAGE_END] = self.image_end.to(image_states.dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(image_states.dtype)
        span[types == IMAGE] = image_states
        return span


__all__ = [
    "DeepseekV41Aligner",
    "DeepseekV41VisionMixin",
    "DeepseekV41VisionModel",
    "shard_deepseek_v41_vision_tensor",
]
