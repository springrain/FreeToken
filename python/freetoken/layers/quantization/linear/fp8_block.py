"""fp8 e4m3 weight with square 32x32 or 128x128 block scales."""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import FP8_BLOCK_SIZES, QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _e8m0(cfg: LinearConfig) -> bool:
    return cfg.scheme is not None and cfg.scheme.weight.scale == "e8m0"


def _block_shape(cfg: LinearConfig) -> tuple[int, int]:
    group = None if cfg.scheme is None else cfg.scheme.weight.group
    if group is None or len(group) != 2:
        raise ValueError(f"block-fp8 needs a two-dimensional group, got {group}")
    return int(group[0]), int(group[1])


class Dsv4Fp8BlockLinearKernel(LinearKernel):
    """DeepSeek-V4's reference path: activations quantized to fp8 with power-of-two block scales, e8m0 weight scales read as codes."""

    name = "dsv4"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if not _e8m0(cfg):
            return "serves e8m0 block scales only"
        block_n, block_k = _block_shape(cfg)
        if block_n != block_k or block_n not in FP8_BLOCK_SIZES:
            return (
                "serves square block-fp8 groups with size "
                f"{sorted(FP8_BLOCK_SIZES)}, got {(block_n, block_k)}"
            )
        return None

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear

        # Kernels are selected with a LinearConfig but are intentionally stateless;
        # recover the block geometry from the rank-local tensors at execution time.
        n, k = layer.weight.shape
        scale_n, scale_k = layer.weight_scale_inv.shape
        block_n, block_k = n // scale_n, k // scale_k
        assert block_n == block_k
        return block_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale_inv,
            layer.bias,
            block_size=block_k,
        )


class TritonFp8BlockLinearKernel(LinearKernel):
    """W8A16 GEMV at M=1, dynamic 1x128 W8A8 GEMM above."""

    name = "triton"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if _e8m0(cfg):
            return "reads float block scales; e8m0 codes go to the dsv4 kernel"
        return (
            None
            if _block_shape(cfg) == (128, 128)
            else "the generic triton kernel serves 128x128 block scales only"
        )

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


@register_method(QuantKind.FP8_BLOCK, LayerKind.LINEAR)
class Fp8BlockLinearMethod(LinearMethod):
    candidates = (Dsv4Fp8BlockLinearKernel, TritonFp8BlockLinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        block_n, block_k = _block_shape(g)
        if block_n != block_k or block_n not in FP8_BLOCK_SIZES:
            raise ValueError(
                f"block-fp8 supports square block sizes {sorted(FP8_BLOCK_SIZES)}, "
                f"got {(block_n, block_k)}"
            )
        if g.in_features % block_k or any(o % block_n for o in g.output_sizes):
            raise ValueError(
                f"block-fp8 needs K divisible by {block_k} and every fused N segment "
                f"divisible by {block_n}, got K={g.in_features} N={g.output_sizes}"
            )
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=FP8)
        # e8m0 codes stay codes for the dsv4 kernel; float scales are bf16 as the readers push them today
        scale_dtype = E8M0 if _e8m0(g) else torch.bfloat16
        layer.weight_scale_inv = torch.empty(
            g.out_features // block_n,
            g.in_features // block_k,
            dtype=scale_dtype,
        )
