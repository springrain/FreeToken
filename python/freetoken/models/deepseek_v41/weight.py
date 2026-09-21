"""Rank-local checkpoint readers for DeepSeek-V4.1-Flash.

The released checkpoint stores the language tower at the safetensors root. Dense FP8
weights use 32x32 E8M0 block scales, routed experts use MXFP4, and the two very large
Engram tables are loaded separately into pinned host banks by ``load_host_tables``.
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache
from freetoken.moe.ownership import ExpertOwnership
from freetoken.utils import download_hf_weight

from .args import load_args
from .config import VisionConfig
from .vision import shard_deepseek_v41_vision_tensor


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict[str, str], device) -> None:
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict[str, str]:
    with open(
        os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8"
    ) as handle:
        return json.load(handle)["weight_map"]


def _dequant_fp8_block(
    weight: torch.Tensor, scale: torch.Tensor, block: int = 32
) -> torch.Tensor:
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    values = torch.exp2(codes - 127.0)
    values = values.repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return (weight.float() * values).to(torch.bfloat16)


def _tp_slice(
    tensor: torch.Tensor,
    dim: int,
    *,
    rank: int,
    world_size: int,
    name: str,
    scale: torch.Tensor | None = None,
    block: int = 32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if world_size == 1:
        return tensor, scale
    total = tensor.shape[dim]
    if total % world_size:
        raise ValueError(
            f"TP shard of {name} needs dim{dim} divisible by tp_size, "
            f"got {total} % {world_size}"
        )
    part = total // world_size
    start = rank * part
    tensor = tensor.narrow(dim, start, part).clone()
    if scale is not None:
        if total % (block * world_size):
            raise ValueError(
                f"TP shard of {name} scale needs {block}-block alignment, "
                f"got {total} % {block * world_size}"
            )
        scale = scale.narrow(dim, start // block, part // block).clone()
    return tensor, scale


def _vocab_shard(
    tensor: torch.Tensor, *, rank: int, world_size: int
) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rows = (tensor.shape[0] + world_size - 1) // world_size
    start = min(rank * rows, tensor.shape[0])
    shard = tensor[start : min(start + rows, tensor.shape[0])].clone()
    if shard.shape[0] < rows:
        shard = torch.cat(
            (shard, tensor.new_zeros((rows - shard.shape[0], *tensor.shape[1:]))),
            dim=0,
        )
    return shard.contiguous()


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    include_vision: bool = True,
    tp_shard: bool = False,
    config=None,
):
    """Yield the resident rank-local state dict; routed experts and Engram rows are separate."""
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4.1 routed experts are served from the offload cache; "
            "run with --moe-strategy offload"
        )
    if not include_non_moe:
        return

    tp = get_tp_info()
    if tp.size > 1 and not tp_shard:
        raise NotImplementedError(
            "DeepSeek-V4.1 runtime TP requires iter_weights(tp_shard=True)"
        )
    args = config.dsv4_args if config is not None else load_args(model_path)
    folder = download_hf_weight(model_path)
    weight_map = _weight_map(folder)
    reader = _ShardReader(folder, weight_map, device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(src: str, dst: str, shard_dim: int | None = None):
        weight = get(f"{src}.weight")
        scale = get(f"{src}.scale") if reader.has(f"{src}.scale") else None
        if shard_dim is not None and tp_shard:
            weight, scale = _tp_slice(
                weight,
                shard_dim,
                rank=tp.rank,
                world_size=tp.size,
                name=dst,
                scale=scale,
            )
        yield f"{dst}.weight", weight
        if scale is not None:
            yield f"{dst}.weight_scale_inv", scale

    try:
        world_size = tp.size if tp_shard else 1
        yield "model.embed.weight", _vocab_shard(
            get("embed.weight"), rank=tp.rank, world_size=world_size
        )
        yield "model.norm.weight", get("norm.weight")
        yield "head.weight", _vocab_shard(
            get("head.weight"), rank=tp.rank, world_size=world_size
        )

        for layer_id in range(args.n_layers):
            layer = f"layers.{layer_id}"
            attn = f"{layer}.attn"
            dst_attn = f"model.{attn}"
            yield from linear(f"{attn}.wq_a", f"{dst_attn}.wq_a")
            yield f"{dst_attn}.q_norm.weight", get(f"{attn}.q_norm.weight")
            yield from linear(f"{attn}.wq_b", f"{dst_attn}.wq_b", 0)
            yield from linear(f"{attn}.wkv", f"{dst_attn}.wkv")
            yield f"{dst_attn}.kv_norm.weight", get(f"{attn}.kv_norm.weight")

            wo_a_weight = get(f"{attn}.wo_a.weight")
            wo_a_scale = get(f"{attn}.wo_a.scale")
            if tp_shard:
                if args.o_groups % tp.size:
                    raise ValueError(
                        f"DeepSeek-V4.1 o_groups={args.o_groups} is not divisible by TP={tp.size}"
                    )
                wo_a_weight, wo_a_scale = _tp_slice(
                    wo_a_weight,
                    0,
                    rank=tp.rank,
                    world_size=tp.size,
                    name=f"{dst_attn}.wo_a",
                    scale=wo_a_scale,
                )
            yield f"{dst_attn}.wo_a", _dequant_fp8_block(
                wo_a_weight, wo_a_scale
            )
            yield from linear(f"{attn}.wo_b", f"{dst_attn}.wo_b", 1)

            sink = get(f"{attn}.attn_sink")
            if tp_shard:
                if args.n_heads % tp.size:
                    raise ValueError(
                        f"DeepSeek-V4.1 n_heads={args.n_heads} is not divisible by TP={tp.size}"
                    )
                local = args.n_heads // tp.size
                sink = sink.narrow(0, tp.rank * local, local).clone()
            yield f"{dst_attn}.attn_sink", sink

            if layer_id in args.kv_source_layers:
                comp = f"{attn}.compressor"
                yield f"model.{comp}.wkv_weight", get(f"{comp}.wkv.weight")
                if reader.has(f"{comp}.wgate.weight"):
                    yield f"model.{comp}.wgate_weight", get(f"{comp}.wgate.weight")
                yield f"model.{comp}.norm.weight", get(f"{comp}.norm.weight")

            if layer_id in args.index_source_layers:
                indexer = f"{attn}.indexer"
                yield from linear(f"{indexer}.wq_b", f"model.{indexer}.wq_b", 0)
                weight = get(f"{indexer}.weights_proj.weight")
                if tp_shard:
                    weight, _ = _tp_slice(
                        weight,
                        0,
                        rank=tp.rank,
                        world_size=tp.size,
                        name=f"model.{indexer}.weights_proj",
                    )
                yield f"model.{indexer}.weights_proj.weight", weight
                if reader.has(f"{indexer}.wk.weight"):
                    yield f"model.{indexer}.wk.weight", get(f"{indexer}.wk.weight")
                    yield f"model.{indexer}.k_norm.weight", get(
                        f"{indexer}.k_norm.weight"
                    )

            yield f"model.{layer}.attn_norm.weight", get(
                f"{layer}.attn_norm.weight"
            )
            yield f"model.{layer}.ffn_norm.weight", get(f"{layer}.ffn_norm.weight")

            gate = f"{layer}.ffn.gate"
            yield f"model.{gate}.weight", get(f"{gate}.weight")
            yield f"model.{gate}.bias", get(f"{gate}.bias")
            if include_vision and args.vision_enabled and reader.has(f"{gate}.bias_vl"):
                yield f"model.{gate}.bias_vl", get(f"{gate}.bias_vl")

            for projection in ("w1", "w3"):
                src = f"{layer}.ffn.shared_experts.{projection}"
                yield from linear(src, f"model.{src}", 0)
            src = f"{layer}.ffn.shared_experts.w2"
            yield from linear(src, f"model.{src}", 1)

            for name in (
                "hc_attn_fn",
                "hc_ffn_fn",
                "hc_attn_base",
                "hc_ffn_base",
                "hc_attn_scale",
                "hc_ffn_scale",
            ):
                yield f"model.{layer}.{name}", get(f"{layer}.{name}")

            if layer_id in args.engram_layer_ids:
                engram = f"{layer}.engram"
                yield f"model.{engram}.q_weight", get(f"{engram}.q_weight")
                yield f"model.{engram}.k_weight", get(f"{engram}.k_weight")
                yield from linear(f"{engram}.wkv", f"model.{engram}.wkv")

        if include_vision and args.vision_enabled:
            vision_config = getattr(config, "vision_config", None)
            if vision_config is None:
                vision_config = VisionConfig(
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
            prefixes = ("vision.", "aligner.")
            for raw_name in tqdm(
                (name for name in weight_map if name.startswith(prefixes)),
                desc="Loading DeepSeek-V4.1 vision weights",
                disable=not tp.is_primary(),
            ):
                tensor = get(raw_name)
                if tp_shard:
                    tensor = shard_deepseek_v41_vision_tensor(
                        raw_name,
                        tensor,
                        config=vision_config,
                        rank=tp.rank,
                        world_size=tp.size,
                    )
                yield raw_name, tensor
            for name in ("image_start", "image_end", "image_newline"):
                yield name, get(name)
    finally:
        reader.close()


def iter_vision_weights(model_path: str, device):
    """Read the complete encoder for tools that operate outside a TP engine."""
    if get_tp_info().size > 1:
        raise NotImplementedError(
            "DeepSeek-V4.1 encoder-only loading is TP1-only; runtime TP uses iter_weights"
        )
    folder = download_hf_weight(model_path)
    reader = _ShardReader(folder, _weight_map(folder), device)
    try:
        for name in reader._weight_map:
            if name.startswith(("vision.", "aligner.")) or name in (
                "image_start",
                "image_end",
                "image_newline",
            ):
                yield name, reader.get(name)
    finally:
        reader.close()


_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|scale)$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}
_KIND_SUFFIX = {"weight": "", "scale": "_scale"}


def iter_expert_pieces(
    model_path: str,
    config,
    kind: QuantKind,
    *,
    parallel: bool | None = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    ownership: ExpertOwnership | None = None,
):
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    args = load_args(model_path)
    num_layers, num_experts = args.n_layers, args.n_routed_experts
    tp = get_tp_info()
    if tp.size > 1 and ownership is None:
        raise NotImplementedError(
            "DeepSeek-V4.1 TP>1 routed experts require owner-local EP"
        )
    if ownership is not None:
        if ownership.global_num_experts != num_experts:
            raise ValueError(
                f"ownership has {ownership.global_num_experts} experts, checkpoint has {num_experts}"
            )
        if ownership.world_size != tp.size or ownership.rank != tp.rank:
            raise ValueError("expert ownership must match the tensor-parallel group")
        expert_start, expert_end = ownership.global_start, ownership.global_end
    else:
        expert_start, expert_end = 0, num_experts

    def locate(raw_name: str):
        match = _EXPERT_RE.match(raw_name)
        if match is None or int(match["layer"]) >= num_layers:
            return None
        expert = int(match["expert"])
        if not expert_start <= expert < expert_end:
            return None
        return (
            int(match["layer"]),
            expert - expert_start,
            _PROJ_ROLE[match["proj"]] + _KIND_SUFFIX[match["kind"]],
        )

    if parallel:
        tensors = iter_expert_tensors_parallel(
            model_path,
            lambda name: locate(name) is not None,
            workers=workers,
            chunk=chunk,
        )
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def serial():
        folder = download_hf_weight(model_path)
        reader = _ShardReader(folder, _weight_map(folder), torch.device("cpu"))
        try:
            for layer_id in tqdm(
                range(num_layers),
                desc="Loading DeepSeek-V4.1 experts",
                disable=not tp.is_primary(),
            ):
                for expert in range(expert_start, expert_end):
                    base = f"layers.{layer_id}.ffn.experts.{expert}"
                    for projection in ("w1", "w3", "w2"):
                        for value_kind in ("weight", "scale"):
                            name = f"{base}.{projection}.{value_kind}"
                            yield name, reader.get(name)
        finally:
            reader.close()

    return per_expert_pieces(serial(), locate, tensors_per_expert=6)


__all__ = ["iter_expert_pieces", "iter_vision_weights", "iter_weights"]
