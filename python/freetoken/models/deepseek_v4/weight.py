"""Weight loading for DeepSeek-V4 Flash/Pro (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed by the model's
    attribute paths (``model.`` + the checkpoint name). ``wo_a`` dequantized to bf16 to match
    the reference bf16 einsum.
  - :func:`iter_expert_pieces` streams the routed MXFP4 experts (e2m1 pairs + e8m0 per-32
    scales, no global) as per-expert pieces for the expert quant method's banks.
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch
from tqdm import tqdm

from freetoken.layers.quantization import QuantKind
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache
from freetoken.moe.ownership import ExpertOwnership
from freetoken.utils import download_hf_weight, init_logger

from .args import load_args


logger = init_logger(__name__)


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
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


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Dequantize 128x128 block-scaled FP8 (e4m3) to bf16.

    scale is e8m0 exponent codes, ``value = 2^(code-127)`` (Triton FP8 GEMM convention).
    Used for ``wo_a`` to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _tp_slice(
    tensor: torch.Tensor,
    dim: int,
    *,
    rank: int,
    world_size: int,
    name: str,
    scale: torch.Tensor | None = None,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Slice a dense weight and its block scale along one tensor-parallel axis."""
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
        scale_part = part // block
        scale = scale.narrow(dim, start // block, scale_part).clone()
    return tensor, scale


def _vocab_shard(tensor: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    """Match VocabParallelEmbedding/ParallelLMHead's ceil-divided row layout."""
    if world_size == 1:
        return tensor
    rows = (tensor.shape[0] + world_size - 1) // world_size
    start = min(rank * rows, tensor.shape[0])
    shard = tensor[start : min(start + rows, tensor.shape[0])].clone()
    if shard.shape[0] < rows:
        padding = tensor.new_zeros((rows - shard.shape[0], *tensor.shape[1:]))
        shard = torch.cat((shard, padding), dim=0)
    return shard.contiguous()


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
    tp_shard: bool = False,
    config=None,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed MXFP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4 only runs ``--moe-strategy offload``). Tensors yielded in checkpoint
    dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    tp = get_tp_info()
    if tp.size > 1 and not tp_shard:
        raise NotImplementedError(
            "DeepSeek-V4 runtime TP requires iter_weights(tp_shard=True); "
            "the default reader path is TP1-only"
        )

    args = load_args(model_path, max_batch_size=1)
    folder = download_hf_weight(model_path)
    logger.debug(
        "[DSV4_TRACE] weights.resident.begin folder=%s layers=%d tp=%d/%d shard=%s",
        folder,
        args.n_layers,
        tp.rank,
        tp.size,
        tp_shard,
    )
    reader = _ShardReader(folder, _weight_map(folder), device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(src: str, dst: str, shard_dim: int | None = None):
        weight = get(f"{src}.weight")
        # fp8 linears declare the e8m0 block scale under the quant method's role name
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
        yield "model.embed.weight", _vocab_shard(
            get("embed.weight"), rank=tp.rank, world_size=tp.size if tp_shard else 1
        )
        yield "model.norm.weight", get("norm.weight")
        yield "model.head.weight", _vocab_shard(
            get("head.weight"), rank=tp.rank, world_size=tp.size if tp_shard else 1
        )
        for nm in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            yield f"model.{nm}", get(nm)

        for L in range(args.n_layers):
            logger.debug("[DSV4_TRACE] weights.resident.layer.begin layer=%d/%d", L, args.n_layers)
            a = f"layers.{L}.attn"
            m = f"model.{a}"
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", f"{m}.wq_b", shard_dim=0)
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a is grouped by output heads. Slice its FP8 rows and scale rows first so
            # every rank dequantizes only its local groups (important for V4-Pro's matrix).
            wo_a_weight = get(f"{a}.wo_a.weight")
            wo_a_scale = get(f"{a}.wo_a.scale")
            if tp_shard and tp.size > 1:
                if args.o_groups % tp.size:
                    raise ValueError(
                        "DeepSeek-V4 TP needs o_groups divisible by tp_size, "
                        f"got {args.o_groups} % {tp.size}"
                    )
                wo_a_weight, wo_a_scale = _tp_slice(
                    wo_a_weight,
                    0,
                    rank=tp.rank,
                    world_size=tp.size,
                    name=f"{m}.wo_a",
                    scale=wo_a_scale,
                )
            yield f"{m}.wo_a", _dequant_fp8_block(wo_a_weight, wo_a_scale)
            yield from linear(f"{a}.wo_b", f"{m}.wo_b", shard_dim=1)
            sink = get(f"{a}.attn_sink")
            if tp_shard and tp.size > 1:
                if args.n_heads % tp.size:
                    raise ValueError(
                        "DeepSeek-V4 TP needs n_heads divisible by tp_size, "
                        f"got {args.n_heads} % {tp.size}"
                    )
                heads = args.n_heads // tp.size
                sink = sink.narrow(0, tp.rank * heads, heads).clone()
            yield f"{m}.attn_sink", sink

            ratio = args.compress_ratios[L]
            if ratio:
                c = f"{a}.compressor"
                for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                    yield f"model.{c}.{nm}", get(f"{c}.{nm}")
                if ratio == 4:
                    idx = f"{a}.indexer"
                    yield from linear(f"{idx}.wq_b", f"model.{idx}.wq_b")
                    yield f"model.{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                    ic = f"{idx}.compressor"
                    for nm in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                        yield f"model.{ic}.{nm}", get(f"{ic}.{nm}")

            yield f"model.layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"model.layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"model.{g}.weight", get(f"{g}.weight")
            if L < args.n_hash_layers:
                yield f"model.{g}.tid2eid", get(f"{g}.tid2eid")
            else:
                yield f"model.{g}.bias", get(f"{g}.bias")
            for proj in ("w1", "w3"):
                src = f"layers.{L}.ffn.shared_experts.{proj}"
                yield from linear(src, f"model.{src}", shard_dim=0)
            src = f"layers.{L}.ffn.shared_experts.w2"
            yield from linear(src, f"model.{src}", shard_dim=1)

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"model.layers.{L}.{nm}", get(f"layers.{L}.{nm}")
            logger.debug("[DSV4_TRACE] weights.resident.layer.done layer=%d/%d", L, args.n_layers)
    finally:
        reader.close()
        logger.debug("[DSV4_TRACE] weights.resident.done")


# --------------------------------------------------------------------------------------
# Routed MXFP4 expert pieces.
# --------------------------------------------------------------------------------------
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
    """Routed experts, one piece per expert: ``{gate, up, down}`` e2m1 pairs and their e8m0
    ``_scale`` companions (``w1`` / ``w3`` / ``w2``). The MTP layer's experts are skipped."""
    if kind is not QuantKind.MXFP4:
        return None
    from freetoken.models.weight import iter_expert_tensors_parallel
    from freetoken.moe.expert_pieces import per_expert_pieces

    args = load_args(model_path, max_batch_size=1)
    L, E = args.n_layers, args.n_routed_experts
    tp = get_tp_info()
    if tp.size > 1 and ownership is None:
        raise NotImplementedError(
            "DeepSeek-V4 TP>1 routed experts require owner-local EP; "
            "set --moe-ep-size to the tensor-parallel group size"
        )
    if ownership is not None:
        if ownership.global_num_experts != E:
            raise ValueError(
                f"expert ownership has global_num_experts={ownership.global_num_experts}, "
                f"but DeepSeek-V4 has {E}"
            )
        if ownership.world_size != tp.size or ownership.rank != tp.rank:
            raise ValueError(
                "expert ownership must match the active tensor-parallel group: "
                f"ownership=({ownership.rank}, {ownership.world_size}), "
                f"tp=({tp.rank}, {tp.size})"
            )
        expert_start = ownership.global_start
        expert_end = ownership.global_end
    else:
        expert_start, expert_end = 0, E

    logger.debug(
        "[DSV4_TRACE] weights.experts.begin model=%s layers=%d experts=%d range=%d:%d "
        "parallel=%s tp=%d/%d",
        model_path,
        L,
        E,
        expert_start,
        expert_end,
        parallel,
        tp.rank,
        tp.size,
    )

    def locate(raw_name: str):
        m = _EXPERT_RE.match(raw_name)
        if m is None or int(m["layer"]) >= L:
            return None
        expert = int(m["expert"])
        if not expert_start <= expert < expert_end:
            return None
        return (
            int(m["layer"]),
            expert - expert_start,
            _PROJ_ROLE[m["proj"]] + _KIND_SUFFIX[m["kind"]],
        )

    if parallel:
        tensors = iter_expert_tensors_parallel(model_path, lambda n: locate(n) is not None, workers=workers, chunk=chunk)
        return per_expert_pieces(tensors, locate, tensors_per_expert=6)

    def _serial():
        folder = download_hf_weight(model_path)
        reader = _ShardReader(folder, _weight_map(folder), torch.device("cpu"))
        try:
            for li in tqdm(range(L), desc="Loading DSV4 experts (serial)", disable=not get_tp_info().is_primary()):
                logger.debug(
                    "[DSV4_TRACE] weights.experts.layer.begin layer=%d/%d range=%d:%d",
                    li,
                    L,
                    expert_start,
                    expert_end,
                )
                for e in range(expert_start, expert_end):
                    base = f"layers.{li}.ffn.experts.{e}"
                    for proj in ("w1", "w3", "w2"):
                        for kind_ in ("weight", "scale"):
                            name = f"{base}.{proj}.{kind_}"
                            yield name, reader.get(name)
                logger.debug("[DSV4_TRACE] weights.experts.layer.done layer=%d/%d", li, L)
        finally:
            reader.close()
            logger.debug("[DSV4_TRACE] weights.experts.serial.done")

    return per_expert_pieces(_serial(), locate, tensors_per_expert=6)


__all__ = ["iter_weights", "iter_expert_pieces"]
