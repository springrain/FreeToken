"""DeepSeek-V4.1 image preprocessing and prompt-span expansion."""

from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np
import torch
from PIL import ImageOps

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(
    height: int, width: int, patch_size: int, downsample_ratio: int
) -> tuple[int, int]:
    return (
        math.ceil((height // patch_size) / downsample_ratio),
        math.ceil((width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_tokens: int,
) -> tuple[int, int]:
    ratio = height / width
    max_w = math.sqrt((max_tokens - 2) / ratio + 0.25) - 0.5
    max_h = max_w * ratio
    cell = patch_size * downsample_ratio
    if max_w < 1.0:
        return (max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_tokens - 3) * cell
    beta = min(
        math.floor(max_w) * cell / width,
        math.floor(max_h) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_tokens: int,
) -> tuple[int, int, int, int]:
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_tokens:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_tokens
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        if num_image_tokens(n_llm_h, n_llm_w) > max_tokens:
            raise ValueError("failed to fit the image into the configured token budget")
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(
    width: int,
    height: int,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_tokens: int,
    min_pixels: int,
    max_wh_ratio: int | None,
) -> tuple[int, int, int, int]:
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    return safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
        max_tokens,
    )


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV41MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = int(hf_config.image_token_id)
        self.placeholder = [self.image_token_id]
        self.patch_size = int(vc.patch_size)
        self.downsample_ratio = int(vc.downsample_ratio)
        self.default_max_tokens = int(vc.max_image_tokens)
        self.min_pixels = int(vc.min_pixels)
        self.max_wh_ratio = getattr(vc, "max_wh_ratio", None)

    def get_mm_processor_kwargs(self, mm: MultimodalConfig) -> dict[str, Any]:
        max_tokens = (
            self.default_max_tokens
            if mm.image_max_tokens is None
            else int(mm.image_max_tokens)
        )
        min_pixels = self.min_pixels
        if mm.image_min_tokens is not None:
            cell = self.patch_size * self.downsample_ratio
            min_pixels = int(mm.image_min_tokens) * cell * cell
        return {
            "max_tokens": max_tokens,
            "min_pixels": min_pixels,
            **mm.processor_kwargs,
        }

    def _process_one(self, image: Any, kwargs: dict[str, Any]) -> MMItem:
        image = image.convert("RGB")
        max_tokens = int(kwargs.pop("max_tokens"))
        min_pixels = int(kwargs.pop("min_pixels"))
        max_wh_ratio = kwargs.pop("max_wh_ratio", self.max_wh_ratio)
        if kwargs:
            raise TypeError(
                "unsupported DeepSeek-V4.1 image processor kwargs: "
                + ", ".join(sorted(kwargs))
            )
        n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
            image.width,
            image.height,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_tokens=max_tokens,
            min_pixels=min_pixels,
            max_wh_ratio=max_wh_ratio,
        )
        n_vit_h = best_height // self.patch_size
        n_vit_w = best_width // self.patch_size
        if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(
                image, (best_width, best_height), color=(127, 127, 127)
            )
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        pixels = pixels.permute(2, 0, 1) / 255
        pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
        patches = (
            pixels.reshape(
                3,
                n_vit_h,
                self.patch_size,
                n_vit_w,
                self.patch_size,
            )
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, self.patch_size, self.patch_size)
            .contiguous()
        )
        token_types = image_token_types(n_llm_h, n_llm_w)
        extra = struct.pack("<4i", n_vit_h, n_vit_w, n_llm_h, n_llm_w)
        item_hash = content_hash(patches, extra + token_types.numpy().tobytes())
        return MMItem(
            modality="image",
            hash=item_hash,
            pad_value=mm_pad_value(item_hash),
            offsets=[],
            feature=patches,
            model_specific_data={
                "n_vit_h": n_vit_h,
                "n_vit_w": n_vit_w,
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "token_types": token_types,
            },
        )

    def process(self, images: list[Any]) -> list[MMItem]:
        base = self.get_mm_processor_kwargs(self.mm)
        return [self._process_one(image, dict(base)) for image in images]

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        # Every span row is overwritten by the encoder: image features or one of the
        # learned start/newline/end embeddings.
        return PromptReplacement(
            [self.image_token_id] * item.token_types.numel(),
            [True] * item.token_types.numel(),
        )

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        ratio = self.downsample_ratio
        token_types = image_token_types(1, 1).to(device)
        patch_shape = (3, self.patch_size, self.patch_size)
        return [
            MMItem(
                modality="image",
                hash=0,
                pad_value=0,
                offsets=[[0, token_types.numel()]],
                feature=torch.zeros(
                    ratio * ratio, *patch_shape, dtype=dtype, device=device
                ),
                model_specific_data={
                    "n_vit_h": ratio,
                    "n_vit_w": ratio,
                    "n_llm_h": 1,
                    "n_llm_w": 1,
                    "token_types": token_types,
                },
            )
        ]


__all__ = [
    "DeepseekV41MMProcessor",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_START",
    "TEXT",
    "image_token_types",
    "llm_grid",
    "num_image_tokens",
    "plan_image_grid",
    "safe_resize",
    "solve_resize_ratio",
]
