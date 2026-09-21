from __future__ import annotations

import io
from types import SimpleNamespace

import torch
from PIL import Image

from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processors.deepseek_v41 import (
    DeepseekV41MMProcessor,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    image_token_types,
    num_image_tokens,
    plan_image_grid,
)


def _config():
    return SimpleNamespace(
        image_token_id=129264,
        vision_config=SimpleNamespace(
            patch_size=2,
            downsample_ratio=2,
            max_image_tokens=20,
            min_pixels=0,
            max_wh_ratio=None,
        ),
    )


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 127, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_grid_plan_and_token_types_follow_the_official_layout():
    # 5x3 -> pad to 6x4 -> 2x3 ViT patches -> ceil /2 = 1x2 LLM cells.
    assert plan_image_grid(
        5,
        3,
        patch_size=2,
        downsample_ratio=2,
        max_tokens=20,
        min_pixels=0,
        max_wh_ratio=None,
    ) == (1, 2, 4, 6)
    types = image_token_types(2, 3)
    assert types.tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]
    assert types.numel() == num_image_tokens(2, 3) == 10


def test_processor_builds_patches_and_replaces_the_whole_learned_span():
    proc = DeepseekV41MMProcessor(_config(), "/unused", MultimodalConfig())
    result = proc.apply(torch.tensor([7, 129264, 8], dtype=torch.int32), [_png(5, 3)])
    (item,) = result.mm_items
    assert item.feature.shape == (6, 3, 2, 2)
    assert item.feature.dtype is torch.bfloat16
    assert (item.n_vit_h, item.n_vit_w, item.n_llm_h, item.n_llm_w) == (2, 3, 1, 2)
    assert item.token_types.tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]
    assert item.offsets == [[1, 6]] and item.num_tokens == 5
    assert result.input_ids.tolist() == [7, *[item.pad_value] * 5, 8]
    assert result.mrope_positions is None and result.mrope_delta == 0


def test_token_budget_and_dummy_item_use_complete_span_rows():
    proc = DeepseekV41MMProcessor(
        _config(), "/unused", MultimodalConfig(image_max_tokens=6)
    )
    (item,) = proc.process([Image.new("RGB", (100, 100))])
    assert item.token_types.numel() <= 6
    assert item.feature.shape[0] == item.n_vit_h * item.n_vit_w

    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 3, 2, 2)
    assert dummy.token_types.tolist() == [
        IMAGE_START,
        IMAGE,
        IMAGE_NEW_LINE,
        IMAGE_END,
    ]
    assert dummy.offsets == [[0, 4]] and dummy.num_tokens == 4


def test_processor_kwargs_override_the_official_resize_limits():
    proc = DeepseekV41MMProcessor(
        _config(),
        "/unused",
        MultimodalConfig(
            image_min_tokens=3,
            image_max_tokens=12,
            processor_kwargs={"max_wh_ratio": 4},
        ),
    )
    assert proc.get_mm_processor_kwargs(proc.mm) == {
        "max_tokens": 12,
        "min_pixels": 3 * (2 * 2) ** 2,
        "max_wh_ratio": 4,
    }
