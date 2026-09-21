from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import DistributedCommunicator
from freetoken.message import MMItem
from freetoken.mm.processors.deepseek_v41 import image_token_types
from freetoken.models.deepseek_v41.config import VisionConfig
from freetoken.models.deepseek_v41.vision import (
    DeepseekV41Aligner,
    DeepseekV41VisionAttention,
    DeepseekV41VisionMLP,
    DeepseekV41VisionMixin,
    DeepseekV41VisionModel,
    shard_deepseek_v41_vision_tensor,
)


def _vc(num_layers: int = 1) -> VisionConfig:
    return VisionConfig(
        num_layers=num_layers,
        hidden_size=64,
        num_heads=8,
        intermediate_size=96,
        patch_size=2,
        rope_theta=10000.0,
        downsample_ratio=2,
        max_image_tokens=64,
        min_pixels=0,
        max_wh_ratio=None,
        out_hidden_size=64,
        image_token_id=129264,
    )


def _fill(op, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    for name, value in op.state_dict().items():
        if "norm" in name:
            value.copy_(1 + 0.05 * torch.randn(value.shape, generator=gen))
        else:
            value.copy_(0.05 * torch.randn(value.shape, generator=gen))


def _rms(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(
        x.dtype
    )


def _rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


def test_tiny_tower_and_aligner_match_the_official_torch_graph(monkeypatch):
    import freetoken.distributed.info as info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    vc = _vc()
    vision = DeepseekV41VisionModel(vc)
    aligner = DeepseekV41Aligner(vc)
    _fill(vision, 1)
    _fill(aligner, 2)
    patches = torch.randn(6, 3, 2, 2, generator=torch.Generator().manual_seed(3))

    state = vision.state_dict()
    x = F.linear(
        patches.flatten(1),
        state["patch_embed.proj.weight"],
        state["patch_embed.proj.bias"],
    )
    dim = (vc.hidden_size // vc.num_heads) // 2
    inv = 1.0 / (vc.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    hp = torch.arange(2).unsqueeze(1).expand(2, 3)
    wp = torch.arange(3).unsqueeze(0).expand(2, 3)
    freqs = (torch.stack((hp, wp), -1).reshape(-1, 2, 1).float() * inv).flatten(1)
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)
    norm = _rms(x, state["blocks.0.norm1.weight"])
    q, k, v = (
        F.linear(
            norm, state["blocks.0.attn.wqkv.weight"], state["blocks.0.attn.wqkv.bias"]
        )
        .view(6, 3, vc.num_heads, -1)
        .unbind(1)
    )
    q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
    attn = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    )
    x = x + F.linear(
        attn.transpose(0, 1).reshape(6, -1),
        state["blocks.0.attn.wo.weight"],
        state["blocks.0.attn.wo.bias"],
    )
    norm = _rms(x, state["blocks.0.norm2.weight"])
    gate, up = F.linear(norm, state["blocks.0.mlp.w1.weight"]).chunk(2, -1)
    x = x + F.linear(F.silu(gate) * up, state["blocks.0.mlp.w2.weight"])
    x = _rms(x, state["norm.weight"])
    astate = aligner.state_dict()
    unfolded = F.unfold(
        F.pad(x.view(2, 3, -1).permute(2, 0, 1), (0, 1, 0, 0)).unsqueeze(0), 2, stride=2
    )
    unfolded = unfolded.squeeze(0).transpose(0, 1)
    expected = F.linear(
        F.gelu(F.linear(unfolded, astate["w1.weight"], astate["w1.bias"])),
        astate["w2.weight"],
        astate["w2.bias"],
    )

    got = aligner.forward(vision.forward(patches, 2, 3), 2, 3)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tp_shapes_match_the_vision_partition_contract(monkeypatch, world_size):
    import freetoken.distributed.info as info

    monkeypatch.setattr(
        info, "_TP_INFO", info.DistributedInfo(rank=world_size - 1, size=world_size)
    )
    vc = _vc()
    vision = DeepseekV41VisionModel(vc)
    aligner = DeepseekV41Aligner(vc)
    state = vision.state_dict()
    assert state["patch_embed.proj.weight"].shape == (64, 12)
    assert state["blocks.0.attn.wqkv.weight"].shape == (3 * 64 // world_size, 64)
    assert state["blocks.0.attn.wqkv.bias"].shape == (3 * 64 // world_size,)
    assert state["blocks.0.attn.wo.weight"].shape == (64, 64 // world_size)
    assert state["blocks.0.attn.wo.bias"].shape == (64,)
    assert state["blocks.0.mlp.w1.weight"].shape == (2 * 96 // world_size, 64)
    assert state["blocks.0.mlp.w2.weight"].shape == (64, 96 // world_size)
    assert state["blocks.0.norm1.weight"].dtype is torch.float32
    astate = aligner.state_dict()
    assert astate["w1.weight"].shape == (64 // world_size, 4 * 64)
    assert astate["w1.bias"].shape == (64 // world_size,)
    assert astate["w2.weight"].shape == (64, 64 // world_size)
    assert astate["w2.bias"].shape == (64,)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_raw_vision_sharder_preserves_fused_groups_and_bias_once(world_size):
    vc = _vc()
    tensors = {
        "vision.patch_embed.proj.weight": torch.randn(64, 12),
        "vision.blocks.0.norm1.weight": torch.randn(64),
        "vision.blocks.0.attn.wqkv.weight": torch.randn(192, 64),
        "vision.blocks.0.attn.wqkv.bias": torch.randn(192),
        "vision.blocks.0.attn.wo.weight": torch.randn(64, 64),
        "vision.blocks.0.attn.wo.bias": torch.randn(64),
        "vision.blocks.0.mlp.w1.weight": torch.randn(192, 64),
        "vision.blocks.0.mlp.w2.weight": torch.randn(64, 96),
        "aligner.w1.weight": torch.randn(64, 256),
        "aligner.w1.bias": torch.randn(64),
        "aligner.w2.weight": torch.randn(64, 64),
        "aligner.w2.bias": torch.randn(64),
        "image_start": torch.randn(64),
    }
    shards = {
        name: [
            shard_deepseek_v41_vision_tensor(
                name, tensor, config=vc, rank=rank, world_size=world_size
            )
            for rank in range(world_size)
        ]
        for name, tensor in tensors.items()
    }
    for name in (
        "vision.patch_embed.proj.weight",
        "vision.blocks.0.norm1.weight",
        "image_start",
    ):
        assert all(torch.equal(shard, tensors[name]) for shard in shards[name])
    for name, group_rows in (
        ("vision.blocks.0.attn.wqkv.weight", (64, 64, 64)),
        ("vision.blocks.0.attn.wqkv.bias", (64, 64, 64)),
        ("vision.blocks.0.mlp.w1.weight", (96, 96)),
    ):
        groups = tensors[name].split(group_rows, dim=0)
        local_rows = [size // world_size for size in group_rows]
        local = [shard.split(local_rows, dim=0) for shard in shards[name]]
        rebuilt = torch.cat(
            [
                torch.cat([local[rank][group] for rank in range(world_size)])
                for group in range(len(groups))
            ]
        )
        assert torch.equal(rebuilt, tensors[name])
    for name in (
        "vision.blocks.0.attn.wo.weight",
        "vision.blocks.0.mlp.w2.weight",
        "aligner.w2.weight",
    ):
        assert torch.equal(torch.cat(shards[name], dim=1), tensors[name])
    for name in ("aligner.w1.weight", "aligner.w1.bias"):
        assert torch.equal(torch.cat(shards[name], dim=0), tensors[name])
    for name in ("vision.blocks.0.attn.wo.bias", "aligner.w2.bias"):
        assert torch.equal(shards[name][0], tensors[name])
        assert all(torch.count_nonzero(shard) == 0 for shard in shards[name][1:])


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_tp_attention_mlp_and_aligner_partials_sum_to_tp1(monkeypatch, world_size):
    import freetoken.distributed.info as info

    vc = _vc()
    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    attention = DeepseekV41VisionAttention(vc)
    mlp = DeepseekV41VisionMLP(vc)
    aligner = DeepseekV41Aligner(vc)
    _fill(attention, 4)
    _fill(mlp, 5)
    _fill(aligner, 6)
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(7))
    rope_width = (vc.hidden_size // vc.num_heads) // 2
    cos = torch.randn(
        4, 1, rope_width, generator=torch.Generator().manual_seed(8)
    ).cos()
    sin = torch.randn(
        4, 1, rope_width, generator=torch.Generator().manual_seed(9)
    ).sin()
    attn_ref = attention.forward(x, cos, sin)
    mlp_ref = mlp.forward(x)
    align_input = torch.randn(6, 64, generator=torch.Generator().manual_seed(10))
    align_ref = aligner.forward(align_input, 2, 3)
    attn_state, mlp_state, align_state = (
        {k: v.clone() for k, v in op.state_dict().items()}
        for op in (attention, mlp, aligner)
    )
    monkeypatch.setattr(
        DistributedCommunicator, "all_reduce", lambda _self, value: value
    )

    attn_parts, mlp_parts, align_parts = [], [], []
    for rank in range(world_size):
        monkeypatch.setattr(
            info, "_TP_INFO", info.DistributedInfo(rank=rank, size=world_size)
        )
        local_attn = DeepseekV41VisionAttention(vc)
        local_attn.load_state_dict(
            {
                key: shard_deepseek_v41_vision_tensor(
                    "vision.blocks.0.attn." + key,
                    value,
                    config=vc,
                    rank=rank,
                    world_size=world_size,
                )
                for key, value in attn_state.items()
            }
        )
        attn_parts.append(local_attn.forward(x, cos, sin))

        local_mlp = DeepseekV41VisionMLP(vc)
        local_mlp.load_state_dict(
            {
                key: shard_deepseek_v41_vision_tensor(
                    "vision.blocks.0.mlp." + key,
                    value,
                    config=vc,
                    rank=rank,
                    world_size=world_size,
                )
                for key, value in mlp_state.items()
            }
        )
        mlp_parts.append(local_mlp.forward(x))

        local_aligner = DeepseekV41Aligner(vc)
        local_aligner.load_state_dict(
            {
                key: shard_deepseek_v41_vision_tensor(
                    "aligner." + key,
                    value,
                    config=vc,
                    rank=rank,
                    world_size=world_size,
                )
                for key, value in align_state.items()
            }
        )
        align_parts.append(local_aligner.forward(align_input, 2, 3))

    torch.testing.assert_close(sum(attn_parts), attn_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sum(mlp_parts), mlp_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sum(align_parts), align_ref, rtol=1e-5, atol=1e-5)


def test_mixin_returns_start_rows_newlines_and_end(monkeypatch):
    class _Vision:
        def forward(self, feature, n_h, n_w):
            return torch.zeros(n_h * n_w, 3)

    class _Aligner:
        def forward(self, states, n_h, n_w):
            return torch.arange(4 * 3, dtype=torch.float32).view(4, 3)

    class _Host(DeepseekV41VisionMixin):
        vision = _Vision()
        aligner = _Aligner()
        image_start = torch.tensor([100.0, 101.0, 102.0])
        image_newline = torch.tensor([200.0, 201.0, 202.0])
        image_end = torch.tensor([300.0, 301.0, 302.0])

    types = image_token_types(2, 2)
    item = MMItem(
        modality="image",
        hash=1,
        pad_value=1,
        offsets=[[0, types.numel()]],
        feature=torch.zeros(4, 3, 2, 2),
        model_specific_data={
            "n_vit_h": 2,
            "n_vit_w": 2,
            "n_llm_h": 2,
            "n_llm_w": 2,
            "token_types": types,
        },
    )
    got = _Host().encode(item)
    assert got.shape == (8, 3)
    assert torch.equal(got[0], _Host.image_start)
    assert torch.equal(got[3], _Host.image_newline)
    assert torch.equal(got[6], _Host.image_newline)
    assert torch.equal(got[7], _Host.image_end)
    assert torch.equal(got[[1, 2, 4, 5]], torch.arange(12).view(4, 3).float())
