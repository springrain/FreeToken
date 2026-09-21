"""``_validate_owner_ep_config`` must reject what the owner runtime cannot honour.

Two settings used to be accepted and then silently ignored:

* ``--moe-cpu-layers`` -- ``_decode_owner`` is selected before the ``is_cpu_layer``
  branch, and ``OwnerOffloadMoeCache`` hard-codes a GPU inner cache, so the flag
  had no effect at all.
* FTW checkpoints -- ``load_ftw_banks`` rebuilds ``[num_experts, ...]`` GLOBAL
  expert rows with no ownership filter, so the banks cannot bind to the
  owner-local geometry.

Fail-fast is the contract for owner EP (``moe_ep_size > 1`` is opt-in and
validated before any allocation), so both must raise rather than serve a
different configuration than the one requested.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from freetoken.engine.engine import (
    _resolve_owner_ep_defaults,
    _validate_owner_ep_config,
)


def _config(**over):
    base = dict(
        moe_ep_size=2,
        tp_info=SimpleNamespace(size=2, rank=0),
        moe_strategy="offload",
        moe_cache_rate=None,
        moe_cache_size=4096,
        moe_cache_auto=False,
        moe_cpu_layers=None,
        model_path="/models/unit-model",
        model_config=SimpleNamespace(
            model_type="qwen4_exp",
            expert_quant="nvfp4",
            owner_ep_expert_quants=("nvfp4",),
            num_experts=512,
        ),
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _not_ftw(monkeypatch):
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)


@pytest.mark.parametrize("size", [2, 3, 4, 5, 6, 8, 10, 12, 16])
def test_owner_topology_accepts_any_divisible_group_size(size):
    _validate_owner_ep_config(
        _config(
            moe_ep_size=size,
            tp_info=SimpleNamespace(size=size, rank=0),
            model_config=SimpleNamespace(
                model_type="unit_owner_model",
                expert_quant="test_format",
                owner_ep_expert_quants=("test_format",),
                num_experts=480,
            ),
        )
    )


def test_moe_cache_auto_satisfies_the_cache_requirement():
    _validate_owner_ep_config(_config(moe_cache_size=0, moe_cache_auto=True))


def test_owner_ep_auto_resolves_to_offload_and_local_cache_auto():
    config = _config(
        moe_strategy="auto", moe_cache_size=0, moe_cache_auto=False
    )
    _resolve_owner_ep_defaults(config)
    assert config.moe_strategy == "offload"
    assert config.moe_cache_auto is True
    _validate_owner_ep_config(config)


def test_owner_ep_auto_preserves_an_explicit_cache_size():
    config = _config(moe_strategy="auto", moe_cache_size=1024)
    _resolve_owner_ep_defaults(config)
    assert config.moe_strategy == "offload"
    assert config.moe_cache_size == 1024
    assert config.moe_cache_auto is False


def test_ep_size_one_is_a_no_op(monkeypatch):
    monkeypatch.setattr(
        "freetoken.checkpoint.ftw.is_ftw_checkpoint",
        lambda path: pytest.fail("must not probe the checkpoint when EP is off"),
    )
    _validate_owner_ep_config(_config(moe_ep_size=1))


@pytest.mark.parametrize("layers", ["0", "0,1", "0-3"])
def test_cpu_layers_are_rejected_instead_of_silently_ignored(layers):
    with pytest.raises(ValueError, match="moe-cpu-layers"):
        _validate_owner_ep_config(_config(moe_cpu_layers=layers))


def test_ftw_checkpoints_are_rejected(monkeypatch):
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: True)
    with pytest.raises(ValueError, match="FTW"):
        _validate_owner_ep_config(_config())


def test_an_explicit_cache_size_is_still_required_without_auto():
    with pytest.raises(ValueError, match="moe-cache-size"):
        _validate_owner_ep_config(_config(moe_cache_size=0))


def test_moe_cache_rate_is_still_rejected():
    with pytest.raises(ValueError, match="moe-cache-rate"):
        _validate_owner_ep_config(_config(moe_cache_rate=0.5))


def test_a_non_offload_strategy_is_still_rejected():
    with pytest.raises(ValueError, match="offload"):
        _validate_owner_ep_config(_config(moe_strategy="resident"))


def test_unsupported_model_format_is_rejected_before_allocation():
    with pytest.raises(NotImplementedError, match="does not implement"):
        _validate_owner_ep_config(
            _config(
                model_config=SimpleNamespace(
                    model_type="other",
                    expert_quant="nvfp4",
                    owner_ep_expert_quants=(),
                    num_experts=512,
                )
            )
        )


def test_expert_count_must_be_divisible_by_owner_group():
    with pytest.raises(ValueError, match="num_experts divisible"):
        _validate_owner_ep_config(
            _config(
                moe_ep_size=8,
                tp_info=SimpleNamespace(size=8, rank=0),
                model_config=SimpleNamespace(
                    model_type="deepseek_v41",
                    expert_quant="ds_fp4",
                    owner_ep_expert_quants=("ds_fp4",),
                    num_experts=10,
                ),
            )
        )


@pytest.mark.parametrize(
    ("ep_size", "tp_size"),
    [(2, 4), (4, 2), (4, 8), (8, 4)],
)
def test_owner_ep_size_must_equal_tp_size(ep_size, tp_size):
    with pytest.raises(ValueError, match="must equal"):
        _validate_owner_ep_config(
            _config(
                moe_ep_size=ep_size,
                tp_info=SimpleNamespace(size=tp_size, rank=0),
            )
        )
