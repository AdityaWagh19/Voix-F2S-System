"""Tests for the manual LoRA adapter used by the StyleTTS2 pilot."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from voix.training.lora import (
    LoRAConfig,
    LoRALinear,
    adapter_statistics,
    count_trainable_parameters,
    freeze_module,
    inject_lora,
    iter_lora_modules,
    load_lora,
    lora_named_parameters,
    lora_state_dict,
    save_lora,
    set_lora_enabled,
)


def test_zero_init_is_exactly_the_base_layer():
    torch.manual_seed(0)
    base = nn.Linear(6, 4)
    adapter = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 6)
    assert torch.equal(adapter(x), base(x))
    with torch.no_grad():
        adapter.lora_B.normal_(0.0, 0.1)
    assert not torch.allclose(adapter(x), base(x))


def test_base_is_frozen_and_only_lora_receives_gradients():
    module = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    injected = inject_lora(module, rank=2, alpha=4.0)
    assert injected == ["0", "2"]
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if ".base." in name
    }
    module(torch.randn(3, 4)).sum().backward()

    for name, parameter in module.named_parameters():
        if ".base." in name:
            assert parameter.grad is None, name
            assert torch.equal(parameter, before[name]), name
    lora_b = [
        parameter
        for name, parameter in lora_named_parameters(module)
        if name.endswith("lora_B")
    ]
    assert len(lora_b) == 2
    assert all(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in lora_b
    )


def test_save_load_roundtrip_contains_only_adapters(tmp_path):
    def build() -> nn.Sequential:
        torch.manual_seed(1)
        module = nn.Sequential(nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 2))
        inject_lora(module, rank=2, alpha=4.0)
        return module

    first, second = build(), build()
    with torch.no_grad():
        for _, adapter in iter_lora_modules(first):
            adapter.lora_B.normal_(0.0, 0.1)

    state = lora_state_dict(first)
    assert state
    assert all("lora_" in key for key in state)
    assert not any("base" in key for key in state)

    path = tmp_path / "adapter.pt"
    save_lora(first, path, metadata={"epoch": 3, "note": "roundtrip"})
    payload = load_lora(second, path)
    assert payload["metadata"]["epoch"] == 3
    assert payload["targets"] == ["0", "2"]

    x = torch.randn(4, 5)
    assert torch.allclose(first(x), second(x))


def test_target_filter_and_reinjection_does_not_double_wrap():
    module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    first = inject_lora(
        module,
        rank=2,
        alpha=2.0,
        target=lambda name, _linear: name == "1",
    )
    assert first == ["1"]
    assert isinstance(module[0], nn.Linear)
    assert isinstance(module[1], LoRALinear)

    second = inject_lora(module, rank=2, alpha=2.0)
    assert second == ["0"]
    assert isinstance(module[1], LoRALinear)
    assert sum(1 for _ in iter_lora_modules(module)) == 2


def test_adapter_statistics_detect_zero_initialisation():
    module = nn.Sequential(nn.Linear(3, 3))
    inject_lora(module, rank=2, alpha=2.0)
    initial = adapter_statistics(module)
    assert initial["layers"] == 1
    assert initial["lora_b_norm_mean"] == 0.0
    with torch.no_grad():
        for _, adapter in iter_lora_modules(module):
            adapter.lora_B.normal_(0.0, 0.1)
    moved = adapter_statistics(module)
    assert moved["lora_b_norm_mean"] > 0.0
    assert moved["lora_b_norm_min"] > 0.0


def test_enable_disable_and_freeze_helpers():
    adapter = LoRALinear(nn.Linear(3, 3), rank=2, alpha=2.0)
    with torch.no_grad():
        adapter.lora_B.normal_(0.0, 0.1)
    x = torch.randn(2, 3)
    enabled = adapter(x)

    set_lora_enabled(adapter, False)
    assert torch.equal(adapter(x), adapter.base(x))
    set_lora_enabled(adapter, True)
    assert torch.allclose(adapter(x), enabled)

    freeze_module(adapter)
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert count_trainable_parameters(adapter) == 0


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, alpha=0.0)
    with pytest.raises(ValueError):
        LoRAConfig(rank=2, alpha=2.0, dropout=1.0)
