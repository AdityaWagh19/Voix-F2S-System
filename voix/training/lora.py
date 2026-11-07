"""Manual LoRA adapters for PyTorch ``nn.Linear`` layers.

``peft`` is not installed in this environment, so this module implements the
small subset the StyleTTS2 prosody-diffusion pilot needs:

* a zero-initialised low-rank update on an existing ``nn.Linear``,
* recursive injection into a module tree,
* freezing everything except the adapter,
* saving/loading adapter weights only (not the frozen base model),
* a switch to disable the adapter for paired base-vs-LoRA evaluation.

The base layer is kept as a child module.  At initialisation ``lora_B`` is
zero, so ``LoRALinear(x)`` is exactly ``base(x)``; that property is pinned by
``tests/test_lora.py``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

SCHEMA_VERSION = "voix.lora.v1"
TargetFilter = Optional[Callable[[str, nn.Linear], bool]]


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter hyper-parameters shared by every injected layer."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if int(self.rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if float(self.alpha) <= 0.0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(
                f"LoRA dropout must be in [0, 1), got {self.dropout}"
            )

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


class LoRALinear(nn.Module):
    """``base(x) + scaling * B(A(dropout(x)))`` with a frozen base layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"base must be nn.Linear, got {type(base).__name__}")
        config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.dropout = float(config.dropout)
        self.scaling = config.scaling
        self.lora_dropout = (
            nn.Dropout(float(config.dropout))
            if float(config.dropout) > 0.0
            else nn.Identity()
        )
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=device, dtype=dtype)
        )
        # Plain Python attribute: not part of state_dict, so checkpoints stay
        # adapter-only and enabling/disabling never mutates saved weights.
        self.lora_enabled = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.lora_enabled:
            return out
        delta = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        )
        return out + delta.to(out.dtype) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


def _inject(
    parent: nn.Module,
    prefix: str,
    config: LoRAConfig,
    target: TargetFilter,
) -> List[str]:
    injected: List[str] = []
    for name, child in list(parent.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and (target is None or target(full_name, child)):
            setattr(
                parent,
                name,
                LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ),
            )
            injected.append(full_name)
        else:
            injected.extend(_inject(child, full_name, config, target))
    return injected


def inject_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target: TargetFilter = None,
) -> List[str]:
    """Replace matching ``nn.Linear`` children with ``LoRALinear``.

    Returns the dotted names of the injected layers.  Calling this twice does
    not wrap an adapter again.  ``target`` receives ``(dotted_name, linear)``
    and can restrict which layers are adapted.
    """
    config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
    return _inject(module, "", config, target)


def iter_lora_modules(
    module: nn.Module,
    prefix: str = "",
) -> Iterator[Tuple[str, LoRALinear]]:
    """Yield ``(dotted_name, adapter)`` for every adapter in the tree."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRALinear):
            yield full_name, child
        else:
            yield from iter_lora_modules(child, full_name)


def lora_named_parameters(
    module: nn.Module,
) -> Iterator[Tuple[str, nn.Parameter]]:
    for name, adapter in iter_lora_modules(module):
        yield f"{name}.lora_A", adapter.lora_A
        yield f"{name}.lora_B", adapter.lora_B


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Adapter-only state dict (CPU tensors), never the frozen base weights."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in lora_named_parameters(module)
    }


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def adapter_statistics(module: nn.Module) -> Dict[str, Any]:
    """Report whether the zero-initialised ``lora_B`` matrices have moved.

    A fine-tune that leaves every ``lora_B`` at zero is indistinguishable from
    the base model; this makes that failure visible instead of reporting a
    misleading "no metric change" result.
    """
    norms = [
        float(adapter.lora_B.detach().float().norm())
        for _, adapter in iter_lora_modules(module)
    ]
    if not norms:
        return {"layers": 0, "lora_b_norm_mean": None}
    return {
        "layers": len(norms),
        "lora_b_norm_mean": round(float(sum(norms) / len(norms)), 8),
        "lora_b_norm_min": round(float(min(norms)), 8),
        "lora_b_norm_max": round(float(max(norms)), 8),
    }


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    """Toggle every adapter without touching its weights."""
    if isinstance(module, LoRALinear):
        module.lora_enabled = bool(enabled)
    for _, adapter in iter_lora_modules(module):
        adapter.lora_enabled = bool(enabled)


def _infer_config(module: nn.Module) -> LoRAConfig:
    adapters = [adapter for _, adapter in iter_lora_modules(module)]
    if not adapters:
        raise ValueError("module contains no LoRA adapters")
    first = adapters[0]
    config = LoRAConfig(
        rank=first.rank,
        alpha=first.alpha,
        dropout=first.dropout,
    )
    for adapter in adapters[1:]:
        if adapter.rank != config.rank or adapter.alpha != config.alpha:
            raise ValueError(
                "all LoRA adapters must share rank/alpha before saving"
            )
    return config


def save_lora(
    module: nn.Module,
    path: Union[str, Path],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[LoRAConfig] = None,
) -> Path:
    """Save adapter weights plus enough metadata to reconstruct the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = config or _infer_config(module)
    targets = [name for name, _ in iter_lora_modules(module)]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(resolved),
        "targets": targets,
        "state_dict": lora_state_dict(module),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return path


def load_lora(
    module: nn.Module,
    path: Union[str, Path],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load adapter weights into an already-injected module.

    Returns the checkpoint payload (including its ``metadata``).  Raises when
    the checkpoint's rank/alpha do not match the injected adapters, because
    silently loading a differently-shaped adapter would corrupt the model.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a voix LoRA checkpoint")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported LoRA schema {payload.get('schema_version')!r}"
        )

    saved = payload.get("config") or {}
    adapters = [adapter for _, adapter in iter_lora_modules(module)]
    if not adapters:
        raise ValueError("module contains no LoRA adapters to load into")
    for adapter in adapters:
        if int(saved.get("rank", -1)) != adapter.rank:
            raise ValueError(
                f"rank mismatch: checkpoint {saved.get('rank')} vs "
                f"adapter {adapter.rank}"
            )
        if abs(float(saved.get("alpha", -1.0)) - adapter.alpha) > 1e-9:
            raise ValueError(
                f"alpha mismatch: checkpoint {saved.get('alpha')} vs "
                f"adapter {adapter.alpha}"
            )

    state = payload["state_dict"]
    current = dict(lora_named_parameters(module))
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if strict and (missing or unexpected):
        raise ValueError(
            "LoRA state dict does not match the injected modules; "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    for name, tensor in state.items():
        parameter = current.get(name)
        if parameter is None:
            continue
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return payload
