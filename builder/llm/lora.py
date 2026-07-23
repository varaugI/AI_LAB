"""Lightweight LoRA adapters for memory-efficient fine-tuning."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(values), self.lora_a), self.lora_b)
        return self.base(values) + update * self.scale


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_names: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"),
) -> list[str]:
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
    if not replaced:
        raise ValueError("No matching linear layers were found for LoRA")
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def save_lora(model: nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    torch.save({"adapters": lora_state_dict(model), "metadata": metadata or {}}, path)


def load_lora(model: nn.Module, path: str | Path, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["adapters"], strict=False)
    if strict and result.unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys: {result.unexpected_keys}")
    return payload.get("metadata", {})


def merge_lora(model: nn.Module) -> list[str]:
    """Merge LoRA updates into ordinary Linear layers for standalone inference."""
    merged: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            base = child.base
            replacement = nn.Linear(
                base.in_features, base.out_features,
                bias=base.bias is not None,
                device=base.weight.device, dtype=base.weight.dtype,
            )
            update = torch.matmul(child.lora_b, child.lora_a) * child.scale
            replacement.weight.data.copy_(base.weight.data + update.to(base.weight.dtype))
            if base.bias is not None:
                replacement.bias.data.copy_(base.bias.data)
            setattr(module, child_name, replacement)
            merged.append(full_name)
    return merged
