"""A modern decoder-only transformer implemented directly with PyTorch.

The architecture uses RMSNorm, rotary position embeddings, grouped-query
attention, SwiGLU feed-forward blocks, tied token embeddings, causal masking,
and optional key/value caching for autoregressive generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        normalized = values.float() * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.epsilon)
        return (normalized * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head dimension must be even for rotary embeddings")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos_cache", frequencies.cos(), persistent=False)
        self.register_buffer("sin_cache", frequencies.sin(), persistent=False)

    def forward(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # values: [batch, heads, time, head_dim]
        cos = self.cos_cache.index_select(0, positions).to(dtype=values.dtype)[None, None, :, :]
        sin = self.sin_cache.index_select(0, positions).to(dtype=values.dtype)[None, None, :, :]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return values
    batch, heads, time, dimension = values.shape
    return (
        values[:, :, None, :, :]
        .expand(batch, heads, repeats, time, dimension)
        .reshape(batch, heads * repeats, time, dimension)
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = int(config.n_kv_heads)
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.dropout = float(config.dropout)
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, time, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_length = 0 if past_key_value is None else past_key_value[0].size(2)
        positions = torch.arange(past_length, past_length + time, device=hidden.device)
        query = self.rope(query, positions)
        key = self.rope(key, positions)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None

        repeated_key = _repeat_kv(key, self.kv_repeats)
        repeated_value = _repeat_kv(value, self.kv_repeats)
        key_length = repeated_key.size(2)

        scores = torch.matmul(query, repeated_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(past_length, past_length + time, device=hidden.device)[:, None]
        key_positions = torch.arange(key_length, device=hidden.device)[None, :]
        causal_mask = key_positions <= query_positions
        scores = scores.masked_fill(~causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = torch.matmul(probabilities, repeated_value)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.residual_dropout(self.out_proj(attended)), present


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, int(config.d_ff), bias=config.use_bias)
        self.up_proj = nn.Linear(config.d_model, int(config.d_ff), bias=config.use_bias)
        self.down_proj = nn.Linear(int(config.d_ff), config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_output, present = self.attention(
            self.attention_norm(hidden), past_key_value=past_key_value, use_cache=use_cache
        )
        hidden = hidden + attention_output
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden, present


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)
        # Residual projections use scaled initialization.
        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> TransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, time = input_ids.shape
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].size(2)
        if past_length + time > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {past_length + time} exceeds max_seq_len={self.config.max_seq_len}"
            )
        hidden = self.dropout(self.token_embedding(input_ids))
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = block(hidden, past_key_value=past, use_cache=use_cache)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return TransformerOutput(logits=logits, loss=loss, past_key_values=presents or None)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.05,
        eos_token_ids: int | Iterable[int] | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if isinstance(eos_token_ids, int):
            stop_ids = {eos_token_ids}
        else:
            stop_ids = set(eos_token_ids or [])
        generated = input_ids
        cache = None
        model_input = generated[:, -self.config.max_seq_len :]
        for _ in range(max_new_tokens):
            cached_length = 0 if cache is None else cache[0][0].size(2)
            if use_cache and cache is not None and cached_length < self.config.max_seq_len:
                model_input = generated[:, -1:]
            else:
                # Rebuild cache whenever the context window has filled or slid.
                cache = None
                model_input = generated[:, -self.config.max_seq_len :]
            output = self(model_input, past_key_values=cache, use_cache=use_cache)
            cache = output.past_key_values if use_cache else None
            logits = output.logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                for row in range(generated.size(0)):
                    unique_ids = generated[row].unique()
                    selected = logits[row, unique_ids]
                    logits[row, unique_ids] = torch.where(
                        selected < 0,
                        selected * repetition_penalty,
                        selected / repetition_penalty,
                    )
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None and 0 < top_k < logits.size(-1):
                    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, sorted_indices, sorted_logits)
                    logits = filtered
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if stop_ids and all(int(token) in stop_ids for token in next_token.squeeze(-1)):
                break
        return generated

    def save_pretrained(self, directory: str | Path, extra: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / "config.json")
        payload = {"model": self.state_dict(), "extra": extra or {}}
        torch.save(payload, directory / "model.pt")

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TransformerLM":
        directory = Path(directory)
        config = ModelConfig.load(directory / "config.json")
        model = cls(config)
        payload = torch.load(directory / "model.pt", map_location=map_location, weights_only=False)
        state = payload.get("model", payload)
        model.load_state_dict(state)
        return model
