"""Configuration objects for transformer training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int | None = None
    d_ff: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 1:
            raise ValueError("max_seq_len must be greater than 1")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.d_ff is None:
            # LLaMA-style rounded SwiGLU width.
            raw = int((8 * self.d_model) / 3)
            self.d_ff = 256 * ((raw + 255) // 256)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TrainingConfig:
    output_dir: str = "data/checkpoints/run"
    seed: int = 42
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 500
    log_interval: int = 10
    num_workers: int = 0
    precision: str = "auto"  # auto, fp32, fp16, bf16
    compile_model: bool = False
    resume_from: str | None = None
    keep_last_checkpoints: int = 3

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
