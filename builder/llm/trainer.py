"""GPU-aware language-model trainer with checkpoint/resume support."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

from .config import TrainingConfig
from .model import TransformerLM


class LanguageModelTrainer:
    def __init__(
        self,
        model: TransformerLM,
        train_dataset: Dataset,
        validation_dataset: Dataset | None,
        config: TrainingConfig,
        *,
        collate_fn=None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config
        self.collate_fn = collate_fn
        self.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        if self.distributed and not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank if self.distributed else 0)
            if self.distributed:
                torch.cuda.set_device(self.device)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.is_main = self.rank == 0
        self._seed_everything(config.seed + self.rank)
        self.model.to(self.device)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            )
        self.raw_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            fused=self.device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames,
        )
        self.step = 0
        self.best_validation_loss = float("inf")
        self.output_dir = Path(config.output_dir)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config.save(self.output_dir / "training_config.json")
        self.autocast_dtype = self._resolve_precision(config.precision)
        scaler_enabled = self.device.type == "cuda" and self.autocast_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _resolve_precision(self, precision: str) -> torch.dtype | None:
        if precision == "fp32":
            return None
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        if self.device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # CPU defaults to fp32 for maximum operator compatibility.
        return None

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        device_type = self.device.type if self.device.type in {"cuda", "cpu"} else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)

    def learning_rate_at(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + cosine * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def _loader(self, dataset: Dataset, training: bool) -> DataLoader:
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=training, seed=self.config.seed)
        else:
            sampler = RandomSampler(dataset) if training else None
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def _cycle(loader: DataLoader):
        epoch = 0
        while True:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

    @torch.no_grad()
    def evaluate(self) -> float | None:
        if self.validation_dataset is None:
            return None
        self.model.eval()
        loader = self._loader(self.validation_dataset, training=False)
        losses = []
        for index, batch in enumerate(loader):
            if index >= self.config.eval_batches:
                break
            batch = self._move_batch(batch)
            with self._autocast():
                output = self.model(batch["input_ids"], labels=batch["labels"])
            losses.append(output.loss.detach().float())
        if not losses:
            self.model.train()
            return None
        mean = torch.stack(losses).mean().to(self.device)
        if self.distributed:
            torch.distributed.all_reduce(mean, op=torch.distributed.ReduceOp.SUM)
            mean /= torch.distributed.get_world_size()
        self.model.train()
        return float(mean.item())

    def save_checkpoint(self, name: str | None = None, validation_loss: float | None = None) -> Path | None:
        if not self.is_main:
            return None
        checkpoint_dir = self.output_dir / (name or f"checkpoint-{self.step:08d}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.raw_model.config.save(checkpoint_dir / "config.json")
        model_state = self.raw_model.state_dict()
        torch.save({"model": model_state, "extra": {"step": self.step}}, checkpoint_dir / "model.pt")
        torch.save(
            {
                "model": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "best_validation_loss": self.best_validation_loss,
                "validation_loss": validation_loss,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            checkpoint_dir / "training_state.pt",
        )
        self._prune_checkpoints()
        return checkpoint_dir

    def _prune_checkpoints(self) -> None:
        checkpoints = sorted(
            path for path in self.output_dir.glob("checkpoint-*") if path.is_dir()
        )
        keep = max(1, self.config.keep_last_checkpoints)
        for old in checkpoints[:-keep]:
            shutil.rmtree(old, ignore_errors=True)

    def load_checkpoint(self, directory: str | Path) -> None:
        directory = Path(directory)
        payload = torch.load(directory / "training_state.pt", map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload.get("step", 0))
        self.best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        rng = payload.get("rng") or {}
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    def train(self) -> dict:
        loader = self._loader(self.train_dataset, training=True)
        iterator = self._cycle(loader)
        self.model.train()
        started = time.time()
        running_loss = 0.0
        tokens_since_log = 0
        while self.step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(self.config.gradient_accumulation_steps):
                batch = self._move_batch(next(iterator))
                tokens_since_log += int((batch["labels"] != -100).sum().item())
                with self._autocast():
                    output = self.model(batch["input_ids"], labels=batch["labels"])
                    loss = output.loss / self.config.gradient_accumulation_steps
                accumulated_loss += float(loss.detach().item())
                self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            lr = self.learning_rate_at(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            running_loss += accumulated_loss

            if self.step % self.config.log_interval == 0 and self.is_main:
                elapsed = max(time.time() - started, 1e-6)
                record = {
                    "step": self.step,
                    "loss": running_loss / self.config.log_interval,
                    "learning_rate": lr,
                    "tokens_per_second": tokens_since_log / elapsed,
                    "elapsed_seconds": elapsed,
                }
                print(json.dumps(record))
                with (self.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                running_loss = 0.0
                tokens_since_log = 0
                started = time.time()

            validation_loss = None
            if self.validation_dataset is not None and self.step % self.config.eval_interval == 0:
                validation_loss = self.evaluate()
                if self.is_main and validation_loss is not None:
                    eval_record = {
                        "step": self.step,
                        "validation_loss": validation_loss,
                        "perplexity": math.exp(min(validation_loss, 20.0)),
                    }
                    print(json.dumps(eval_record))
                    with (self.output_dir / "eval_log.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(eval_record) + "\n")
                    if validation_loss < self.best_validation_loss:
                        self.best_validation_loss = validation_loss
                        self.save_checkpoint("best", validation_loss)

            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(validation_loss=validation_loss)

        final = self.save_checkpoint("final")
        if self.distributed:
            torch.distributed.barrier()
        return {
            "steps": self.step,
            "best_validation_loss": self.best_validation_loss,
            "checkpoint": str(final) if final else "",
        }
