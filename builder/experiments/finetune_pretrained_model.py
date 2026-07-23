"""Continue-pretrain or response-tune a Hugging Face causal language model.

This is the higher-quality path when you have enough GPU memory: start from a
capable open model, continue pretraining on your books, then LoRA/full fine-tune
on approved conversations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def require_dependencies():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Install the production stack first: pip install -r requirements-production.txt"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def read_jsonl(path: str | Path) -> list[dict]:
    values = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                values.append(json.loads(line))
    if not values:
        raise ValueError("Dataset is empty")
    return values


class ContinuedPretrainingDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        text = str(self.records[index].get("text", ""))
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {"input_ids": encoded["input_ids"], "labels": encoded["input_ids"].copy()}


class ResponseOnlySFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        pieces = [f"{item['role'].upper()}: {item['content']}" for item in messages]
        if add_generation_prompt:
            pieces.append("ASSISTANT:")
        return "\n\n".join(pieces)

    def __getitem__(self, index):
        record = self.records[index]
        messages = record.get("messages")
        if not messages:
            messages = [
                {"role": "user", "content": str(record.get("instruction", ""))},
                {"role": "assistant", "content": str(record.get("output", record.get("response", "")))},
            ]
        if messages[-1].get("role") != "assistant":
            raise ValueError("Each SFT example must end with an assistant message")
        prompt_text = self._render(messages[:-1], add_generation_prompt=True)
        full_text = self._render(messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=True
        )["input_ids"]
        labels = full_ids.copy()
        masked = min(len(prompt_ids), len(labels))
        labels[:masked] = [-100] * masked
        return {"input_ids": full_ids, "labels": labels}


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(item["input_ids"]) for item in features)
        input_rows, label_rows, masks = [], [], []
        for item in features:
            padding = maximum - len(item["input_ids"])
            input_rows.append(item["input_ids"] + [self.pad_token_id] * padding)
            label_rows.append(item["labels"] + [-100] * padding)
            masks.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model name or local path")
    parser.add_argument("dataset", help="Corpus JSONL or approved chat JSONL")
    parser.add_argument("--output", default="data/checkpoints/hf-finetuned")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments = require_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("Install peft to use --lora") from exc
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            target_modules="all-linear",
        ))
        model.print_trainable_parameters()

    records = read_jsonl(args.dataset)
    dataset = (
        ContinuedPretrainingDataset(records, tokenizer, args.max_length)
        if args.mode == "pretrain"
        else ResponseOnlySFTDataset(records, tokenizer, args.max_length)
    )
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.1 if args.mode == "pretrain" else 0.01,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({"output": args.output, "examples": len(dataset), "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
