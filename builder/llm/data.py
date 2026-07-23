"""Datasets and corpus preparation for pretraining and chat fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ByteBPETokenizer


@dataclass
class CorpusRecord:
    text: str
    title: str = ""
    source: str = ""
    domain: str = "general"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "domain": self.domain,
        }


def iter_jsonl_records(path: str | Path) -> Iterator[CorpusRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc
            text = str(value.get("text", "")).strip()
            if text:
                yield CorpusRecord(
                    text=text,
                    title=str(value.get("title", "")),
                    source=str(value.get("source", "")),
                    domain=str(value.get("domain", "general")),
                )


def deduplicate_records(records: Iterable[CorpusRecord]) -> Iterator[CorpusRecord]:
    seen: set[str] = set()
    for record in records:
        normalized = " ".join(record.text.split()).casefold()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        yield record


def prepare_binary_dataset(
    records: Iterable[CorpusRecord] | str | Path,
    tokenizer: ByteBPETokenizer,
    output_dir: str | Path,
    *,
    validation_fraction: float = 0.02,
    seed: int = 42,
    add_document_boundaries: bool = True,
) -> dict:
    """Tokenize a corpus into efficient uint32 train/validation streams."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records = iter_jsonl_records(records) if isinstance(records, (str, Path)) else records
    materialized = list(deduplicate_records(source_records))
    if not materialized:
        raise ValueError("No non-empty corpus records were supplied")
    random.Random(seed).shuffle(materialized)
    split = max(1, int(len(materialized) * (1.0 - validation_fraction)))
    if validation_fraction > 0 and len(materialized) > 1:
        split = min(split, len(materialized) - 1)
    train_records = materialized[:split]
    validation_records = materialized[split:]

    def write_stream(path: Path, subset: list[CorpusRecord]) -> int:
        count = 0
        with path.open("wb") as handle:
            for record in subset:
                ids = tokenizer.encode(
                    record.text,
                    add_bos=add_document_boundaries,
                    add_eos=add_document_boundaries,
                )
                np.asarray(ids, dtype=np.uint32).tofile(handle)
                count += len(ids)
        return count

    train_tokens = write_stream(output_dir / "train.bin", train_records)
    val_tokens = write_stream(output_dir / "val.bin", validation_records)
    tokenizer.save(output_dir / "tokenizer.json")
    metadata = {
        "version": 1,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "train_documents": len(train_records),
        "validation_documents": len(validation_records),
        "dtype": "uint32",
    }
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


class BinaryTokenDataset(Dataset):
    """Random-access next-token blocks backed by a memory-mapped token file."""

    def __init__(self, path: str | Path, block_size: int, stride: int | None = None) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)
        if self.block_size <= 1 or self.stride <= 0:
            raise ValueError("block_size must be > 1 and stride must be positive")
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        if len(self.tokens) < self.block_size + 1:
            raise ValueError(f"Not enough tokens in {self.path} for block_size={self.block_size}")
        self.length = 1 + (len(self.tokens) - self.block_size - 1) // self.stride

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        start = index * self.stride
        window = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(window[:-1].copy()),
            "labels": torch.from_numpy(window[1:].copy()),
        }


def format_chat_messages(messages: list[dict], tokenizer: ByteBPETokenizer) -> tuple[list[int], list[int]]:
    """Format chat messages and mask all non-assistant tokens from SFT loss."""
    input_ids: list[int] = [tokenizer.bos_id]
    labels: list[int] = [-100]
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        prefix = f"<|{role}|>\n"
        segment = tokenizer.encode(prefix + content + "\n", allow_special=True)
        input_ids.extend(segment)
        if role == "assistant":
            labels.extend(segment)
        else:
            labels.extend([-100] * len(segment))
    input_ids.append(tokenizer.eos_id)
    if any(label != -100 for label in labels):
        labels.append(tokenizer.eos_id)
    else:
        labels.append(-100)
    return input_ids, labels


def load_sft_examples(path: str | Path) -> list[list[dict]]:
    examples: list[list[dict]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if "messages" in value:
                messages = value["messages"]
            else:
                instruction = str(value.get("instruction", "")).strip()
                additional_input = str(value.get("input", "")).strip()
                output = str(value.get("output", value.get("response", ""))).strip()
                prompt = instruction + (("\n\n" + additional_input) if additional_input else "")
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ]
            if not isinstance(messages, list):
                raise ValueError(f"messages must be a list at line {line_number}")
            examples.append(messages)
    return examples


class SFTDataset(Dataset):
    def __init__(
        self,
        examples: list[list[dict]] | str | Path,
        tokenizer: ByteBPETokenizer,
        block_size: int,
    ) -> None:
        self.examples = load_sft_examples(examples) if isinstance(examples, (str, Path)) else examples
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        if not self.examples:
            raise ValueError("No SFT examples were supplied")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = format_chat_messages(self.examples[index], self.tokenizer)
        input_ids = input_ids[: self.block_size]
        labels = labels[: self.block_size]
        if len(input_ids) < 2:
            raise ValueError("SFT example is too short")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sft_collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    maximum = max(item["input_ids"].numel() for item in batch)
    input_rows = []
    label_rows = []
    for item in batch:
        padding = maximum - item["input_ids"].numel()
        input_rows.append(torch.cat((item["input_ids"], torch.full((padding,), pad_id, dtype=torch.long))))
        label_rows.append(torch.cat((item["labels"], torch.full((padding,), -100, dtype=torch.long))))
    return {"input_ids": torch.stack(input_rows), "labels": torch.stack(label_rows)}
