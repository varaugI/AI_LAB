"""A dependency-free byte-level BPE tokenizer.

It begins with all 256 byte values, so every Unicode string is representable.
Training adds frequently occurring byte/token pairs to improve compression.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable, Iterator


DEFAULT_SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class ByteBPETokenizer:
    def __init__(
        self,
        special_tokens: list[str] | None = None,
        merges: list[tuple[int, int, int]] | None = None,
    ) -> None:
        self.special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be unique")
        self.special_to_id = {token: index for index, token in enumerate(self.special_tokens)}
        self.id_to_special = {index: token for token, index in self.special_to_id.items()}
        self.byte_offset = len(self.special_tokens)
        self.merges = list(merges or [])
        self._rebuild_tables()

    def _rebuild_tables(self) -> None:
        self.merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self.token_bytes: dict[int, bytes] = {
            self.byte_offset + value: bytes([value]) for value in range(256)
        }
        for rank, (left, right, new_id) in enumerate(self.merges):
            if left not in self.token_bytes or right not in self.token_bytes:
                raise ValueError("merge references an unknown token")
            self.merge_ranks[(left, right)] = (rank, new_id)
            self.token_bytes[new_id] = self.token_bytes[left] + self.token_bytes[right]
        escaped = sorted((re.escape(token) for token in self.special_tokens), key=len, reverse=True)
        self._special_pattern = re.compile("(" + "|".join(escaped) + ")") if escaped else None

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    def _base_encode(self, text: str) -> list[int]:
        return [self.byte_offset + byte for byte in text.encode("utf-8")]

    def _apply_merges(self, tokens: list[int]) -> list[int]:
        if len(tokens) < 2 or not self.merge_ranks:
            return tokens
        tokens = list(tokens)
        while len(tokens) > 1:
            best_pair = None
            best_rank = None
            best_new_id = None
            for left, right in zip(tokens, tokens[1:]):
                ranked = self.merge_ranks.get((left, right))
                if ranked is not None and (best_rank is None or ranked[0] < best_rank):
                    best_pair = (left, right)
                    best_rank, best_new_id = ranked
            if best_pair is None:
                break
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1] == best_pair[1]
                ):
                    merged.append(best_new_id)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = True,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        if allow_special and self._special_pattern:
            for piece in self._special_pattern.split(text):
                if not piece:
                    continue
                special_id = self.special_to_id.get(piece)
                if special_id is not None:
                    ids.append(special_id)
                else:
                    ids.extend(self._apply_merges(self._base_encode(piece)))
        else:
            ids.extend(self._apply_merges(self._base_encode(text)))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = False) -> str:
        chunks: list[str] = []
        byte_buffer = bytearray()

        def flush() -> None:
            if byte_buffer:
                chunks.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                flush()
                if not skip_special:
                    chunks.append(self.id_to_special[token_id])
                continue
            data = self.token_bytes.get(token_id)
            if data is None:
                raise ValueError(f"Unknown token id: {token_id}")
            byte_buffer.extend(data)
        flush()
        return "".join(chunks)

    def train(
        self,
        texts: Iterable[str],
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_training_bytes: int | None = None,
        verbose: bool = False,
    ) -> "ByteBPETokenizer":
        if vocab_size < self.byte_offset + 256:
            raise ValueError("vocab_size is smaller than the byte vocabulary")
        sequences: list[list[int]] = []
        consumed = 0
        for text in texts:
            if not text:
                continue
            encoded = self._base_encode(text)
            if max_training_bytes is not None:
                remaining = max_training_bytes - consumed
                if remaining <= 0:
                    break
                encoded = encoded[:remaining]
            if encoded:
                sequences.append(encoded)
                consumed += len(encoded)
            if max_training_bytes is not None and consumed >= max_training_bytes:
                break

        target_merges = vocab_size - (self.byte_offset + 256)
        self.merges = []
        self._rebuild_tables()
        for merge_index in range(target_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, frequency = pair_counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            new_id = self.byte_offset + 256 + len(self.merges)
            self.merges.append((pair[0], pair[1], new_id))
            for seq_index, sequence in enumerate(sequences):
                merged: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                sequences[seq_index] = merged
            self._rebuild_tables()
            if verbose and (merge_index + 1) % 100 == 0:
                print(f"Tokenizer merges: {merge_index + 1}/{target_merges}")
        return self

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "special_tokens": self.special_tokens,
            "merges": [list(item) for item in self.merges],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            special_tokens=payload["special_tokens"],
            merges=[tuple(map(int, item)) for item in payload.get("merges", [])],
        )

    @staticmethod
    def iter_text_files(paths: Iterable[str | Path]) -> Iterator[str]:
        for path in paths:
            value = Path(path)
            if value.is_dir():
                for child in sorted(value.rglob("*")):
                    if child.is_file() and child.suffix.lower() in {".txt", ".md", ".jsonl"}:
                        yield child.read_text(encoding="utf-8", errors="replace")
            elif value.is_file():
                yield value.read_text(encoding="utf-8", errors="replace")
