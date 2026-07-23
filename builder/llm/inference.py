"""Local transformer chat backend."""

from __future__ import annotations

from pathlib import Path
import threading

import torch

from builder.books.backends import BackendResponse, ChatBackend
from .model import TransformerLM
from .tokenizer import ByteBPETokenizer


class LocalTransformerBackend(ChatBackend):
    name = "ai-lab-transformer"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.model_name = checkpoint_dir.name
        self.model = TransformerLM.from_pretrained(checkpoint_dir, map_location="cpu")
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else checkpoint_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            candidate = checkpoint_dir.parent / "tokenizer.json"
            if candidate.exists():
                tokenizer_path = candidate
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self.model_name

    def available(self) -> bool:
        return True

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        pieces = ["<|bos|>"]
        for message in messages:
            role = str(message.get("role", "user")).lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            pieces.append(f"<|{role}|>\n{str(message.get('content', '')).strip()}\n")
        pieces.append("<|assistant|>\n")
        return "".join(pieces)

    def generate(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 900) -> BackendResponse:
        prompt = self._format_messages(messages)
        ids = self.tokenizer.encode(prompt, allow_special=True)
        ids = ids[-self.model.config.max_seq_len + 1 :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock:
            generated = self.model.generate(
                input_ids,
                max_new_tokens=min(int(max_tokens), self.model.config.max_seq_len - 1),
                temperature=float(temperature),
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.08,
                eos_token_ids=[self.tokenizer.eos_id, self.tokenizer.special_to_id["<|user|>"]],
            )
        new_ids = generated[0, input_ids.size(1) :].tolist()
        text = self.tokenizer.decode(new_ids, skip_special=True).strip()
        return BackendResponse(text=text, model=self.model_name, backend=self.name)
