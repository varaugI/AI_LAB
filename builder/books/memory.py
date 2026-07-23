"""Persistent conversational memory for follow-up questions.

This memory stores only recent chat turns and lightweight retrieval hints. It is
not a trainable language model and never replaces evidence from the books.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .corpus import tokenize


PRONOUN_TERMS = {
    "he", "she", "they", "them", "him", "her", "his", "hers", "their",
    "it", "its", "this", "that", "these", "those", "there", "then",
}


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    timestamp: str
    sources: list[dict]

    @classmethod
    def create(cls, user: str, assistant: str, sources=None):
        return cls(
            user=user,
            assistant=assistant,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sources=list(sources or []),
        )


class ConversationMemory:
    def __init__(self, max_turns: int = 30, turns=None):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        self.max_turns = int(max_turns)
        self.turns: list[ConversationTurn] = list(turns or [])[-self.max_turns:]

    def add(self, user: str, assistant: str, sources=None):
        self.turns.append(ConversationTurn.create(user, assistant, sources))
        self.turns = self.turns[-self.max_turns:]

    def clear(self):
        self.turns.clear()

    def recent(self, limit: int = 4):
        return self.turns[-max(0, limit):]

    def context_text(self, limit: int = 3, assistant_characters: int = 500) -> str:
        pieces = []
        for turn in self.recent(limit):
            pieces.append(turn.user)
            pieces.append(turn.assistant[:assistant_characters])
        return " ".join(pieces)

    @staticmethod
    def _looks_like_follow_up(question: str) -> bool:
        tokens = tokenize(question)
        if len(tokens) <= 6:
            return True
        return bool(set(tokens) & PRONOUN_TERMS)

    def expand_query(self, question: str, max_context_terms: int = 18) -> str:
        """Add useful recent terms to short/pronominal follow-up questions."""
        if not self.turns or not self._looks_like_follow_up(question):
            return question

        current = set(tokenize(question))
        candidates = []
        # Prefer capitalized names from the user's previous question and source
        # titles, then fall back to distinctive words from recent turns.
        for turn in reversed(self.recent(3)):
            combined_text = f"{turn.user} {turn.assistant[:500]}"
            for name in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", combined_text):
                candidates.extend(tokenize(name))
            for source in turn.sources:
                candidates.extend(tokenize(str(source.get("title", ""))))
            candidates.extend(tokenize(turn.user))

        additions = []
        seen = set(current)
        for term in candidates:
            if len(term) <= 2 or term in seen or term in PRONOUN_TERMS:
                continue
            seen.add(term)
            additions.append(term)
            if len(additions) >= max_context_terms:
                break
        return question if not additions else f"{question} {' '.join(additions)}"

    def to_dict(self):
        return {
            "version": 1,
            "type": "conversation_memory",
            "max_turns": self.max_turns,
            "turns": [asdict(turn) for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "conversation_memory":
            raise ValueError("Not an AI LAB conversation memory file.")
        turns = [ConversationTurn(**item) for item in data.get("turns", [])]
        return cls(max_turns=data.get("max_turns", 30), turns=turns)

    def save(self, filename):
        Path(filename).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filename):
        return cls.from_dict(json.loads(Path(filename).read_text(encoding="utf-8")))
