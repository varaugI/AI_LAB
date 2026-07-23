"""Heuristic character discovery and relationship tracking for novels."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from .corpus import TextChunk, split_sentences


NAME_PATTERN = re.compile(
    r"\b(?:(?:Mr|Mrs|Ms|Miss|Dr|Lord|Lady|Sir|Dame|King|Queen|Prince|Princess|"
    r"Captain|Professor)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)
FALSE_NAMES = {
    "A", "An", "And", "After", "At", "Before", "Beyond", "But", "By", "Chapter", "During",
    "For", "Finally", "From", "He", "Her", "Here", "His", "How", "I", "If", "In", "Inside",
    "Into", "It", "Its", "Later", "Meanwhile", "Near", "No", "Not", "Now", "Of", "Outside",
    "On", "Once", "Only", "Or", "Page", "She", "So", "That", "The", "Their",
    "Then", "There", "They", "This", "Those", "Through", "To", "Under", "Until",
    "We", "What", "When", "Where", "While", "Who", "Why", "With", "Without", "Yet",
    "Yes", "You",
}
TITLE_PATTERN = re.compile(r"^(Mr|Mrs|Ms|Miss|Dr|Lord|Lady|Sir|Dame|King|Queen|Prince|Princess|Captain|Professor)\.?\s+")


@dataclass
class CharacterProfile:
    name: str
    mentions: int
    first_appearance: str
    locations: list[str]
    descriptions: list[str]
    relationships: dict[str, int]

    def to_dict(self):
        return asdict(self)


class CharacterTracker:
    def __init__(self, profiles=None):
        self.profiles: dict[str, CharacterProfile] = dict(profiles or {})

    @staticmethod
    def _extract_names(sentence: str):
        names = []
        for match in NAME_PATTERN.finditer(sentence):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            bare = TITLE_PATTERN.sub("", value)
            parts = bare.split()
            while len(parts) > 1 and parts[0] in FALSE_NAMES:
                parts.pop(0)
            bare = " ".join(parts)
            value = bare if not TITLE_PATTERN.match(value) else value
            if bare in FALSE_NAMES or value in FALSE_NAMES or len(bare) < 2:
                continue
            # Reject likely sentence-openers unless repeated/capitalized as a
            # multi-word name. This keeps ordinary prose from becoming a cast.
            if match.start() == 0 and " " not in bare and bare in FALSE_NAMES:
                continue
            names.append(value if TITLE_PATTERN.match(value) else bare)
        return list(dict.fromkeys(names))

    def analyze(self, chunks: list[TextChunk], minimum_mentions: int = 2, max_descriptions: int = 5):
        mention_counts = Counter()
        locations = defaultdict(list)
        descriptions = defaultdict(list)
        relationships = defaultdict(Counter)
        first = {}

        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                names = self._extract_names(sentence)
                if not names:
                    continue
                for name in names:
                    mention_counts[name] += 1
                    if name not in first:
                        first[name] = chunk.location
                    if chunk.location not in locations[name]:
                        locations[name].append(chunk.location)
                    if sentence not in descriptions[name] and len(descriptions[name]) < max_descriptions * 3:
                        descriptions[name].append(sentence)
                for name in names:
                    for other in names:
                        if other != name:
                            relationships[name][other] += 1

        profiles = {}
        for name, count in mention_counts.most_common():
            if count < minimum_mentions:
                continue
            # Prefer sentences that describe an action or identity rather than
            # isolated dialogue tags.
            ranked = sorted(
                descriptions[name],
                key=lambda sentence: (
                    bool(re.search(r"\b(was|were|is|became|carried|found|entered|left|wanted|feared|loved|knew|said)\b", sentence.lower())),
                    len(sentence),
                ),
                reverse=True,
            )[:max_descriptions]
            profiles[name] = CharacterProfile(
                name=name,
                mentions=count,
                first_appearance=first[name],
                locations=locations[name][:20],
                descriptions=ranked,
                relationships=dict(relationships[name].most_common(12)),
            )
        self.profiles = profiles
        return self

    def list(self, limit=30):
        return sorted(self.profiles.values(), key=lambda item: (-item.mentions, item.name))[:limit]

    def get(self, name: str):
        target = name.strip().lower()
        if not target:
            return None
        exact = next((profile for key, profile in self.profiles.items() if key.lower() == target), None)
        if exact:
            return exact
        matches = [
            profile for key, profile in self.profiles.items()
            if target in key.lower() or key.lower() in target
        ]
        return max(matches, key=lambda item: item.mentions) if matches else None

    def to_dict(self):
        return {
            "version": 1,
            "type": "character_index",
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "character_index":
            raise ValueError("Not an AI LAB character index.")
        return cls({name: CharacterProfile(**profile) for name, profile in data.get("profiles", {}).items()})

    def save(self, filename):
        Path(filename).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, filename):
        return cls.from_dict(json.loads(Path(filename).read_text(encoding="utf-8")))
