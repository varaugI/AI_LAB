"""Dependency-free extractive summaries for imported novels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .assistant import STOP_WORDS
from .corpus import TextChunk, split_sentences, tokenize


@dataclass
class SummarySource:
    number: int
    title: str
    location: str
    source: str
    excerpt: str


@dataclass
class SummaryResult:
    title: str
    summary: str
    sources: list[SummarySource]
    sentence_count: int
    source_word_count: int


class ExtractiveSummarizer:
    """Select representative sentences without inventing new plot details."""

    def __init__(self, chunks: list[TextChunk]):
        self.chunks = list(chunks)

    @staticmethod
    def _content_terms(text: str):
        return [
            token for token in tokenize(text)
            if token not in STOP_WORDS and len(token) > 2 and not token.isdigit()
        ]

    def _sentence_candidates(self, chunks):
        frequencies = Counter()
        all_sentences = []
        for chunk_index, chunk in enumerate(chunks):
            sentences = split_sentences(chunk.text)
            for sentence_index, sentence in enumerate(sentences):
                terms = self._content_terms(sentence)
                if len(terms) < 3 or len(sentence) < 25:
                    continue
                frequencies.update(set(terms))
                all_sentences.append((chunk_index, sentence_index, sentence, terms, chunk))

        if not all_sentences:
            return []
        maximum = max(frequencies.values(), default=1)
        candidates = []
        for chunk_index, sentence_index, sentence, terms, chunk in all_sentences:
            unique_terms = set(terms)
            lexical = sum(frequencies[term] / maximum for term in unique_terms)
            lexical /= math.sqrt(max(1, len(terms)))
            # Opening sentences often establish setting/characters; do not let
            # them dominate, but give them a small narrative-position bonus.
            position = 0.35 / (1 + sentence_index) + 0.15 / (1 + chunk_index)
            name_bonus = min(0.6, 0.12 * len(re.findall(r"\b[A-Z][a-z]+\b", sentence)))
            candidates.append({
                "score": lexical + position + name_bonus,
                "sentence": sentence,
                "terms": unique_terms,
                "chunk": chunk,
                "order": (chunk.chunk_id, sentence_index),
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    def summarize(
        self,
        title: str | None = None,
        max_sentences: int = 8,
        max_characters: int = 3000,
    ) -> SummaryResult:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive.")
        chunks = [
            chunk for chunk in self.chunks
            if title is None or chunk.title.lower() == title.lower()
        ]
        display_title = title or (chunks[0].title if len({c.title for c in chunks}) == 1 and chunks else "Library")
        if not chunks:
            return SummaryResult(display_title, "No matching book text was found.", [], 0, 0)

        candidates = self._sentence_candidates(chunks)
        selected = []
        covered_terms = set()
        character_count = 0
        for candidate in candidates:
            novelty = len(candidate["terms"] - covered_terms) / max(1, len(candidate["terms"]))
            if selected and novelty < 0.28:
                continue
            addition = len(candidate["sentence"]) + 1
            if selected and character_count + addition > max_characters:
                continue
            selected.append(candidate)
            covered_terms.update(candidate["terms"])
            character_count += addition
            if len(selected) >= max_sentences:
                break

        if not selected:
            first = chunks[0]
            text = first.text[:max_characters].strip()
            selected = [{"sentence": text, "chunk": first, "order": (first.chunk_id, 0)}]

        # Present in narrative order, while preserving source links.
        selected.sort(key=lambda item: item["order"])
        source_number = {}
        sources = []
        parts = []
        for item in selected:
            chunk = item["chunk"]
            if chunk.chunk_id not in source_number:
                number = len(sources) + 1
                source_number[chunk.chunk_id] = number
                sources.append(SummarySource(
                    number=number,
                    title=chunk.title,
                    location=chunk.location,
                    source=chunk.source,
                    excerpt=chunk.text[:320].strip() + ("..." if len(chunk.text) > 320 else ""),
                ))
            parts.append(f"{item['sentence']} [{source_number[chunk.chunk_id]}]")

        return SummaryResult(
            title=display_title,
            summary=" ".join(parts),
            sources=sources,
            sentence_count=len(selected),
            source_word_count=sum(chunk.word_count for chunk in chunks),
        )

    def available_titles(self):
        return sorted({chunk.title for chunk in self.chunks})
