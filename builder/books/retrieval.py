"""A dependency-free BM25 index for searching imported books."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .corpus import TextChunk, tokenize


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float
    matched_terms: list[str]


class BM25Index:
    def __init__(self, chunks=None, k1=1.5, b=0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")
        self.k1 = float(k1)
        self.b = float(b)
        self.chunks: list[TextChunk] = []
        self.term_frequencies: list[dict[str, int]] = []
        self.document_frequencies: dict[str, int] = {}
        self.document_lengths: list[int] = []
        self.average_document_length = 0.0
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[TextChunk]):
        if not chunks:
            raise ValueError("Cannot build an index without chunks.")
        self.chunks = list(chunks)
        self.term_frequencies = []
        self.document_frequencies = {}
        self.document_lengths = []

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            frequencies = Counter(tokens)
            self.term_frequencies.append(dict(frequencies))
            self.document_lengths.append(len(tokens))
            for term in frequencies:
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths else 0.0
        )
        return self

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        frequency = self.document_frequencies.get(term, 0)
        # Robertson/Sparck Jones IDF with +1 for stable positive scores.
        return math.log(1.0 + (n - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        query: str,
        limit: int = 5,
        minimum_score: float = 0.0,
        domain: str | None = None,
        titles: list[str] | set[str] | None = None,
    ) -> list[SearchResult]:
        """Search with BM25 plus small phrase/title boosts and metadata filters."""
        if limit <= 0:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not self.chunks:
            return []

        normalized_query = " ".join(query_terms)
        title_filter = {item.lower() for item in titles or []}
        results: list[SearchResult] = []
        average = self.average_document_length or 1.0
        for index, chunk in enumerate(self.chunks):
            if domain and domain != "all" and getattr(chunk, "domain", "general") != domain:
                continue
            if title_filter and chunk.title.lower() not in title_filter:
                continue

            length = self.document_lengths[index]
            frequencies = self.term_frequencies[index]
            score = 0.0
            matched = []
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                matched.append(term)
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / average
                )
                score += self._idf(term) * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )

            lowered_text = " ".join(tokenize(chunk.text))
            if len(query_terms) >= 2 and normalized_query in lowered_text:
                score += 2.5
            lowered_title = chunk.title.lower()
            title_matches = sum(1 for term in query_terms if term in lowered_title)
            score += 0.35 * title_matches

            if score > minimum_score:
                results.append(SearchResult(chunk, score, matched))

        results.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return results[:limit]

    def to_dict(self):
        return {
            "version": 2,
            "type": "bm25_book_index",
            "k1": self.k1,
            "b": self.b,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "term_frequencies": self.term_frequencies,
            "document_frequencies": self.document_frequencies,
            "document_lengths": self.document_lengths,
            "average_document_length": self.average_document_length,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "bm25_book_index":
            raise ValueError("Not an AI LAB book index.")
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.chunks = [TextChunk.from_dict(item) for item in data["chunks"]]
        index.term_frequencies = data["term_frequencies"]
        index.document_frequencies = data["document_frequencies"]
        index.document_lengths = data["document_lengths"]
        index.average_document_length = data["average_document_length"]
        return index

    def save(self, filename: str | Path):
        Path(filename).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filename: str | Path):
        return cls.from_dict(json.loads(Path(filename).read_text(encoding="utf-8")))
