"""Convert loaded novels into overlapping, searchable text chunks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from .document_reader import LoadedDocument, normalize_text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in TOKEN_PATTERN.finditer(text)]


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text).replace("\n", " ")
    if not text:
        return []
    return [piece.strip() for piece in SENTENCE_PATTERN.split(text) if piece.strip()]


@dataclass
class TextChunk:
    chunk_id: int
    source: str
    title: str
    location: str
    text: str
    word_count: int
    domain: str = "general"
    kind: str = "text"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def _window_words(text: str, max_words: int, overlap_words: int):
    words = text.split()
    if not words:
        return
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        window = words[start:start + max_words]
        if not window:
            continue
        yield start, " ".join(window)
        if start + max_words >= len(words):
            break


def chunk_documents(
    documents: list[LoadedDocument],
    max_words: int = 180,
    overlap_words: int = 35,
    minimum_words: int = 8,
) -> list[TextChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be >= 0 and smaller than max_words.")
    if minimum_words <= 0:
        raise ValueError("minimum_words must be positive.")

    chunks: list[TextChunk] = []
    next_id = 0
    for document in documents:
        for section in document.sections:
            section_chunk_start = len(chunks)
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section.text) if part.strip()]
            if not paragraphs:
                paragraphs = [section.text]

            # Combine short neighbouring paragraphs before making windows.
            buffer: list[str] = []
            buffer_count = 0
            groups: list[str] = []
            for paragraph in paragraphs:
                count = len(paragraph.split())
                if buffer and buffer_count + count > max_words:
                    groups.append("\n\n".join(buffer))
                    buffer = []
                    buffer_count = 0
                buffer.append(paragraph)
                buffer_count += count
            if buffer:
                groups.append("\n\n".join(buffer))

            for group_number, group in enumerate(groups, start=1):
                for word_offset, text in _window_words(group, max_words, overlap_words):
                    count = len(text.split())
                    if count < minimum_words and len(chunks) > section_chunk_start:
                        # Skip only a tiny trailing fragment from this same section.
                        # A short standalone note or source file must remain searchable.
                        continue
                    chunks.append(TextChunk(
                        chunk_id=next_id,
                        source=section.source,
                        title=section.title,
                        location=f"{section.location}, part {group_number}, word {word_offset + 1}",
                        text=text,
                        word_count=count,
                        domain=getattr(document, "domain", "general"),
                        kind=getattr(document, "kind", "text"),
                    ))
                    next_id += 1
    return chunks
