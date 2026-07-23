"""High-level API combining document ingestion, search, replies, and generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assistant import NovelAssistant
from .corpus import TextChunk, chunk_documents
from .document_reader import DocumentSection, LoadedDocument, read_documents
from .language_models import NGramLanguageModel
from .retrieval import BM25Index


@dataclass
class LearningReport:
    documents: int
    sections: int
    chunks: int
    words: int


class NovelLearningSystem:
    """Convenient façade for the complete novel-learning workflow.

    ``learn_files`` imports books and rebuilds searchable memory. ``ask`` uses
    grounded retrieval. ``train_style_model`` optionally learns an n-gram model
    for text continuation. The two abilities are deliberately separate so a
    creative generator cannot silently invent facts in question answers.
    """

    def __init__(self, index: BM25Index | None = None):
        self.index = index or BM25Index()
        self.assistant = NovelAssistant(self.index)
        self.documents: list[LoadedDocument] = []
        self.style_model: NGramLanguageModel | None = None

    def learn_documents(
        self,
        documents: list[LoadedDocument],
        append: bool = True,
        max_words: int = 180,
        overlap_words: int = 35,
        minimum_words: int = 8,
    ) -> LearningReport:
        if not append:
            self.documents = []
            existing_chunks = []
        else:
            existing_chunks = list(self.index.chunks)
        self.documents.extend(documents)
        new_chunks = chunk_documents(
            documents,
            max_words=max_words,
            overlap_words=overlap_words,
            minimum_words=minimum_words,
        )
        # Re-number chunks because IDs are used as stable keys within one index.
        all_chunks = existing_chunks + new_chunks
        for chunk_id, chunk in enumerate(all_chunks):
            chunk.chunk_id = chunk_id
        self.index.build(all_chunks)
        self.assistant = NovelAssistant(self.index)
        return LearningReport(
            documents=len(documents),
            sections=sum(len(document.sections) for document in documents),
            chunks=len(new_chunks),
            words=sum(document.word_count for document in documents),
        )

    def learn_files(self, paths, **options) -> LearningReport:
        reader_keys = {"recursive", "max_pages", "max_sections", "ocr_scanned"}
        reader_options = {
            key: options.pop(key) for key in list(options) if key in reader_keys
        }
        documents = read_documents(list(paths), **reader_options)
        return self.learn_documents(documents, **options)

    def learn_text(self, text: str, title="User text", source="memory://user-text", append=True):
        document = LoadedDocument(
            path=source,
            title=title,
            kind="text",
            sections=[DocumentSection(source, title, "document", text)],
        )
        return self.learn_documents([document], append=append)

    def ask(self, question: str, **options):
        return self.assistant.answer(question, **options)

    def search(self, query: str, limit=5):
        return self.index.search(query, limit=limit)

    def train_style_model(self, order=4):
        texts = [section.text for document in self.documents for section in document.sections]
        if not texts:
            # A loaded index may not retain original LoadedDocument objects, but
            # its chunks still provide enough text for a style experiment.
            texts = [chunk.text for chunk in self.index.chunks]
        if not texts:
            raise RuntimeError("Learn or load at least one book first.")
        self.style_model = NGramLanguageModel(order=order).train(texts)
        return self.style_model

    def continue_text(self, seed_text="", **options):
        if self.style_model is None:
            raise RuntimeError("Call train_style_model() or load a style model first.")
        return self.style_model.generate(seed_text=seed_text, **options)

    def save_library(self, filename):
        self.index.save(filename)

    @classmethod
    def load_library(cls, filename):
        return cls(BM25Index.load(filename))

    def save_style_model(self, filename):
        if self.style_model is None:
            raise RuntimeError("No style model has been trained.")
        self.style_model.save(filename)

    def load_style_model(self, filename):
        self.style_model = NGramLanguageModel.load(filename)
        return self.style_model
