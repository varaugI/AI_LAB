"""High-level API combining ingestion, search, replies, summaries, and memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assistant import NovelAssistant
from .backends import ChatBackend, backend_from_environment
from .knowledge_assistant import KnowledgeAssistant
from .characters import CharacterTracker
from .corpus import chunk_documents
from .document_reader import DocumentSection, LoadedDocument, read_documents
from .language_models import NGramLanguageModel
from .memory import ConversationMemory
from .retrieval import BM25Index
from .summarization import ExtractiveSummarizer


@dataclass
class LearningReport:
    documents: int
    sections: int
    chunks: int
    words: int


class NovelLearningSystem:
    """High-level mixed-document learning and conversational retrieval system.

    The historical class name is retained for compatibility. It now supports
    novels, textbooks, law material, programming references, notes and data
    files. Imported content is indexed for retrieval; an optional local LLM
    backend turns retrieved passages into natural ChatGPT-like replies.
    """

    def __init__(
        self,
        index: BM25Index | None = None,
        memory: ConversationMemory | None = None,
        backend: ChatBackend | None = None,
        use_environment_backend: bool = True,
    ):
        self.index = index or BM25Index()
        self.documents: list[LoadedDocument] = []
        self.style_model: NGramLanguageModel | None = None
        self.memory = memory or ConversationMemory()
        self.backend = backend
        if self.backend is None and use_environment_backend:
            self.backend = backend_from_environment()
        self.assistant = NovelAssistant(self.index)
        self.knowledge_assistant = KnowledgeAssistant(self.index, self.memory, self.backend)
        self._character_tracker: CharacterTracker | None = None

    def _refresh_helpers(self):
        self.assistant = NovelAssistant(self.index)
        self.knowledge_assistant = KnowledgeAssistant(self.index, self.memory, self.backend)
        self._character_tracker = None

    def set_backend(self, backend: ChatBackend | None):
        self.backend = backend
        self._refresh_helpers()
        return self

    def learn_documents(
        self,
        documents: list[LoadedDocument],
        append: bool = True,
        max_words: int = 180,
        overlap_words: int = 35,
        minimum_words: int = 8,
        domain: str | None = None,
    ) -> LearningReport:
        if not documents:
            raise ValueError("No readable documents were supplied.")
        if domain and domain != "auto":
            for document in documents:
                document.domain = domain
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
        all_chunks = existing_chunks + new_chunks
        for chunk_id, chunk in enumerate(all_chunks):
            chunk.chunk_id = chunk_id
        self.index.build(all_chunks)
        self._refresh_helpers()
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

    def learn_text(
        self, text: str, title="User text", source="memory://user-text",
        append=True, **chunk_options
    ):
        document = LoadedDocument(
            path=source,
            title=title,
            kind="text",
            sections=[DocumentSection(source, title, "document", text)],
        )
        return self.learn_documents([document], append=append, **chunk_options)

    def ask(self, question: str, mode: str = "chat", **options):
        return self.knowledge_assistant.answer(question, mode=mode, **options)

    def chat(self, question: str, remember: bool = True, mode: str = "chat", **options):
        """Answer naturally with retrieval and recent-turn context."""
        expanded_query = self.memory.expand_query(question)
        reply = self.knowledge_assistant.answer(
            question, mode=mode, search_query=expanded_query, **options
        )
        if remember:
            self.memory.add(
                question,
                reply.answer,
                [
                    {"title": source.title, "location": source.location, "source": source.source}
                    for source in reply.sources
                ],
            )
        return reply

    def search(self, query: str, limit=5, **filters):
        return self.index.search(query, limit=limit, **filters)

    def domains(self):
        return sorted({getattr(chunk, "domain", "general") for chunk in self.index.chunks})

    def library_stats(self):
        domains = {}
        kinds = {}
        for chunk in self.index.chunks:
            domain = getattr(chunk, "domain", "general")
            kind = getattr(chunk, "kind", "text")
            domains[domain] = domains.get(domain, 0) + 1
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "chunks": len(self.index.chunks),
            "titles": self.titles(),
            "domains": domains,
            "kinds": kinds,
            "backend": getattr(self.backend, "name", "extractive") if self.backend else "extractive",
            "model": getattr(self.backend, "model", "") if self.backend else "",
        }

    def summarize(self, title=None, max_sentences=8, max_characters=3000):
        return ExtractiveSummarizer(self.index.chunks).summarize(
            title=title,
            max_sentences=max_sentences,
            max_characters=max_characters,
        )

    def titles(self):
        return ExtractiveSummarizer(self.index.chunks).available_titles()

    def analyze_characters(self, minimum_mentions=2, refresh=False):
        if refresh or self._character_tracker is None:
            self._character_tracker = CharacterTracker().analyze(
                self.index.chunks,
                minimum_mentions=minimum_mentions,
            )
        return self._character_tracker

    def character(self, name: str, minimum_mentions=1):
        return self.analyze_characters(minimum_mentions=minimum_mentions).get(name)

    def train_style_model(self, order=4):
        texts = [section.text for document in self.documents for section in document.sections]
        if not texts:
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
    def load_library(cls, filename, memory_file=None):
        memory = None
        if memory_file and Path(memory_file).exists():
            try:
                memory = ConversationMemory.load(memory_file)
            except (ValueError, OSError):
                memory = ConversationMemory()
        return cls(BM25Index.load(filename), memory=memory)

    def save_memory(self, filename):
        self.memory.save(filename)

    def load_memory(self, filename):
        self.memory = ConversationMemory.load(filename)
        return self.memory

    def clear_memory(self):
        self.memory.clear()

    def save_style_model(self, filename):
        if self.style_model is None:
            raise RuntimeError("No style model has been trained.")
        self.style_model.save(filename)

    def load_style_model(self, filename):
        self.style_model = NGramLanguageModel.load(filename)
        return self.style_model


# Clearer modern name; the old name remains supported.
KnowledgeLearningSystem = NovelLearningSystem
