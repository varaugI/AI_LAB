"""Optional semantic and hybrid retrieval using sentence-transformers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .catalog import DocumentCatalog, SearchHit


class SemanticIndex:
    def __init__(
        self,
        catalog: DocumentCatalog,
        index_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.catalog = catalog
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".json")
        self.model_name = model_name
        self._model = None
        self._embeddings = None
        self._chunk_ids = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install requirements-retrieval.txt to use semantic search."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def rebuild(self, batch_size: int = 64) -> dict:
        chunks = self.catalog.get_chunks()
        if not chunks:
            raise ValueError("The catalog has no chunks to embed")
        model = self._load_model()
        embeddings = model.encode(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        chunk_ids = np.asarray([chunk.chunk_id for chunk in chunks], dtype=np.int64)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_path, embeddings=embeddings, chunk_ids=chunk_ids)
        metadata = {
            "model_name": self.model_name,
            "chunks": len(chunks),
            "catalog_stats": self.catalog.stats(),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._embeddings, self._chunk_ids = embeddings, chunk_ids
        return metadata

    def _load(self) -> None:
        if self._embeddings is not None:
            return
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        payload = np.load(self.index_path)
        self._embeddings = payload["embeddings"]
        self._chunk_ids = payload["chunk_ids"]

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        domain: str | None = None,
        document_ids: list[int] | None = None,
    ) -> list[SearchHit]:
        self._load()
        model = self._load_model()
        query_vector = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        scores = self._embeddings @ query_vector
        candidate_count = min(len(scores), max(limit * 8, limit))
        indexes = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        indexes = indexes[np.argsort(-scores[indexes])]
        ids = [int(self._chunk_ids[index]) for index in indexes]
        chunks = self.catalog.get_chunks_by_ids(ids)
        allowed_documents = set(document_ids or [])
        results = []
        for index in indexes:
            chunk_id = int(self._chunk_ids[index])
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            if domain and domain != "all" and chunk.domain != domain:
                continue
            if allowed_documents and chunk.document_id not in allowed_documents:
                continue
            results.append(replace(chunk, score=float(scores[index])))
            if len(results) >= limit:
                break
        return results


class HybridRetriever:
    def __init__(
        self,
        catalog: DocumentCatalog,
        semantic_index: SemanticIndex | None = None,
        lexical_weight: float = 0.55,
        semantic_weight: float = 0.45,
    ) -> None:
        self.catalog = catalog
        self.semantic_index = semantic_index
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def search(self, query: str, *, limit: int = 6, **filters) -> list[SearchHit]:
        lexical = self.catalog.search(query, limit=max(limit * 3, 12), **filters)
        semantic = []
        if self.semantic_index is not None:
            try:
                semantic = self.semantic_index.search(
                    query, limit=max(limit * 3, 12), **filters
                )
            except (FileNotFoundError, RuntimeError):
                semantic = []
        combined: dict[int, tuple[SearchHit, float]] = {}
        for rank, hit in enumerate(lexical, start=1):
            combined[hit.chunk_id] = (hit, self.lexical_weight / (60 + rank))
        for rank, hit in enumerate(semantic, start=1):
            previous = combined.get(hit.chunk_id)
            score = self.semantic_weight / (60 + rank)
            if previous:
                combined[hit.chunk_id] = (previous[0], previous[1] + score)
            else:
                combined[hit.chunk_id] = (hit, score)
        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return [replace(hit, score=score) for hit, score in ranked[:limit]]
