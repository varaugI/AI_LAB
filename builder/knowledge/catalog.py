"""SQLite-backed knowledge library with duplicate detection and true deletion."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable

from builder.books.corpus import chunk_documents, tokenize
from builder.books.document_reader import infer_domain, read_document


@dataclass
class DocumentRecord:
    id: int
    sha256: str
    original_name: str
    stored_path: str
    title: str
    domain: str
    kind: str
    size_bytes: int
    chunk_count: int
    added_at: str
    content_sha256: str | None = None


@dataclass
class ImportResult:
    status: str
    document: DocumentRecord
    message: str


@dataclass
class SearchHit:
    chunk_id: int
    document_id: int
    title: str
    location: str
    text: str
    domain: str
    kind: str
    source: str
    score: float


class DocumentCatalog:
    def __init__(self, database_path: str | Path, upload_dir: str | Path) -> None:
        self.database_path = Path(database_path)
        self.upload_dir = Path(upload_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._fts_available = True
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    added_at TEXT NOT NULL,
                    content_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    location TEXT NOT NULL,
                    text TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    UNIQUE(document_id, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(domain);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            if "content_sha256" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN content_sha256 TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash "
                "ON documents(content_sha256) WHERE content_sha256 IS NOT NULL"
            )
            try:
                connection.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        title, location, text, domain, kind,
                        content='chunks', content_rowid='id',
                        tokenize='unicode61 remove_diacritics 2'
                    );
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, title, location, text, domain, kind)
                        VALUES (new.id, new.title, new.location, new.text, new.domain, new.kind);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, title, location, text, domain, kind)
                        VALUES ('delete', old.id, old.title, old.location, old.text, old.domain, old.kind);
                    END;
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, title, location, text, domain, kind)
                        VALUES ('delete', old.id, old.title, old.location, old.text, old.domain, old.kind);
                        INSERT INTO chunks_fts(rowid, title, location, text, domain, kind)
                        VALUES (new.id, new.title, new.location, new.text, new.domain, new.kind);
                    END;
                    """
                )
            except sqlite3.OperationalError:
                self._fts_available = False

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(**dict(row))

    def get_document(self, document_id: int) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (int(document_id),)).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_content_hash(self, content_sha256: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, domain: str | None = None) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        values: tuple = ()
        if domain and domain != "all":
            query += " WHERE domain = ?"
            values = (domain,)
        query += " ORDER BY added_at DESC, id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row_to_document(row) for row in rows]

    def import_file(
        self,
        path: str | Path,
        *,
        domain: str = "auto",
        copy_file: bool = True,
        replace_same_name: bool = False,
        max_words: int = 220,
        overlap_words: int = 40,
        ocr_scanned: bool = False,
    ) -> ImportResult:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        sha256 = self._hash_file(source)
        duplicate = self.find_by_hash(sha256)
        if duplicate:
            return ImportResult(
                status="duplicate",
                document=duplicate,
                message=f"Exact duplicate skipped; already stored as {duplicate.original_name}.",
            )
        if replace_same_name:
            for existing in self.list_documents():
                if existing.original_name.casefold() == source.name.casefold():
                    self.delete_document(existing.id, delete_file=True)

        loaded = read_document(source, ocr_scanned=ocr_scanned)
        normalized_content = " ".join(loaded.text.split()).casefold()
        content_sha256 = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()
        content_duplicate = self.find_by_content_hash(content_sha256)
        if content_duplicate:
            return ImportResult(
                status="duplicate",
                document=content_duplicate,
                message=(
                    "Duplicate text skipped even though the file bytes differ; "
                    f"already stored as {content_duplicate.original_name}."
                ),
            )
        if domain not in {"", "auto"}:
            loaded.domain = domain
        elif not getattr(loaded, "domain", None):
            sample = "\n".join(section.text[:4000] for section in loaded.sections)
            loaded.domain = infer_domain(loaded.title, sample, source.suffix)
        chunks = chunk_documents(
            [loaded], max_words=max_words, overlap_words=overlap_words, minimum_words=1
        )
        if not chunks:
            raise ValueError(f"No searchable text was extracted from {source.name}")

        stored_path = source
        if copy_file:
            destination = self.upload_dir / f"{sha256[:12]}_{source.name}"
            if source != destination:
                shutil.copy2(source, destination)
            stored_path = destination
        added_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO documents(
                    sha256, original_name, stored_path, title, domain, kind,
                    size_bytes, chunk_count, added_at, content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sha256, source.name, str(stored_path), loaded.title,
                    loaded.domain, loaded.kind, source.stat().st_size,
                    len(chunks), added_at, content_sha256,
                ),
            )
            document_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO chunks(
                    document_id, chunk_index, title, location, text, domain,
                    kind, source, token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id, index, chunk.title, chunk.location, chunk.text,
                        chunk.domain, chunk.kind, str(stored_path), len(tokenize(chunk.text)),
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )
        record = self.get_document(document_id)
        assert record is not None
        return ImportResult("added", record, f"Imported {record.original_name} with {record.chunk_count} chunks.")

    def update_document_metadata(
        self,
        document_id: int,
        *,
        title: str | None = None,
        domain: str | None = None,
        kind: str | None = None,
    ) -> DocumentRecord:
        record = self.get_document(document_id)
        if record is None:
            raise KeyError(document_id)
        next_title = title.strip() if title and title.strip() else record.title
        next_domain = domain.strip() if domain and domain.strip() else record.domain
        next_kind = kind.strip() if kind and kind.strip() else record.kind
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET title = ?, domain = ?, kind = ? WHERE id = ?",
                (next_title, next_domain, next_kind, int(document_id)),
            )
            connection.execute(
                "UPDATE chunks SET title = ?, domain = ?, kind = ? WHERE document_id = ?",
                (next_title, next_domain, next_kind, int(document_id)),
            )
        updated = self.get_document(document_id)
        assert updated is not None
        return updated

    def delete_document(self, document_id: int, *, delete_file: bool = True) -> bool:
        record = self.get_document(document_id)
        if record is None:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE id = ?", (int(document_id),))
        if delete_file:
            path = Path(record.stored_path)
            try:
                if path.is_file() and self.upload_dir.resolve() in path.resolve().parents:
                    path.unlink()
            except OSError:
                pass
        return True

    def clear(self, *, delete_files: bool = True) -> None:
        records = self.list_documents()
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")
        if delete_files:
            for record in records:
                try:
                    path = Path(record.stored_path)
                    if path.is_file() and self.upload_dir.resolve() in path.resolve().parents:
                        path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _fts_query(query: str) -> str:
        words = [word.replace('"', "") for word in tokenize(query)]
        return " OR ".join(f'"{word}"' for word in words)

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        domain: str | None = None,
        document_ids: Iterable[int] | None = None,
    ) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        document_ids = list(document_ids or [])
        filters = []
        values: list = []
        if domain and domain != "all":
            filters.append("c.domain = ?")
            values.append(domain)
        if document_ids:
            filters.append("c.document_id IN (" + ",".join("?" for _ in document_ids) + ")")
            values.extend(document_ids)
        where_suffix = (" AND " + " AND ".join(filters)) if filters else ""

        with self._connect() as connection:
            if self._fts_available:
                fts_query = self._fts_query(query)
                if fts_query:
                    try:
                        rows = connection.execute(
                            f"""
                            SELECT c.id AS chunk_id, c.document_id, c.title, c.location,
                                   c.text, c.domain, c.kind, c.source,
                                   -bm25(chunks_fts, 2.0, 0.5, 1.0, 0.2, 0.2) AS score
                            FROM chunks_fts
                            JOIN chunks c ON c.id = chunks_fts.rowid
                            WHERE chunks_fts MATCH ? {where_suffix}
                            ORDER BY score DESC
                            LIMIT ?
                            """,
                            [fts_query, *values, int(limit)],
                        ).fetchall()
                        if rows:
                            return [SearchHit(**dict(row)) for row in rows]
                    except sqlite3.OperationalError:
                        pass
            # Portable fallback: score by matched query terms.
            terms = tokenize(query)
            rows = connection.execute(
                "SELECT c.* FROM chunks c WHERE 1=1" + where_suffix,
                values,
            ).fetchall()
        scored = []
        for row in rows:
            haystack = f"{row['title']} {row['text']}".casefold()
            score = sum(haystack.count(term.casefold()) for term in terms)
            if score:
                scored.append(SearchHit(
                    chunk_id=row["id"], document_id=row["document_id"],
                    title=row["title"], location=row["location"], text=row["text"],
                    domain=row["domain"], kind=row["kind"], source=row["source"],
                    score=float(score),
                ))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


    def get_chunks_by_ids(self, chunk_ids: Iterable[int]) -> dict[int, SearchHit]:
        ids = [int(value) for value in chunk_ids]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
            ).fetchall()
        return {
            row["id"]: SearchHit(
                chunk_id=row["id"], document_id=row["document_id"],
                title=row["title"], location=row["location"], text=row["text"],
                domain=row["domain"], kind=row["kind"], source=row["source"],
                score=0.0,
            )
            for row in rows
        }

    def get_chunks(
        self,
        *,
        document_id: int | None = None,
        domain: str | None = None,
        limit: int | None = None,
    ) -> list[SearchHit]:
        filters = []
        values: list = []
        if document_id is not None:
            filters.append("document_id = ?")
            values.append(int(document_id))
        if domain and domain != "all":
            filters.append("domain = ?")
            values.append(domain)
        query = "SELECT * FROM chunks"
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY document_id, chunk_index"
        if limit is not None:
            query += " LIMIT ?"
            values.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [
            SearchHit(
                chunk_id=row["id"], document_id=row["document_id"],
                title=row["title"], location=row["location"], text=row["text"],
                domain=row["domain"], kind=row["kind"], source=row["source"],
                score=1.0,
            )
            for row in rows
        ]

    def export_corpus(self, path: str | Path, *, domain: str | None = None) -> int:
        """Export original document sections for training, avoiding chunk overlap."""
        records = self.list_documents(None if domain in {None, "all"} else domain)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        exported_documents = 0
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                source_path = Path(record.stored_path)
                wrote = False
                if source_path.is_file():
                    try:
                        document = read_document(source_path)
                        for section in document.sections:
                            text = section.text.strip()
                            if not text:
                                continue
                            handle.write(json.dumps({
                                "text": text,
                                "title": record.title,
                                "source": record.stored_path,
                                "location": section.location,
                                "domain": record.domain,
                                "kind": record.kind,
                                "document_id": record.id,
                                "document_sha256": record.sha256,
                            }, ensure_ascii=False) + "\n")
                            wrote = True
                    except Exception:
                        wrote = False
                if not wrote:
                    chunks = self.get_chunks(document_id=record.id)
                    if chunks:
                        handle.write(json.dumps({
                            "text": "\n\n".join(chunk.text for chunk in chunks),
                            "title": record.title,
                            "source": record.stored_path,
                            "domain": record.domain,
                            "kind": record.kind,
                            "document_id": record.id,
                            "document_sha256": record.sha256,
                        }, ensure_ascii=False) + "\n")
                        wrote = True
                if wrote:
                    exported_documents += 1
        return exported_documents

    def stats(self) -> dict:
        with self._connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            rows = connection.execute(
                "SELECT domain, COUNT(*) AS count FROM documents GROUP BY domain"
            ).fetchall()
        return {"documents": documents, "chunks": chunks, "domains": {row[0]: row[1] for row in rows}}
