"""Store rated/corrected conversations and export approved SFT data."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


class FeedbackStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    corrected_response TEXT,
                    rating INTEGER,
                    approved INTEGER NOT NULL DEFAULT 0,
                    context_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(
        self,
        prompt: str,
        response: str,
        *,
        corrected_response: str = "",
        rating: int | None = None,
        approved: bool = False,
        context: list[dict] | None = None,
    ) -> int:
        if not prompt.strip() or not response.strip():
            raise ValueError("prompt and response are required")
        if rating is not None and rating not in {-1, 0, 1}:
            raise ValueError("rating must be -1, 0, 1, or None")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback(
                    prompt, response, corrected_response, rating, approved,
                    context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt.strip(), response.strip(), corrected_response.strip(), rating,
                    int(approved), json.dumps(context or [], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def export_sft(self, path: str | Path, *, approved_only: bool = True) -> int:
        query = "SELECT * FROM feedback"
        if approved_only:
            query += " WHERE approved = 1 OR corrected_response != '' OR rating = 1"
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                answer = row["corrected_response"].strip() or row["response"].strip()
                if not answer:
                    continue
                handle.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": answer},
                    ]
                }, ensure_ascii=False) + "\n")
                count += 1
        return count
