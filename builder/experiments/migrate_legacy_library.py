"""Migrate an old knowledge_library.json into the SQLite document catalog."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

from builder.knowledge import DocumentCatalog


def merge_overlapping_chunks(chunks: list[dict]) -> str:
    chunks = sorted(chunks, key=lambda item: int(item.get("chunk_id", 0)))
    merged: list[str] = []
    for chunk in chunks:
        words = str(chunk.get("text", "")).split()
        if not words:
            continue
        if not merged:
            merged.extend(words)
            continue
        maximum = min(120, len(merged), len(words))
        overlap = 0
        for size in range(maximum, 0, -1):
            if merged[-size:] == words[:size]:
                overlap = size
                break
        merged.extend(words[overlap:])
    return " ".join(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_json")
    parser.add_argument("--database", default="data/runtime/library.sqlite3")
    parser.add_argument("--uploads", default="data/runtime/uploads")
    args = parser.parse_args()

    payload = json.loads(Path(args.legacy_json).read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for chunk in payload.get("chunks", []):
        key = (chunk.get("source", ""), chunk.get("title", "Untitled"), chunk.get("domain", "general"))
        groups[key].append(chunk)
    catalog = DocumentCatalog(args.database, args.uploads)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, ((source, title, domain), chunks) in enumerate(groups.items(), start=1):
            text = merge_overlapping_chunks(chunks)
            safe = "".join(character if character.isalnum() else "_" for character in title).strip("_") or f"document_{index}"
            path = root / f"{safe[:80]}_{index}.txt"
            path.write_text(text, encoding="utf-8")
            result = catalog.import_file(path, domain=domain)
            if result.status == "added":
                updated = catalog.update_document_metadata(
                    result.document.id, title=title, domain=domain,
                    kind=str(chunks[0].get("kind", "text")) if chunks else "text",
                )
                result.document = updated
            results.append({"status": result.status, "title": title, "message": result.message})
    print(json.dumps({"migrated": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
