"""Extract PDF/EPUB/DOCX/text/code files into deduplicated pretraining JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from builder.books import read_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Files or folders to extract")
    parser.add_argument("--output", default="data/datasets/corpus.jsonl")
    parser.add_argument("--domain", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--ocr-scanned", action="store_true")
    args = parser.parse_args()

    documents = read_documents(
        args.paths,
        recursive=args.recursive,
        ocr_scanned=args.ocr_scanned,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records = 0
    characters = 0
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            domain = document.domain if args.domain == "auto" else args.domain
            for section in document.sections:
                text = section.text.strip()
                normalized = " ".join(text.split()).casefold()
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if not text or digest in seen:
                    continue
                seen.add(digest)
                handle.write(json.dumps({
                    "text": text,
                    "title": document.title,
                    "source": document.path,
                    "location": section.location,
                    "domain": domain,
                    "kind": document.kind,
                    "sha256": digest,
                }, ensure_ascii=False) + "\n")
                records += 1
                characters += len(text)
    print(json.dumps({
        "output": str(destination),
        "documents": len(documents),
        "records": records,
        "characters": characters,
    }, indent=2))


if __name__ == "__main__":
    main()
