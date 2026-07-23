"""Build a searchable AI LAB knowledge index from TXT, PDF, and EPUB books."""

from __future__ import annotations

import argparse
from pathlib import Path

from builder.books import BM25Index, chunk_documents, read_documents


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read novels and create a searchable book-memory index."
    )
    parser.add_argument("paths", nargs="+", help="Book files or directories")
    parser.add_argument("--output", default="novel_library.json", help="Output index JSON")
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=35)
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages per PDF")
    parser.add_argument("--max-sections", type=int, default=None, help="Limit chapters per EPUB")
    parser.add_argument("--ocr-scanned", action="store_true", help="OCR PDF pages with little/no text")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    documents = read_documents(
        args.paths,
        recursive=not args.no_recursive,
        max_pages=args.max_pages,
        max_sections=args.max_sections,
        ocr_scanned=args.ocr_scanned,
    )
    chunks = chunk_documents(
        documents,
        max_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    index = BM25Index(chunks)
    index.save(args.output)

    print(f"Imported documents: {len(documents)}")
    print(f"Searchable chunks: {len(chunks)}")
    print(f"Total words: {sum(document.word_count for document in documents):,}")
    print(f"Saved book memory: {Path(args.output).resolve()}")
    for document in documents:
        print(f"- {document.title} ({document.kind}, {document.word_count:,} words)")


if __name__ == "__main__":
    main()
