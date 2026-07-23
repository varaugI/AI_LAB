"""Build a mixed-domain AI LAB knowledge library."""

from __future__ import annotations

import argparse
from pathlib import Path

from builder.books import KnowledgeLearningSystem


DOMAINS = ("auto", "general", "fiction", "school", "law", "programming")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Import books, documents, tables, notes, and source-code files."
    )
    parser.add_argument("paths", nargs="+", help="Files or directories")
    parser.add_argument("--output", default="knowledge_library.json")
    parser.add_argument("--domain", choices=DOMAINS, default="auto")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=45)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument("--ocr-scanned", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    system = KnowledgeLearningSystem(use_environment_backend=False)
    report = system.learn_files(
        args.paths,
        append=False,
        domain=args.domain,
        max_words=args.chunk_words,
        overlap_words=args.overlap_words,
        recursive=not args.no_recursive,
        max_pages=args.max_pages,
        max_sections=args.max_sections,
        ocr_scanned=args.ocr_scanned,
    )
    system.save_library(args.output)
    print(f"Imported documents: {report.documents}")
    print(f"Sections: {report.sections}")
    print(f"Searchable passages: {report.chunks}")
    print(f"Total words: {report.words:,}")
    print(f"Domains: {system.library_stats()['domains']}")
    print(f"Saved library: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
