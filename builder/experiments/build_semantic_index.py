"""Build an optional sentence-transformer embedding index for hybrid retrieval."""

from __future__ import annotations

import argparse
import json

from builder.knowledge import DocumentCatalog
from builder.knowledge.semantic import SemanticIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/runtime/library.sqlite3")
    parser.add_argument("--uploads", default="data/runtime/uploads")
    parser.add_argument("--output", default="data/runtime/semantic_index.npz")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    catalog = DocumentCatalog(args.database, args.uploads)
    metadata = SemanticIndex(catalog, args.output, args.model).rebuild(args.batch_size)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
