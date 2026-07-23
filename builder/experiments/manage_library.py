"""Import, list, search, delete, and export the persistent knowledge library."""

from __future__ import annotations

import argparse
import json

from builder.knowledge import DocumentCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/runtime/library.sqlite3")
    parser.add_argument("--uploads", default="data/runtime/uploads")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("paths", nargs="+")
    add.add_argument("--domain", default="auto")
    add.add_argument("--replace-same-name", action="store_true")
    sub.add_parser("list")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--domain", default="all")
    delete = sub.add_parser("delete")
    delete.add_argument("document_id", type=int)
    export = sub.add_parser("export-corpus")
    export.add_argument("output")
    export.add_argument("--domain", default="all")
    args = parser.parse_args()

    catalog = DocumentCatalog(args.database, args.uploads)
    if args.command == "add":
        results = [catalog.import_file(
            path, domain=args.domain, replace_same_name=args.replace_same_name
        ) for path in args.paths]
        print(json.dumps([
            {"status": item.status, "message": item.message, "document": item.document.__dict__}
            for item in results
        ], indent=2))
    elif args.command == "list":
        print(json.dumps([item.__dict__ for item in catalog.list_documents()], indent=2))
    elif args.command == "search":
        print(json.dumps([
            item.__dict__ for item in catalog.search(args.query, domain=args.domain)
        ], indent=2))
    elif args.command == "delete":
        print(json.dumps({"deleted": catalog.delete_document(args.document_id)}, indent=2))
    elif args.command == "export-corpus":
        count = catalog.export_corpus(args.output, domain=args.domain)
        print(json.dumps({"documents": count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
