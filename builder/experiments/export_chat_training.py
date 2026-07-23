"""Export positively rated or corrected chats as supervised fine-tuning JSONL."""

from __future__ import annotations

import argparse
import json

from builder.knowledge import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/runtime/feedback.sqlite3")
    parser.add_argument("--output", default="data/datasets/approved_chats.jsonl")
    parser.add_argument("--all", action="store_true", help="Include unapproved conversations")
    args = parser.parse_args()
    count = FeedbackStore(args.database).export_sft(args.output, approved_only=not args.all)
    print(json.dumps({"examples": count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
