"""Train AI LAB's byte-level BPE tokenizer on a JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from builder.llm.data import iter_jsonl_records
from builder.llm.tokenizer import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="JSONL created by build_training_corpus")
    parser.add_argument("--output", default="data/tokenizers/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--max-training-bytes", type=int, default=None)
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer().train(
        (record.text for record in iter_jsonl_records(args.corpus)),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_training_bytes=args.max_training_bytes,
        verbose=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    print(json.dumps({"output": str(output), "vocab_size": tokenizer.vocab_size}, indent=2))


if __name__ == "__main__":
    main()
