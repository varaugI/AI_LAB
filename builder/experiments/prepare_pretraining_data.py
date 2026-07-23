"""Tokenize corpus JSONL into memory-mapped train.bin and val.bin streams."""

from __future__ import annotations

import argparse
import json

from builder.llm import ByteBPETokenizer, prepare_binary_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus")
    parser.add_argument("--tokenizer", default="data/tokenizers/tokenizer.json")
    parser.add_argument("--output", default="data/datasets/pretraining")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = prepare_binary_dataset(
        args.corpus,
        ByteBPETokenizer.load(args.tokenizer),
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
