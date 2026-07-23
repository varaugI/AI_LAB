"""Generate a short continuation from a saved AI LAB language model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from builder.books import NGramLanguageModel, TinyNeuralLanguageModel


def build_parser():
    parser = argparse.ArgumentParser(description="Generate text from a learned novel model.")
    parser.add_argument("model")
    parser.add_argument("--seed-text", default="")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = json.loads(Path(args.model).read_text(encoding="utf-8"))
    model_type = data.get("type")
    if model_type == "ngram_language_model":
        model = NGramLanguageModel.from_dict(data)
    elif model_type == "tiny_neural_language_model":
        model = TinyNeuralLanguageModel.load(args.model)
    else:
        raise ValueError("Unknown AI LAB language model file.")
    print(model.generate(
        seed_text=args.seed_text,
        max_tokens=args.tokens,
        temperature=args.temperature,
        seed=args.random_seed,
    ))


if __name__ == "__main__":
    main()
