"""Train either a practical n-gram model or a tiny from-scratch neural model."""

from __future__ import annotations

import argparse
from pathlib import Path

from builder.books import NGramLanguageModel, TinyNeuralLanguageModel, read_documents


def build_parser():
    parser = argparse.ArgumentParser(description="Learn word patterns from TXT/PDF/EPUB novels.")
    parser.add_argument("paths", nargs="+", help="Book files or directories")
    parser.add_argument("--mode", choices=("ngram", "neural"), default="ngram")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-sections", type=int, default=None)
    parser.add_argument("--ocr-scanned", action="store_true")
    parser.add_argument("--order", type=int, default=4, help="N-gram order")
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vocabulary", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    documents = read_documents(
        args.paths,
        max_pages=args.max_pages,
        max_sections=args.max_sections,
        ocr_scanned=args.ocr_scanned,
    )
    texts = [section.text for document in documents for section in document.sections]
    if not texts:
        raise ValueError("No readable text was found.")

    if args.mode == "ngram":
        model = NGramLanguageModel(order=args.order).train(texts)
        output = args.output or "novel_ngram.json"
        model.save(output)
        print(f"Learned {model.token_count:,} tokens across {len(model.transitions):,} contexts.")
    else:
        print("Training the educational neural model. Pure-Python training is intentionally small and slow.")
        model = TinyNeuralLanguageModel(
            context_size=args.context_size,
            hidden_size=args.hidden_size,
        )
        model.train(
            texts,
            max_vocabulary=args.vocabulary,
            max_samples=args.max_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        output = args.output or "novel_neural_language.json"
        model.save(output)
        print(f"Vocabulary size: {len(model.vocabulary)}")

    print(f"Saved language model: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
