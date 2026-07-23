"""One-command from-scratch training pipeline over the SQLite knowledge library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from builder.knowledge import DocumentCatalog
from builder.llm import (
    BinaryTokenDataset,
    ByteBPETokenizer,
    LanguageModelTrainer,
    ModelConfig,
    TrainingConfig,
    TransformerLM,
    prepare_binary_dataset,
)
from builder.llm.data import iter_jsonl_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="data/runtime/library.sqlite3")
    parser.add_argument("--uploads", default="data/runtime/uploads")
    parser.add_argument("--work-dir", default="data/training_run")
    parser.add_argument("--model-config", default="configs/model_tiny.json")
    parser.add_argument("--training-config", default="configs/training_tiny.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer", default="", help="Reuse an existing tokenizer JSON")
    parser.add_argument("--domain", default="all")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    catalog = DocumentCatalog(args.database, args.uploads)
    corpus = work / "corpus.jsonl"
    document_count = catalog.export_corpus(corpus, domain=args.domain)
    if not document_count:
        raise SystemExit("The library contains no matching documents.")

    if args.tokenizer:
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
    else:
        tokenizer = ByteBPETokenizer().train(
            (record.text for record in iter_jsonl_records(corpus)),
            vocab_size=args.vocab_size,
            min_frequency=3,
            verbose=True,
        )
    tokenizer_path = work / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    dataset_dir = work / "tokenized"
    metadata = prepare_binary_dataset(corpus, tokenizer, dataset_dir)

    model_config = ModelConfig.load(args.model_config)
    model_config.vocab_size = tokenizer.vocab_size
    training_config = TrainingConfig.load(args.training_config)
    model = TransformerLM(model_config)
    train = BinaryTokenDataset(dataset_dir / "train.bin", model.config.max_seq_len)
    val_path = dataset_dir / "val.bin"
    validation = (
        BinaryTokenDataset(val_path, model.config.max_seq_len)
        if val_path.exists() and val_path.stat().st_size >= (model.config.max_seq_len + 1) * 4
        else None
    )
    trainer = LanguageModelTrainer(model, train, validation, training_config)
    result = trainer.train()
    if trainer.is_main:
        ready = Path(training_config.output_dir) / "ready"
        model.save_pretrained(ready, extra={"dataset": metadata, "training": result})
        shutil.copy2(tokenizer_path, ready / "tokenizer.json")
        print(json.dumps({
            "ready_checkpoint": str(ready),
            "documents": document_count,
            "dataset": metadata,
            "training": result,
            "parameters": model.parameter_count(),
        }, indent=2))


if __name__ == "__main__":
    main()
