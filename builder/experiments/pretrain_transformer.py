"""Pretrain or continue-pretrain AI LAB's own transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from builder.llm import (
    BinaryTokenDataset,
    ByteBPETokenizer,
    LanguageModelTrainer,
    ModelConfig,
    TrainingConfig,
    TransformerLM,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Directory containing train.bin, val.bin and tokenizer.json")
    parser.add_argument("--model-config", default="configs/model_tiny.json")
    parser.add_argument("--training-config", default="configs/training_tiny.json")
    parser.add_argument("--initialize-from", default="", help="Existing checkpoint directory for continued pretraining")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    tokenizer = ByteBPETokenizer.load(dataset_dir / "tokenizer.json")
    model_config = ModelConfig.load(args.model_config)
    model_config.vocab_size = tokenizer.vocab_size
    if args.initialize_from:
        model = TransformerLM.from_pretrained(args.initialize_from)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("The checkpoint and tokenizer vocabularies do not match")
    else:
        model = TransformerLM(model_config)
    training_config = TrainingConfig.load(args.training_config)
    train_dataset = BinaryTokenDataset(dataset_dir / "train.bin", model.config.max_seq_len)
    validation_path = dataset_dir / "val.bin"
    validation_dataset = (
        BinaryTokenDataset(validation_path, model.config.max_seq_len)
        if validation_path.exists() and validation_path.stat().st_size >= (model.config.max_seq_len + 1) * 4
        else None
    )
    trainer = LanguageModelTrainer(model, train_dataset, validation_dataset, training_config)
    result = trainer.train()
    if trainer.is_main:
        output_dir = Path(training_config.output_dir)
        ready_dir = output_dir / "ready"
        model.save_pretrained(ready_dir, extra={"training_result": result})
        shutil.copy2(dataset_dir / "tokenizer.json", ready_dir / "tokenizer.json")
        shutil.copy2(dataset_dir / "tokenizer.json", output_dir / "tokenizer.json")
        for checkpoint in output_dir.iterdir():
            if checkpoint.is_dir() and (checkpoint / "model.pt").exists():
                shutil.copy2(dataset_dir / "tokenizer.json", checkpoint / "tokenizer.json")
        print(json.dumps({
            **result,
            "parameters": model.parameter_count(),
            "trainable_parameters": model.parameter_count(trainable_only=True),
        }, indent=2))


if __name__ == "__main__":
    main()
