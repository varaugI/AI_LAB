"""Supervised fine-tune a checkpoint on approved chat JSONL."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import shutil

from builder.llm import ByteBPETokenizer, LanguageModelTrainer, SFTDataset, TrainingConfig, TransformerLM
from builder.llm.data import sft_collate
from builder.llm.lora import inject_lora, merge_lora, save_lora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("dataset", help="JSONL with messages or instruction/output")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--training-config", default="configs/training_sft.json")
    parser.add_argument("--lora-rank", type=int, default=0, help="Use LoRA when greater than zero")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else checkpoint / "tokenizer.json"
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    model = TransformerLM.from_pretrained(checkpoint)
    if args.lora_rank > 0:
        for parameter in model.parameters():
            parameter.requires_grad = False
        replaced = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f"Injected LoRA into {len(replaced)} linear layers")
    dataset = SFTDataset(args.dataset, tokenizer, model.config.max_seq_len)
    config = TrainingConfig.load(args.training_config)
    # A deterministic holdout split for validation.
    validation_size = max(1, len(dataset) // 20) if len(dataset) >= 10 else 0
    if validation_size:
        train_dataset, validation_dataset = __import__("torch").utils.data.random_split(
            dataset, [len(dataset) - validation_size, validation_size],
            generator=__import__("torch").Generator().manual_seed(config.seed),
        )
    else:
        train_dataset, validation_dataset = dataset, None
    trainer = LanguageModelTrainer(
        model,
        train_dataset,
        validation_dataset,
        config,
        collate_fn=partial(sft_collate, pad_id=tokenizer.pad_id),
    )
    result = trainer.train()
    if trainer.is_main:
        output_dir = Path(config.output_dir)
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
        if args.lora_rank > 0:
            save_lora(model, output_dir / "adapter.pt", {
                "base_checkpoint": str(checkpoint),
                "rank": args.lora_rank,
                "alpha": args.lora_alpha,
            })
            merge_lora(model)
        ready_dir = output_dir / "ready"
        model.save_pretrained(ready_dir, extra={"training_result": result})
        shutil.copy2(tokenizer_path, ready_dir / "tokenizer.json")
        print(json.dumps({**result, "ready_checkpoint": str(ready_dir)}, indent=2))


if __name__ == "__main__":
    main()
