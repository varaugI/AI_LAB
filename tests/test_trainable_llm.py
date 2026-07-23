import json
from pathlib import Path
import tempfile
import unittest

import torch

from builder.llm import ByteBPETokenizer, ModelConfig, TrainingConfig, TransformerLM
from builder.llm.data import BinaryTokenDataset, CorpusRecord, SFTDataset, format_chat_messages, prepare_binary_dataset
from builder.llm.lora import inject_lora, merge_lora
from builder.llm.trainer import LanguageModelTrainer
from builder.knowledge import DocumentCatalog, FeedbackStore, KnowledgeRuntime


class TrainableLLMTests(unittest.TestCase):
    def test_byte_bpe_round_trip_and_save(self):
        tokenizer = ByteBPETokenizer().train(
            ["hello hello world", "Unicode: नमस्ते"], vocab_size=280, min_frequency=2
        )
        text = "<|user|>\nhello नमस्ते"
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_transformer_forward_generation_and_reload(self):
        config = ModelConfig(
            vocab_size=270, max_seq_len=32, d_model=32,
            n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64,
        )
        model = TransformerLM(config)
        inputs = torch.randint(0, config.vocab_size, (2, 12))
        output = model(inputs, labels=inputs)
        self.assertEqual(output.logits.shape, (2, 12, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        generated = model.generate(inputs[:1, :4], max_new_tokens=3, temperature=0)
        self.assertEqual(generated.shape[1], 7)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = TransformerLM.from_pretrained(directory)
            self.assertEqual(loaded.config.d_model, 32)

    def test_sft_masks_prompt(self):
        tokenizer = ByteBPETokenizer()
        ids, labels = format_chat_messages([
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ], tokenizer)
        self.assertEqual(len(ids), len(labels))
        self.assertIn(-100, labels)
        self.assertTrue(any(value >= 0 for value in labels))
        dataset = SFTDataset([[
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]], tokenizer, block_size=128)
        self.assertEqual(len(dataset), 1)

    def test_catalog_duplicate_search_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "python_notes.txt"
            source.write_text("Python functions use def. A function can return a value.", encoding="utf-8")
            catalog = DocumentCatalog(root / "library.sqlite3", root / "uploads")
            first = catalog.import_file(source, domain="programming")
            second = catalog.import_file(source, domain="programming")
            self.assertEqual(first.status, "added")
            self.assertEqual(second.status, "duplicate")
            hits = catalog.search("How do Python functions return values?", domain="programming")
            self.assertTrue(hits)
            stored = Path(first.document.stored_path)
            self.assertTrue(stored.exists())
            self.assertTrue(catalog.delete_document(first.document.id))
            self.assertFalse(catalog.search("Python functions"))
            self.assertFalse(stored.exists())

    def test_feedback_exports_only_useful_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeedbackStore(root / "feedback.sqlite3")
            store.add("bad?", "bad", rating=-1)
            store.add("good?", "good", approved=True)
            store.add("fix?", "wrong", corrected_response="correct")
            destination = root / "sft.jsonl"
            self.assertEqual(store.export_sft(destination), 2)
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(rows[1]["messages"][-1]["content"], "correct")

    def test_content_duplicate_with_different_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "book_utf8.txt"
            second_path = root / "book_utf16.txt"
            text = "A unique chapter about gravity and orbital motion."
            first_path.write_text(text, encoding="utf-8")
            second_path.write_text(text, encoding="utf-16")
            catalog = DocumentCatalog(root / "library.sqlite3", root / "uploads")
            first = catalog.import_file(first_path)
            second = catalog.import_file(second_path)
            self.assertEqual(first.status, "added")
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(first.document.id, second.document.id)

    def test_runtime_answers_from_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "science.txt"
            source.write_text("Photosynthesis uses light energy to help plants make glucose.", encoding="utf-8")
            catalog = DocumentCatalog(root / "library.sqlite3", root / "uploads")
            catalog.import_file(source, domain="school")
            runtime = KnowledgeRuntime(catalog)
            reply = runtime.answer("What does photosynthesis use?", domain="school")
            self.assertTrue(reply.used_library)
            self.assertIn("light", reply.answer.lower())
            self.assertTrue(reply.sources)

    def test_lora_merge_restores_plain_linears(self):
        config = ModelConfig(vocab_size=270, max_seq_len=16, d_model=32, n_layers=1, n_heads=4, d_ff=64)
        model = TransformerLM(config)
        replaced = inject_lora(model, rank=2)
        self.assertTrue(replaced)
        merged = merge_lora(model)
        self.assertEqual(set(replaced), set(merged))
        self.assertFalse(any("lora_a" in name for name, _ in model.named_parameters()))

    def test_mini_trainer_writes_reloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = ByteBPETokenizer()
            prepare_binary_dataset([
                CorpusRecord("alpha beta gamma delta " * 80),
                CorpusRecord("one two three four " * 80),
            ], tokenizer, root / "dataset", validation_fraction=0.5)
            config = ModelConfig(
                vocab_size=tokenizer.vocab_size, max_seq_len=16, d_model=32,
                n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64,
            )
            model = TransformerLM(config)
            train = BinaryTokenDataset(root / "dataset/train.bin", 16)
            validation = BinaryTokenDataset(root / "dataset/val.bin", 16)
            training = TrainingConfig(
                output_dir=str(root / "run"), batch_size=2,
                gradient_accumulation_steps=1, max_steps=1, warmup_steps=1,
                eval_interval=1, eval_batches=1, save_interval=1, log_interval=1,
                precision="fp32",
            )
            result = LanguageModelTrainer(model, train, validation, training).train()
            self.assertEqual(result["steps"], 1)
            loaded = TransformerLM.from_pretrained(root / "run/final")
            self.assertEqual(loaded.config.vocab_size, tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
