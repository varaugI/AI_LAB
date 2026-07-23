# AI LAB — Trainable Private Knowledge Assistant

AI LAB now contains **two complete language-model paths** and a persistent document system:

1. **AI LAB Transformer** — your own decoder-only transformer, tokenizer, dataset format, trainer, checkpoints, generation code, and LoRA implementation.
2. **Production fine-tuning path** — continue-pretrain or LoRA/full fine-tune a capable Hugging Face causal language model on your books and approved conversations.
3. **Knowledge library (RAG)** — immediately search and cite imported PDF, EPUB, DOCX, web, data, text, and source-code files.

The earlier project mostly searched documents. This rebuild keeps that useful capability but no longer calls it neural learning. Importing, pretraining, chat fine-tuning, and conversation memory are separate and visible operations.

## What each operation really does

| Operation | Immediate effect | Changes neural weights? | Deleting the document reverses it? |
|---|---|---:|---:|
| Import document | Makes it searchable and citable | No | Yes, for retrieval |
| Continue pretraining | Learns token patterns and domain knowledge | Yes | No |
| Supervised chat tuning | Learns how to answer instructions | Yes | No |
| Clear chat memory | Removes recent conversation context | No | Not applicable |
| Delete document | Removes stored file and search chunks | No | Does not untrain checkpoints |

A checkpoint trained on a book may retain learned patterns after the book is deleted. To exclude that material, resume from a checkpoint created before it entered the training dataset or retrain without it.

## Major capabilities

- Modern decoder-only transformer implemented with PyTorch
- RMSNorm, rotary embeddings, grouped-query attention, SwiGLU, tied embeddings
- Universal byte-level BPE tokenizer implemented in this project
- Mixed precision, AdamW, cosine decay, warmup, gradient accumulation
- Multi-GPU Distributed Data Parallel through `torchrun`
- Validation, perplexity, JSONL logs, automatic checkpoints and resume
- Full fine-tuning and custom LoRA adapters
- Response-only supervised chat masking
- Hugging Face/PEFT production fine-tuning path
- Local transformer, Hugging Face, Ollama, and compatible-server backends
- SQLite document catalog with FTS search
- Duplicate detection by exact file hash **and extracted text hash**
- True document deletion with cascading chunk deletion
- Persistent conversation memory
- Positive/corrected chat feedback export for fine-tuning
- PDF, scanned-PDF OCR, EPUB, DOCX, HTML, text, data, and code ingestion
- Law, programming, school, fiction, and general categories
- 36 automated tests plus an end-to-end miniature pretraining validation

## Install

Create and activate a virtual environment first.

### Core from-scratch training

```bash
pip install -r requirements-core.txt
```

### Web interface

```bash
pip install -r requirements-web.txt
pip install -r requirements-optional.txt
```

### Production model fine-tuning

```bash
pip install -r requirements-production.txt
```

Scanned-PDF OCR also requires the Tesseract application installed on the computer.

## Start the application

```bash
python run_ai_lab.py
```

Open:

```text
http://127.0.0.1:5000
```

The application stores runtime data under `data/runtime/`:

```text
data/runtime/library.sqlite3
data/runtime/feedback.sqlite3
data/runtime/conversation_memory.json
data/runtime/uploads/
data/runtime/exports/
```

## Import and manage documents from the terminal

```bash
python -m builder.experiments.manage_library add books/physics.pdf --domain school
python -m builder.experiments.manage_library add src/project.py --domain programming
python -m builder.experiments.manage_library list
python -m builder.experiments.manage_library search "Newton's second law"
python -m builder.experiments.manage_library delete 3
```

Adding the same bytes twice is skipped. A differently encoded PDF/EPUB containing the same extracted text is also skipped.


## Optional semantic retrieval

Lexical FTS works without extra model downloads. To add embedding-based retrieval:

```bash
pip install -r requirements-retrieval.txt
python -m builder.experiments.build_semantic_index
```

Set `AI_LAB_SEMANTIC_MODEL` before starting the web app, or keep the generated `data/runtime/semantic_index.npz`. Rebuild the semantic index after major library changes.

## Path A: train your own transformer from scratch

### 1. Build a clean corpus

```bash
python -m builder.experiments.build_training_corpus books/ notes/ code/ \
  --recursive \
  --output data/datasets/corpus.jsonl
```

### 2. Train the tokenizer

```bash
python -m builder.experiments.train_tokenizer \
  data/datasets/corpus.jsonl \
  --vocab-size 8192 \
  --output data/tokenizers/main.json
```

### 3. Create token streams

```bash
python -m builder.experiments.prepare_pretraining_data \
  data/datasets/corpus.jsonl \
  --tokenizer data/tokenizers/main.json \
  --output data/datasets/pretraining
```

### 4. Pretrain

Start with the tiny profile to verify the pipeline:

```bash
python -m builder.experiments.pretrain_transformer \
  data/datasets/pretraining \
  --model-config configs/model_tiny.json \
  --training-config configs/training_tiny.json
```

For multiple NVIDIA GPUs:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m builder.experiments.pretrain_transformer \
  data/datasets/pretraining \
  --model-config configs/model_medium.json \
  --training-config configs/training_gpu.json
```

The ready-to-load checkpoint is written to:

```text
data/checkpoints/<run>/ready/
```

### One-command version using the web library

```bash
python -m builder.experiments.train_from_library \
  --model-config configs/model_small.json \
  --training-config configs/training_gpu.json \
  --vocab-size 8192
```

## Teach it to answer through approved chats

In the web interface, use:

- **Good** for a correct reply
- **Wrong** for a bad reply
- **Correct** to supply the desired answer

Export the useful conversations:

```bash
python -m builder.experiments.export_chat_training
```

Then supervised fine-tune:

```bash
python -m builder.experiments.instruction_tune \
  data/checkpoints/pretrain-gpu/ready \
  data/datasets/approved_chats.jsonl \
  --training-config configs/training_sft.json
```

For lower GPU memory, use the custom LoRA path:

```bash
python -m builder.experiments.instruction_tune \
  data/checkpoints/pretrain-gpu/ready \
  data/datasets/approved_chats.jsonl \
  --lora-rank 16
```

A merged standalone checkpoint is created under the run's `ready/` directory.

## Path B: adapt a capable pretrained open model

This path reaches useful conversation quality much sooner than training a small model from random weights.

Continue pretraining on your document corpus:

```bash
python -m builder.experiments.finetune_pretrained_model \
  MODEL_NAME_OR_PATH \
  data/runtime/exports/knowledge_corpus.jsonl \
  --mode pretrain \
  --output data/checkpoints/domain-model
```

LoRA response tuning on approved chats:

```bash
python -m builder.experiments.finetune_pretrained_model \
  MODEL_NAME_OR_PATH \
  data/runtime/exports/approved_chats.jsonl \
  --mode sft \
  --lora \
  --output data/checkpoints/chat-adapter
```

Use only models and training data whose licenses permit your intended use.

## Chat directly with your from-scratch checkpoint

```bash
python -m builder.experiments.chat_transformer \
  data/checkpoints/instruction-tuned/ready
```

Or select **My from-scratch AI LAB transformer** in the web interface and enter the checkpoint directory.

## Migrate the previous JSON library

```bash
python -m builder.experiments.migrate_legacy_library knowledge_library.json
```

The migration reconstructs documents from the old chunks and imports them into the duplicate-aware SQLite catalog.

## Scaling guidance

- Verify correctness with `model_tiny.json` before spending GPU hours.
- Increase model size only after you have enough clean tokens.
- Keep a validation split and watch validation loss, not only training loss.
- Continue pretraining teaches domain text; it does not automatically teach helpful conversation.
- Use diverse, reviewed instruction examples for SFT.
- Never automatically train on every model reply. Bad answers would become new training targets.
- Keep immutable dataset exports and checkpoint manifests so you know what each model learned from.
- Use RAG for exact, changing, or legally sensitive facts even after training.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Important limitations

- A randomly initialized tiny model will produce poor language until trained on substantial clean data.
- A private model will not become equivalent to ChatGPT merely by reading several books.
- Training quality depends on data quality, token count, model size, optimization, and evaluation.
- Deleting a source removes retrieval knowledge, not information already distributed through model weights.
- Legal answers are informational and must be checked against authoritative, current sources and qualified counsel.
