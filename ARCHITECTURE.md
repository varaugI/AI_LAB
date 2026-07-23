# Architecture

## 1. Knowledge plane

`DocumentCatalog` stores documents and chunks in SQLite. It uses file SHA-256 and normalized extracted-text SHA-256 for duplicate detection. SQLite FTS5 performs local retrieval. Foreign-key cascading makes document deletion remove all associated chunks.

## 2. Learning plane

### From-scratch model

`builder/llm/model.py` implements a causal transformer:

```text
bytes → BPE tokens → token embeddings
      → repeated transformer blocks
      → RMSNorm → tied LM head → next-token logits
```

Each transformer block contains:

```text
RMSNorm → grouped-query causal attention + residual
RMSNorm → SwiGLU feed-forward + residual
```

### Training objectives

Pretraining minimizes next-token cross entropy over all corpus tokens.

Supervised chat tuning masks system/user prompt tokens with `-100`, so loss is applied to assistant response tokens only.

### Production path

`finetune_pretrained_model.py` supports continued pretraining and response-only SFT of Hugging Face causal models, with optional PEFT LoRA.

## 3. Conversation plane

`KnowledgeRuntime` retrieves sources, constructs a protected prompt, adds recent conversation context, calls the selected backend, and returns citations.

Backends:

- AI LAB from-scratch transformer
- Hugging Face local model/checkpoint
- Ollama
- OpenAI-compatible local/hosted endpoint
- extractive offline fallback

## 4. Feedback plane

Feedback is stored separately in `feedback.sqlite3`. Only positive, approved, or corrected replies are exported for SFT. This avoids self-training on hallucinations.

## 5. Data lifecycle

```text
Document upload
  ├─ file hash duplicate check
  ├─ text extraction
  ├─ content hash duplicate check
  ├─ SQLite chunks for immediate RAG
  └─ optional export to immutable training corpus

Approved chats
  └─ SFT JSONL

Training corpus + tokenizer
  └─ binary token streams
      └─ checkpoint series
          └─ ready inference checkpoint
```
