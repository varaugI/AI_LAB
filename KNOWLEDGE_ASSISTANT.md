# AI LAB General Knowledge Assistant

AI LAB can now chat with a mixed local library containing novels, school books,
law material, coding books, notes, structured data, and source-code files.

## What happens when you feed it a document

Importing a file does **not** retrain billions of language-model parameters.
Instead, AI LAB:

1. Extracts readable text.
2. Detects a broad domain: fiction, school, law, programming, or general.
3. Splits the text into overlapping passages.
4. Builds a searchable BM25 knowledge index.
5. Retrieves relevant passages for every question.
6. Gives those passages to the selected reply engine.
7. Returns the answer together with page, chapter, or file citations.

This retrieval-augmented approach is much faster and more reliable on a normal
computer than attempting to train a ChatGPT-sized model from the uploaded books.

## Supported files

- Books and documents: PDF, EPUB, DOCX, TXT, Markdown, RST, HTML
- Structured material: CSV, JSON, XML, YAML, TOML, INI
- Code: Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, PHP, Ruby,
  SQL, shell, PowerShell, Kotlin, and Swift

Ordinary PDFs need `pypdf`. Image-only PDFs additionally need Tesseract OCR,
PyMuPDF, Pillow, and pytesseract.

## Start the browser application

```bash
pip install -r requirements-web.txt
pip install -r requirements-optional.txt
python -m builder.app
```

Open `http://127.0.0.1:5000`.

Upload documents in the left sidebar. Choose automatic categorization or force
a category for a particular import.

## Reply engines

### Built-in offline engine

This works immediately. It retrieves and combines sentences from the imported
library. It is grounded and lightweight, but its prose is less natural than a
large language model.

### Local Ollama model — recommended

Ollama runs a language model on your computer. After installing Ollama, pull a
model in a terminal:

```bash
ollama pull llama3.2:3b
ollama serve
```

In the AI LAB sidebar choose **Local Ollama model**, enter
`llama3.2:3b`, and click **Apply reply engine**.

Larger models generally write better replies but require more RAM. A 3B model is
a practical starting point. The imported passages remain the evidence supplied
to the model.

You can also configure Ollama before starting the app:

### Windows PowerShell

```powershell
$env:AI_LAB_BACKEND="ollama"
$env:AI_LAB_MODEL="llama3.2:3b"
python -m builder.app
```

### macOS or Linux

```bash
AI_LAB_BACKEND=ollama AI_LAB_MODEL=llama3.2:3b python -m builder.app
```

### OpenAI-compatible local server

AI LAB can call LM Studio, llama.cpp servers, or another compatible endpoint:

```bash
AI_LAB_BACKEND=openai-compatible \
AI_LAB_API_BASE=http://127.0.0.1:1234 \
AI_LAB_MODEL=your-model-name \
python -m builder.app
```

An API key can be supplied through `AI_LAB_API_KEY` when the server requires it.

## Assistant modes

- **Chat** — natural grounded conversation with follow-up memory
- **Explain** — simple step-by-step explanation
- **Study** — tutor-style answer with self-check questions
- **Code** — programming-focused explanation and examples
- **Law** — explains imported legal material with a legal-information notice
- **Search** — returns passages without composing an answer
- **Summary** — summarizes one document or the complete library
- **Characters** — extracts recurring names and relationships from fiction

Use the domain selector to search only school, law, programming, fiction, or
general material.

## Command-line workflow

Build a library:

```bash
python -m builder.experiments.ingest_knowledge books/ notes/ source_code/ \
  --output knowledge_library.json
```

Chat with it:

```bash
python -m builder.experiments.chat_with_knowledge \
  --index knowledge_library.json
```

Start directly in programming mode:

```bash
python -m builder.experiments.chat_with_knowledge \
  --index knowledge_library.json \
  --mode code \
  --domain programming
```

Inside the command-line chat, use `/mode`, `/domain`, `/search`, `/summary`,
`/memory`, `/clear`, `/stats`, and `/quit`.

## Python API

```python
from builder.books import KnowledgeLearningSystem, OllamaBackend

brain = KnowledgeLearningSystem(
    backend=OllamaBackend(model="llama3.2:3b"),
    use_environment_backend=False,
)

brain.learn_files([
    "library/novel.epub",
    "library/physics.pdf",
    "library/contracts.docx",
    "project/server.py",
])

reply = brain.chat(
    "Explain how this server handles authentication",
    mode="code",
    domain="programming",
)

print(reply.answer)
for source in reply.sources:
    print(source.title, source.location)

brain.save_library("knowledge_library.json")
brain.save_memory("conversation_memory.json")
```

## Important limitations

- The assistant knows the imported material only through retrieved passages.
- A local language model can still make mistakes; citations should be checked.
- Legal mode explains source material but is not a replacement for a lawyer.
- Medical and financial material should be checked against current official or
  professional sources.
- Scanned or badly formatted PDFs may require OCR and manual correction.
- The from-scratch tiny neural language model remains educational. It is not a
  substitute for a pretrained transformer model.
