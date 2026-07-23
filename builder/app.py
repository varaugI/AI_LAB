"""AI LAB web application: persistent library, trainable local model, and chat."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, jsonify, render_template, request
    from werkzeug.utils import secure_filename
except ImportError as exc:  # pragma: no cover - dependency guidance
    raise SystemExit(
        "Flask is required for the web interface. Run: pip install -r requirements-web.txt"
    ) from exc

from builder.books import (
    OllamaBackend,
    OpenAICompatibleBackend,
    SUPPORTED_EXTENSIONS,
)
from builder.knowledge import DocumentCatalog, HybridRetriever, KnowledgeRuntime, SemanticIndex
from builder.llm import HuggingFaceBackend, LocalTransformerBackend


DATA_DIR = Path(os.environ.get("AI_LAB_DATA_DIR", PROJECT_ROOT / "data" / "runtime")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "library.sqlite3"
MEMORY_PATH = DATA_DIR / "conversation_memory.json"
FEEDBACK_PATH = DATA_DIR / "feedback.sqlite3"
EXPORT_DIR = DATA_DIR / "exports"
for directory in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

catalog = DocumentCatalog(DATABASE_PATH, UPLOAD_DIR)
semantic_path = DATA_DIR / "semantic_index.npz"
semantic = None
if semantic_path.exists() or os.environ.get("AI_LAB_SEMANTIC_MODEL"):
    semantic = SemanticIndex(
        catalog, semantic_path,
        os.environ.get("AI_LAB_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
retriever = HybridRetriever(catalog, semantic)
runtime = KnowledgeRuntime(
    catalog,
    retriever=retriever,
    memory_path=MEMORY_PATH,
    feedback_path=FEEDBACK_PATH,
)
state_lock = Lock()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("AI_LAB_MAX_UPLOAD_BYTES", 500 * 1024 * 1024))


def source_payload(sources):
    return [item.__dict__ for item in sources]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    stats = catalog.stats()
    backend = runtime.backend
    return jsonify({
        "ready": stats["documents"] > 0,
        **stats,
        "memory_turns": len(runtime.memory.turns),
        "backend": getattr(backend, "name", "extractive") if backend else "extractive",
        "model": getattr(backend, "model", getattr(backend, "model_id", "")) if backend else "",
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    })


@app.get("/api/documents")
def documents():
    domain = request.args.get("domain", "all")
    records = catalog.list_documents(None if domain == "all" else domain)
    return jsonify({"documents": [record.__dict__ for record in records]})


@app.delete("/api/documents/<int:document_id>")
def delete_document(document_id: int):
    with state_lock:
        deleted = catalog.delete_document(document_id, delete_file=True)
    if not deleted:
        return jsonify({"error": "Document not found."}), 404
    return jsonify({"message": "Document, searchable chunks, and stored upload were deleted. Existing trained checkpoints were not untrained."})


@app.post("/api/upload")
def upload_documents():
    files = request.files.getlist("files")
    requested_domain = str(request.form.get("domain", "auto")).strip().lower()
    replace_same_name = str(request.form.get("replace_same_name", "false")).lower() == "true"
    ocr_scanned = str(request.form.get("ocr_scanned", "false")).lower() == "true"
    if requested_domain not in {"auto", "general", "fiction", "school", "law", "programming"}:
        return jsonify({"error": "Unsupported domain."}), 400
    if not files:
        return jsonify({"error": "Select at least one document."}), 400

    results = []
    rejected = []
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as temporary:
        temp_dir = Path(temporary)
        for uploaded in files:
            original = uploaded.filename or ""
            filename = secure_filename(original)
            if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                rejected.append(original or "unnamed file")
                continue
            temporary_path = temp_dir / filename
            uploaded.save(temporary_path)
            try:
                with state_lock:
                    result = catalog.import_file(
                        temporary_path,
                        domain=requested_domain,
                        replace_same_name=replace_same_name,
                        ocr_scanned=ocr_scanned,
                    )
                results.append({
                    "status": result.status,
                    "message": result.message,
                    "document": result.document.__dict__,
                })
            except Exception as exc:
                rejected.append(f"{original}: {exc}")
    if not results:
        return jsonify({"error": "No documents were imported.", "rejected": rejected}), 400
    return jsonify({
        "message": "Import completed.",
        "added": sum(item["status"] == "added" for item in results),
        "duplicates": sum(item["status"] == "duplicate" for item in results),
        "results": results,
        "rejected": rejected,
        **catalog.stats(),
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    mode = str(payload.get("mode", "chat")).strip().lower()
    domain = str(payload.get("domain", "all")).strip().lower()
    scope = str(payload.get("scope", "hybrid")).strip().lower()
    document_ids = [int(value) for value in payload.get("document_ids", [])]
    if mode == "summary":
        document_id = document_ids[0] if document_ids else None
        reply = runtime.summarize(
            document_id=document_id,
            domain=None if domain == "all" else domain,
        )
    else:
        if not message:
            return jsonify({"error": "Enter a message."}), 400
        with state_lock:
            reply = runtime.answer(
                message,
                mode=mode,
                domain=None if domain == "all" else domain,
                document_ids=document_ids or None,
                allow_general_knowledge=scope != "library",
            )
    return jsonify({
        "answer": reply.answer,
        "sources": source_payload(reply.sources),
        "backend": reply.backend,
        "model": reply.model,
        "used_library": reply.used_library,
        "confidence": reply.confidence,
    })


@app.post("/api/search")
def search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    domain = str(payload.get("domain", "all"))
    if not query:
        return jsonify({"error": "Enter a search query."}), 400
    hits = catalog.search(query, domain=None if domain == "all" else domain, limit=10)
    return jsonify({"results": [hit.__dict__ for hit in hits]})


@app.post("/api/backend")
def configure_backend():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("backend", "extractive")).strip().lower()
    try:
        with state_lock:
            if name in {"extractive", "none", "offline"}:
                runtime.set_backend(None)
                return jsonify({"message": "Using the built-in grounded extractive engine."})
            if name == "local":
                checkpoint = str(payload.get("checkpoint", "")).strip()
                tokenizer = str(payload.get("tokenizer", "")).strip() or None
                if not checkpoint:
                    return jsonify({"error": "Enter a local checkpoint directory."}), 400
                runtime.set_backend(LocalTransformerBackend(checkpoint, tokenizer))
                return jsonify({"message": f"Loaded AI LAB checkpoint: {checkpoint}"})
            if name == "huggingface":
                checkpoint = str(payload.get("checkpoint", "")).strip()
                adapter = str(payload.get("adapter", "")).strip()
                if not checkpoint:
                    return jsonify({"error": "Enter a Hugging Face model name or checkpoint path."}), 400
                runtime.set_backend(HuggingFaceBackend(checkpoint, adapter_path=adapter))
                return jsonify({"message": f"Loaded Hugging Face model: {checkpoint}"})
            if name == "ollama":
                model = str(payload.get("model", "llama3.2:3b")).strip()
                base_url = str(payload.get("base_url", "http://127.0.0.1:11434")).strip()
                backend = OllamaBackend(model=model, base_url=base_url)
                if not backend.available():
                    return jsonify({"error": "Ollama is not reachable. Start it and pull the selected model."}), 400
                runtime.set_backend(backend)
                return jsonify({"message": f"Connected to Ollama model {model}."})
            if name == "compatible":
                model = str(payload.get("model", "")).strip()
                base_url = str(payload.get("base_url", "")).strip()
                api_key = str(payload.get("api_key", ""))
                runtime.set_backend(OpenAICompatibleBackend(model, base_url, api_key))
                return jsonify({"message": f"Configured compatible model {model}."})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"error": "Unsupported backend."}), 400


@app.post("/api/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    try:
        feedback_id = runtime.feedback.add(
            str(payload.get("prompt", "")),
            str(payload.get("response", "")),
            corrected_response=str(payload.get("corrected_response", "")),
            rating=payload.get("rating"),
            approved=bool(payload.get("approved", False)),
            context=payload.get("sources") or [],
        )
        return jsonify({"message": "Feedback saved for supervised fine-tuning.", "id": feedback_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/training/export")
def export_training_data():
    corpus_path = EXPORT_DIR / "knowledge_corpus.jsonl"
    chats_path = EXPORT_DIR / "approved_chats.jsonl"
    documents_count = catalog.export_corpus(corpus_path)
    chat_count = runtime.feedback.export_sft(chats_path)
    return jsonify({
        "message": "Training datasets exported.",
        "corpus": str(corpus_path),
        "documents": documents_count,
        "chats": str(chats_path),
        "chat_examples": chat_count,
    })


@app.post("/api/memory/clear")
def clear_memory():
    with state_lock:
        runtime.clear_memory()
    return jsonify({"message": "Conversation memory cleared. Documents and trained weights were not deleted."})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("AI_LAB_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True,
    )
