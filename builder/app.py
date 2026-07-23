"""Local web interface for AI LAB's mixed-document knowledge assistant."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from builder.books import NovelLearningSystem, OllamaBackend, SUPPORTED_EXTENSIONS


DATA_DIR = Path(os.environ.get("AI_LAB_DATA_DIR", PROJECT_ROOT)).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = Path(os.environ.get("AI_LAB_INDEX", DATA_DIR / "knowledge_library.json"))
MEMORY_PATH = Path(os.environ.get("AI_LAB_MEMORY", DATA_DIR / "conversation_memory.json"))
UPLOAD_DIR = Path(os.environ.get("AI_LAB_UPLOADS", DATA_DIR / "uploaded_documents"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
state_lock = Lock()
system: NovelLearningSystem | None = None


def load_system():
    global system
    if not INDEX_PATH.exists():
        system = None
        return
    try:
        system = NovelLearningSystem.load_library(INDEX_PATH, memory_file=MEMORY_PATH)
    except Exception as exc:
        print(f"[AI LAB] Could not load {INDEX_PATH}: {exc}")
        system = None


def require_system():
    if system is None:
        return None, (jsonify({
            "error": "No knowledge library is loaded. Upload documents or source files first."
        }), 503)
    return system, None


def source_payload(sources):
    return [
        {
            "number": source.number,
            "title": source.title,
            "location": source.location,
            "excerpt": source.excerpt,
            "source": source.source,
        }
        for source in sources
    ]


load_system()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    stats = system.library_stats() if system else {
        "chunks": 0, "titles": [], "domains": {}, "kinds": {},
        "backend": "extractive", "model": "",
    }
    return jsonify({
        "ready": system is not None,
        **stats,
        "memory_turns": len(system.memory.turns) if system else 0,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    })


@app.post("/api/chat")
def chat():
    active, error = require_system()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    mode = str(payload.get("mode", "chat")).strip().lower()
    domain = str(payload.get("domain", "all")).strip().lower()
    scope = str(payload.get("scope", "hybrid")).strip().lower()
    if not message and mode not in {"summary", "characters"}:
        return jsonify({"error": "Enter a message."}), 400

    try:
        with state_lock:
            if mode == "search":
                results = active.search(message, limit=5, domain=None if domain in {"", "all"} else domain)
                return jsonify({
                    "answer": f"Found {len(results)} matching passages.",
                    "results": [
                        {
                            "title": item.chunk.title,
                            "location": item.chunk.location,
                            "excerpt": item.chunk.text[:500],
                            "score": item.score,
                        }
                        for item in results
                    ],
                    "sources": [],
                })
            if mode == "summary":
                title = message or None
                result = active.summarize(title=title)
                return jsonify({
                    "answer": result.summary,
                    "heading": f"Summary — {result.title}",
                    "sources": source_payload(result.sources),
                })
            if mode == "characters":
                profiles = active.analyze_characters().list(limit=30)
                return jsonify({
                    "answer": "Recurring characters detected in the library.",
                    "characters": [profile.to_dict() for profile in profiles],
                    "sources": [],
                })
            if mode == "character":
                profile = active.character(message)
                if not profile:
                    return jsonify({"answer": f"No recurring character named {message!r} was found.", "sources": []})
                return jsonify({"answer": f"Character profile — {profile.name}", "character": profile.to_dict(), "sources": []})

            reply = active.chat(
                message,
                mode="chat" if mode == "ask" else mode,
                domain=None if domain in {"", "all"} else domain,
                allow_general_knowledge=scope != "library",
            )
            active.save_memory(MEMORY_PATH)
            return jsonify({
                "answer": reply.answer,
                "confidence": reply.confidence,
                "sources": source_payload(reply.sources),
                "backend": getattr(reply, "backend", "extractive"),
                "model": getattr(reply, "model", ""),
                "mode": getattr(reply, "mode", mode),
                "used_library": getattr(reply, "used_library", bool(reply.sources)),
            })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/upload")
def upload_books():
    global system
    files = request.files.getlist("files")
    requested_domain = str(request.form.get("domain", "auto")).strip().lower()
    allowed_domains = {"auto", "general", "fiction", "school", "law", "programming"}
    if requested_domain not in allowed_domains:
        return jsonify({"error": "Unsupported domain selection."}), 400
    if not files:
        return jsonify({"error": "Select at least one document."}), 400

    saved = []
    rejected = []
    for uploaded in files:
        original = uploaded.filename or ""
        filename = secure_filename(original)
        extension = Path(filename).suffix.lower()
        if not filename or extension not in SUPPORTED_EXTENSIONS:
            rejected.append(original or "unnamed file")
            continue
        destination = UPLOAD_DIR / filename
        stem, suffix = destination.stem, destination.suffix
        counter = 2
        while destination.exists():
            destination = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        uploaded.save(destination)
        saved.append(destination)

    if not saved:
        return jsonify({"error": "No supported files were uploaded.", "rejected": rejected}), 400

    try:
        with state_lock:
            if system is None:
                system = NovelLearningSystem()
            report = system.learn_files(
                saved,
                append=bool(system.index.chunks),
                domain=requested_domain,
            )
            system.save_library(INDEX_PATH)
        return jsonify({
            "message": "Documents imported successfully.",
            "documents": report.documents,
            "sections": report.sections,
            "chunks": report.chunks,
            "words": report.words,
            "titles": system.titles(),
            "domains": system.library_stats()["domains"],
            "rejected": rejected,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "rejected": rejected}), 500


@app.post("/api/backend")
def configure_backend():
    active, error_response = require_system()
    if error_response:
        return error_response
    payload = request.get_json(silent=True) or {}
    backend_name = str(payload.get("backend", "extractive")).strip().lower()
    try:
        with state_lock:
            if backend_name in {"extractive", "none", "offline"}:
                active.set_backend(None)
                return jsonify({"message": "Using the built-in extractive reply engine.", **active.library_stats()})
            if backend_name == "ollama":
                model = str(payload.get("model", "llama3.2:3b")).strip() or "llama3.2:3b"
                base_url = str(payload.get("base_url", "http://127.0.0.1:11434")).strip()
                backend = OllamaBackend(model=model, base_url=base_url)
                if not backend.available():
                    return jsonify({
                        "error": f"Ollama is not reachable at {base_url}. Start Ollama first, then pull {model}."
                    }), 400
                active.set_backend(backend)
                return jsonify({"message": f"Connected to local Ollama model {model}.", **active.library_stats()})
            return jsonify({"error": "Unsupported backend."}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/memory/clear")
def clear_memory():
    active, error = require_system()
    if error:
        return error
    with state_lock:
        active.clear_memory()
        active.save_memory(MEMORY_PATH)
    return jsonify({"message": "Conversation memory cleared. The imported books remain indexed."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
