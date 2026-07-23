import os
import sys
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add parent dir to sys.path to resolve 'builder' imports if run from inside builder/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from builder.framework import Sequential
from builder.books import NovelAssistant

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load Models
NOVEL_INDEX_PATH = "novel_library.json"
novel_assistant = None
try:
    if os.path.exists(NOVEL_INDEX_PATH):
        novel_assistant = NovelAssistant.load(NOVEL_INDEX_PATH)
        print("[OK] Loaded Novel Assistant.")
    else:
        print("[WARNING] novel_library.json not found.")
except Exception as e:
    print(f"[ERROR] Failed to load Novel Assistant: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "novel": novel_assistant is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    if novel_assistant is None:
        return jsonify({"error": "Novel Assistant index is not loaded. Try running 'ingest_novels.py' first!"}), 503
        
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({"error": "Empty message."}), 400
        
    try:
        # Search command
        if question.startswith("/search "):
            query = question[8:].strip()
            results = novel_assistant.search(query, limit=3)
            search_reply = "<strong>Search Results:</strong><br><br>"
            for r in results:
                search_reply += f"<em>{r.chunk.title} - {r.chunk.location}</em><br>{r.chunk.text[:200]}...<br><br>"
            return jsonify({"answer": search_reply, "sources": []})
            
        # Normal Q&A
        reply = novel_assistant.answer(question)
        sources = []
        if reply.sources:
            for s in reply.sources:
                sources.append({
                    "title": s.title,
                    "location": s.location,
                    "excerpt": s.excerpt
                })
                
        return jsonify({
            "answer": reply.answer,
            "confidence": reply.confidence,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000, visible to localhost
    app.run(host='127.0.0.1', port=5000, debug=True)
