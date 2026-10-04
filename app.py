import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import llm
import memory
import rag

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads
UPLOAD_DIR = "data/uploads"
ALLOWED = {".pdf", ".txt", ".md"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

SYSTEM = (
    "You are NexaChat, a helpful multimodal assistant. Reply in the language the user writes in "
    "(English, Hindi or Hinglish). Be clear and concise. "
    "When document excerpts are provided, use them if they are relevant and cite them like [1] or [2]. "
    "If the user asks about the documents and the answer is not in the excerpts, say you could not "
    "find it in the documents instead of guessing."
)


def build_prompt(question, summary, turns, chunks):
    parts = []
    if summary:
        parts.append(f"Summary of earlier conversation:\n{summary}")
    if turns:
        parts.append("Recent conversation:\n" + "\n".join(f"User: {u}\nNexaChat: {b}" for u, b in turns))
    if chunks:
        ctx = "\n\n".join(f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}" for i, c in enumerate(chunks, 1))
        parts.append(f"Document excerpts:\n{ctx}")
    parts.append(f"User question: {question}")
    return "\n\n".join(parts)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/docs")
def docs():
    return jsonify(rag.list_docs())


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file received"), 400
    name = secure_filename(f.filename)
    if os.path.splitext(name)[1].lower() not in ALLOWED:
        return jsonify(error="Only PDF, TXT and MD files are supported"), 400
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)
    try:
        n = rag.ingest(path, name)
    except Exception as e:
        return jsonify(error=f"Could not index file: {e}"), 500
    if n == 0:
        return jsonify(error="No readable text found (scanned PDFs need OCR)"), 400
    return jsonify(name=name, chunks=n)


@app.delete("/api/docs/<name>")
def delete_doc(name):
    rag.delete_doc(secure_filename(name))
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    question = (data.get("message") or "").strip()
    sid = data.get("session_id") or "default"
    if not question:
        return jsonify(error="Empty message"), 400
    try:
        chunks = rag.retrieve(question) if data.get("use_docs") else []
        summary, turns = memory.get(sid)
        answer = llm.generate(build_prompt(question, summary, turns, chunks), system=SYSTEM)
    except Exception as e:
        return jsonify(error=f"Model error: {e}"), 500
    memory.add_turn(sid, question, answer)
    sources = [{"n": i, "source": c["source"], "page": c["page"], "score": c["score"]} for i, c in enumerate(chunks, 1)]
    return jsonify(answer=answer, sources=sources)


@app.post("/api/transcribe")
def transcribe():
    audio = request.files.get("audio")
    if not audio:
        return jsonify(error="No audio received"), 400
    try:
        return jsonify(text=llm.transcribe(audio.read(), audio.filename or "audio.webm"))
    except Exception as e:
        return jsonify(error=f"Transcription failed: {e}"), 500


@app.post("/api/reset")
def reset():
    memory.reset((request.get_json(silent=True) or {}).get("session_id") or "default")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)
