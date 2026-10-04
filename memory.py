"""Conversation memory: keep recent turns verbatim, compress older turns into a running summary."""
import json
import os
import threading
import llm

FILE = "data/sessions.json"
KEEP_TURNS = 6
_lock = threading.Lock()
_sessions = {}

if os.path.exists(FILE):
    try:
        with open(FILE, encoding="utf-8") as f:
            _sessions = json.load(f)
    except (OSError, ValueError):
        _sessions = {}


def _save():
    os.makedirs("data", exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(_sessions, f, ensure_ascii=False)


def get(sid):
    with _lock:
        s = _sessions.setdefault(sid, {"summary": "", "turns": []})
        return s["summary"], [tuple(t) for t in s["turns"]]


def add_turn(sid, user, bot):
    with _lock:
        s = _sessions.setdefault(sid, {"summary": "", "turns": []})
        s["turns"].append([user, bot])
        if len(s["turns"]) > KEEP_TURNS:
            older, s["turns"] = s["turns"][:-KEEP_TURNS], s["turns"][-KEEP_TURNS:]
            s["summary"] = _summarize(s["summary"], older)
        _save()


def reset(sid):
    with _lock:
        _sessions.pop(sid, None)
        _save()


def _summarize(old_summary, turns):
    convo = "\n".join(f"User: {u}\nAssistant: {b}" for u, b in turns)
    prompt = (
        "Update this running conversation summary. Keep names, facts, preferences and open "
        "questions. Max 120 words.\n\n"
        f"Existing summary:\n{old_summary or '(none)'}\n\nNew messages:\n{convo}"
    )
    try:
        return llm.generate(prompt)
    except Exception:
        return old_summary  # never lose the chat because summarizing failed
