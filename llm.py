"""Thin wrappers around Gemini (chat + embeddings) and Groq Whisper (speech-to-text)."""
import os

CHAT_MODEL = "gemini-3.8-flash"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

_gemini = None
_groq = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        from google import genai
        _gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini


def embed(texts, query=False):
    """Return one embedding vector per text. Uses different task types for docs vs queries."""
    from google.genai import types
    task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    vectors = []
    for i in range(0, len(texts), 50):  # batch to stay inside API limits
        resp = _gemini_client().models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors += [e.values for e in resp.embeddings]
    return vectors


def generate(prompt, system=None):
    from google.genai import types
    resp = _gemini_client().models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.3),
    )
    return resp.text or ""


def transcribe(audio_bytes, filename="audio.webm"):
    global _groq
    if _groq is None:
        from groq import Groq
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = _groq.audio.transcriptions.create(file=(filename, audio_bytes), model=WHISPER_MODEL)
    return resp.text
