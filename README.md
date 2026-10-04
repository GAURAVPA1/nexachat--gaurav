# NexaChat: Multimodal AI Chatbot with RAG and Memory

A Python (Flask) and HTML chatbot that answers questions from your own documents, remembers the conversation, and accepts voice input in English, Hindi or Hinglish.

## Features

- **RAG (Retrieval-Augmented Generation):** upload PDF, TXT or MD files. Text is chunked, embedded with Gemini embeddings, stored in ChromaDB, and the most relevant chunks are retrieved for every question. Answers cite their sources like `[1] file.pdf, p.3`.
- **Conversation memory:** the last 6 turns are kept verbatim; older turns are compressed into a running summary by the LLM, so long chats stay coherent without growing the prompt.
- **Voice input:** microphone audio is transcribed by Groq Whisper. Replies can be read aloud with the browser's speech synthesis.
- **Multilingual:** replies in the language the user writes in.

## Architecture

```
Browser (HTML/JS)  ->  Flask API (app.py)
                         |-- rag.py     : extract -> chunk -> embed -> ChromaDB -> retrieve
                         |-- memory.py  : recent turns + running summary (data/sessions.json)
                         |-- llm.py     : Gemini (chat, embeddings) and Groq Whisper
```

## Setup

1. Install Python 3.10 to 3.12.
2. In this folder run:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your keys:
   - Gemini key: https://aistudio.google.com/apikey
   - Groq key: https://console.groq.com/keys
4. Run `python app.py` and open http://127.0.0.1:5000

## Limitations

- Scanned (image-only) PDFs have no extractable text and need OCR.
- Memory is stored in a local JSON file, which suits a single user. A database would be needed for many users.
- No user login. Documents are shared across all sessions on the same server.

## Ideas to extend

- Hybrid search (keyword + vector) and re-ranking
- Streaming responses
- ElevenLabs text-to-speech
- Evaluation set to measure answer quality
