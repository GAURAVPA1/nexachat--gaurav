"""RAG pipeline: extract text -> chunk -> embed -> store in ChromaDB -> retrieve."""
import re
import chromadb
from pypdf import PdfReader
import llm

_client = chromadb.PersistentClient(path="data/chroma")
_col = _client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def extract_pages(path, name):
    """Return a list of (page_number, text). Plain text files count as page 1."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [(1, f.read())]


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", i + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return [c for c in chunks if len(c) > 40]


def ingest(path, name):
    """Index a file. Re-uploading the same name replaces the old version."""
    delete_doc(name)
    texts, metas = [], []
    for page, text in extract_pages(path, name):
        for c in chunk_text(text):
            texts.append(c)
            metas.append({"source": name, "page": page})
    if not texts:
        return 0
    ids = [f"{name}::{i}" for i in range(len(texts))]
    _col.add(ids=ids, documents=texts, metadatas=metas, embeddings=llm.embed(texts))
    return len(texts)


def retrieve(query, k=4):
    if _col.count() == 0:
        return []
    emb = llm.embed([query], query=True)[0]
    res = _col.query(query_embeddings=[emb], n_results=min(k, _col.count()))
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": text, "source": meta["source"], "page": meta["page"], "score": round(1 - dist, 3)})
    return out


def list_docs():
    metas = _col.get(include=["metadatas"])["metadatas"]
    counts = {}
    for m in metas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]


def delete_doc(name):
    _col.delete(where={"source": name})
