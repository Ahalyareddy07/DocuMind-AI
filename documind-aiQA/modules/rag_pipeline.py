"""RAG pipeline: question -> embedding -> similarity search -> LLM (Ollama) -> answer + sources."""
import re

import requests

NOT_FOUND = "I couldn't find this information in the uploaded documents."
MIN_SCORE = 0.15  # below this similarity a chunk is considered irrelevant
GENERIC = re.compile(r"\b(about|summar\w*|overview|key points?|main (topic|idea|concept)|important points?)\b", re.I)


class LLMUnavailable(Exception):
    pass


def list_models(base_url: str):
    """Installed Ollama models, or None if Ollama is unreachable."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def llm_chat(cfg: dict, system: str, user: str, history=None, json_mode=False) -> str:
    """Single place that talks to the LLM -- replace this function to use another backend."""
    msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
    payload = {"model": cfg["model"], "messages": msgs, "stream": False,
               "options": {"temperature": cfg["temperature"]}}
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{cfg['ollama_base_url']}/api/chat", json=payload, timeout=600)
    except requests.RequestException as e:
        raise LLMUnavailable(
            f"Cannot reach Ollama at {cfg['ollama_base_url']}. Install it from https://ollama.com, "
            f"start it (`ollama serve`) and run `ollama pull {cfg['model']}`.") from e
    if r.status_code == 404:
        raise LLMUnavailable(f"Model '{cfg['model']}' is not installed. Run: ollama pull {cfg['model']}")
    if not r.ok:
        raise LLMUnavailable(f"The AI model returned an error (HTTP {r.status_code}).")
    return r.json()["message"]["content"].strip()


def answer_question(question, store, embedder, cfg, doc_ids, history=()):
    question = question.strip()
    if len(question) < 3:
        raise ValueError("Please enter a valid question (at least 3 characters).")
    if not doc_ids or store.count == 0:
        raise ValueError("Please select at least one processed document.")
    if store.model and store.model != embedder.name:
        raise ValueError("The embedding model changed. Go to Settings -> Rebuild index.")

    hits = [h for h in store.search(embedder.encode([question])[0], cfg["top_k"], set(doc_ids))
            if h["score"] >= MIN_SCORE]
    if GENERIC.search(question):  # broad questions: also include the opening of each document
        seen = {h["chunk_id"] for h in hits}
        for d in doc_ids:
            for c in store.chunks_for(d)[:2]:
                if c["chunk_id"] not in seen:
                    hits.append({**c, "score": 0.0})
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(f"[{i}] ({h['doc_name']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))
    system = ("You are DocuMind AI, a document assistant. Answer ONLY from the provided context. "
              f"If the context does not contain the answer, reply exactly: {NOT_FOUND} "
              "Never invent facts. Be clear and concise, use bullet points when helpful, "
              "and cite sources inline like [1], [2].")
    hist = [{"role": m["role"], "content": m["content"]} for m in list(history)[-6:]]
    answer = llm_chat(cfg, system, f"Context:\n{context}\n\nQuestion: {question}", hist)
    if NOT_FOUND in answer:
        return {"answer": NOT_FOUND, "sources": []}
    return {"answer": answer, "sources": hits}
