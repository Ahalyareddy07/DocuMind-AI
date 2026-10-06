"""Quiz generation from document chunks (JSON output from the LLM)."""
import json
import random
import re

from modules.rag_pipeline import llm_chat

TYPES = {"Multiple Choice": "mcq", "True/False": "tf", "Short Answer": "short"}
SCHEMA = {
    "mcq": '{"question":"...","options":["...","...","...","..."],"answer":"A","explanation":"..."}',
    "tf": '{"question":"<a factual statement>","answer":"True","explanation":"..."}',
    "short": '{"question":"...","answer":"<1-2 sentence model answer>","explanation":"..."}',
}


def _normalise(q, kind):
    question, ans = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
    if not question or not ans:
        return None
    item = {"type": kind, "question": question, "answer": ans,
            "explanation": str(q.get("explanation", "")).strip()}
    if kind == "mcq":
        opts = [re.sub(r"^[A-Da-d][\).:]\s*", "", str(o)).strip() for o in q.get("options", [])][:4]
        if len(opts) < 4:
            return None
        letter = ans[:1].upper()
        if letter not in list("ABCD") or len(ans) > 2:
            match = [i for i, o in enumerate(opts) if o.lower() == ans.lower()]
            if not match:
                return None
            letter = "ABCD"[match[0]]
        item.update(options=opts, answer=letter)
    elif kind == "tf":
        item["answer"] = "True" if ans.lower().startswith("t") else "False"
    return item


def generate_quiz(doc_ids, store, cfg, n, qtype):
    kind = TYPES[qtype]
    chunks = [c for d in doc_ids for c in store.chunks_for(d)]
    if not chunks:
        raise ValueError("No processed content found for the selected documents.")
    sample = random.sample(chunks, min(len(chunks), 10))
    context = "\n\n".join(f"({c['doc_name']}, p.{c['page']}) {c['text']}" for c in sample)
    prompt = (f"Create exactly {n} {qtype} questions based ONLY on the text below. "
              f'Return JSON: {{"questions":[{SCHEMA[kind]}, ...]}}\n\nText:\n{context}')
    raw = llm_chat(cfg, "You are a careful exam-question writer. Output valid JSON only.", prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    items = [_normalise(q, kind) for q in data.get("questions", []) if isinstance(q, dict)]
    items = [i for i in items if i]
    if not items:
        raise ValueError("The model returned an unusable quiz. Please try again (a larger model helps).")
    return items[:n]
