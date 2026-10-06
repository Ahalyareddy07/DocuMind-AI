"""Document summarisation (map-reduce over stored chunks)."""
from modules.rag_pipeline import llm_chat

BUDGET = {"Short": 6, "Medium": 12, "Detailed": 24}  # max chunks fed to the model
LENGTH = {"Short": "about 100-150 words", "Medium": "about 250-350 words",
          "Detailed": "a thorough write-up of 500+ words"}
FORMATS = {
    "Short Summary": "Write a concise summary ({length}).",
    "Detailed Summary": "Write a section-by-section explanation with a heading per section ({length}).",
    "Key Points": "List the most important points as markdown bullets ({length}).",
    "Executive Summary": "Write an executive summary with the headings Overview, Main Concepts, "
                         "Important Findings and Conclusions ({length}).",
}
SYSTEM = "You summarise documents faithfully. Use ONLY the supplied text; never add outside facts."


def summarize(doc_id, store, cfg, kind="Short Summary", length="Medium") -> str:
    chunks = store.chunks_for(doc_id)
    if not chunks:
        raise ValueError("This document has no processed content.")
    n = BUDGET[length]
    if len(chunks) > n:  # evenly sample across the whole document
        chunks = [chunks[int(i * len(chunks) / n)] for i in range(n)]
    texts = ["\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks[i:i + 6])
             for i in range(0, len(chunks), 6)]
    if len(texts) > 1:  # map step: condense each batch, then reduce
        texts = [llm_chat(cfg, SYSTEM, "Summarise the key content of this excerpt in ~120 words:\n\n" + t)
                 for t in texts]
    prompt = FORMATS[kind].format(length=LENGTH[length]) + "\n\nDocument text:\n\n" + "\n\n".join(texts)
    return llm_chat(cfg, SYSTEM, prompt)
