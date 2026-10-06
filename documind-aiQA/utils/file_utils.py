"""Upload validation, filename sanitising and the document registry."""
import re
from pathlib import Path

from .helpers import DATA, UPLOADS, read_json, write_json

ALLOWED = {".pdf", ".docx", ".txt"}
DOCS_FILE = DATA / "documents.json"


def sanitize_filename(name: str) -> str:
    """Strip any path and unsafe characters from a user-supplied filename."""
    p = Path(name).name
    stem = re.sub(r"[^\w\-. ]", "_", Path(p).stem)[:80].strip(" .") or "document"
    return stem + Path(p).suffix.lower()


def validate_upload(name: str, data: bytes, max_mb: int):
    """Raise ValueError with a friendly message if the upload is unacceptable."""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported file type '{ext or 'unknown'}'. Please upload PDF, DOCX or TXT.")
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"File is too large (limit {max_mb} MB).")
    # Check magic bytes so a renamed file cannot sneak through.
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise ValueError("This file is not a valid PDF.")
    if ext == ".docx" and not data.startswith(b"PK"):
        raise ValueError("This file is not a valid DOCX.")
    if ext == ".txt" and b"\x00" in data[:4096]:
        raise ValueError("This file does not look like plain text.")


def save_upload(doc_id: str, name: str, data: bytes) -> Path:
    path = UPLOADS / f"{doc_id}_{name}"
    path.write_bytes(data)  # stored only; never executed
    return path


def load_docs() -> dict:
    return read_json(DOCS_FILE, {})


def add_doc(meta: dict):
    docs = load_docs()
    docs[meta["id"]] = meta
    write_json(DOCS_FILE, docs)


def update_doc(doc_id: str, **fields):
    docs = load_docs()
    if doc_id in docs:
        docs[doc_id].update(fields)
        write_json(DOCS_FILE, docs)


def delete_doc(doc_id: str):
    docs = load_docs()
    meta = docs.pop(doc_id, None)
    if meta:
        (UPLOADS / meta.get("path", "")).unlink(missing_ok=True)
    write_json(DOCS_FILE, docs)


def clear_docs():
    for d in list(load_docs()):
        delete_doc(d)
