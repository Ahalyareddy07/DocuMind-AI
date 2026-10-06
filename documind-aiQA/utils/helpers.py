"""Shared helpers: paths, persistent settings and JSON storage."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
VSTORE = DATA / "vectorstore"
for _p in (UPLOADS, VSTORE):
    _p.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA / "settings.json"


def defaults() -> dict:
    """Default settings; environment variables provide the initial values."""
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "gemma3"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "temperature": 0.2, "top_k": 5,
        "chunk_size": 900, "chunk_overlap": 150,
        "max_upload_mb": int(os.getenv("MAX_UPLOAD_MB", "20")),
        "theme": "dark",
    }


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_settings() -> dict:
    cfg = defaults()
    cfg.update(read_json(SETTINGS_FILE, {}))
    return cfg


def save_settings(cfg: dict):
    write_json(SETTINGS_FILE, cfg)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
