"""Persistent chat history stored in data/chats.json."""
from utils.helpers import DATA, new_id, now, read_json, write_json

FILE = DATA / "chats.json"


def _load() -> dict:
    return read_json(FILE, {})


def create_chat(title: str) -> str:
    chats, cid = _load(), new_id()
    chats[cid] = {"id": cid, "title": title or "New chat", "created": now(), "updated": now(), "messages": []}
    write_json(FILE, chats)
    return cid


def list_chats():
    return sorted(_load().values(), key=lambda c: c["updated"], reverse=True)


def get_chat(cid):
    return _load().get(cid)


def add_messages(cid, msgs):
    chats = _load()
    if cid in chats:
        chats[cid]["messages"] += msgs
        chats[cid]["updated"] = now()
        write_json(FILE, chats)


def rename_chat(cid, title):
    chats = _load()
    if cid in chats and title.strip():
        chats[cid]["title"] = title.strip()[:60]
        write_json(FILE, chats)


def delete_chat(cid):
    chats = _load()
    chats.pop(cid, None)
    write_json(FILE, chats)


def clear_chats():
    write_json(FILE, {})


def count_questions() -> int:
    return sum(1 for c in _load().values() for m in c["messages"] if m["role"] == "user")
