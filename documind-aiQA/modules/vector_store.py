"""Persistent FAISS vector store (cosine similarity via normalised inner product)."""
import faiss
import numpy as np

from utils.helpers import VSTORE, read_json, write_json


class VectorStore:
    def __init__(self, path=VSTORE):
        self.idx_file, self.meta_file = path / "index.faiss", path / "meta.json"
        self.index, self.chunks, self.model = None, [], None
        self._load()

    def _load(self):
        try:
            if self.idx_file.exists() and self.meta_file.exists():
                self.index = faiss.read_index(str(self.idx_file))
                m = read_json(self.meta_file, {})
                self.chunks, self.model = m.get("chunks", []), m.get("model")
                if self.index.ntotal != len(self.chunks):
                    raise ValueError("index/metadata mismatch")
        except Exception:  # corrupted store -> start fresh instead of crashing
            self.index, self.chunks, self.model = None, [], None

    def save(self):
        if self.index is None or not self.chunks:
            self.idx_file.unlink(missing_ok=True)
            self.meta_file.unlink(missing_ok=True)
            return
        faiss.write_index(self.index, str(self.idx_file))
        write_json(self.meta_file, {"model": self.model, "chunks": self.chunks})

    @property
    def count(self) -> int:
        return len(self.chunks)

    def add(self, vecs, chunks, model: str):
        if self.model and self.model != model:
            raise ValueError("The embedding model changed. Rebuild the index in Settings first.")
        vecs = np.asarray(vecs, dtype="float32")
        if self.index is None:
            self.index, self.model = faiss.IndexFlatIP(vecs.shape[1]), model
        self.index.add(vecs)
        self.chunks += chunks
        self.save()

    def search(self, qvec, k: int, doc_ids=None):
        """Top-k chunks (optionally restricted to doc_ids) with similarity scores."""
        if self.index is None:
            return []
        n = self.index.ntotal if doc_ids else min(k, self.index.ntotal)
        scores, ids = self.index.search(np.asarray(qvec, dtype="float32").reshape(1, -1), n)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i >= 0 and (not doc_ids or self.chunks[i]["doc_id"] in doc_ids):
                out.append({**self.chunks[i], "score": float(s)})
            if len(out) == k:
                break
        return out

    def chunks_for(self, doc_id: str):
        return [c for c in self.chunks if c["doc_id"] == doc_id]

    def _rebuild(self, keep):
        if not keep:
            self.index, self.chunks, self.model = None, [], None
        else:
            vecs = self.index.reconstruct_n(0, self.index.ntotal)[keep]
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            self.chunks = [self.chunks[i] for i in keep]
        self.save()

    def remove_document(self, doc_id: str):
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        if len(keep) != len(self.chunks):
            self._rebuild(keep)

    def clear(self):
        self._rebuild([])

    def reindex(self, embedder):
        """Re-embed all stored chunks (use after changing the embedding model)."""
        if not self.chunks:
            return
        vecs = embedder.encode([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.model = embedder.name
        self.save()
