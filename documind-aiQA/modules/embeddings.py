"""Embedding wrapper (swap the model by changing the name in Settings/.env)."""
import numpy as np


class EmbeddingError(Exception):
    pass


class Embedder:
    def __init__(self, name: str):
        self.name, self._model = name, None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.name)
            except Exception as e:
                raise EmbeddingError(f"Could not load embedding model '{self.name}'. "
                                     "The first run needs internet to download it.") from e
        return self._model

    def encode(self, texts) -> np.ndarray:
        try:
            return np.asarray(self._load().encode(list(texts), batch_size=32, normalize_embeddings=True,
                                                  show_progress_bar=False), dtype="float32")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError("Embedding generation failed.") from e


_CACHE: dict = {}


def get_embedder(name: str) -> Embedder:
    if name not in _CACHE:
        _CACHE[name] = Embedder(name)
    return _CACHE[name]
