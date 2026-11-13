import threading
from typing import Any


class _LazyEmbeddingModel:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
        return self._model.encode(texts, normalize_embeddings=True)


embedding_model = _LazyEmbeddingModel()