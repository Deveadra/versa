from __future__ import annotations

import numpy as np


class Embeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))
