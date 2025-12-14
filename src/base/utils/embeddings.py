# base/utils/embeddings.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from config.config import settings


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def get_embedder() -> tuple[Embedder, int]:
    """
    Returns (embedder, dimension) based on settings.embeddings_provider / embeddings_model.
    Supported providers: "sentence_transformers" (default), "openai"
    """
    provider = (settings.embeddings_provider or "sentence_transformers").lower()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        model = settings.embeddings_model or "text-embedding-3-small"

        class OpenAIEmbedder:
            def encode(self, texts: Sequence[str]) -> np.ndarray:
                resp = client.embeddings.create(model=model, input=list(texts))
                return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

        # Text-embedding-3-small is 1536-d
        dim = 1536
        return OpenAIEmbedder(), dim

    # default: sentence-transformers (local, fast)
    from sentence_transformers import SentenceTransformer

    st_model_name = settings.embeddings_model or "all-MiniLM-L6-v2"
    st_model = SentenceTransformer(st_model_name)

    class STEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.asarray(
                st_model.encode(
                    list(texts), convert_to_numpy=True, batch_size=32, show_progress_bar=False
                ),
                dtype=np.float32,
            )

    dim = 384  # MiniLM-L6-v2
    return STEmbedder(), dim
