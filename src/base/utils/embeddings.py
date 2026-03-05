# base/utils/embeddings.py
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from config.config import settings


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


# Process-wide cache to avoid loading large embedding models multiple times.
# Key: (provider, model, openai_api_key_if_used)
_EMBEDDER_CACHE: dict[tuple[str, str, str], tuple[Embedder, int]] = {}


def get_embedder() -> tuple[Embedder, int]:
    """
    Returns (embedder, dimension) based on settings.embeddings_provider / embeddings_model.
    Supported providers: "sentence_transformers" (default), "openai"

    Uses a process-wide cache so we don't load large models multiple times.
    """
    provider = (settings.embeddings_provider or "sentence_transformers").lower()

    if provider == "openai":
        from openai import OpenAI

        api_key = settings.openai_api_key or ""
        model = settings.embeddings_model or "text-embedding-3-small"
        cache_key = (provider, model, api_key)

        cached = _EMBEDDER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        client = OpenAI(api_key=api_key)

        class OpenAIEmbedder:
            def encode(self, texts: Sequence[str]) -> np.ndarray:
                resp = client.embeddings.create(model=model, input=list(texts))
                return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

        dim = 1536  # text-embedding-3-small
        out: tuple[Embedder, int] = (OpenAIEmbedder(), dim)
        _EMBEDDER_CACHE[cache_key] = out
        return out

    # default: sentence-transformers (local)
    from sentence_transformers import SentenceTransformer

    st_model_name = settings.embeddings_model or "all-MiniLM-L6-v2"
    cache_key = (provider, st_model_name, "")

    cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    st_model = SentenceTransformer(st_model_name)

    dim_opt = st_model.get_sentence_embedding_dimension()
    dim = int(dim_opt) if dim_opt is not None else 384

    class STEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.asarray(
                st_model.encode(
                    list(texts), convert_to_numpy=True, batch_size=32, show_progress_bar=False
                ),
                dtype=np.float32,
            )

    out = (STEmbedder(), dim)
    _EMBEDDER_CACHE[cache_key] = out
    return out
