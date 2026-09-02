"""Embedding phi — Definition 7.

phi(Th) maps a threat to R^n via a transformer language model B operating on the textual
serialization of its event sequence:

  (i)  |E| = 1 :  phi(Th) = B(serialize(e1))
  (ii) |E| > 1 :  phi(Th) = AGG(B(serialize(e1)), ..., B(serialize(ek)))

B is a local sentence-transformer (default all-MiniLM-L6-v2), so embeddings are free,
offline-capable and reproducible. AGG defaults to mean pooling (Def 7 lists mean / attention
pooling / transformer encoder as options).
"""

from __future__ import annotations

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model_cache: dict[str, object] = {}


def get_model(model_name: str = _DEFAULT_MODEL):
    """Lazily load and cache the sentence-transformer (uses MPS/CUDA if available)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        device = _pick_device()
        _model_cache[model_name] = SentenceTransformer(model_name, device=device)
    return _model_cache[model_name]


def _pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def embed_texts(texts: list[str], model_name: str = _DEFAULT_MODEL, batch_size: int = 256) -> np.ndarray:
    """B(serialize(e)) for a batch of serialized events -> (N, n) float array (Def 7, B)."""
    model = get_model(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float64)


def aggregate(embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
    """AGG : (R^n)* -> R^n  (Def 7, aggregation for multi-event threats).

    `embeddings` is (k, n) for the k events of one threat. Returns (n,).
    """
    if embeddings.ndim == 1:
        return embeddings
    if embeddings.shape[0] == 1:
        return embeddings[0]
    if method == "mean":
        return embeddings.mean(axis=0)
    if method == "max":
        return embeddings.max(axis=0)
    raise ValueError(f"unknown aggregation method: {method}")
