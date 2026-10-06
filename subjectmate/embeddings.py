"""Sentence-transformers embeddings, computed locally or through the Hugging Face Inference API.

Both backends use the same model, so an index built locally can be queried from Vercel,
where installing PyTorch is not possible.
"""
import importlib.util
import os

import numpy as np
from langchain_core.embeddings import Embeddings

from . import config


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class LocalEmbeddings(Embeddings):
    """Runs the model on this machine with sentence-transformers (used to build the index)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 500
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HFInferenceEmbeddings(Embeddings):
    """Calls the same sentence-transformers model through the Hugging Face Inference API."""

    def __init__(self, model_name: str, token: str | None = None):
        from huggingface_hub import InferenceClient

        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Create a free read token at "
                "https://huggingface.co/settings/tokens and add it as an environment variable."
            )
        self.client = InferenceClient(provider="hf-inference", api_key=token)
        self.model = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            vectors = np.asarray(self.client.feature_extraction(batch, model=self.model), dtype=np.float32)
            if vectors.ndim == 3:  # token-level output -> mean pooling
                vectors = vectors.mean(axis=1)
            out.append(_normalize(vectors.reshape(len(batch), -1)))
        return np.vstack(out).tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = np.asarray(self.client.feature_extraction(text, model=self.model), dtype=np.float32)
        while vector.ndim > 1:  # (1, dim) or token-level (tokens, dim) -> (dim,)
            vector = vector.mean(axis=0)
        return _normalize(vector).tolist()


def get_embeddings(model_name: str | None = None, backend: str | None = None) -> Embeddings:
    model_name = model_name or config.EMBEDDING_MODEL
    backend = backend or config.EMBEDDING_BACKEND
    if backend == "auto":
        has_local = importlib.util.find_spec("sentence_transformers") is not None
        backend = "local" if has_local and not os.getenv("VERCEL") else "hf-api"
    if backend == "local":
        return LocalEmbeddings(model_name)
    if backend == "hf-api":
        return HFInferenceEmbeddings(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")
