"""Retrieval over the prebuilt index, exposed as a LangChain retriever.

Chunks are indexed with a contextual header (subject | file | page) prepended to their text.

Modes (RETRIEVAL_MODE):
  dense          FAISS cosine similarity on sentence-transformer embeddings (default: best on
                 eval/retrieval_gold.csv - see scripts/benchmark_retrieval.py)
  hybrid         dense + BM25 keyword search, combined with weighted Reciprocal Rank Fusion
  hybrid+rerank  hybrid, then a cross-encoder re-scores the top candidates (local only: needs PyTorch)

All modes apply the optional subject filter and drop duplicate passages before returning top-k.
"""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from . import config
from .embeddings import get_embeddings

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"

RRF_K = 60           # standard Reciprocal Rank Fusion constant
CANDIDATES = 40      # candidates taken from each of dense and BM25 search
RERANK_TOP = 25      # fused candidates passed to the cross-encoder


class IndexNotBuiltError(RuntimeError):
    pass


def chunk_header(chunk: dict) -> str:
    """Context prepended to a chunk for embedding / keyword matching (file, subject, location)."""
    parts = [chunk.get("subject", ""), Path(chunk["source"]).stem, f"{chunk.get('unit', 'p.')} {chunk['page']}"]
    return " | ".join(p for p in parts if p)


# ---------------------------------------------------------------- BM25

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by for from has have how in into is it its of on or that the this
to was were what when where which who why will with do does did can you your we our i me my they them their
there these those than then so if not no but about also more most other some such only own same very just
s t should would could""".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        # light plural folding so "queues"/"queue" and "trees"/"tree" match
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        docs = [tokenize(t) for t in texts]
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.k1, self.b = k1, b
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings[term].append((i, tf))
        self.postings = {
            term: (np.array([i for i, _ in p]), np.array([tf for _, tf in p], dtype=np.float32))
            for term, p in postings.items()
        }
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            scores[ids] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores


# ---------------------------------------------------------------- reranker

@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL, max_length=512)


def rerank_available() -> bool:
    import importlib.util
    import os

    return config.RERANK_MODEL != "none" and not os.getenv("VERCEL") and \
        importlib.util.find_spec("sentence_transformers") is not None


def default_mode() -> str:
    mode = config.RETRIEVAL_MODE
    if mode == "hybrid+rerank" and not rerank_available():
        return "hybrid"
    return mode


# ---------------------------------------------------------------- retriever

class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any  # _Store
    k: int = config.TOP_K
    mode: str = "hybrid"
    subject: str | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        s = self.store
        allowed = s.subject_mask(self.subject)
        q = np.asarray(s.embeddings.embed_query(s.query_instruction + query), dtype=np.float32)
        dense = s.vectors @ q  # cosine similarity (vectors are normalized)
        dense_masked = np.where(allowed, dense, -np.inf)
        dense_rank = np.argsort(-dense_masked)[:CANDIDATES]
        dense_rank = [i for i in dense_rank if allowed[i]]

        if self.mode == "dense":
            ranked = dense_rank
        else:
            sparse = np.where(allowed, s.bm25.scores(query), 0)
            sparse_rank = [i for i in np.argsort(-sparse)[:CANDIDATES] if sparse[i] > 0]
            fused: dict[int, float] = defaultdict(float)
            for ranking, weight in ((dense_rank, 1.0), (sparse_rank, config.BM25_WEIGHT)):
                for rank, i in enumerate(ranking):
                    fused[i] += weight / (RRF_K + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)

        rerank_scores = {}
        if self.mode == "hybrid+rerank" and ranked:
            top = ranked[:RERANK_TOP]
            pairs = [(query, s.search_text[i]) for i in top]
            ce = get_reranker().predict(pairs, show_progress_bar=False)
            rerank_scores = {i: float(sc) for i, sc in zip(top, ce)}
            ranked = sorted(top, key=rerank_scores.get, reverse=True)

        docs, seen = [], set()
        for i in ranked:
            if s.text_hash[i] in seen:  # same passage in several files (e.g. duplicated quizzes)
                continue
            seen.add(s.text_hash[i])
            chunk = s.chunks[i]
            docs.append(Document(page_content=chunk["text"], metadata={
                "source": chunk["source"],
                "subject": chunk.get("subject", ""),
                "unit": chunk.get("unit", "p."),
                "page": chunk["page"],
                "score": float(dense[i]),
                "rerank_score": rerank_scores.get(i),
            }))
            if len(docs) == self.k:
                break
        return docs


class _Store:
    """Everything loaded once per process: index vectors, chunks, BM25, embeddings."""

    def __init__(self, index_dir: Path):
        self.meta = load_meta(index_dir)
        index = faiss.read_index(str(index_dir / INDEX_FILE))
        self.vectors = index.reconstruct_n(0, index.ntotal)
        self.chunks = json.loads((index_dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        self.search_text = [f"{chunk_header(c)}\n{c['text']}" for c in self.chunks]
        self.bm25 = BM25(self.search_text)
        self.text_hash = [hashlib.md5(c["text"].encode("utf-8")).hexdigest() for c in self.chunks]
        self.subjects = np.array([c.get("subject", "") for c in self.chunks])
        self.query_instruction = self.meta.get("query_instruction", "")
        self.embeddings = get_embeddings(self.meta["embedding_model"])

    def subject_mask(self, subject: str | None) -> np.ndarray:
        if not subject:
            return np.ones(len(self.chunks), dtype=bool)
        return self.subjects == subject


def load_meta(index_dir: Path = config.INDEX_DIR) -> dict:
    path = Path(index_dir) / META_FILE
    if not path.exists():
        raise IndexNotBuiltError(
            "The search index has not been built yet. Put the course files in data/pdfs/ "
            "and run: python scripts/build_index.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def _store(index_dir: str) -> _Store:
    return _Store(Path(index_dir))


def min_score(index_dir: Path = config.INDEX_DIR) -> float:
    """Abstention threshold: env MIN_SCORE, else the value calibrated for the index's embedding model."""
    if config.MIN_SCORE is not None:
        return config.MIN_SCORE
    return float(_store(str(index_dir)).meta.get("min_score", 0.15))


def get_retriever(k: int | None = None, mode: str | None = None, subject: str | None = None,
                  index_dir: Path = config.INDEX_DIR) -> HybridRetriever:
    return HybridRetriever(store=_store(str(index_dir)), k=k or config.TOP_K,
                           mode=mode or default_mode(), subject=subject or None)
