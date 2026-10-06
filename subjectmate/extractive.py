"""Answer without an LLM: quote the sentences from the retrieved passages that best match the question.

Used when no LLM is configured, so the app still answers (with citations) from the course material.
"""
import re

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_SENTENCES = 6
MIN_SENTENCE_SCORE = 0.3


def _location(d: Document) -> str:
    return f"{d.metadata['source']}, {d.metadata.get('unit', 'p.')} {d.metadata['page']}"


def extractive_answer(question: str, docs: list[Document], embeddings: Embeddings) -> str:
    candidates: list[tuple[str, Document]] = []
    seen = set()
    for d in docs:
        for sentence in _SPLIT.split(d.page_content):
            sentence = " ".join(sentence.split())
            letters = sum(c.isalpha() for c in sentence)
            if 25 <= len(sentence) <= 500 and letters >= 15 and sentence.lower() not in seen:
                seen.add(sentence.lower())
                candidates.append((sentence, d))
    if not candidates:
        return "\n\n".join(f"> {d.page_content[:600]}\n\n[{_location(d)}]" for d in docs[:2])

    vectors = np.asarray(embeddings.embed_documents([s for s, _ in candidates]), dtype=np.float32)
    query = np.asarray(embeddings.embed_query(question), dtype=np.float32)
    scores = vectors @ query
    best = np.argsort(-scores)[:MAX_SENTENCES]
    picked = sorted(i for i in best if scores[i] >= MIN_SENTENCE_SCORE) or sorted(best[:2])

    lines = [f"- {candidates[i][0]} [{_location(candidates[i][1])}]" for i in picked]
    return "Most relevant points from the course material:\n\n" + "\n".join(lines)
