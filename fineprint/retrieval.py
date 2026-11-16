"""Retrieval (the RAG step): per-category top-k clause retrieval over a FAISS index.

Why retrieve at all when a lease fits in context?  Two honest reasons:
  1. The same pipeline has to scale to lease + addenda + house rules + a state
     tenant handbook, where stuffing everything in context hurts accuracy and cost.
  2. Per-category retrieval gives the planner a checklist-shaped view and lets us
     measure retrieval recall separately from planning quality.
`mode="full"` bypasses retrieval so the eval can quantify what RAG costs/gains here.
"""
from __future__ import annotations

import functools
import hashlib
import math
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import RISK_CATEGORIES, SETTINGS
from .schemas import Clause


class HashingEmbeddings(Embeddings):
    """Dependency-free lexical embeddings (hashed bag of words, L2-normalised).

    Used for tests and as an offline fallback. Quality is worse than a real
    sentence-embedding model, which is the point of keeping both selectable.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9']+", text.lower()):
            if len(tok) < 3:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@functools.lru_cache(maxsize=4)
def _cached_embeddings(kind: str, model_name: str) -> Embeddings:
    if kind == "hashing":
        return HashingEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings  # imported lazily: heavy
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def get_embeddings(kind: str | None = None) -> Embeddings:
    """Embedding models are cached so repeated pipeline runs do not reload weights."""
    return _cached_embeddings(kind or SETTINGS.embeddings, SETTINGS.embedding_model)


def build_index(clauses: list[Clause], embeddings: Embeddings | None = None) -> FAISS:
    embeddings = embeddings or get_embeddings()
    docs = [Document(page_content=f"{c.title}. {c.text}", metadata={"clause_id": c.id}) for c in clauses]
    return FAISS.from_documents(docs, embeddings)


def retrieve_for_checklist(index: FAISS, clauses: list[Clause], k: int | None = None) -> tuple[list[Clause], dict[str, list[str]]]:
    """Return (clauses in document order, {category: [clause_ids]}) for the checklist."""
    k = k or SETTINGS.top_k_per_category
    by_id = {c.id: c for c in clauses}
    hits: dict[str, list[str]] = {}
    chosen: set[str] = set()
    for cat, spec in RISK_CATEGORIES.items():
        docs = index.similarity_search(spec["query"], k=k)
        ids = [d.metadata["clause_id"] for d in docs]
        hits[cat] = ids
        chosen.update(ids)
    ordered = [by_id[cid] for cid in sorted(chosen)]
    return ordered, hits
