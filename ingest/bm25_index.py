from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import Chroma
from ingest.embeddings import build_embeddings

load_dotenv()

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_texts: List[str]
    chunk_meta: List[Dict[str, Any]]

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in ranked:
            out.append(
                {
                    "text": self.chunk_texts[i],
                    "score": float(scores[i]),
                    "citation": self.chunk_meta[i],
                    "source": "bm25",
                }
            )
        return out


def _get_vs() -> Chroma:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    collection = os.getenv("CHROMA_COLLECTION", "incident_copilot")
    embeddings = build_embeddings()
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def build_bm25_from_chroma(index_path: str | None = None) -> str:
    """
    Builds a BM25 index from ALL chunks in the Chroma collection.
    Stores chunk texts + minimal citation metadata.
    """
    index_path = index_path or os.getenv("BM25_INDEX_PATH", ".bm25/index.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    vs = _get_vs()
    # Pull everything from Chroma. Chroma supports get with limit/offset; keep MVP simple:
    store = vs._collection  # chroma internal
    all_docs = store.get(include=["documents", "metadatas"])
    docs = all_docs.get("documents") or []
    metas = all_docs.get("metadatas") or []

    if not docs:
        raise RuntimeError("No documents found in Chroma. Run ingestion first.")

    tokenized = [_tokenize(t) for t in docs]
    bm25 = BM25Okapi(tokenized)

    payload = {
        "chunk_texts": docs,
        "chunk_meta": metas,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return index_path


def load_bm25(index_path: str | None = None) -> BM25Index:
    index_path = index_path or os.getenv("BM25_INDEX_PATH", ".bm25/index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    chunk_texts = payload["chunk_texts"]
    chunk_meta = payload["chunk_meta"]
    tokenized = [_tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(bm25=bm25, chunk_texts=chunk_texts, chunk_meta=chunk_meta)
