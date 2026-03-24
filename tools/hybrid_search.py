from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from tools.kb_search import search_incident_knowledge_base
from ingest.bm25_index import load_bm25

load_dotenv()


def _normalize_scores(items: List[Dict[str, Any]], key: str = "score") -> List[Dict[str, Any]]:
    if not items:
        return items
    scores = [float(x[key]) for x in items]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        for x in items:
            x[f"{key}_norm"] = 1.0
        return items
    for x in items:
        x[f"{key}_norm"] = (float(x[key]) - mn) / (mx - mn)
    return items


def hybrid_search(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval without KG:
      - vector retrieval from Chroma (semantic)
      - BM25 retrieval over same chunks (lexical)
      - merge + rerank
    """
    top_k = k or int(os.getenv("TOP_K", "6"))

    vec = search_incident_knowledge_base(query, k=top_k)
    for r in vec:
        r["source"] = "vector"

    bm25 = load_bm25()
    lex = bm25.search(query, k=top_k)

    vec = _normalize_scores(vec, "score")
    lex = _normalize_scores(lex, "score")

    # Merge by chunk_id if present
    merged: Dict[str, Dict[str, Any]] = {}

    def key_of(item: Dict[str, Any]) -> str:
        c = item.get("citation") or {}
        return c.get("chunk_id") or f"{item['source']}::{c.get('source_path')}::{c.get('chunk_local_id')}"

    for item in vec + lex:
        kkey = key_of(item)
        if kkey not in merged:
            merged[kkey] = item
            merged[kkey]["hybrid_score"] = 0.0
        # Weighted sum: semantic slightly stronger by default
        w = 0.65 if item["source"] == "vector" else 0.35
        merged[kkey]["hybrid_score"] += w * float(item.get("score_norm", 0.0))

    reranked = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

    # Keep payload compact
    out = []
    for r in reranked:
        out.append(
            {
                "text": r["text"],
                "hybrid_score": float(r["hybrid_score"]),
                "source": r["source"],
                "citation": r.get("citation", {}),
            }
        )
    return out
