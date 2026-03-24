from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma

from ingest.embeddings import build_embeddings

load_dotenv()


def _get_vs() -> Chroma:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    collection = os.getenv("CHROMA_COLLECTION", "incident_copilot")
    embeddings = build_embeddings()
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def search_incident_knowledge_base(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """
    Tool 1 (Required): Retrieve relevant past incidents and runbooks.
    Returns chunks with citation metadata.
    """
    top_k = k or int(os.getenv("TOP_K", "6"))
    vs = _get_vs()
    results = vs.similarity_search_with_score(query, k=top_k)

    out = []
    for doc, score in results:
        out.append(
            {
                "text": doc.page_content,
                "score": float(score),
                "citation": {
                    "source_path": doc.metadata.get("source_path"),
                    "filename": doc.metadata.get("filename"),
                    "doc_type": doc.metadata.get("doc_type"),
                    "section": doc.metadata.get("section"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                },
            }
        )
    return out
