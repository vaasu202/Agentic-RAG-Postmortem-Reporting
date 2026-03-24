from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


EmbBackend = Literal["openai", "st"]


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: EmbBackend
    openai_model: str = "text-embedding-3-small"
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def build_embeddings(cfg: Optional[EmbeddingConfig] = None):
    if cfg is None:
        cfg = EmbeddingConfig(
            backend=os.getenv("EMBEDDINGS_BACKEND", "openai"),
            openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            st_model=os.getenv("ST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )

    if cfg.backend == "openai":
        return OpenAIEmbeddings(model=cfg.openai_model)
    if cfg.backend == "st":
        return HuggingFaceEmbeddings(model_name=cfg.st_model)

    raise ValueError(f"Unknown embeddings backend: {cfg.backend}")
