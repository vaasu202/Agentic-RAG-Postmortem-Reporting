from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

from ingest.loaders import load_documents
from ingest.chunking import chunk_documents
from ingest.embeddings import build_embeddings


load_dotenv()


def build_or_rebuild(data_dir: str = "data") -> None:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    collection = os.getenv("CHROMA_COLLECTION", "incident_copilot")

    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    embeddings = build_embeddings()

    # Rebuild by deleting existing collection directory if you want a clean slate
    # For MVP, we just upsert into a persisted store.
    vs = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vs.add_documents(chunks)
    vs.persist()
    print(f"Ingested {len(docs)} docs -> {len(chunks)} chunks into Chroma ({persist_dir}/{collection}).")


if __name__ == "__main__":
    build_or_rebuild()
