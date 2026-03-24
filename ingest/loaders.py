from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document


SUPPORTED_EXTS = {".md", ".txt", ".log", ".json"}


@dataclass(frozen=True)
class LoadedDoc:
    doc: Document


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> str:
    # Store JSON as pretty text for embeddings, keep raw for metadata if desired
    data = json.loads(_read_text(path))
    return json.dumps(data, indent=2, sort_keys=True)


def load_documents(data_dir: str) -> list[Document]:
    """
    Loads postmortems/runbooks/logs from a /data folder.
    Expected layout:
      data/postmortems, data/runbooks, data/logs
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base.resolve()}")

    docs: list[Document] = []
    for sub in ["postmortems", "runbooks", "logs"]:
        folder = base / sub
        if not folder.exists():
            continue

        for p in folder.rglob("*"):
            if p.is_dir():
                continue
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue

            if p.suffix.lower() == ".json":
                text = _read_json(p)
            else:
                text = _read_text(p)

            doc_type = sub[:-1] if sub.endswith("s") else sub
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_path": str(p.as_posix()),
                        "filename": p.name,
                        "doc_type": doc_type,  # postmortem|runbook|log
                    },
                )
            )
    return docs
