from __future__ import annotations

import re
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _detect_markdown_sections(text: str) -> list[tuple[str, int, int]]:
    """
    Returns (section_title, start, end) spans for markdown.
    Simple but effective: split at headers, keep ranges.
    """
    matches = list(_MD_HEADER_RE.finditer(text))
    if not matches:
        return [("Document", 0, len(text))]

    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(2).strip()
        spans.append((title, start, end))
    return spans


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Intelligent chunking:
      - postmortems/runbooks: markdown-section aware, then recursive splitter
      - logs: line-window splitter via recursive splitter with smaller sizes
    """
    out: list[Document] = []

    for d in docs:
        doc_type = d.metadata.get("doc_type", "unknown")
        text = d.page_content

        if doc_type in {"postmortem", "runbook"}:
            spans = _detect_markdown_sections(text)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            for section_title, s, e in spans:
                section_text = text[s:e].strip()
                if not section_text:
                    continue
                chunks = splitter.split_text(section_text)
                for idx, c in enumerate(chunks):
                    out.append(
                        Document(
                            page_content=c,
                            metadata={
                                **d.metadata,
                                "section": section_title,
                                "chunk_local_id": idx,
                            },
                        )
                    )

        elif doc_type == "log":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n", " ", ""],
            )
            chunks = splitter.split_text(text)
            for idx, c in enumerate(chunks):
                out.append(
                    Document(
                        page_content=c,
                        metadata={
                            **d.metadata,
                            "chunk_local_id": idx,
                        },
                    )
                )
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
            chunks = splitter.split_text(text)
            for idx, c in enumerate(chunks):
                out.append(
                    Document(
                        page_content=c,
                        metadata={
                            **d.metadata,
                            "chunk_local_id": idx,
                        },
                    )
                )

    # Add globally unique chunk_id
    for i, d in enumerate(out):
        d.metadata["chunk_id"] = f"chunk_{i:06d}"
    return out
