from __future__ import annotations

import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

load_dotenv()


class EvidenceItem(BaseModel):
    claim: str
    citations: List[dict] = Field(default_factory=list)


class IncidentReport(BaseModel):
    incident_summary: str
    probable_root_causes: List[str]
    supporting_evidence: List[EvidenceItem]
    recommended_remediation_steps: List[str]
    confidence_level: str  # Low/Medium/High
    missing_information: List[str]


def generate_incident_report(context: dict[str, Any]) -> dict[str, Any]:
    """
    Tool 4: Produce a structured incident report.
    Strong guardrails:
      - Facts come only from user_incident + user_answers + log_signals.
      - Retrieved docs are used as *patterns* / supporting references, never as facts about "what happened" unless user said it.
      - Every supporting evidence item must have citations copied from retrieved[*].citation (or log_citation if provided).
    """
    retrieved = context.get("retrieved") or []
    MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.82"))
    def _top_score(items) -> float:
        scores = [r.get("score") for r in items if isinstance(r.get("score"), (int, float))]
        return max(scores) if scores else 0.0

    top_score = _top_score(retrieved)
    def _count_above(items, thr: float, k: int = 5) -> int:
        n = 0
        for r in (items or [])[:k]:
            s = r.get("score")
            if isinstance(s, (int, float)) and s >= thr:
                n += 1
        return n

    above = _count_above(retrieved, MIN_RELEVANCE, k=5)

    # Tune this threshold after a couple tests; start here.

    incident = (context.get("incident") or "").strip()

    if retrieved and (top_score < MIN_RELEVANCE or above < 2):
        return {
            "incident_summary": incident or "Insufficient input.",
            "probable_root_causes": [],
            "supporting_evidence": [],
            "recommended_remediation_steps": [],
            "confidence_level": "Low",
            "missing_information": [
                f"Query does not match the incident knowledge base (top retrieval score {top_score:.2f} < {MIN_RELEVANCE:.2f}).",
                "Provide service name, timeframe, scope (one user vs many), and any error codes/log snippets.",
            ],
        }

    # Minimal required inputs
    incident = (context.get("incident") or "").strip()
    log_signals = context.get("log_signals") or None
    user_answers = (context.get("user_answers") or {}).get("raw", {}) or {}
    log_citation = context.get("log_citation")  # may be None

    if not incident:
        return {
            "incident_summary": "Insufficient input: incident description is empty.",
            "probable_root_causes": [],
            "supporting_evidence": [],
            "recommended_remediation_steps": [],
            "confidence_level": "Low",
            "missing_information": ["Provide an incident description."],
        }

    if not retrieved:
        # We can still produce a report, but we must be honest and cite nothing.
        # Your schema requires citations list, but EvidenceItem allows empty list.
        return {
            "incident_summary": incident,
            "probable_root_causes": ["Insufficient evidence: no relevant historical incidents/runbooks were retrieved."],
            "supporting_evidence": [],
            "recommended_remediation_steps": [
                "Run a knowledge base search over postmortems/runbooks and re-run the analysis.",
                "Attach logs or provide error codes, affected services, and timeframe.",
            ],
            "confidence_level": "Low",
            "missing_information": ["Retrieved documents with citations (RAG results are empty)."],
        }

    # Keep prompt bounded: pass only top chunks + their citations
    top_k = min(8, len(retrieved))
    kb_chunks = []
    for r in retrieved[:top_k]:
        kb_chunks.append({
            "text": (r.get("text") or r.get("content") or "")[:1600],  # prevent huge prompt
            "citation": r.get("citation") or {},
            "score": r.get("score"),
        })

    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1"), temperature=0.1)

    system = """
You are a senior SRE + incident commander assistant.

You must produce an IncidentReport JSON object.

Non-negotiable rules:
- Do NOT invent incident-specific facts.
- "Incident Summary" must describe ONLY what the user reported + any explicit log_signals + explicit user_answers.
  Do NOT claim "traffic surge" or "deploy happened" unless the incident text/log_signals/answers explicitly say so.
- Retrieved KB chunks are historical references. Use them to suggest plausible root causes and mitigations,
  but never claim they happened in THIS incident unless the user said so.
- Every supporting_evidence item must:
  - be a single-sentence claim
  - include citations copied EXACTLY from the provided citations list.
- If you cannot cite a claim, it MUST go into missing_information instead of supporting_evidence.
- Confidence must reflect evidence quality:
  - High: logs + strong matching KB chunks + specific symptoms
  - Medium: some specifics + relevant KB but missing key telemetry
  - Low: mostly generic incident statement and/or weak match

CITATION RULES:
- For KB-derived evidence: cite using the provided KB citations.
"""

    user = {
        "incident": incident,
        "user_answers": user_answers,
        "log_signals": log_signals,
        "log_citation": log_citation,
        "kb_chunks": kb_chunks,
        "required_output": {
            "incident_summary": "string",
            "probable_root_causes": ["string"],
            "supporting_evidence": [{"claim": "string", "citations": ["citation objects"]}],
            "recommended_remediation_steps": ["string"],
            "confidence_level": "Low|Medium|High",
            "missing_information": ["string"],
        }
    }

    structured = model.with_structured_output(IncidentReport, method="function_calling")
    resp = structured.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"INPUT:\n{user}"}
    ])
    return resp.model_dump()
