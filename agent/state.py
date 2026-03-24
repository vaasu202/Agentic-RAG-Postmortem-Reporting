from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    user_incident: str
    logs: str

    retrieved: List[Dict[str, Any]]
    log_signals: Dict[str, Any]
    clarifying_questions: List[str]
    user_answers: Dict[str, str]

    report: Dict[str, Any]
    log_citation: Dict[str, Any]

    step_count: int
    done: bool
    next_action: str
    notes: List[str]

    tool_trace: List[Dict[str, Any]]  # NEW: tool calls + summaries
