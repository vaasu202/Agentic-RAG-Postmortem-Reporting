from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from agent.graph import build_graph  # adjust if your builder name differs


def run_incident_copilot(
    incident: str,
    logs: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    user_answers: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (report, tool_trace, raw_state)
    """
    app = build_graph()

    state: Dict[str, Any] = {
        "user_incident": incident,
        "logs": logs or "",
        "user_answers": user_answers or {},
        "clarifying_questions": [],
        "retrieved": [],
        "log_signals": None,
        "kb_query": None,
        "next_action": "log_signals" if (logs and logs.strip()) else "kb_search",
        "done": False,
        "step_count": 0,
        "tool_trace": [],
        "log_citation": {"source": "user_provided_logs"} if (logs and logs.strip()) else None,
        "chat_history": chat_history or [],
    }

    # Single invoke: your LangGraph should execute the loop internally
    state = app.invoke(state)

    report = state.get("report") or {}
    trace = state.get("tool_trace") or []
    return report, trace, state
