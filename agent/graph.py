from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional
import json
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.prompts import ROUTER_SYSTEM, ROUTER_USER_TEMPLATE

from tools.kb_search import search_incident_knowledge_base
from tools.hybrid_search import hybrid_search
from tools.log_signals import extract_log_signals
from tools.clarifier import ask_clarifying_questions
from tools.report import generate_incident_report
print("LOADED graph.py FROM:", __file__)

load_dotenv()


class RouterDecision(BaseModel):
    next_action: Literal["log_signals", "hybrid_search", "kb_search", "clarify", "report"]
    query: Optional[str] = None


def _trace(state: AgentState, tool: str, payload: Dict[str, Any]) -> None:
    state.setdefault("tool_trace", []).append({"tool": tool, **payload})


def _router_node(state: AgentState) -> AgentState:
    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1"), temperature=0.3)
    state["step_count"] = int(state.get("step_count", 0)) + 1
    incident = state.get("user_incident", "")
    logs = state.get("logs", "") or ""
    has_logs = bool(logs.strip())

    prompt = ROUTER_USER_TEMPLATE.format(
        incident=incident,
        has_logs=has_logs,
        n_retrieved=len(state.get("retrieved", []) or []),
        has_signals=bool(state.get("log_signals")),
        n_questions=len(state.get("clarifying_questions", []) or []),
        n_answers=len(state.get("user_answers", {}) or {}),
    )

    router_instruction = ROUTER_SYSTEM + """
Return ONLY valid JSON (no markdown) with this schema:
{
  "next_action": "kb_search" | "hybrid_search" | "log_signals" | "clarify" | "report",
  "query": string | null
}
"""


    msg = model.invoke(
        [{"role": "system", "content": router_instruction},
         {"role": "user", "content": prompt}]
    )

    text = (msg.content or "").strip()
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        # Safe fallback
        decision = {"next_action": "log_signals" if has_logs and not state.get("log_signals") else "kb_search", "query": None}

    next_action = decision.get("next_action", "clarify")
    query = decision.get("query")

    if next_action not in {"kb_search", "hybrid_search", "log_signals", "clarify", "report"}:
        next_action = "clarify"


    state["next_action"] = next_action

    if isinstance(query, str) and query.strip():
        state["kb_query"] = query.strip()  # type: ignore
        state.setdefault("notes", []).append(f"router_query={state['kb_query']}")
    else:
        state.pop("kb_query", None)

    # IMPORTANT: ensure no RouterDecision object is ever stored
    state.pop("router_decision", None)  # in case you added it earlier

    return state

def _can_write_report(state: AgentState) -> bool:
    has_retrieved = bool(state.get("retrieved"))
    has_signals = bool(state.get("log_signals"))
    has_answers = bool(state.get("user_answers"))
    relevant = _retrieval_is_relevant(state, threshold=0.80)

    # Must have relevant retrieval, and either logs or answers
    return has_retrieved and relevant and (has_signals or has_answers)



def _log_signals_node(state: AgentState) -> AgentState:
    # Prevent repeated parsing
    if state.get("log_signals"):
        state.setdefault("tool_trace", []).append({
            "tool": "extract_log_signals",
            "skipped": True,
            "reason": "already_present"
        })
        return state

    logs = state.get("logs", "") or ""
    state["log_signals"] = extract_log_signals(logs)

    state.setdefault("tool_trace", []).append({
        "tool": "extract_log_signals",
        "summary": {
            "services": state["log_signals"].get("services", []),
            "error_codes": state["log_signals"].get("error_codes", []),
            "time_range": state["log_signals"].get("time_range"),
            "repeated_failures": len(state["log_signals"].get("repeated_failures", [])),
        }
    })

    return state



def _hybrid_search_node(state: AgentState) -> AgentState:
    q = state.get("search_query") or state.get("user_incident", "")
    results = hybrid_search(q)
    existing = state.get("retrieved") or []
    state["retrieved"] = existing + results
    _trace(state, "hybrid_search", {"query": q, "top": [
        {"hybrid_score": r["hybrid_score"], "citation": r.get("citation", {})} for r in results[:3]
    ]})
    return state


def _kb_search_node(state: AgentState) -> AgentState:
    q = state.get("kb_query")

    if not q:
        sig = state.get("log_signals") or {}
        codes = " ".join(sig.get("error_codes", [])[:5])
        svcs = " ".join(sig.get("services", [])[:3])
        q = f"{state.get('user_incident','')} {svcs} {codes}".strip()

    results = search_incident_knowledge_base(q)
    existing = state.get("retrieved") or []
    state["retrieved"] = existing + results

    state.setdefault("tool_trace", []).append({
        "tool": "search_incident_knowledge_base",
        "query": q,
        "results": len(results),
        "top_sources": [r["citation"]["filename"] for r in results[:3]],
    })

    return state




def _clarify_node(state: AgentState) -> AgentState:
    ctx = {
        "incident": state.get("user_incident", ""),
        "log_signals": state.get("log_signals"),
        "retrieved": (state.get("retrieved") or [])[:8],
        "user_answers": state.get("user_answers", {}),
    }

    q = ask_clarifying_questions(ctx)["questions"]
    state["clarifying_questions"] = q

    # IMPORTANT:
    # Do NOT mark done here. Clarify is a "pause", not a terminal state.
    # The UI will collect answers and re-invoke the graph with user_answers filled.
    _trace(state, "ask_clarifying_questions", {"questions": q})

    # Put the agent into a "waiting for user" state
    state["waiting_for_user"] = True
    return state



def _report_node(state: AgentState) -> AgentState:
    ctx = {
        "incident": state.get("user_incident", ""),
        "log_signals": state.get("log_signals"),
        "log_citation": state.get("log_citation"),
        "retrieved": state.get("retrieved", [])[:10],
        "user_answers": state.get("user_answers", {}),
        "instructions": {
            "citations_format": "Use citation objects exactly as provided.",
            "no_fabrication": True,
        },
    }

    state["report"] = generate_incident_report(ctx)
    state["done"] = True

    state.setdefault("tool_trace", []).append({
        "tool": "generate_incident_report",
        "confidence": state.get("report", {}).get("confidence_level"),
    })

    return state

def _top_retrieval_score(state: AgentState) -> float:
    retrieved = state.get("retrieved") or []
    if not retrieved:
        return 0.0
    scores = [r.get("score") for r in retrieved if isinstance(r.get("score"), (int, float))]
    return max(scores) if scores else 0.0

def _retrieval_is_relevant(state: AgentState, threshold: float = 0.80) -> bool:
    return _top_retrieval_score(state) >= threshold


def _should_continue(state: AgentState) -> str:
    step = int(state.get("step_count", 0))
    if state.get("done"):
        return END

    has_retrieved = bool(state.get("retrieved"))
    has_signals = bool(state.get("log_signals"))
    has_answers = bool(state.get("user_answers", {}).get("raw"))

    # EXIT RAMP: if we have enough context, write the report now.
    if has_retrieved and (has_signals or has_answers):
        state["next_action"] = "report"
        return "report"

    
    # If user provided answers but we still have no retrieval, retrieve now.
    if state.get("user_answers") and not state.get("retrieved"):
        state["next_action"] = "kb_search"
        incident = state.get("user_incident", "")
        ans = (state.get("user_answers", {}) or {}).get("raw", {}) or {}
        ans_text = " ".join([str(v) for v in ans.values() if v])
        state["kb_query"] = f"{incident} {ans_text}".strip()
        return "kb_search"

    
    # Always retrieve after signals (if not done yet)
    if state.get("log_signals") and not state.get("retrieved"):
        state["next_action"] = "kb_search"
        sig = state["log_signals"]
        codes = " ".join(sig.get("error_codes", [])[:5])
        svcs = " ".join(sig.get("services", [])[:3])
        state["kb_query"] = f"{state.get('user_incident','')} {svcs} {codes}".strip()
        return "kb_search"


    # hard gate: don't allow report without citations
    # If router wants report but we still have nothing to work with, clarify.
    if state.get("next_action") == "report" and not _can_write_report(state):
        if state.get("retrieved") and not _retrieval_is_relevant(state, threshold=0.80):
            state["next_action"] = "clarify"
            state["force_insufficient"] = True
            return "clarify"


    if step >= 8:
        state["next_action"] = "clarify"
        return "clarify"

    return state.get("next_action", "clarify")



def build_agent_graph():
    g = StateGraph(AgentState)

    g.add_node("router", _router_node)
    g.add_node("log_signals", _log_signals_node)
    g.add_node("hybrid_search", _hybrid_search_node)
    g.add_node("kb_search", _kb_search_node)
    g.add_node("clarify", _clarify_node)
    g.add_node("report", _report_node)

    g.set_entry_point("router")

    g.add_conditional_edges("router", _should_continue, {
        "log_signals": "log_signals",
        "hybrid_search": "hybrid_search",
        "kb_search": "kb_search",
        "clarify": "clarify",
        "report": "report",
        END: END,
    })

    for node in ["log_signals", "hybrid_search", "kb_search"]:
        g.add_edge(node, "router")

    g.add_edge("clarify", END)
    g.add_edge("report", END)

    return g.compile()

def build_graph():
    """Return the compiled LangGraph app for Streamlit/runner."""
    return build_agent_graph()

