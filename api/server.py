from __future__ import annotations

import json
import os
from typing import Any, Dict, Generator

from dotenv import load_dotenv
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from agent.graph import build_agent_graph
from api.models import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="Incident Copilot API", version="0.1.0")
agent_app = build_agent_graph()


def _run_agent(req: ChatRequest) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "user_incident": req.incident,
        "logs": req.logs,
        "retrieved": [],
        "log_signals": None,
        "clarifying_questions": [],
        "user_answers": req.answers,
        "step_count": 0,
        "done": False,
        "notes": [],
        "tool_trace": [],
    }

    while not state.get("done"):
        state["step_count"] += 1
        state = agent_app.invoke(state)

    return state


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    state = _run_agent(req)
    return ChatResponse(
        tool_trace=state.get("tool_trace", []),
        clarifying_questions=state.get("clarifying_questions", []),
        report=state.get("report"),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def gen() -> Generator[dict, None, None]:
        # Run agent once, but stream events as if “live”
        state = _run_agent(req)

        # Stream tool trace events first
        for ev in state.get("tool_trace", []):
            yield {"event": "tool", "data": json.dumps(ev)}

        if state.get("clarifying_questions"):
            yield {"event": "clarify", "data": json.dumps({"questions": state["clarifying_questions"]})}
            return

        # Stream report in chunks (simple token chunking)
        report = state.get("report", {})
        report_str = json.dumps(report, indent=2)
        for i in range(0, len(report_str), 400):
            yield {"event": "report_chunk", "data": report_str[i:i+400]}

        yield {"event": "done", "data": json.dumps({"ok": True})}

    return EventSourceResponse(gen())
