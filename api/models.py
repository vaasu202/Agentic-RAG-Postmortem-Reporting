from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class ChatRequest(BaseModel):
    incident: str = Field(..., description="Incident description")
    logs: str = Field("", description="Optional logs")
    answers: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    tool_trace: list[dict[str, Any]]
    clarifying_questions: list[str] = Field(default_factory=list)
    report: Optional[dict[str, Any]] = None
