import os
from typing import Dict, Any, List
import json
from langchain_openai import ChatOpenAI

print("LOADED clarifier.py FROM:", __file__)
print("CLARIFIER VERSION: v2-incident-first")

def _looks_like_consumer_account_issue(text: str) -> bool:
    t = (text or "").lower()
    keywords = [
        "valorant", "riot", "fortnite", "steam", "epic", "xbox", "psn", "playstation",
        "login", "log in", "sign in", "password", "account", "verify", "2fa", "locked out",
        "ban", "suspended", "email code"
    ]
    return any(k in t for k in keywords)


def ask_clarifying_questions(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask ONLY the minimum necessary questions based on missing information.
    Returns {"questions": [...], "reason": optional_str}
    """

    # ALWAYS define incident first, and never reassign it later.
    incident_text = (context.get("incident") or "").strip()

    # Consumer / out-of-scope gate
    if _looks_like_consumer_account_issue(incident_text):
        return {
            "questions": [
                "Is this affecting only your account/device, or are many users affected (an outage)?",
                "What exact error message/code do you see, and when (timestamp + timezone)?"
            ],
            "reason": "Likely consumer account/login issue; need scope + error details."
        }

    signals = context.get("log_signals", {}) or {}
    retrieved = context.get("retrieved", []) or []
    answers = context.get("user_answers", {}) or {}

    model = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.1"),
        temperature=0.2,
    )

    prompt = f"""
You are a senior site reliability engineer investigating a production incident.

Incident:
{incident_text}

Log signals:
{signals}

Retrieved references available:
{len(retrieved)}

Existing answers:
{answers}

Your job:
Ask ONLY the minimum number of clarifying questions required to determine root cause.

Rules:
- Ask max 3 questions
- Ask ONLY if information is missing
- Questions must be specific to THIS incident
- If sufficient info exists, return an empty list

Return JSON only:
{{
  "questions": ["question1", "question2"]
}}
"""

    resp = model.invoke([{"role": "user", "content": prompt}])
    text = (resp.content or "").strip()

    try:
        data = json.loads(text)
        questions = data.get("questions", [])
    except Exception:
        questions = []

    questions = (questions or [])[:3]
    return {"questions": questions}