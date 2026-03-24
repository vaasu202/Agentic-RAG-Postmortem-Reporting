import json
import sys
from pathlib import Path

import streamlit as st

# --- Make project root importable ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runner import run_incident_copilot


# ----------------------------
# Session state initialization
# ----------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"  # chat | clarify
    if "pending_payload" not in st.session_state:
        st.session_state.pending_payload = None
    if "last_state" not in st.session_state:
        st.session_state.last_state = None


def reset_to_chat(reason: str | None = None):
    st.session_state.mode = "chat"
    st.session_state.pending_payload = None
    if reason:
        st.session_state.messages.append({"role": "assistant", "content": f"(reset) {reason}"})


def render_retrieved_sources(raw_state: dict):
    retrieved = raw_state.get("retrieved") or []
    if not retrieved:
        st.info("No retrieved sources found.")
        return

    top_n = min(6, len(retrieved))
    for i in range(top_n):
        hit = retrieved[i]
        citation = hit.get("citation") or {}
        text = hit.get("text") or hit.get("content") or ""
        score = hit.get("score")

        fn = citation.get("filename") or Path(citation.get("source_path", "")).name or "unknown"
        sec = citation.get("section", "")
        chunk = citation.get("chunk_id", "")
        label = f"{i+1}) {fn} | {sec} | {chunk}"
        if score is not None:
            label += f" | score={score:.4f}" if isinstance(score, (int, float)) else f" | score={score}"

        with st.expander(label, expanded=(i == 0)):
            if citation:
                st.code(json.dumps(citation, indent=2), language="json")
            snippet = (text[:1200] + "…") if len(text) > 1200 else text
            st.write(snippet if snippet.strip() else "(empty chunk text)")


def format_report(report: dict) -> str:
    parts = []
    parts.append(f"### Incident Summary\n{report.get('incident_summary','')}\n")

    rc = report.get("probable_root_causes", [])
    if rc:
        parts.append("### Probable Root Causes")
        for i, x in enumerate(rc, 1):
            parts.append(f"{i}. {x}")
        parts.append("")

    ev = report.get("supporting_evidence", [])
    if ev:
        parts.append("### Supporting Evidence")
        for i, item in enumerate(ev, 1):
            claim = item.get("claim", "")
            parts.append(f"{i}. {claim}")
            cits = item.get("citations", []) or []
            for c in cits:
                fn = c.get("filename") or Path(c.get("source_path", "")).name
                chunk = c.get("chunk_id", "")
                sec = c.get("section", "")
                parts.append(f"   - Source: `{fn}` | {sec} | {chunk}")
        parts.append("")

    steps = report.get("recommended_remediation_steps", [])
    if steps:
        parts.append("### Recommended Remediation Steps")
        for i, x in enumerate(steps, 1):
            parts.append(f"{i}. {x}")
        parts.append("")



    missing = report.get("missing_information", [])
    if missing:
        parts.append("### Missing Information")
        for i, x in enumerate(missing, 1):
            parts.append(f"{i}. {x}")
        parts.append("")

    return "\n".join(parts)


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Incident Copilot", layout="wide")
init_state()

st.title("Agentic RAG Incident Copilot")
st.caption("You have a problem, I have the answer. For example - I cannot login to my Cloudfare account. Help me")
st.caption("Knowledge with citations limited to service companies like Cloudfare, Google etc.")

# Sidebar
with st.sidebar:
    #st.header("UI")
    #show_trace = st.checkbox("Show tool trace", value=False)
    #show_retrieved = st.checkbox("Show retrieved sources", value=False)
    #show_raw_state = st.checkbox("Show raw state (debug)", value=False)

    #st.divider()
    #st.header("Inputs")
    #input_mode = st.radio("Logs input", ["Paste logs", "Upload log file", "No logs"], index=2)

    st.divider()
    st.caption("Debug")
    st.write("mode:", st.session_state.mode)
    st.write("pending_payload:", st.session_state.pending_payload is not None)
    if st.button("Reset conversation"):
        st.session_state.messages = []
        reset_to_chat("manual reset")
        st.rerun()

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Logs input
#logs_text = ""
#if input_mode == "Paste logs":
#    logs_text = st.text_area("Logs (optional)", height=160, placeholder="Paste logs here…")
#elif input_mode == "Upload log file":
#    up = st.file_uploader("Upload a log file", type=["log", "txt", "md"])
#    if up is not None:
#        logs_text = up.read().decode("utf-8", errors="ignore")
#        st.success(f"Loaded {up.name} ({len(logs_text):,} chars)")
#else:
 #   logs_text = ""

# ----------------------------
# Clarify mode
# ----------------------------
if st.session_state.mode == "clarify":
    pending = st.session_state.pending_payload

    if not pending:
        reset_to_chat("clarify mode had no payload")
        st.rerun()

    questions = pending.get("questions") or []
    if not questions:
        reset_to_chat("clarify mode had no questions")
        st.rerun()

    st.subheader("Clarifying Questions")
    st.write("Answer these and I’ll rerun the analysis.")

    with st.form("clarify_form"):
        answers = {}
        for i, q in enumerate(questions, 1):
            answers[q] = st.text_input(q, key=f"clarify_{i}")
        submitted = st.form_submit_button("Submit answers and rerun")

    if submitted:
        user_answers = {"raw": answers}

        with st.chat_message("assistant"):
            with st.spinner("Re-analyzing…"):
                report, trace, raw_state = run_incident_copilot(
                    incident=pending["incident"],
                    logs=pending.get("logs", ""),
                    chat_history=pending.get("history") or st.session_state.messages[-8:],
                    user_answers=user_answers,
                )

            has_report = bool(report and report.get("incident_summary"))

            if not has_report:
                st.error("No report generated after clarifications. Staying in clarify mode.")
                st.subheader("Tool Trace")
                st.code(json.dumps(trace, indent=2), language="json")
                st.subheader("Raw State")
                st.code(json.dumps(raw_state, indent=2), language="json")

                # Keep clarify mode + keep payload so user can edit answers and resubmit
                st.session_state.last_state = raw_state
                st.session_state.pending_payload = pending
                st.stop()

            # Success path: show report and exit clarify mode
#            if show_retrieved:
 #               st.subheader("Retrieved Sources")
#                render_retrieved_sources(raw_state)

            st.markdown(format_report(report))

#            if show_trace:
#                st.subheader("Tool Trace")
 #               st.code(json.dumps(trace, indent=2), language="json")
#            if show_raw_state:
#                st.subheader("Raw State")
#                st.code(json.dumps(raw_state, indent=2), language="json")

            st.session_state.messages.append({"role": "assistant", "content": format_report(report)})
            st.session_state.last_state = raw_state

        reset_to_chat()
        st.rerun()


    st.stop()

# ----------------------------
# Chat mode
# ----------------------------
incident_text = st.chat_input("Describe the incident in plain English…")

if incident_text:
    st.session_state.messages.append({"role": "user", "content": incident_text})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            history = st.session_state.messages[-8:]
            report, trace, raw_state = run_incident_copilot(
                incident=incident_text,
 #               logs=logs_text,
                chat_history=history,
                user_answers={},
            )

        questions = raw_state.get("clarifying_questions") or []
        has_report = bool(report and report.get("incident_summary"))

        # Clarify path: ALWAYS rerun so the clarify UI is rendered
        if questions and not has_report:
            st.session_state.mode = "clarify"
            st.session_state.pending_payload = {
                "incident": incident_text,
#                "logs": logs_text,
                "questions": questions,
                "history": history,
            }
            st.session_state.last_state = raw_state
            st.rerun()

        # No-output fallback: don’t show blank
        if not has_report and not questions:
            st.error("Agent produced no report and no clarifying questions.")
#            if show_trace:
#                st.subheader("Tool Trace")
#                st.code(json.dumps(trace, indent=2), language="json")
            st.subheader("Raw State")
            st.code(json.dumps(raw_state, indent=2), language="json")
            st.stop()

        # Report path
 #       if show_retrieved:
#            st.subheader("Retrieved Sources")
#            render_retrieved_sources(raw_state)

        st.markdown(format_report(report))

 #       if show_trace:
 #           st.subheader("Tool Trace")
 #           st.code(json.dumps(trace, indent=2), language="json")
 #       if show_raw_state:
 #           st.subheader("Raw State")
 #           st.code(json.dumps(raw_state, indent=2), language="json")

    st.session_state.messages.append({"role": "assistant", "content": format_report(report)})
    st.session_state.last_state = raw_state
