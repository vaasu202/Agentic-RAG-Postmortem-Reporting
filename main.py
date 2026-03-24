from __future__ import annotations

import argparse
import json
from dotenv import load_dotenv
from rich import print

from ingest.build_vectorstore import build_or_rebuild
from ingest.bm25_index import build_bm25_from_chroma
from agent.graph import build_agent_graph

load_dotenv()


def run_agent(incident: str, logs: str = "", answers_json: str | None = None):
    app = build_agent_graph()
    user_answers = json.loads(answers_json) if answers_json else {}

    state = {
        "user_incident": incident,
        "logs": logs,
        "retrieved": [],
        "log_signals": None,
        "clarifying_questions": [],
        "user_answers": user_answers,
        "step_count": 0,
        "done": False,
        "notes": [],
        "tool_trace": [],
        "log_citation": {"source": "user_provided_logs" if logs else "none"},
    }

    while not state.get("done"):
        state["step_count"] += 1
        state = app.invoke(state)

    if state.get("clarifying_questions"):
        print("\n[bold yellow]Missing info. Answer these to continue:[/bold yellow]")
        for i, q in enumerate(state["clarifying_questions"], 1):
            print(f"{i}. {q}")
        print("\nRerun with --answers '{\"env\":\"prod\", ...}'")
        return

    print("\n[bold green]Incident Report[/bold green]")
    print(json.dumps(state.get("report", {}), indent=2))
    print("\n[bold cyan]Tool Trace[/bold cyan]")
    print(json.dumps(state.get("tool_trace", []), indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--data", default="data")

    p_run = sub.add_parser("run")
    p_run.add_argument("--incident", required=True)
    p_run.add_argument("--logs", default="")
    p_run.add_argument("--answers", default=None)

    p_bm = sub.add_parser("build-bm25")

    args = parser.parse_args()

    if args.cmd == "ingest":
        build_or_rebuild(args.data)
        path = build_bm25_from_chroma()
        print(f"[green]BM25 index built:[/green] {path}")
    elif args.cmd == "build-bm25":
        path = build_bm25_from_chroma()
        print(f"[green]BM25 index built:[/green] {path}")
    elif args.cmd == "run":
        run_agent(args.incident, args.logs, args.answers)


if __name__ == "__main__":
    main()
