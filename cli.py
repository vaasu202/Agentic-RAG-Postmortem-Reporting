from __future__ import annotations

import argparse
import json
import requests

from rich import print


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incident", required=True)
    p.add_argument("--logs", default="")
    p.add_argument("--answers", default="{}")
    p.add_argument("--url", default="http://localhost:8000/chat/stream")
    args = p.parse_args()

    payload = {
        "incident": args.incident,
        "logs": args.logs,
        "answers": json.loads(args.answers or "{}"),
    }

    with requests.post(args.url, json=payload, stream=True) as r:
        r.raise_for_status()
        print("[bold]Streaming:[/bold]\n")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE format: "event: X" / "data: Y"
            if line.startswith("event:"):
                event = line.split("event:", 1)[1].strip()
                continue
            if line.startswith("data:"):
                data = line.split("data:", 1)[1].strip()
                if event == "tool":
                    obj = json.loads(data)
                    print(f"[cyan]TOOL[/cyan] {obj.get('tool')}: {obj}")
                elif event == "clarify":
                    obj = json.loads(data)
                    print("\n[bold yellow]Clarifying questions:[/bold yellow]")
                    for q in obj.get("questions", []):
                        print("-", q)
                elif event == "report_chunk":
                    print(data, end="")
                elif event == "done":
                    print("\n\n[green]DONE[/green]")
                else:
                    print(data)


if __name__ == "__main__":
    main()
