from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

TS_PATTERNS = [
    # 2026-02-18T12:34:56Z or with offset
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"),
    # 2026-02-18 12:34:56
    re.compile(r"\b\d{4}-\d{2}-\d{2}[ ]\d{2}:\d{2}:\d{2}\b"),
]
ERROR_CODE_RE = re.compile(r"\b(?:ERR|ERROR|E)\d{3,6}\b|\b5\d{2}\b")
SERVICE_RE = re.compile(r"\b(service|svc|component)=([a-zA-Z0-9_-]+)\b", re.IGNORECASE)
EXC_RE = re.compile(r"\b(Exception|Traceback|panic|segfault|OOM|OutOfMemory)\b", re.IGNORECASE)
LAT_RE = re.compile(r"\b(latency|p99|p95|duration|elapsed)=?([0-9]+(?:\.[0-9]+)?)(ms|s)\b", re.IGNORECASE)


def extract_log_signals(log_text: str) -> dict[str, Any]:
    """
    Tool 2 (Required): Extract error codes, services, timestamps, repeated failures.
    Pure deterministic extraction (fast + reliable).
    """
    timestamps = []
    for pat in TS_PATTERNS:
        timestamps.extend(pat.findall(log_text))

    error_codes = ERROR_CODE_RE.findall(log_text)
    services = [m.group(2) for m in SERVICE_RE.finditer(log_text)]
    exceptions = EXC_RE.findall(log_text)
    latencies = [{"metric": m.group(1), "value": float(m.group(2)), "unit": m.group(3)} for m in LAT_RE.finditer(log_text)]

    # repeated lines heuristic
    lines = [ln.strip() for ln in log_text.splitlines() if ln.strip()]
    top_repeats = Counter(lines).most_common(5)
    repeated_failures = [{"line": ln, "count": c} for ln, c in top_repeats if c >= 3]

    time_range = None
    if timestamps:
        # Keep it simple: just show min/max lexicographically (works for ISO-ish)
        time_range = {"start": min(timestamps), "end": max(timestamps), "count": len(timestamps)}

    return {
        "time_range": time_range,
        "services": sorted(set(services))[:20],
        "error_codes": sorted(set(error_codes))[:50],
        "exceptions_keywords": sorted(set(exceptions)),
        "latency_observations": latencies[:30],
        "repeated_failures": repeated_failures,
        "raw_stats": {
            "lines": len(lines),
            "unique_lines": len(set(lines)),
        },
    }
