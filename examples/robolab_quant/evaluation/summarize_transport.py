"""Summarize completed lane request latency without confusing it with GPU latency."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(root):
    result = {}
    for path in sorted(root.glob("*/requests.jsonl")):
        values = []
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["event"] == "transport":
                    values.append(row["transport_latency_s"])
        if not values:
            continue
        values.sort()
        result[path.parent.name] = dict(
            requests=len(values),
            median_s=statistics.median(values),
            p95_s=values[int(0.95 * (len(values) - 1))],
            p99_s=values[int(0.99 * (len(values) - 1))],
        )
    return dict(
        lanes=result,
        metric="Client transport includes server queue and inference; excludes injected delay.",
        weighting="Event-weighted, not episode-weighted; longer episodes contribute more requests.",
        quantiles="Sorted sample at floor(p*(n-1)), no interpolation.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.root), indent=2) + "\n")
