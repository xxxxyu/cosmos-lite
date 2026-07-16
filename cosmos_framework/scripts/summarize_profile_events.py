#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize Cosmos3 policy-server profile_events.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _summarize_field(items: list[dict[str, Any]], field: str, skip_first: bool = False) -> dict[str, float] | None:
    vals = [float(item[field]) for item in items if field in item]
    if skip_first and len(vals) > 1:
        vals = vals[1:]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": float(_pct(vals, 50) or 0.0),
        "p95": float(_pct(vals, 95) or 0.0),
        "max": max(vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    events_path = run_dir / "profile_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]

    by_event: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_event.setdefault(event["event"], []).append(event)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "event_counts": {key: len(value) for key, value in by_event.items()},
    }
    fields_by_event = {
        "model_load": ["elapsed_ms"],
        "policy_init": ["elapsed_ms"],
        "transform_init": ["elapsed_ms"],
        "infer_one": ["elapsed_ms", "build_sample_ms", "build_batch_ms", "generate_ms", "postprocess_ms"],
        "get_action": ["elapsed_ms"],
        "request": ["elapsed_ms", "recv_ms", "decode_ms", "dispatch_ms", "encode_ms"],
    }
    for event_name, fields in fields_by_event.items():
        items = by_event.get(event_name, [])
        summary[event_name] = {}
        for field in fields:
            summary[event_name][field] = _summarize_field(items, field, skip_first=False)
            if event_name in {"infer_one", "get_action", "request"}:
                summary[event_name][f"{field}_steady_excluding_first"] = _summarize_field(
                    items,
                    field,
                    skip_first=True,
                )

    mem_events = [event for event in events if "cuda_max_allocated_gb" in event]
    model_load = by_event["model_load"][0]
    summary["memory"] = {
        "max_allocated_gb": max(event.get("cuda_max_allocated_gb", 0.0) for event in mem_events),
        "max_reserved_gb": max(event.get("cuda_max_reserved_gb", 0.0) for event in mem_events),
        "post_load_allocated_gb": model_load["cuda_allocated_gb"],
        "post_load_reserved_gb": model_load["cuda_reserved_gb"],
    }
    if by_event.get("request"):
        first_request = by_event["request"][0]
        summary["first_request"] = {
            "dispatch_ms": first_request.get("dispatch_ms"),
            "elapsed_ms": first_request.get("elapsed_ms"),
        }
    if by_event.get("infer_one"):
        first_infer = by_event["infer_one"][0]
        summary.setdefault("first_request", {})["generate_ms"] = first_infer.get("generate_ms")

    (run_dir / "profile_bf16.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    steady_req = summary["request"]["elapsed_ms_steady_excluding_first"]
    steady_dispatch = summary["request"]["dispatch_ms_steady_excluding_first"]
    steady_gen = summary["infer_one"]["generate_ms_steady_excluding_first"]
    steady_build = summary["infer_one"]["build_sample_ms_steady_excluding_first"]
    steady_encode = summary["request"]["encode_ms_steady_excluding_first"]
    first_req = summary.get("first_request", {})

    md = [
        "# M2 BF16 Profile",
        "",
        f"Run dir: {run_dir}",
        "",
        "## Key Results",
        f"- Model load: {summary['model_load']['elapsed_ms']['mean']:.2f} ms",
        f"- Policy init total: {summary['policy_init']['elapsed_ms']['mean']:.2f} ms",
        (
            "- Post-load CUDA allocated/reserved: "
            f"{summary['memory']['post_load_allocated_gb']:.2f} / "
            f"{summary['memory']['post_load_reserved_gb']:.2f} GB"
        ),
        (
            "- Peak CUDA allocated/reserved during replay: "
            f"{summary['memory']['max_allocated_gb']:.2f} / "
            f"{summary['memory']['max_reserved_gb']:.2f} GB"
        ),
        f"- First profiled request dispatch: {first_req.get('dispatch_ms', 0.0):.2f} ms",
        f"- First profiled request generate: {first_req.get('generate_ms', 0.0):.2f} ms",
        f"- Steady request elapsed p50/p95: {steady_req['p50']:.2f} / {steady_req['p95']:.2f} ms",
        f"- Steady server dispatch p50/p95: {steady_dispatch['p50']:.2f} / {steady_dispatch['p95']:.2f} ms",
        f"- Steady generate p50/p95: {steady_gen['p50']:.2f} / {steady_gen['p95']:.2f} ms",
        f"- Steady input build p50/p95: {steady_build['p50']:.2f} / {steady_build['p95']:.2f} ms",
        f"- Steady response encode p50/p95: {steady_encode['p50']:.2f} / {steady_encode['p95']:.2f} ms",
        "",
        "## Artifacts",
        "- profile_events.jsonl",
        "- profile_bf16.json",
        "- replay_bf16_profile/metrics.json",
    ]
    (run_dir / "profile_bf16.md").write_text("\n".join(md) + "\n")

    print(
        json.dumps(
            {
                "memory": summary["memory"],
                "request_steady": steady_req,
                "dispatch_steady": steady_dispatch,
                "generate_steady": steady_gen,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
