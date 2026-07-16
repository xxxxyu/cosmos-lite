#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize a direct replay benchmark sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _get(item: dict[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = item
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _fmt(value: Any, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    if "_bench_" in name:
        name = name.split("_bench_", 1)[1]
    if "_replay" in name:
        name = name.split("_replay", 1)[0]
    parts = name.split("_")
    for idx in range(len(parts)):
        candidate = "_".join(parts[idx:])
        if candidate in {"bf16", "full_w8", "full_w4", "attention_w8", "gen_branch_w8"}:
            name = candidate
            break

    profile = _load_json(run_dir / "profile_bf16.json")
    replay_metrics = _load_json(run_dir / "replay32" / "metrics.json")
    aggregate = replay_metrics["aggregate"]
    memory = profile["memory"]
    request = profile["request"]["elapsed_ms_steady_excluding_first"]
    dispatch = profile["request"]["dispatch_ms_steady_excluding_first"]
    generate = profile["infer_one"]["generate_ms_steady_excluding_first"]
    model_load = profile["model_load"]["elapsed_ms"]["mean"]
    policy_init = profile["policy_init"]["elapsed_ms"]["mean"]

    return {
        "candidate": name,
        "run_dir": str(run_dir),
        "n_replay": aggregate["n"],
        "model_load_ms": model_load,
        "policy_init_ms": policy_init,
        "post_load_alloc_gb": memory["post_load_allocated_gb"],
        "post_load_reserved_gb": memory["post_load_reserved_gb"],
        "peak_alloc_gb": memory["max_allocated_gb"],
        "peak_reserved_gb": memory["max_reserved_gb"],
        "client_elapsed_p50_ms": aggregate["elapsed_ms_p50"],
        "client_elapsed_p95_ms": aggregate["elapsed_ms_p95"],
        "server_request_p50_ms": request["p50"],
        "server_request_p95_ms": request["p95"],
        "server_dispatch_p50_ms": dispatch["p50"],
        "server_dispatch_p95_ms": dispatch["p95"],
        "generate_p50_ms": generate["p50"],
        "generate_p95_ms": generate["p95"],
        "action_l1_mean": aggregate["action_l1_mean_mean"],
        "action_l1_p95": aggregate["action_l1_mean_p95"],
        "action_linf_p95": aggregate["action_linf_p95"],
        "action_linf_max": aggregate["action_linf_max"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(Path(path)) for path in args.run_dirs]

    order = {"bf16": 0, "full_w8": 1, "full_w4": 2, "attention_w8": 3, "gen_branch_w8": 4}
    rows.sort(key=lambda row: order.get(str(row["candidate"]), 99))

    columns = [
        "candidate",
        "n_replay",
        "post_load_alloc_gb",
        "peak_alloc_gb",
        "client_elapsed_p50_ms",
        "client_elapsed_p95_ms",
        "generate_p50_ms",
        "generate_p95_ms",
        "action_l1_mean",
        "action_l1_p95",
        "action_linf_p95",
        "action_linf_max",
        "model_load_ms",
        "policy_init_ms",
        "run_dir",
    ]
    csv_path = output_dir / "direct_replay_benchmark_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    md_path = output_dir / "direct_replay_benchmark_summary.md"
    md_columns = [
        "candidate",
        "n_replay",
        "post_load_alloc_gb",
        "peak_alloc_gb",
        "client_elapsed_p50_ms",
        "client_elapsed_p95_ms",
        "generate_p50_ms",
        "generate_p95_ms",
        "action_l1_mean",
        "action_linf_p95",
    ]
    lines = [
        "# Direct Replay Benchmark Summary",
        "",
        "| " + " | ".join(md_columns) + " |",
        "| " + " | ".join(["---"] * len(md_columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col)) for col in md_columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- Memory is CUDA allocated memory reported by the profiled policy server.",
            "- Latency excludes the first profiled request for steady-state server metrics.",
            "- Action parity is measured against the captured BF16 response files in the replay dataset.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))

    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
