#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize Nsight Systems CUDA kernel/API CSV for quantized RoboCasa365 replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_first_kernel_report(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(errors="replace").splitlines()
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("Time (%),Total Time (ns),Instances"):
            start = idx + 1
            continue
        if start is not None and line.startswith("Processing "):
            end = idx
            break
    if start is None:
        raise ValueError(f"Could not find cuda_gpu_kern_sum report in {path}")

    rows: list[dict[str, Any]] = []
    for row in csv.reader(lines[start:end]):
        if len(row) < 9:
            continue
        try:
            rows.append(
                {
                    "time_pct": float(row[0]),
                    "total_time_ns": int(row[1]),
                    "instances": int(row[2]),
                    "avg_ns": float(row[3]),
                    "median_ns": float(row[4]),
                    "name": row[8],
                }
            )
        except ValueError:
            continue
    return rows


def _category(name: str) -> str:
    if "marlin::Marlin" in name:
        return "marlin"
    if "natten::cuda::fmha" in name or "fmha_cutlass" in name:
        return "attention_fmha"
    if "cudnn" in name or "nchwToNhwc" in name or "nhwcToNchw" in name:
        return "cudnn_conv_or_layout"
    if any(
        token in name
        for token in (
            "elementwise",
            "reduce_kernel",
            "direct_copy",
            "CatArray",
            "index",
            "FillFunctor",
            "copy_kernel",
            "silu_kernel",
            "rsqrt_kernel",
        )
    ):
        return "pytorch_elementwise_reduce_copy_cat"
    if "gemm" in name or "gemv" in name or "cublas" in name:
        return "cublas_gemm"
    return "other"


def summarize(path: Path, top_n: int) -> dict[str, Any]:
    rows = _read_first_kernel_report(path)
    total_ns = sum(row["total_time_ns"] for row in rows)
    total_instances = sum(row["instances"] for row in rows)
    categories: dict[str, dict[str, float | int]] = {}
    for row in rows:
        category = _category(str(row["name"]))
        item = categories.setdefault(category, {"time_ns": 0, "instances": 0, "time_pct": 0.0})
        item["time_ns"] = int(item["time_ns"]) + int(row["total_time_ns"])
        item["instances"] = int(item["instances"]) + int(row["instances"])

    for item in categories.values():
        item["time_ms"] = float(item["time_ns"]) / 1e6
        item["time_pct"] = 100.0 * float(item["time_ns"]) / float(total_ns or 1)

    return {
        "source": str(path),
        "kernel_total_ms": total_ns / 1e6,
        "kernel_rows": len(rows),
        "kernel_instances": total_instances,
        "categories": dict(sorted(categories.items(), key=lambda kv: -float(kv[1]["time_ns"]))),
        "top_kernels": rows[:top_n],
    }


def _write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Nsight Systems Kernel Summary",
        "",
        f"Source: `{summary['source']}`",
        "",
        f"- Total CUDA kernel time: `{summary['kernel_total_ms']:.1f} ms`",
        f"- Kernel launches in report: `{summary['kernel_instances']}`",
        "",
        "## Categories",
        "",
        "| Category | Time ms | Share | Instances |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["categories"].items():
        lines.append(
            f"| `{name}` | {float(item['time_ms']):.1f} | "
            f"{float(item['time_pct']):.1f}% | {int(item['instances'])} |"
        )
    lines.extend(["", "## Top Kernels", "", "| Share | Time ms | Instances | Name |", "|---:|---:|---:|---|"])
    for row in summary["top_kernels"]:
        kernel = str(row["name"]).replace("|", "\\|")
        lines.append(
            f"| {float(row['time_pct']):.1f}% | {int(row['total_time_ns']) / 1e6:.1f} | "
            f"{int(row['instances'])} | `{kernel}` |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = summarize(Path(args.stats_csv), args.top_n)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")
    if args.output_md:
        _write_markdown(summary, Path(args.output_md))
    print(text)


if __name__ == "__main__":
    main()
