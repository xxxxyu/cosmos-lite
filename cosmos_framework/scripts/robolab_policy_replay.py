# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay captured RoboLab OpenPI requests and summarize latency/action parity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--reference-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    capture_root = Path(args.capture_dir).expanduser().resolve()
    reference_root = Path(args.reference_dir).expanduser().resolve() if args.reference_dir else capture_root
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    request_files = sorted(capture_root.glob("sample_*.request.msgpack"))[args.skip :]
    if args.limit > 0:
        request_files = request_files[: args.limit]
    if not request_files:
        raise FileNotFoundError(f"No captured RoboLab requests found in {capture_root}")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")

    client = WebsocketClientPolicy(args.host, args.port)
    packer = msgpack_numpy.Packer()
    request_ms: list[float] = []
    server_ms: list[float] = []
    l1_values: list[float] = []
    linf_values: list[float] = []
    rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        for request_file in request_files:
            request = msgpack_numpy.unpackb(request_file.read_bytes())
            start = time.perf_counter()
            response = client.infer(request)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            request_ms.append(elapsed_ms)
            timing = response.get("server_timing", {})
            if isinstance(timing, dict) and isinstance(timing.get("infer_ms"), (int, float)):
                server_ms.append(float(timing["infer_ms"]))
            action = np.asarray(response["action"], dtype=np.float32)
            if not np.isfinite(action).all():
                raise ValueError(f"Non-finite action returned for {request_file.name}")
            reference_file = reference_root / request_file.name.replace(".request.", ".response.")
            l1 = None
            linf = None
            if reference_file.is_file():
                reference = msgpack_numpy.unpackb(reference_file.read_bytes())
                reference_action = np.asarray(reference["action"], dtype=np.float32)
                diff = np.abs(action - reference_action)
                l1 = float(diff.mean())
                linf = float(diff.max())
                l1_values.append(l1)
                linf_values.append(linf)
            response_name = request_file.name.replace(".request.", ".response.")
            if args.repeat > 1:
                response_name = f"repeat_{repeat_index:03d}.{response_name}"
            (output_root / response_name).write_bytes(packer.pack(response))
            rows.append(
                {
                    "sample": request_file.stem.split(".")[0],
                    "repeat": repeat_index,
                    "request_ms": elapsed_ms,
                    "server_ms": (
                        float(timing["infer_ms"])
                        if isinstance(timing, dict) and isinstance(timing.get("infer_ms"), (int, float))
                        else None
                    ),
                    "l1_mean": l1,
                    "linf": linf,
                }
            )

    metrics = {
        "capture_dir": str(capture_root),
        "reference_dir": str(reference_root) if reference_root != capture_root else None,
        "samples": len(rows),
        "repeat": args.repeat,
        "request_ms": {"p50": _percentile(request_ms, 50), "p95": _percentile(request_ms, 95)},
        "server_ms": {"p50": _percentile(server_ms, 50), "p95": _percentile(server_ms, 95)},
        "action_error": {
            "l1_mean": float(np.mean(l1_values)) if l1_values else None,
            "l1_p95": _percentile(l1_values, 95),
            "linf_p95": _percentile(linf_values, 95),
            "linf_max": max(linf_values) if linf_values else None,
        },
        "rows": rows,
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
