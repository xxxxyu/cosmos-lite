#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Replay captured RLDX policy requests and compare actions.

This script expects request/response files created by the RoboCasa365 capture
hook:

  sample_00000.request.msgpack
  sample_00000.response.msgpack

It sends each request to a running policy server and compares the returned
action dict against the captured BF16 reference response.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


def _as_response_tuple(response: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not isinstance(response, (list, tuple)) or len(response) != 2:
        raise TypeError(f"Expected response pair, got {type(response)}")
    actions, info = response
    if not isinstance(actions, dict):
        raise TypeError(f"Expected action dict, got {type(actions)}")
    if not isinstance(info, dict):
        info = {}
    return actions, info


def _flatten_actions(actions: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in sorted(actions):
        value = np.asarray(actions[key], dtype=np.float32)
        parts.append(value.reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError(f"Shape mismatch candidate={candidate.shape} reference={reference.shape}")
    diff = candidate - reference
    return {
        "action_l1_mean": float(np.mean(np.abs(diff))),
        "action_l2_mean": float(np.sqrt(np.mean(diff * diff))),
        "action_linf": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "action_ref_abs_mean": float(np.mean(np.abs(reference))) if reference.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=300000)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_dir = output_dir / "responses"
    response_dir.mkdir(exist_ok=True)

    request_files = sorted(capture_dir.glob("sample_*.request.msgpack"))
    if args.skip > 0:
        request_files = request_files[args.skip :]
    if args.limit > 0:
        request_files = request_files[: args.limit]
    if not request_files:
        raise FileNotFoundError(f"No request files found in {capture_dir}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.connect(f"tcp://{args.host}:{args.port}")

    rows: list[dict[str, Any]] = []
    try:
        for request_file in request_files:
            stem = request_file.name.replace(".request.msgpack", "")
            reference_file = capture_dir / f"{stem}.response.msgpack"
            if not reference_file.is_file():
                raise FileNotFoundError(reference_file)

            payload = request_file.read_bytes()
            start = time.perf_counter()
            socket.send(payload)
            response_bytes = socket.recv()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            candidate_response = MsgSerializer.from_bytes(response_bytes)
            reference_response = MsgSerializer.from_bytes(reference_file.read_bytes())
            if isinstance(candidate_response, dict) and "error" in candidate_response:
                raise RuntimeError(f"Server error for {stem}: {candidate_response['error']}")

            candidate_actions, candidate_info = _as_response_tuple(candidate_response)
            reference_actions, reference_info = _as_response_tuple(reference_response)
            candidate_flat = _flatten_actions(candidate_actions)
            reference_flat = _flatten_actions(reference_actions)
            item_metrics = _metrics(candidate_flat, reference_flat)
            item_metrics.update(
                {
                    "sample": stem,
                    "elapsed_ms": elapsed_ms,
                    "candidate_action_steps": candidate_info.get("action_steps"),
                    "reference_action_steps": reference_info.get("action_steps"),
                }
            )
            rows.append(item_metrics)
            (response_dir / f"{stem}.response.msgpack").write_bytes(response_bytes)
            with open(response_dir / f"{stem}.metrics.json", "w") as f:
                json.dump(item_metrics, f, indent=2, sort_keys=True)
    finally:
        socket.close()
        context.term()

    aggregate: dict[str, Any] = {
        "n": len(rows),
        "host": args.host,
        "port": args.port,
        "capture_dir": str(capture_dir),
        "output_dir": str(output_dir),
    }
    for key in ("action_l1_mean", "action_l2_mean", "action_linf", "elapsed_ms"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_p50"] = float(np.percentile(values, 50))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
        aggregate[f"{key}_max"] = float(np.max(values))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"aggregate": aggregate, "samples": rows}, f, indent=2, sort_keys=True)

    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
