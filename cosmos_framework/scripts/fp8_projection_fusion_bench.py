#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Benchmark shared-quant and packed-projection FP8 fusion candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class ProjectionGroup:
    name: str
    tokens: int
    in_features: int
    out_features: tuple[int, ...]


GROUPS = {
    "edge_qkv": ProjectionGroup("edge_qkv", 3093, 2048, (2048, 1024, 1024)),
    "nano_qkv": ProjectionGroup("nano_qkv", 3093, 4096, (4096, 1024, 1024)),
    "nano_gate_up": ProjectionGroup("nano_gate_up", 3093, 4096, (12288, 12288)),
}


def _time_ms(fn: Callable[[], tuple[torch.Tensor, ...]], warmup: int, iters: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), statistics.mean(samples)


def _run_group(group: ProjectionGroup, warmup: int, iters: int, seed: int) -> dict[str, object]:
    import vllm._C  # noqa: F401
    from vllm import _custom_ops as ops

    torch.manual_seed(seed)
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    capability_int = capability[0] * 10 + capability[1]
    if capability_int < 89 or not ops.cutlass_scaled_mm_supports_fp8(capability_int):
        raise RuntimeError(f"FP8 CUTLASS benchmark requires SM89+, got SM{capability_int}")

    x = torch.randn(group.tokens, group.in_features, dtype=torch.bfloat16, device=device)
    weights = [
        torch.randn(n, group.in_features, dtype=torch.bfloat16, device=device)
        .mul_(group.in_features**-0.5)
        .to(torch.float8_e4m3fn)
        .contiguous()
        for n in group.out_features
    ]
    scales = [torch.ones(1, n, dtype=torch.float32, device=device) for n in group.out_features]
    packed_weight = torch.cat(weights, dim=0).contiguous()
    packed_scale = torch.cat(scales, dim=1).contiguous()

    def quantize() -> tuple[torch.Tensor, torch.Tensor]:
        return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

    def gemm(q_x: torch.Tensor, scale_a: torch.Tensor, weight: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
        return ops.cutlass_scaled_mm(q_x, weight.t(), scale_a, scale_b, torch.bfloat16)

    def separate() -> tuple[torch.Tensor, ...]:
        outputs = []
        for weight, scale_b in zip(weights, scales, strict=True):
            q_x, scale_a = quantize()
            outputs.append(gemm(q_x, scale_a, weight, scale_b))
        return tuple(outputs)

    def shared_quant() -> tuple[torch.Tensor, ...]:
        q_x, scale_a = quantize()
        return tuple(gemm(q_x, scale_a, weight, scale_b) for weight, scale_b in zip(weights, scales, strict=True))

    def packed() -> tuple[torch.Tensor, ...]:
        q_x, scale_a = quantize()
        output = gemm(q_x, scale_a, packed_weight, packed_scale)
        return tuple(output.split(group.out_features, dim=-1))

    separate_out = separate()
    shared_out = shared_quant()
    packed_out = packed()
    torch.cuda.synchronize()

    parity: dict[str, dict[str, float]] = {}
    for name, outputs in (("shared_quant", shared_out), ("packed", packed_out)):
        diffs = [(actual.float() - expected.float()).abs() for actual, expected in zip(outputs, separate_out, strict=True)]
        parity[name] = {
            "l1_mean": float(torch.cat([diff.flatten() for diff in diffs]).mean().item()),
            "linf": max(float(diff.max().item()) for diff in diffs),
        }

    timings = {}
    for name, fn in (("separate", separate), ("shared_quant", shared_quant), ("packed", packed)):
        median_ms, mean_ms = _time_ms(fn, warmup=warmup, iters=iters)
        timings[name] = {"median_ms": median_ms, "mean_ms": mean_ms}
    baseline = float(timings["separate"]["median_ms"])
    for values in timings.values():
        values["speedup_vs_separate"] = baseline / float(values["median_ms"])

    return {
        "group": asdict(group),
        "timings": timings,
        "parity_vs_separate": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--tokens", type=int, default=0, help="Override the captured token count for every group.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = []
    for name in (item.strip() for item in args.groups.split(",")):
        if name not in GROUPS:
            raise ValueError(f"Unknown projection group {name!r}; choose from {sorted(GROUPS)}")
        group = GROUPS[name]
        if args.tokens > 0:
            group = ProjectionGroup(group.name, args.tokens, group.in_features, group.out_features)
        selected.append(group)

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_capability": torch.cuda.get_device_capability(),
        "warmup": args.warmup,
        "iters": args.iters,
        "groups": [_run_group(group, args.warmup, args.iters, args.seed + index) for index, group in enumerate(selected)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
