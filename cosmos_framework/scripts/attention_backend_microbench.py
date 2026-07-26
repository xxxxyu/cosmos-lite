#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare optional inference attention kernels on Cosmos policy shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=3093)
    parser.add_argument("--kv-len", type=int, default=3175)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _measure(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        output = fn()
        end.record()
    torch.cuda.synchronize()
    return output, [start.elapsed_time(end) for start, end in zip(starts, ends)]


def main() -> None:
    args = _parse_args()
    if args.causal and args.q_len != args.kv_len:
        raise SystemExit("Causal attention requires equal Q and KV sequence lengths")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape_q = (1, args.q_len, args.q_heads, args.head_dim)
    shape_kv = (1, args.kv_len, args.kv_heads, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=generator)

    from flash_attn import flash_attn_func
    from sageattention import sageattn_qk_int8_pv_fp8_cuda

    backends: dict[str, Callable[[], torch.Tensor]] = {
        "flash2": lambda: flash_attn_func(q, k, v, causal=args.causal),
        "sage_fp8_fp32+fp32": lambda: sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=args.causal,
            pv_accum_dtype="fp32+fp32",
        ),
        "sage_fp8_fp32": lambda: sageattn_qk_int8_pv_fp8_cuda(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=args.causal,
            pv_accum_dtype="fp32",
        ),
    }

    outputs: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, fn in backends.items():
        try:
            output, latencies = _measure(fn, args.warmup, args.iters)
        except Exception as error:
            rows.append({"backend": name, "status": f"{type(error).__name__}: {error}"})
            continue
        outputs[name] = output
        rows.append(
            {
                "backend": name,
                "status": "ok",
                "latency_p50_ms": statistics.median(latencies),
                "latency_min_ms": min(latencies),
            }
        )

    reference = outputs.get("flash2")
    if reference is not None:
        for row in rows:
            output = outputs.get(str(row["backend"]))
            if output is None:
                continue
            error = (output.float() - reference.float()).abs().flatten()
            row.update(
                l1_mean=float(error.mean()),
                element_p95=float(torch.quantile(error, 0.95)),
                linf=float(error.max()),
            )

    print(
        json.dumps(
            {
                "shape": {
                    "q": shape_q,
                    "kv": shape_kv,
                    "causal": args.causal,
                    "dtype": "bfloat16",
                },
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
