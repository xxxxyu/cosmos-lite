# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shape-tuned SM89 FP8 GEMM used by the optional RoboLab runtime backend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_scaled_mm_kernel(
    a,
    b,
    scale_a,
    scale_b,
    out,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    a_ptrs = a + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, k, block_k):
        a_tile = tl.load(
            a_ptrs,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] + k_start < k),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptrs,
            mask=(offsets_k[:, None] + k_start < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    accumulator *= tl.load(scale_a + offsets_m, mask=offsets_m < m, other=0.0)[:, None]
    accumulator *= tl.load(scale_b + offsets_n, mask=offsets_n < n, other=0.0)[None, :]
    out_ptrs = out + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(out_ptrs, accumulator, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


# Exact released Edge Gen shapes validated on RTX 4090. Unknown shapes stay on
# vLLM CUTLASS instead of relying on an unvalidated generic Triton policy.
_KERNEL_CONFIGS: dict[tuple[int, int, int], tuple[int, int, int, int, int, int]] = {
    # M, K, N: block M, block N, block K, warps, stages, grouped M
    (3093, 2048, 9216): (64, 128, 64, 4, 3, 1),
    (3093, 2048, 2048): (64, 128, 64, 4, 3, 1),
    (3093, 2048, 1024): (128, 64, 64, 4, 3, 1),
    (3093, 9216, 2048): (128, 64, 64, 4, 3, 1),
}


def supports_shape(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.ndim == 2 and b.ndim == 2 and (a.shape[0], a.shape[1], b.shape[1]) in _KERNEL_CONFIGS


@torch.library.triton_op("cosmos_lite::sm89_fp8_scaled_mm", mutates_args=())
def _scaled_mm_op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    group_m: int,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

    torch.library.wrap_triton(_fp8_scaled_mm_kernel)[grid](
        a,
        b,
        scale_a,
        scale_b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("SM89 FP8 GEMM requires E4M3 inputs")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("SM89 FP8 GEMM requires inputs on the same CUDA device")
    if b.shape[0] != a.shape[1]:
        raise ValueError(f"Incompatible FP8 GEMM shapes: {tuple(a.shape)} and {tuple(b.shape)}")
    config = _KERNEL_CONFIGS.get((a.shape[0], a.shape[1], b.shape[1]))
    if config is None:
        raise ValueError(f"No tuned SM89 FP8 kernel for M={a.shape[0]}, K={a.shape[1]}, N={b.shape[1]}")
    return _scaled_mm_op(a, b, scale_a, scale_b, *config)
