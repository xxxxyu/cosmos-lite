#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Microbenchmark linear backends for Cosmos3 quantization candidates.

This is intentionally independent from the policy server. It validates the
minimum contract needed before adapting a backend into Cosmos:

- module-local replacement,
- BF16 activation input/output handoff,
- mixed precision chains, and
- latency/memory/error against a BF16 Linear baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class LinearShape:
    batch_tokens: int
    in_features: int
    out_features: int


@dataclass(frozen=True)
class LinearShapeSpec:
    shape: LinearShape
    count: int = 1
    source_backend_class: str = ""
    example_name: str = ""


class Bf16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight).to(torch.bfloat16)


def _pack_rows(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    if q_w.shape != (size_k, size_n):
        raise ValueError(f"Expected q_w {(size_k, size_n)}, got {tuple(q_w.shape)}")
    pack_factor = 32 // num_bits
    if size_k % pack_factor != 0:
        raise ValueError(f"size_k={size_k} must be divisible by pack_factor={pack_factor}")
    q_w_cpu = q_w.detach().to("cpu", torch.int32)
    q_res = torch.zeros((size_k // pack_factor, size_n), dtype=torch.int32)
    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        q_res |= (q_w_cpu[i::pack_factor, :] & mask) << (num_bits * i)
    return q_res.to(q_w.device, non_blocking=True).contiguous()


def _marlin_permute_scales(scales: torch.Tensor, size_k: int, size_n: int, group_size: int) -> torch.Tensor:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        return scales.reshape((-1, len(scale_perm)))[:, scale_perm].reshape((-1, size_n)).contiguous()
    return scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single].reshape((-1, size_n)).contiguous()


def _vllm_uint4b8_id() -> int:
    return _vllm_uintxb_id(mantissa=4, bias=8)


def _vllm_uint8b128_id() -> int:
    return _vllm_uintxb_id(mantissa=8, bias=128)


def _vllm_uintxb_id(mantissa: int, bias: int) -> int:
    exponent = 0
    signed = 0
    finite_values_only = 0
    nan_repr_ieee754 = 1
    return (
        exponent
        | (mantissa << 8)
        | (signed << 16)
        | (bias << 17)
        | (finite_values_only << 49)
        | (nan_repr_ieee754 << 50)
    )


def _gptq_uint4b8_quantize_vllm_compatible(
    weight_kn: torch.Tensor, group_size: int, collect_ref: bool = False
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Equivalent to vLLM quant_utils.gptq_quantize_weights for uint4b8/no act_order."""
    orig_type = weight_kn.dtype
    size_k, size_n = weight_kn.shape
    if group_size == -1:
        group_size = size_k

    work = weight_kn
    grouped_layout = group_size < size_k
    if grouped_layout:
        work = work.reshape((-1, group_size, size_n))
        work = work.permute(1, 0, 2)
        work = work.reshape((group_size, -1))

    max_val = torch.max(work, 0, keepdim=True).values
    min_val = torch.min(work, 0, keepdim=True).values
    scales = torch.max(torch.abs(max_val / 7.0), torch.abs(min_val / -8.0)).clamp(min=1e-5)
    q_weight = torch.round(work / scales).to(torch.int32).clamp(min=-8, max=7)
    quant_ref = q_weight.to(orig_type) * scales if collect_ref else None
    q_weight = q_weight + 8

    if grouped_layout:
        def restore(t: torch.Tensor) -> torch.Tensor:
            return t.reshape((group_size, -1, size_n)).permute(1, 0, 2).reshape((size_k, size_n)).contiguous()

        q_weight = restore(q_weight)
        quant_ref = restore(quant_ref) if quant_ref is not None else None
        scales = scales.reshape((-1, size_n)).contiguous()

    return quant_ref, q_weight.contiguous(), scales.contiguous()


def _gptq_uint8b128_quantize_per_channel(
    weight_kn: torch.Tensor, collect_ref: bool = False
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    max_val = torch.max(weight_kn, 0, keepdim=True).values
    min_val = torch.min(weight_kn, 0, keepdim=True).values
    scales = torch.max(torch.abs(max_val / 127.0), torch.abs(min_val / -128.0)).clamp(min=1e-5)
    q_weight = torch.round(weight_kn / scales).to(torch.int32).clamp(min=-128, max=127)
    quant_ref = q_weight.to(weight_kn.dtype) * scales if collect_ref else None
    q_weight = (q_weight + 128).to(torch.uint8).contiguous()
    return quant_ref, q_weight, scales.contiguous()


def _collect_quant_weight_debug() -> bool:
    return os.environ.get("COSMOS3_QUANT_WEIGHT_DEBUG", "0") == "1"


class VllmGptqMarlinW4A16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, group_size: int = 128, input_scale: torch.Tensor | None = None) -> None:
        super().__init__()
        if not torch.cuda.is_available() or weight.device.type != "cuda":
            raise RuntimeError("vLLM Marlin backend requires CUDA tensors")
        self.size_n, self.size_k = weight.shape
        self.group_size = group_size
        self.num_bits = 4
        if self.size_n % 64 != 0:
            raise ValueError(f"Marlin requires out_features divisible by 64, got {self.size_n}")
        if self.size_k % 128 != 0:
            raise ValueError(f"Marlin requires in_features divisible by 128, got {self.size_k}")
        if group_size not in {-1, 32, 64, 128}:
            raise ValueError(f"Unsupported Marlin group_size={group_size}")
        if group_size != -1 and self.size_k % group_size != 0:
            raise ValueError(f"in_features={self.size_k} must be divisible by group_size={group_size}")

        import vllm._C  # noqa: F401  # Registers torch.ops._C Marlin kernels.

        if input_scale is not None:
            if input_scale.numel() != self.size_k:
                raise ValueError(f"input_scale must have {self.size_k} elements, got {input_scale.numel()}")
            scale = input_scale.detach().to(device=weight.device, dtype=torch.bfloat16).reshape(1, self.size_k)
            scaled_weight = weight.detach().to(torch.bfloat16) * scale
            self.register_buffer("input_scale", scale.reshape(self.size_k), persistent=False)
        else:
            scaled_weight = weight.detach().to(torch.bfloat16)
            self.input_scale = None

        weight_kn = scaled_weight.t().contiguous()
        collect_ref = _collect_quant_weight_debug()
        quant_ref, q_weight, scales = _gptq_uint4b8_quantize_vllm_compatible(weight_kn, group_size, collect_ref)
        if quant_ref is not None:
            diff = (quant_ref.float() - weight_kn.float()).abs()
            self.quant_weight_l1_mean = float(diff.mean().item())
            self.quant_weight_linf = float(diff.max().item())
        else:
            self.quant_weight_l1_mean = float("nan")
            self.quant_weight_linf = float("nan")
        q_packed = _pack_rows(q_weight, self.num_bits, self.size_k, self.size_n)
        empty_perm = torch.empty(0, dtype=torch.int, device=weight.device)
        q_marlin = torch.ops._C.gptq_marlin_repack(
            q_packed.contiguous(),
            empty_perm,
            self.size_k,
            self.size_n,
            self.num_bits,
            False,
        )
        permuted_scales = _marlin_permute_scales(
            scales.contiguous(),
            self.size_k,
            self.size_n,
            group_size,
        )
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        workspace = torch.zeros(sms, dtype=torch.int, device=weight.device)
        self.register_buffer("qweight", q_marlin, persistent=False)
        self.register_buffer("scales", permuted_scales, persistent=False)
        self.register_buffer("workspace", workspace, persistent=False)
        self.register_buffer("empty", empty_perm, persistent=False)
        self.wtype_id = _vllm_uint4b8_id()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if self.input_scale is not None:
            x_2d = (x_2d / self.input_scale).contiguous()
        out = torch.ops._C.marlin_gemm(
            x_2d,
            None,
            self.qweight,
            None,
            self.scales,
            None,
            None,
            self.empty,
            self.empty,
            self.empty,
            self.workspace,
            self.wtype_id,
            x_2d.shape[0],
            self.size_n,
            self.size_k,
            True,
            False,
            True,
            False,
        )
        return out.reshape(x.shape[:-1] + (self.size_n,)).to(torch.bfloat16)


class VllmGptqMarlinW8A16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, group_size: int = -1) -> None:
        super().__init__()
        if not torch.cuda.is_available() or weight.device.type != "cuda":
            raise RuntimeError("vLLM Marlin backend requires CUDA tensors")
        self.size_n, self.size_k = weight.shape
        self.group_size = group_size
        self.num_bits = 8
        if self.size_n % 64 != 0:
            raise ValueError(f"Marlin requires out_features divisible by 64, got {self.size_n}")
        if self.size_k % 128 != 0:
            raise ValueError(f"Marlin requires in_features divisible by 128, got {self.size_k}")
        if group_size not in {-1, 32, 64, 128}:
            raise ValueError(f"Unsupported Marlin group_size={group_size}")
        if group_size != -1 and self.size_k % group_size != 0:
            raise ValueError(f"in_features={self.size_k} must be divisible by group_size={group_size}")

        import vllm._C  # noqa: F401  # Registers torch.ops._C Marlin kernels.

        weight_kn = weight.detach().to(torch.bfloat16).t().contiguous()
        collect_ref = _collect_quant_weight_debug()
        quant_ref, q_weight, scales = _gptq_uint8b128_quantize_per_channel(weight_kn, collect_ref)
        if quant_ref is not None:
            diff = (quant_ref.float() - weight_kn.float()).abs()
            self.quant_weight_l1_mean = float(diff.mean().item())
            self.quant_weight_linf = float(diff.max().item())
        else:
            self.quant_weight_l1_mean = float("nan")
            self.quant_weight_linf = float("nan")
        q_packed = _pack_rows(q_weight.to(torch.int32), self.num_bits, self.size_k, self.size_n)
        empty_perm = torch.empty(0, dtype=torch.int, device=weight.device)
        q_marlin = torch.ops._C.gptq_marlin_repack(
            q_packed.contiguous(),
            empty_perm,
            self.size_k,
            self.size_n,
            self.num_bits,
            False,
        )
        permuted_scales = _marlin_permute_scales(
            scales.contiguous(),
            self.size_k,
            self.size_n,
            group_size,
        )
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        workspace = torch.zeros(sms, dtype=torch.int, device=weight.device)
        self.register_buffer("qweight", q_marlin, persistent=False)
        self.register_buffer("scales", permuted_scales, persistent=False)
        self.register_buffer("workspace", workspace, persistent=False)
        self.register_buffer("empty", empty_perm, persistent=False)
        self.wtype_id = _vllm_uint8b128_id()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.ops._C.marlin_gemm(
            x_2d,
            None,
            self.qweight,
            None,
            self.scales,
            None,
            None,
            self.empty,
            self.empty,
            self.empty,
            self.workspace,
            self.wtype_id,
            x_2d.shape[0],
            self.size_n,
            self.size_k,
            True,
            False,
            True,
            False,
        )
        return out.reshape(x.shape[:-1] + (self.size_n,)).to(torch.bfloat16)


class VllmCutlassFp8W8A8Linear(nn.Module):
    """CUTLASS FP8 linear with per-channel weights and dynamic per-token activations."""

    def __init__(self, weight: torch.Tensor, input_scale: torch.Tensor | None = None) -> None:
        super().__init__()
        if not torch.cuda.is_available() or weight.device.type != "cuda":
            raise RuntimeError("vLLM CUTLASS FP8 backend requires CUDA tensors")
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("vLLM CUTLASS FP8 backend requires torch.float8_e4m3fn")

        import vllm._C  # noqa: F401  # Registers vLLM CUTLASS and quant kernels.
        from vllm import _custom_ops as ops

        capability = torch.cuda.get_device_capability(weight.device)
        capability_int = capability[0] * 10 + capability[1]
        if capability_int < 89 or not ops.cutlass_scaled_mm_supports_fp8(capability_int):
            raise RuntimeError(
                f"vLLM CUTLASS FP8 W8A8 requires native FP8 support (SM89+), got SM{capability_int}"
            )

        self.size_n, self.size_k = weight.shape
        self.num_bits = 8
        self.activation_bits = 8
        self.fp8_max = 448.0
        weight_nk = weight.detach().to(torch.bfloat16).contiguous()
        if input_scale is not None:
            if input_scale.numel() != self.size_k:
                raise ValueError(f"input_scale must have {self.size_k} elements, got {input_scale.numel()}")
            scale = input_scale.detach().to(device=weight.device, dtype=torch.bfloat16).reshape(1, self.size_k)
            weight_nk = weight_nk * scale
            self.register_buffer("input_scale", scale.reshape(self.size_k), persistent=False)
        else:
            self.input_scale = None
        scale_b = (weight_nk.abs().amax(dim=1, keepdim=True).float() / self.fp8_max).clamp(min=1e-8)
        qweight_nk = (weight_nk / scale_b).clamp(-self.fp8_max, self.fp8_max).to(torch.float8_e4m3fn)
        self.register_buffer("qweight_nk", qweight_nk.contiguous(), persistent=False)
        self.register_buffer("scale_b", scale_b.reshape(1, self.size_n).contiguous(), persistent=False)

    @staticmethod
    def _ops():
        from vllm import _custom_ops as ops

        return ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if self.input_scale is not None:
            x_2d = (x_2d / self.input_scale).contiguous()
        q_x, scale_a = self._ops().scaled_fp8_quant(x_2d, use_per_token_if_dynamic=True)
        out = self._ops().cutlass_scaled_mm(
            q_x,
            self.qweight_nk.t(),
            scale_a,
            self.scale_b,
            torch.bfloat16,
        )
        return out.reshape(x.shape[:-1] + (self.size_n,)).to(torch.bfloat16)


class VllmAllSparkW8A16Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if not torch.cuda.is_available() or weight.device.type != "cuda":
            raise RuntimeError("vLLM AllSpark backend requires CUDA tensors")
        self.size_n, self.size_k = weight.shape
        props = torch.cuda.get_device_properties(weight.device)
        self.sm_count = props.multi_processor_count
        self.sm_version = props.major * 10 + props.minor
        if not (80 <= self.sm_version < 90):
            raise RuntimeError(f"AllSpark W8A16 supports 80 <= SM < 90; got SM{self.sm_version}")
        if self.size_k % 16 != 0 or self.size_n % 16 != 0:
            raise ValueError("AllSpark W8A16 requires in/out features divisible by 16")

        import vllm._C  # noqa: F401  # Registers torch.ops._C AllSpark kernels.

        weight_kn = weight.detach().to(torch.bfloat16).t().contiguous()
        collect_ref = _collect_quant_weight_debug()
        quant_ref, q_weight, scales = _gptq_uint8b128_quantize_per_channel(weight_kn, collect_ref)
        if quant_ref is not None:
            diff = (quant_ref.float() - weight_kn.float()).abs()
            self.quant_weight_l1_mean = float(diff.mean().item())
            self.quant_weight_linf = float(diff.max().item())
        else:
            self.quant_weight_l1_mean = float("nan")
            self.quant_weight_linf = float("nan")
        n_aligned = ((self.size_n + 31) // 32) * 32
        q_reorder = torch.empty((n_aligned, self.size_k), device=weight.device, dtype=torch.uint8)
        scale_reorder = torch.empty((1, n_aligned), device=weight.device, dtype=scales.dtype)
        torch.ops._C.rearrange_kn_weight_as_n32k16_order(
            q_weight,
            scales,
            None,
            False,
            q_reorder,
            scale_reorder,
            None,
            self.size_k,
            self.size_n,
            n_aligned,
        )
        self.register_buffer("qweight", q_reorder, persistent=False)
        self.register_buffer("scales", scale_reorder, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.ops._C.allspark_w8a16_gemm(
            x_2d,
            self.qweight,
            self.scales,
            None,
            self.size_n,
            -1,
            self.sm_count,
            self.sm_version,
            1024,
            False,
            True,
        )
        return out.reshape(x.shape[:-1] + (self.size_n,)).to(torch.bfloat16)


def _make_torchao_linear(weight: torch.Tensor, mode: str) -> nn.Module:
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16, device=weight.device)
    layer.weight.data.copy_(weight.to(torch.bfloat16))
    layer.eval()
    if mode == "torchao_int8wo":
        config = Int8WeightOnlyConfig()
    elif mode == "torchao_int4wo":
        config = Int4WeightOnlyConfig(group_size=128)
    else:
        raise ValueError(mode)
    quantize_(layer, config)
    return layer


def _make_backend(name: str, weight: torch.Tensor) -> nn.Module:
    if name == "bf16":
        return Bf16Linear(weight)
    if name in {"torchao_int8wo", "torchao_int4wo"}:
        return _make_torchao_linear(weight, name)
    if name == "vllm_gptq_marlin_w4a16":
        return VllmGptqMarlinW4A16Linear(weight)
    if name == "vllm_gptq_marlin_w8a16":
        return VllmGptqMarlinW8A16Linear(weight)
    if name == "vllm_cutlass_fp8_w8a8":
        return VllmCutlassFp8W8A8Linear(weight)
    if name == "vllm_allspark_w8a16":
        return VllmAllSparkW8A16Linear(weight)
    raise ValueError(f"Unsupported backend {name!r}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _mem_gb() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def _module_storage_gb(module: nn.Module) -> float:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total / 1e9


def _quant_debug(module: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("quant_weight_l1_mean", "quant_weight_linf"):
        if hasattr(module, key):
            out[key] = float(getattr(module, key))
    return out


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = None
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
        _sync()
        start = time.perf_counter()
        for _ in range(iters):
            out = fn()
        _sync()
    assert out is not None
    return (time.perf_counter() - start) * 1000.0 / max(1, iters), out


def _one_backend(shape: LinearShape, backend: str, warmup: int, iters: int, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    x = torch.randn(shape.batch_tokens, shape.in_features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(shape.out_features, shape.in_features, device=device, dtype=torch.bfloat16) / (
        shape.in_features**0.5
    )
    ref = Bf16Linear(weight).eval()
    _sync()
    build_start = time.perf_counter()
    candidate = _make_backend(backend, weight).eval()
    _sync()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    def ref_fn() -> torch.Tensor:
        return ref(x)

    def cand_fn() -> torch.Tensor:
        return candidate(x).to(torch.bfloat16)

    ref_ms, ref_out = _time_ms(ref_fn, warmup=warmup, iters=iters)
    cand_ms, cand_out = _time_ms(cand_fn, warmup=warmup, iters=iters)
    diff = (cand_out.float() - ref_out.float()).abs()
    return {
        "backend": backend,
        "shape": shape.__dict__,
        "ref_bf16_ms": ref_ms,
        "candidate_ms": cand_ms,
        "candidate_build_ms": build_ms,
        "ref_storage_gb": _module_storage_gb(ref),
        "candidate_storage_gb": _module_storage_gb(candidate),
        "candidate_storage_ratio_vs_bf16": _module_storage_gb(candidate) / _module_storage_gb(ref),
        **_quant_debug(candidate),
        "latency_ratio_vs_bf16": cand_ms / ref_ms if ref_ms > 0 else None,
        "l1_mean": float(diff.mean().item()),
        "linf": float(diff.max().item()),
        **_mem_gb(),
    }


def _chain(shape: LinearShape, backend_a: str, backend_b: str, warmup: int, iters: int, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    hidden = shape.out_features
    x = torch.randn(shape.batch_tokens, shape.in_features, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(hidden, shape.in_features, device=device, dtype=torch.bfloat16) / (shape.in_features**0.5)
    w2 = torch.randn(shape.in_features, hidden, device=device, dtype=torch.bfloat16) / (hidden**0.5)
    ref1 = Bf16Linear(w1).eval()
    ref2 = Bf16Linear(w2).eval()
    _sync()
    build_start = time.perf_counter()
    cand1 = _make_backend(backend_a, w1).eval()
    cand2 = _make_backend(backend_b, w2).eval()
    _sync()
    build_ms = (time.perf_counter() - build_start) * 1000.0

    def ref_fn() -> torch.Tensor:
        return ref2(torch.nn.functional.silu(ref1(x)).to(torch.bfloat16))

    def cand_fn() -> torch.Tensor:
        y = cand1(x).to(torch.bfloat16)
        return cand2(torch.nn.functional.silu(y).to(torch.bfloat16)).to(torch.bfloat16)

    ref_ms, ref_out = _time_ms(ref_fn, warmup=warmup, iters=iters)
    cand_ms, cand_out = _time_ms(cand_fn, warmup=warmup, iters=iters)
    diff = (cand_out.float() - ref_out.float()).abs()
    return {
        "backend_a": backend_a,
        "backend_b": backend_b,
        "shape": shape.__dict__,
        "ref_bf16_chain_ms": ref_ms,
        "candidate_chain_ms": cand_ms,
        "candidate_build_ms": build_ms,
        "ref_storage_gb": _module_storage_gb(ref1) + _module_storage_gb(ref2),
        "candidate_storage_gb": _module_storage_gb(cand1) + _module_storage_gb(cand2),
        "candidate_storage_ratio_vs_bf16": (_module_storage_gb(cand1) + _module_storage_gb(cand2))
        / (_module_storage_gb(ref1) + _module_storage_gb(ref2)),
        "latency_ratio_vs_bf16": cand_ms / ref_ms if ref_ms > 0 else None,
        "l1_mean": float(diff.mean().item()),
        "linf": float(diff.max().item()),
        **_mem_gb(),
    }


def _load_shape_specs(path: Path, max_shapes: int) -> list[LinearShapeSpec]:
    grouped: dict[tuple[int, int, int, str], dict[str, object]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") not in {None, "quant_linear_shape"}:
            continue
        try:
            batch_tokens = int(record["batch_tokens"])
            in_features = int(record["in_features"])
            out_features = int(record["out_features"])
        except KeyError:
            continue
        if batch_tokens <= 0 or in_features <= 0 or out_features <= 0:
            continue
        backend_class = str(record.get("backend_class", ""))
        key = (batch_tokens, in_features, out_features, backend_class)
        entry = grouped.setdefault(
            key,
            {
                "count": 0,
                "example_name": str(record.get("name", "")),
            },
        )
        entry["count"] = int(entry["count"]) + int(record.get("count", 1))
    specs = [
        LinearShapeSpec(
            shape=LinearShape(batch_tokens, in_features, out_features),
            count=int(entry["count"]),
            source_backend_class=backend_class,
            example_name=str(entry["example_name"]),
        )
        for (batch_tokens, in_features, out_features, backend_class), entry in grouped.items()
    ]
    specs.sort(key=lambda item: item.count, reverse=True)
    return specs[:max_shapes] if max_shapes > 0 else specs


def _weighted_single_backend_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        if "error" in row:
            continue
        backend = str(row["backend"])
        count = float(row.get("shape_count", 1))
        entry = grouped.setdefault(
            backend,
            {
                "weighted_candidate_ms": 0.0,
                "weighted_ref_bf16_ms": 0.0,
                "count": 0.0,
                "shapes": 0.0,
            },
        )
        entry["weighted_candidate_ms"] += float(row["candidate_ms"]) * count
        entry["weighted_ref_bf16_ms"] += float(row["ref_bf16_ms"]) * count
        entry["count"] += count
        entry["shapes"] += 1.0
    summary: list[dict[str, object]] = []
    for backend, entry in sorted(grouped.items()):
        count = max(entry["count"], 1.0)
        cand = entry["weighted_candidate_ms"] / count
        ref = entry["weighted_ref_bf16_ms"] / count
        summary.append(
            {
                "backend": backend,
                "weighted_avg_candidate_ms": cand,
                "weighted_avg_ref_bf16_ms": ref,
                "weighted_latency_ratio_vs_bf16": cand / ref if ref > 0 else None,
                "total_shape_calls": int(entry["count"]),
                "benchmarked_shapes": int(entry["shapes"]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backends",
        default="bf16,torchao_int8wo,torchao_int4wo,vllm_gptq_marlin_w4a16,vllm_allspark_w8a16",
    )
    parser.add_argument(
        "--chain",
        default=(
            "bf16:torchao_int8wo,torchao_int8wo:bf16,torchao_int4wo:bf16,"
            "bf16:vllm_gptq_marlin_w4a16,vllm_gptq_marlin_w4a16:bf16,"
            "bf16:vllm_gptq_marlin_w8a16,vllm_gptq_marlin_w8a16:bf16"
        ),
    )
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=12288)
    parser.add_argument("--shape-file", default="", help="Optional JSONL from COSMOS3_LINEAR_SHAPES_JSONL.")
    parser.add_argument("--max-shapes", type=int, default=16, help="Top shape groups to benchmark when --shape-file is set.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.shape_file:
        shape_specs = _load_shape_specs(Path(args.shape_file), args.max_shapes)
        if not shape_specs:
            raise ValueError(f"No valid linear shape records found in {args.shape_file}")
    else:
        shape_specs = [LinearShapeSpec(LinearShape(args.batch_tokens, args.in_features, args.out_features))]
    results: dict[str, object] = {
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        "shape_file": args.shape_file or None,
        "shape_specs": [
            {
                "shape": spec.shape.__dict__,
                "count": spec.count,
                "source_backend_class": spec.source_backend_class,
                "example_name": spec.example_name,
            }
            for spec in shape_specs
        ],
        "notes": {
            "vllm_gptq_marlin_w4a16": "offline symmetric per-group W4 quantization; forward uses vLLM Marlin ops only",
            "vllm_gptq_marlin_w8a16": "offline symmetric per-channel W8 quantization; forward uses vLLM Marlin ops only",
            "vllm_cutlass_fp8_w8a8": "offline per-channel FP8 weights and dynamic per-token FP8 activations; requires SM89+",
            "vllm_allspark_w8a16": "offline symmetric per-channel W8 quantization; forward uses vLLM AllSpark ops; gated to 80 <= SM < 90",
            "storage_gb": "module parameter/buffer bytes only; excludes temporary original BF16 tensors held by harness",
        },
        "single_backend": [],
        "mixed_chain": [],
    }
    backend_specs = [] if args.backends.strip().lower() in {"", "none", "null", "off"} else args.backends.split(",")
    for shape_index, shape_spec in enumerate(shape_specs):
        shape = shape_spec.shape
        for backend in [item.strip() for item in backend_specs if item.strip()]:
            try:
                row = _one_backend(shape, backend, args.warmup, args.iters, args.seed + shape_index)
                row.update(
                    {
                        "shape_count": shape_spec.count,
                        "source_backend_class": shape_spec.source_backend_class,
                        "example_name": shape_spec.example_name,
                    }
                )
                results["single_backend"].append(row)
            except Exception as exc:
                results["single_backend"].append(
                    {
                        "backend": backend,
                        "shape": shape.__dict__,
                        "shape_count": shape_spec.count,
                        "source_backend_class": shape_spec.source_backend_class,
                        "example_name": shape_spec.example_name,
                        "error": repr(exc),
                    }
                )
        chain_specs = [] if args.chain.strip().lower() in {"", "none", "null", "off"} else args.chain.split(",")
        for spec in [item.strip() for item in chain_specs if item.strip()]:
            backend_a, backend_b = spec.split(":", 1)
            try:
                row = _chain(shape, backend_a, backend_b, args.warmup, args.iters, args.seed + shape_index)
                row.update(
                    {
                        "shape_count": shape_spec.count,
                        "source_backend_class": shape_spec.source_backend_class,
                        "example_name": shape_spec.example_name,
                    }
                )
                results["mixed_chain"].append(row)
            except Exception as exc:
                results["mixed_chain"].append(
                    {
                        "backend_a": backend_a,
                        "backend_b": backend_b,
                        "shape": shape.__dict__,
                        "shape_count": shape_spec.count,
                        "source_backend_class": shape_spec.source_backend_class,
                        "example_name": shape_spec.example_name,
                        "error": repr(exc),
                    }
                )
    results["single_backend_weighted_summary"] = _weighted_single_backend_summary(results["single_backend"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
