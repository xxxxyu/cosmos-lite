# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-FileCopyrightText: Copyright (c) 2024 SageAttention team.
# SPDX-License-Identifier: Apache-2.0

"""SageAttention adapter for heads-last Cosmos tensors."""

import torch
from sageattention import _fused
from sageattention.core import per_thread_int8_triton, sm89_compile
from torch import Tensor

from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.sage.checks import sage_attention_check


@torch.library.custom_op("cosmos_lite::sage_transpose_pad_permute", mutates_args=("output",), device_types="cuda")
def _transpose_pad_permute(value: Tensor, output: Tensor) -> None:
    _fused.transpose_pad_permute_cuda(value, output, 0)


@_transpose_pad_permute.register_fake
def _transpose_pad_permute_fake(value: Tensor, output: Tensor) -> None:
    del value, output


@torch.library.custom_op(
    "cosmos_lite::sage_scale_fuse_quant",
    mutates_args=("output", "output_scale"),
    device_types="cuda",
)
def _scale_fuse_quant(value: Tensor, output: Tensor, output_scale: Tensor, sequence_length: int) -> None:
    _fused.scale_fuse_quant_cuda(value, output, output_scale, sequence_length, 448.0, 0)


@_scale_fuse_quant.register_fake
def _scale_fuse_quant_fake(value: Tensor, output: Tensor, output_scale: Tensor, sequence_length: int) -> None:
    del value, output, output_scale, sequence_length


def _sage_attention_sm89(query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
    """Graph-native SageAttention 2.2 dispatch without its Dynamo-hostile set_device call."""
    key_mean = key.mean(dim=1, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
        query,
        key,
        key_mean,
        tensor_layout="NHD",
        BLKQ=128,
        WARPQ=32,
        BLKK=64,
        WARPK=64,
    )
    batch, sequence_length, kv_heads, head_dim = value.shape
    padded_length = (sequence_length + 63) // 64 * 64
    value_permuted = torch.empty(
        (batch, head_dim, kv_heads, padded_length),
        dtype=value.dtype,
        device=value.device,
    )
    _transpose_pad_permute(value, value_permuted)
    value_fp8 = torch.empty_like(value_permuted, dtype=torch.float8_e4m3fn)
    value_scale = torch.empty((batch, kv_heads, head_dim), dtype=torch.float32, device=value.device)
    _scale_fuse_quant(value_permuted, value_fp8, value_scale, sequence_length)
    output = torch.empty_like(query)
    sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        q_int8,
        k_int8,
        value_fp8,
        output,
        q_scale,
        k_scale,
        value_scale,
        0,  # NHD
        0,  # non-causal
        3,  # per-thread Q/K quantization
        scale,
        0,  # no LSE output
    )
    return output


def sage_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    del max_seqlen_Q, max_seqlen_KV
    is_varlen = cumulative_seqlen_Q is not None or cumulative_seqlen_KV is not None
    assert sage_attention_check(
        query.shape,
        key.shape,
        value.shape,
        query.dtype,
        query.device,
        query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal,
        causal_type,
        is_varlen,
        deterministic,
        raise_error=True,
    )
    if backend_kwargs:
        raise ValueError(f"The Cosmos SageAttention backend has no runtime options, got {sorted(backend_kwargs)}")
    if return_lse:
        raise ValueError("The Cosmos SageAttention backend does not expose log-sum-exp outputs")
    resolved_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _sage_attention_sm89(query, key, value, resolved_scale)
