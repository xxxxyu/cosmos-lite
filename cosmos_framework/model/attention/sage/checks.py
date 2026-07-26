# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compatibility checks for the opt-in SageAttention SM89 backend."""

from functools import partial

import torch

from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.sage import SAGE_SUPPORTED
from cosmos_framework.model.attention.utils import get_arch_tag, log_or_raise_error


def sage_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType | None,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    del causal_type
    reject = partial(log_or_raise_error, raise_error=raise_error)

    def unsupported(reason: str) -> bool:
        reject(reason, exception=RuntimeError)
        return False

    if not SAGE_SUPPORTED:
        return unsupported("SageAttention is disabled or its optional SM89 extension is unavailable.")
    if get_arch_tag(device) != 89:
        return unsupported("The Cosmos SageAttention backend is validated only on SM89.")
    if requires_grad:
        return unsupported("The Cosmos SageAttention backend is inference-only.")
    if deterministic:
        return unsupported("SageAttention does not provide deterministic execution guarantees.")
    if is_varlen:
        return unsupported("The Cosmos SageAttention backend currently supports dense attention only.")
    if is_causal:
        return unsupported("Short causal policy attention remains faster and more accurate with FlashAttention2.")
    if dtype not in {torch.float16, torch.bfloat16}:
        return unsupported("SageAttention requires FP16 or BF16 Q/K/V tensors.")
    if len(query_shape) != 4 or len(key_shape) != 4 or len(value_shape) != 4:
        return unsupported("SageAttention requires four-dimensional BSHD tensors.")
    if query_shape[1] < 512:
        return unsupported("SageAttention is reserved for long policy attention (Q length >= 512).")
    if query_shape[-1] not in {64, 128} or key_shape[-1] != query_shape[-1]:
        return unsupported("SageAttention requires matching Q/K head dimensions of 64 or 128.")
    if value_shape[-1] != query_shape[-1]:
        return unsupported("The Cosmos SageAttention backend does not support MLA value dimensions.")
    if query_shape[2] % key_shape[2] != 0 or key_shape[2] != value_shape[2]:
        return unsupported("Q heads must be divisible by matching K/V head counts.")
    return True
