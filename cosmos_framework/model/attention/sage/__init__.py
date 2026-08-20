# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional SageAttention backend for SM89 policy inference."""

import os

import torch

from cosmos_framework.model.attention.utils.safe_ops import log


def sage_supported() -> bool:
    if os.environ.get("COSMOS3_SAGE_ATTENTION", "0") != "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from sageattention.core import SM89_ENABLED
    except Exception as error:
        log.debug(f"SageAttention is unavailable: {error}")
        return False
    if not SM89_ENABLED:
        log.debug("SageAttention was installed without its SM89 kernel.")
        return False
    return True


SAGE_SUPPORTED = sage_supported()

if SAGE_SUPPORTED:
    from cosmos_framework.model.attention.sage.functions import sage_attention
else:
    from cosmos_framework.model.attention.sage.stubs import sage_attention

__all__ = ["SAGE_SUPPORTED", "sage_attention"]
