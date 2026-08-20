# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stub used when the optional SageAttention backend is unavailable."""


def sage_attention(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("SageAttention is disabled or unavailable")
