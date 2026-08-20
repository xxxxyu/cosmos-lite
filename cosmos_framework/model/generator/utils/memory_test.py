# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.utils.memory import ConditionKVCacheState


@pytest.mark.L0
def test_condition_kv_cache_populates_then_becomes_read_only() -> None:
    state = ConditionKVCacheState()
    state.init({}, torch.device("cpu"))
    assert not state.is_gen_only()
    assert state.read_for_layer(0).frame_idx == 0

    gen_k = torch.randn(1, 2, 1, 4)
    gen_v = torch.randn_like(gen_k)
    und_k = torch.randn(1, 3, 1, 4)
    und_v = torch.randn_like(und_k)
    state.write_for_layer(0, (gen_k, gen_v, und_k, und_v))

    state.init({}, torch.device("cpu"))
    assert state.is_gen_only()
    cached = state.read_for_layer(0)
    assert cached.frame_idx == 1
    torch.testing.assert_close(cached.und_k, und_k)
    torch.testing.assert_close(cached.und_v, und_v)

    replacement = torch.zeros_like(und_k)
    state.write_for_layer(0, (gen_k, gen_v, replacement, replacement))
    torch.testing.assert_close(state.read_for_layer(0).und_k, und_k)


@pytest.mark.L0
def test_condition_kv_cache_rejects_partial_population() -> None:
    state = ConditionKVCacheState()
    state.init({}, torch.device("cpu"))
    tensor = torch.randn(1, 1, 1, 4)
    state.write_for_layer(0, (tensor, tensor, tensor, tensor))
    state.init({}, torch.device("cpu"))

    with pytest.raises(RuntimeError, match="incomplete at layer 1"):
        state.read_for_layer(1)
