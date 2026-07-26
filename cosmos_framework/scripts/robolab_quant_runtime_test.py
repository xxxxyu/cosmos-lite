# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.scripts import robolab_quant_runtime as runtime

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_quant_linear_shape_recorder_groups_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "linear_shapes.jsonl"
    backend = torch.nn.Identity()
    backend.size_n = 9216  # type: ignore[attr-defined]
    x = torch.empty(2, 41, 2048, dtype=torch.bfloat16)

    monkeypatch.setattr(runtime, "_LINEAR_SHAPES_JSONL", str(output))
    runtime._LINEAR_SHAPE_COUNTS.clear()
    runtime._record_quant_linear_shape("net.layers.0.mlp.up_proj", backend, x)
    runtime._record_quant_linear_shape("net.layers.0.mlp.up_proj", backend, x)
    runtime.flush_quant_linear_shapes()

    assert json.loads(output.read_text()) == {
        "event": "quant_linear_shape",
        "name": "net.layers.0.mlp.up_proj",
        "backend_class": "Identity",
        "batch_tokens": 82,
        "in_features": 2048,
        "out_features": 9216,
        "input_dtype": "bfloat16",
        "count": 2,
    }


def test_loader_installs_compatible_fp8_projection_groups() -> None:
    class FakeFp8Backend(torch.nn.Module):
        pass

    class FakeGroup(torch.nn.Module):
        def __init__(self, projections: list[torch.nn.Module]) -> None:
            super().__init__()
            self.projections = torch.nn.ModuleList(projections)

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen"):
                setattr(self, name, runtime.QuantLinearWithOptionalBias(FakeFp8Backend(), None, name=name))

    class FakeMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in ("gate_proj", "up_proj"):
                setattr(self, name, runtime.QuantLinearWithOptionalBias(FakeFp8Backend(), None, name=name))

    network = torch.nn.Module()
    network.attention = FakeAttention()
    network.mlp = FakeMlp()
    loader = object.__new__(runtime.RobolabDirectQuantLoader)
    loader.fp8_projection_fusion = "shared"
    loader.fp8_projection_groups = {"qkv": 0, "gate_up": 0}

    loader._install_fp8_projection_groups(
        network,
        SimpleNamespace(
            VllmCutlassFp8W8A8Linear=FakeFp8Backend,
            VllmCutlassFp8ProjectionGroup=FakeGroup,
        ),
    )

    assert isinstance(network.attention.qkv_proj_moe_gen, FakeGroup)
    assert isinstance(network.attention.q_proj_moe_gen, torch.nn.Identity)
    assert isinstance(network.attention.k_proj_moe_gen, torch.nn.Identity)
    assert isinstance(network.attention.v_proj_moe_gen, torch.nn.Identity)
    assert isinstance(network.mlp.gate_up_proj, FakeGroup)
    assert isinstance(network.mlp.gate_proj, torch.nn.Identity)
    assert isinstance(network.mlp.up_proj, torch.nn.Identity)
    assert loader.fp8_projection_groups == {"qkv": 1, "gate_up": 1}
