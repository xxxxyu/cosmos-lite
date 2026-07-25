# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

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
