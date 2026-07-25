# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.scripts import quant_backend_microbench as backends
from cosmos_framework.scripts import robolab_quant_runtime as runtime

pytestmark = pytest.mark.gpus(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_backend_payload_round_trip() -> None:
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] * 10 + capability[1] < 89:
        pytest.skip("Native FP8 requires SM89 or newer")

    generator = torch.Generator(device=device).manual_seed(7)
    weight = torch.randn(256, 128, dtype=torch.bfloat16, device=device, generator=generator)
    x = torch.randn(3, 11, 128, dtype=torch.bfloat16, device=device, generator=generator)
    input_scale = torch.linspace(0.5, 1.5, 128, dtype=torch.bfloat16, device=device)
    exported = backends.VllmCutlassFp8W8A8Linear(weight, input_scale=input_scale)
    payload = {
        "qweight_nk": exported.qweight_nk.cpu(),
        "scale_b": exported.scale_b.cpu(),
        "input_scale": exported.input_scale.cpu(),
    }
    metadata = {
        "backend_class": "VllmCutlassFp8W8A8Linear",
        "size_k": 128,
        "size_n": 256,
    }

    restored = runtime._make_backend(backends, metadata, payload, device)
    torch.testing.assert_close(restored(x), exported(x), rtol=0, atol=0)
    assert restored.qweight_nk.dtype == torch.float8_e4m3fn
    assert restored.scale_b.dtype == torch.float32
    torch.testing.assert_close(restored.input_scale, input_scale, rtol=0, atol=0)
