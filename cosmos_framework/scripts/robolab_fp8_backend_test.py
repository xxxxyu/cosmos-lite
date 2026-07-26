# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest
import torch

from cosmos_framework.scripts import quant_backend_microbench as backends
from cosmos_framework.scripts import robolab_quant_runtime as runtime
from cosmos_framework.scripts import sm89_fp8_gemm

pytestmark = pytest.mark.gpus(1)


def test_sm89_triton_shape_allowlist_is_edge_only() -> None:
    edge_a = torch.empty((3093, 2048), dtype=torch.float8_e4m3fn, device="meta")
    edge_b = torch.empty((2048, 9216), dtype=torch.float8_e4m3fn, device="meta")
    nano_a = torch.empty((3093, 4096), dtype=torch.float8_e4m3fn, device="meta")
    nano_b = torch.empty((4096, 12288), dtype=torch.float8_e4m3fn, device="meta")

    assert sm89_fp8_gemm.supports_shape(edge_a, edge_b)
    assert not sm89_fp8_gemm.supports_shape(nano_a, nano_b)


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

    restored = runtime._make_backend(backends, metadata, payload, device, fp8_gemm_backend="triton_sm89")
    torch.testing.assert_close(restored(x), exported(x), rtol=0, atol=0)
    assert restored.gemm_backend == "triton_sm89"
    assert restored.qweight_nk.dtype == torch.float8_e4m3fn
    assert restored.scale_b.dtype == torch.float32
    torch.testing.assert_close(restored.input_scale, input_scale, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sm89_triton_fp8_backend_matches_cutlass() -> None:
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (8, 9):
        pytest.skip("Tuned Triton backend requires SM89")

    generator = torch.Generator(device=device).manual_seed(17)
    weight = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device, generator=generator)
    x = torch.randn(3093, 2048, dtype=torch.bfloat16, device=device, generator=generator)
    backend = backends.VllmCutlassFp8W8A8Linear(weight)
    expected = backend(x)
    backend.gemm_backend = "triton_sm89"

    actual = backend(x)

    error = (actual.float() - expected.float()).abs()
    assert (error != 0).float().mean().item() < 1e-4
    assert error.max().item() <= 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_projection_group_matches_independent_linears() -> None:
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] * 10 + capability[1] < 89:
        pytest.skip("Native FP8 requires SM89 or newer")

    generator = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(5, 17, 128, dtype=torch.bfloat16, device=device, generator=generator)
    projections = [
        backends.VllmCutlassFp8W8A8Linear(
            torch.randn(size_n, 128, dtype=torch.bfloat16, device=device, generator=generator)
        )
        for size_n in (256, 128, 128)
    ]
    expected = tuple(projection(x) for projection in projections)
    group = backends.VllmCutlassFp8ProjectionGroup(projections)

    actual = group(x)

    for grouped, independent in zip(actual, expected, strict=True):
        torch.testing.assert_close(grouped, independent, rtol=0, atol=0)
