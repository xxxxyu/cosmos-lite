# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.attention import backends
from cosmos_framework.model.attention.sage import checks


def _supported(**overrides: object) -> bool:
    args: dict[str, object] = {
        "query_shape": torch.Size((1, 3093, 16, 128)),
        "key_shape": torch.Size((1, 3175, 8, 128)),
        "value_shape": torch.Size((1, 3175, 8, 128)),
        "dtype": torch.bfloat16,
        "device": torch.device("cuda"),
        "requires_grad": False,
        "is_causal": False,
        "causal_type": None,
        "is_varlen": False,
    }
    args.update(overrides)
    return checks.sage_attention_check(**args)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _mock_sm89(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "SAGE_SUPPORTED", True)
    monkeypatch.setattr(checks, "get_arch_tag", lambda _device: 89)


@pytest.mark.L0
def test_sage_policy_accepts_long_dense_sm89_inference() -> None:
    assert _supported()


@pytest.mark.L0
@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("requires_grad", True),
        ("is_causal", True),
        ("is_varlen", True),
        ("query_shape", torch.Size((1, 82, 16, 128))),
    ],
)
def test_sage_policy_rejects_training_causal_varlen_and_short_attention(override: str, value: object) -> None:
    assert not _supported(**{override: value})


@pytest.mark.L0
def test_sage_policy_rejects_non_sm89(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checks, "get_arch_tag", lambda _device: 90)
    assert not _supported()


@pytest.mark.L0
def test_backend_selector_skips_sage_when_call_requires_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    backends.choose_backend.cache_clear()
    monkeypatch.setattr(backends, "get_arch_tag", lambda _device: 89)
    monkeypatch.setattr(backends, "get_backend_list", lambda _arch: ["sage", "flash2"])
    monkeypatch.setattr(backends, "is_backend_compatible", lambda **_kwargs: True)

    selected = backends.choose_backend(
        query_shape=torch.Size((1, 3093, 16, 128)),
        key_shape=torch.Size((1, 3175, 8, 128)),
        value_shape=torch.Size((1, 3175, 8, 128)),
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        requires_grad=False,
        is_causal=False,
        causal_type=None,
        is_varlen=False,
        excluded_backends=("sage",),
    )

    assert selected == "flash2"
    backends.choose_backend.cache_clear()
