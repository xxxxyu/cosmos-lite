# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for public-facing RoboCasa365 policy server security defaults."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

with patch("cosmos_framework.inference.common.init._init_script", lambda **kwargs: None):
    module_name = "cosmos_framework.scripts.action_policy_server_robocasa365_quant"
    sys.modules.pop(module_name, None)
    from cosmos_framework.scripts import action_policy_server_robocasa365_quant as robocasa_server  # noqa: E402

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.12.3.4", "::1", "[::1]"])
def test_loopback_host_detection_accepts_only_loopback_addresses(host: str) -> None:
    assert robocasa_server._is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "*", "192.0.2.1", "policy.internal"])
def test_loopback_host_detection_rejects_exposed_addresses(host: str) -> None:
    assert not robocasa_server._is_loopback_host(host)
