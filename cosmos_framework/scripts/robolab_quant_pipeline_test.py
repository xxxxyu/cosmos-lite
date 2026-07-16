# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cosmos_framework.scripts.robolab_quant_pipeline import _DROID_REVISION, _prepare_public_sources

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_prepare_public_sources_records_resolved_revisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        local_dir = Path(str(kwargs["local_dir"]))
        local_dir.mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    def hf_hub_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"vae")
        return str(path)

    class FakeHfApi:
        def model_info(self, repo_id: str, revision: str) -> SimpleNamespace:
            return SimpleNamespace(sha=f"resolved-{repo_id}-{revision}")

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = FakeHfApi
    fake_hub.hf_hub_download = hf_hub_download
    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = _prepare_public_sources(
        tmp_path,
        droid_revision=_DROID_REVISION,
        qwen_revision="qwen-rev",
        wan_revision="wan-rev",
    )

    assert len(calls) == 3
    assert result["requested_revisions"]["droid"] == _DROID_REVISION
    assert result["resolved_revisions"]["qwen"].endswith("-qwen-rev")
    assert Path(result["vae_path"]).read_bytes() == b"vae"
    assert json.loads((tmp_path / "sources.json").read_text()) == result
