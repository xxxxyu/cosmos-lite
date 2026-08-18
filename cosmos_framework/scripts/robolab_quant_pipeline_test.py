# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cosmos_framework.scripts.robolab_quant_pipeline import (
    _DROID_REVISION,
    _EDGE_DROID_REPOSITORY,
    _EDGE_DROID_REVISION,
    _load_source_provenance,
    _parser,
    _prepare_public_sources,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_public_builder_exposes_only_release_strategies() -> None:
    for strategy in ("full_w8", "gen_branch_w8a8"):
        args = _parser().parse_args(
            [
                "build-public",
                "--asset-dir",
                "/tmp/assets",
                "--output-dir",
                "/tmp/output",
                "--strategy",
                strategy,
            ]
        )
        assert args.strategy == strategy

    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "build-public",
                "--asset-dir",
                "/tmp/assets",
                "--output-dir",
                "/tmp/output",
                "--strategy",
                "full_w4",
            ]
        )


def test_load_source_provenance_requires_release_mappings(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    value = {
        "repositories": {"droid": "nvidia/policy", "wan": "Wan-AI/vae"},
        "requested_revisions": {"droid": "main", "wan": "main"},
        "resolved_revisions": {"droid": "droid-sha", "wan": "wan-sha"},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    assert _load_source_provenance(path) == value

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        _load_source_provenance(path)


def test_prepare_public_sources_records_resolved_revisions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_prepare_edge_public_sources_reuses_bundled_processor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        model_family="cosmos3_edge",
        droid_revision=_EDGE_DROID_REVISION,
        qwen_revision="unused-qwen-rev",
        wan_revision="wan-rev",
    )

    assert len(calls) == 2
    assert calls[0]["repo_id"] == _EDGE_DROID_REPOSITORY
    assert "qwen" not in result["repositories"]
    assert "qwen" not in result["requested_revisions"]
    assert "qwen" not in result["resolved_revisions"]
    assert result["checkpoint_dir"] == result["tokenizer_dir"]
