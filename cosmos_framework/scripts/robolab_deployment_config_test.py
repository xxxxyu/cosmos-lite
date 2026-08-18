# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.scripts.robolab_deployment_config import (
    DeploymentRuntimeConfig,
    RuntimeProbe,
    apply_cli_overrides,
    configure_backend_environment,
    load_deployment_config,
    resolve_deployment_config,
    server_argument_values,
    write_resolved_deployment_record,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "examples/robolab_quant/configs"


def _write_bundle(root: Path, *, family: str, strategy: str) -> Path:
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"model_family": family, "quantization": {"strategy": strategy}}) + "\n",
        encoding="utf-8",
    )
    return root


def _probe(
    capability: tuple[int, int] = (8, 9),
    *,
    sage: bool = True,
    triton: bool = True,
) -> RuntimeProbe:
    return RuntimeProbe(
        cuda_available=True,
        gpu_name="test GPU",
        compute_capability=capability,
        torch_version="test",
        cuda_version="test",
        sageattention_installed=sage,
        sageattention_sm89=sage,
        triton_installed=triton,
    )


@pytest.mark.parametrize(
    ("filename", "family", "strategy"),
    [
        ("nano_w8.yaml", "cosmos3_nano", "full_w8"),
        ("nano_genw8a8_fast_4090.yaml", "cosmos3_nano", "gen_branch_w8a8"),
        ("edge_w8.yaml", "cosmos3_edge", "full_w8"),
        ("edge_genw8a8_fast_4090.yaml", "cosmos3_edge", "gen_branch_w8a8"),
    ],
)
def test_public_deployment_presets_are_valid(filename: str, family: str, strategy: str) -> None:
    config = load_deployment_config(_CONFIG_DIR / filename)

    assert config.model.family == family
    assert config.model.strategy == strategy
    assert config.model.bundle_dir is None
    assert config.runtime.backend_policy == "strict"
    assert config.sampling.guidance == 3.0
    assert config.sampling.denoise_steps == 2


def test_runtime_rejects_triton_without_compile() -> None:
    with pytest.raises(ValueError, match="triton_sm89 requires torch_compile"):
        DeploymentRuntimeConfig(fp8_gemm_backend="triton_sm89")


def test_resolve_w8_profile_checks_manifest_and_builds_server_args(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family="cosmos3_nano", strategy="full_w8")
    config = apply_cli_overrides(
        load_deployment_config(_CONFIG_DIR / "nano_w8.yaml"),
        bundle_dir=bundle,
        output_dir=tmp_path / "run",
        port=9000,
    )

    resolution = resolve_deployment_config(config, probe=_probe(sage=False))
    args = server_argument_values(resolution.effective)

    assert resolution.fallback_decisions == []
    assert args["quant_import_dir"] == str(bundle.resolve())
    assert args["port"] == 9000
    assert args["profile_jsonl"] == (tmp_path / "run/profile.jsonl").resolve()
    assert args["format_prompt_as_json"] is False


def test_edge_public_presets_use_json_prompts() -> None:
    for filename in ("edge_w8.yaml", "edge_genw8a8_fast_4090.yaml"):
        config = load_deployment_config(_CONFIG_DIR / filename)

        assert config.server.format_prompt_as_json is True


def test_strict_sage_profile_fails_when_optional_backend_is_missing(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family="cosmos3_nano", strategy="gen_branch_w8a8")
    config = apply_cli_overrides(load_deployment_config(_CONFIG_DIR / "nano_genw8a8_fast_4090.yaml"), bundle_dir=bundle)

    with pytest.raises(RuntimeError, match="SageAttention is not installed"):
        resolve_deployment_config(config, probe=_probe(sage=False))


def test_best_available_profile_records_sage_and_triton_fallbacks(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family="cosmos3_edge", strategy="gen_branch_w8a8")
    config = apply_cli_overrides(load_deployment_config(_CONFIG_DIR / "edge_genw8a8_fast_4090.yaml"), bundle_dir=bundle)
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend_policy": "best_available"})}
    )

    resolution = resolve_deployment_config(config, probe=_probe((9, 0), sage=False, triton=False))

    assert resolution.effective.runtime.fp8_gemm_backend == "cutlass"
    assert resolution.effective.runtime.attention_backend == "flash_attention_2"
    assert len(resolution.fallback_decisions) == 2


def test_genw8a8_never_falls_back_on_gpu_without_native_fp8(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family="cosmos3_nano", strategy="gen_branch_w8a8")
    config = apply_cli_overrides(load_deployment_config(_CONFIG_DIR / "nano_genw8a8_fast_4090.yaml"), bundle_dir=bundle)
    config = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"backend_policy": "best_available"})}
    )

    with pytest.raises(RuntimeError, match="native FP8 Tensor Cores"):
        resolve_deployment_config(config, probe=_probe((8, 6), sage=False))


def test_configure_backend_environment_clears_stale_sage_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COSMOS3_SAGE_ATTENTION", "1")
    monkeypatch.setenv("COSMOS3_SAGE_PV", "fp16_fp32")
    config = load_deployment_config(_CONFIG_DIR / "nano_w8.yaml")

    configure_backend_environment(config)

    assert "COSMOS3_SAGE_ATTENTION" not in __import__("os").environ
    assert "COSMOS3_SAGE_PV" not in __import__("os").environ


def test_write_resolved_record_contains_effective_runtime_and_bundle_hash(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family="cosmos3_nano", strategy="full_w8")
    config_path = _CONFIG_DIR / "nano_w8.yaml"
    config = apply_cli_overrides(load_deployment_config(config_path), bundle_dir=bundle, output_dir=tmp_path / "run")
    resolution = resolve_deployment_config(config, probe=_probe(sage=False))

    output = write_resolved_deployment_record(resolution, config_file=config_path)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["effective"]["profile"] == "nano_w8"
    assert record["bundle"]["path"] == str(bundle.resolve())
    assert len(record["bundle"]["manifest_sha256"]) == 64
    assert record["runtime_probe"]["compute_capability"] == [8, 9]
