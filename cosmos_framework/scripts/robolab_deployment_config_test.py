# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
from pathlib import Path

import pydantic
import pytest

from cosmos_framework.scripts.robolab_deployment_config import (
    DeploymentModelConfig,
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


def _write_bundle(root: Path, *, family: str | None, strategy: str, checkpoint_path: str | None = None) -> Path:
    root.mkdir()
    manifest = {"model_family": family, "quantization": {"strategy": strategy}}
    if checkpoint_path is not None:
        manifest["source"] = {"checkpoint_path": checkpoint_path}
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
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


def test_edge_bf16_preset_separates_checkpoint_from_sampling() -> None:
    config = load_deployment_config(_CONFIG_DIR / "edge_bf16.yaml")

    assert config.model.artifact == "bf16_checkpoint"
    assert config.model.checkpoint_path == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert config.model.strategy is None
    assert config.sampling.denoise_steps == 2
    assert config.server.format_prompt_as_json is True


def test_bf16_sampling_overrides_do_not_change_model_artifact() -> None:
    config = apply_cli_overrides(
        load_deployment_config(_CONFIG_DIR / "edge_bf16.yaml"),
        denoise_steps=4,
        guidance=3.0,
    )

    assert config.model.artifact == "bf16_checkpoint"
    assert config.model.checkpoint_path == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert config.sampling.denoise_steps == 4


@pytest.mark.parametrize(
    "model",
    [
        {
            "artifact": "bf16_checkpoint",
            "family": "cosmos3_edge",
            "checkpoint_path": "nvidia/Cosmos3-Edge-Policy-DROID",
            "strategy": "full_w8",
        },
        {
            "artifact": "quantized_bundle",
            "family": "cosmos3_edge",
            "checkpoint_path": "nvidia/Cosmos3-Edge-Policy-DROID",
            "strategy": "full_w8",
        },
    ],
)
def test_model_artifact_rejects_incompatible_quantization_fields(model: dict[str, str]) -> None:
    with pytest.raises(pydantic.ValidationError):
        DeploymentModelConfig.model_validate(model)


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


def test_resolve_legacy_nano_w8_bundle_infers_family_from_pinned_source(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path / "bundle",
        family=None,
        strategy="full_w8",
        checkpoint_path=(
            "hf://nvidia/Cosmos3-Nano-Policy-DROID@6706d7680581c255ff61e0f3bb49d90eac55c79e"
        ),
    )
    config = apply_cli_overrides(load_deployment_config(_CONFIG_DIR / "nano_w8.yaml"), bundle_dir=bundle)

    resolution = resolve_deployment_config(config, probe=_probe(sage=False))

    assert resolution.effective.model.family == "cosmos3_nano"


def test_resolve_bundle_rejects_missing_and_unrecognized_family(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle", family=None, strategy="full_w8")
    config = apply_cli_overrides(load_deployment_config(_CONFIG_DIR / "nano_w8.yaml"), bundle_dir=bundle)

    with pytest.raises(ValueError, match="bundle contains None"):
        resolve_deployment_config(config, probe=_probe(sage=False))


def test_resolve_bf16_profile_builds_checkpoint_server_args() -> None:
    config = load_deployment_config(_CONFIG_DIR / "edge_bf16.yaml")
    resolution = resolve_deployment_config(config, probe=_probe(sage=False))
    args = server_argument_values(resolution.effective)

    assert resolution.bundle_manifest == {}
    assert resolution.bundle_manifest_sha256 is None
    assert args["checkpoint_path"] == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert args["quant_import_dir"] is None
    assert args["num_steps"] == 2


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
    monkeypatch.setenv("COSMOS_TRAINING", "0")
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


def test_write_resolved_bf16_record_separates_model_and_sampler(tmp_path: Path) -> None:
    config_path = _CONFIG_DIR / "edge_bf16.yaml"
    config = apply_cli_overrides(
        load_deployment_config(config_path),
        output_dir=tmp_path / "run",
        denoise_steps=4,
    )
    resolution = resolve_deployment_config(config, probe=_probe(sage=False))

    output = write_resolved_deployment_record(resolution, config_file=config_path)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert record["model"]["artifact"] == "bf16_checkpoint"
    assert record["model"]["checkpoint_path"] == "nvidia/Cosmos3-Edge-Policy-DROID"
    assert record["effective"]["sampling"]["denoise_steps"] == 4
    assert record["bundle"] is None
