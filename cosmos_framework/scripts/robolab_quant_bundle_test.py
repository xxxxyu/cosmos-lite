# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest
import torch

from cosmos_framework.scripts import robolab_quant_bundle as bundle_module
from cosmos_framework.scripts.robolab_quant_bundle import (
    ROBOLAB_BUNDLE_ARTIFACT_TYPE,
    ROBOLAB_BUNDLE_ROOT_TOKEN,
    _manifest_source,
    _quant_module_name,
    _read_weight_map,
    _strategy_backend,
    discover_quant_targets,
    materialize_robolab_bundle_config,
    validate_robolab_quant_bundle,
)


def test_weight_map_merges_transformer_component_without_overriding_root(tmp_path: Path) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (tmp_path / "root.safetensors").write_bytes(b"")
    (transformer_dir / "component.safetensors").write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "shared": "root.safetensors",
                    "layers.0.self_attn.k_norm_und_for_gen.weight": "root.safetensors",
                }
            }
        )
    )
    (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"shared": "ignored.safetensors", "generation": "component.safetensors"}})
    )

    assert _read_weight_map(tmp_path) == {
        "shared": "root.safetensors",
        "generation": "transformer/component.safetensors",
    }


def test_quant_module_name_maps_diffusers_mot_branches() -> None:
    assert (
        _quant_module_name("layers.3.self_attn.add_q_proj.weight")
        == "net.language_model.model.layers.3.self_attn.q_proj_moe_gen"
    )
    assert (
        _quant_module_name("layers.3.mlp_moe_gen.down_proj.weight")
        == "net.language_model.model.layers.3.mlp_moe_gen.down_proj"
    )


def test_edge_target_discovery_fails_closed_on_missing_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model": {"config": {"vlm_config": {"model_name": "Cosmos3-Edge"}}}})
    )
    monkeypatch.setattr(
        bundle_module,
        "_read_weight_map",
        lambda _root: {"layers.0.self_attn.to_q.weight": "transformer/model.safetensors"},
    )

    with pytest.raises(ValueError, match="Expected 336 cosmos3_edge"):
        discover_quant_targets(tmp_path, "full_w8")


def _strategy_keys(*, layers: int, include_gates: bool) -> list[str]:
    keys: list[str] = []
    for layer in range(layers):
        keys.extend(
            f"layers.{layer}.self_attn.{name}.weight"
            for name in (
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
            )
        )
        keys.extend(
            f"layers.{layer}.{branch}.{name}.weight"
            for branch in ("mlp", "mlp_moe_gen")
            for name in (("gate_proj", "up_proj", "down_proj") if include_gates else ("up_proj", "down_proj"))
        )
    return keys


def test_strategy_precision_maps_match_nano_release_counts() -> None:
    keys = _strategy_keys(layers=36, include_gates=True)

    expected = {
        "full_w8": (0, 504),
        "full_w4": (504, 0),
        "attention_w8": (216, 288),
        "gen_branch_w8": (252, 252),
        "gen_branch_w8a8": (252, 252),
        "full_w8a8": (0, 504),
    }
    for strategy, counts in expected.items():
        bits = [_strategy_backend(strategy, key)[1] for key in keys]  # type: ignore[arg-type, union-attr]
        assert (bits.count(4), bits.count(8)) == counts
    assert sum(_strategy_backend("gen_branch_w8a8", key)[0] == "VllmCutlassFp8W8A8Linear" for key in keys) == 252
    assert sum(_strategy_backend("full_w8a8", key)[0] == "VllmCutlassFp8W8A8Linear" for key in keys) == 504


def test_strategy_precision_maps_match_edge_release_counts() -> None:
    keys = _strategy_keys(layers=28, include_gates=False)
    expected = {
        "full_w8": (0, 336),
        "full_w4": (336, 0),
        "attention_w8": (112, 224),
        "gen_branch_w8": (168, 168),
        "gen_branch_w8a8": (168, 168),
        "full_w8a8": (0, 336),
    }
    for strategy, counts in expected.items():
        bits = [_strategy_backend(strategy, key)[1] for key in keys]  # type: ignore[arg-type, union-attr]
        assert (bits.count(4), bits.count(8)) == counts
    assert sum(_strategy_backend("gen_branch_w8a8", key)[0] == "VllmCutlassFp8W8A8Linear" for key in keys) == 168
    assert sum(_strategy_backend("full_w8a8", key)[0] == "VllmCutlassFp8W8A8Linear" for key in keys) == 336


def test_public_edge_manifest_uses_revision_uris_without_local_paths(tmp_path: Path) -> None:
    source = _manifest_source(
        source_root=tmp_path / "checkpoint",
        processor_source=tmp_path / "processor",
        vae_source=tmp_path / "Wan2.2_VAE.pth",
        model_family="cosmos3_edge",
        provenance={
            "repositories": {"droid": "nvidia/edge", "wan": "Wan-AI/wan"},
            "resolved_revisions": {"droid": "edge-sha", "wan": "wan-sha"},
        },
    )

    assert source["checkpoint_path"] == "hf://nvidia/edge@edge-sha"
    assert source["tokenizer_dir"] == source["checkpoint_path"]
    assert source["vae_path"] == "hf://Wan-AI/wan@wan-sha/Wan2.2_VAE.pth"
    assert str(tmp_path) not in json.dumps(source)


def test_local_manifest_records_names_without_build_machine_paths(tmp_path: Path) -> None:
    source = _manifest_source(
        source_root=tmp_path / "Cosmos3-Edge-Policy-DROID",
        processor_source=tmp_path / "processor",
        vae_source=tmp_path / "Wan2.2_VAE.pth",
        model_family="cosmos3_edge",
        provenance=None,
    )

    assert source == {
        "checkpoint_path": "Cosmos3-Edge-Policy-DROID",
        "tokenizer_dir": "processor",
        "vae_path": "Wan2.2_VAE.pth",
        "provenance": {},
    }
    assert str(tmp_path) not in json.dumps(source)


def _minimal_bundle(root: Path, *, layers: int = 36, linears_per_layer: int = 14) -> None:
    (root / "tensors").mkdir(parents=True)
    (root / "runtime").mkdir()
    (root / "assets/qwen3_vl_tokenizer").mkdir(parents=True)
    (root / "tensors/shared.pt").write_bytes(b"packed")
    (root / "model-00001.safetensors").write_bytes(b"residual")
    (root / "assets/Wan2.2_VAE.pth").write_bytes(b"vae")
    (root / "assets/qwen3_vl_tokenizer/tokenizer.json").write_text("{}")
    (root / "runtime/config.json").write_text(
        json.dumps({"asset": f"{ROBOLAB_BUNDLE_ROOT_TOKEN}/assets/Wan2.2_VAE.pth"})
    )
    (root / "config.json").write_text("{}")
    residual_index = {
        "metadata": {"total_size": 1},
        "weight_map": {"net.norm.weight": "model-00001.safetensors"},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(residual_index))

    modules = []
    for layer in range(layers):
        for index in range(linears_per_layer):
            modules.append(
                {
                    "name": f"net.language_model.model.layers.{layer}.test_linear_{index}",
                    "backend_class": "VllmGptqMarlinW8A16Linear",
                    "format": "vllm_marlin_wna16",
                    "num_bits": 8,
                    "group_size": -1,
                    "size_k": 128,
                    "size_n": 64,
                    "wtype_id": 1,
                    "tensor_file": "tensors/shared.pt",
                }
            )
    runtime_files = [
        "tensors/shared.pt",
        "model-00001.safetensors",
        "assets/Wan2.2_VAE.pth",
        "assets/qwen3_vl_tokenizer/tokenizer.json",
        "runtime/config.json",
        "config.json",
        "model.safetensors.index.json",
    ]
    files = {rel: {"size": (root / rel).stat().st_size, "sha256": "not-checked"} for rel in runtime_files}
    manifest = {
        "schema_version": 1,
        "artifact_type": ROBOLAB_BUNDLE_ARTIFACT_TYPE,
        "self_contained": True,
        "quantization": {
            "strategy": "full_w8",
            "weight_only": True,
            "activation_quantization": False,
        },
        "runtime": {
            "config_file": "runtime/config.json",
            "tokenizer_dir": "assets/qwen3_vl_tokenizer",
            "vae_path": "assets/Wan2.2_VAE.pth",
            "residual_index": "model.safetensors.index.json",
        },
        "modules": modules,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_bundle_validation_and_runtime_materialization(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _minimal_bundle(bundle)

    result = validate_robolab_quant_bundle(bundle, expected_strategy="full_w8")

    assert result["modules"] == 504
    assert result["residual_state_keys"] == 1
    materialized = materialize_robolab_bundle_config(bundle, tmp_path / "runtime")
    text = materialized.read_text()
    assert ROBOLAB_BUNDLE_ROOT_TOKEN not in text
    assert str(bundle.resolve()) in text


def test_fp8_bundle_validation_checks_activation_metadata_and_payload(tmp_path: Path) -> None:
    bundle = tmp_path / "fp8-bundle"
    _minimal_bundle(bundle)
    torch.save(
        {
            "qweight_nk": torch.zeros(64, 128).to(torch.float8_e4m3fn),
            "scale_b": torch.ones(1, 64, dtype=torch.float32),
            "input_scale": None,
        },
        bundle / "tensors/shared.pt",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["quantization"].update(
        {
            "strategy": "gen_branch_w8a8",
            "weight_only": False,
            "activation_quantization": True,
            "activation_dtype": "fp8_e4m3fn",
            "w8a8_modules": 504,
            "w8a16_modules": 0,
        }
    )
    for entry in manifest["modules"]:
        entry.update(
            {
                "backend_class": "VllmCutlassFp8W8A8Linear",
                "format": "vllm_cutlass_fp8_w8a8",
                "activation_bits": 8,
            }
        )
        entry.pop("group_size")
        entry.pop("wtype_id")
    manifest["files"]["tensors/shared.pt"]["size"] = (bundle / "tensors/shared.pt").stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    result = validate_robolab_quant_bundle(bundle, expected_strategy="gen_branch_w8a8", check_tensors=True)
    assert result["modules"] == 504

    manifest["quantization"].pop("weight_only")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="weight_only"):
        validate_robolab_quant_bundle(bundle)


def test_edge_bundle_uses_manifest_module_count_and_local_vision_tower(tmp_path: Path) -> None:
    bundle = tmp_path / "edge-bundle"
    _minimal_bundle(bundle, layers=28, linears_per_layer=12)
    (bundle / "vision_encoder").mkdir()
    (bundle / "vision_encoder/model.safetensors").write_bytes(b"vision")

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_family"] = "cosmos3_edge"
    manifest["model"] = {
        "quant_module_count": 336,
        "quant_module_prefix": "net.language_model.model.layers.",
    }
    manifest["runtime"]["vision_encoder_dir"] = "vision_encoder"
    manifest_path.write_text(json.dumps(manifest))

    result = validate_robolab_quant_bundle(bundle)

    assert result["modules"] == 336
