# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

from cosmos_framework.scripts.robolab_quant_bundle import (
    ROBOLAB_BUNDLE_ARTIFACT_TYPE,
    ROBOLAB_BUNDLE_ROOT_TOKEN,
    _strategy_backend,
    materialize_robolab_bundle_config,
    validate_robolab_quant_bundle,
)


def test_strategy_precision_maps_match_release_counts() -> None:
    keys: list[str] = []
    for layer in range(36):
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
            for name in ("gate_proj", "up_proj", "down_proj")
        )

    expected = {
        "full_w8": (0, 504),
        "full_w4": (504, 0),
        "attention_w8": (216, 288),
        "gen_branch_w8": (252, 252),
    }
    for strategy, counts in expected.items():
        bits = [_strategy_backend(strategy, key)[1] for key in keys]  # type: ignore[arg-type, union-attr]
        assert (bits.count(4), bits.count(8)) == counts


def _minimal_bundle(root: Path) -> None:
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
    for layer in range(36):
        for index in range(14):
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
    files = {
        rel: {"size": (root / rel).stat().st_size, "sha256": "not-checked"} for rel in runtime_files
    }
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
