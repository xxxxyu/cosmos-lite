# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.scripts.robocasa365_quant_bundle import (
    BUNDLE_ROOT_TOKEN,
    build_self_contained_bundle,
    materialize_bundle_config,
    validate_self_contained_bundle,
)
from cosmos_framework.scripts.robocasa365_quant_pipeline import (
    STRATEGIES,
    _calibration_input_scale,
    _is_quantizable_linear,
    serve_command,
    validate_quant_artifact,
    write_strategy_configs,
)


@pytest.mark.parametrize(
    "name",
    [
        "net.language_model.model.layers.0.self_attn.q_proj",
        "net.language_model.model.layers.35.self_attn.o_proj_moe_gen",
        "net.language_model.model.layers.4.mlp.down_proj",
        "net.language_model.model.layers.9.mlp_moe_gen.gate_proj",
    ],
)
def test_stream_export_linear_filter_accepts_mot_linears(name: str) -> None:
    assert _is_quantizable_linear(name)


@pytest.mark.parametrize(
    "name",
    [
        "net.language_model.model.layers.0.input_layernorm",
        "net.language_model.model.layers.0.self_attn.q_norm_moe_gen",
        "net.language_model.model.embed_tokens",
        "net.time_embedder.mlp.0",
    ],
)
def test_stream_export_linear_filter_rejects_non_targets(name: str) -> None:
    assert not _is_quantizable_linear(name)


def test_calibration_scale_rejects_wrong_channel_count() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="expected 4"):
        _calibration_input_scale(torch.ones(3), size_k=4, alpha=0.5)


def _write_minimal_artifact(root: Path, *, backend_class: str = "VllmGptqMarlinW4A16Linear") -> None:
    (root / "tensors").mkdir(parents=True)
    tensor_file = "tensors/net.language_model.model.layers.0.self_attn.q_proj.pt"
    (root / tensor_file).write_bytes(b"placeholder")
    num_bits = 4 if backend_class.endswith("W4A16Linear") else 8
    manifest = {
        "schema_version": 1,
        "checkpoint_path": "/ckpt",
        "config_file": "/config.yaml",
        "modules": [
            {
                "name": "net.language_model.model.layers.0.self_attn.q_proj",
                "backend_class": backend_class,
                "format": "vllm_marlin_wna16",
                "num_bits": num_bits,
                "group_size": 128,
                "size_k": 4096,
                "size_n": 4096,
                "tensor_file": tensor_file,
                "wtype_id": 1,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_write_strategy_configs_contains_retained_candidates(tmp_path: Path) -> None:
    write_strategy_configs(tmp_path)

    manifest = json.loads((tmp_path / "strategies_manifest.json").read_text())
    assert set(manifest) == set(STRATEGIES)
    assert json.loads((tmp_path / "attention_w8.json").read_text())[1]["backend"] == "vllm_gptq_marlin_w8a16"


def test_validate_quant_artifact_accepts_minimal_manifest(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)

    result = validate_quant_artifact(tmp_path)

    assert result["modules"] == 1
    assert result["counts"] == {"VllmGptqMarlinW4A16Linear": 1}


def test_validate_quant_artifact_can_require_self_contained(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)

    with pytest.raises(ValueError, match="self-contained schema-v2"):
        validate_quant_artifact(tmp_path, require_self_contained=True)


def test_validate_quant_artifact_rejects_wrong_bits(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["num_bits"] = 8
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="num_bits"):
        validate_quant_artifact(tmp_path)


def test_validate_quant_artifact_rejects_strategy_mismatch(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)

    with pytest.raises(ValueError, match="expects"):
        validate_quant_artifact(tmp_path, expected_strategy="full_w8")


def test_validate_quant_artifact_rejects_escaping_tensor_file(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["tensor_file"] = "../outside.pt"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact root"):
        validate_quant_artifact(tmp_path)


def test_self_contained_bundle_survives_source_removal(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    dcp = pytest.importorskip("torch.distributed.checkpoint")
    safetensors = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    checkpoint = source / "checkpoint"
    checkpoint.mkdir(parents=True)
    source_state = {
        "net.language_model.model.layers.0.self_attn.q_proj.weight": torch.arange(
            16, dtype=torch.bfloat16
        ).reshape(4, 4),
        "net.norm.weight": torch.arange(4, dtype=torch.bfloat16),
        "net_ema.language_model.model.layers.0.self_attn.q_proj.weight": torch.full(
            (4, 4), 9, dtype=torch.bfloat16
        ),
    }
    dcp.save(source_state, checkpoint_id=checkpoint)

    quant = source / "quant"
    _write_minimal_artifact(quant)
    (quant / "cosmos3_quant_metadata.json").write_text(
        json.dumps({"strategy": "full_w4", "weight_only": True, "activation_quant": False})
    )

    tokenizer = source / "Qwen3-VL-8B-Instruct"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    vae = source / "Wan2.2_VAE.pth"
    vae.write_bytes(b"vae")
    config = source / "config.yaml"
    config.write_text(
        "checkpoint:\n"
        f"  load_path: {checkpoint}\n"
        "runtime:\n"
        f"  tokenizer: {tokenizer}\n"
        f"  vae: {vae}\n"
    )

    bundle = tmp_path / "bundle"
    result = build_self_contained_bundle(
        strategy="full_w4",
        strategy_plan=STRATEGIES["full_w4"].plan,
        quant_artifact_dir=quant,
        checkpoint_path=checkpoint,
        config_file=config,
        tokenizer_dir=tokenizer,
        vae_path=vae,
        output_dir=bundle,
        max_shard_size_bytes=1024,
    )
    assert result["state_keys"] == 1
    assert validate_quant_artifact(bundle, expected_strategy="full_w4")["self_contained"] is True
    validate_self_contained_bundle(bundle, check_hashes=True)

    moved_source = tmp_path / "source-removed"
    source.rename(moved_source)
    runtime_config = materialize_bundle_config(bundle, tmp_path / "runtime")
    config_text = runtime_config.read_text()
    assert BUNDLE_ROOT_TOKEN not in config_text
    assert str(bundle) in config_text
    assert str(moved_source) not in config_text

    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    shard = bundle / index["weight_map"]["net.norm.weight"]
    residual = safetensors.load_file(shard)
    torch.testing.assert_close(residual["net.norm.weight"], source_state["net.norm.weight"])

    restored = {"net.norm.weight": torch.empty_like(source_state["net.norm.weight"])}
    reader = torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader(str(bundle))
    dcp.load(state_dict=restored, storage_reader=reader)
    torch.testing.assert_close(restored["net.norm.weight"], source_state["net.norm.weight"])

    serve_args = SimpleNamespace(
        python="python",
        checkpoint_path="",
        config_file="",
        output_dir="/tmp/server",
        host="127.0.0.1",
        port=5577,
        served_action_steps=8,
        quant_import_dir=str(bundle),
    )
    command = serve_command(serve_args)
    assert "--quant-import-dir" in command
    assert "--checkpoint-path" not in command
    assert "--config-file" not in command
