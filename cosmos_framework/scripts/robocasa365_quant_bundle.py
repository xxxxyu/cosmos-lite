# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Portable, self-contained artifacts for quantized RoboCasa365 policies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_ARTIFACT_TYPE = "cosmos3_robocasa365_quantized_policy"
BUNDLE_ROOT_TOKEN = "__COSMOS3_QUANT_BUNDLE_ROOT__"


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"quant bundle field {field!r} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"quant bundle field {field!r} must stay under the bundle root: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path, *, copy_mode: str) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        os.link(source, target)
    elif copy_mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported copy_mode={copy_mode!r}; expected 'copy' or 'hardlink'")
    return {"size": target.stat().st_size, "sha256": _sha256(target)}


def _copy_tree(
    source: Path,
    target: Path,
    *,
    copy_mode: str,
    bundle_root: Path,
    files: dict[str, dict[str, Any]],
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required quant bundle directory does not exist: {source}")
    for source_path in sorted(source.rglob("*")):
        if source_path.is_dir():
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Quant bundle source contains a broken link: {source_path}")
        target_path = target / source_path.relative_to(source)
        files[target_path.relative_to(bundle_root).as_posix()] = _copy_file(
            source_path.resolve(), target_path, copy_mode=copy_mode
        )


def _replace_exact_strings(value: Any, replacements: dict[str, str]) -> tuple[Any, int]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        count = 0
        for key, child in value.items():
            output[key], child_count = _replace_exact_strings(child, replacements)
            count += child_count
        return output, count
    if isinstance(value, list):
        output_list: list[Any] = []
        count = 0
        for child in value:
            replaced, child_count = _replace_exact_strings(child, replacements)
            output_list.append(replaced)
            count += child_count
        return output_list, count
    if isinstance(value, str) and value in replacements:
        return replacements[value], 1
    return value, 0


def _portable_runtime_config(
    source: Path,
    target: Path,
    *,
    checkpoint_path: Path,
    tokenizer_dir: Path,
    vae_path: Path,
) -> int:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(source), resolve=False)
    checkpoint_dir = checkpoint_path.parent if checkpoint_path.name == "model" else checkpoint_path
    replacements = {
        str(tokenizer_dir): f"{BUNDLE_ROOT_TOKEN}/assets/qwen3_vl_tokenizer",
        str(tokenizer_dir.resolve()): f"{BUNDLE_ROOT_TOKEN}/assets/qwen3_vl_tokenizer",
        str(vae_path): f"{BUNDLE_ROOT_TOKEN}/assets/Wan2.2_VAE.pth",
        str(vae_path.resolve()): f"{BUNDLE_ROOT_TOKEN}/assets/Wan2.2_VAE.pth",
        str(checkpoint_path): BUNDLE_ROOT_TOKEN,
        str(checkpoint_path.resolve()): BUNDLE_ROOT_TOKEN,
        str(checkpoint_dir): BUNDLE_ROOT_TOKEN,
        str(checkpoint_dir.resolve()): BUNDLE_ROOT_TOKEN,
    }
    portable, replacement_count = _replace_exact_strings(config, replacements)
    serialized = OmegaConf.to_yaml(OmegaConf.create(portable), resolve=False)
    if str(tokenizer_dir) in serialized or str(vae_path) in serialized:
        raise ValueError("Failed to remove source tokenizer/VAE paths from portable runtime config")
    if f"{BUNDLE_ROOT_TOKEN}/assets/qwen3_vl_tokenizer" not in serialized:
        raise ValueError("Runtime config does not reference the bundled Qwen tokenizer")
    if f"{BUNDLE_ROOT_TOKEN}/assets/Wan2.2_VAE.pth" not in serialized:
        raise ValueError("Runtime config does not reference the bundled Wan VAE")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized)
    return replacement_count


def _tensor_nbytes(metadata: Any) -> int:
    numel = 1
    for dim in metadata.size:
        numel *= int(dim)
    return numel * metadata.properties.dtype.itemsize


def _residual_state_metadata(checkpoint_path: Path, modules: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(str(checkpoint_path))
    checkpoint_metadata = reader.read_metadata()
    net_metadata = {
        key: value
        for key, value in checkpoint_metadata.state_dict_metadata.items()
        if key.startswith("net.") and isinstance(value, TensorStorageMetadata)
    }
    quant_weight_keys = {f"{entry['name']}.weight" for entry in modules}
    missing = sorted(quant_weight_keys - set(net_metadata))
    if missing:
        raise KeyError(f"Quant artifact modules are missing from source checkpoint: {missing[:8]}")
    residual = {key: value for key, value in net_metadata.items() if key not in quant_weight_keys}
    if not residual:
        raise ValueError("Quant bundle export selected no residual model tensors")
    return reader, residual


def _write_residual_safetensors(
    checkpoint_path: Path,
    modules: list[dict[str, Any]],
    output_root: Path,
    *,
    max_shard_size_bytes: int,
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    import torch
    import torch.distributed.checkpoint as dcp
    from safetensors.torch import save_file

    reader, residual = _residual_state_metadata(checkpoint_path, modules)
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in sorted(residual):
        size = _tensor_nbytes(residual[key])
        if current and current_size + size > max_shard_size_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size
    if current:
        groups.append(current)

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard_index, keys in enumerate(groups, start=1):
        tensors = {
            key: torch.empty(tuple(residual[key].size), dtype=residual[key].properties.dtype, device="cpu")
            for key in keys
        }
        dcp.load(state_dict=tensors, storage_reader=reader)
        shard_name = f"model-{shard_index:05d}-of-{len(groups):05d}.safetensors"
        shard_path = output_root / shard_name
        save_file(tensors, str(shard_path), metadata={"format": "pt"})
        for key in keys:
            weight_map[key] = shard_name
            total_size += tensors[key].numel() * tensors[key].element_size()
        files[shard_name] = {"size": shard_path.stat().st_size, "sha256": _sha256(shard_path)}
        del tensors

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path = output_root / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    files[index_path.name] = {"size": index_path.stat().st_size, "sha256": _sha256(index_path)}
    return {
        "format": "hf_safetensors",
        "index_file": index_path.name,
        "state_key_count": len(weight_map),
        "total_size": total_size,
        "shard_count": len(groups),
    }


def build_self_contained_bundle(
    *,
    strategy: str,
    strategy_description: str = "",
    strategy_plan: list[dict[str, str]] | None = None,
    quant_artifact_dir: str | Path,
    checkpoint_path: str | Path,
    config_file: str | Path,
    tokenizer_dir: str | Path,
    vae_path: str | Path,
    output_dir: str | Path,
    copy_mode: str = "copy",
    max_shard_size_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    """Convert a legacy packed artifact plus DCP into one portable bundle."""

    quant_root = Path(quant_artifact_dir).expanduser().resolve()
    if not strategy:
        raise ValueError("A reviewed quantization strategy name is required for a self-contained bundle")
    checkpoint_root = Path(checkpoint_path).expanduser().resolve()
    config_source = Path(config_file).expanduser().resolve()
    tokenizer_source = Path(tokenizer_dir).expanduser().resolve()
    vae_source = Path(vae_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing quant bundle: {output_root}")
    if not (checkpoint_root / ".metadata").is_file():
        raise FileNotFoundError(f"DCP metadata does not exist: {checkpoint_root / '.metadata'}")
    if not config_source.is_file():
        raise FileNotFoundError(f"Runtime config does not exist: {config_source}")
    if not vae_source.is_file():
        raise FileNotFoundError(f"Wan VAE does not exist: {vae_source}")

    source_manifest_path = quant_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema_version") != 1:
        raise ValueError("Self-contained conversion currently expects a schema-v1 packed quant artifact")
    modules = source_manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError("Source quant artifact has no modules")
    source_metadata_path = quant_root / "cosmos3_quant_metadata.json"
    source_metadata: dict[str, Any] = {}
    if source_metadata_path.is_file():
        loaded_metadata = json.loads(source_metadata_path.read_text())
        if not isinstance(loaded_metadata, dict):
            raise TypeError(f"Quant artifact metadata must be a dict: {source_metadata_path}")
        source_metadata = loaded_metadata

    temp_root = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temp_root.exists():
        raise FileExistsError(f"Temporary quant bundle path already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    try:
        copied_modules: list[dict[str, Any]] = []
        for entry in modules:
            copied_entry = dict(entry)
            rel_path = _safe_relative_path(copied_entry.get("tensor_file"), "modules[].tensor_file")
            source_tensor = quant_root / rel_path
            target_tensor = temp_root / rel_path
            if not source_tensor.is_file():
                raise FileNotFoundError(f"Missing packed quant tensor: {source_tensor}")
            files[rel_path.as_posix()] = _copy_file(source_tensor, target_tensor, copy_mode=copy_mode)
            copied_modules.append(copied_entry)

        model = _write_residual_safetensors(
            checkpoint_root,
            copied_modules,
            temp_root,
            max_shard_size_bytes=max_shard_size_bytes,
            files=files,
        )

        tokenizer_target = temp_root / "assets/qwen3_vl_tokenizer"
        _copy_tree(
            tokenizer_source,
            tokenizer_target,
            copy_mode=copy_mode,
            bundle_root=temp_root,
            files=files,
        )
        vae_target = temp_root / "assets/Wan2.2_VAE.pth"
        files[vae_target.relative_to(temp_root).as_posix()] = _copy_file(
            vae_source, vae_target, copy_mode=copy_mode
        )

        runtime_config = temp_root / "runtime/config.yaml"
        replacement_count = _portable_runtime_config(
            config_source,
            runtime_config,
            checkpoint_path=checkpoint_root,
            tokenizer_dir=tokenizer_source,
            vae_path=vae_source,
        )
        files[runtime_config.relative_to(temp_root).as_posix()] = {
            "size": runtime_config.stat().st_size,
            "sha256": _sha256(runtime_config),
        }

        hf_config = temp_root / "config.json"
        hf_config.write_text(
            json.dumps(
                {
                    "artifact_type": BUNDLE_ARTIFACT_TYPE,
                    "model_type": "cosmos3_omni_quant_bundle",
                    "schema_version": BUNDLE_SCHEMA_VERSION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files[hf_config.name] = {"size": hf_config.stat().st_size, "sha256": _sha256(hf_config)}

        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "artifact_type": BUNDLE_ARTIFACT_TYPE,
            "self_contained": True,
            "created_unix": time.time(),
            "source": {
                "checkpoint_path": str(checkpoint_root),
                "config_file": str(config_source),
                "quant_artifact_dir": str(quant_root),
                "quant_schema_version": source_manifest.get("schema_version"),
                "quant_plan_file": source_manifest.get("quant_plan_file", ""),
                "calibration": source_manifest.get("calibration", {}),
            },
            "quantization": {
                "strategy": strategy,
                "description": strategy_description or source_metadata.get("description", ""),
                "weight_only": source_metadata.get("weight_only", True),
                "activation_quant": source_metadata.get("activation_quant", False),
                "runtime_backend": source_metadata.get("runtime_backend", "vllm_marlin_wna16"),
                "plan": strategy_plan if strategy_plan is not None else source_metadata.get("plan", []),
                "quant_backend": source_manifest.get("quant_backend", ""),
                "quant_target_prefix": source_manifest.get("quant_target_prefix", ""),
            },
            "model": model,
            "runtime": {
                "config_file": runtime_config.relative_to(temp_root).as_posix(),
                "bundle_root_token": BUNDLE_ROOT_TOKEN,
                "config_replacement_count": replacement_count,
                "tokenizer_dir": tokenizer_target.relative_to(temp_root).as_posix(),
                "vae_path": vae_target.relative_to(temp_root).as_posix(),
            },
            "modules": copied_modules,
            "files": files,
        }
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temp_root, output_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return validate_self_contained_bundle(output_root, check_hashes=False)


def validate_self_contained_bundle(
    bundle_dir: str | Path,
    *,
    check_hashes: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing quant bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Expected quant bundle schema_version={BUNDLE_SCHEMA_VERSION}")
    if manifest.get("artifact_type") != BUNDLE_ARTIFACT_TYPE:
        raise ValueError(f"Unsupported quant bundle artifact_type={manifest.get('artifact_type')!r}")
    if manifest.get("self_contained") is not True:
        raise ValueError("Quant bundle must declare self_contained=true")

    model = manifest.get("model")
    runtime = manifest.get("runtime")
    files = manifest.get("files")
    modules = manifest.get("modules")
    if not isinstance(model, dict) or not isinstance(runtime, dict) or not isinstance(files, dict):
        raise TypeError("Quant bundle manifest requires model, runtime, and files objects")
    if not isinstance(modules, list) or not modules:
        raise TypeError("Quant bundle manifest requires a non-empty modules list")

    required_paths = [
        model.get("index_file"),
        "config.json",
        runtime.get("config_file"),
        runtime.get("tokenizer_dir"),
        runtime.get("vae_path"),
        *(entry.get("tensor_file") for entry in modules),
    ]
    for index, value in enumerate(required_paths):
        rel_path = _safe_relative_path(value, f"required_paths[{index}]")
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing required quant bundle path: {path}")

    index_path = root / _safe_relative_path(model.get("index_file"), "model.index_file")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Quant bundle safetensors index has no weight_map")
    if len(weight_map) != int(model.get("state_key_count", -1)):
        raise ValueError("Quant bundle state_key_count does not match safetensors index")
    for key, rel_file in weight_map.items():
        if not isinstance(key, str):
            raise TypeError("Quant bundle state keys must be strings")
        shard = root / _safe_relative_path(rel_file, f"weight_map[{key!r}]")
        if not shard.is_file():
            raise FileNotFoundError(f"Missing quant bundle residual shard: {shard}")

    for rel_file, expected in files.items():
        path = root / _safe_relative_path(rel_file, f"files[{rel_file!r}]")
        if not path.is_file():
            raise FileNotFoundError(f"Missing quant bundle file: {path}")
        if not isinstance(expected, dict) or path.stat().st_size != int(expected.get("size", -1)):
            raise ValueError(f"Quant bundle file size mismatch: {path}")
        if check_hashes and _sha256(path) != expected.get("sha256"):
            raise ValueError(f"Quant bundle SHA256 mismatch: {path}")

    runtime_config = root / _safe_relative_path(runtime.get("config_file"), "runtime.config_file")
    config_text = runtime_config.read_text()
    if BUNDLE_ROOT_TOKEN not in config_text:
        raise ValueError("Quant bundle runtime config does not contain the bundle-root token")
    return {
        "root": str(root),
        "manifest": manifest,
        "modules": len(modules),
        "state_keys": len(weight_map),
        "model_bytes": int(model.get("total_size", 0)),
        "file_bytes": sum(int(value["size"]) for value in files.values()),
        "hashes_checked": check_hashes,
    }


def materialize_bundle_config(bundle_dir: str | Path, output_dir: str | Path) -> Path:
    validation = validate_self_contained_bundle(bundle_dir, check_hashes=False)
    root = Path(validation["root"])
    manifest = validation["manifest"]
    runtime_config = root / _safe_relative_path(manifest["runtime"]["config_file"], "runtime.config_file")
    text = runtime_config.read_text().replace(BUNDLE_ROOT_TOKEN, str(root))
    if BUNDLE_ROOT_TOKEN in text:
        raise ValueError("Failed to resolve the quant bundle root in runtime config")
    target = Path(output_dir).expanduser() / "quant_bundle.runtime.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target
