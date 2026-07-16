# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Self-contained packed W4/W8 bundles for Cosmos3 DROID/RoboLab policies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

ROBOLAB_BUNDLE_SCHEMA_VERSION = 1
ROBOLAB_BUNDLE_ARTIFACT_TYPE = "cosmos3_robolab_quantized_policy"
ROBOLAB_BUNDLE_ROOT_TOKEN = "__COSMOS3_ROBOLAB_QUANT_BUNDLE_ROOT__"

StrategyName = Literal["full_w8", "full_w4", "attention_w8", "gen_branch_w8"]

_LINEAR_WEIGHT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?:"
    r"(?P<mlp>mlp|mlp_moe_gen)\.(?:down_proj|gate_proj|up_proj)"
    r"|self_attn\.(?P<attn>to_q|to_k|to_v|to_out|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
    r")\.weight$"
)


@dataclass(frozen=True)
class QuantTarget:
    source_key: str
    module_name: str
    backend_class: str
    num_bits: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _copy_file(source: Path, target: Path, *, copy_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, target)
    elif copy_mode == "hardlink":
        os.link(source, target)
    else:
        raise ValueError(f"Unsupported copy_mode={copy_mode!r}; expected 'copy' or 'hardlink'")


def _copy_tree(source: Path, target: Path, *, copy_mode: str) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {source}")
    copied: list[Path] = []
    for source_path in sorted(source.rglob("*")):
        if source_path.is_dir():
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"Source tree contains a broken link: {source_path}")
        target_path = target / source_path.relative_to(source)
        _copy_file(source_path.resolve(), target_path, copy_mode=copy_mode)
        copied.append(target_path)
    return copied


def _read_weight_map(source_root: Path) -> dict[str, str]:
    index_path = source_root / "model.safetensors.index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Missing Diffusers weight_map in {index_path}")
    result = {str(key): str(value) for key, value in weight_map.items()}
    missing = sorted({rel for rel in result.values() if not (source_root / rel).is_file()})
    if missing:
        raise FileNotFoundError(f"Source checkpoint is missing indexed shard(s): {missing[:8]}")
    return result


def _strategy_backend(strategy: StrategyName, source_key: str) -> tuple[str, int] | None:
    match = _LINEAR_WEIGHT_RE.fullmatch(source_key)
    if match is None:
        return None
    if strategy == "full_w8":
        return "VllmGptqMarlinW8A16Linear", 8
    if strategy == "full_w4":
        return "VllmGptqMarlinW4A16Linear", 4
    if strategy == "attention_w8":
        return (
            ("VllmGptqMarlinW8A16Linear", 8)
            if match.group("attn") is not None
            else ("VllmGptqMarlinW4A16Linear", 4)
        )
    if strategy == "gen_branch_w8":
        is_generation = match.group("mlp") == "mlp_moe_gen" or match.group("attn") in {
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        }
        return (
            ("VllmGptqMarlinW8A16Linear", 8)
            if is_generation
            else ("VllmGptqMarlinW4A16Linear", 4)
        )
    raise ValueError(f"Unsupported RoboLab quantization strategy: {strategy}")


def discover_quant_targets(source_root: str | Path, strategy: StrategyName) -> list[QuantTarget]:
    """Map Diffusers Linear weights to the native OmniMoT module hierarchy."""

    from cosmos_framework.inference.model import _diffusers_to_net_key

    root = Path(source_root).expanduser().resolve()
    targets: list[QuantTarget] = []
    for source_key, rel_path in sorted(_read_weight_map(root).items()):
        backend = _strategy_backend(strategy, source_key)
        if backend is None:
            continue
        net_key = _diffusers_to_net_key(source_key, rel_path)
        if net_key is None or not net_key.endswith(".weight"):
            raise KeyError(f"Quantized Diffusers key has no native model mapping: {source_key!r}")
        backend_class, num_bits = backend
        targets.append(
            QuantTarget(
                source_key=source_key,
                module_name=f"net.{net_key.removesuffix('.weight')}",
                backend_class=backend_class,
                num_bits=num_bits,
            )
        )
    if len(targets) != 504:
        raise ValueError(f"Expected 504 DROID language/MoT Linear targets, found {len(targets)}")
    if len({target.module_name for target in targets}) != len(targets):
        raise ValueError("Multiple Diffusers weights map to the same quantized module")
    return targets


def _load_backend_module() -> Any:
    module_path = Path(__file__).resolve().parent / "quant_backend_microbench.py"
    spec = importlib.util.spec_from_file_location("cosmos3_robolab_quant_backends", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import quant backend module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_calibration_stats(path: str | Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    loaded = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TypeError("Calibration stats must be a dict of module name to input-channel amax tensor")
    result: dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise TypeError(f"Invalid calibration entry for {name!r}")
        result[name] = value.detach().float().cpu()
    return result


def _input_scale(stats: torch.Tensor, *, size_k: int, alpha: float) -> torch.Tensor:
    if stats.numel() != size_k:
        raise ValueError(f"Calibration vector has {stats.numel()} channels, expected {size_k}")
    values = stats.clamp(min=1e-6)
    normalized = values / values.mean().clamp(min=1e-6)
    return normalized.pow(alpha).clamp(min=1e-2, max=1e2).to(torch.bfloat16)


def _portable_runtime_config(
    source_config: Path,
    target: Path,
    *,
    source_checkpoint: Path,
    tokenizer_dir: Path,
    vae_path: Path,
) -> None:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    model_config = config["model"]["config"]
    model_config["tokenizer"]["bucket_name"] = ""
    model_config["tokenizer"]["object_store_credential_path_pretrained"] = ""
    model_config["tokenizer"]["vae_path"] = f"{ROBOLAB_BUNDLE_ROOT_TOKEN}/assets/Wan2.2_VAE.pth"
    tokenizer_config = model_config["vlm_config"]["tokenizer"]
    tokenizer_config.pop("repository", None)
    tokenizer_config.pop("revision", None)
    tokenizer_config.pop("subdir", None)
    tokenizer_config["tokenizer_type"] = f"{ROBOLAB_BUNDLE_ROOT_TOKEN}/assets/qwen3_vl_tokenizer"
    pretrained = model_config["vlm_config"].get("pretrained_weights")
    if isinstance(pretrained, dict):
        pretrained["enabled"] = False
        pretrained["backbone_path"] = ""
        pretrained["credentials_path"] = ""
        pretrained["enable_gcs_patch_in_boto3"] = False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = target.read_text(encoding="utf-8")
    for forbidden in (str(source_checkpoint), str(tokenizer_dir), str(vae_path)):
        if forbidden and forbidden in text:
            raise ValueError(f"Portable runtime config retained an export-time path: {forbidden}")


def materialize_robolab_bundle_config(bundle_dir: str | Path, output_dir: str | Path) -> Path:
    root = Path(bundle_dir).expanduser().resolve()
    validate_robolab_quant_bundle(root)
    template = root / "runtime/config.json"
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "robolab_quant_runtime_config.json"
    target.write_text(
        template.read_text(encoding="utf-8").replace(ROBOLAB_BUNDLE_ROOT_TOKEN, str(root)),
        encoding="utf-8",
    )
    if ROBOLAB_BUNDLE_ROOT_TOKEN in target.read_text(encoding="utf-8"):
        raise ValueError("Failed to materialize RoboLab quant bundle root")
    return target


def _flush_residual_shard(
    tensors: dict[str, torch.Tensor],
    output_root: Path,
    shard_index: int,
) -> tuple[str, int]:
    name = f"model-{shard_index:05d}.safetensors"
    save_file(tensors, str(output_root / name), metadata={"format": "pt"})
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    return name, total_size


def build_robolab_quant_bundle(
    *,
    strategy: StrategyName,
    source_checkpoint: str | Path,
    tokenizer_dir: str | Path,
    vae_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    calibration_stats: str | Path | None = None,
    calibration_alpha: float = 0.5,
    allow_uncalibrated_w4: bool = False,
    copy_mode: str = "copy",
    max_residual_shard_size: int = 2 * 1024**3,
    source_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream a Diffusers DROID checkpoint into one deployable packed bundle."""

    source_root = Path(source_checkpoint).expanduser().resolve()
    tokenizer_source = Path(tokenizer_dir).expanduser().resolve()
    vae_source = Path(vae_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    source_config = source_root / "config.json"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing RoboLab quant bundle: {output_root}")
    if not source_config.is_file():
        raise FileNotFoundError(f"DROID config does not exist: {source_config}")
    if not vae_source.is_file():
        raise FileNotFoundError(f"Wan VAE does not exist: {vae_source}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to pack Marlin weights")

    targets = discover_quant_targets(source_root, strategy)
    target_by_source = {target.source_key: target for target in targets}
    stats = _load_calibration_stats(calibration_stats)
    w4_names = {target.module_name for target in targets if target.num_bits == 4}
    missing_stats = sorted(w4_names - set(stats))
    if missing_stats and not allow_uncalibrated_w4:
        raise ValueError(
            f"Strategy {strategy!r} has {len(w4_names)} W4 modules but calibration stats are missing for "
            f"{len(missing_stats)} modules. Pass DROID training calibration stats or explicitly use "
            "allow_uncalibrated_w4 for an exploratory artifact."
        )

    temp_root = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temp_root.exists():
        raise FileExistsError(f"Temporary bundle path already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    (temp_root / "tensors").mkdir()
    files: dict[str, dict[str, Any]] = {}
    modules: list[dict[str, Any]] = []
    residual_weight_map: dict[str, str] = {}
    residual_tensors: dict[str, torch.Tensor] = {}
    residual_bytes = 0
    residual_total_size = 0
    residual_shard_index = 0
    backend_module = _load_backend_module()
    weight_map = _read_weight_map(source_root)
    source_by_shard: dict[str, list[str]] = {}
    for source_key, rel_path in weight_map.items():
        source_by_shard.setdefault(rel_path, []).append(source_key)

    from cosmos_framework.inference.model import _diffusers_to_net_key

    try:
        for rel_path, source_keys in sorted(source_by_shard.items()):
            logger.info(
                "Packing source shard %s (%d source tensors; %d/%d modules complete)",
                rel_path,
                len(source_keys),
                len(modules),
                len(targets),
            )
            with safe_open(str(source_root / rel_path), framework="pt", device="cpu") as shard:
                for source_key in sorted(source_keys):
                    net_key = _diffusers_to_net_key(source_key, rel_path)
                    if net_key is None:
                        continue
                    target = target_by_source.get(source_key)
                    tensor = shard.get_tensor(source_key)
                    if target is not None:
                        weight = tensor.to(device=device, dtype=torch.bfloat16)
                        input_scale = None
                        if target.num_bits == 4 and target.module_name in stats:
                            input_scale = _input_scale(
                                stats[target.module_name], size_k=int(weight.shape[1]), alpha=calibration_alpha
                            )
                        if target.num_bits == 8:
                            backend = backend_module.VllmGptqMarlinW8A16Linear(weight)
                        else:
                            backend = backend_module.VllmGptqMarlinW4A16Linear(weight, input_scale=input_scale)
                        torch.cuda.synchronize(device)
                        payload = {
                            "backend_class": target.backend_class,
                            "bias": None,
                            "qweight": backend.qweight.detach().cpu().contiguous(),
                            "scales": backend.scales.detach().cpu().contiguous(),
                            "input_scale": (
                                backend.input_scale.detach().cpu().contiguous()
                                if isinstance(getattr(backend, "input_scale", None), torch.Tensor)
                                else None
                            ),
                        }
                        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.module_name)
                        tensor_rel = f"tensors/{safe_name}.pt"
                        torch.save(payload, temp_root / tensor_rel)
                        files[tensor_rel] = _file_record(temp_root / tensor_rel)
                        modules.append(
                            {
                                "name": target.module_name,
                                "source_key": source_key,
                                "backend_class": target.backend_class,
                                "format": "vllm_marlin_wna16",
                                "num_bits": target.num_bits,
                                "group_size": int(backend.group_size),
                                "size_k": int(backend.size_k),
                                "size_n": int(backend.size_n),
                                "wtype_id": int(backend.wtype_id),
                                "tensor_file": tensor_rel,
                            }
                        )
                        del payload, backend, weight, tensor
                        torch.cuda.empty_cache()
                        continue

                    target_key = f"net.{net_key}"
                    tensor_size = tensor.numel() * tensor.element_size()
                    if residual_tensors and residual_bytes + tensor_size > max_residual_shard_size:
                        residual_shard_index += 1
                        shard_name, shard_size = _flush_residual_shard(
                            residual_tensors, temp_root, residual_shard_index
                        )
                        files[shard_name] = _file_record(temp_root / shard_name)
                        for key in residual_tensors:
                            residual_weight_map[key] = shard_name
                        residual_total_size += shard_size
                        residual_tensors = {}
                        residual_bytes = 0
                    if target_key in residual_tensors or target_key in residual_weight_map:
                        raise KeyError(f"Duplicate residual target key: {target_key}")
                    residual_tensors[target_key] = tensor.contiguous()
                    residual_bytes += tensor_size

        if residual_tensors:
            residual_shard_index += 1
            shard_name, shard_size = _flush_residual_shard(residual_tensors, temp_root, residual_shard_index)
            files[shard_name] = _file_record(temp_root / shard_name)
            for key in residual_tensors:
                residual_weight_map[key] = shard_name
            residual_total_size += shard_size

        if len(modules) != 504:
            raise ValueError(f"Packed module count changed during export: {len(modules)}")

        residual_index = {
            "metadata": {"total_size": residual_total_size},
            "weight_map": residual_weight_map,
        }
        index_path = temp_root / "model.safetensors.index.json"
        index_path.write_text(json.dumps(residual_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files[index_path.name] = _file_record(index_path)

        runtime_config = temp_root / "runtime/config.json"
        _portable_runtime_config(
            source_config,
            runtime_config,
            source_checkpoint=source_root,
            tokenizer_dir=tokenizer_source,
            vae_path=vae_source,
        )
        files[runtime_config.relative_to(temp_root).as_posix()] = _file_record(runtime_config)

        logger.info("Copying tokenizer and Wan VAE into the deployment bundle")
        tokenizer_target = temp_root / "assets/qwen3_vl_tokenizer"
        for path in _copy_tree(tokenizer_source, tokenizer_target, copy_mode=copy_mode):
            files[path.relative_to(temp_root).as_posix()] = _file_record(path)
        vae_target = temp_root / "assets/Wan2.2_VAE.pth"
        _copy_file(vae_source, vae_target, copy_mode=copy_mode)
        files[vae_target.relative_to(temp_root).as_posix()] = _file_record(vae_target)

        hf_config = temp_root / "config.json"
        hf_config.write_text(
            json.dumps(
                {
                    "artifact_type": ROBOLAB_BUNDLE_ARTIFACT_TYPE,
                    "model_type": "cosmos3_omni_quant_bundle",
                    "schema_version": ROBOLAB_BUNDLE_SCHEMA_VERSION,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        files[hf_config.name] = _file_record(hf_config)

        counts = {
            "w4": sum(target.num_bits == 4 for target in targets),
            "w8": sum(target.num_bits == 8 for target in targets),
        }
        manifest = {
            "schema_version": ROBOLAB_BUNDLE_SCHEMA_VERSION,
            "artifact_type": ROBOLAB_BUNDLE_ARTIFACT_TYPE,
            "self_contained": True,
            "created_unix": time.time(),
            "source": {
                "checkpoint_path": str(source_root),
                "tokenizer_dir": str(tokenizer_source),
                "vae_path": str(vae_source),
                "provenance": source_provenance or {},
            },
            "quantization": {
                "strategy": strategy,
                "weight_only": True,
                "activation_quantization": False,
                "w4_modules": counts["w4"],
                "w8_modules": counts["w8"],
                "calibration_stats": str(Path(calibration_stats).expanduser()) if calibration_stats else "",
                "calibration_alpha": calibration_alpha,
                "uncalibrated_w4": bool(missing_stats),
            },
            "runtime": {
                "config_file": "runtime/config.json",
                "tokenizer_dir": "assets/qwen3_vl_tokenizer",
                "vae_path": "assets/Wan2.2_VAE.pth",
                "residual_index": "model.safetensors.index.json",
            },
            "model": {
                "quant_module_count": len(modules),
                "residual_state_key_count": len(residual_weight_map),
                "residual_total_size": residual_total_size,
                "residual_shard_count": residual_shard_index,
            },
            "modules": sorted(modules, key=lambda item: item["name"]),
            "files": dict(sorted(files.items())),
        }
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_root.rename(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    validation = validate_robolab_quant_bundle(output_root)
    return {
        "bundle_dir": str(output_root),
        "strategy": strategy,
        "modules": validation["modules"],
        "w4_modules": counts["w4"],
        "w8_modules": counts["w8"],
        "residual_state_keys": len(residual_weight_map),
        "bundle_bytes": sum(record["size"] for record in files.values()),
    }


def validate_robolab_quant_bundle(
    bundle_dir: str | Path,
    *,
    expected_strategy: str | None = None,
    check_hashes: bool = False,
    check_tensors: bool = False,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RoboLab quant manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ROBOLAB_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported RoboLab quant bundle schema: {manifest.get('schema_version')!r}")
    if manifest.get("artifact_type") != ROBOLAB_BUNDLE_ARTIFACT_TYPE or manifest.get("self_contained") is not True:
        raise ValueError("RoboLab quant artifact is not a self-contained deployment bundle")
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("activation_quantization") is not False:
        raise ValueError("RoboLab quant bundle must declare weight-only quantization")
    strategy = str(quantization.get("strategy", ""))
    if expected_strategy is not None and strategy != expected_strategy:
        raise ValueError(f"Expected strategy {expected_strategy!r}, found {strategy!r}")
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != 504:
        raise ValueError(f"RoboLab quant bundle must contain 504 packed modules, found {len(modules or [])}")
    names: set[str] = set()
    for entry in modules:
        if not isinstance(entry, dict):
            raise TypeError("RoboLab quant module metadata must be dictionaries")
        name = str(entry.get("name", ""))
        if not name.startswith("net.language_model.model.layers.") or name in names:
            raise ValueError(f"Invalid or duplicate quant module name: {name!r}")
        names.add(name)
        num_bits = int(entry.get("num_bits", 0))
        backend_class = str(entry.get("backend_class", ""))
        if num_bits not in {4, 8} or f"W{num_bits}A16" not in backend_class:
            raise ValueError(f"Inconsistent quant backend metadata for {name}")
        rel = Path(str(entry.get("tensor_file", "")))
        if rel.is_absolute() or ".." in rel.parts or not (root / rel).is_file():
            raise ValueError(f"Quant tensor path must stay under the bundle root: {rel}")
        if check_tensors:
            payload = torch.load(root / rel, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not {"qweight", "scales"}.issubset(payload):
                raise ValueError(f"Invalid packed payload: {rel}")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("RoboLab quant bundle has no runtime metadata")
    for field in ("config_file", "tokenizer_dir", "vae_path", "residual_index"):
        rel = Path(str(runtime.get(field, "")))
        if rel.is_absolute() or ".." in rel.parts or not (root / rel).exists():
            raise ValueError(f"Invalid runtime path {field}: {rel}")
    runtime_text = (root / str(runtime["config_file"])).read_text(encoding="utf-8")
    if ROBOLAB_BUNDLE_ROOT_TOKEN not in runtime_text:
        raise ValueError("Portable runtime config has no bundle-root token")

    residual_index = json.loads((root / str(runtime["residual_index"])).read_text(encoding="utf-8"))
    residual_map = residual_index.get("weight_map")
    if not isinstance(residual_map, dict) or not residual_map:
        raise ValueError("RoboLab quant bundle residual index is empty")
    for key, rel_value in residual_map.items():
        if not str(key).startswith("net."):
            raise ValueError(f"Residual key is outside OmniMoT net: {key!r}")
        rel = Path(str(rel_value))
        if rel.is_absolute() or ".." in rel.parts or not (root / rel).is_file():
            raise ValueError(f"Invalid residual shard path: {rel}")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("RoboLab quant bundle has no file integrity manifest")
    for rel_value, expected in files.items():
        rel = Path(str(rel_value))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Integrity path escapes bundle root: {rel}")
        path = root / rel
        if not path.is_file() or path.stat().st_size != int(expected["size"]):
            raise ValueError(f"Bundle file size mismatch: {rel}")
        if check_hashes and _sha256(path) != str(expected["sha256"]):
            raise ValueError(f"Bundle file SHA256 mismatch: {rel}")

    return {
        "manifest": manifest,
        "strategy": strategy,
        "modules": len(modules),
        "residual_state_keys": len(residual_map),
        "bundle_bytes": sum(int(item["size"]) for item in files.values()),
    }
