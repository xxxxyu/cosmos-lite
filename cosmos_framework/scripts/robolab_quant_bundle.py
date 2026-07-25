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

from cosmos_framework.scripts._export_model_helpers import build_vision_encoder_bundle_config

logger = logging.getLogger(__name__)

ROBOLAB_BUNDLE_SCHEMA_VERSION = 1
ROBOLAB_BUNDLE_ARTIFACT_TYPE = "cosmos3_robolab_quantized_policy"
ROBOLAB_BUNDLE_ROOT_TOKEN = "__COSMOS3_ROBOLAB_QUANT_BUNDLE_ROOT__"

StrategyName = Literal["full_w8", "full_w4", "attention_w8", "gen_branch_w8", "gen_branch_w8a8"]
ModelFamily = Literal["cosmos3_nano", "cosmos3_edge"]

_QUANT_MODULE_PREFIX = "net.language_model.model.layers."
_EDGE_PROCESSOR_FILES = (
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)

_LINEAR_WEIGHT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?:"
    r"(?P<mlp>mlp|mlp_moe_gen)\.(?:down_proj|gate_proj|up_proj)"
    r"|self_attn\.(?P<attn>to_q|to_k|to_v|to_out|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
    r")\.weight$"
)

_ATTENTION_MODULE_NAMES = {
    "to_q": "q_proj",
    "to_k": "k_proj",
    "to_v": "v_proj",
    "to_out": "o_proj",
    "add_q_proj": "q_proj_moe_gen",
    "add_k_proj": "k_proj_moe_gen",
    "add_v_proj": "v_proj_moe_gen",
    "to_add_out": "o_proj_moe_gen",
}


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
    def read_index(path: Path) -> dict[str, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing Diffusers weight_map in {path}")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
            raise TypeError(f"Diffusers weight_map must contain string keys and values: {path}")
        return dict(weight_map)

    root_index = source_root / "model.safetensors.index.json"
    result = read_index(root_index) if root_index.is_file() else {}
    result = {key: value for key, value in result.items() if ".k_norm_und_for_gen." not in key}
    transformer_index = source_root / "transformer/diffusion_pytorch_model.safetensors.index.json"
    if transformer_index.is_file():
        for key, rel_path in read_index(transformer_index).items():
            result.setdefault(key, f"transformer/{rel_path}")
    if not result:
        raise FileNotFoundError(f"No Diffusers safetensors index found under {source_root}")
    missing = sorted({rel for rel in result.values() if not (source_root / rel).is_file()})
    if missing:
        raise FileNotFoundError(f"Source checkpoint is missing indexed shard(s): {missing[:8]}")
    return result


def _quant_module_name(source_key: str) -> str:
    stem = source_key.removesuffix(".weight")
    prefix, separator, attention_name = stem.rpartition(".self_attn.")
    if separator:
        stem = f"{prefix}.self_attn.{_ATTENTION_MODULE_NAMES[attention_name]}"
    return f"net.language_model.model.{stem}"


def _strategy_backend(strategy: StrategyName, source_key: str) -> tuple[str, int] | None:
    match = _LINEAR_WEIGHT_RE.fullmatch(source_key)
    if match is None:
        return None
    if strategy == "full_w8":
        return "VllmGptqMarlinW8A16Linear", 8
    if strategy == "full_w4":
        return "VllmGptqMarlinW4A16Linear", 4
    if strategy == "attention_w8":
        return ("VllmGptqMarlinW8A16Linear", 8) if match.group("attn") is not None else ("VllmGptqMarlinW4A16Linear", 4)
    if strategy in {"gen_branch_w8", "gen_branch_w8a8"}:
        is_generation = match.group("mlp") == "mlp_moe_gen" or match.group("attn") in {
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        }
        if is_generation:
            backend = "VllmCutlassFp8W8A8Linear" if strategy == "gen_branch_w8a8" else "VllmGptqMarlinW8A16Linear"
            return backend, 8
        return "VllmGptqMarlinW4A16Linear", 4
    raise ValueError(f"Unsupported RoboLab quantization strategy: {strategy}")


def detect_model_family(source_root: str | Path) -> ModelFamily:
    root = Path(source_root).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model_config = config.get("model", {}).get("config", {})
    model_name = str(model_config.get("vlm_config", {}).get("model_name", ""))
    transformer_layers = (model_config.get("net") or {}).get("num_hidden_layers")
    if "Cosmos3-Edge" in model_name or transformer_layers == 28:
        return "cosmos3_edge"
    return "cosmos3_nano"


def _manifest_source(
    *,
    source_root: Path,
    processor_source: Path,
    vae_source: Path,
    model_family: ModelFamily,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    if not provenance:
        return {
            "checkpoint_path": source_root.name,
            "tokenizer_dir": processor_source.name,
            "vae_path": vae_source.name,
            "provenance": {},
        }

    repositories = provenance.get("repositories") or {}
    revisions = provenance.get("resolved_revisions") or {}
    droid_uri = f"hf://{repositories['droid']}@{revisions['droid']}"
    processor_uri = droid_uri if model_family == "cosmos3_edge" else f"hf://{repositories['qwen']}@{revisions['qwen']}"
    return {
        "checkpoint_path": droid_uri,
        "tokenizer_dir": processor_uri,
        "vae_path": f"hf://{repositories['wan']}@{revisions['wan']}/{vae_source.name}",
        "provenance": provenance,
    }


def discover_quant_targets(source_root: str | Path, strategy: StrategyName) -> list[QuantTarget]:
    """Map Diffusers Linear weights to the native OmniMoT module hierarchy."""

    root = Path(source_root).expanduser().resolve()
    model_family = detect_model_family(root)
    targets: list[QuantTarget] = []
    for source_key, rel_path in sorted(_read_weight_map(root).items()):
        backend = _strategy_backend(strategy, source_key)
        if backend is None:
            continue
        backend_class, num_bits = backend
        targets.append(
            QuantTarget(
                source_key=source_key,
                module_name=_quant_module_name(source_key),
                backend_class=backend_class,
                num_bits=num_bits,
            )
        )
    expected_targets = 336 if model_family == "cosmos3_edge" else 504
    if len(targets) != expected_targets:
        raise ValueError(
            f"Expected {expected_targets} {model_family} DROID language/MoT Linear targets, found {len(targets)}"
        )
    if len({target.module_name for target in targets}) != len(targets):
        raise ValueError("Multiple Diffusers weights map to the same quantized module")
    layers = [int(target.module_name.split(".layers.", 1)[1].split(".", 1)[0]) for target in targets]
    if sorted(set(layers)) != list(range(max(layers) + 1)):
        raise ValueError("Quantized DROID layers must be contiguous and zero-indexed")
    per_layer = {layer: layers.count(layer) for layer in set(layers)}
    if len(set(per_layer.values())) != 1:
        raise ValueError(f"Quantized DROID target count differs by layer: {per_layer}")
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
    processor_dir: Path,
    processor_asset_path: str,
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
    tokenizer_config["tokenizer_type"] = f"{ROBOLAB_BUNDLE_ROOT_TOKEN}/{processor_asset_path}"
    pretrained = model_config["vlm_config"].get("pretrained_weights")
    if isinstance(pretrained, dict):
        pretrained["enabled"] = False
        pretrained["backbone_path"] = ""
        pretrained["credentials_path"] = ""
        pretrained["enable_gcs_patch_in_boto3"] = False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = target.read_text(encoding="utf-8")
    for forbidden in (str(source_checkpoint), str(processor_dir), str(vae_path)):
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

    model_family = detect_model_family(source_root)
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
            if model_family == "cosmos3_edge" and rel_path == "vision_encoder/model.safetensors":
                continue
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
                        if (
                            target.num_bits == 4 or target.backend_class == "VllmCutlassFp8W8A8Linear"
                        ) and target.module_name in stats:
                            input_scale = _input_scale(
                                stats[target.module_name], size_k=int(weight.shape[1]), alpha=calibration_alpha
                            )
                        if target.backend_class == "VllmCutlassFp8W8A8Linear":
                            backend = backend_module.VllmCutlassFp8W8A8Linear(weight, input_scale=input_scale)
                        elif target.num_bits == 8:
                            backend = backend_module.VllmGptqMarlinW8A16Linear(weight)
                        else:
                            backend = backend_module.VllmGptqMarlinW4A16Linear(weight, input_scale=input_scale)
                        torch.cuda.synchronize(device)
                        if target.backend_class == "VllmCutlassFp8W8A8Linear":
                            payload = {
                                "backend_class": target.backend_class,
                                "bias": None,
                                "qweight_nk": backend.qweight_nk.detach().cpu().contiguous(),
                                "scale_b": backend.scale_b.detach().cpu().contiguous(),
                                "input_scale": (
                                    backend.input_scale.detach().cpu().contiguous()
                                    if isinstance(getattr(backend, "input_scale", None), torch.Tensor)
                                    else None
                                ),
                            }
                            module_metadata = {
                                "format": "vllm_cutlass_fp8_w8a8",
                                "num_bits": 8,
                                "activation_bits": 8,
                                "weight_scale": "per_output_channel",
                                "activation_scale": "dynamic_per_token",
                            }
                        else:
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
                            module_metadata = {
                                "format": "vllm_marlin_wna16",
                                "num_bits": target.num_bits,
                                "activation_bits": 16,
                                "group_size": int(backend.group_size),
                                "wtype_id": int(backend.wtype_id),
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
                                **module_metadata,
                                "size_k": int(backend.size_k),
                                "size_n": int(backend.size_n),
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

        if len(modules) != len(targets):
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
            processor_dir=tokenizer_source,
            processor_asset_path=(
                "assets/cosmos3_edge_processor" if model_family == "cosmos3_edge" else "assets/qwen3_vl_tokenizer"
            ),
            vae_path=vae_source,
        )
        files[runtime_config.relative_to(temp_root).as_posix()] = _file_record(runtime_config)

        logger.info("Copying processor and Wan VAE into the deployment bundle")
        processor_asset_path = (
            "assets/cosmos3_edge_processor" if model_family == "cosmos3_edge" else "assets/qwen3_vl_tokenizer"
        )
        tokenizer_target = temp_root / processor_asset_path
        if model_family == "cosmos3_edge":
            for name in _EDGE_PROCESSOR_FILES:
                source = tokenizer_source / name
                if not source.is_file():
                    raise FileNotFoundError(f"Required Cosmos3 Edge processor file is missing: {source}")
                target = tokenizer_target / name
                _copy_file(source, target, copy_mode=copy_mode)
                files[target.relative_to(temp_root).as_posix()] = _file_record(target)
        else:
            for path in _copy_tree(tokenizer_source, tokenizer_target, copy_mode=copy_mode):
                files[path.relative_to(temp_root).as_posix()] = _file_record(path)
        vae_target = temp_root / "assets/Wan2.2_VAE.pth"
        _copy_file(vae_source, vae_target, copy_mode=copy_mode)
        files[vae_target.relative_to(temp_root).as_posix()] = _file_record(vae_target)
        if model_family == "cosmos3_edge":
            vision_weights = temp_root / "vision_encoder/model.safetensors"
            _copy_file(
                source_root / "vision_encoder/model.safetensors",
                vision_weights,
                copy_mode=copy_mode,
            )
            files[vision_weights.relative_to(temp_root).as_posix()] = _file_record(vision_weights)
            vision_config = temp_root / "vision_encoder/config.json"
            merged_vision_config = build_vision_encoder_bundle_config(source_root / "vision_encoder", source_root)
            vision_config.write_text(
                json.dumps(merged_vision_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            files[vision_config.relative_to(temp_root).as_posix()] = _file_record(vision_config)

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
            "w8a8": sum(target.backend_class == "VllmCutlassFp8W8A8Linear" for target in targets),
        }
        counts["w8a16"] = counts["w8"] - counts["w8a8"]
        manifest = {
            "schema_version": ROBOLAB_BUNDLE_SCHEMA_VERSION,
            "artifact_type": ROBOLAB_BUNDLE_ARTIFACT_TYPE,
            "self_contained": True,
            "model_family": model_family,
            "created_unix": time.time(),
            "source": _manifest_source(
                source_root=source_root,
                processor_source=tokenizer_source,
                vae_source=vae_source,
                model_family=model_family,
                provenance=source_provenance,
            ),
            "quantization": {
                "strategy": strategy,
                "weight_only": counts["w8a8"] == 0,
                "activation_quantization": counts["w8a8"] > 0,
                "activation_dtype": "fp8_e4m3fn" if counts["w8a8"] else "bf16",
                "w4_modules": counts["w4"],
                "w8_modules": counts["w8"],
                "w8a16_modules": counts["w8a16"],
                "w8a8_modules": counts["w8a8"],
                "calibration_stats": Path(calibration_stats).name if calibration_stats else "",
                "calibration_stats_sha256": _sha256(Path(calibration_stats).expanduser()) if calibration_stats else "",
                "calibration_alpha": calibration_alpha,
                "uncalibrated_w4": bool(missing_stats),
            },
            "runtime": {
                "config_file": "runtime/config.json",
                "tokenizer_dir": processor_asset_path,
                "vae_path": "assets/Wan2.2_VAE.pth",
                "residual_index": "model.safetensors.index.json",
                "vision_encoder_dir": "vision_encoder" if model_family == "cosmos3_edge" else "",
            },
            "model": {
                "quant_module_count": len(modules),
                "quant_module_prefix": _QUANT_MODULE_PREFIX,
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


def convert_gen_w8_bundle_to_w8a8(
    *,
    base_bundle: str | Path,
    source_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    copy_mode: str = "hardlink",
    calibration_stats: str | Path | None = None,
    calibration_alpha: float = 0.5,
) -> dict[str, Any]:
    """Reuse calibrated W4 payloads and replace generation W8A16 payloads with FP8 W8A8."""

    base_root = Path(base_bundle).expanduser().resolve()
    source_root = Path(source_checkpoint).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing RoboLab quant bundle: {output_root}")
    base_validation = validate_robolab_quant_bundle(base_root, expected_strategy="gen_branch_w8")
    manifest = base_validation["manifest"]
    model_family = detect_model_family(source_root)
    if manifest.get("model_family", model_family) != model_family:
        raise ValueError("Base bundle and BF16 source checkpoint use different Cosmos3 model families")
    targets = discover_quant_targets(source_root, "gen_branch_w8a8")
    fp8_targets = {target.source_key: target for target in targets if target.backend_class == "VllmCutlassFp8W8A8Linear"}
    stats = _load_calibration_stats(calibration_stats)
    missing_stats = sorted(target.module_name for target in fp8_targets.values() if target.module_name not in stats)
    if calibration_stats is not None and missing_stats:
        raise ValueError(f"Activation calibration is missing {len(missing_stats)} generation-branch modules")
    modules_by_source = {str(entry["source_key"]): entry for entry in manifest["modules"]}
    if set(fp8_targets) - set(modules_by_source):
        raise ValueError("Base bundle does not contain every generation-branch module required by the source checkpoint")

    temp_root = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}")
    if temp_root.exists():
        raise FileExistsError(f"Temporary bundle path already exists: {temp_root}")
    temp_root.mkdir(parents=True)
    backend_module = _load_backend_module()
    weight_map = _read_weight_map(source_root)
    source_by_shard: dict[str, list[str]] = {}
    for source_key in fp8_targets:
        source_by_shard.setdefault(weight_map[source_key], []).append(source_key)

    try:
        _copy_tree(base_root, temp_root, copy_mode=copy_mode)
        converted = 0
        for rel_path, source_keys in sorted(source_by_shard.items()):
            logger.info("Converting generation weights from source shard %s", rel_path)
            with safe_open(str(source_root / rel_path), framework="pt", device="cpu") as shard:
                for source_key in sorted(source_keys):
                    target = fp8_targets[source_key]
                    weight = shard.get_tensor(source_key).to(device=device, dtype=torch.bfloat16)
                    input_scale = None
                    if target.module_name in stats:
                        input_scale = _input_scale(
                            stats[target.module_name], size_k=int(weight.shape[1]), alpha=calibration_alpha
                        )
                    backend = backend_module.VllmCutlassFp8W8A8Linear(weight, input_scale=input_scale)
                    torch.cuda.synchronize(device)
                    payload = {
                        "backend_class": target.backend_class,
                        "bias": None,
                        "qweight_nk": backend.qweight_nk.detach().cpu().contiguous(),
                        "scale_b": backend.scale_b.detach().cpu().contiguous(),
                        "input_scale": (
                            backend.input_scale.detach().cpu().contiguous()
                            if isinstance(getattr(backend, "input_scale", None), torch.Tensor)
                            else None
                        ),
                    }
                    entry = modules_by_source[source_key]
                    tensor_rel = str(entry["tensor_file"])
                    tensor_path = temp_root / tensor_rel
                    tensor_path.unlink()
                    torch.save(payload, tensor_path)
                    for stale_key in ("group_size", "wtype_id"):
                        entry.pop(stale_key, None)
                    entry.update(
                        {
                            "backend_class": target.backend_class,
                            "format": "vllm_cutlass_fp8_w8a8",
                            "num_bits": 8,
                            "activation_bits": 8,
                            "weight_scale": "per_output_channel",
                            "activation_scale": "dynamic_per_token",
                            "size_k": int(backend.size_k),
                            "size_n": int(backend.size_n),
                        }
                    )
                    manifest["files"][tensor_rel] = _file_record(tensor_path)
                    converted += 1
                    del payload, backend, weight
                    torch.cuda.empty_cache()

        if converted != len(fp8_targets):
            raise ValueError(f"Converted {converted} FP8 modules, expected {len(fp8_targets)}")
        quantization = manifest["quantization"]
        quantization.update(
            {
                "strategy": "gen_branch_w8a8",
                "weight_only": False,
                "activation_quantization": True,
                "activation_dtype": "fp8_e4m3fn",
                "w8a8_modules": converted,
                "w8a16_modules": 0,
                "activation_calibration": "input_equalization" if stats else "dynamic_per_token_only",
                "activation_calibration_alpha": calibration_alpha if stats else None,
            }
        )
        calibration_name = str(quantization.get("calibration_stats", ""))
        quantization["calibration_stats"] = Path(calibration_name).name if calibration_name else ""
        if calibration_stats is not None:
            calibration_path = Path(calibration_stats).expanduser()
            quantization["activation_calibration_stats"] = calibration_path.name
            quantization["activation_calibration_stats_sha256"] = _sha256(calibration_path)
        source = manifest.setdefault("source", {})
        source["derived_from_bundle"] = base_root.name
        source["checkpoint_path"] = source_root.name
        manifest["created_unix"] = time.time()
        manifest_path = temp_root / "manifest.json"
        manifest_path.unlink()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_root.rename(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    validation = validate_robolab_quant_bundle(output_root, expected_strategy="gen_branch_w8a8")
    return {
        "bundle_dir": str(output_root),
        "strategy": validation["strategy"],
        "modules": validation["modules"],
        "w4_modules": int(manifest["quantization"]["w4_modules"]),
        "w8a8_modules": converted,
        "bundle_bytes": validation["bundle_bytes"],
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
    if not isinstance(quantization, dict) or not isinstance(quantization.get("activation_quantization"), bool):
        raise ValueError("RoboLab quant bundle must declare whether activations are quantized")
    activation_quantization = bool(quantization["activation_quantization"])
    weight_only = quantization.get("weight_only")
    if not isinstance(weight_only, bool) or weight_only == activation_quantization:
        raise ValueError("RoboLab quant bundle has inconsistent weight_only and activation_quantization flags")
    if activation_quantization and quantization.get("activation_dtype") != "fp8_e4m3fn":
        raise ValueError("RoboLab activation-quantized bundle must declare FP8 E4M3 activations")
    strategy = str(quantization.get("strategy", ""))
    if expected_strategy is not None and strategy != expected_strategy:
        raise ValueError(f"Expected strategy {expected_strategy!r}, found {strategy!r}")
    model = manifest.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("RoboLab quant bundle model metadata must be a dictionary")
    expected_modules = int(model.get("quant_module_count", 504))
    module_prefix = str(model.get("quant_module_prefix", _QUANT_MODULE_PREFIX))
    modules = manifest.get("modules")
    if not isinstance(modules, list) or len(modules) != expected_modules:
        raise ValueError(
            f"RoboLab quant bundle must contain {expected_modules} packed modules, found {len(modules or [])}"
        )
    names: set[str] = set()
    fp8_w8a8_modules = 0
    marlin_w8a16_modules = 0
    for entry in modules:
        if not isinstance(entry, dict):
            raise TypeError("RoboLab quant module metadata must be dictionaries")
        name = str(entry.get("name", ""))
        if not name.startswith(module_prefix) or name in names:
            raise ValueError(f"Invalid or duplicate quant module name: {name!r}")
        names.add(name)
        num_bits = int(entry.get("num_bits", 0))
        backend_class = str(entry.get("backend_class", ""))
        is_fp8_w8a8 = backend_class == "VllmCutlassFp8W8A8Linear"
        fp8_w8a8_modules += int(is_fp8_w8a8)
        is_marlin = backend_class in {"VllmGptqMarlinW4A16Linear", "VllmGptqMarlinW8A16Linear"}
        marlin_w8a16_modules += int(backend_class == "VllmGptqMarlinW8A16Linear")
        if num_bits not in {4, 8} or not (is_fp8_w8a8 or (is_marlin and f"W{num_bits}A16" in backend_class)):
            raise ValueError(f"Inconsistent quant backend metadata for {name}")
        if is_fp8_w8a8 and (
            entry.get("format") != "vllm_cutlass_fp8_w8a8" or int(entry.get("activation_bits", 0)) != 8
        ):
            raise ValueError(f"Inconsistent FP8 activation metadata for {name}")
        rel = Path(str(entry.get("tensor_file", "")))
        if rel.is_absolute() or ".." in rel.parts or not (root / rel).is_file():
            raise ValueError(f"Quant tensor path must stay under the bundle root: {rel}")
        if check_tensors:
            payload = torch.load(root / rel, map_location="cpu", weights_only=True)
            required_tensors = {"qweight_nk", "scale_b"} if is_fp8_w8a8 else {"qweight", "scales"}
            if not isinstance(payload, dict) or not required_tensors.issubset(payload):
                raise ValueError(f"Invalid packed payload: {rel}")
            if is_fp8_w8a8:
                qweight = payload["qweight_nk"]
                scale = payload["scale_b"]
                expected_weight_shape = (int(entry["size_n"]), int(entry["size_k"]))
                expected_scale_shape = (1, int(entry["size_n"]))
                if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"FP8 payload has invalid weight dtype: {rel}")
                if tuple(qweight.shape) != expected_weight_shape:
                    raise ValueError(f"FP8 payload has invalid weight shape: {rel}")
                if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
                    raise ValueError(f"FP8 payload has invalid scale dtype: {rel}")
                if (
                    tuple(scale.shape) != expected_scale_shape
                    or not torch.all(torch.isfinite(scale))
                    or not torch.all(scale > 0)
                ):
                    raise ValueError(f"FP8 payload has invalid scale values or shape: {rel}")
                input_scale = payload.get("input_scale")
                if input_scale is not None:
                    if (
                        not isinstance(input_scale, torch.Tensor)
                        or input_scale.dtype != torch.bfloat16
                        or tuple(input_scale.shape) != (int(entry["size_k"]),)
                        or not torch.all(torch.isfinite(input_scale))
                        or not torch.all(input_scale > 0)
                    ):
                        raise ValueError(f"FP8 payload has invalid input equalization scale: {rel}")
    if activation_quantization != (fp8_w8a8_modules > 0):
        raise ValueError("RoboLab quant bundle activation declaration does not match its module backends")
    if int(quantization.get("w8a8_modules", fp8_w8a8_modules)) != fp8_w8a8_modules:
        raise ValueError("RoboLab quant bundle W8A8 module count is inconsistent")
    if int(quantization.get("w8a16_modules", marlin_w8a16_modules)) != marlin_w8a16_modules:
        raise ValueError("RoboLab quant bundle W8A16 module count is inconsistent")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("RoboLab quant bundle has no runtime metadata")
    for field in ("config_file", "tokenizer_dir", "vae_path", "residual_index"):
        rel = Path(str(runtime.get(field, "")))
        if rel.is_absolute() or ".." in rel.parts or not (root / rel).exists():
            raise ValueError(f"Invalid runtime path {field}: {rel}")
    vision_encoder_dir = str(runtime.get("vision_encoder_dir", ""))
    if vision_encoder_dir:
        rel = Path(vision_encoder_dir)
        vision_weights = root / rel / "model.safetensors"
        if rel.is_absolute() or ".." in rel.parts or not vision_weights.is_file():
            raise ValueError(f"Invalid runtime path vision_encoder_dir: {rel}")
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
