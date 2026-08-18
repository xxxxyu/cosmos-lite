# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Validated deployment configuration for quantized Cosmos3 RoboLab policies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pydantic
import yaml

ModelFamily = Literal["cosmos3_nano", "cosmos3_edge"]
PublicStrategy = Literal["full_w8", "gen_branch_w8a8"]
BackendPolicy = Literal["strict", "best_available"]
AttentionBackend = Literal["flash_attention_2", "sage_attention"]
SagePvMode = Literal["fp8_fp32_fp32", "fp16_fp32"]


class DeploymentModelConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    family: ModelFamily
    bundle_dir: Path | None = None
    strategy: PublicStrategy


class DeploymentSamplingConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    guidance: float = 3.0
    denoise_steps: int = 2
    shift: float = 5.0
    seed: int = 0
    deterministic_seed: bool = True

    @pydantic.field_validator("denoise_steps")
    @classmethod
    def _positive_denoise_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("denoise_steps must be positive")
        return value


class DeploymentRuntimeConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    backend_policy: BackendPolicy = "strict"
    torch_compile: bool = False
    cuda_graphs: bool = False
    compiled_region: Literal["all", "language"] = "language"
    compile_dynamic: bool = True
    fp8_projection_fusion: Literal["none", "shared"] = "none"
    fp8_gemm_backend: Literal["cutlass", "triton_sm89"] = "cutlass"
    attention_backend: AttentionBackend = "flash_attention_2"
    sage_pv: SagePvMode = "fp8_fp32_fp32"
    condition_kv_cache: bool = False
    sparse_video_transform: bool = True
    vae_encode_chunk_frames: int = 8

    @pydantic.model_validator(mode="after")
    def _validate_runtime_combinations(self) -> "DeploymentRuntimeConfig":
        if self.cuda_graphs and not self.torch_compile:
            raise ValueError("cuda_graphs requires torch_compile")
        if self.fp8_gemm_backend == "triton_sm89" and not self.torch_compile:
            raise ValueError("triton_sm89 requires torch_compile")
        if self.vae_encode_chunk_frames <= 0 or self.vae_encode_chunk_frames % 4 != 0:
            raise ValueError("vae_encode_chunk_frames must be a positive multiple of 4")
        return self


class DeploymentServerConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8000
    output_dir: Path = Path("outputs/robolab_deploy")
    profile_jsonl: Path | None = None
    format_prompt_as_json: bool = False

    @pydantic.field_validator("port")
    @classmethod
    def _valid_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be in [1, 65535]")
        return value


class RobolabDeploymentConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profile: str
    model: DeploymentModelConfig
    sampling: DeploymentSamplingConfig = pydantic.Field(default_factory=DeploymentSamplingConfig)
    runtime: DeploymentRuntimeConfig = pydantic.Field(default_factory=DeploymentRuntimeConfig)
    server: DeploymentServerConfig = pydantic.Field(default_factory=DeploymentServerConfig)

    @pydantic.model_validator(mode="after")
    def _validate_public_profile(self) -> "RobolabDeploymentConfig":
        if self.model.strategy == "full_w8" and self.runtime.fp8_projection_fusion != "none":
            raise ValueError("full_w8 does not support FP8 projection fusion")
        if self.model.strategy == "gen_branch_w8a8" and self.runtime.fp8_projection_fusion != "shared":
            raise ValueError("gen_branch_w8a8 public profiles require shared FP8 projection fusion")
        return self


class RuntimeProbe(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    cuda_available: bool
    gpu_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    sageattention_installed: bool = False
    sageattention_sm89: bool = False
    triton_installed: bool = False


class DeploymentResolution(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    requested: RobolabDeploymentConfig
    effective: RobolabDeploymentConfig
    runtime_probe: RuntimeProbe
    fallback_decisions: list[str] = pydantic.Field(default_factory=list)
    bundle_manifest: dict[str, Any]
    bundle_manifest_sha256: str


def load_deployment_config(path: str | Path) -> RobolabDeploymentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Deployment config must contain a YAML mapping: {config_path}")
    return RobolabDeploymentConfig.model_validate(raw)


def apply_cli_overrides(
    config: RobolabDeploymentConfig,
    *,
    bundle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> RobolabDeploymentConfig:
    model = config.model
    if bundle_dir is not None:
        model = model.model_copy(update={"bundle_dir": Path(bundle_dir)})
    server = config.server
    server_updates: dict[str, Any] = {}
    if output_dir is not None:
        server_updates["output_dir"] = Path(output_dir)
        server_updates["profile_jsonl"] = None
    if host is not None:
        server_updates["host"] = host
    if port is not None:
        server_updates["port"] = port
    if server_updates:
        server = server.model_copy(update=server_updates)
    return RobolabDeploymentConfig.model_validate(
        config.model_dump(mode="python") | {"model": model.model_dump(), "server": server.model_dump()}
    )


def probe_runtime() -> RuntimeProbe:
    import torch

    cuda_available = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    sageattention_installed = importlib.util.find_spec("sageattention") is not None
    sageattention_sm89 = False
    if sageattention_installed:
        try:
            from sageattention.core import SM89_ENABLED

            sageattention_sm89 = bool(SM89_ENABLED)
        except Exception:
            sageattention_sm89 = False
    return RuntimeProbe(
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compute_capability=capability,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        sageattention_installed=sageattention_installed,
        sageattention_sm89=sageattention_sm89,
        triton_installed=importlib.util.find_spec("triton") is not None,
    )


def _load_bundle_manifest(bundle_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Quant bundle has no manifest.json: {bundle_dir}")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError(f"Quant bundle manifest must contain a JSON object: {manifest_path}")
    return manifest, hashlib.sha256(payload).hexdigest()


def resolve_deployment_config(
    config: RobolabDeploymentConfig,
    *,
    probe: RuntimeProbe | None = None,
) -> DeploymentResolution:
    if config.model.bundle_dir is None:
        raise ValueError("model.bundle_dir is required; set it in YAML or pass --bundle-dir")
    bundle_dir = config.model.bundle_dir.expanduser().resolve()
    requested = config.model_copy(update={"model": config.model.model_copy(update={"bundle_dir": bundle_dir})})
    manifest, manifest_sha256 = _load_bundle_manifest(bundle_dir)
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Quant bundle manifest has no quantization metadata")
    actual_strategy = quantization.get("strategy")
    if actual_strategy != requested.model.strategy:
        raise ValueError(f"Profile expects strategy {requested.model.strategy!r}, bundle contains {actual_strategy!r}")
    actual_family = manifest.get("model_family")
    if actual_family != requested.model.family:
        raise ValueError(f"Profile expects model family {requested.model.family!r}, bundle contains {actual_family!r}")

    runtime_probe = probe or probe_runtime()
    if not runtime_probe.cuda_available or runtime_probe.compute_capability is None:
        raise RuntimeError("A CUDA GPU is required for RoboLab policy deployment")
    capability = runtime_probe.compute_capability
    capability_int = capability[0] * 10 + capability[1]
    effective_runtime = requested.runtime
    fallback_decisions: list[str] = []

    if requested.model.strategy == "gen_branch_w8a8" and capability_int < 89:
        raise RuntimeError(
            f"GenW8A8 requires native FP8 Tensor Cores (SM89+); detected SM{capability_int}. "
            "Use the W8A16 profile on this GPU."
        )

    if effective_runtime.fp8_gemm_backend == "triton_sm89":
        reason = None
        if capability_int != 89:
            reason = f"Triton SM89 FP8 GEMM requires SM89, detected SM{capability_int}"
        elif not runtime_probe.triton_installed:
            reason = "Triton SM89 FP8 GEMM requires the triton package"
        if reason is not None:
            if effective_runtime.backend_policy == "strict":
                raise RuntimeError(reason)
            fallback_decisions.append(f"{reason}; using CUTLASS FP8 GEMM")
            effective_runtime = effective_runtime.model_copy(update={"fp8_gemm_backend": "cutlass"})

    if effective_runtime.attention_backend == "sage_attention":
        reason = None
        if capability_int != 89:
            reason = f"Cosmos Lite SageAttention requires SM89, detected SM{capability_int}"
        elif not runtime_probe.sageattention_installed:
            reason = "SageAttention is not installed"
        elif not runtime_probe.sageattention_sm89:
            reason = "SageAttention was installed without its SM89 kernel"
        if reason is not None:
            if effective_runtime.backend_policy == "strict":
                raise RuntimeError(
                    f"{reason}. Run examples/quantized_robot_policy/install_sage_attention.sh "
                    "or choose a compatible deployment profile."
                )
            fallback_decisions.append(f"{reason}; using FlashAttention 2")
            effective_runtime = effective_runtime.model_copy(update={"attention_backend": "flash_attention_2"})

    effective = requested.model_copy(update={"runtime": effective_runtime})
    return DeploymentResolution(
        requested=requested,
        effective=effective,
        runtime_probe=runtime_probe,
        fallback_decisions=fallback_decisions,
        bundle_manifest=manifest,
        bundle_manifest_sha256=manifest_sha256,
    )


def configure_backend_environment(config: RobolabDeploymentConfig) -> None:
    os.environ.setdefault("COSMOS_TRAINING", "0")
    os.environ.pop("COSMOS3_SAGE_ATTENTION", None)
    os.environ.pop("COSMOS3_SAGE_PV", None)
    if config.runtime.attention_backend == "sage_attention":
        os.environ["COSMOS3_SAGE_ATTENTION"] = "1"
        os.environ["COSMOS3_SAGE_PV"] = config.runtime.sage_pv


def server_argument_values(config: RobolabDeploymentConfig) -> dict[str, Any]:
    if config.model.bundle_dir is None:
        raise ValueError("Resolved deployment config has no bundle_dir")
    output_dir = config.server.output_dir.expanduser().resolve()
    profile_jsonl = config.server.profile_jsonl
    if profile_jsonl is None:
        profile_jsonl = output_dir / "profile.jsonl"
    return {
        "quant_import_dir": str(config.model.bundle_dir),
        "host": config.server.host,
        "port": config.server.port,
        "output_dir": output_dir,
        "profile_jsonl": profile_jsonl.expanduser().resolve(),
        "guidance": config.sampling.guidance,
        "num_steps": config.sampling.denoise_steps,
        "shift": config.sampling.shift,
        "seed": config.sampling.seed,
        "deterministic_seed": config.sampling.deterministic_seed,
        "use_torch_compile": config.runtime.torch_compile,
        "use_cuda_graphs": config.runtime.cuda_graphs,
        "compiled_region": config.runtime.compiled_region,
        "compile_dynamic": config.runtime.compile_dynamic,
        "fp8_projection_fusion": config.runtime.fp8_projection_fusion,
        "fp8_gemm_backend": config.runtime.fp8_gemm_backend,
        "condition_kv_cache": config.runtime.condition_kv_cache,
        "sparse_video_transform": config.runtime.sparse_video_transform,
        "vae_encode_chunk_frames": config.runtime.vae_encode_chunk_frames,
        "format_prompt_as_json": config.server.format_prompt_as_json,
    }


def _repository_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {"root": str(repo_root), "revision": revision, "dirty": dirty}


def write_resolved_deployment_record(
    resolution: DeploymentResolution,
    *,
    config_file: str | Path,
) -> Path:
    effective = resolution.effective
    output_dir = effective.server.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "resolved_deployment_config.json"
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_file": str(Path(config_file).expanduser().resolve()),
        "requested": resolution.requested.model_dump(mode="json"),
        "effective": effective.model_dump(mode="json"),
        "fallback_decisions": resolution.fallback_decisions,
        "bundle": {
            "path": str(effective.model.bundle_dir),
            "manifest_sha256": resolution.bundle_manifest_sha256,
            "model_family": resolution.bundle_manifest.get("model_family"),
            "strategy": resolution.bundle_manifest.get("quantization", {}).get("strategy"),
        },
        "runtime_probe": resolution.runtime_probe.model_dump(mode="json"),
        "repository": _repository_state(),
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
