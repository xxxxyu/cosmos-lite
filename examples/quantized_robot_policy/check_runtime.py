#!/usr/bin/env python3
"""Validate the minimal Cosmos3 quantized policy runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> None:
    # This entry point validates the inference-only runtime. Set the feature
    # flag before importing Cosmos modules so fresh shells do not require the
    # optional training and cloud-storage stack.
    os.environ.setdefault("COSMOS_TRAINING", "0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--deployment-config", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--checkpoint-path")
    args = parser.parse_args()
    if args.deployment_config is None and (args.bundle_dir is not None or args.checkpoint_path is not None):
        parser.error("--deployment-config is required with --bundle-dir or --checkpoint-path")

    modules = {
        name: importlib.import_module(name)
        for name in (
            "cosmos_framework",
            "cosmos_framework.model.attention.flash2",
            "cosmos_framework.model.attention.natten",
            "cosmos_framework.scripts.action_policy_server_robolab",
            "cosmos_framework.scripts.action_policy_server_robolab_deploy",
            "openpi_client",
            "openpi_server",
            "qwen_vl_utils",
            "transformers_cosmos3",
            "vllm._C",
            "vllm._custom_ops",
            "vllm_cosmos3",
            "zmq",
        )
    }

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    required_marlin_ops = ("gptq_marlin_repack", "marlin_gemm")
    missing_marlin_ops = [name for name in required_marlin_ops if not hasattr(torch.ops._C, name)]
    if missing_marlin_ops:
        raise RuntimeError(f"vLLM loaded but required Marlin ops are unavailable: {missing_marlin_ops}")
    if not modules["cosmos_framework.model.attention.flash2"].FLASH2_SUPPORTED:
        raise RuntimeError("FlashAttention 2 is not supported by the installed CUDA runtime")
    if not modules["cosmos_framework.model.attention.natten"].NATTEN_SUPPORTED:
        raise RuntimeError("NATTEN is not supported by the installed CUDA runtime")

    repo_root = args.repo_root.resolve() if args.repo_root else None
    module_paths: dict[str, str] = {}
    for name, module in modules.items():
        value = getattr(module, "__file__", None)
        module_paths[name] = str(Path(value).resolve()) if value else "<extension>"
    if repo_root is not None:
        for name in ("cosmos_framework", "transformers_cosmos3", "vllm_cosmos3"):
            path = Path(module_paths[name])
            if not path.is_relative_to(repo_root):
                raise RuntimeError(f"{name} imported from stale checkout: {path}")

    cuda: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        cuda.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        )

    result = {
        "status": "ok",
        "versions": {
            name: _distribution_version(name)
            for name in (
                "cosmos-framework",
                "numpy",
                "flash-attn",
                "natten",
                "openpi-client",
                "openpi-server",
                "sageattention",
                "torch",
                "triton",
                "vllm",
            )
        },
        "cuda": cuda,
        "marlin_gemm": True,
        "module_paths": module_paths,
    }
    if args.deployment_config is not None:
        from cosmos_framework.scripts.robolab_deployment_config import (
            apply_cli_overrides,
            load_deployment_config,
            resolve_deployment_config,
        )

        requested = apply_cli_overrides(
            load_deployment_config(args.deployment_config),
            bundle_dir=args.bundle_dir,
            checkpoint_path=args.checkpoint_path,
        )
        resolution = resolve_deployment_config(requested)
        result["deployment"] = {
            "status": "compatible",
            "profile": resolution.effective.profile,
            "model_family": resolution.effective.model.family,
            "strategy": resolution.effective.model.strategy,
            "artifact": resolution.effective.model.artifact,
            "bundle_manifest_sha256": resolution.bundle_manifest_sha256,
            "effective_runtime": resolution.effective.runtime.model_dump(mode="json"),
            "fallback_decisions": resolution.fallback_decisions,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
