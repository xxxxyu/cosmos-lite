#!/usr/bin/env python3
"""Validate the minimal Cosmos3 quantized policy runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()

    modules = {
        name: importlib.import_module(name)
        for name in (
            "cosmos_framework",
            "cosmos_framework.model.attention.flash2",
            "cosmos_framework.model.attention.natten",
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
                "torch",
                "vllm",
            )
        },
        "cuda": cuda,
        "marlin_gemm": True,
        "module_paths": module_paths,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
