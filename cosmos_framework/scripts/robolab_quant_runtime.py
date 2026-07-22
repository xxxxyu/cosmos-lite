# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Direct-load runtime support for self-contained RoboLab quant bundles."""

from __future__ import annotations

import atexit
import importlib.util
import json
import math
import os
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cosmos_framework.scripts.robolab_quant_bundle import validate_robolab_quant_bundle

_LINEAR_SHAPES_JSONL = os.environ.get("COSMOS3_LINEAR_SHAPES_JSONL", "")
_LINEAR_SHAPE_COUNTS: Counter[tuple[str, str, int, int, int, str]] = Counter()
_LINEAR_SHAPE_LOCK = threading.Lock()


def _record_quant_linear_shape(name: str, backend: nn.Module, x: torch.Tensor) -> None:
    if not _LINEAR_SHAPES_JSONL or not isinstance(x, torch.Tensor) or x.ndim == 0:
        return
    batch_tokens = int(math.prod(x.shape[:-1])) if x.ndim > 1 else 1
    key = (
        name,
        type(backend).__name__,
        batch_tokens,
        int(x.shape[-1]),
        int(backend.size_n),
        str(x.dtype).removeprefix("torch."),
    )
    with _LINEAR_SHAPE_LOCK:
        _LINEAR_SHAPE_COUNTS[key] += 1


def _flush_quant_linear_shapes() -> None:
    if not _LINEAR_SHAPES_JSONL:
        return
    with _LINEAR_SHAPE_LOCK:
        items = list(_LINEAR_SHAPE_COUNTS.items())
        _LINEAR_SHAPE_COUNTS.clear()
    if not items:
        return
    path = Path(_LINEAR_SHAPES_JSONL).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for (name, backend_class, batch_tokens, in_features, out_features, dtype), count in sorted(items):
            f.write(
                json.dumps(
                    {
                        "event": "quant_linear_shape",
                        "name": name,
                        "backend_class": backend_class,
                        "batch_tokens": batch_tokens,
                        "in_features": in_features,
                        "out_features": out_features,
                        "input_dtype": dtype,
                        "count": count,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


atexit.register(_flush_quant_linear_shapes)


class QuantLinearWithOptionalBias(nn.Module):
    def __init__(self, backend: nn.Module, bias: torch.Tensor | None, *, name: str) -> None:
        super().__init__()
        self.backend = backend
        self.quant_module_name = name
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _record_quant_linear_shape(self.quant_module_name, self.backend, x)
        output = self.backend(x)
        return output if self.bias is None else output + self.bias

    def _apply(self, fn: Any, recurse: bool = True) -> "QuantLinearWithOptionalBias":
        # The packed buffers are already on their deployment device when the
        # parent network calls to_empty(). Reapplying to_empty would destroy them.
        return self


def _load_backend_module() -> Any:
    module_path = Path(__file__).resolve().parent / "quant_backend_microbench.py"
    spec = importlib.util.spec_from_file_location("cosmos3_robolab_quant_runtime_backends", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import quant backend module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _make_backend(
    backend_module: Any,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    backend_class = str(metadata["backend_class"])
    if backend_class == "VllmGptqMarlinW4A16Linear":
        cls = backend_module.VllmGptqMarlinW4A16Linear
    elif backend_class == "VllmGptqMarlinW8A16Linear":
        cls = backend_module.VllmGptqMarlinW8A16Linear
    else:
        raise ValueError(f"Unsupported RoboLab direct-load backend: {backend_class}")

    import vllm._C  # noqa: F401

    backend = cls.__new__(cls)
    nn.Module.__init__(backend)
    backend.size_k = int(metadata["size_k"])
    backend.size_n = int(metadata["size_n"])
    backend.group_size = int(metadata["group_size"])
    backend.num_bits = int(metadata["num_bits"])
    backend.wtype_id = int(metadata["wtype_id"])
    backend.quant_weight_l1_mean = float("nan")
    backend.quant_weight_linf = float("nan")
    backend.register_buffer("qweight", payload["qweight"].to(device=device).contiguous(), persistent=False)
    backend.register_buffer("scales", payload["scales"].to(device=device).contiguous(), persistent=False)
    input_scale = payload.get("input_scale")
    if isinstance(input_scale, torch.Tensor):
        backend.register_buffer(
            "input_scale", input_scale.to(device=device, dtype=torch.bfloat16).contiguous(), persistent=False
        )
    else:
        backend.input_scale = None
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    backend.register_buffer("workspace", torch.zeros(sms, dtype=torch.int, device=device), persistent=False)
    backend.register_buffer("empty", torch.empty(0, dtype=torch.int, device=device), persistent=False)
    return backend


class RobolabDirectQuantLoader:
    """Process-lifetime patch that replaces Linear modules before CUDA materialization."""

    def __init__(self, bundle_dir: str | Path) -> None:
        self.root = Path(bundle_dir).expanduser().resolve()
        self.validation = validate_robolab_quant_bundle(self.root)
        self.manifest = self.validation["manifest"]
        self._applied = False
        self._original_from_pretrained: Any = None
        self._original_to_empty: Any = None

    def _load_packed_modules(self, network: nn.Module) -> None:
        if self._applied:
            return
        backend_module = _load_backend_module()
        device = torch.device("cuda", torch.cuda.current_device())
        for metadata in self.manifest["modules"]:
            name = str(metadata["name"])
            set_name = name.removeprefix("net.")
            payload = torch.load(self.root / str(metadata["tensor_file"]), map_location="cpu", weights_only=True)
            backend = _make_backend(backend_module, metadata, payload, device)
            bias = payload.get("bias")
            wrapped = QuantLinearWithOptionalBias(
                backend,
                bias.to(device=device) if isinstance(bias, torch.Tensor) else None,
                name=name,
            )
            _set_module(network, set_name, wrapped)
            del payload
        self._applied = True

    def install(self) -> None:
        from cosmos_framework.inference import model as inference_model

        if self._original_from_pretrained is not None:
            raise RuntimeError("RoboLab direct quant loader is already installed")
        expected_root = self.root
        loader = self
        self._original_from_pretrained = inference_model.Cosmos3OmniModel.from_pretrained_dcp.__func__
        self._original_to_empty = torch.nn.Module.to_empty
        original_from_pretrained = self._original_from_pretrained
        original_to_empty = self._original_to_empty

        def to_empty_with_quant(module: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
            if type(module).__name__ == "Cosmos3VFMNetwork" and not loader._applied:
                loader._load_packed_modules(module)
            return original_to_empty(module, *args, **kwargs)

        def from_pretrained_dcp_direct(
            cls: type[Any],
            checkpoint_path: Path,
            config: Any = None,
            parallelism_config: Any = None,
            compile_config: Any = None,
            quantization_config: Any = None,
        ) -> Any:
            checkpoint_root = Path(checkpoint_path).expanduser().resolve()
            if checkpoint_root != expected_root:
                return original_from_pretrained(
                    cls,
                    checkpoint_path,
                    config=config,
                    parallelism_config=parallelism_config,
                    compile_config=compile_config,
                    quantization_config=quantization_config,
                )
            if config is None:
                raise ValueError("A RoboLab quant bundle must use its bundled runtime config")
            if parallelism_config is None:
                parallelism_config = inference_model.ParallelismConfig()
            if compile_config is None:
                compile_config = inference_model.CompileConfig()
            if quantization_config is None:
                quantization_config = inference_model.QuantizationConfig()
            config.parallelism = inference_model.attrs.asdict(parallelism_config)
            config.compile = inference_model.attrs.asdict(compile_config)
            config.quantization = inference_model.attrs.asdict(quantization_config)
            model = cls(config)
            language_model = getattr(getattr(model.model, "net", None), "language_model", None)
            if language_model is not None:
                language_model._local_checkpoint_dir = str(expected_root)
            if not loader._applied:
                raise RuntimeError(
                    "Packed modules were not installed before model materialization; refusing a hidden BF16 fallback"
                )
            state_dict = inference_model.get_model_state_dict(model.model)
            storage_reader = inference_model.HuggingFaceStorageReader(str(expected_root))
            inference_model.dcp.load(state_dict=state_dict, storage_reader=storage_reader)
            return model

        torch.nn.Module.to_empty = to_empty_with_quant
        inference_model.Cosmos3OmniModel.from_pretrained_dcp = classmethod(from_pretrained_dcp_direct)

    def uninstall(self) -> None:
        if self._original_from_pretrained is None:
            return
        from cosmos_framework.inference import model as inference_model

        inference_model.Cosmos3OmniModel.from_pretrained_dcp = classmethod(self._original_from_pretrained)
        torch.nn.Module.to_empty = self._original_to_empty
        self._original_from_pretrained = None
        self._original_to_empty = None
