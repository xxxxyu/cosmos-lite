# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""ZMQ policy server that evaluates a Cosmos3 RoboCasa365 action checkpoint with RLDX.

RLDX's RoboCasa365 rollout harness can call an external policy through
``rldx.policy.server_client.PolicyClient``. This server speaks that protocol and
adapts native RoboCasa365 observations to the Cosmos3 action SFT transform.
"""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import atexit
import ipaddress
import importlib.util
import io
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
import zmq
from omegaconf import OmegaConf
from torch import nn

try:
    from cosmos_framework.data.generator.action.domain_utils import get_domain_id
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
except ModuleNotFoundError:
    from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
    from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
    from cosmos_framework.data.vfm.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from cosmos_framework.scripts.robocasa365_quant_bundle import (
    BUNDLE_SCHEMA_VERSION,
    materialize_bundle_config,
)
from cosmos_framework.scripts.robocasa365_quant_pipeline import validate_quant_artifact
from cosmos_framework.utils import log

_DEFAULT_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robocasa365_rldx"
_ACTION_KEYS = (
    "action.base_motion",
    "action.control_mode",
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
)
_VIDEO_KEY_CANDIDATES = {
    "left": ("video.robot0_agentview_left", "video.res256_image_side_0"),
    "right": ("video.robot0_agentview_right", "video.res256_image_side_1"),
    "wrist": ("video.robot0_eye_in_hand", "video.res256_image_wrist_0"),
}
_LANGUAGE_KEY_CANDIDATES = (
    "annotation.human.task_description",
    "annotation.human.action.task_description",
)
_CONCAT_VIEW_DESCRIPTION = (
    "The top row is from the robot wrist camera. "
    "The bottom row contains two third-person agent views from the left and right sides."
)

_PROFILE_JSONL = os.environ.get("COSMOS3_PROFILE_JSONL", "")
_LINEAR_SHAPES_JSONL = os.environ.get("COSMOS3_LINEAR_SHAPES_JSONL", "")
_DYNAMO_DISABLE_QUANT_LINEAR = os.environ.get("COSMOS3_DYNAMO_DISABLE_QUANT_LINEAR", "0") == "1"
_PROFILE_LOCK = threading.Lock()
_LINEAR_SHAPE_LOCK = threading.Lock()
_LINEAR_SHAPE_COUNTS: Counter[tuple[str, str, int, int, int, str]] = Counter()


def _cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "cuda_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "cuda_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "cuda_max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def _profile_event(event: str, **fields: Any) -> None:
    if not _PROFILE_JSONL:
        return
    record = {
        "event": event,
        "ts": time.time(),
        **fields,
        **_cuda_mem(),
    }
    with _PROFILE_LOCK:
        with open(_PROFILE_JSONL, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _record_quant_linear_shape(name: str, backend_class: str, x: torch.Tensor, out_features: int) -> None:
    if not _LINEAR_SHAPES_JSONL:
        return
    if not isinstance(x, torch.Tensor) or x.ndim == 0:
        return
    in_features = int(x.shape[-1])
    backend = backend_class
    batch_tokens = int(math.prod(x.shape[:-1])) if x.ndim > 1 else 1
    key = (name, backend, batch_tokens, in_features, int(out_features), str(x.dtype).replace("torch.", ""))
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
    path = Path(_LINEAR_SHAPES_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
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


def _sync_elapsed_ms(start: float) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _maybe_disable_dynamo(fn: Any) -> Any:
    if not _DYNAMO_DISABLE_QUANT_LINEAR:
        return fn
    try:
        return torch._dynamo.disable(fn)
    except Exception:
        return fn


@dataclass(frozen=True)
class ServerConfig:
    checkpoint_path: str
    config_file: str
    output_dir: str
    host: str
    port: int
    sampler: str
    seed: int
    deterministic_seed: bool
    guidance: float
    num_steps: int
    shift: float
    conditioning_fps: float
    resolution: str
    action_chunk_size: int
    served_action_steps: int
    raw_action_dim: int
    max_action_dim: int
    camera_size: int
    guardrails: bool
    use_torch_compile: bool
    torchao_quant: str
    torchao_target_prefix: str
    quant_backend: str
    quant_target_prefix: str
    quant_plan_file: str
    quant_calib_capture_dir: str
    quant_calib_skip: int
    quant_calib_limit: int
    quant_calib_alpha: float
    quant_calibration_stats_output: str
    quant_export_dir: str
    quant_import_dir: str
    allow_legacy_quant_artifact: bool


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            if tensor.dtype in (torch.bfloat16, torch.float16):
                tensor = tensor.float()
            obj = tensor.numpy()
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class QuantLinearWithOptionalBias(nn.Module):
    def __init__(self, backend: nn.Module, bias: torch.Tensor | None, name: str = "") -> None:
        super().__init__()
        self.backend = backend
        self.profile_name = name
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)

    @_maybe_disable_dynamo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _LINEAR_SHAPES_JSONL:
            _record_quant_linear_shape(
                self.profile_name,
                type(self.backend).__name__,
                x,
                int(getattr(self.backend, "size_n", self.bias.numel() if self.bias is not None else -1)),
            )
        y = self.backend(x)
        if self.bias is not None:
            y = y + self.bias
        return y

    def _apply(self, fn: Any, recurse: bool = True) -> "QuantLinearWithOptionalBias":
        # These modules own already-packed runtime buffers. During direct-load
        # pre-materialization they are inserted before the parent network's
        # to_empty(cuda), so ordinary _apply would replace qweight/scales with
        # empty tensors. Keep them untouched.
        return self


def _infer_config_file(checkpoint_path: str) -> str:
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    candidate = checkpoint_dir.parent.parent / "config.yaml"
    if candidate.is_file():
        return str(candidate)
    raise ValueError(
        "--config-file was not provided and could not be inferred from "
        f"checkpoint path {checkpoint_path!r}"
    )


_CONFIG_COMPAT_REPLACEMENTS = (
    ("cosmos_framework.data.vfm", "cosmos_framework.data.generator"),
    ("cosmos_framework.model.vfm", "cosmos_framework.model.generator"),
    (
        "cosmos_framework/model/vfm/vlm/qwen3_vl",
        "cosmos_framework/model/generator/reasoner/qwen3_vl",
    ),
    (
        "cosmos_framework.configs.base.defaults.vlm",
        "cosmos_framework.configs.base.defaults.reasoner",
    ),
)


def _materialize_compat_config(config_file: str, output_dir: str) -> str:
    source = Path(config_file).expanduser()
    text = source.read_text()
    patched = text
    applied: list[dict[str, str]] = []
    for old, new in _CONFIG_COMPAT_REPLACEMENTS:
        if old in patched:
            patched = patched.replace(old, new)
            applied.append({"old": old, "new": new})
    if not applied:
        return str(source)

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    compat_file = out_dir / f"{source.stem}.compat{source.suffix or '.yaml'}"
    compat_file.write_text(patched)
    log.info(
        "[robocasa365-rldx-server] materialized compatible config "
        f"source={str(source)!r} compat={str(compat_file)!r} replacements={len(applied)}"
    )
    _profile_event(
        "compat_config",
        source=str(source),
        compat=str(compat_file),
        replacements=applied,
    )
    return str(compat_file)


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]
        else:
            data_batch[key] = [value]
    return data_batch


def _load_dataset_config(config_file: str) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
    assert isinstance(cfg, dict)
    try:
        dataset_cfg = cfg["dataloader_train"]["dataloader"]["datasets"]["robocasa365"]["dataset"]
    except KeyError as exc:
        raise KeyError(f"Could not find robocasa365 dataset config in {config_file}") from exc
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"robocasa365 dataset config must be a dict, got {type(dataset_cfg)}")
    return dataset_cfg


def _modality_config(delta_indices: list[int], modality_keys: list[str]) -> dict[str, Any]:
    return {
        "__ModalityConfig_class__": True,
        "as_json": {
            "delta_indices": delta_indices,
            "modality_keys": modality_keys,
            "action_configs": None,
        },
    }


def _batch_size(observation: dict[str, Any]) -> int:
    for candidates in _VIDEO_KEY_CANDIDATES.values():
        for key in candidates:
            value = observation.get(key)
            if isinstance(value, np.ndarray) and value.ndim >= 5:
                return int(value.shape[0])
    raise ValueError("Observation does not contain any batched RoboCasa365 video key")


def _extract_frame(observation: dict[str, Any], candidates: tuple[str, ...], batch_idx: int) -> np.ndarray:
    for key in candidates:
        value = observation.get(key)
        if isinstance(value, np.ndarray):
            if value.ndim == 5:
                frame = value[batch_idx, -1]
            elif value.ndim == 4:
                frame = value[-1]
            elif value.ndim == 3:
                frame = value
            else:
                continue
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Video key {key!r} must resolve to an HWC RGB frame, got {frame.shape}")
            return frame
    raise KeyError(f"Missing any of video keys {candidates}")


def _extract_text_item(value: Any, batch_idx: int) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _extract_text_item(value.item(), batch_idx)
        return _extract_text_item(value[batch_idx], 0)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        return _extract_text_item(value[batch_idx], 0)
    return str(value)


def _extract_prompt(observation: dict[str, Any], batch_idx: int) -> str:
    for key in _LANGUAGE_KEY_CANDIDATES:
        if key in observation:
            prompt = _extract_text_item(observation[key], batch_idx)
            if prompt:
                return prompt
    raise KeyError(f"Missing any of language keys {_LANGUAGE_KEY_CANDIDATES}")


def _to_chw_uint8(frame: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
    if tensor.shape[-2:] == (size, size):
        return tensor
    resized = F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.round().clamp_(0, 255).to(torch.uint8)


def _compose_concat_view(left: np.ndarray, right: np.ndarray, wrist: np.ndarray, camera_size: int) -> torch.Tensor:
    wrist_chw = _to_chw_uint8(wrist, camera_size)
    left_chw = _to_chw_uint8(left, camera_size)
    right_chw = _to_chw_uint8(right, camera_size)
    half = camera_size // 2
    left_half = F.interpolate(left_chw.unsqueeze(0).float(), size=(half, half), mode="bilinear", align_corners=False)[
        0
    ].round().clamp_(0, 255).to(torch.uint8)
    right_half = F.interpolate(
        right_chw.unsqueeze(0).float(),
        size=(half, half),
        mode="bilinear",
        align_corners=False,
    )[0].round().clamp_(0, 255).to(torch.uint8)
    bottom = torch.cat([left_half, right_half], dim=-1)
    return torch.cat([wrist_chw, bottom], dim=-2)


class CosmosRoboCasa365Policy:
    def __init__(self, cfg: ServerConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Cosmos3 action policy inference.")
        self.cfg = cfg
        init_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        maybe_init_distributed()
        self._runtime_config_file = _materialize_compat_config(cfg.config_file, cfg.output_dir)
        setup_start = time.perf_counter()
        setup_args = self._build_setup_args(cfg, self._runtime_config_file)
        _profile_event("build_setup_args", elapsed_ms=(time.perf_counter() - setup_start) * 1000.0)
        log.info(
            "[robocasa365-rldx-server] loading model "
            f"checkpoint_path={setup_args.checkpoint_path!r} config_file={setup_args.config_file!r}"
        )
        if cfg.quant_import_dir:
            self._install_direct_quant_loader_patch(cfg)
        load_start = time.perf_counter()
        self.pipe = OmniInference.create(setup_args)
        _profile_event("model_load", elapsed_ms=_sync_elapsed_ms(load_start))
        self.model = self.pipe.model
        self.model.eval()
        assert isinstance(self.pipe.setup_args, OmniSetupArgs)

        transform_start = time.perf_counter()
        dataset_cfg = _load_dataset_config(self._runtime_config_file)
        tokenizer_config = dataset_cfg.get("tokenizer_config")
        self.transform = ActionTransformPipeline(
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=0.0,
            max_action_dim=cfg.max_action_dim,
            append_viewpoint_info=bool(dataset_cfg.get("append_viewpoint_info", True)),
            append_duration_fps_timestamps=bool(dataset_cfg.get("append_duration_fps_timestamps", True)),
            append_resolution_info=bool(dataset_cfg.get("append_resolution_info", True)),
            append_idle_frames=False,
        )
        _profile_event("transform_init", elapsed_ms=(time.perf_counter() - transform_start) * 1000.0)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(cfg.seed)
        self._clip_warning_count = 0
        if cfg.torchao_quant != "none":
            self._apply_torchao_quantization(cfg)
        if cfg.quant_backend != "none" or cfg.quant_plan_file:
            self._apply_linear_backend_replacement(cfg)
        if cfg.quant_export_dir:
            self._export_quant_artifacts(cfg)
        if cfg.quant_calibration_stats_output:
            self._collect_direct_quant_calibration(cfg)
        _profile_event("policy_init", elapsed_ms=_sync_elapsed_ms(init_start))
        log.info(
            "[robocasa365-rldx-server] ready "
            f"resolution={cfg.resolution} fps={cfg.conditioning_fps} "
            f"chunk={cfg.action_chunk_size} served_steps={cfg.served_action_steps} "
            f"raw_action_dim={cfg.raw_action_dim} max_action_dim={cfg.max_action_dim}"
        )

    def _apply_torchao_quantization(self, cfg: ServerConfig) -> None:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_

        quant_start = time.perf_counter()
        target_prefix = cfg.torchao_target_prefix

        def filter_fn(module: nn.Module, name: str) -> bool:
            return isinstance(module, nn.Linear) and name.startswith(target_prefix)

        if cfg.torchao_quant == "int8wo":
            quant_config = Int8WeightOnlyConfig()
        elif cfg.torchao_quant == "int4wo":
            quant_config = Int4WeightOnlyConfig(group_size=128)
        else:
            raise ValueError(f"Unsupported torchao quantization mode: {cfg.torchao_quant}")

        log.info(
            "[robocasa365-rldx-server] applying torchao quantization "
            f"mode={cfg.torchao_quant} target_prefix={target_prefix!r}"
        )
        quantize_(self.model, quant_config, filter_fn=filter_fn)
        _profile_event(
            "torchao_quantize",
            mode=cfg.torchao_quant,
            target_prefix=target_prefix,
            elapsed_ms=_sync_elapsed_ms(quant_start),
        )

    def _load_quant_backend_module(self) -> Any:
        module_path = Path(__file__).resolve().parent / "quant_backend_microbench.py"
        spec = importlib.util.spec_from_file_location("cosmos3_quant_backend_microbench", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import quant backend module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _load_quant_plan(self, cfg: ServerConfig) -> list[dict[str, str]]:
        if not cfg.quant_plan_file:
            return [{"prefix": cfg.quant_target_prefix, "backend": cfg.quant_backend}]
        plan_path = Path(cfg.quant_plan_file).expanduser()
        plan = json.loads(plan_path.read_text())
        if not isinstance(plan, list):
            raise TypeError(f"quant plan must be a list, got {type(plan)}")
        rows: list[dict[str, str]] = []
        for item in plan:
            if not isinstance(item, dict):
                raise TypeError(f"quant plan entries must be dicts, got {type(item)}")
            prefix = str(item.get("prefix", ""))
            backend = str(item.get("backend", "none"))
            name_regex = str(item.get("name_regex", ""))
            exclude_regex = str(item.get("exclude_regex", ""))
            rows.append(
                {
                    "prefix": prefix,
                    "name_regex": name_regex,
                    "backend": backend,
                    "exclude_regex": exclude_regex,
                }
            )
        return rows

    @staticmethod
    def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    def _apply_linear_backend_replacement(self, cfg: ServerConfig) -> None:
        quant_start = time.perf_counter()
        backend_module = self._load_quant_backend_module()
        plan = self._load_quant_plan(cfg)

        def backend_for_name(name: str) -> str:
            selected = "none"
            for row in plan:
                prefix = row.get("prefix", "")
                if prefix and not name.startswith(prefix):
                    continue
                name_regex = row.get("name_regex", "")
                if name_regex and not re.search(name_regex, name):
                    continue
                exclude_regex = row.get("exclude_regex", "")
                if exclude_regex and re.search(exclude_regex, name):
                    continue
                selected = row.get("backend", "none")
            return selected

        def make_torchao_int8wo(linear: nn.Linear) -> nn.Module:
            from torchao.quantization import Int8WeightOnlyConfig, quantize_

            layer = nn.Linear(
                linear.in_features,
                linear.out_features,
                bias=False,
                dtype=torch.bfloat16,
                device=linear.weight.device,
            )
            layer.weight.data.copy_(linear.weight.detach().to(torch.bfloat16))
            layer.eval()
            quantize_(layer, Int8WeightOnlyConfig())
            return layer

        replacements: list[tuple[str, nn.Linear, str]] = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                backend = backend_for_name(name)
                if backend != "none":
                    replacements.append((name, module, backend))
        input_scales = self._collect_quant_input_scales(cfg, replacements)

        counts: dict[str, int] = {}
        bytes_before = 0
        bytes_after = 0
        log.info(
            "[robocasa365-rldx-server] replacing Linear backends "
            f"n={len(replacements)} plan={plan}"
        )
        for name, linear, backend_name in replacements:
            bytes_before += linear.weight.numel() * linear.weight.element_size()
            if linear.bias is not None:
                bytes_before += linear.bias.numel() * linear.bias.element_size()
            if backend_name == "vllm_gptq_marlin_w4a16":
                backend = backend_module.VllmGptqMarlinW4A16Linear(linear.weight, input_scale=input_scales.get(name))
            elif backend_name == "vllm_gptq_marlin_w8a16":
                backend = backend_module.VllmGptqMarlinW8A16Linear(linear.weight)
            elif backend_name == "vllm_allspark_w8a16":
                backend = backend_module.VllmAllSparkW8A16Linear(linear.weight)
            elif backend_name == "torchao_int8wo":
                backend = make_torchao_int8wo(linear)
            else:
                raise ValueError(f"Unsupported quant backend {backend_name!r} for module {name}")
            wrapped = QuantLinearWithOptionalBias(backend, linear.bias, name=name)
            bytes_after += sum(t.numel() * t.element_size() for t in list(wrapped.parameters()) + list(wrapped.buffers()))
            self._set_module(self.model, name, wrapped)
            counts[backend_name] = counts.get(backend_name, 0) + 1

        _profile_event(
            "linear_backend_replace",
            plan=plan,
            counts=counts,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            storage_ratio=(bytes_after / bytes_before) if bytes_before else None,
            calib_capture_dir=cfg.quant_calib_capture_dir,
            calib_skip=cfg.quant_calib_skip,
            calib_limit=cfg.quant_calib_limit,
            calib_alpha=cfg.quant_calib_alpha,
            calibrated_w4_modules=len(input_scales),
            elapsed_ms=_sync_elapsed_ms(quant_start),
        )

    def _make_marlin_backend_from_artifact(
        self,
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
            raise ValueError(f"Unsupported direct-load backend class {backend_class!r}")

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
        backend.register_buffer("qweight", payload["qweight"].to(device=device, non_blocking=True).contiguous(), persistent=False)
        backend.register_buffer("scales", payload["scales"].to(device=device, non_blocking=True).contiguous(), persistent=False)
        input_scale = payload.get("input_scale")
        if input_scale is None:
            backend.input_scale = None
        else:
            backend.register_buffer(
                "input_scale",
                input_scale.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous(),
                persistent=False,
            )
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        backend.register_buffer("workspace", torch.zeros(sms, dtype=torch.int, device=device), persistent=False)
        backend.register_buffer("empty", torch.empty(0, dtype=torch.int, device=device), persistent=False)
        return backend

    def _load_quant_artifact_into_model(self, model: nn.Module, quant_import_dir: str) -> dict[str, int]:
        import_start = time.perf_counter()
        root = Path(quant_import_dir).expanduser()
        validation = validate_quant_artifact(root)
        manifest = validation["manifest"]
        modules = manifest.get("modules", [])
        backend_module = self._load_quant_backend_module()
        device = torch.device("cuda", torch.cuda.current_device())
        counts: dict[str, int] = {}
        bytes_loaded = 0
        log.info(
            "[robocasa365-rldx-server] direct-loading quant artifact "
            f"dir={str(root)!r} modules={len(modules)}"
        )
        for metadata in modules:
            if not isinstance(metadata, dict):
                raise TypeError(f"quant artifact module entry must be a dict, got {type(metadata)}")
            name = str(metadata["name"])
            set_name = name
            if set_name.startswith("net.") and not hasattr(model, "net"):
                set_name = set_name[len("net.") :]
            tensor_file = root / str(metadata["tensor_file"])
            payload = torch.load(tensor_file, map_location="cpu", weights_only=True)
            if str(metadata.get("format")) != "vllm_marlin_wna16":
                raise ValueError(f"Unsupported quant artifact format for {name}: {metadata.get('format')!r}")
            backend = self._make_marlin_backend_from_artifact(backend_module, metadata, payload, device)
            bias = payload.get("bias")
            wrapped = QuantLinearWithOptionalBias(
                backend,
                bias.to(device=device) if isinstance(bias, torch.Tensor) else None,
                name=name,
            )
            self._set_module(model, set_name, wrapped)
            counts[str(metadata["backend_class"])] = counts.get(str(metadata["backend_class"]), 0) + 1
            for value in payload.values():
                if isinstance(value, torch.Tensor):
                    bytes_loaded += value.numel() * value.element_size()
            del payload

        self._direct_quant_import_applied = True

        _profile_event(
            "quant_artifact_import",
            import_dir=str(root),
            modules=len(modules),
            counts=counts,
            artifact_tensor_bytes=bytes_loaded,
            elapsed_ms=_sync_elapsed_ms(import_start),
        )
        return counts

    def _install_direct_quant_loader_patch(self, cfg: ServerConfig) -> None:
        if getattr(CosmosRoboCasa365Policy, "_direct_quant_loader_installed", False):
            return
        quant_import_dir = cfg.quant_import_dir
        policy_self = self
        artifact_validation = validate_quant_artifact(quant_import_dir)
        artifact_manifest = artifact_validation["manifest"]
        self_contained = artifact_manifest.get("schema_version") == BUNDLE_SCHEMA_VERSION

        from cosmos_framework.inference import model as inference_model

        original = inference_model.Cosmos3OmniModel.from_pretrained_dcp.__func__

        def from_pretrained_dcp_direct(
            cls: type[Any],
            checkpoint_path: Path,
            config: Any = None,
            parallelism_config: Any = None,
            compile_config: Any = None,
        ) -> Any:
            if config is None and self_contained:
                raise ValueError("A self-contained quant bundle must be loaded with its bundled runtime config")
            if config is None:
                config = inference_model.Cosmos3OmniConfig.from_pretrained(checkpoint_path)
            if parallelism_config is None:
                parallelism_config = inference_model.ParallelismConfig()
            if compile_config is None:
                compile_config = inference_model.CompileConfig()
            config.parallelism = inference_model.attrs.asdict(parallelism_config)
            config.compile = inference_model.attrs.asdict(compile_config)
            model = cls(config)
            checkpoint_type = inference_model.CheckpointType.from_path(checkpoint_path)
            if self_contained:
                expected_root = Path(quant_import_dir).expanduser().resolve()
                if checkpoint_path.expanduser().resolve() != expected_root:
                    raise ValueError(
                        "Self-contained quant bundle checkpoint path must be the artifact root: "
                        f"expected {expected_root}, got {checkpoint_path}"
                    )
                if not getattr(policy_self, "_direct_quant_import_applied", False):
                    policy_self._load_quant_artifact_into_model(model.model, quant_import_dir)
                state_dict = inference_model.get_model_state_dict(model.model)
                storage_reader = inference_model.HuggingFaceStorageReader(str(checkpoint_path))
                load_start = time.perf_counter()
                inference_model.dcp.load(state_dict=state_dict, storage_reader=storage_reader)
                _profile_event(
                    "direct_quant_bundle_load",
                    state_dict_keys=len(state_dict),
                    bundle_path=str(checkpoint_path),
                    elapsed_ms=_sync_elapsed_ms(load_start),
                )
                return model

            if checkpoint_type != inference_model.CheckpointType.DCP:
                return original(
                    cls,
                    checkpoint_path,
                    config=config,
                    parallelism_config=parallelism_config,
                    compile_config=compile_config,
                )

            if not getattr(policy_self, "_direct_quant_import_applied", False):
                policy_self._load_quant_artifact_into_model(model.model, quant_import_dir)
            state_dict = inference_model.get_model_state_dict(model.model)
            storage_reader = inference_model.FileSystemReader(str(checkpoint_path))
            load_start = time.perf_counter()
            inference_model.dcp.load(state_dict=state_dict, storage_reader=storage_reader)
            _profile_event(
                "direct_quant_dcp_load",
                state_dict_keys=len(state_dict),
                checkpoint_path=str(checkpoint_path),
                elapsed_ms=_sync_elapsed_ms(load_start),
            )
            return model

        inference_model.Cosmos3OmniModel.from_pretrained_dcp = classmethod(from_pretrained_dcp_direct)
        original_to_empty = torch.nn.Module.to_empty

        def to_empty_with_direct_quant(module: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
            if (
                type(module).__name__ == "Cosmos3VFMNetwork"
                and not getattr(policy_self, "_direct_quant_import_applied", False)
            ):
                _profile_event("direct_quant_pre_to_empty_start", module=type(module).__name__)
                policy_self._load_quant_artifact_into_model(module, quant_import_dir)
                result = original_to_empty(module, *args, **kwargs)
                _profile_event("direct_quant_pre_to_empty_done", module=type(module).__name__)
                return result
            return original_to_empty(module, *args, **kwargs)

        torch.nn.Module.to_empty = to_empty_with_direct_quant
        CosmosRoboCasa365Policy._direct_quant_loader_installed = True
        log.info(
            "[robocasa365-rldx-server] installed direct quant loader patch "
            f"quant_import_dir={quant_import_dir!r}"
        )

    def _export_quant_artifacts(self, cfg: ServerConfig) -> None:
        export_start = time.perf_counter()
        root = Path(cfg.quant_export_dir).expanduser()
        tensor_dir = root / "tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        modules: list[dict[str, Any]] = []

        def cpu_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value.detach().to("cpu").contiguous()

        for name, module in self.model.named_modules():
            backend = getattr(module, "backend", None)
            if backend is None:
                continue
            backend_name = type(backend).__name__
            payload: dict[str, Any] = {
                "backend_class": backend_name,
                "bias": cpu_tensor(getattr(module, "bias", None)),
            }
            metadata: dict[str, Any] = {
                "name": name,
                "backend_class": backend_name,
            }

            if backend_name in {"VllmGptqMarlinW4A16Linear", "VllmGptqMarlinW8A16Linear"}:
                metadata.update(
                    {
                        "format": "vllm_marlin_wna16",
                        "num_bits": int(getattr(backend, "num_bits")),
                        "group_size": int(getattr(backend, "group_size")),
                        "size_k": int(getattr(backend, "size_k")),
                        "size_n": int(getattr(backend, "size_n")),
                        "wtype_id": int(getattr(backend, "wtype_id")),
                    }
                )
                payload.update(
                    {
                        "qweight": cpu_tensor(getattr(backend, "qweight")),
                        "scales": cpu_tensor(getattr(backend, "scales")),
                        "input_scale": cpu_tensor(getattr(backend, "input_scale", None)),
                    }
                )
            elif backend_name == "VllmAllSparkW8A16Linear":
                metadata.update(
                    {
                        "format": "vllm_allspark_w8a16",
                        "size_k": int(getattr(backend, "size_k")),
                        "size_n": int(getattr(backend, "size_n")),
                        "sm_count": int(getattr(backend, "sm_count")),
                        "sm_version": int(getattr(backend, "sm_version")),
                    }
                )
                payload.update(
                    {
                        "qweight": cpu_tensor(getattr(backend, "qweight")),
                        "scales": cpu_tensor(getattr(backend, "scales")),
                    }
                )
            else:
                metadata.update({"format": "unsupported"})
                payload["state_dict"] = {k: cpu_tensor(v) for k, v in backend.state_dict().items()}

            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            rel_path = f"tensors/{safe_name}.pt"
            torch.save(payload, root / rel_path)
            metadata["tensor_file"] = rel_path
            modules.append(metadata)

        manifest = {
            "schema_version": 1,
            "created_unix": time.time(),
            "checkpoint_path": cfg.checkpoint_path,
            "config_file": cfg.config_file,
            "quant_plan_file": cfg.quant_plan_file,
            "quant_backend": cfg.quant_backend,
            "quant_target_prefix": cfg.quant_target_prefix,
            "calibration": {
                "capture_dir": cfg.quant_calib_capture_dir,
                "skip": cfg.quant_calib_skip,
                "limit": cfg.quant_calib_limit,
                "alpha": cfg.quant_calib_alpha,
            },
            "modules": modules,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _profile_event(
            "quant_artifact_export",
            export_dir=str(root),
            modules=len(modules),
            elapsed_ms=_sync_elapsed_ms(export_start),
        )

    def _collect_quant_input_scales(
        self,
        cfg: ServerConfig,
        replacements: list[tuple[str, nn.Linear, str]],
    ) -> dict[str, torch.Tensor]:
        if not cfg.quant_calib_capture_dir:
            return {}
        calib_start = time.perf_counter()
        capture_dir = Path(cfg.quant_calib_capture_dir).expanduser()
        request_files = sorted(capture_dir.glob("sample_*.request.msgpack"))
        if cfg.quant_calib_skip > 0:
            request_files = request_files[cfg.quant_calib_skip :]
        if cfg.quant_calib_limit > 0:
            request_files = request_files[: cfg.quant_calib_limit]
        if not request_files:
            raise FileNotFoundError(f"No calibration request files found in {capture_dir}")

        target_names = {
            name
            for name, _linear, backend in replacements
            if backend == "vllm_gptq_marlin_w4a16"
        }
        if not target_names:
            return {}

        modules = dict(self.model.named_modules())
        stats: dict[str, torch.Tensor] = {}
        hook_handles: list[Any] = []

        def make_hook(name: str) -> Any:
            def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs:
                    return
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.numel() == 0:
                    return
                x_2d = x.detach().reshape(-1, x.shape[-1]).abs()
                value = x_2d.amax(dim=0).float().cpu()
                previous = stats.get(name)
                stats[name] = value if previous is None else torch.maximum(previous, value)

            return hook

        for name in sorted(target_names):
            module = modules.get(name)
            if isinstance(module, nn.Linear):
                hook_handles.append(module.register_forward_pre_hook(make_hook(name)))

        log.info(
            "[robocasa365-rldx-server] collecting quant calibration stats "
            f"capture_dir={str(capture_dir)!r} n_requests={len(request_files)} "
            f"target_w4_modules={len(target_names)} alpha={cfg.quant_calib_alpha}"
        )
        try:
            with torch.inference_mode():
                for request_file in request_files:
                    request = MsgSerializer.from_bytes(request_file.read_bytes())
                    if not isinstance(request, dict):
                        continue
                    data = request.get("data", {})
                    if not isinstance(data, dict):
                        continue
                    observation = data.get("observation")
                    options = data.get("options")
                    if isinstance(observation, dict):
                        self.get_action(observation, options if isinstance(options, dict) else None)
        finally:
            for handle in hook_handles:
                handle.remove()

        input_scales: dict[str, torch.Tensor] = {}
        alpha = float(cfg.quant_calib_alpha)
        for name, values in stats.items():
            values = values.clamp(min=1e-6)
            normalized = values / values.mean().clamp(min=1e-6)
            scale = normalized.pow(alpha).clamp(min=1e-2, max=1e2).to(torch.bfloat16)
            input_scales[name] = scale

        _profile_event(
            "quant_calibration",
            capture_dir=str(capture_dir),
            requested=len(request_files),
            skip=cfg.quant_calib_skip,
            target_w4_modules=len(target_names),
            calibrated_w4_modules=len(input_scales),
            missing_w4_modules=len(target_names) - len(input_scales),
            alpha=cfg.quant_calib_alpha,
            elapsed_ms=_sync_elapsed_ms(calib_start),
        )
        return input_scales

    def _collect_direct_quant_calibration(self, cfg: ServerConfig) -> None:
        if not cfg.quant_import_dir:
            raise ValueError("--quant-calibration-stats-output requires --quant-import-dir")
        capture_dir = Path(cfg.quant_calib_capture_dir).expanduser()
        request_files = sorted(capture_dir.glob("sample_*.request.msgpack"))
        if cfg.quant_calib_skip > 0:
            request_files = request_files[cfg.quant_calib_skip :]
        if cfg.quant_calib_limit > 0:
            request_files = request_files[: cfg.quant_calib_limit]
        if not request_files:
            raise FileNotFoundError(f"No calibration request files found in {capture_dir}")

        stats: dict[str, torch.Tensor] = {}
        hook_handles: list[Any] = []

        def make_hook(name: str) -> Any:
            def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor) or inputs[0].numel() == 0:
                    return
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).abs()
                value = x.amax(dim=0).float().cpu()
                previous = stats.get(name)
                stats[name] = value if previous is None else torch.maximum(previous, value)

            return hook

        for module in self.model.modules():
            if isinstance(module, QuantLinearWithOptionalBias):
                name = str(module.profile_name)
                hook_handles.append(module.register_forward_pre_hook(make_hook(name)))
        if len(hook_handles) != 504:
            raise ValueError(f"Expected 504 direct-quant calibration hooks, found {len(hook_handles)}")

        calibration_start = time.perf_counter()
        processed_requests = 0
        try:
            with torch.inference_mode():
                for request_file in request_files:
                    request = MsgSerializer.from_bytes(request_file.read_bytes())
                    if not isinstance(request, dict):
                        raise TypeError(f"Calibration request must contain a dict: {request_file}")
                    data = request.get("data", {})
                    if not isinstance(data, dict):
                        raise TypeError(f"Calibration request data must contain a dict: {request_file}")
                    observation = data.get("observation")
                    options = data.get("options")
                    if not isinstance(observation, dict):
                        raise TypeError(f"Calibration request has no observation dict: {request_file}")
                    self.get_action(observation, options if isinstance(options, dict) else None)
                    processed_requests += 1
        finally:
            for handle in hook_handles:
                handle.remove()
        if len(stats) != 504:
            raise ValueError(f"Calibration exercised {len(stats)} of 504 quantized Linear modules")

        output_path = Path(cfg.quant_calibration_stats_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        torch.save(dict(sorted(stats.items())), temp_path)
        os.replace(temp_path, output_path)
        _profile_event(
            "direct_quant_calibration",
            capture_dir=str(capture_dir),
            requested=len(request_files),
            processed=processed_requests,
            modules=len(stats),
            output_path=str(output_path),
            elapsed_ms=_sync_elapsed_ms(calibration_start),
        )

    def _build_setup_args(self, cfg: ServerConfig, config_file: str) -> OmniSetupArgs:
        overrides = OmniSetupOverrides.model_validate(
            {
                "checkpoint_path": cfg.checkpoint_path,
                "config_file": config_file,
                "output_dir": cfg.output_dir,
                "sampler": cfg.sampler,
                "guardrails": cfg.guardrails,
                "use_torch_compile": cfg.use_torch_compile,
            }
        )
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _next_seed(self) -> int:
        if self.cfg.deterministic_seed:
            return self.cfg.seed
        return int(self._rng.integers(0, 2**31))

    def get_modality_config(self) -> dict[str, Any]:
        return {
            "video": _modality_config(
                [0],
                ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            ),
            "action": _modality_config(
                list(range(self.cfg.served_action_steps)),
                [
                    "base_motion",
                    "control_mode",
                    "end_effector_position",
                    "end_effector_rotation",
                    "gripper_close",
                ],
            ),
            "language": _modality_config([0], ["annotation.human.task_description"]),
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def _build_sample(self, observation: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        left = _extract_frame(observation, _VIDEO_KEY_CANDIDATES["left"], batch_idx)
        right = _extract_frame(observation, _VIDEO_KEY_CANDIDATES["right"], batch_idx)
        wrist = _extract_frame(observation, _VIDEO_KEY_CANDIDATES["wrist"], batch_idx)
        image = _compose_concat_view(left=left, right=right, wrist=wrist, camera_size=self.cfg.camera_size)
        video = torch.zeros(
            (3, self.cfg.action_chunk_size + 1, image.shape[-2], image.shape[-1]),
            dtype=torch.uint8,
        )
        video[:, 0] = image

        action = torch.zeros((self.cfg.action_chunk_size, self.cfg.raw_action_dim), dtype=torch.float32)
        sample = {
            "ai_caption": _extract_prompt(observation, batch_idx),
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(int(round(self.cfg.conditioning_fps)), dtype=torch.long),
            "mode": "policy",
            "domain_id": torch.tensor(get_domain_id("robocasa365"), dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        return self.transform(sample, self.cfg.resolution)

    def _infer_one(self, observation: dict[str, Any], batch_idx: int, action_steps: int) -> np.ndarray:
        infer_start = time.perf_counter()
        build_start = time.perf_counter()
        sample = self._build_sample(observation, batch_idx)
        build_ms = (time.perf_counter() - build_start) * 1000.0
        batch_start = time.perf_counter()
        data_batch = _build_data_batch_from_sample(sample)
        batch_ms = (time.perf_counter() - batch_start) * 1000.0
        seed = self._next_seed()
        generate_start = time.perf_counter()
        with self._lock:
            with torch.inference_mode():
                output = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )
        generate_ms = _sync_elapsed_ms(generate_start)
        post_start = time.perf_counter()
        action = output["action"][0][:, : self.cfg.raw_action_dim]
        action_np = action[:action_steps].detach().cpu().float().numpy()
        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(action_np, -1.0, 1.0)
        if np.any(clipped != action_np) and self._clip_warning_count < 5:
            log.warning(
                "[robocasa365-rldx-server] clipping action output to [-1, 1]: "
                f"min={float(action_np.min()):.4f} max={float(action_np.max()):.4f}"
            )
            self._clip_warning_count += 1
        action_np = clipped
        post_ms = _sync_elapsed_ms(post_start)
        _profile_event(
            "infer_one",
            batch_idx=batch_idx,
            action_steps=action_steps,
            build_sample_ms=build_ms,
            build_batch_ms=batch_ms,
            generate_ms=generate_ms,
            postprocess_ms=post_ms,
            elapsed_ms=_sync_elapsed_ms(infer_start),
        )
        return action_np.astype(np.float32, copy=False)

    @staticmethod
    def _split_action(action: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "action.base_motion": action[:, 0:4],
            "action.control_mode": action[:, 4:5],
            "action.end_effector_position": action[:, 5:8],
            "action.end_effector_rotation": action[:, 8:11],
            "action.gripper_close": action[:, 11:12],
        }

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        get_action_start = time.perf_counter()
        if not isinstance(observation, dict):
            raise TypeError(f"observation must be a dict, got {type(observation)}")
        batch = _batch_size(observation)
        requested_steps = self.cfg.served_action_steps
        if isinstance(options, dict) and options.get("executed_action_steps") is not None:
            requested_steps = int(options["executed_action_steps"])
        action_steps = max(1, min(requested_steps, self.cfg.action_chunk_size))

        per_env = [self._infer_one(observation, idx, action_steps) for idx in range(batch)]
        split = {key: [] for key in _ACTION_KEYS}
        for action in per_env:
            for key, value in self._split_action(action).items():
                split[key].append(value)
        actions = {key: np.stack(values, axis=0).astype(np.float32) for key, values in split.items()}
        _profile_event(
            "get_action",
            batch=batch,
            action_steps=action_steps,
            elapsed_ms=_sync_elapsed_ms(get_action_start),
        )
        return actions, {"action_steps": action_steps}


class PolicyServer:
    def __init__(self, policy: CosmosRoboCasa365Policy, host: str, port: int) -> None:
        self.policy = policy
        self.running = True
        self.allow_admin_endpoints = _is_loopback_host(host)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def _dispatch(self, request: dict[str, Any]) -> Any:
        endpoint = request.get("endpoint", "get_action")
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "kill":
            if not self.allow_admin_endpoints:
                raise PermissionError("The kill endpoint is available only on a loopback-bound server")
            self.running = False
            _flush_quant_linear_shapes()
            return {"status": "ok"}
        if endpoint == "get_modality_config":
            return self.policy.get_modality_config()
        if endpoint == "reset":
            return self.policy.reset(**request.get("data", {}))
        if endpoint == "get_action":
            return self.policy.get_action(**request.get("data", {}))
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}", flush=True)
        while self.running:
            try:
                recv_start = time.perf_counter()
                message = self.socket.recv()
                recv_ms = (time.perf_counter() - recv_start) * 1000.0
                request_start = time.perf_counter()
                decode_start = time.perf_counter()
                request = MsgSerializer.from_bytes(message)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                endpoint = request.get("endpoint", "get_action")
                dispatch_start = time.perf_counter()
                response = self._dispatch(request)
                dispatch_ms = _sync_elapsed_ms(dispatch_start)
                encode_start = time.perf_counter()
                response_bytes = MsgSerializer.to_bytes(response)
                encode_ms = (time.perf_counter() - encode_start) * 1000.0
                self.socket.send(response_bytes)
                _profile_event(
                    "request",
                    endpoint=endpoint,
                    request_bytes=len(message),
                    response_bytes=len(response_bytes),
                    recv_ms=recv_ms,
                    decode_ms=decode_ms,
                    dispatch_ms=dispatch_ms,
                    encode_ms=encode_ms,
                    elapsed_ms=_sync_elapsed_ms(request_start),
                )
            except Exception as exc:
                log.exception("[robocasa365-rldx-server] request failed")
                self.socket.send(MsgSerializer.to_bytes({"error": str(exc)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--config-file", default="")
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5577)
    parser.add_argument("--sampler", default="unipc", choices=["unipc", "edm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--conditioning-fps", type=float, default=20.0)
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--action-chunk-size", type=int, default=32)
    parser.add_argument("--served-action-steps", type=int, default=8)
    parser.add_argument("--raw-action-dim", type=int, default=12)
    parser.add_argument("--max-action-dim", type=int, default=64)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--guardrails", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torchao-quant", choices=["none", "int8wo", "int4wo"], default="none")
    parser.add_argument("--torchao-target-prefix", default="net.language_model.model.layers")
    parser.add_argument(
        "--quant-backend",
        choices=[
            "none",
            "vllm_gptq_marlin_w4a16",
            "vllm_gptq_marlin_w8a16",
            "vllm_allspark_w8a16",
            "torchao_int8wo",
        ],
        default="none",
    )
    parser.add_argument("--quant-target-prefix", default="net.language_model.model.layers")
    parser.add_argument("--quant-plan-file", default="")
    parser.add_argument("--quant-calib-capture-dir", default="")
    parser.add_argument("--quant-calib-skip", type=int, default=0)
    parser.add_argument("--quant-calib-limit", type=int, default=0)
    parser.add_argument("--quant-calib-alpha", type=float, default=0.5)
    parser.add_argument("--quant-calibration-stats-output", default="")
    parser.add_argument("--quant-export-dir", default="")
    parser.add_argument("--quant-import-dir", default="")
    parser.add_argument("--allow-legacy-quant-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path
    config_file = args.config_file
    if args.quant_import_dir:
        validation = validate_quant_artifact(args.quant_import_dir)
        if validation["self_contained"]:
            bundle_root = Path(validation["root"])
            if checkpoint_path and Path(checkpoint_path).expanduser().resolve() != bundle_root:
                raise ValueError("Do not pass an external --checkpoint-path with a self-contained quant bundle")
            if config_file:
                raise ValueError("Do not pass an external --config-file with a self-contained quant bundle")
            checkpoint_path = str(bundle_root)
            config_file = str(materialize_bundle_config(bundle_root, args.output_dir))
        elif not args.allow_legacy_quant_artifact:
            raise ValueError(
                "Legacy schema-v1 quant artifacts depend on an external DCP. Convert this artifact with "
                "robocasa365_quant_pipeline build-self-contained-bundle, or explicitly pass "
                "--allow-legacy-quant-artifact for rollback-only use."
            )
    if not checkpoint_path:
        raise ValueError("--checkpoint-path is required unless --quant-import-dir is a self-contained bundle")
    config_file = config_file or _infer_config_file(checkpoint_path)
    cfg = ServerConfig(
        checkpoint_path=str(Path(checkpoint_path).expanduser()),
        config_file=str(Path(config_file).expanduser()),
        output_dir=str(Path(args.output_dir).expanduser()),
        host=args.host,
        port=args.port,
        sampler=args.sampler,
        seed=args.seed,
        deterministic_seed=args.deterministic_seed,
        guidance=args.guidance,
        num_steps=args.num_steps,
        shift=args.shift,
        conditioning_fps=args.conditioning_fps,
        resolution=str(args.resolution),
        action_chunk_size=args.action_chunk_size,
        served_action_steps=args.served_action_steps,
        raw_action_dim=args.raw_action_dim,
        max_action_dim=args.max_action_dim,
        camera_size=args.camera_size,
        guardrails=args.guardrails,
        use_torch_compile=args.torch_compile,
        torchao_quant=args.torchao_quant,
        torchao_target_prefix=args.torchao_target_prefix,
        quant_backend=args.quant_backend,
        quant_target_prefix=args.quant_target_prefix,
        quant_plan_file=args.quant_plan_file,
        quant_calib_capture_dir=args.quant_calib_capture_dir,
        quant_calib_skip=args.quant_calib_skip,
        quant_calib_limit=args.quant_calib_limit,
        quant_calib_alpha=args.quant_calib_alpha,
        quant_calibration_stats_output=args.quant_calibration_stats_output,
        quant_export_dir=args.quant_export_dir,
        quant_import_dir=args.quant_import_dir,
        allow_legacy_quant_artifact=args.allow_legacy_quant_artifact,
    )
    policy = CosmosRoboCasa365Policy(cfg)
    PolicyServer(policy, cfg.host, cfg.port).run()


if __name__ == "__main__":
    main()
