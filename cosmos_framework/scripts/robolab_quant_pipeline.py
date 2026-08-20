# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""CLI for exporting and validating self-contained Cosmos3 RoboLab bundles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from cosmos_framework.scripts.robolab_quant_bundle import (
    build_robolab_quant_bundle,
    convert_gen_w8_bundle_to_w8a8,
    convert_w8_bundle_to_w8a8,
    discover_quant_targets,
    validate_robolab_quant_bundle,
)

_DROID_REPOSITORY = "nvidia/Cosmos3-Nano-Policy-DROID"
_DROID_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
_EDGE_DROID_REPOSITORY = "nvidia/Cosmos3-Edge-Policy-DROID"
_EDGE_DROID_REVISION = "3ea407af3e156c0af3b4bb6edd85842cc9a58777"
_QWEN_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
_QWEN_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
_WAN_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
_WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
_PUBLIC_STRATEGIES = ("full_w8", "gen_branch_w8a8")


def _prepare_public_sources(
    asset_dir: str | Path,
    *,
    model_family: str = "cosmos3_nano",
    droid_revision: str,
    qwen_revision: str,
    wan_revision: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root = Path(asset_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if model_family not in {"cosmos3_nano", "cosmos3_edge"}:
        raise ValueError(f"Unsupported model family: {model_family}")
    is_edge = model_family == "cosmos3_edge"
    droid_repository = _EDGE_DROID_REPOSITORY if is_edge else _DROID_REPOSITORY
    checkpoint_dir = root / ("Cosmos3-Edge-Policy-DROID" if is_edge else "Cosmos3-Nano-Policy-DROID")
    tokenizer_dir = root / "Qwen3-VL-8B-Instruct-tokenizer"
    vae_dir = root / "Wan2.2-TI2V-5B"

    snapshot_download(
        repo_id=droid_repository,
        revision=droid_revision,
        local_dir=checkpoint_dir,
        allow_patterns=[
            "config.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "model.safetensors.index.json",
            "transformer/*.safetensors",
            "vision_encoder/*.safetensors",
        ],
    )
    if not is_edge:
        snapshot_download(
            repo_id=_QWEN_REPOSITORY,
            revision=qwen_revision,
            local_dir=tokenizer_dir,
            allow_patterns=[
                "README.md",
                "chat_template.json",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model.safetensors.index.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "video_preprocessor_config.json",
                "vocab.json",
            ],
        )
    else:
        tokenizer_dir = checkpoint_dir
    vae_path = Path(
        hf_hub_download(
            repo_id=_WAN_REPOSITORY,
            filename="Wan2.2_VAE.pth",
            revision=wan_revision,
            local_dir=vae_dir,
        )
    ).resolve()

    api = HfApi()
    resolved_revisions = {
        "droid": api.model_info(droid_repository, revision=droid_revision).sha,
        "wan": api.model_info(_WAN_REPOSITORY, revision=wan_revision).sha,
    }
    if not is_edge:
        resolved_revisions["qwen"] = api.model_info(_QWEN_REPOSITORY, revision=qwen_revision).sha
    requested_revisions = {
        "droid": droid_revision,
        "wan": wan_revision,
    }
    if not is_edge:
        requested_revisions["qwen"] = qwen_revision
    result = {
        "asset_dir": str(root),
        "checkpoint_dir": str(checkpoint_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "vae_path": str(vae_path),
        "repositories": {
            "droid": droid_repository,
            "wan": _WAN_REPOSITORY,
        },
        "requested_revisions": requested_revisions,
        "resolved_revisions": resolved_revisions,
    }
    if not is_edge:
        result["repositories"]["qwen"] = _QWEN_REPOSITORY
    (root / "sources.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _add_public_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--model-family", choices=("cosmos3_nano", "cosmos3_edge"), default="cosmos3_nano")
    parser.add_argument("--droid-revision")
    parser.add_argument("--qwen-revision", default=_QWEN_REVISION)
    parser.add_argument("--wan-revision", default=_WAN_REVISION)


def _load_source_provenance(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    provenance_path = Path(path).expanduser().resolve()
    value = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Source provenance must contain a JSON object: {provenance_path}")
    for field in ("repositories", "resolved_revisions"):
        if not isinstance(value.get(field), dict):
            raise ValueError(f"Source provenance has no {field!r} mapping: {provenance_path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the DROID quantization map")
    inspect_parser.add_argument("--source-checkpoint", required=True)
    inspect_parser.add_argument("--strategy", required=True)

    build_parser = subparsers.add_parser("build-bundle", help="Stream-pack one self-contained bundle")
    build_parser.add_argument("--strategy", required=True)
    build_parser.add_argument("--source-checkpoint", required=True)
    build_parser.add_argument("--tokenizer-dir", required=True)
    build_parser.add_argument("--vae-path", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--device", default="cuda:0")
    build_parser.add_argument("--calibration-stats")
    build_parser.add_argument("--calibration-alpha", type=float, default=0.5)
    build_parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    build_parser.add_argument("--max-residual-shard-size", type=int, default=2 * 1024**3)
    build_parser.add_argument("--source-provenance-json")

    convert_parser = subparsers.add_parser(
        "convert-gen-w8-to-w8a8",
        help="Reuse a calibrated GenW8 bundle and convert its generation branch to FP8 W8A8",
    )
    convert_parser.add_argument("--base-bundle", required=True)
    convert_parser.add_argument("--source-checkpoint", required=True)
    convert_parser.add_argument("--output-dir", required=True)
    convert_parser.add_argument("--device", default="cuda:0")
    convert_parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    convert_parser.add_argument("--calibration-stats")
    convert_parser.add_argument("--calibration-alpha", type=float, default=0.5)

    convert_full_parser = subparsers.add_parser(
        "convert-full-w8-to-w8a8",
        help="Convert every W8A16 Linear in a FullW8 bundle to FP8 W8A8",
    )
    convert_full_parser.add_argument("--base-bundle", required=True)
    convert_full_parser.add_argument("--source-checkpoint", required=True)
    convert_full_parser.add_argument("--output-dir", required=True)
    convert_full_parser.add_argument("--device", default="cuda:0")
    convert_full_parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    convert_full_parser.add_argument("--calibration-stats")
    convert_full_parser.add_argument("--calibration-alpha", type=float, default=0.5)

    validate_parser = subparsers.add_parser("validate", help="Validate a deployment bundle")
    validate_parser.add_argument("--bundle-dir", required=True)
    validate_parser.add_argument("--expected-strategy")
    validate_parser.add_argument("--check-hashes", action="store_true")
    validate_parser.add_argument("--check-tensors", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-sources", help="Download the public DROID export inputs")
    _add_public_source_arguments(prepare_parser)

    public_parser = subparsers.add_parser(
        "build-public", help="Download the public DROID policy inputs and stream-pack a bundle"
    )
    _add_public_source_arguments(public_parser)
    public_parser.add_argument("--strategy", choices=_PUBLIC_STRATEGIES, default="full_w8")
    public_parser.add_argument("--output-dir", required=True)
    public_parser.add_argument("--device", default="cuda:0")
    public_parser.add_argument("--calibration-stats")
    public_parser.add_argument("--calibration-alpha", type=float, default=0.5)
    public_parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    public_parser.add_argument("--max-residual-shard-size", type=int, default=2 * 1024**3)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    if args.command == "inspect":
        targets = discover_quant_targets(args.source_checkpoint, args.strategy)
        result = {
            "strategy": args.strategy,
            "modules": len(targets),
            "w4_modules": sum(target.num_bits == 4 for target in targets),
            "w8_modules": sum(target.num_bits == 8 for target in targets),
        }
    elif args.command == "build-bundle":
        result = build_robolab_quant_bundle(
            strategy=args.strategy,
            source_checkpoint=args.source_checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            vae_path=args.vae_path,
            output_dir=args.output_dir,
            device=args.device,
            calibration_stats=args.calibration_stats,
            calibration_alpha=args.calibration_alpha,
            copy_mode=args.copy_mode,
            max_residual_shard_size=args.max_residual_shard_size,
            source_provenance=_load_source_provenance(args.source_provenance_json),
        )
    elif args.command == "convert-gen-w8-to-w8a8":
        result = convert_gen_w8_bundle_to_w8a8(
            base_bundle=args.base_bundle,
            source_checkpoint=args.source_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            copy_mode=args.copy_mode,
            calibration_stats=args.calibration_stats,
            calibration_alpha=args.calibration_alpha,
        )
    elif args.command == "convert-full-w8-to-w8a8":
        result = convert_w8_bundle_to_w8a8(
            base_bundle=args.base_bundle,
            source_checkpoint=args.source_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            copy_mode=args.copy_mode,
            calibration_stats=args.calibration_stats,
            calibration_alpha=args.calibration_alpha,
            strategy="full_w8a8",
        )
    elif args.command == "validate":
        result = validate_robolab_quant_bundle(
            args.bundle_dir,
            expected_strategy=args.expected_strategy,
            check_hashes=args.check_hashes,
            check_tensors=args.check_tensors,
        )
        result.pop("manifest", None)
    elif args.command == "prepare-sources":
        droid_revision = args.droid_revision or (
            _EDGE_DROID_REVISION if args.model_family == "cosmos3_edge" else _DROID_REVISION
        )
        result = _prepare_public_sources(
            args.asset_dir,
            model_family=args.model_family,
            droid_revision=droid_revision,
            qwen_revision=args.qwen_revision,
            wan_revision=args.wan_revision,
        )
    elif args.command == "build-public":
        droid_revision = args.droid_revision or (
            _EDGE_DROID_REVISION if args.model_family == "cosmos3_edge" else _DROID_REVISION
        )
        sources = _prepare_public_sources(
            args.asset_dir,
            model_family=args.model_family,
            droid_revision=droid_revision,
            qwen_revision=args.qwen_revision,
            wan_revision=args.wan_revision,
        )
        result = build_robolab_quant_bundle(
            strategy=args.strategy,
            source_checkpoint=sources["checkpoint_dir"],
            tokenizer_dir=sources["tokenizer_dir"],
            vae_path=sources["vae_path"],
            output_dir=args.output_dir,
            device=args.device,
            calibration_stats=args.calibration_stats,
            calibration_alpha=args.calibration_alpha,
            copy_mode=args.copy_mode,
            max_residual_shard_size=args.max_residual_shard_size,
            source_provenance={
                "repositories": sources["repositories"],
                "requested_revisions": sources["requested_revisions"],
                "resolved_revisions": sources["resolved_revisions"],
            },
        )
        result["sources"] = sources
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
