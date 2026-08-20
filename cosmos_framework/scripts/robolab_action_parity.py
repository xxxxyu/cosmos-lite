# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Compare two directories of captured RoboLab policy responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client import msgpack_numpy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output")
    return parser


def summarize(reference_dir: str | Path, candidate_dir: str | Path) -> dict[str, object]:
    reference_root = Path(reference_dir).expanduser().resolve()
    candidate_root = Path(candidate_dir).expanduser().resolve()
    candidate_files = sorted(candidate_root.glob("sample_*.response.msgpack"))
    if not candidate_files:
        raise FileNotFoundError(f"No RoboLab responses found in {candidate_root}")

    references: list[np.ndarray] = []
    differences: list[np.ndarray] = []
    sample_l1: list[float] = []
    sample_linf: list[float] = []
    for candidate_file in candidate_files:
        reference_file = reference_root / candidate_file.name
        if not reference_file.is_file():
            raise FileNotFoundError(f"Reference response is missing: {reference_file}")
        reference = np.asarray(msgpack_numpy.unpackb(reference_file.read_bytes())["action"], dtype=np.float32)
        candidate = np.asarray(msgpack_numpy.unpackb(candidate_file.read_bytes())["action"], dtype=np.float32)
        if candidate.shape != reference.shape:
            raise ValueError(f"Action shape mismatch for {candidate_file.name}: {candidate.shape} != {reference.shape}")
        difference = np.abs(candidate - reference)
        references.append(reference)
        differences.append(difference)
        sample_l1.append(float(difference.mean()))
        sample_linf.append(float(difference.max()))

    reference_values = np.concatenate(references, axis=0)
    difference_values = np.concatenate(differences, axis=0)
    per_dimension = []
    for index in range(reference_values.shape[1]):
        reference_dimension = reference_values[:, index]
        difference_dimension = difference_values[:, index]
        per_dimension.append(
            {
                "dimension": index,
                "reference_abs_mean": float(np.mean(np.abs(reference_dimension))),
                "reference_std": float(np.std(reference_dimension)),
                "l1_mean": float(np.mean(difference_dimension)),
                "abs_error_p95": float(np.percentile(difference_dimension, 95)),
                "linf": float(np.max(difference_dimension)),
            }
        )
    return {
        "reference_dir": str(reference_root),
        "candidate_dir": str(candidate_root),
        "samples": len(candidate_files),
        "action_shape": list(references[0].shape),
        "action_error": {
            "l1_mean": float(np.mean(difference_values)),
            "abs_error_p95": float(np.percentile(difference_values, 95)),
            "linf_sample_p95": float(np.percentile(sample_linf, 95)),
            "linf": float(np.max(difference_values)),
            "sample_l1_p95": float(np.percentile(sample_l1, 95)),
        },
        "per_dimension": per_dimension,
    }


def main() -> None:
    args = _parser().parse_args()
    result = summarize(args.reference_dir, args.candidate_dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
