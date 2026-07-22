# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Summarize paired RoboLab rollout directories with an initial-state gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Episode:
    run: int
    success: bool
    final_step: int
    initial_state_sha256: str


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return center - radius, center + radius


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def _initial_state_hash(hdf5_path: Path) -> str:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("initial-state validation requires h5py; run this with the RoboLab Python") from exc

    digest = hashlib.sha256()
    with h5py.File(hdf5_path, "r") as handle:
        root = handle["data/demo_0/initial_state"]

        def update(name: str, obj: object) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            value = obj[()]
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())

        root.visititems(update)
    return digest.hexdigest()


def _load_episodes(task_dir: Path, expected_runs: int) -> dict[int, Episode]:
    episodes: dict[int, Episode] = {}
    for log_path in task_dir.glob("log_*_env0.json"):
        record = json.loads(log_path.read_text())
        run = int(record["run"])
        if run in episodes:
            raise ValueError(f"duplicate run {run} in {task_dir}")
        hdf5_path = task_dir / f"run_{run}.hdf5"
        if not hdf5_path.is_file():
            raise FileNotFoundError(hdf5_path)
        episodes[run] = Episode(
            run=run,
            success=bool(record["success"]),
            final_step=int(record["final_step"]),
            initial_state_sha256=_initial_state_hash(hdf5_path),
        )

    expected = set(range(expected_runs))
    actual = set(episodes)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{task_dir}: expected runs 0..{expected_runs - 1}; missing={missing}, extra={extra}")
    return episodes


def _parse_setting(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("setting must be LABEL=TASK_DIR")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setting", nargs="+", type=_parse_setting, help="LABEL=TASK_DIR; first setting is baseline")
    parser.add_argument("--expected-runs", type=int, default=50)
    args = parser.parse_args()

    loaded = [(label, _load_episodes(path, args.expected_runs)) for label, path in args.setting]
    baseline_label, baseline = loaded[0]
    baseline_hashes = {run: episode.initial_state_sha256 for run, episode in baseline.items()}

    print("| Setting | Success (Wilson 95% CI) | Successful step median | Paired wins/losses | McNemar p |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for label, episodes in loaded:
        hashes = {run: episode.initial_state_sha256 for run, episode in episodes.items()}
        mismatches = [run for run in sorted(baseline) if hashes[run] != baseline_hashes[run]]
        if mismatches:
            raise ValueError(f"initial states differ between {baseline_label} and {label}: runs {mismatches}")

        successes = sum(episode.success for episode in episodes.values())
        low, high = _wilson_interval(successes, len(episodes))
        successful_steps = [episode.final_step for episode in episodes.values() if episode.success]
        median = statistics.median(successful_steps) if successful_steps else "-"
        if label == baseline_label:
            paired = "reference"
            p_value = "-"
        else:
            wins = sum(episodes[run].success and not baseline[run].success for run in baseline)
            losses = sum(not episodes[run].success and baseline[run].success for run in baseline)
            paired = f"{wins}/{losses}"
            p_value = f"{_mcnemar_exact(wins, losses):.4g}"
        print(
            f"| `{label}` | {successes}/{len(episodes)} = {successes / len(episodes):.2f} "
            f"[{low:.3f}, {high:.3f}] | {median} | {paired} | {p_value} |"
        )


if __name__ == "__main__":
    main()
