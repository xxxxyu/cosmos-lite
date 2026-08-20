# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail when public-release invariants regress in tracked text files."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = {
    "/home/lixiangyu": "private home path",
    "/mnt/100T": "private shared-storage path",
    "/mnt/lixiangyu": "private mount path",
    "4090-NX": "private host name",
    "DINGXIN_SITE": "private deployment identifier",
}
QUANT_PATH_PARTS = ("quant", "action_policy_server_robolab.py")
REQUIRED_SPDX_PATHS = (
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "UPSTREAM_README.md",
    "examples/quantized_robot_policy/README.md",
    "examples/quantized_robot_policy/RELEASE_CHECKLIST.md",
    "examples/robocasa365_quant/README.md",
    "examples/robocasa365_quant/BENCHMARKS.md",
    "examples/robolab_quant/README.md",
    "examples/robolab_quant/DATA_GENERATION.md",
    "integrations/robolab/README.md",
    "docs/README.md",
    "docs/benchmarks/robolab.md",
    "docs/benchmarks/robolab_ablations.md",
    "docs/model_build.md",
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    )
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def main() -> int:
    failures: list[str] = []
    paths = tracked_files()
    tracked_relative = {path.relative_to(ROOT).as_posix() for path in paths}
    for path in paths:
        if not path.is_file():
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for needle, reason in FORBIDDEN_TEXT.items():
            if needle in text:
                failures.append(f"{relative}: contains {reason}: {needle!r}")
        if any(part in relative for part in QUANT_PATH_PARTS):
            if "weights_only=False" in text:
                failures.append(f"{relative}: unsafe torch/easy_io load mode")

    lfs = subprocess.check_output(
        ["git", "lfs", "ls-files", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    missing = [
        item for item in lfs if item in tracked_relative and not (ROOT / item).is_file()
    ]
    if missing:
        failures.append("missing LFS worktree files: " + ", ".join(missing))

    for relative in REQUIRED_SPDX_PATHS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "SPDX-License-Identifier: OpenMDW-1.1" not in text:
            failures.append(f"{relative}: missing OpenMDW-1.1 SPDX header")

    if failures:
        print("Public release check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
