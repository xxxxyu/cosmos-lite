# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import random

import numpy as np
import pytest

from cosmos_framework.scripts.export_robolab_train_calibration_requests import (
    _compose_views,
    _parser,
    _select_episodes,
    _select_frame,
    _task_variants,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_select_episodes_is_distinct_deterministic_and_stratified() -> None:
    episodes = [{"episode_index": index} for index in range(40)]

    first = _select_episodes(episodes, 8, seed=7)
    second = _select_episodes(episodes, 8, seed=7)

    assert first == second
    assert len({item["episode_index"] for item in first}) == 8
    for interval, item in enumerate(first):
        assert interval * 5 <= item["episode_index"] < (interval + 1) * 5


def test_select_episodes_rejects_insufficient_multiview_data() -> None:
    with pytest.raises(ValueError, match="Only 2 episodes"):
        _select_episodes([{"episode_index": 0}, {"episode_index": 1}], 3, seed=0)


def test_select_frame_avoids_episode_edges() -> None:
    global_index, frame_index = _select_frame(
        {"episode_index": 3, "length": 100, "dataset_from_index": 500}, random.Random(0)
    )

    assert 10 <= frame_index < 90
    assert global_index == 500 + frame_index


def test_compose_views_matches_robolab_layout() -> None:
    wrist = np.full((4, 6, 3), 10, dtype=np.uint8)
    left = np.full((4, 6, 3), 20, dtype=np.uint8)
    right = np.full((4, 6, 3), 30, dtype=np.uint8)

    result = _compose_views(wrist, left, right)

    assert result.shape == (6, 6, 3)
    assert np.all(result[:4] == 10)
    assert np.all(result[4:, :3] == 20)
    assert np.all(result[4:, 3:] == 30)


def test_task_variants_deduplicates_episode_and_task_table_text() -> None:
    result = _task_variants(
        {"tasks": ["place cup | move cup"]},
        {5: "move cup | put cup down"},
        5,
    )

    assert result == ["place cup", "move cup", "put cup down"]


def test_dataset_revision_is_required() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-root", "/dataset", "--output-dir", "/output"])

    args = parser.parse_args(
        [
            "--dataset-root",
            "/dataset",
            "--output-dir",
            "/output",
            "--dataset-revision",
            "immutable-commit",
        ]
    )
    assert args.dataset_revision == "immutable-commit"
