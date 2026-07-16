# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export deterministic OpenPI calibration requests from Cosmos3-DROID training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_CAMERA_KEYS = (
    "observation.image.wrist_image_left",
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
)
_DATA_COLUMNS = (
    "index",
    "episode_index",
    "frame_index",
    "task_index",
    "timestamp",
    "observation.state.joint_positions",
    "observation.state.gripper_position",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_path(root: Path, template: str, *, chunk_index: int, file_index: int, **fields: Any) -> Path:
    return root / template.format(chunk_index=chunk_index, file_index=file_index, **fields)


def _eligible_episodes(root: Path, info: dict[str, Any]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    episodes: list[dict[str, Any]] = []
    for metadata_file in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        for episode in pq.read_table(metadata_file).to_pylist():
            data_chunk = int(episode["data/chunk_index"])
            data_file = int(episode["data/file_index"])
            data_path = _available_path(
                root,
                str(info["data_path"]),
                chunk_index=data_chunk,
                file_index=data_file,
            )
            if not data_path.is_file():
                continue
            video_paths: dict[str, str] = {}
            for camera_key in _CAMERA_KEYS:
                prefix = f"videos/{camera_key}"
                video_path = _available_path(
                    root,
                    str(info["video_path"]),
                    video_key=camera_key,
                    chunk_index=int(episode[f"{prefix}/chunk_index"]),
                    file_index=int(episode[f"{prefix}/file_index"]),
                )
                if not video_path.is_file():
                    break
                video_paths[camera_key] = str(video_path)
            else:
                episode["_data_path"] = str(data_path)
                episode["_video_paths"] = video_paths
                episodes.append(episode)
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def _select_episodes(episodes: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--samples must be positive")
    if len(episodes) < count:
        raise ValueError(
            f"Only {len(episodes)} episodes have local parquet and all three RGB views; "
            f"{count} distinct training episodes were requested"
        )
    rng = random.Random(seed)
    # Stratification preserves broad coverage of the locally available shard while
    # randomizing the exact episode selected inside each equal-sized interval.
    selected: list[dict[str, Any]] = []
    for sample_index in range(count):
        begin = sample_index * len(episodes) // count
        end = (sample_index + 1) * len(episodes) // count
        selected.append(episodes[rng.randrange(begin, end)])
    return selected


def _select_frame(episode: dict[str, Any], rng: random.Random) -> tuple[int, int]:
    length = int(episode["length"])
    if length <= 0:
        raise ValueError(f"Episode {episode['episode_index']} has no frames")
    # Avoid reset/terminal frames while retaining temporal diversity.
    margin = min(max(length // 10, 1), max((length - 1) // 2, 0))
    frame_index = rng.randint(margin, max(margin, length - margin - 1))
    return int(episode["dataset_from_index"]) + frame_index, frame_index


def _read_selected_rows(selected: list[dict[str, Any]], global_indices: list[int]) -> dict[int, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = set(global_indices)
    value_set = pa.array(sorted(wanted), type=pa.int64())
    rows: dict[int, dict[str, Any]] = {}
    for data_path in sorted({str(episode["_data_path"]) for episode in selected}):
        table = pq.read_table(data_path, columns=list(_DATA_COLUMNS))
        for row in table.filter(pc.is_in(table["index"], value_set=value_set)).to_pylist():
            rows[int(row["index"])] = row
    missing = sorted(wanted - set(rows))
    if missing:
        raise KeyError(f"Selected DROID rows were not present in local parquet files: {missing[:8]}")
    return rows


def _decode_rgb_frame(
    video_path: str,
    absolute_frame: int,
    *,
    fps: float,
    height: int,
    width: int,
    ffmpeg: str,
) -> np.ndarray:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{absolute_frame / fps:.9f}",
        "-c:v",
        "libdav1d",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected_bytes = height * width * 3
    if result.returncode != 0 or len(result.stdout) != expected_bytes:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(
            f"Could not decode {video_path} frame {absolute_frame} with {ffmpeg}; "
            f"returncode={result.returncode} bytes={len(result.stdout)}/{expected_bytes}: {stderr}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _compose_views(wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import cv2

    if wrist.ndim != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"Wrist image must be HWC RGB, got {wrist.shape}")
    half_size = (wrist.shape[1] // 2, wrist.shape[0] // 2)
    left_small = cv2.resize(left, half_size, interpolation=cv2.INTER_LINEAR)
    right_small = cv2.resize(right, half_size, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.concatenate((wrist, np.concatenate((left_small, right_small), axis=1))))


def _task_variants(episode: dict[str, Any], task_by_index: dict[int, str], task_index: int) -> list[str]:
    values: list[str] = []
    for raw in [*episode.get("tasks", []), task_by_index.get(task_index, "")]:
        values.extend(part.strip() for part in str(raw).split(" | ") if part.strip())
    return list(dict.fromkeys(values))


def export_requests(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from openpi_client import msgpack_numpy

    root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a local LeRobot dataset root: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _eligible_episodes(root, info)
    selected = _select_episodes(episodes, args.samples, args.seed)

    rng = random.Random(args.seed)
    selections = [_select_frame(episode, rng) for episode in selected]
    rows = _read_selected_rows(selected, [global_index for global_index, _ in selections])
    task_table = pq.read_table(root / "meta/tasks.parquet", columns=["task_index", "task"])
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_table.to_pylist()}

    output_root.mkdir(parents=True, exist_ok=True)
    packer = msgpack_numpy.Packer()
    provenance_rows: list[dict[str, Any]] = []
    for sample_index, (episode, (global_index, frame_index)) in enumerate(zip(selected, selections)):
        row = rows[global_index]
        images: dict[str, np.ndarray] = {}
        video_provenance: dict[str, Any] = {}
        for camera_key in _CAMERA_KEYS:
            prefix = f"videos/{camera_key}"
            from_timestamp = float(episode[f"{prefix}/from_timestamp"])
            fps = float(info["fps"])
            absolute_frame = round((from_timestamp + frame_index / fps) * fps)
            video_path = str(episode["_video_paths"][camera_key])
            camera_info = info["features"][camera_key]["info"]
            images[camera_key] = _decode_rgb_frame(
                video_path,
                absolute_frame,
                fps=fps,
                height=int(camera_info["video.height"]),
                width=int(camera_info["video.width"]),
                ffmpeg=args.ffmpeg,
            )
            video_provenance[camera_key] = {
                "path": str(Path(video_path).relative_to(root)),
                "absolute_frame": absolute_frame,
            }
        task_index = int(row["task_index"])
        variants = _task_variants(episode, task_by_index, task_index)
        if not variants:
            raise ValueError(f"Episode {episode['episode_index']} has no training task text")
        prompt = variants[rng.randrange(len(variants))]
        request = {
            "observation/image": _compose_views(
                images["observation.image.wrist_image_left"],
                images["observation.image.exterior_image_1_left"],
                images["observation.image.exterior_image_2_left"],
            ),
            "observation/joint_position": np.asarray(row["observation.state.joint_positions"], dtype=np.float32),
            "observation/gripper_position": np.asarray(
                row["observation.state.gripper_position"], dtype=np.float32
            ),
            "prompt": prompt,
        }
        request_path = output_root / f"sample_{sample_index:05d}.request.msgpack"
        request_path.write_bytes(packer.pack(request))
        unpacked = msgpack_numpy.unpackb(request_path.read_bytes())
        if np.asarray(unpacked["observation/image"]).shape != (540, 640, 3):
            raise ValueError(f"Round-trip request image has wrong shape: {request_path}")
        provenance_rows.append(
            {
                "sample": sample_index,
                "request_file": request_path.name,
                "request_sha256": _sha256(request_path),
                "episode_index": int(episode["episode_index"]),
                "episode_id": str(episode.get("episode_id", "")),
                "frame_index": int(row["frame_index"]),
                "global_index": int(row["index"]),
                "timestamp": float(row["timestamp"]),
                "task_index": task_index,
                "prompt": prompt,
                "videos": video_provenance,
            }
        )

    provenance_path = output_root / "provenance.jsonl"
    provenance_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in provenance_rows), encoding="utf-8"
    )
    manifest = {
        "artifact_type": "cosmos3_robolab_training_calibration_requests",
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_split": "train",
        "dataset_local_root": str(root),
        "samples": len(provenance_rows),
        "distinct_episodes": len({row["episode_index"] for row in provenance_rows}),
        "seed": args.seed,
        "fps": float(info["fps"]),
        "ffmpeg": args.ffmpeg,
        "video_decoder": "libdav1d",
        "selection": "one central-80-percent frame from each stratified distinct episode",
        "request_schema": "OpenPI Cosmos3 RoboLab joint_pos with client-composed concat3 RGB",
        "provenance_file": provenance_path.name,
        "provenance_sha256": _sha256(provenance_path),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="nvidia/Cosmos3-DROID")
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="Immutable source dataset revision, such as a Hugging Face commit SHA",
    )
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    return parser


def main() -> None:
    manifest = export_requests(_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
