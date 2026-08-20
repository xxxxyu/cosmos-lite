<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboLab Training-Data Generation

Cosmos Lite can turn successful RoboLab rollouts into a LeRobot v3 dataset
that the Cosmos-DROID joint-position training loader reads directly. The
recorder keeps image memory bounded. In the measured 10-environment RTX 4090
setup, the current default adds `11.0%` wall time.

## What It Saves

Every training row is aligned as `observation[t] -> executed_action[t]`:

| Data            | Contract                                                      |
| --------------- | ------------------------------------------------------------- |
| RGB             | left shoulder, right shoulder, wrist; `640x360`; 15 FPS       |
| Robot state     | 7 joints, 1 gripper, 6 Cartesian values                       |
| Executed action | 7 joint-position commands, 1 gripper command                  |
| Episode data    | indexes, timestamps, task, seed, success, length, termination |

Only successful episodes enter `success/`. JSON and HDF5 sidecars retain the
initial simulator state, object poses, bounding boxes, camera extrinsics,
environment config, checkpoint provenance, timing, and failure reason.

The LeRobot dataset is the training input. HDF5 is a reproduction sidecar, not
an alternate training format.

The CLI uses a fixed Cosmos-DROID preset: three cameras, `640x360`, 15 FPS, and
the DROID state/action schema above. RoboLab's underlying streaming layer is
separate and configurable for custom LeRobot datasets, including stream
mappings, feature schemas, resolution, FPS, codec, quality, encoder threads per
stream, queue size, and FFmpeg options. See
`robolab/core/logging/lerobot_trajectory.py` after applying the integration
patches.

## 1. Install The RoboLab Integration

Use the exact RoboLab revision and apply both patches in order:

```bash
git -C "$ROBOLAB_DIR" switch --detach 9db0aaf
git -C "$ROBOLAB_DIR" am \
  /path/to/cosmos-lite/integrations/robolab/0001-bounded-cpu-streaming-image-recorder.patch \
  /path/to/cosmos-lite/integrations/robolab/0002-Add-training-ready-Cosmos-DROID-trajectory-export.patch

cd "$ROBOLAB_DIR"
uv sync --extra isaac50 --extra trajectory
```

The optional trajectory dependencies do not enter the Cosmos Lite policy
runtime. RoboLab's `uv` configuration preserves Isaac Sim's required
`packaging==23.0`; the pinned LeRobot APIs used by the recorder are tested with
that version.

## 2. Configure Provenance And Storage

The patched checkout includes
`examples/configs/droid_lerobot_trajectory.yaml`. Copy it to the run directory
and set `checkpoint` to the exact Hugging Face ID, local artifact digest, or
immutable revision served by Cosmos Lite. An unset value is rejected.

Keep `output_root: null` to place trajectories below the matching RoboLab run.
Write to node-local NVMe, validate the completed dataset, then copy it to
shared storage. Do not stream live rollout output directly to a network mount.

The release defaults are H.264, CRF 23, one encoder thread per camera, and a
bounded four-frame queue. A larger 16-frame queue did not improve throughput
and increased host memory by about 450 MiB in the qualified lane. LeRobot image
statistics sample every fourth temporal frame while every RGB frame remains in
the videos. On the 1,421-frame validation set, mean and standard-deviation
changes stayed below `0.00011`; quantiles changed by at most one 8-bit value.

## 3. Collect

Use the normal config-driven rollout command and add `TRAJECTORY_CONFIG`:

```bash
BUNDLE_DIR=/data/cosmos_lite/edge_genw8a8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/edge_genw8a8_fast_4090.yaml \
ROBOLAB_DIR=/path/to/RoboLab \
ROBOLAB_PYTHON=/path/to/RoboLab/.venv/bin/python \
TRAJECTORY_CONFIG=/data/cosmos_lite/runs/droid_trajectory.yaml \
RUN_DIR=/data/cosmos_lite/runs/droid_collection \
OUTPUT_NAME=droid_collection \
POLICY_GPU=0 SIM_GPU=1 \
TASK=BananaOnPlateTask NUM_ENVS=10 NUM_RUNS=1 \
examples/robolab_quant/pipeline.sh rollout
```

`VIDEO_MODE=none` is appropriate. It disables the human-viewable evaluation
video, not the three training camera streams. Do not enable RoboLab's separate
`--record-image-data` flag unless full-resolution RGB is also required in the
HDF5 sidecar; doing so adds a second, much heavier image pipeline.

## 4. Validate

The output is under:

```text
$ROBOLAB_DIR/output/droid_collection/
`-- droid_plus_lerobot_640x360_20260412/
    |-- success/                 # LeRobot v3 training dataset
    |   |-- data/
    |   |-- videos/
    |   `-- meta/
    |-- sidecars/
    |   |-- env_configs/
    |   |-- success/
    |   `-- failure/
    `-- manifest.json
```

Run the strict validator before training or transfer:

```bash
cd "$ROBOLAB_DIR"
uv run robolab-validate-droid-trajectory \
  output/droid_collection/droid_plus_lerobot_640x360_20260412
```

It checks schema, global and per-episode indexes, timestamps, H.264 frame
counts, success-only metadata, sidecars, checkpoint provenance, and exact
agreement between LeRobot actions and the executed HDF5 actions. Use
`--check-cosmos-loader` from an environment where Cosmos Framework is also
importable to exercise the real training loader.

An interrupted batch publishes no partial training episode. Failed temporary
videos are discarded; lightweight failure sidecars are retained by default.

## Qualified RTX 4090 Result

The fixed-window A/B used the same initial state, Edge GenW8A8 server, 10 active
RoboLab environments, and exactly 128 vector steps. The server and simulator
ran on separate RTX 4090 GPUs; evaluation video was disabled.

| Training trajectory | Stats frame stride | Wall time | Vector steps/s | Recorder time |
| ------------------- | -----------------: | --------: | -------------: | ------------: |
| Disabled            |                  - |  108.20 s |           1.18 |             - |
| Enabled             |                  1 |  125.04 s |           1.02 |        8.33 s |
| Enabled (default)   |                  4 |  120.14 s |           1.07 |        7.71 s |

The default recorder added `11.0%` wall time, down from `15.5%` with
full-frame statistics. The earlier enabled run used 496 MiB more simulator
VRAM and reached 10.11 GiB host RSS; the fixed-window stride-4 repeat did not
sample memory. A complete Banana-on-plate run produced 9 successful episodes,
1,421 aligned frames, and 27 H.264 videos in 37 MB. Storage size depends on
visual complexity, episode length, and CRF, so qualify the target task before
launching several lanes.

Leave CPU scheduling to the operating system for a single simulator process.
When several simulator processes share a host, assign each process a wide,
disjoint CPU/NUMA set and benchmark the complete lane. Narrow process affinity
also constrains its FFmpeg children and can slow both simulation and encoding.

## Optional Full-Resolution HDF5

The first integration patch also fixes RoboLab's original HDF5 image recorder:
recorded tensors move to CPU, histories are chunked, and writes are bounded.
This path is useful when a downstream consumer explicitly needs original
camera resolution or every episode, including failures:

```bash
python policies/cosmos3/run.py \
  --task BananaOnPlateTask --num-envs 10 --headless \
  --record-image-data --record-storage-profile fast
```

This is not the recommended Cosmos-DROID training path. It uses substantially
more disk bandwidth and must be qualified independently.
