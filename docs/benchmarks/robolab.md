<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboLab-120 Benchmark

This is the primary release benchmark for Cosmos3 Nano and Edge DROID policies
in Cosmos Lite. It reports four deployment profiles under one common protocol.
Quantization and sampler comparisons are kept in
[RoboLab ablations](robolab_ablations.md).

## Main Results

| Model    | Artifact    | VRAM (GB) | Request p50 (ms) |   Successes | SR (%) |  95% CI (%) |
| -------- | ----------- | --------: | ---------------: | ----------: | -----: | ----------: |
| Edge 4B  | BF16        |      9.20 |            582.0 | 251 / 1,200 |  20.92 | 18.71-23.31 |
| Edge 4B  | **GenW8A8** |      8.79 |        **331.4** | 231 / 1,200 |  19.25 | 17.12-21.58 |
| Nano 16B | W8A16       |     21.42 |          2,403.0 | 378 / 1,200 |  31.50 | 28.93-34.18 |
| Nano 16B | **GenW8A8** | **15.51** |        **958.5** | 380 / 1,200 |  31.67 | 29.10-34.35 |

The Nano confidence intervals overlap almost completely. GenW8A8 therefore
retains the measured success rate while materially reducing memory and request
latency. Edge BF16 has the higher point estimate, but its interval overlaps
GenW8A8. Use Edge BF16 for original-weight inference and Edge GenW8A8 when
latency matters.

The overlapping confidence intervals support comparable aggregate success,
with the deployment choice driven by latency, memory, and weight provenance.

## What Was Measured

Request latency and success rate have different boundaries:

- **Request latency** is measured with one batch-one policy instance on one
  RTX 4090. It begins when the server receives an observation and ends when the
  32x8 action response is ready. Simulator time is excluded.
- **VRAM** is peak CUDA reserved memory for that policy instance.
- **Success rate** uses all 120 RoboLab tasks in `default` instruction mode,
  with 10 episodes per task. Full-suite evaluation replicates the same policy
  profile across multiple servers to improve throughput.

## Common Protocol

| Setting      | Value                                 |
| ------------ | ------------------------------------- |
| GPU          | RTX 4090 24GB, SM89                   |
| Sampler      | guidance 3, two UniPC steps, shift 5  |
| Seed         | 0, deterministic per request          |
| Client input | 640x540 RGB composition, three views  |
| Model bucket | 736x544 (W x H)                       |
| Action chunk | 32x8, all 32 actions executed         |
| Tasks        | 120, `default` instruction mode       |
| Episodes     | 10 per task, 1,200 per profile        |
| Horizon      | RoboLab task default, 20-300s at 15Hz |

Model precision is fixed by the artifact. The sampler is selected at runtime.
Changing denoise steps does not rebuild or modify the checkpoint.

## Release Profiles

**Nano GenW8A8** uses `nano_genw8a8_fast_4090.yaml` and is the recommended
tradeoff. **Nano W8A16** uses `nano_w8.yaml` and is the calibration-free 24GB
option.

**Edge GenW8A8** uses `edge_genw8a8_fast_4090.yaml` and is the low-latency
option. **Edge BF16** uses `edge_bf16.yaml` and preserves NVIDIA's original
weights.

GenW8A8 applies dynamic per-token FP8 W8A8 to the action-generation branch and
calibrated packed W4A16 to the remaining target layers. Calibration uses 128
distinct episodes from revision `5c11a20accb11497270a5247a7f1e66ad04c956c`
of `nvidia/Cosmos3-DROID` `train/success`. It does not update model parameters
or use RoboLab evaluation episodes. W8A16 is weight-only and calibration-free.

## Evaluation Topology

OmniGibson simulator steps dominate one-environment wall time, so full-suite
evaluation shares policy servers across simulator lanes.

| Profile      | Policy GPUs | Simulator GPUs | Environments per simulator |
| ------------ | ----------: | -------------: | -------------------------: |
| Nano W8A16   |           3 |              5 |                         10 |
| Nano GenW8A8 |           2 |              6 |                         10 |
| Edge BF16    |           2 |              6 |                         10 |
| Edge GenW8A8 |           2 |              6 |                         10 |

Topology changes throughput, not the episode protocol or success accounting.
These layouts were measured with training-trajectory recording disabled.
Saving LeRobot training streams or full-resolution HDF5 images changes
simulator GPU, host-memory, CPU, and disk requirements; follow the separate
[data-generation guide](../../examples/robolab_quant/DATA_GENERATION.md).

## Runtime Stability

Each profile completed 128 requests in one server process. The first lazy
initialization request is excluded. Shared-host latency below is a stability
check and is not substituted for the idle-GPU headline latency.

| Model    | Artifact | p50 / p95 (ms) | Last / peak VRAM (GB) |
| -------- | -------- | -------------: | --------------------: |
| Edge 4B  | BF16     |      579 / 593 |           9.21 / 9.21 |
| Edge 4B  | GenW8A8  |      408 / 420 |           5.70 / 8.79 |
| Nano 16B | W8A16    |  1,746 / 1,758 |         19.07 / 19.07 |
| Nano 16B | GenW8A8  |  1,010 / 1,015 |         15.51 / 15.51 |

All four runs completed without a crash, out-of-memory error, non-finite
action, unexpected fallback, or increasing reserved memory.

See the [deployment guide](../../examples/robolab_quant/README.md) to reproduce
the runtime workflow and [runtime architecture](../runtime_architecture.md) for
implementation details.
