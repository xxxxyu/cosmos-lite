<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite RoboLab Ablations

This document preserves the experiments that informed the public deployment
profiles. These tables compare quantization and runtime choices. The four
promoted profiles and their common 1,200-rollout protocol are in the
[main benchmark](robolab.md).

## Full-Suite Denoise Control

These rows cover all 120 RoboLab tasks, 10 episodes per task, in `default`
instruction mode. Within each model artifact, only the number of UniPC denoise
steps changes. Guidance remains 3 and shift remains 5.

| Model    | Artifact | Steps | Successes | SR (%) |  95% CI (%) |
| -------- | -------- | ----: | --------: | -----: | ----------: |
| Nano 16B | W8A16    |     2 |       378 |  31.50 | 28.93-34.18 |
| Nano 16B | W8A16    |     4 |       373 |  31.08 | 28.53-33.76 |
| Nano 16B | GenW8A8  |     2 |       380 |  31.67 | 29.10-34.35 |
| Nano 16B | GenW8A8  |     4 |       382 |  31.83 | 29.26-34.52 |
| Edge 4B  | BF16     |     2 |       251 |  20.92 | 18.71-23.31 |
| Edge 4B  | BF16     |     4 |       260 |  21.67 | 19.43-24.09 |
| Edge 4B  | W8A16    |     2 |       239 |  19.92 | 17.75-22.27 |
| Edge 4B  | W8A16    |     4 |       240 |  20.00 | 17.83-22.36 |
| Edge 4B  | GenW8A8  |     2 |       231 |  19.25 | 17.12-21.58 |
| Edge 4B  | GenW8A8  |     4 |       238 |  19.83 | 17.68-22.18 |

At full-suite scale, two and four steps have overlapping confidence intervals
for every retained artifact. Two steps are the release default because they
substantially reduce request latency and do not show a meaningful aggregate SR
loss. The stronger two-step gain once observed on Banana does not generalize
as an SR improvement across all 120 tasks.

## Edge Banana: Quantization At Two Steps

This paired 50-episode `BananaInBowlTask` experiment compares older weight-only
strategies at guidance 3 and two denoise steps. `Avg. episode steps` is the
mean `final_step` over all 50 episodes; failures contribute the 750-step
horizon. It therefore captures both completion speed and the evaluation cost
of failures.

| Artifact     | VRAM (GB) | Latency (ms) | Successes | SR (%) | Mean episode steps |
| ------------ | --------: | -----------: | --------: | -----: | -----------------: |
| W8A16        |  **8.71** |          576 |        36 |     72 |          **393.8** |
| W4A16        |      8.87 |      **563** |        37 |     74 |              422.2 |
| W4A16-AttnW8 |      8.74 |          571 |        29 |     58 |              492.6 |
| W4A16-GenW8  |      8.79 |          570 |        40 | **80** |              398.2 |

The 50-episode intervals overlap for W8A16, W4A16, and W4A16-GenW8. These
older artifacts remain reproducible development strategies but do not add a
clear deployment role beside BF16 and GenW8A8.

## Edge Banana: Sampler Choice

The same 50 initial states were used across rows. The table again averages
`final_step` over all episodes and counts failures at 750 steps.

| Artifact | Guidance | Steps | Latency (ms) | Successes | SR (%) | Mean episode steps |
| -------- | -------: | ----: | -----------: | --------: | -----: | -----------------: |
| BF16     |        3 |     4 |        1,042 |        21 |     42 |              579.6 |
| BF16     |        3 |     2 |          582 |        34 |     68 |              463.9 |
| W8A16    |        3 |     4 |        1,017 |        29 |     58 |              548.0 |
| W8A16    |        3 |     2 |          576 |        36 |     72 |          **393.8** |
| W8A16    |        1 |     4 |          576 |         4 |      8 |              732.4 |
| W8A16    |        1 |     2 |      **367** |        11 |     22 |              647.6 |

On this task, two steps improved both request latency and closed-loop outcome.
Removing classifier-free guidance reduced request latency further but caused a
large failure increase, so guidance 1 is not a release preset.

## Nano Banana: Legacy Weight-Only Strategies

The original Nano comparison used 50 paired episodes at guidance 3 and four
steps. The raw episode JSON used for the Edge mean-step calculation is no
longer retained for these legacy Nano runs, so this table reports the
previously audited median step among successful episodes instead of presenting
an unrecoverable mean.

| Artifact     | VRAM (GB) | Latency (ms) | Successes | SR (%) | Median successful steps |
| ------------ | --------: | -----------: | --------: | -----: | ----------------------: |
| W8A16        |     21.42 |        4,110 |        43 |     86 |                     213 |
| W4A16        | **14.67** |        4,248 |        26 |     52 |                   292.5 |
| W4A16-AttnW8 |     16.21 |        4,153 |        42 |     84 |                   373.5 |
| W4A16-GenW8  |     18.03 |    **4,104** |        45 | **90** |                 **208** |

Full W4 showed a clear quality loss. AttentionW8 and generation-branch W8A16
were useful sensitivity probes, but GenW8A8 now provides a stronger,
non-redundant speed and memory profile and is the promoted mixed artifact.

## Nano Banana: W8A16 Sampler Choice

| Guidance | Steps | Latency (ms) | Successes | SR (%) |
| -------: | ----: | -----------: | --------: | -----: |
|        3 |     4 |        4,110 |        43 |     86 |
|        3 |     2 |        2,403 |        45 | **90** |
|        1 |     4 |        2,431 |        32 |     64 |
|        1 |     2 |    **1,565** |        40 |     80 |

As on Edge, guidance 3 with two steps retained CFG and offered the best
measured speed/quality balance. Full-suite results above remain the basis for
the public sampler recommendation.

## GenW8A8 Focused Gate

Before the full RoboLab-120 campaign, dynamic GenW8A8 passed a paired
50-episode Banana gate under guidance 3 and two steps.

| Model    | Runtime candidate            | Successes / 50 | SR (%) | Wilson 95% CI (%) |
| -------- | ---------------------------- | -------------: | -----: | ----------------: |
| Nano 16B | GenW8A8 eager                |             47 |     94 |         83.5-97.9 |
| Edge 4B  | GenW8A8 eager                |             39 |     78 |         64.8-87.2 |
| Edge 4B  | GenW8A8 + tuned Sage FP16-PV |             40 |     80 |         67.0-88.8 |

These task-local gates qualified the optimized runtime for the full-suite
evaluation. Public success rates come from the 1,200-rollout results above.

## Open-Loop Diagnostics

Replay action error is useful for catching broken quantization but is not a
success-rate proxy:

- `L1 mean` averages absolute error over every action step and dimension.
- `Linf p95` takes the largest element error in each request, then reports the
  95th percentile across requests.

Legacy Edge replay at guidance 3 / four steps produced:

| Artifact     | Request p50/p95 | Peak VRAM (GB) | L1 mean | Linf p95 |
| ------------ | --------------: | -------------: | ------: | -------: |
| BF16         |   1,042/1,047ms |           9.20 |       0 |        0 |
| W8A16        |   1,017/1,032ms |           8.71 | 0.01864 |   0.4093 |
| W4A16        |   1,040/1,047ms |           8.87 | 0.03214 |   0.5400 |
| W4A16-AttnW8 |   1,043/1,051ms |           8.74 | 0.03133 |   0.3397 |
| W4A16-GenW8  |     988/1,009ms |           8.79 | 0.02352 |   0.4404 |

## Detailed Experiment Records

- [FP8 W8A8 exploration](../experiments/fp8_w8a8.md)
- [Compile and CUDA Graph exploration](../experiments/graph_optimization.md)
- [RTX 4090 SM89 kernel and graph optimization](../experiments/rtx4090_sm89.md)
- [Full optimization report](../cosmos_lite_optimization_report.md)

These records archive experimental backends and rejected variants for
reproduction. They stay outside the release quickstart and main benchmark.
