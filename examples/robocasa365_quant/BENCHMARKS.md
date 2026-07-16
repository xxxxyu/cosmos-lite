<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboCasa365 Quantization and Inference Benchmarks

This document is the deployment-oriented benchmark record for the Cosmos 3
Nano CloseFridge policy. It compares closed-loop success, model memory, replay
latency, quantization strategies, and denoising settings. For setup, artifact,
replay, and rollout commands, see `README.md`; for the tag gate, see
`../quantized_robot_policy/RELEASE_CHECKLIST.md`.

## Benchmark Summary

### Quantization Comparison

The closed-loop table fixes the CloseFridge checkpoint, guidance 3.0, four
UniPC steps, 1,200 maximum episode steps, and 50 episodes. M12 and M13 are
independent repeated H100 runs under the same protocol.

| Strategy        | H100 SR M12 | H100 SR M13 | RTX 4090 peak alloc. (GB) | RTX 4090 replay request median (ms) |
| --------------- | ----------: | ----------: | ------------------------: | ----------------------------------: |
| `full_w8`       |     **96%** |     **96%** |                     19.20 |                           **1,176** |
| `full_w4`       |         92% |         92% |                 **12.47** |                               1,324 |
| `attention_w8`  |     **96%** |     **96%** |                     13.93 |                               1,231 |
| `gen_branch_w8` |         94% |         94% |                     15.84 |                               1,325 |

The success and performance columns intentionally name their hardware. H100
rollouts establish the repeated quality ranking; RTX 4090 replay establishes
the deployment memory and request-latency ranking. Absolute H100 and RTX 4090
latencies are not compared.

### RTX 4090 Sampler Comparison

These 50-episode rows fix the quantization strategy to `full_w8`. The
conservative guidance-3/four-step setting was established in the H100 gate but
was not repeated in this 4090 matrix, so it is not backfilled into the table.

| Denoise steps |   Guidance | Request median (ms) | CloseFridge SR |
| ------------: | ---------: | ------------------: | -------------: |
|             4 |        1.0 |                 479 |        **90%** |
|      <u>2</u> | <u>3.0</u> |          <u>472</u> |            84% |
|             2 |        1.0 |             **304** |            78% |

The confidence intervals overlap. Keep guidance 3.0 / four steps as the
conservative default; treat faster samplers as checkpoint-specific candidates
until repeated paired rollout resolves the ranking.

### Rollout Reference

[![Watch the RoboCasa365 CloseFridge reference](../../docs/assets/robocasa_closefridge_bf16_reference_poster.png)](../../docs/assets/robocasa_closefridge_bf16_reference.mp4)

[Open the CloseFridge reference video](../../docs/assets/robocasa_closefridge_bf16_reference.mp4).
This cherry-picked three-episode video comes from the step-8000 BF16 reference
evaluation. It demonstrates the task and observation layout; it is not a
quantized rollout comparison. Quantized quality claims use the full matched
tables above.

## Scope and Measurement Rules

Unless a table says otherwise:

- Model checkpoint: `iter_000008000` from the CloseFridge single-task SFT.
- Quantization is packed weight-only W4A16/W8A16. Activations remain BF16.
- Quantized Linear backend: vLLM Marlin direct load.
- The four release strategies quantize 504 language/MoT Linear modules.
- One policy request returns an action chunk; rollout executes 8 actions before
  requesting the next chunk.

The reported latency fields have different boundaries:

- **Generate latency** is model preprocessing/inference/postprocessing inside
  the policy server's generate call.
- **Request latency** includes the client/server request path around generate.
- **Rollout success** is closed-loop task completion and is the quality gate.
- **Replay action error** is an open-loop diagnostic against saved BF16
  responses. It is not a success-rate proxy.
- **Peak allocated** is PyTorch CUDA allocated memory. Reserved memory can be
  higher and is shown where it was captured.

Do not compare absolute latency across H100 and RTX 4090, across different
software stacks, or between `N_ENVS=5` rollout and single-request replay. Use
only rows in the same table/runtime environment for latency ranking.

## Release Pipeline Smoke

The release entry point was rebuilt on one RTX 4090 from the 8,000-step BF16
DCP. This gate validates the product pipeline itself; the larger rollout
tables below remain the quality evidence.

| Strategy       |           Calibration |  Packed modules |   Bundle bytes | Stream-pack peak alloc/reserved |
| -------------- | --------------------: | --------------: | -------------: | ------------------------------: |
| `full_w8`      |          not required |   0 W4 / 504 W8 | 19,292,105,862 |                   0.654/0.656GB |
| `attention_w8` | 128 training captures | 216 W4 / 288 W8 | 14,025,041,490 |                   0.808/1.034GB |

Both one-command builds completed residual conversion and strong hash/tensor
validation. The mixed build first direct-loaded a temporary W8 model under
24GB, exercised all 504 Linear hooks on all 128 CloseFridge training requests,
then removed every temporary packed artifact and calibration-stat file.

The newly built `full_w8` bundle returned finite actions for 2/2 replay
requests. Post-load memory was 17.88GB allocated; peak inference memory was
19.21/19.58GB allocated/reserved. The one steady measured generation took
1,107ms; use the larger latency samples below for deployment estimates.

A fresh Git checkout and fresh locked runtime on a dual-GPU RTX 4090 host then
ran the public `pipeline.sh rollout` entry point with `attention_w8`, guidance 3.0,
four steps, one environment, and the 1,200-step horizon. CloseFridge succeeded
1/1; collection took 106.8s, the episode progress bar took 77.9s, steady
generation was about 0.80-0.83s, and peak allocated/reserved memory was
13.94/14.28GB. This single episode is a deployment smoke, not additional
quality evidence.

## Quantization Strategies

| Strategy        | W4 modules | W8 modules | Precision map                                | Deployment role                    |
| --------------- | ---------: | ---------: | -------------------------------------------- | ---------------------------------- |
| `full_w8`       |          0 |        504 | All language/MoT Linear W8A16                | Quality-first, highest memory      |
| `full_w4`       |        504 |          0 | All language/MoT Linear W4A16                | Minimum-memory frontier            |
| `attention_w8`  |        216 |        288 | All self-attention W8A16, MLP W4A16          | Recommended memory/quality balance |
| `gen_branch_w8` |        252 |        252 | MoT generation branch W8A16, remainder W4A16 | Intermediate memory option         |

Every W4-containing strategy uses training-set calibration during export: 128
CloseFridge frames, activation-aware input-channel scaling, and `alpha=0.5`.
`full_w8` has no calibration-dependent input scaling. Calibration frames and
RGB videos came from the RoboCasa365 training split, not evaluation captures.
Activation quantization is not enabled.

## H100 Closed-Loop Quality Gate

M12 and M13 are two complete repeats under the same corrected protocol:

```text
task/split: CloseFridge/target
N_EPISODES: 50
N_ENVS: 5
N_ACTION_STEPS: 8
MAX_EPISODE_STEPS: 1200
USE_TASK_HORIZON: 0
video: disabled
guidance: 3.0
UniPC steps: 4
calibration: 128 training frames, alpha=0.5
```

`USE_TASK_HORIZON=0` is essential. Enabling the task horizon reduced the
effective horizon to 600 and materially changed the result.

| Strategy               |  M12 success |  M13 success | Post-load alloc | Peak alloc | M13 generate p50/p95 |
| ---------------------- | -----------: | -----------: | --------------: | ---------: | -------------------: |
| BF16                   | 45/50 = 0.90 | 45/50 = 0.90 |         31.76GB |    33.09GB |            740/911ms |
| `full_w8`              | 48/50 = 0.96 | 48/50 = 0.96 |         17.88GB |    19.20GB |            814/980ms |
| `full_w4`              | 46/50 = 0.92 | 46/50 = 0.92 |         11.15GB |    12.48GB |           830/1038ms |
| `attention_w8`         | 48/50 = 0.96 | 48/50 = 0.96 |         12.61GB |    13.94GB |           822/1020ms |
| `gen_branch_w8`        | 47/50 = 0.94 | 47/50 = 0.94 |         14.52GB |    15.84GB |           839/1035ms |
| `mixed_best` (demoted) | 42/50 = 0.84 | not repeated |         15.24GB |    16.57GB |       M12 820/1025ms |

Interpretation:

- All retained quantized strategies fit the 24GB allocation target; BF16 does
  not.
- `full_w8` and `attention_w8` had the highest point estimate and reproduced
  exactly across these two runs.
- `full_w4` retained the external 0.92 reference success while using the least
  memory.
- A quantized point estimate above BF16 does not establish that quantization
  improves the policy. With 50 Bernoulli trials the confidence intervals are
  broad, and simulator seeds/trajectory divergence introduce variance.
- The H100 Marlin path reduced memory but was not faster than BF16. The 4090
  measurements below are the relevant deployment latency results.

## RTX 4090 Quantization Replay

The first replay32 comparison used one local 4090 runtime, guidance 3.0,
4 UniPC steps, batch size 1, and 8 served action steps.

| Strategy        | Peak alloc/reserved | Request p50/p95 | Generate p50/p95 |
| --------------- | ------------------: | --------------: | ---------------: |
| `full_w4`       |       12.47/12.75GB |     1324/1490ms |      1108/1191ms |
| `full_w8`       |       19.20/19.50GB |     1176/1453ms |      1015/1098ms |
| `attention_w8`  |       13.93/14.28GB |     1231/1523ms |      1029/1143ms |
| `gen_branch_w8` |       15.84/16.11GB |     1325/1448ms |      1048/1138ms |

A later fresh-checkout NX validation tested schema-v2 self-contained bundles.
These rows validate direct loading without any source DCP/config dependency.
They came from a different runtime environment and must not be merged into the
latency ranking above.

| Bundle                 |  On-disk bytes | Peak alloc/reserved | Generate p50/p95 |        Legacy parity |
| ---------------------- | -------------: | ------------------: | ---------------: | -------------------: |
| `attention_w8_v2_fast` | 14,025,041,490 |       13.93/14.28GB |    792.7/803.7ms | 32/32 byte-identical |
| `full_w8_v2_fast`      | 19,292,105,862 |       19.20/19.50GB |    786.3/790.8ms | 32/32 byte-identical |

The exact parity result verifies artifact conversion and direct-load behavior;
it does not mean the quantized output is byte-identical to BF16.

## Sampling and Denoiser Acceleration

For this path, the first-order denoiser cost is:

```text
forwards_per_request = num_steps * (2 if guidance > 1 else 1)
```

Guidance above 1 uses conditional and unconditional classifier-free guidance
passes. Guidance 1 removes the unconditional pass. Reducing UniPC steps
coarsens numerical integration of the denoising trajectory. Both alter model
behavior and therefore require rollout validation.

Replay32 quality/latency gate on one RTX 4090 with `attention_w8`:

| Guidance | Steps | Forwards/request | Request p50/p95 | Generate p50/p95 | Generate speedup |
| -------: | ----: | ---------------: | --------------: | ---------------: | ---------------: |
|      3.0 |     4 |                8 |     1250/1419ms |      1063/1190ms |            1.00x |
|      1.0 |     4 |                4 |       771/985ms |        614/666ms |            1.73x |
|      3.0 |     2 |                4 |       793/951ms |        633/694ms |            1.68x |
|      1.0 |     2 |                2 |       532/695ms |        382/444ms |            2.78x |

Short H100 rollout gate, 5 episodes per setting:

| Guidance | Steps | Success | Generate p50/p95 |
| -------: | ----: | ------: | ---------------: |
|      1.0 |     4 |     5/5 |        419/444ms |
|      3.0 |     2 |     5/5 |        430/445ms |
|      1.0 |     2 |     4/5 |        242/262ms |

## RTX 4090 Full Rollout Matrix

This is the same-machine comparison most relevant to 4090 deployment. Each row
used a schema-v2 self-contained bundle and the following protocol:

```text
GPU: RTX 4090 24GB
N_EPISODES: 50
N_ENVS: 1
N_ACTION_STEPS: 8
MAX_EPISODE_STEPS: 1200
USE_TASK_HORIZON: 0
DETERMINISTIC_SEED: 0
USE_TORCH_COMPILE: 0
VIEW_MODE: concat3
```

| Strategy       | Guidance | Steps |      Success |  Wilson 95% CI | Request p50/p95 | Generate p50/p95 |
| -------------- | -------: | ----: | -----------: | -------------: | --------------: | ---------------: |
| `attention_w8` |      1.0 |     4 | 40/50 = 0.80 | [0.670, 0.888] |       484/501ms |        474/488ms |
| `attention_w8` |      3.0 |     2 | 43/50 = 0.86 | [0.738, 0.930] |       492/516ms |        481/502ms |
| `attention_w8` |      1.0 |     2 | 41/50 = 0.82 | [0.692, 0.902] |       302/322ms |        291/308ms |
| `full_w8`      |      1.0 |     4 | 45/50 = 0.90 | [0.786, 0.957] |       479/496ms |        458/471ms |
| `full_w8`      |      3.0 |     2 | 42/50 = 0.84 | [0.715, 0.917] |       472/495ms |        460/479ms |
| `full_w8`      |      1.0 |     2 | 39/50 = 0.78 | [0.648, 0.872] |       304/318ms |        283/291ms |

The confidence intervals overlap. A prior local `attention_w8` repeat reached
44/50 for both guidance 1.0/steps 4 and guidance 3.0/steps 2, showing that a
single 50-episode run cannot resolve small quality differences. The consistent
conclusion is narrower: guidance 1.0/steps 2 is the latency leader but has not
been shown quality-neutral; the two four-forward settings are safer balanced
candidates.

## Acceleration Methods Not Promoted

| Method                      | Main result                                                             | Decision                                     |
| --------------------------- | ----------------------------------------------------------------------- | -------------------------------------------- |
| Graph-break `torch.compile` | ~1031ms eager vs ~1053ms compiled generate; ~20.2s first request        | Keep eager default                           |
| Action chunk 16/8           | Lower linear-work proxy, worse measured latency                         | Do not deploy                                |
| Camera pre-resize to 192    | No material speedup; model still consumes 256 resolution                | Do not deploy                                |
| Wrist-only/fewer views      | Changed policy input and quality without compelling speedup             | Do not deploy                                |
| Dynamic FP8 W8A8            | Did not beat Marlin on the full real-shape mix                          | No production integration                    |
| Selective static-scale FP8  | 30.7% large-MLP operator saving; estimated 14.5% end-to-end upper bound | Insufficient benefit for added A8 complexity |
| PyTorch INT8 `_int_mm`      | Slower and unsupported for the token-10 shape                           | Reject backend                               |

Nsight attribution found the denoiser dominant, with Marlin kernels about
65.9% of GPU kernel time, eager elementwise/copy about 19%, and attention
kernels only about 4.4%. This explains why changing denoiser forward count
produced 1.7-2.8x gains while attention-only or token-count tweaks did not.

## Deployment Selection

| Priority                      | Recommended setting                                | Evidence and tradeoff                                                    |
| ----------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| Conservative memory/quality   | `attention_w8`, guidance 3.0, 4 steps              | 13.94GB peak; repeated H100 success 0.96; no sampler approximation       |
| Quality-first under 24GB      | `full_w8`, guidance 3.0, 4 steps                   | 19.20GB peak; repeated H100 success 0.96                                 |
| Minimum memory                | `full_w4`, guidance 3.0, 4 steps                   | 12.48GB peak; repeated H100 success 0.92                                 |
| Balanced experimental latency | `full_w8` or `attention_w8`, guidance 1.0, 4 steps | Four denoiser forwards; 4090 matrix favors full-W8 but intervals overlap |
| Alternative balanced sampler  | guidance 3.0, 2 steps                              | Also four forwards; preserves CFG but uses coarser integration           |
| Latency-first, quality-risky  | guidance 1.0, 2 steps                              | ~0.29s sustained 4090 generate, but 50-episode quality was lower/noisy   |

For a new robot, task, checkpoint, runtime, or sampler setting, repeat in this
order: artifact validation, replay32, 2-5 episode smoke, then at least one
50-episode closed-loop gate. Never infer closed-loop quality from latency or
open-loop action error alone.
