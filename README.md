<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite

**Deploy Cosmos 3 robot policies on a single 24GB RTX 4090.** Cosmos Lite
supports both the 4B Cosmos3 Edge and 16B Cosmos3 Nano policy families with
streaming low-bit quantization, packed Marlin inference, self-contained
artifacts, and reproducible RoboLab and RoboCasa365 evaluation pipelines for
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework).

**[Models](#prebuilt-models) | [RoboLab Nano benchmark](examples/robolab_quant/NANO_BENCHMARKS.md) | [RoboLab Edge benchmark](examples/robolab_quant/EDGE_BENCHMARKS.md) | [RoboCasa365 benchmark](examples/robocasa365_quant/BENCHMARKS.md) | [Setup](#setup) | [Roadmap](#roadmap)**

## News

- **2026-07-22:** Cosmos Lite now supports W4A16, W8A16, and mixed-precision
  inference for the 4B Cosmos3 Edge DROID policy on RTX 4090, with end-to-end
  deployment and evaluation in RoboLab.
- **2026-07-16:** [Cosmos Lite v0.1.0](https://github.com/xxxxyu/cosmos-lite/releases/tag/v0.1.0)
  brought W4A16, W8A16, and mixed-precision inference for the 16B Cosmos3 Nano
  policy to a single 24GB RTX 4090, with reproducible RoboLab and RoboCasa365
  deployment and evaluation.

## RoboLab Nano Rollout Demo

<!-- rumdl-disable MD034 -->
https://github.com/user-attachments/assets/76e09f1c-4f20-47e5-ac09-6dc64f30d7a3
<!-- rumdl-enable MD034 -->

Each Nano quantized setting was evaluated for 50 paired Banana rollouts at
guidance 3 and four denoise steps. The demo shows three selected episodes per
setting; the success rates below come from all 50 rollouts, not the displayed
subset.

## RoboLab Edge Rollout Demo

<!-- rumdl-disable MD034 -->
https://github.com/user-attachments/assets/12f2ef89-6576-4b92-9114-22b500382272
<!-- rumdl-enable MD034 -->

Each Edge quantized setting was evaluated for 50 paired Banana rollouts at
guidance 3 and two denoise steps. The demo shows three selected episodes per
setting; the success rates below come from all 50 rollouts, not the displayed
subset.

## At A Glance

| Env & Task              | Quant.       | Denoise | VRAM (GB) | Latency (ms) | SR (%) |
| ----------------------- | ------------ | ------: | --------: | -----------: | -----: |
| RoboLab Banana (Edge)   | W8A16        |       2 |      8.71 |          576 |    72% |
| RoboLab Banana (Edge)   | W4A16-GenW8  |       2 |      8.79 |          570 |    80% |
| RoboLab Banana (Nano)   | W8A16        |       2 |     21.42 |        2,403 |    90% |
| RoboLab Banana (Nano)   | W4A16-GenW8  |       2 |     18.03 |        2,433 |   100% |
| RoboCasa365 CloseFridge | W4A16-AttnW8 |       4 |     14.28 |        1,231 |    96% |

VRAM is peak CUDA reserved memory and latency is p50 end-to-end policy request
latency on RTX 4090. RoboLab success rates are 50-rollout RTX 4090 results.
The RoboCasa success rate was reproduced in two 50-episode H100 runs with the
same quantization and sampler settings; its VRAM and latency are RTX 4090
measurements. All rows use guidance 3.

## RoboLab Nano Quantization Comparison

| Quant.                                                                                      | VRAM (GB) | Latency (ms) |  SR (%) |
| ------------------------------------------------------------------------------------------- | --------: | -----------: | ------: |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16)               |     21.42 |        4,110 |     86% |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16)               | **14.67** |        4,248 |     52% |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-AttnW8) |     16.21 |        4,153 |     84% |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-GenW8)   |     18.03 |    **4,104** | **90%** |

This comparison fixes guidance to 3 and denoise steps to 4. Request latency is
p50 on RTX 4090; success rates use 50 paired Banana rollouts.

## RoboLab Edge Quantization Comparison

| Quant.                                                                                      | VRAM (GB) | Latency (ms) |  SR (%) |
| ------------------------------------------------------------------------------------------- | --------: | -----------: | ------: |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16)               |  **8.71** |          576 |     72% |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16)               |      8.87 |      **563** |     74% |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-AttnW8) |      8.74 |          571 |     58% |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-GenW8)   |      8.79 |          570 | **80%** |

This comparison fixes guidance to 3 and denoise steps to 2. Edge request
latency is p50 on RTX 4090; success rates use 50 paired Banana rollouts. Two
steps significantly outperformed four steps for BF16 (68% vs. 42%, exact
McNemar `p=0.0146`) and also improved several quantized variants, so g3/s2 is
the validated Edge comparison protocol rather than a latency-only setting.
This is a task- and checkpoint-specific result; see the
[RoboLab Edge benchmark](examples/robolab_quant/EDGE_BENCHMARKS.md).

## Prebuilt Models

The Nano bundles below are self-contained quantized derivatives of
[`nvidia/Cosmos3-Nano-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID).
They include packed weights, residual weights, config, tokenizer, VAE,
provenance, sizes, and SHA256 hashes.

| Model                                                                                       | Precision       |  Bundle size | Notes                                      |
| ------------------------------------------------------------------------------------------- | --------------- | -----------: | ------------------------------------------ |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16)               | 504 W8 modules  |     20.44 GB | **General default**                        |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16)               | 504 W4 modules  | **13.72 GB** | Minimum memory; failed Banana quality gate |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-AttnW8) | 216 W4 / 288 W8 |     15.18 GB | Lower memory; verified on Banana           |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-GenW8)   | 252 W4 / 252 W8 |     17.08 GB | Verified on Banana; validate transfer      |

W4 and mixed bundles were calibrated with 128 frames from 128 distinct
episodes in the official DROID `train/success` split. No Banana evaluation
episode was used for calibration. The Notes column describes rollout evidence,
not task-specific calibration or training.

These are weight-only bundles; activations remain BF16. They use the Cosmos
Lite self-contained bundle schema and vLLM Marlin WNA16 kernels, not a generic GPTQ/AWQ
loader contract. RoboCasa365 uses a task-fine-tuned checkpoint, so build its
bundle with the provided pipeline instead of using the DROID weights above.

Cosmos3 Edge DROID bundles use the same download, validation, replay, and
rollout commands with `MODEL_FAMILY=cosmos3_edge` when building from the public
checkpoint. Unlike Nano, Edge bundles include the native Edge processor and
local `vision_encoder/` tower because Edge initializes its vision encoder
lazily. See the [RoboLab Edge benchmark and deployment record](examples/robolab_quant/EDGE_BENCHMARKS.md).

| Edge model                                                                                  | Precision       | Bundle size | Notes                                 |
| ------------------------------------------------------------------------------------------- | --------------- | ----------: | ------------------------------------- |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16)               | 336 W8 modules  |     7.75 GB | **General default; calibration-free** |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16)               | 336 W4 modules  |     6.38 GB | Minimum model allocation              |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-AttnW8) | 112 W4 / 224 W8 |     6.72 GB | Verified on Banana                    |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-GenW8)   | 168 W4 / 168 W8 |     7.07 GB | Highest observed Banana SR            |

## What Cosmos Lite Adds

- Streaming export from BF16, Diffusers, or DCP checkpoints without placing
  the full source model on GPU.
- Packed W4A16, W8A16, and fixed mixed-precision plans with direct loading
  below 24GB on the tested RTX 4090 paths.
- Calibration from training data for W4 and mixed strategies; full W8 does not
  require calibration.
- Strong artifact validation covering schema, precision map, tensor payloads,
  sizes, provenance, and SHA256 hashes.
- Deterministic replay, latency profiling, and closed-loop rollout entry points
  for RoboLab and RoboCasa365.
- A locked CUDA 12.8 policy runtime isolated from simulator dependencies.

## Roadmap

- [ ] Model and quantization coverage
  - [x] Support W4A16, W8A16, and mixed-precision pipelines for Nano
  - [x] Support the same pipelines for Edge
  - [ ] More quantization formats and backends
- [ ] Quantized accuracy
  - [x] Training-data calibration and layer sensitivity analysis
  - [ ] Stronger calibration and task-aware precision selection
- [ ] Runtime and deployment
  - [x] Streaming export and self-contained 24GB RTX 4090 deployment
  - [ ] Kernel and model-level inference acceleration
- [ ] Evaluation and robotics
  - [x] RoboLab and RoboCasa365 replay, profiling, and rollout
  - [ ] More simulation environments
  - [ ] Real-robot evaluation

## Setup

Requirements: Linux x86-64, Python 3.13, `uv`, a CUDA 12.8-compatible NVIDIA
driver, and an Ampere-or-newer GPU. RTX 4090 24GB is the tested target.

```bash
git clone https://github.com/xxxxyu/cosmos-lite.git
cd cosmos-lite
CUDA_VISIBLE_DEVICES=0 examples/quantized_robot_policy/setup.sh
```

## Inference

Choose a pipeline:

- [RoboLab](examples/robolab_quant/README.md): download a prebuilt Nano or
  Edge model, or build from the public NVIDIA DROID policy, validate, replay,
  and roll out.
- [RoboCasa365](examples/robocasa365_quant/README.md): stream-pack a
  task-fine-tuned BF16 DCP checkpoint, then validate, replay, and roll out.

Both use the same lifecycle:

```text
setup -> build or download -> validate -> replay -> rollout
```

Source checkpoints and calibration captures are build-time inputs only. A
deployment needs this checkout, the locked runtime, and one validated
self-contained bundle. Start with the
[runtime guide](examples/quantized_robot_policy/README.md) and use the
[release checklist](examples/quantized_robot_policy/RELEASE_CHECKLIST.md) for
a fresh machine.

## Deployment Guidance

| Scope                                 | Recommended setting | Rollback       |
| ------------------------------------- | ------------------- | -------------- |
| General RoboLab (Edge)                | W8A16, 2 steps      | W8A16, 4 steps |
| RoboLab Banana (Edge, low allocation) | W4A16, 2 steps      | W8A16, 2 steps |
| General RoboLab (Nano)                | W8A16, 2 steps      | W8A16, 4 steps |
| RoboLab Banana (Nano, lower VRAM)     | AttnW8, 4 steps     | W8A16, 4 steps |
| RoboCasa365 CloseFridge               | AttnW8, 4 steps     | W8A16, 4 steps |

All deployment-guidance rows use guidance 3.

Quantization and sampling are separate runtime controls, but they are not
quality-orthogonal. Validate a new task, embodiment, camera contract, or
checkpoint with replay and paired closed-loop rollouts before promoting a
mixed strategy or accelerated sampler.

## Scope And Safety

The release validates batch-one Cosmos robot-policy serving on an RTX 4090.
It does not establish quality transfer to untested tasks, robots,
cameras, action contracts, or non-NVIDIA hardware. It is not real-robot safety
certified.

Policy servers bind to loopback by default and have no built-in TLS or
authentication. Real-robot integration must add independent E-stop, watchdog,
workspace/joint/action limits, stale-command rejection, rate limits, and
operator supervision. See [SECURITY.md](SECURITY.md).

## Project Status And License

Cosmos Lite is an unofficial, community-maintained extension of NVIDIA Cosmos
Framework, not an NVIDIA product or a new model family. The upstream overview
is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

This repository retains the upstream [OpenMDW-1.1 license](LICENSE) and
[notices](NOTICE). Model weights may have additional terms at their download
locations. OpenMDW-1.1 is a model-material license; no claim is made that it is
OSI approved. Contributions require DCO sign-off as described in
[CONTRIBUTING.md](CONTRIBUTING.md).
