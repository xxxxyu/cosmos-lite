<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite

**Deploy 16B Cosmos 3 robot policies on a single 24GB RTX 4090.** Cosmos Lite
adds streaming low-bit quantization, packed Marlin inference, self-contained
artifacts, and reproducible RoboLab and RoboCasa365 evaluation pipelines to
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework).

**[Models](#prebuilt-models) | [RoboLab benchmark](examples/robolab_quant/BENCHMARKS.md) | [RoboCasa365 benchmark](examples/robocasa365_quant/BENCHMARKS.md) | [Setup](#setup) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)**

## RoboLab Rollout Demo

<!-- rumdl-disable MD034 -->
https://github.com/user-attachments/assets/76e09f1c-4f20-47e5-ac09-6dc64f30d7a3
<!-- rumdl-enable MD034 -->

Each quantized setting was evaluated for 50 paired Banana rollouts at guidance
3 and four denoise steps. The demo shows three selected episodes per setting;
the success rates below come from all 50 rollouts, not the displayed subset.

## At A Glance

| Env & Task              | Quant.       | Denoise | VRAM (GB) | Latency (ms) | SR (%) |
| ----------------------- | ------------ | ------: | --------: | -----------: | -----: |
| RoboLab Banana          | W8A16        |       2 |     21.42 |        2,403 |    90% |
| RoboLab Banana          | W4A16-GenW8  |       2 |     18.03 |        2,433 |   100% |
| RoboCasa365 CloseFridge | W4A16-AttnW8 |       4 |     14.28 |        1,231 |    96% |

VRAM is peak CUDA reserved memory and latency is p50 end-to-end policy request
latency on RTX 4090. RoboLab success rates are 50-rollout RTX 4090 results.
The RoboCasa success rate was reproduced in two 50-episode H100 runs with the
same quantization and sampler settings; its VRAM and latency are RTX 4090
measurements. All rows use guidance 3.

## RoboLab Quantization Comparison

| Quant.                                                                                      | VRAM (GB) | Latency (ms) |  SR (%) |
| ------------------------------------------------------------------------------------------- | --------: | -----------: | ------: |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16)               |     21.42 |        4,110 |     86% |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16)               | **14.67** |        4,248 |     52% |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-AttnW8) |     16.21 |        4,153 |     84% |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W4A16-GenW8)   |     18.03 |    **4,104** | **90%** |

This comparison fixes guidance to 3 and denoise steps to 4. Request latency is
p50 on RTX 4090; success rates use 50 paired Banana rollouts.

## Prebuilt Models

The public bundles below are self-contained quantized derivatives of
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
Lite schema-v2 format and vLLM Marlin WNA16 kernels, not a generic GPTQ/AWQ
loader contract. RoboCasa365 uses a task-fine-tuned checkpoint, so build its
bundle with the provided pipeline instead of using the DROID weights above.

## What Cosmos Lite Adds

- Streaming export from BF16 or DCP checkpoints without placing the full
  source model on GPU.
- Packed W4A16, W8A16, and fixed mixed-precision plans with direct loading
  below 24GB on the tested RTX 4090 paths.
- Calibration from training data for W4 and mixed strategies; full W8 does not
  require calibration.
- Strong artifact validation covering schema, precision map, tensor payloads,
  sizes, provenance, and SHA256 hashes.
- Deterministic replay, latency profiling, and closed-loop rollout entry points
  for RoboLab and RoboCasa365.
- A locked CUDA 12.8 policy runtime isolated from simulator dependencies.

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

- [RoboLab](examples/robolab_quant/README.md): download a prebuilt model or
  build from the public NVIDIA DROID policy, validate, replay, and roll out.
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

| Scope                             | Recommended setting | Rollback       |
| --------------------------------- | ------------------- | -------------- |
| General RoboLab                   | W8A16, 2 steps      | W8A16, 4 steps |
| RoboLab Banana under lower memory | AttnW8, 4 steps     | W8A16, 4 steps |
| RoboCasa365 CloseFridge           | AttnW8, 4 steps     | W8A16, 4 steps |

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
