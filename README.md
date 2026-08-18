<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite

**Deploy Cosmos 3 robot policies on one 24GB RTX 4090.** Cosmos Lite adds
self-contained quantized checkpoints, RTX 4090 inference optimizations, and
reproducible RoboLab evaluation to
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework). It
supports both Cosmos3 Edge 4B and Cosmos3 Nano 16B DROID policies.

**[Models](#models) | [RoboLab Quickstart](#quickstart) | [8-GPU rollout](examples/robolab_quant/ROLLOUT_THROUGHPUT.md) | [Nano benchmark](examples/robolab_quant/NANO_BENCHMARKS.md) | [Edge benchmark](examples/robolab_quant/EDGE_BENCHMARKS.md) | [Optimization report](docs/cosmos_lite_optimization_report.md) | [RoboCasa365](examples/robocasa365_quant/README.md) | [Roadmap](#roadmap)**

## News

- **2026-07-22:** Added end-to-end quantized inference and RoboLab evaluation
  for Cosmos3 Edge DROID policies on RTX 4090.
- **2026-07-16:** Released the first single-RTX-4090 quantization and
  deployment pipeline for Cosmos3 Nano, with RoboLab and RoboCasa365 support.

## At A Glance

The release has two checkpoint types per model family:

- **W8A16:** calibration-free, portable quality-first fallback.
- **GenW8A8:** recommended RTX 4090 profile. The generation branch uses FP8
  W8A8; remaining target Linear modules use calibrated W4A16.

| Env & Task     | Model    | Quant.      | Denoise | VRAM (GB) | Latency (ms) | SR (%) |
| -------------- | -------- | ----------- | ------: | --------: | -----------: | -----: |
| RoboLab Banana | Edge 4B  | W8A16       |       2 |      8.71 |        576.0 |     72 |
| RoboLab Banana | Edge 4B  | **GenW8A8** |       2 |      8.79 |    **331.4** | **80** |
| RoboLab Banana | Nano 16B | W8A16       |       2 |     21.42 |      2,403.0 |     90 |
| RoboLab Banana | Nano 16B | **GenW8A8** |       2 | **15.51** |    **958.5** | **98** |

VRAM is peak CUDA reserved memory and latency is p50 end-to-end policy request
latency on RTX 4090. Success rates use 50 paired RoboLab
`BananaInBowlTask` rollouts at guidance 3, two UniPC denoise steps, and 32
generated and executed actions. These point estimates show that the fast
profiles passed the existing quality gate; they do not prove quantization
improves policy quality.

## Models

The bundles are self-contained derivatives of NVIDIA's public DROID policies.
They include packed and residual weights, model config, tokenizer or processor,
VAE, provenance, file sizes, and SHA256 hashes.

| Family   | Checkpoint                                                                                                              | Deployment role                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Nano 16B | [`XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16`](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16) | Calibration-free fallback        |
| Nano 16B | `XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8`                                                                              | **Recommended RTX 4090 profile** |
| Edge 4B  | [`XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16`](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16) | Calibration-free fallback        |
| Edge 4B  | `XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8`                                                                              | **Recommended RTX 4090 profile** |

W8A16 does not require calibration. In GenW8A8, the W4A16 remainder uses input
statistics from 128 distinct episodes in the official DROID `train/success`
split, while the FP8 generation branch uses dynamic per-token activation
scales. No RoboLab evaluation episode is used. The source BF16 checkpoint and
calibration data are build-time inputs only.

Full W4, AttentionW8, generation-branch W8A16, full W8A8, and equalized
variants remain in the codebase and experiment reports for reproduction. They
are not recommended release checkpoints because they did not provide a clear,
non-redundant deployment benefit over W8A16 or GenW8A8.

## What Cosmos Lite Adds

- Streaming export from public BF16 DROID policies without placing the full
  source model on GPU.
- Direct loading of self-contained W8A16 and GenW8A8 bundles below 24GB on the
  validated RTX 4090 paths.
- RTX 4090 FP8 kernels, `torch.compile`, shared FP8 projections,
  shape-aware attention, condition K/V caching, and request data-path
  optimizations.
- YAML deployment presets with strict backend checks and a complete resolved
  config record for every run.
- Strong artifact validation covering schema, precision map, packed tensors,
  sizes, provenance, and SHA256 hashes.
- Fixed-input request replay plus closed-loop RoboLab rollout, latency, and
  memory measurement.
- A locked CUDA 12.8 policy environment isolated from IsaacSim and training
  dependencies.

The [English](docs/cosmos_lite_optimization_report.md) and
[Chinese](docs/cosmos_lite_optimization_report_zh.md) optimization reports
explain what was implemented, measured, rejected, and retained.

## Quickstart

Requirements: Linux x86-64, Python 3.13, `uv`, a CUDA 12.8-compatible NVIDIA
driver, and an Ampere-or-newer NVIDIA GPU. RTX 4090 24GB is the primary tested
target; the fast GenW8A8 presets require SM89. Building their optional
SageAttention backend also requires a CUDA 12.4-or-newer toolkit with `nvcc`
and a C++ build toolchain. The locked policy environment occupies about 13GB
on the tested host, excluding checkpoints and the `uv` download cache.

```bash
git clone https://github.com/xxxxyu/cosmos-lite.git
cd cosmos-lite

# W8A16 minimal runtime
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup

# Or install the optional SageAttention SM89 backend for GenW8A8
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup --with-sage
```

Download and validate a bundle:

```bash
hf download "XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16" \
  --local-dir /data/cosmos_lite/nano_w8

BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
STRATEGY=full_w8 \
examples/robolab_quant/pipeline.sh validate
```

Serve it from the release YAML:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
RUN_DIR=/data/cosmos_lite/runs/nano_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

The launch writes `resolved_deployment_config.json` with requested and
effective settings, bundle manifest hash, git revision, GPU capability, and
all fallback decisions. Release presets are strict: an incompatible GPU,
bundle, or optional backend fails before model loading instead of silently
changing the benchmark configuration.

Continue with the [RoboLab guide](examples/robolab_quant/README.md) for the
deployment doctor, request replay, closed-loop rollout, and building W8A16 or
GenW8A8 from NVIDIA's public policies.

## Evaluation Terms

**Request replay** sends saved policy observations without environment
feedback. It measures action differences, request latency, peak memory, and
runtime stability. It cannot measure task success.

**Rollout** executes actions in the simulator. Each subsequent observation
depends on earlier actions, so rollout measures closed-loop task success. The
two gates answer different questions and both are required before promoting a
new deployment profile.

## RoboCasa365

RoboCasa365 is secondary support. It starts from a user-provided,
task-fine-tuned BF16 Cosmos3 Nano checkpoint rather than the public DROID
policy. The maintained recipes are:

| Recipe     | Quantization | Sampling            | Purpose                       |
| ---------- | ------------ | ------------------- | ----------------------------- |
| `balanced` | Attention W8 | guidance 3, 4 steps | Lower-memory default          |
| `quality`  | Full W8      | guidance 3, 4 steps | Conservative quality fallback |

See the [RoboCasa365 pipeline](examples/robocasa365_quant/README.md) and
[benchmark](examples/robocasa365_quant/BENCHMARKS.md). RoboLab remains the
primary release and full benchmark target.

## Roadmap

- [ ] Release-quality model and platform coverage
  - [x] Nano and Edge W8A16/GenW8A8 on RTX 4090
  - [ ] Additional NVIDIA GPU architectures
  - [ ] Additional WAM architectures
- [ ] Quantized policy quality
  - [x] Training-data calibration and layer sensitivity analysis
  - [ ] Stronger calibration and precision allocation
  - [ ] Broader paired RoboLab evaluation
- [ ] Runtime and deployment
  - [x] Streaming export and self-contained bundles
  - [x] Config-driven serving and reproducibility records
  - [ ] Additional kernels and inference backends
- [ ] Robotics evaluation
  - [x] RoboLab request replay and closed-loop rollout
  - [x] Secondary RoboCasa365 support
  - [ ] Real-robot evaluation and safety integration

## Scope And Safety

This project validates batch-one Cosmos robot-policy serving on RTX 4090. It
does not establish quality transfer to untested tasks, robots, cameras, action
contracts, or non-NVIDIA hardware, and it is not real-robot safety certified.

Policy servers bind to loopback by default and provide no built-in TLS or
authentication. Hardware integrations need independent E-stop, watchdog,
workspace/joint/action limits, stale-command rejection, rate limits, and
operator supervision. See [SECURITY.md](SECURITY.md).

## Project And License

Cosmos Lite is an unofficial, community-maintained extension of NVIDIA Cosmos
Framework, not an NVIDIA product or a new model family. The upstream overview
is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

This repository retains the upstream [OpenMDW-1.1 license](LICENSE) and
[notices](NOTICE). Model weights may have additional terms at their download
locations. Contributions require DCO sign-off as described in
[CONTRIBUTING.md](CONTRIBUTING.md).
