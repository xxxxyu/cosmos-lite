<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite

**An efficient inference runtime for Cosmos 3 robot policies, from on-device
serving to parallel simulation rollout.** Cosmos Lite adds deployable low-bit
artifacts, an optimized RTX 4090 runtime, and reproducible RoboLab evaluation
to [NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework).

**[Quickstart](#quickstart) | [Models](#model-profiles) | [Benchmark](docs/benchmarks/robolab.md) | [RoboLab guide](examples/robolab_quant/README.md) | [How it works](docs/runtime_architecture.md) | [Documentation](docs/README.md)**

## Demo

<!-- github native video -->
<!-- rumdl-disable MD034 -->
https://github.com/user-attachments/assets/5d3d3649-71ef-4423-bb74-e0e98265f934
<!-- rumdl-enable MD034 -->

## Results

The headline benchmark covers all 120 RoboLab tasks in `default` instruction
mode, with 10 episodes per task. Every row uses the same runtime sampler:
guidance 3, two UniPC denoise steps, shift 5, and a 32-action chunk.

| Model    | Artifact    | VRAM (GB) | Request p50 (ms) | SR (%) |
| -------- | ----------- | --------: | ---------------: | -----: |
| Edge 4B  | BF16        |      9.20 |            582.0 |  20.92 |
| Edge 4B  | **GenW8A8** |      8.79 |        **331.4** |  19.25 |
| Nano 16B | W8A16       |     21.42 |          2,403.0 |  31.50 |
| Nano 16B | **GenW8A8** | **15.51** |        **958.5** |  31.67 |

Latency is one batch-one policy request on an RTX 4090 and excludes simulator
time. Success rate uses 1,200 closed-loop rollouts per row. See the
[main benchmark](docs/benchmarks/robolab.md) for confidence intervals,
measurement details, and evaluation topology. Sampler and quantization
comparisons are kept in [ablations](docs/benchmarks/robolab_ablations.md).

## Model Profiles

Model precision is fixed by the artifact. Guidance, denoise steps, and shift
are runtime controls and can be changed without rebuilding it.

### Nano 16B

- **GenW8A8** is the recommended speed, memory, and quality tradeoff:
  [`XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8`](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8).
- **W8A16** is the calibration-free option that fits 24GB:
  [`XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16`](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16).

### Edge 4B

- **GenW8A8** is the recommended low-latency profile:
  [`XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8`](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8).
- **BF16** is the original-weight quality-first path:
  [`nvidia/Cosmos3-Edge-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID).

GenW8A8 uses FP8 W8A8 in the action-generation branch and calibrated packed
W4A16 in the remaining target layers. Calibration uses 128 DROID training
episodes; no RoboLab evaluation episode is used. W8A16 and BF16 require no
calibration.

<a id="setup"></a>

## Quickstart

Requirements: Linux x86-64, Python 3.13, `uv`, and a CUDA 12.8-compatible
NVIDIA driver. RTX 4090 24GB is the primary tested target. GenW8A8 additionally
requires SM89 and the optional SageAttention build.

```bash
git clone https://github.com/xxxxyu/cosmos-lite.git
cd cosmos-lite

# Recommended GenW8A8 runtime. Omit --with-sage for BF16 or W8A16.
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup --with-sage

hf download XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8 \
  --local-dir /data/cosmos_lite/nano_genw8a8

BUNDLE_DIR=/data/cosmos_lite/nano_genw8a8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_genw8a8_fast_4090.yaml \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

The release YAML is the source of truth for model, sampler, and backend
settings. Each launch writes a resolved config with the artifact identity,
manifest hash, Git revision, GPU capability, and fallback decisions.

Continue with the [RoboLab guide](examples/robolab_quant/README.md) for bundle
validation, request replay, closed-loop rollout, and all four release profiles.

<a id="inference"></a>

## Runtime Capabilities

- Self-contained W8A16 and GenW8A8 bundles with strong hash and tensor checks.
- Streaming export without placing the complete BF16 source model on GPU.
- RTX 4090 FP8 kernels, compiled language blocks, shared FP8 projections,
  shape-aware attention, condition K/V caching, and a reduced request data path.
- Strict YAML presets with explicit failure instead of silent backend fallback.
- Batch-one serving plus shared-policy-server, multi-environment RoboLab rollout.
- Success-only RoboLab trajectory export in the LeRobot v3 schema consumed by
  Cosmos-DROID joint-position training.
- A locked policy environment isolated from IsaacSim and training dependencies.

The [runtime architecture](docs/runtime_architecture.md) explains the request
path in plain language. The [optimization report](docs/cosmos_lite_optimization_report.md)
records what was implemented, measured, rejected, and retained.

## Project Status

- [x] Nano W8A16/GenW8A8 and Edge BF16/GenW8A8 on RTX 4090
- [x] Config-driven serving, artifact validation, replay, and RoboLab rollout
- [x] Full RoboLab-120 evaluation for the four release profiles
- [x] Bounded, training-ready RoboLab trajectory collection
- [ ] Additional NVIDIA GPU architectures and inference backends
- [ ] Additional world-action models
- [ ] Real-robot evaluation and safety integration

## Scope And Safety

RoboLab Nano and Edge are the primary supported workflows. A secondary
[RoboCasa365 integration](examples/robocasa365_quant/README.md) is retained for
user-provided fine-tuned Nano checkpoints.

The reported results cover the listed RoboLab tasks, robot, cameras, action
contract, and NVIDIA hardware. The policy server has no built-in TLS or
authentication. Real-robot use requires an independent E-stop, watchdog,
motion limits, stale-command rejection, and operator supervision. See
[SECURITY.md](SECURITY.md).

## Project And License

Cosmos Lite is an unofficial, community-maintained extension of NVIDIA Cosmos
Framework, not an NVIDIA product or a new model family. The upstream overview
is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

The repository retains the upstream [OpenMDW-1.1 license](LICENSE) and
[notices](NOTICE). Model weights may have additional terms at their download
locations. Contributions require DCO sign-off; see [CONTRIBUTING.md](CONTRIBUTING.md).
