<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboLab Deployment Guide

This guide serves Cosmos3 Nano and Edge DROID policies with Cosmos Lite, then
validates them with request replay or closed-loop RoboLab rollout. Model
precision and runtime sampling are independent choices.

## Workflow

1. **Install** the isolated policy runtime.
2. **Download** one BF16 checkpoint or self-contained quantized bundle.
3. **Select** a release YAML and, if needed, override sampler fields.
4. **Validate and serve** the exact artifact and backend combination.
5. **Replay** recorded observations to check latency, memory, and action parity.
6. **Roll out** in RoboLab to measure closed-loop success.

Replay has no simulator feedback and cannot measure task success. During a
rollout, every action changes the next observation; small action differences
can therefore accumulate. Use replay as the fast engineering gate and rollout
as the policy-quality gate.

The four release profiles and their 1,200-rollout results are in the
[RoboLab benchmark](../../docs/benchmarks/robolab.md).

## 1. Install

Use the minimal runtime for Edge BF16 or Nano W8A16:

```bash
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup
```

GenW8A8 on RTX 4090 also uses the pinned SageAttention SM89 extension. Its
source build requires a CUDA 12.4-or-newer toolkit, `nvcc`, and a C++ toolchain:

```bash
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup --with-sage
```

RoboLab and IsaacSim remain in their own environment. They are not installed
into the policy runtime.

## 2. Download A Model

- Nano GenW8A8: `XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8`
- Nano W8A16: `XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16`
- Edge GenW8A8: `XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8`
- Edge BF16: `nvidia/Cosmos3-Edge-Policy-DROID`

For example:

```bash
hf download XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16 \
  --local-dir /data/cosmos_lite/nano_w8
```

A quantized bundle includes packed and residual weights, model config,
tokenizer or processor, VAE, provenance, file sizes, and SHA256 hashes. Source
BF16 weights and calibration data are build-time inputs, not deployment
dependencies. Edge BF16 is loaded directly from NVIDIA's checkpoint.

## 3. Choose A Config

| Preset                                                               | Profile               |
| -------------------------------------------------------------------- | --------------------- |
| [`nano_genw8a8_fast_4090.yaml`](configs/nano_genw8a8_fast_4090.yaml) | Nano recommended      |
| [`nano_w8.yaml`](configs/nano_w8.yaml)                               | Nano calibration-free |
| [`edge_genw8a8_fast_4090.yaml`](configs/edge_genw8a8_fast_4090.yaml) | Edge low latency      |
| [`edge_bf16.yaml`](configs/edge_bf16.yaml)                           | Edge original weights |

Release configs use `backend_policy: strict`. A wrong artifact, unsupported
GPU, or missing required backend fails before model loading. Development
configs may choose `best_available`; all fallback decisions are then recorded.

The release sampler defaults to guidance 3, two UniPC steps, shift 5, and
deterministic seed 0. Override `GUIDANCE`, `NUM_STEPS`, or `SHIFT` at launch
without changing the artifact.

## 4. Validate And Serve

Strong bundle validation is CPU-only:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
STRATEGY=full_w8 \
examples/robolab_quant/pipeline.sh validate
```

Start the policy server:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
RUN_DIR=/data/cosmos_lite/runs/nano_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

For Edge BF16, replace `BUNDLE_DIR` with:

```bash
CHECKPOINT_PATH=nvidia/Cosmos3-Edge-Policy-DROID
```

The server defaults to `127.0.0.1:8000`. It has no built-in TLS or
authentication; keep it on loopback or behind a trusted authenticated
transport.

Each launch writes `resolved_deployment_config.json` and `profile.jsonl` below
`RUN_DIR/server/`. Keep both with benchmark or deployment records.

## 5. Request Replay

Replay fixed RoboLab-format observations without starting IsaacSim:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
CAPTURE_DIR=/data/cosmos_lite/request_replay \
RUN_DIR=/data/cosmos_lite/runs/nano_w8_replay \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh replay
```

Replay reports request latency, CUDA memory, runtime stability, and action
difference against a reference. It does not report success rate.

## 6. Closed-Loop Rollout

Use separate physical GPUs for the policy and IsaacSim. Vulkan device
selection is not fully controlled by the policy's CUDA device setting.

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
ROBOLAB_DIR=/path/to/RoboLab \
ROBOLAB_PYTHON=/path/to/robolab/python \
RUN_DIR=/data/cosmos_lite/runs/nano_w8_banana \
POLICY_GPU=0 SIM_GPU=1 \
TASK=BananaInBowlTask NUM_ENVS=1 NUM_RUNS=1 \
examples/robolab_quant/pipeline.sh rollout
```

The client composes a 640x540 RGB input from wrist and two exterior views. The
server maps it to a 736x544 model bucket and returns a 32x8 action chunk. The
standard RoboLab Cosmos3 client executes all 32 actions before the next policy
request.

For large evaluations, share policy servers across simulator lanes. The
validated 8-GPU layouts are listed in the
[main benchmark](../../docs/benchmarks/robolab.md#evaluation-topology).

To turn the same rollout into success-only Cosmos-DROID training data, apply
the pinned RoboLab integration and set `TRAJECTORY_CONFIG` to its trajectory
YAML. The result is a validated LeRobot v3 dataset with three `640x360` RGB
streams, aligned state/action rows, and reproduction sidecars. See
[RoboLab data generation](DATA_GENERATION.md). Recording changes simulator
CPU, memory, and disk requirements; the evaluation VRAM figures above do not
include it.

## Outputs And Reproduction

Retain these files with every result:

- the deployment YAML and generated `resolved_deployment_config.json`;
- bundle manifest hash or BF16 checkpoint revision;
- `profile.jsonl`, rollout logs, task list, and episode count;
- GPU model, sampler values, action chunk, and simulator horizon.

To build W8A16 or GenW8A8 instead of downloading it, see
[Building model artifacts](../../docs/model_build.md). Experimental strategies
and backend studies are indexed under [Documentation](../../docs/README.md).
