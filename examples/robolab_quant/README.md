<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos3 RoboLab Quantized Pipeline

This is the release entry point for serving Cosmos3 Nano and Edge DROID
policies on one 24GB RTX 4090 and evaluating them in RoboLab. The public path
has two checkpoint types per model family:

For multi-GPU evaluation, start with the
[8-GPU rollout throughput guide](ROLLOUT_THROUGHPUT.md). It covers vectorized
RoboLab environments, policy/simulator GPU layouts, request staggering, and
scene-aware task sharding.

| Checkpoint | Purpose                         | Calibration                 | RTX 4090 backend                                   |
| ---------- | ------------------------------- | --------------------------- | -------------------------------------------------- |
| W8A16      | Portable quality-first fallback | None                        | Marlin + FlashAttention 2                          |
| GenW8A8    | Recommended low-latency profile | 128 DROID training requests | FP8 generation branch + calibrated W4A16 remainder |

GenW8A8 uses FP8 W8A8 only in the generation branch. It is not full-model
FP8. Other W4, W8A16 mixed, and full-W8A8 strategies remain available for
research reproduction, but they are not release checkpoints or recommended
deployment profiles.

## Validated Results

All rows use RoboLab `BananaInBowlTask`, guidance 3, two UniPC denoise steps,
32 generated and executed actions, and 50 paired rollouts. Latency is p50
end-to-end policy request latency; VRAM is peak CUDA reserved memory on RTX
4090.

| Model    | Checkpoint  | VRAM (GB) | Latency (ms) | Success Rate (%) |
| -------- | ----------- | --------: | -----------: | ---------------: |
| Edge 4B  | W8A16       |      8.71 |        576.0 |               72 |
| Edge 4B  | **GenW8A8** |      8.79 |    **331.4** |           **80** |
| Nano 16B | W8A16       |     21.42 |      2,403.0 |               90 |
| Nano 16B | **GenW8A8** | **15.51** |    **958.5** |           **98** |

The success-rate point estimates do not establish that GenW8A8 improves
policy quality. They show that the promoted fast profiles passed the existing
paired Banana gate. See [Nano benchmarks](NANO_BENCHMARKS.md),
[Edge benchmarks](EDGE_BENCHMARKS.md), and the
[SM89 optimization report](SM89_OPTIMIZATION.md) for protocols and ablations.

## 1. Install The Policy Runtime

W8A16 uses the minimal locked runtime:

```bash
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup
```

The RTX 4090 GenW8A8 presets require the optional pinned SageAttention SM89
extension. Its source build requires a CUDA 12.4-or-newer toolkit with `nvcc`
and a C++ build toolchain:

```bash
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup --with-sage
```

RoboLab and IsaacSim stay in their own environment or container. The policy
runtime does not install simulator or training dependencies.

## 2. Download A Self-Contained Bundle

| Family | W8A16                                                                                                                   | GenW8A8                                    |
| ------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Nano   | [`XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16`](https://huggingface.co/XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16) | `XXXXyu/Cosmos3-Nano-Policy-DROID-GenW8A8` |
| Edge   | [`XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16`](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16) | `XXXXyu/Cosmos3-Edge-Policy-DROID-GenW8A8` |

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download "XXXXyu/Cosmos3-Nano-Policy-DROID-Marlin-W8A16" \
  --local-dir /data/cosmos_lite/nano_w8
```

Each bundle includes packed and residual weights, runtime config, tokenizer or
processor, VAE, provenance, file sizes, and SHA256 hashes. The source BF16
checkpoint and calibration data are not deployment dependencies.

## 3. Select A Deployment Config

Formal serving uses YAML. Environment-variable tuning remains available only
through the legacy development entry point.

| Preset                                                               | Use                               |
| -------------------------------------------------------------------- | --------------------------------- |
| [`nano_w8.yaml`](configs/nano_w8.yaml)                               | Nano W8A16 portable fallback      |
| [`nano_genw8a8_fast_4090.yaml`](configs/nano_genw8a8_fast_4090.yaml) | Nano recommended RTX 4090 profile |
| [`edge_w8.yaml`](configs/edge_w8.yaml)                               | Edge W8A16 portable fallback      |
| [`edge_genw8a8_fast_4090.yaml`](configs/edge_genw8a8_fast_4090.yaml) | Edge recommended RTX 4090 profile |

Release presets use strict backend checks. A missing Sage build, incompatible
GPU, wrong bundle family/strategy, or unavailable tuned Triton backend fails
before model loading. There is no silent benchmark fallback. Custom development
configs may set `runtime.backend_policy: best_available`; every fallback is
then written to the resolved deployment record.

Run the deployment doctor before loading the model:

```bash
examples/quantized_robot_policy/.venv/bin/python \
  examples/quantized_robot_policy/check_runtime.py \
  --require-cuda \
  --repo-root "$PWD" \
  --deployment-config examples/robolab_quant/configs/nano_w8.yaml \
  --bundle-dir /data/cosmos_lite/nano_w8
```

## 4. Validate And Serve

Strong bundle validation is CPU-only:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
STRATEGY=full_w8 \
examples/robolab_quant/pipeline.sh validate
```

Start the server from a release preset:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
RUN_DIR=/data/cosmos_lite/runs/nano_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

Every config-driven launch writes
`RUN_DIR/server/resolved_deployment_config.json`. It records requested and
effective settings, bundle manifest hash, model family, strategy, git revision,
GPU capability, dependency probe, and fallback decisions. Request timing and
memory events are written to `profile.jsonl` in the same directory.

The WebSocket endpoint defaults to `127.0.0.1:8000` and has no built-in TLS or
authentication. Keep it on loopback or place it behind a trusted authenticated
transport.

## 5. Request Replay

Request replay sends fixed recorded policy observations without simulator
feedback. It measures action error, latency, memory, and runtime stability; it
does not measure task success.

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/nano_w8.yaml \
CAPTURE_DIR=/data/cosmos_lite/request_replay \
RUN_DIR=/data/cosmos_lite/runs/nano_w8_replay \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh replay
```

## 6. Closed-Loop Rollout

A rollout executes actions in RoboLab, so each new observation depends on the
previous action. It measures closed-loop task success. Use separate physical
GPUs for the policy server and IsaacSim because Vulkan device selection is not
fully controlled by the policy `--device` flag.

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

The client builds one 640x540 RGB input from three RoboLab views: a 640x360
wrist view above two 320x180 exterior views. The server maps it to the policy's
736x544 bucket and returns a 32x8 action chunk. The standard RoboLab Cosmos3
client executes all 32 actions before requesting the next chunk.

## 7. Build From NVIDIA's Public DROID Policy

W8A16 is calibration-free and stream-packed without placing the full BF16
policy on GPU:

```bash
HF_TOKEN=... \
MODEL_FAMILY=cosmos3_nano \
STRATEGY=full_w8 \
ASSET_DIR=/data/cosmos_lite/sources/nano \
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

For Edge, set `MODEL_FAMILY=cosmos3_edge`. Source revisions are pinned and
recorded in `sources.json` and the bundle manifest. Reusing `ASSET_DIR` resumes
downloads; a completed `BUNDLE_DIR` is never overwritten.

GenW8A8 requires input-channel statistics from 128 distinct episodes in the
official `nvidia/Cosmos3-DROID` `train/success` split. No RoboLab Banana
evaluation episode is used. Build it with:

```bash
HF_TOKEN=... \
MODEL_FAMILY=cosmos3_nano \
STRATEGY=gen_branch_w8a8 \
CALIBRATION_STATS=/path/to/droid_train128_input_amax.pt \
ASSET_DIR=/data/cosmos_lite/sources/nano \
BUNDLE_DIR=/data/cosmos_lite/nano_genw8a8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

The calibration export and replay protocol is documented in the Training
Calibration section of the family benchmark reports.

## Development And Reproduction

The implementation still supports full W4, AttentionW8, generation-branch
W8A16, full W8A8, equalization experiments, and direct CLI/environment tuning.
These paths are intentionally absent from the release quickstart because they
did not add a distinct deployment benefit over W8A16 or GenW8A8.

Use the following records to reproduce or extend them:

- [Nano benchmark matrix](NANO_BENCHMARKS.md)
- [Edge benchmark matrix](EDGE_BENCHMARKS.md)
- [FP8 W8A8 experiment](FP8_W8A8_EXPERIMENT.md)
- [Compile and CUDA Graph experiment](GRAPH_OPTIMIZATION_EXPERIMENT.md)
- [RTX 4090 SM89 optimization](SM89_OPTIMIZATION.md)
- [Full optimization report](../../docs/cosmos_lite_optimization_report.md)
