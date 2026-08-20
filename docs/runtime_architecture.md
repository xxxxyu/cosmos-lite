<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# How The Cosmos Lite Runtime Works

This document explains the release implementation without requiring a code
review. It focuses on what is loaded, what is changed, fallback behavior, and
the runtime records written for each launch.

## The Short Version

Cosmos Lite keeps three decisions separate:

1. **Model family:** Cosmos3 Nano 16B or Edge 4B.
2. **Model artifact:** NVIDIA BF16, calibration-free W8A16, or calibrated
   GenW8A8. This choice is fixed by the checkpoint files.
3. **Runtime sampling and backends:** guidance, denoise steps, shift,
   attention backend, compilation, and kernel selection. These are selected at
   launch and do not modify the checkpoint.

A YAML file describes all three. Before loading model weights, the deployment
entry point checks that the requested family, quantization strategy, GPU, and
optional kernels agree. It then writes both the requested config and the
effective config to `resolved_deployment_config.json`.

## What Each Artifact Contains

**BF16** loads NVIDIA's public Edge checkpoint directly. It keeps the original
weights and requires no calibration. Nano BF16 does not fit the single-4090
release target and is therefore not a release preset.

**W8A16** stores language-model linear weights in packed 8-bit form and keeps
activations in BF16. It needs no calibration data. At runtime, Marlin kernels
multiply the packed weights by BF16 activations. The bundle also carries all
non-quantized weights, processor/tokenizer files, VAE files, source revisions,
file sizes, and hashes, so it does not load the source BF16 policy.

**GenW8A8** treats the two MoT branches differently:

- The action-generation branch uses FP8 weights and dynamically quantizes each
  input token to FP8 at runtime. RTX 4090 executes these GEMMs on native FP8
  Tensor Cores.
- The remaining language branch uses packed W4A16. Its weight equalization
  uses input statistics from 128 distinct DROID training episodes.
- Non-language modules and numerically sensitive residual tensors remain in
  their stored higher precision.

This is why GenW8A8 is a mixed artifact rather than a full-model FP8 model.

## What Happens When A Server Starts

1. `pipeline.sh` reads the selected YAML and applies explicit command-line
   overrides such as the local bundle path or denoise-step count.
2. `robolab_deployment_config.py` validates the schema and rejects impossible
   combinations, such as a BF16 checkpoint with a quantization strategy.
3. Quantized bundles are checked against their manifest. Strategy, model
   family, source identity, and manifest hash must match the preset.
4. The runtime probes CUDA capability and optional packages. Strict release
   presets fail if a required SM89 kernel is missing. Development presets may
   request `best_available`; every fallback is recorded.
5. `action_policy_server_robolab_deploy.py` writes the resolved record, sets
   backend controls, and starts the normal Cosmos3 RoboLab policy server.
6. The server loads either the BF16 checkpoint or the self-contained bundle,
   warms lazy modules on the first request, and writes per-request timing and
   memory events to `profile.jsonl`.

The deployment wrapper does not implement a second policy model. It validates
and configures the existing Cosmos policy server, then passes it explicit,
recorded arguments.

## Where The Speedup Comes From

Cosmos Lite combines operator-level and graph-level changes:

- Packed Marlin W8A16/W4A16 kernels reduce weight traffic.
- FP8 Tensor Core GEMMs accelerate the generation branch on SM89.
- Shared FP8 projections avoid quantizing the same activation more than once.
- `torch.compile` covers complete MoT language blocks, fusing eligible
  normalization, residual, and pointwise work around opaque custom GEMMs.
- SageAttention is optional. The strict GenW8A8 RTX 4090 presets require the
  validated build; other profiles use FlashAttention 2.
- Nano condition K/V caching reuses invariant conditioning work across denoise
  steps. It stays disabled for Edge because profiling did not show a repeatable
  gain there.
- Request preprocessing avoids repeated image copies and unnecessary Python
  synchronization.

The custom FP8 and Marlin GEMMs remain separate kernels. Compilation reduces
dispatch and fuses surrounding PyTorch operations; it does not rewrite the
inside of those kernels.

## Replay Versus Rollout

Request replay repeatedly feeds saved observations to the server. It is fast,
deterministic, and useful for finding action drift, memory leaks, latency
regressions, or backend failures. Task success comes from rollout, where each
action changes the next observation.

RoboLab rollout executes every returned action in IsaacSim. The new simulator
state becomes the next model input, so errors can compound. This makes rollout
slower but necessary for reporting closed-loop success rate.

## Failure And Fallback Rules

Release YAML files use `backend_policy: strict`:

- A wrong bundle family or strategy is rejected before weight loading.
- GenW8A8 is rejected on GPUs without native FP8 Tensor Cores.
- A missing required Triton or SageAttention backend is an error.
- A BF16 artifact cannot silently become quantized, and a quantized artifact
  cannot silently load a source checkpoint.

For experiments, `best_available` may fall back from tuned Triton FP8 to
CUTLASS or from SageAttention to FlashAttention 2. These decisions are listed
in `fallback_decisions`; an empty list means the requested backend was used.

## Files To Inspect

| Concern                        | Main file                                                         |
| ------------------------------ | ----------------------------------------------------------------- |
| Public presets                 | `examples/robolab_quant/configs/*.yaml`                           |
| Config validation and fallback | `cosmos_framework/scripts/robolab_deployment_config.py`           |
| Config-driven server entry     | `cosmos_framework/scripts/action_policy_server_robolab_deploy.py` |
| Policy request path            | `cosmos_framework/scripts/action_policy_server_robolab.py`        |
| Bundle build and validation    | `cosmos_framework/scripts/robolab_quant_bundle.py`                |
| End-to-end shell workflow      | `examples/robolab_quant/pipeline.sh`                              |
| Runtime dependency checks      | `examples/quantized_robot_policy/check_runtime.py`                |

The [primary RoboLab benchmark](benchmarks/robolab.md) reports the four release
profiles. The separate [ablation report](benchmarks/robolab_ablations.md)
compares samplers, quantization strategies, and rejected alternatives.
