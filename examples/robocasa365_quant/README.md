<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos3 RoboCasa365 Quantized Pipeline

This is the release entry point for converting a fine-tuned Cosmos3 Nano
RoboCasa365 BF16 checkpoint into a self-contained packed W4A16/W8A16 policy,
then validating it with replay or RLDX rollout on a 24GB RTX 4090.

Read [BENCHMARKS.md](BENCHMARKS.md) before selecting a strategy or sampler.

## Inputs And Output

Export requires:

- a fine-tuned BF16 DCP model directory containing `.metadata`;
- its resolved runtime YAML config;
- 128 training-split calibration request captures with RGB frames for W4 or
  mixed strategies (`full_w8` does not require calibration);
- the Qwen tokenizer directory and `Wan2.2_VAE.pth` referenced by the config.

The output schema-v2 bundle contains packed tensors, residual safetensors,
portable config, tokenizer, VAE, strategy metadata, sizes, and SHA256 hashes.
Serving the bundle never reads the BF16 DCP or calibration captures.

## 1. Setup

```bash
CUDA_VISIBLE_DEVICES=0 examples/robocasa365_quant/pipeline.sh setup
```

This installs the shared locked runtime. It does not install RLDX/RoboCasa.
Prepare the simulator with the upstream RLDX RoboCasa365 setup script,
including its assets and EGL check. The simulator checkout and editable
installs must resolve locally; do not deploy an environment containing
cross-machine or cross-user asset symlinks.

## 2. Build

Use local scratch storage for export and the final bundle. The command never
loads the full BF16 checkpoint onto the GPU: it streams DCP weights through one
Linear layer at a time. W4/mixed builds first stream-pack a temporary W8 model,
use it to collect calibration statistics, then stream-pack the selected plan.
Residual conversion is CPU-streamed and the result receives strong validation.

```bash
BF16_CHECKPOINT=/path/to/iter_000008000/model \
CONFIG_FILE=/path/to/resolved_config.yaml \
CALIBRATION_CAPTURE_DIR=/path/to/train_calibration_requests \
TOKENIZER_DIR=/path/to/Qwen3-VL-8B-Instruct \
VAE_PATH=/path/to/Wan2.2_VAE.pth \
BUNDLE_DIR=/data/cosmos3_quant/attention_w8 \
STRATEGY=attention_w8 \
POLICY_GPU=0 \
examples/robocasa365_quant/pipeline.sh build
```

The build refuses to overwrite `BUNDLE_DIR`. Its temporary packed-only
artifact is removed after successful conversion; `${BUNDLE_DIR}.build.log` is
kept for provenance. If a shared filesystem is needed, build on local `/data*`
and copy the completed bundle afterward.

## 3. Validate Or Transfer

Run after every build or transfer:

```bash
BUNDLE_DIR=/path/to/attention_w8 \
STRATEGY=attention_w8 \
examples/robocasa365_quant/pipeline.sh validate
```

Validation checks the schema, precision map, all packed payloads, file sizes,
and SHA256 hashes. Deployment must stop on any mismatch.

## 4. Replay

Replay verifies direct load, request preprocessing, action output, latency,
and peak memory without a simulator:

```bash
BUNDLE_DIR=/path/to/attention_w8 \
CAPTURE_DIR=/path/to/closefridge_action_parity_v1 \
RUN_DIR=/data/cosmos3_runs/robocasa_replay \
POLICY_GPU=0 \
REPLAY_LIMIT=32 \
examples/robocasa365_quant/pipeline.sh replay
```

Inspect `replay/metrics.json`, `profile_summary.json`, and `server.log` under
`RUN_DIR`.

## 5. Rollout

RLDX/RoboCasa uses its own environment. Point the pipeline to that interpreter
and evaluator; do not install simulator packages into the policy runtime.

```bash
BUNDLE_DIR=/path/to/attention_w8 \
ROBOCASA365_PYTHON=/path/to/robocasa365_uv/.venv/bin/python \
ROBOCASA365_ROLLOUT_SCRIPT=/path/to/RLDX/rldx/eval/rollout_policy.py \
RUN_DIR=/data/cosmos3_runs/robocasa_rollout \
POLICY_GPU=0 \
N_EPISODES=50 N_ENVS=5 MAX_EPISODE_STEPS=1200 \
examples/robocasa365_quant/pipeline.sh rollout
```

The release protocol uses `N_ACTION_STEPS=8`, `MAX_EPISODE_STEPS=1200`, target
split, disabled video, guidance 3.0, and four UniPC steps. Do not silently
restore the task's shorter 600-step horizon. The pipeline infers the RLDX repo
root from `ROBOCASA365_ROLLOUT_SCRIPT`; set `ROBOCASA365_ROOT` only for a
nonstandard checkout layout. It also sources the standard simulator `env.sh`
for EGL/MuJoCo and uses the checkout's `external_dependencies/robocasa365`;
set `ROBOCASA365_ENV_FILE` or `ROBOCASA365_SOURCE` for nonstandard paths.

## Strategy Selection

| Strategy        | W4/W8 modules | 4090 peak allocated | Validated role                     |
| --------------- | ------------: | ------------------: | ---------------------------------- |
| `attention_w8`  |       216/288 |             13.94GB | Recommended memory/quality default |
| `full_w8`       |         0/504 |             19.20GB | Quality-first                      |
| `full_w4`       |         504/0 |             12.48GB | Minimum memory                     |
| `gen_branch_w8` |       252/252 |             15.84GB | Alternative mixed strategy         |

All are weight-only; activations remain BF16. The default sampler remains
guidance 3.0 / four steps. Faster sampler settings are experimental because
the 4090 rollout confidence intervals overlap.

## Calibration Contract

Calibration captures must come from the matching RoboCasa training task, not
evaluation rollouts. The validated configuration uses 128 frames and
`CALIBRATION_ALPHA=0.5`. RGB files are required; parquet-only metadata is not
sufficient. Keep dataset/task/revision/sample provenance with the captures.

## Serving Only

```bash
BUNDLE_DIR=/path/to/attention_w8 POLICY_GPU=0 \
examples/robocasa365_quant/pipeline.sh serve
```

The ZMQ endpoint defaults to `127.0.0.1:5577`. Set `HOST`, `PORT`,
`SERVED_ACTION_STEPS`, `GUIDANCE`, or `NUM_STEPS` explicitly when changing the
validated protocol.

The server has no built-in TLS or authentication. Keep the loopback default,
or expose it only through a trusted network, SSH tunnel, or authenticated
encrypted proxy. Binding to a non-loopback address also disables the remote
`kill` endpoint.

## Developer Profiling

`profile_direct_replay_4090.sh` and `quant_backend_microbench.py` are retained
for operator investigation. They are not part of deployment setup. Production
defaults use eager execution because the tested graph-break compile path did
not improve steady latency.

Legacy schema-v1 artifacts and old overlay/site-package environment templates
are unsupported for new deployments. Roll back with an immutable validated
schema-v2 bundle.
