<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboLab Rollout Throughput Guide

This is the entry point for running many Cosmos3 DROID rollouts on one
8 x RTX 4090 host. It covers resource layout and evaluation throughput; use
the main [RoboLab pipeline guide](README.md) for installation, bundles, and
single-policy serving.

## Recommended Defaults

Use guidance 3, two UniPC denoise steps, shift 5, a 32-action open-loop chunk,
and `--num-envs 10 --num-runs 1`. Keep policy and simulator processes alive
across tasks, group tasks by USD scene, and stagger the first request from
simulators sharing one policy server.

| Model/profile | Policy + simulator GPUs | Simulator setting | First-request stagger per server | RoboLab-120 time | Quality evidence |
| --- | ---: | --- | --- | ---: | ---: |
| Edge BF16, g3/s2 | 2P + 6S | 10 env x 1 run | `0/6/12s` | 5h35m17s measured | 251/1,200 (20.92%) |
| Edge W8A16 | 2P + 6S | 10 env x 1 run | `0/6/12s` | 5.9-7.5h | Banana 36/50 (72%) |
| **Edge GenW8A8** | **2P + 6S** | **10 env x 1 run** | **`0/4/8s`** | **5.3-7.1h** | **Banana 40/50 (80%)** |
| Nano W8A16 | 3P + 5S | 10 env x 1 run | `0/18s` for two-client servers | 10.2-13.9h | Banana 45/50 (90%) |
| **Nano GenW8A8** | **2P + 6S** | **10 env x 1 run** | **`0/10/20s`** | **7.6-10.7h** | **Banana 49/50 (98%)** |

`P` means a policy-server GPU and `S` means an Isaac Sim GPU. Planning ranges
are capacity estimates for 120 tasks x 10 episodes, not measured full-suite
wall times. The quantized quality gates are paired `BananaInBowlTask` rollouts,
not an estimate of success on every RoboLab task.

The Edge BF16 row is a completed full-suite reference rather than a planning
estimate. It directly loads NVIDIA's BF16 checkpoint and is included for
reproduction and comparison; it is not the recommended throughput profile.

### Highest successful-rollout throughput

For both model families, choose the **GenW8A8 fast RTX 4090 preset**. It is
Pareto-better in the current independent gates: it has lower request latency,
higher measured multi-simulator throughput, and passed the paired quality gate
with a higher point estimate than W8A16. Do not interpret the Edge-versus-Nano
rows as a general policy-quality ranking; select the model family first, then
select GenW8A8 within that family.

- Edge: [`configs/edge_genw8a8_fast_4090.yaml`](configs/edge_genw8a8_fast_4090.yaml)
- Nano: [`configs/nano_genw8a8_fast_4090.yaml`](configs/nano_genw8a8_fast_4090.yaml)
- Portable fallback: W8A16 needs no calibration and no optional SageAttention
  build, but completes fewer expected successful rollouts per unit time in the
  current gates.
- Nano BF16 reproduction remains a TODO and is not a release or topology gate.

This recommendation does not multiply a single-task success rate by a
different task's throughput and present the result as a formal metric. Report
successful rollouts per hour only from a matched task distribution and one
recorded end-to-end run.

## Edge BF16 Reference

Edge BF16 fits on one RTX 4090 policy GPU and uses the same `2P + 6S` layout as
the Edge quantized profiles. Both completed RoboLab-120 runs used 120 tasks,
10 episodes per task, `10 env x 1 run` per simulator, a `0/6/12s` first-request
stagger for each three-client server, shift 5, deterministic seed 0, JSON
prompts, and all 32 generated actions executed at 15 Hz.

| BF16 sampler | Evaluation wall time | Success rate | Successful rollouts/hour |
| --- | ---: | ---: | ---: |
| Guidance 3, 4 denoise steps | 9h20m57s | 260/1,200 (21.67%) | 27.8 |
| Guidance 3, 2 denoise steps | 5h35m17s | 251/1,200 (20.92%) | 44.9 |

The two-step BF16 run is the practical BF16 throughput reference. The
four-step row is retained as the original comparison control. They ran on two
different 8 x RTX 4090 hosts. The nearly 2x request-latency reduction is
consistent with halving the denoise steps, while environment-time differences
may also include host and trajectory effects. For comparison, Edge GenW8A8
g3/s2 used the same `2P + 6S` protocol and completed in 4h51m15s with 231/1,200
successes, or 47.6 successful rollouts/hour. It remains the measured Edge
throughput winner, although the margin over BF16 g3/s2 is modest.

BF16 does not use a quantized bundle or deployment YAML. Start each policy
server directly from the local NVIDIA checkpoint:

```bash
CHECKPOINT=/data/cosmos_lite/Cosmos3-Edge-Policy-DROID
POLICY_PY=examples/quantized_robot_policy/.venv/bin/python

CUDA_VISIBLE_DEVICES=0 "$POLICY_PY" \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path "$CHECKPOINT" \
  --output-dir /data/runs/edge_bf16/server0 \
  --profile-jsonl /data/runs/edge_bf16/server0/profile.jsonl \
  --guidance 3 --num-steps 2 --shift 5 \
  --seed 0 --deterministic-seed --format-prompt-as-json True \
  --vae-encode-chunk-frames 8 --host 127.0.0.1 --port 8500
```

Start the second server on the other policy GPU and port, then launch the six
simulator lanes with the same client command shown below. Use
`--num-steps 4` only when reproducing the four-step control.

## What `--num-envs 10` Does

RoboLab officially uses Isaac Lab vector environments. One simulator process
creates ten copies of the same task and advances them with one batched
`env.step(actions)`. Ten environments therefore produce ten episodes in one
run. Prefer:

```bash
--num-envs 10 --num-runs 1
```

over `--num-envs 1 --num-runs 10`, which executes ten episodes sequentially.
The standard Cosmos3 client keeps a separate 32-action cache for every env,
but it does not batch model inference: envs needing a new chunk issue serial
batch-size-one requests. Sharing a policy server is useful only while its
request queue remains below saturation.

## Example 8-GPU Layouts

Discover the machine topology first:

```bash
nvidia-smi topo -m
```

On a symmetric 8 x RTX 4090 host, a practical `2P + 6S` mapping is:

```text
policy GPUs:    0,4
simulator GPUs: 1,2,3,5,6,7
server map:     0,0,0,1,1,1
```

For Nano W8A16 `3P + 5S`:

```text
policy GPUs:    0,4,7
simulator GPUs: 1,2,3,5,6
server map:     0,0,1,1,2
```

These IDs are examples, not portable constants. Place each simulator near its
server's CPU socket when the PCIe topology exposes locality. Bind the
simulator at process launch to the corresponding socket-wide CPU set; narrow
per-process masks did not improve the measured 4-pair workload.

## Launch Pattern

Start one long-lived server per policy GPU with a distinct port and run
directory. For example:

```bash
BUNDLE_DIR=/data/cosmos_lite/edge_genw8a8
DEPLOYMENT_CONFIG=examples/robolab_quant/configs/edge_genw8a8_fast_4090.yaml

POLICY_GPU=0 PORT=8500 RUN_DIR=/data/runs/edge_gen/server0 \
  BUNDLE_DIR="$BUNDLE_DIR" DEPLOYMENT_CONFIG="$DEPLOYMENT_CONFIG" \
  examples/robolab_quant/pipeline.sh serve

POLICY_GPU=4 PORT=8501 RUN_DIR=/data/runs/edge_gen/server1 \
  BUNDLE_DIR="$BUNDLE_DIR" DEPLOYMENT_CONFIG="$DEPLOYMENT_CONFIG" \
  examples/robolab_quant/pipeline.sh serve
```

Run each server in a service manager or its own terminal. Wait for every
`/healthz` endpoint before starting simulators. A simulator lane follows this
pattern:

```bash
COSMOS_ROLLOUT_START_DELAY_S=4 \
CUDA_VISIBLE_DEVICES=2 \
/path/to/RoboLab/.venv/bin/python /path/to/RoboLab/policies/cosmos3/run.py \
  --task <scene-grouped-task-list> \
  --remote-host 127.0.0.1 \
  --remote-port 8500 \
  --num-envs 10 \
  --num-runs 1 \
  --instruction-type default \
  --device cuda:0 \
  --headless \
  --video-mode none \
  --output-folder-name <stable-lane-name>
```

Use a stable output name when resuming. RoboLab reads `episode_results.jsonl`
and skips complete task/episode keys. Before a resume, verify that there are no
duplicate keys and that every task has exactly the requested number of
episodes.

## Scene-Aware Sharding

RoboLab-120 contains 120 tasks in 47 USD scenes. Rebuilding a scene costs much
more than switching tasks within one scene. Build lanes as follows:

1. Group tasks by scene and keep each group intact.
2. Estimate each scene group's cost from task horizons and historical
   environment-step timing.
3. Assign groups to simulator lanes with longest-processing-time greedy
   balancing.
4. Keep all tasks from one scene adjacent in a lane's task list.

Do not split tasks evenly by count: task horizons and scene physics vary
enough to create long tail lanes.

## Operational Guardrails

- Keep video disabled for throughput measurements.
- Record bundle manifest hash, deployment config, Git revision, GPU mapping,
  task manifest, instruction type, start/end times, and server profile JSONL.
- Check server health and the timestamp of the latest request event. Abort a
  lane if its server disappears instead of waiting indefinitely.
- Check GPU ownership before launch and periodically afterward; do not assume
  a previously free shared machine remains reserved.
- Keep one result file per lane and merge only after duplicate and coverage
  checks pass.
- Restart failed simulator lanes against the same stable result directory.
  For formal RNG comparisons, restart the complete run under a new run ID if
  policy-server RNG state was lost.

## Rejected or Experimental Layouts

- Native model batch size 2 was 9-21% slower than two serial batch-size-one
  calls and changed actions. It is not used.
- Edge W8A16 `2P + 8S` saturated policy servers and was slower than `2P + 6S`.
- Edge GenW8A8 colocated `2P + 8S` is experimental. It needs heterogeneous
  lane sizes and dynamic task assignment; equal shards are slower.
- A policy server shared by three BF16 Nano lanes is saturated. Future BF16
  reproduction should use one policy replica per simulator lane, subject to
  available accelerator memory.

For latency, VRAM, and quality protocols, see the
[Edge benchmarks](EDGE_BENCHMARKS.md) and
[Nano benchmarks](NANO_BENCHMARKS.md).
