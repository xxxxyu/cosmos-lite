# FP8 W8A8 Experiment

This document tracks experimental RTX 4090 FP8 paths for Cosmos3 Nano and Edge.
Two self-contained strategies are supported:

- `gen_branch_w8a8` keeps the non-generation branch in calibrated Marlin
  W4A16 and runs the generation branch with native Ada FP8 W8A8 kernels.
- `full_w8a8` runs every quantized Linear module with FP8 W8A8 kernels.

These paths are not part of the stable deployment recommendations yet.
Closed-loop quality, rather than replay parity alone, is the release gate.

## Candidates

| Strategy | Model | W4A16 modules | FP8 W8A8 modules |
| -------- | ----- | ------------: | ---------------: |
| GenW8A8  | Nano  |           252 |              252 |
| FullW8A8 | Nano  |             0 |              504 |
| GenW8A8  | Edge  |           168 |              168 |
| FullW8A8 | Edge  |             0 |              336 |

FP8 weights use E4M3 with one static scale per output channel. Activations use
E4M3 with dynamic per-token scales and BF16 output. The mixed strategy uses
calibrated INT4 weights for its retained W4 branch; dynamic FP8 activation
quantization itself does not require calibration. Both bundle types record
activation quantization in their manifest, directly load their packed payloads,
and do not depend on the BF16 source checkpoint at inference time. They require
native FP8 support (`SM89+`), which includes the RTX 4090 target.

Two activation variants were evaluated:

- **Dynamic**: per-token activation scaling, with no FP8 activation
  calibration. The retained W4 branch still uses the existing DROID train128
  calibration.
- **Equalized**: the same dynamic quantization after input-channel
  equalization from DROID train128 input-amax statistics (`alpha=0.5`).

## Operator Benchmark

Hardware and software: one RTX 4090, SM89, PyTorch 2.10.0+cu128, vLLM 0.19.1,
BF16 output, 10 warmup iterations, and 50 measured iterations. Shapes and call
counts were captured from one real DROID request with guidance 3, two UniPC
steps, and a 32x8 action output. Times below are isolated Linear forward times.

| Tokens x K x N      | Calls | BF16 (ms) | Marlin W4A16 (ms) | Marlin W8A16 (ms) | FP8 W8A8 (ms) |
| ------------------- | ----: | --------: | ----------------: | ----------------: | ------------: |
| 3093 x 4096 x 12288 |   288 |     1.940 |             1.876 |             1.927 |     **1.161** |
| 3093 x 4096 x 1024  |   288 |     0.175 |             0.172 |             0.176 |     **0.137** |
| 3093 x 4096 x 4096  |   288 |     0.658 |             0.631 |             0.646 |     **0.473** |
| 3093 x 12288 x 4096 |   144 |     1.929 |             1.846 |             1.899 |     **1.246** |

Applying the measured backend only to its actual branch gives the following
call-weighted projection for all quantized Linear operations in one request:

| Model Linear policy                        | Projected total (ms) | Reduction vs GenW8A16 |
| ------------------------------------------ | -------------------: | --------------------: |
| Non-generation W4A16 + generation W8A16    |              1,107.1 |                     - |
| Non-generation W4A16 + generation FP8 W8A8 |            **731.5** |             **33.9%** |

The projected 375.6ms Linear saving is close to the 337.4ms measured generation
latency saving below. This confirms that the end-to-end improvement comes from
the dominant large-M generation shapes. Applying FP8 indiscriminately to the
small non-generation shapes is not beneficial; Marlin W4A16 remains faster for
that branch.

## Nano Replay32

Protocol: the same 32 captured DROID train requests, one RTX 4090 on an
eight-GPU host, guidance 3, two UniPC steps, deterministic seed 0, and 32x8
actions. Latency is policy request latency and excludes RoboLab simulation. The
first request is retained,
but does not affect the median materially.

| Candidate           |  Request p50/p95 (ms) | Generate p50 (ms) | Peak alloc/reserved (GB) |     L1 mean | Element error p95 | Sample Linf p95 |
| ------------------- | --------------------: | ----------------: | -----------------------: | ----------: | ----------------: | --------------: |
| GenW8A16 reference  |     1,674.5 / 1,681.6 |           1,632.5 |            15.20 / 15.50 |           - |                 - |               - |
| Dynamic GenW8A8     | **1,340.0 / 1,345.2** |       **1,295.1** |            15.20 / 15.50 | **0.03229** |           0.10157 |         0.32710 |
| Equalized GenW8A8   |     1,361.8 / 1,368.3 |           1,324.3 |            15.20 / 15.50 |     0.03270 |       **0.10156** |     **0.29359** |
| FullW8A16 reference |     1,680.4 / 1,685.1 |           1,643.5 |            18.56 / 19.07 |           - |                 - |               - |
| Dynamic FullW8A8    | **1,329.9 / 1,337.7** |       **1,286.5** |            18.57 / 19.07 | **0.03009** |       **0.09115** |     **0.28891** |

Dynamic W8A8 reduces median request latency by 20.0% relative to GenW8A16. The
equalized variant improves one tail metric, but slightly worsens mean error and
maximum error while adding 21.9ms. It is therefore not a clear improvement and
the simpler dynamic variant was selected for closed-loop validation.

Replay parity is diagnostic. Released Nano GenW8A16 also diverges measurably
from FullW8 (`L1 mean=0.02825`, sample `Linf p95=0.28431`) while retaining strong
closed-loop quality, so an absolute Linf threshold is not used as a rollout
substitute.

FullW8A8 reduces median request latency by 20.9% relative to FullW8A16. It is
only 10.1ms faster than mixed Dynamic GenW8A8, while increasing peak reserved
memory by 3.57GB. This makes the mixed strategy the stronger Nano deployment
tradeoff despite the slightly lower FullW8A8 request latency.

## Edge Replay32

The Edge experiment uses the same replay protocol and DROID train128 requests
as Nano. References are the released Edge W8A16 bundles with matching branch
precision policies.

| Candidate           | Request p50/p95 (ms) | Generate p50 (ms) | Peak alloc/reserved (GB) |     L1 mean | Element error p95 | Sample Linf p95 |
| ------------------- | -------------------: | ----------------: | -----------------------: | ----------: | ----------------: | --------------: |
| GenW8A16 reference  |        573.5 / 589.6 |             525.9 |              6.36 / 8.79 |           - |                 - |               - |
| Dynamic GenW8A8     |    **502.8 / 526.8** |         **464.3** |              6.36 / 8.79 | **0.02685** |       **0.08037** |     **0.27557** |
| FullW8A16 reference |        567.7 / 584.0 |             520.6 |              6.36 / 8.71 |           - |                 - |               - |
| Dynamic FullW8A8    |    **510.6 / 526.3** |         **472.2** |              6.36 / 8.71 | **0.03106** |       **0.09144** |     **0.30098** |

GenW8A8 reduces median request latency by 12.3% relative to GenW8A16;
FullW8A8 reduces it by 10.1% relative to FullW8A16. For Edge, GenW8A8 is both
7.8ms faster and closer to its W8A16 reference than FullW8A8, so quantizing the
remaining small non-generation Linear shapes to FP8 has no measured benefit.

## Nano Closed Loop

RoboLab `BananaInBowlTask`, one environment, guidance 3, two UniPC steps,
32 generated and executed actions per policy request, and no video recording:

| Candidate                       | Episodes | Successes | Success Rate |
| ------------------------------- | -------: | --------: | -----------: |
| Dynamic GenW8A8 short gate      |        5 |         5 |         100% |
| Dynamic GenW8A8 full validation |       50 |        47 |      **94%** |

The full validation used initial states 0-49. It was sharded across four
independent policy/simulator GPU pairs to reduce wall time; every worker still
used one environment and the same model seed and sampler. The merged result
contains one result for every episode ID and three failures. The 94% point
estimate passes the closed-loop quality gate; its interval overlaps the
released FullW8 two-step Banana result (90%).

This result supports retaining the W8A8 path as an experimental high-throughput
option. Stable README recommendations remain unchanged until the implementation
and bundle distribution are reviewed for release.

## Edge Closed Loop

RoboLab `BananaInBowlTask` uses the same protocol as Nano: one environment,
guidance 3, two UniPC steps, 32 generated and executed actions per policy
request, and no video recording.

| Candidate           |        Success (Wilson 95% CI) | Paired wins/losses vs W8A16 | McNemar p |
| ------------------- | -----------------------------: | --------------------------: | --------: |
| GenW8A16 reference  |     40/50 = 80% [67.0%, 88.8%] |                   reference |         - |
| Dynamic GenW8A8     | **39/50 = 78% [64.8%, 87.2%]** |                         8/9 |     1.000 |
| FullW8A16 reference |     36/50 = 72% [58.3%, 82.5%] |                   reference |         - |
| Dynamic FullW8A8    | **37/50 = 74% [60.4%, 84.1%]** |                        10/9 |     1.000 |

Each candidate is evaluated on exactly the initial states 0-49. Two independent
policy/simulator GPU pairs evaluate disjoint 25-episode ranges. The merged
results contain one entry for each episode ID across the two ranges.
Neither FP8 candidate is distinguishable from its precision-matched W8A16
reference on this 50-episode protocol. GenW8A8 retains the higher point estimate
and the lower replay error while also being faster than FullW8A8, so it is the
preferred Edge FP8 candidate.

## Reproduction

Convert only the generation branch of an existing self-contained Nano or Edge
GenW8 bundle:

```bash
python -m cosmos_framework.scripts.robolab_quant_pipeline \
  convert-gen-w8-to-w8a8 \
  --base-bundle /path/to/nano-gen-w8-bundle \
  --source-checkpoint /path/to/Cosmos3-Policy-DROID \
  --output-dir /path/to/nano-gen-w8a8-bundle \
  --device cuda:0

python -m cosmos_framework.scripts.robolab_quant_pipeline validate \
  --bundle-dir /path/to/nano-gen-w8a8-bundle \
  --expected-strategy gen_branch_w8a8 \
  --check-hashes \
  --check-tensors
```

The converter streams source safetensor shards, reuses the existing calibrated
W4 payloads and residual assets, and rewrites only the generation payloads. The
result does not depend on the BF16 checkpoint at inference time.

Convert every quantized Linear module of an existing self-contained FullW8
bundle:

```bash
python -m cosmos_framework.scripts.robolab_quant_pipeline \
  convert-full-w8-to-w8a8 \
  --base-bundle /path/to/full-w8-bundle \
  --source-checkpoint /path/to/Cosmos3-Policy-DROID \
  --output-dir /path/to/full-w8a8-bundle \
  --device cuda:0

python -m cosmos_framework.scripts.robolab_quant_pipeline validate \
  --bundle-dir /path/to/full-w8a8-bundle \
  --expected-strategy full_w8a8 \
  --check-hashes \
  --check-tensors
```
