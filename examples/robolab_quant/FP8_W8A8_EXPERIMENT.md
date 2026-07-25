# Nano FP8 W8A8 Experiment

This document tracks an experimental RTX 4090 path for Cosmos3 Nano. It keeps
the non-generation branch in calibrated Marlin W4A16 and replaces generation
branch W8A16 Linear modules with native Ada FP8 W8A8 kernels.

The path is not part of the stable deployment recommendations yet. Closed-loop
quality, rather than replay parity alone, is the release gate.

## Candidate

| Branch | Modules | Weights | Activations | Backend |
|---|---:|---|---|---|
| Generation | 252 | FP8 E4M3, per output channel | FP8 E4M3, dynamic per token | vLLM CUTLASS scaled GEMM |
| Non-generation | 252 | INT4, calibrated per group | BF16 | vLLM Marlin W4A16 |

The self-contained `gen_branch_w8a8` bundle records activation quantization in
its manifest and directly loads both packed backend formats. It requires native
FP8 support (`SM89+`), which includes the RTX 4090 target.

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

| Tokens x K x N | Calls | BF16 (ms) | Marlin W4A16 (ms) | Marlin W8A16 (ms) | FP8 W8A8 (ms) |
|---|---:|---:|---:|---:|---:|
| 3093 x 4096 x 12288 | 288 | 1.940 | 1.876 | 1.927 | **1.161** |
| 3093 x 4096 x 1024 | 288 | 0.175 | 0.172 | 0.176 | **0.137** |
| 3093 x 4096 x 4096 | 288 | 0.658 | 0.631 | 0.646 | **0.473** |
| 3093 x 12288 x 4096 | 144 | 1.929 | 1.846 | 1.899 | **1.246** |

Applying the measured backend only to its actual branch gives the following
call-weighted projection for all quantized Linear operations in one request:

| Model Linear policy | Projected total (ms) | Reduction vs GenW8A16 |
|---|---:|---:|
| Non-generation W4A16 + generation W8A16 | 1,107.1 | - |
| Non-generation W4A16 + generation FP8 W8A8 | **731.5** | **33.9%** |

The projected 375.6ms Linear saving is close to the 337.4ms measured generation
latency saving below. This confirms that the end-to-end improvement comes from
the dominant large-M generation shapes. Applying FP8 indiscriminately to the
small non-generation shapes is not beneficial; Marlin W4A16 remains faster for
that branch.

## Replay32

Protocol: the same 32 captured DROID train requests, RTX 4090-NX-1, guidance 3,
two UniPC steps, deterministic seed 0, and 32x8 actions. Latency is policy
request latency and excludes RoboLab simulation. The first request is retained,
but does not affect the median materially.

| Candidate | Request p50/p95 (ms) | Generate p50 (ms) | Peak alloc/reserved (GB) | L1 mean | Element error p95 | Sample Linf p95 |
|---|---:|---:|---:|---:|---:|---:|
| GenW8A16 reference | 1,674.5 / 1,681.6 | 1,632.5 | 15.20 / 15.50 | - | - | - |
| Dynamic GenW8A8 | **1,340.0 / 1,345.2** | **1,295.1** | 15.20 / 15.50 | **0.03229** | 0.10157 | 0.32710 |
| Equalized GenW8A8 | 1,361.8 / 1,368.3 | 1,324.3 | 15.20 / 15.50 | 0.03270 | **0.10156** | **0.29359** |

Dynamic W8A8 reduces median request latency by 20.0% relative to GenW8A16. The
equalized variant improves one tail metric, but slightly worsens mean error and
maximum error while adding 21.9ms. It is therefore not a clear improvement and
the simpler dynamic variant was selected for closed-loop validation.

Replay parity is diagnostic. Released Nano GenW8A16 also diverges measurably
from FullW8 (`L1 mean=0.02825`, sample `Linf p95=0.28431`) while retaining strong
closed-loop quality, so an absolute Linf threshold is not used as a rollout
substitute.

## Closed Loop

RoboLab `BananaInBowlTask`, one environment, guidance 3, two UniPC steps,
32 generated and executed actions per policy request, and no video recording:

| Candidate | Episodes | Successes | Success Rate |
|---|---:|---:|---:|
| Dynamic GenW8A8 short gate | 5 | 5 | 100% |
| Dynamic GenW8A8 full validation | 50 | 47 | **94%** |

The full validation used initial states 0-49. It was sharded across four
independent policy/simulator GPU pairs to reduce wall time; every worker still
used one environment and the same model seed and sampler. The canonical result
contains exactly one real result for every episode ID, no resume placeholders,
and three failures. The 94% point estimate passes the closed-loop quality gate
and exceeds the released FullW8 two-step Banana result (90%), although 50
episodes are not enough to claim a statistically significant quality gain.

This result supports retaining the W8A8 path as an experimental high-throughput
option. Stable README recommendations remain unchanged until the implementation
and bundle distribution are reviewed for release.

## Reproduction

Convert only the generation branch of an existing self-contained Nano GenW8
bundle:

```bash
python -m cosmos_framework.scripts.robolab_quant_pipeline \
  convert-gen-w8-to-w8a8 \
  --base-bundle /path/to/nano-gen-w8-bundle \
  --source-checkpoint /path/to/Cosmos3-Nano-Policy-DROID \
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
