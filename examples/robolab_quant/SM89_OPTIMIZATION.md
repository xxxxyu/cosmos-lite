# RTX 4090 SM89 Inference Optimization

This report evaluates training-free operator and compute-graph optimizations
for Cosmos3 Edge and Nano GenW8A8 robot policies on one RTX 4090. The protocol
uses DROID train captures or RoboLab `BananaInBowlTask`, guidance 3, two UniPC
steps, batch size one, and a 32x8 action chunk.

The optimizations are independent of packed quantization. Removing every flag
below restores the previously released CUTLASS FP8 + FlashAttention2 path:

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
FP8_PROJECTION_FUSION=shared \
SAGE_ATTENTION=1 \
CONDITION_KV_CACHE=1 \
examples/robolab_quant/pipeline.sh replay
```

`CONDITION_KV_CACHE=1` is a Nano-only candidate. It is correct on Edge,
but did not produce a repeatable Edge latency reduction.

## Operator Findings

### FP8 GEMM

The dominant Nano FP8 projection has `M=3093`, `K=4096`, and `N=12288`.
Nsight Compute identified the vLLM CUTLASS kernel as a `128x128x64` Stream-K
kernel with 255 registers per thread, 81.92 KB dynamic shared memory per block,
8.33% theoretical occupancy, and about one active warp per scheduler. Compute
and memory throughput were 39.65% and 54.35%; DRAM throughput was only 13.97%
with a 98.96% L2 hit rate. No eligible warp accounted for 90.27% of scheduler
cycles.

The kernel is resource and latency limited rather than Tensor Core or DRAM
bandwidth limited. A lower-register, lower-shared-memory SM89 tile may improve
it, but vLLM does not expose a runtime tile selector for this operation. The
PyTorch cuBLASLt `torch._scaled_mm` path was 3-4x slower on the policy shapes.
The release therefore retains the mature vLLM CUTLASS kernel and the existing
shared activation quantization. A new custom GEMM is not justified by the
available evidence.

### Shape-Aware Attention

The generation tower uses long dense attention while the understanding tower
uses short causal attention. One backend is not optimal for both shapes:

| Model and shape | FlashAttention2 (ms) | SageAttention 2.2.0 (ms) |
| --- | ---: | ---: |
| Edge Gen, Q=3093, KV=3175, Hq/Hkv=16/8 | 0.706 | **0.326** |
| Nano Gen, Q=3093, KV=3175, Hq/Hkv=32/8 | 1.220 | **0.551** |
| Edge Und causal, 82 tokens | **0.084** | 0.245 |
| Nano Und causal, 82 tokens | **0.083** | 0.231 |

The optional policy selects Sage only for SM89 inference with dense,
non-causal attention and Q length at least 512. FlashAttention2 remains active
for short causal Und attention, other architectures, training, varlen inputs,
and installations without SageAttention.

The adapter uses Sage's SM89 INT8-QK/FP8-PV kernels with FP32+FP32
accumulation. SageAttention 2.2.0's default FP32+FP16 SM89 path produced a CUDA
launch failure in the validated PyTorch 2.10 / CUDA 12.8 environment and is not
used. The adapter decomposes the Python scheduler so Inductor can compile the
surrounding graph; only the two upstream pybind quantization operations remain
small opaque custom ops.

### Condition K/V Cache

Within one policy request, Und tokens and their per-layer K/V are invariant
across diffusion timesteps. The cache performs the first cond and uncond
forwards normally, stores separate per-layer Und K/V for the two CFG branches,
and runs only the Gen pathway on the second denoise step. Cache lifetime is one
request; no state crosses robot observations.

The implementation is opt-in and limited to batch-one, two-way, non-CP
inference. Unsupported modes fail explicitly. Cache storage adds about 15 MB
of reserved memory on Nano and does not affect the 24 GB deployment gate.

## Replay Results

Steady request latency excludes the first lazy compile request. All rows use
the same GenW8A8 bundle, sampling settings, and shared FP8 projection fusion.

The ablation below used an isolated replay8 run to separate the attention and
cache contributions:

| Model | Attention | Condition cache | Request p50 (ms) | Peak reserved (GB) |
| --- | --- | --- | ---: | ---: |
| Edge | FlashAttention2 | off | 428.6 | 8.79 |
| Edge | shape-aware Sage | off | **390.6** | 8.79 |
| Edge | shape-aware Sage | on | 396.0 | 8.79 |
| Nano | FlashAttention2 | off | 1119.9 | 15.50 |
| Nano | shape-aware Sage | off | 1001.4 | 15.50 |
| Nano | shape-aware Sage | on | **985.9** | 15.51 |

The selected profiles were then rerun with 32 DROID training captures on an
otherwise idle eight-GPU RTX 4090 host. The first lazy compile request is
excluded from every percentile. Baselines use the same compile and shared-FP8
projection settings with FlashAttention2.

| Model | Runtime | Request p50/p95 (ms) | Server p50/p95 (ms) | Peak reserved (GB) |
| --- | --- | ---: | ---: | ---: |
| Edge | FlashAttention2 baseline | 435.8 / 462.7 | 433.3 / 460.2 | 8.79 |
| Edge | shape-aware Sage | **386.0 / 390.1** | **383.5 / 387.6** | 8.79 |
| Nano | FlashAttention2 baseline | 1126.9 / 1133.1 | 1124.5 / 1130.9 | 15.50 |
| Nano | shape-aware Sage + cache | **987.2 / 992.3** | **984.6 / 989.7** | 15.51 |

The cache changes the compiled graph and therefore floating reduction order.
Against the no-cache decomposed-Sage output, replay8 measured Edge
`L1=0.02186 / Linf p95=0.16125` and Nano
`L1=0.03093 / Linf p95=0.23286`. Replay error is a diagnostic, not a robot
quality metric; the release decision uses paired closed-loop evaluation.

## Closed-Loop Results

All settings use the same Banana task, 50 deterministic run IDs, one
environment, guidance 3, two UniPC steps, and the same GenW8A8 bundle for each
model family.

| Model | Runtime | Success (Wilson 95% CI) | Paired wins/losses vs baseline | McNemar p |
| --- | --- | ---: | ---: | ---: |
| Edge | FlashAttention2 baseline | 37/50 = 74% [60.4%, 84.1%] | reference | - |
| Edge | shape-aware Sage | 34/50 = 68% [54.2%, 79.2%] | 8/11 | 0.648 |
| Nano | FlashAttention2 baseline | 49/50 = 98% [89.5%, 99.6%] | reference | - |
| Nano | shape-aware Sage | 46/50 = 92% [81.2%, 96.8%] | 1/4 | 0.375 |
| Nano | shape-aware Sage + cache | **49/50 = 98% [89.5%, 99.6%]** | 1/1 | 1.000 |

The Edge difference is not statistically significant at this sample size, but
its six-point lower estimate is unfavorable. Sage therefore remains an
experimental Edge opt-in; the stable Edge profile keeps FlashAttention2. Nano
Sage plus the condition cache matches the baseline aggregate and has only two
discordant paired outcomes, so it is the validated Nano low-latency profile.
The cache alone is not claimed to improve quality: its 49/50 result versus
46/50 without cache has McNemar `p=0.375`.

## Deployment Notes

SageAttention is a pinned, optional local CUDA build:

```bash
examples/quantized_robot_policy/install_sage_attention.sh
```

The first request lazily compiles new graph variants and must be used as a
warmup before control starts. The Inductor cache is specific to the PyTorch,
CUDA, GPU, and model environment. Production services should monitor the
`server_ready` and request events in `profile.jsonl`; they record compile,
Sage-request, cache, memory, and latency settings.

For Nano, enable both `SAGE_ATTENTION=1` and `CONDITION_KV_CACHE=1`. For Edge,
omit both for the stable profile; `SAGE_ATTENTION=1` is available when the
latency target outweighs the unresolved closed-loop point-estimate risk.
Removing both variables is the complete rollback and does not require
repacking the model.
