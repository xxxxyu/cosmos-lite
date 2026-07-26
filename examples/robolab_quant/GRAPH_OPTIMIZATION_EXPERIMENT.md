# Training-Free Graph Optimization

This document evaluates training-free compute-graph optimization for Cosmos3
Nano and Edge GenW8A8 on one RTX 4090. Quantization, sampler settings, model
inputs, and action execution remain unchanged.

## Selected Configuration

The retained configuration compiles each complete MoT language block with
symbolic shapes and does not enable Inductor CUDA Graphs:

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
CUDA_GRAPHS=0 \
FP8_PROJECTION_FUSION=shared \
BUNDLE_DIR=/path/to/gen_w8a8_bundle \
examples/robolab_quant/pipeline.sh replay
```

Dynamic compilation is required because prompt token lengths vary between
requests. Compilation is lazy: a cold first request took about 12.5 seconds
for Edge and 14.9 seconds for Nano. Production services must issue a warmup
request before control starts. The Inductor disk cache reduces later process
startup cost but is specific to the PyTorch, CUDA, GPU, and model environment.

`FP8_PROJECTION_FUSION=shared` is specific to FP8 W8A8 bundles. It dynamically
quantizes a common activation once for the generation Q/K/V projections and,
on Nano, for the gated-MLP gate/up projections. The existing CUTLASS GEMMs and
their packed weights are unchanged, so this optimization is bit-identical to
the independent-Linear path.

## Replay32

Protocol: DROID train128 captured requests, guidance 3, two UniPC steps, seed
0, 32x8 actions, and one RTX 4090 GPU per server on an eight-GPU host. Request
latency excludes RoboLab simulation. Peak memory is CUDA reserved memory.

| Model | Runtime | Request p50/p95 (ms) | Generate p50 (ms) | Speedup | Peak (GB) | L1 vs eager | Sample Linf p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Edge | Eager | 502.8 / 524.1 | 462.2 | reference | 8.79 | - | - |
| Edge | Language dynamic | **441.4 / 456.0** | **397.3** | **1.14x** | 8.79 | 0.02436 | 0.23286 |
| Nano | Eager | 1322.7 / 1335.5 | 1277.0 | reference | 15.50 | - | - |
| Nano | Language dynamic | **1130.9 / 1147.8** | **1082.8** | **1.17x** | 15.50 | 0.02930 | 0.23445 |

The optimized path reduces median request latency by 12.2% for Edge and 14.5%
for Nano without increasing peak reserved memory. The gain comes from compiling
the repeated MoT blocks: Inductor fuses eligible pointwise, normalization, and
residual work and removes Python dispatch between operations. The vLLM CUTLASS
FP8 quantization and scaled-GEMM custom operators remain opaque kernel calls.

## Closed Loop

RoboLab `BananaInBowlTask`, the same initial states 0-49, one environment,
guidance 3, two UniPC steps, and all 32 generated actions executed:

| Model | Runtime | Success (Wilson 95% CI) | Paired wins/losses vs eager | McNemar p |
|---|---|---:|---:|---:|
| Edge | Eager | 39/50 = 78% [64.8%, 87.2%] | reference | - |
| Edge | Language dynamic | 37/50 = 74% [60.4%, 84.1%] | 9/11 | 0.824 |
| Nano | Eager | 47/50 = 94% [83.5%, 97.9%] | reference | - |
| Nano | Language dynamic | **49/50 = 98% [89.5%, 99.6%]** | 3/1 | 0.625 |

Neither compiled result is distinguishable from its eager reference on 50
episodes. Compile changes fusion and reduction order, so action tensors are not
bit-identical. Nano passes both latency and closed-loop gates. Edge passes the
statistical gate but has a four-point lower success-rate estimate, so it remains
an opt-in mode pending broader task coverage.

## Candidate Matrix

| Candidate | Result | Decision |
|---|---|---|
| Language blocks, dynamic | 12-15% request reduction | Retain |
| Language blocks, static | New prompt lengths trigger 6-7s recompiles | Reject |
| All VFM heads + language, dynamic | Same steady latency; 2x cold compile and larger parity error | Reject |
| Dynamic + CUDA Graphs | No steady gain; new-shape capture overhead | Reject |
| Dynamic + CUDA Graphs, repeated prompt | 433.6ms Edge p50, no gain over compile alone | Reject |
| MLP-only compile | 1-2% gain and worse parity | Reject |
| Attention-only compile | About 6% gain and worse parity | Reject |
| Inductor `force_same_precision` | Did not improve parity | Reject |

CUDA Graph support exists in the upstream MoT architecture, including padded
sequence packing and graph step markers. It is not selected here because policy
requests have varying text shapes and the block-level graph replay did not beat
dynamic compile without graphs. A manually captured whole request would also
conflict with CPU preprocessing, tokenization, scheduler state, and the final
action transfer.

## Runtime Changes

- The RoboLab server now exposes compile region, dynamic-shape, and CUDA Graph
  controls instead of relying on hidden inference defaults.
- The one-command pipeline forwards the same controls for serve, replay, and
  rollout.
- Disabled Linear shape recording is statically skipped in quantized forwards.
  This removes unnecessary eager Python dispatch and prevents layer-name guards
  from forcing one compiled graph per layer.
- Compile rejects simultaneous `COSMOS3_LINEAR_SHAPES_JSONL` recording because
  the recorder performs Python aggregation that is incompatible with MoT
  fullgraph capture.

No unnecessary explicit CUDA synchronization was present in the request hot
path. The final device-to-CPU action transfer is the required request completion
boundary. Sampler timestep transfers remain, but they are few and were not
changed without evidence that an invasive packing API rewrite would have a
meaningful end-to-end return.

## FP8 Projection Fusion Follow-up

An Nsight Systems trace over four eager GenW8A8 requests measured the kernel
headroom before changing the runtime:

| Model | FP8 GEMM | Dynamic FP8 quant | FlashAttention | Quant time/request |
|---|---:|---:|---:|---:|
| Edge | 41.3% | 3.1% | 18.2% | 10.6 ms |
| Nano | 54.2% | 3.3% | 13.3% | 34.6 ms |

Dynamic quantization alone is not a large end-to-end bottleneck. The useful
opportunity is eliminating repeated quantization of the same activation
without replacing the tuned CUTLASS GEMMs.

The real-shape operator benchmark used 3,093 tokens on RTX 4090:

| Projection group | Separate (ms) | Shared quant (ms) | Packed GEMM (ms) |
|---|---:|---:|---:|
| Edge QKV, K=2,048, N=2,048+1,024+1,024 | 0.375 | 0.357 | **0.311** |
| Nano QKV, K=4,096, N=4,096+1,024+1,024 | 0.836 | **0.791** | 0.797 |
| Nano gate/up, K=4,096, N=12,288+12,288 | 2.345 | 2.306 | **2.220** |

Packed GEMM concatenates weights along the output dimension. Although it wins
two isolated shapes, it changes CUTLASS accumulation tiling, is not
bit-identical, and raises Nano peak reserved memory to 19.0GB when weights are
assembled at load time. It did not consistently beat shared quantization in
replay, so it is not exposed by the deployment runtime.

Sequential replay32 on one otherwise idle RTX 4090 used dynamic language-block
compile in both rows. The first cold-compile request is excluded from latency:

| Model | FP8 projections | Request p50/p95 (ms) | Generate p50 (ms) | Peak reserved (GB) | L1 / Linf p95 |
|---|---|---:|---:|---:|---:|
| Edge | Independent | 436.8 / 438.2 | 398.4 | 8.79 | 0 / 0 |
| Edge | Shared quant | **428.6** / 443.5 | **394.6** | 8.79 | 0 / 0 |
| Nano | Independent | 1144.3 / 1148.5 | 1108.6 | 15.50 | 0 / 0 |
| Nano | Shared quant | **1119.9 / 1124.3** | **1082.3** | 15.50 | 0 / 0 |

Shared quantization reduces median request latency by 1.9% for Edge and 2.1%
for Nano on top of block compilation. All 32 action tensors are bit-identical
to the independent-projection compile reference. Edge p95 includes transient
clock variation late in the run; its p50 and kernel-level saving are the more
stable indicators for this small optimization.

A five-episode closed-loop integration smoke test of the final shared-quant
runtime completed successfully for both models:

| Model | Runtime | Banana success |
|---|---|---:|
| Edge | Language dynamic + shared quant | 5/5 |
| Nano | Language dynamic + shared quant | 4/5 |

This small sample verifies the end-to-end server and RoboLab integration; it is
not used as a success-rate estimate. Quality evidence comes from exact replay32
parity for shared quantization and the 50-episode compile comparison above.
