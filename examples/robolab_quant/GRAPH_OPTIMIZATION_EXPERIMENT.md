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
BUNDLE_DIR=/path/to/gen_w8a8_bundle \
examples/robolab_quant/pipeline.sh replay
```

Dynamic compilation is required because prompt token lengths vary between
requests. Compilation is lazy: a cold first request took about 12.5 seconds
for Edge and 14.9 seconds for Nano. Production services must issue a warmup
request before control starts. The Inductor disk cache reduces later process
startup cost but is specific to the PyTorch, CUDA, GPU, and model environment.

## Replay32

Protocol: DROID train128 captured requests, guidance 3, two UniPC steps, seed
0, 32x8 actions, and one RTX 4090-NX-1 GPU per server. Request latency excludes
RoboLab simulation. Peak memory is CUDA reserved memory.

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
