<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# From Cosmos Framework to Cosmos Lite: Robot Policy Compression and Inference Optimization

[中文版](cosmos_lite_optimization_report_zh.md)

## Executive Summary

Cosmos Lite started with a concrete engineering goal: take the Cosmos3 Nano
16B robot policy from a data-center GPU workflow to a system that could be
deployed, reproduced, and evaluated on one 24GB RTX 4090. The project later
added Cosmos3 Edge 4B and continued optimizing single-robot, batch-one request
latency.

No single optimization produced the final result. The useful gains came from
several layers of work:

1. W4A16/W8A16 weight-only quantization made Nano fit, while self-contained
   bundles made the result directly deployable.
2. Reducing UniPC denoise steps from four to two halved denoiser forwards while
   keeping guidance at 3.
3. FP8 W8A8 on the generation branch accelerated the large matrix
   multiplications that dominate each request.
4. Compiling complete MoT language blocks fused eligible pointwise work and
   removed Python dispatch between operations.
5. Shared FP8 activation quantization avoided quantizing the same Q/K/V or
   gated-MLP input more than once.
6. Attention was selected by real shape: SageAttention for long generation
   attention and FlashAttention2 for short condition attention. Nano also
   reuses invariant condition K/V within a request.
7. Edge uses a shape-tuned Triton FP8 kernel for its dominant RTX 4090/SM89
   GEMMs.
8. The request data path now resizes the one observed frame only, instead of
   also resizing 32 future frames that are known to be zero.

Under the complete RoboLab-120 benchmark in `default` instruction mode,
guidance 3, two denoise steps, and a 32x8 action chunk, the fastest retained
configurations are:

| Model            | Current configuration                                                                 | RTX 4090 request p50 | Peak reserved |     RoboLab-120 SR |
| ---------------- | ------------------------------------------------------------------------------------- | -------------------: | ------------: | -----------------: |
| Cosmos3 Edge 4B  | GenW8A8 + compile + shared FP8 + Triton SM89 + Sage FP16-PV + sparse transform        |         **331.4 ms** |       8.79 GB | 231/1,200 = 19.25% |
| Cosmos3 Nano 16B | GenW8A8 + compile + shared FP8 + Sage FP8-PV + condition K/V cache + sparse transform |         **958.5 ms** |      15.51 GB | 380/1,200 = 31.67% |

Each full-suite row uses 10 episodes for each of 120 tasks. Focused paired
Banana gates reached 40/50 for Edge and 49/50 for Nano; those smaller results
remain useful for controlled ablations but are not the headline quality metric.

These numbers describe the current optimization branch, not the W4A16/W8A16-
only `v0.2.0` stable release. This report explains what was tried, what was
kept, and why other candidates were rejected or deferred.

## 1. Scope and Measurement

### 1.1 This is not a conventional LLM

Cosmos3 Policy uses a MoT, or Mixture of Transformers, structure. Each decoder
layer has two main compute paths:

- The **understanding branch** processes language, state, and visual
  conditioning.
- The **generation branch** processes diffusion latents and produces an action
  chunk.

Each branch has its own attention and MLP projections. Their token counts,
matrix shapes, call frequencies, and sensitivity to lower precision differ.
Describing the model as a language model with a small action head therefore
leads to poor quantization and backend decisions.

The quantization plans target 504 MoT language Linear modules in Nano and 336
in Edge. Action adapters, embeddings, normalization, vision components, and
the VAE remain BF16. In this report, `full_w8a8` means all of these target
Linear modules, not every operation in the complete model.

### 1.2 What W4A16, W8A16, and W8A8 mean

- **W4A16** uses 4-bit weights and BF16/FP16 activations.
- **W8A16** uses 8-bit weights and BF16/FP16 activations.
- **W8A8** uses 8-bit weights and 8-bit matrix-multiply inputs. The fast path
  in this report uses FP8 E4M3 and returns BF16 outputs.

W4A16 and W8A16 save memory first. They accelerate inference only when the
kernel matches the actual model shapes. W8A8 can reduce Tensor Core work
further, but activation quantization has a cost of its own. A fast FP8 GEMM is
not useful if preparing its FP8 input takes longer than the GEMM saves.

### 1.3 Latency and quality use different boundaries

- **Kernel latency** measures one isolated operator. It shows potential, not
  application speed.
- **Generate latency** measures `generate_samples_from_batch` inside the
  server.
- **Request latency** measures server input to returned action, including
  preprocessing, generation, and small postprocessing, but excluding IsaacSim.
- **Success rate (SR)** measures closed-loop task completion and is the final
  quality gate.

Open-loop replay also reports action error:

- **L1 mean** is the mean absolute difference across all action elements.
- **Linf p95** first takes the largest action-element error in each request,
  then reports the 95th percentile across requests.

These diagnostics screen candidates quickly. Rollout remains the quality
measure because a small action difference may grow through closed-loop
interaction or have no effect on task completion.

### 1.4 Short glossary

- A **shape** is the set of matrix dimensions handled by an operator. Kernel
  speed can change sharply when these dimensions change.
- **p50** is the median: half of measured requests are faster and half are
  slower.
- A **paired rollout** runs two policies from the same initial states and
  seeds, making a quality comparison less noisy than two unrelated runs.
- **Peak reserved** is the largest CUDA memory pool reserved by PyTorch. It is
  a conservative deployment number and can exceed memory occupied by live
  tensors.
- In attention, **Q/K/V** are query, key, and value tensors. **PV** is the
  second matrix multiplication, where attention probabilities are multiplied
  by values.

## 2. Three Stages of the System

| Stage                                      | Nano 16B                                                                   | Edge 4B                                 | Main outcome                                                       |
| ------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| Original NVIDIA framework                  | CloseFridge BF16 loaded at 31.76GB and peaked at 33.09GB; no 4090 baseline | BF16 g3/s2: 582ms, 9.20GB, 251/1,200    | Training and inference worked, but Nano was not a 4090 deployment  |
| First stable Cosmos Lite quantized release | Full W8 g3/s2: 2403ms, 21.42GB, 378/1,200                                  | Full W8 g3/s2: 576ms, 8.71GB, 239/1,200 | Self-contained W4/W8 bundles and one-command replay/rollout        |
| Current fastest optimization branch        | 958.5ms, 15.51GB, 380/1,200                                                | 331.4ms, 8.79GB, 231/1,200              | FP8, compile, attention/cache, SM89 kernels, and data-path cleanup |

There is no reliable single-4090 Nano BF16 request number because the model
first fails the 24GB deployment constraint. Relative to the first deployable
Full W8 g3/s2 baseline, current Nano request p50 is about **2.51x faster**.
Edge BF16 already fit; the current path is about **1.76x faster** than its
582ms BF16 g3/s2 baseline.

The original Nano BF16 memory figure was measured on RoboCasa CloseFridge,
whereas the current fastest Nano latency and SR were measured with the DROID
policy on RoboLab-120. The cross-environment row shows the deployment history;
the matched RoboLab Full W8 baseline is the source of the 2.51x speedup.

The headline success rates use 1,200 episodes across 120 tasks. Their
confidence intervals overlap, so differences of a few points remain uncertain
outside this robot and observation contract. Paired 50-episode Banana runs are
used for the ablations below.

### 2.1 Decision map

| Area                 | Technique                                            | Evidence level                                     | Final status                            |
| -------------------- | ---------------------------------------------------- | -------------------------------------------------- | --------------------------------------- |
| Artifact             | Streaming quantization and self-contained bundles    | Fresh deployment, hash validation, replay, rollout | Retained                                |
| Weight precision     | Marlin W8A16                                         | Full replay and rollout on Nano/Edge/RoboCasa      | Stable default                          |
| Weight precision     | Marlin W4A16 and fixed W4/W8 mixes                   | Full replay and rollout                            | Retained as selectable tradeoffs        |
| Calibration          | Training-set AWQ-style input-channel scaling         | 128-frame calibration and held-out replay          | Retained for every W4-containing bundle |
| Sampling             | Guidance 3, two UniPC steps                          | Paired 50-episode RoboLab gates                    | Retained for RoboLab                    |
| Sampling             | Guidance 1                                           | Replay and paired rollout                          | Rejected as a general default           |
| Activation precision | Generation-branch FP8 W8A8                           | Operator, replay32, and 50-episode rollout         | Retained in fastest branch              |
| Activation precision | Full FP8 W8A8                                        | Operator, replay32, and 50-episode rollout         | Supported, not preferred                |
| Activation precision | W4A8                                                 | API and kernel feasibility study only              | Deferred; no model integration          |
| Graph                | Dynamic complete-language-block compile              | Replay32 and 50-episode rollout                    | Retained                                |
| Graph                | Static compile, broad VFM compile, CUDA Graphs       | End-to-end replay                                  | Rejected                                |
| Projection           | Shared FP8 activation quantization                   | Bit-exact replay32                                 | Retained                                |
| Projection           | Concatenated packed projection GEMM                  | Operator and replay                                | Rejected                                |
| Attention            | Shape-aware Sage plus FA2 fallback                   | Operator, replay32, rollout                        | Retained                                |
| Attention            | FlashInfer                                           | Operator benchmark                                 | Rejected                                |
| Cache                | Request-local condition K/V                          | Replay and rollout                                 | Retained for Nano only                  |
| Cache                | PAB/TeaCache/SmoothCache/BAC-style approximate reuse | Literature and headroom analysis                   | Deferred; not implemented               |
| GEMM                 | Shape-tuned Triton SM89 FP8                          | NCU, operator, replay, rollout                     | Retained for Edge only                  |
| Data path            | Sparse future-frame transform                        | Bit-exact replay and end-to-end latency            | Retained by default                     |
| Input/control        | Shorter action chunk, fewer views, pre-resize        | End-to-end tests                                   | Rejected                                |
| CFG execution        | Batch conditional/unconditional branches             | End-to-end replay                                  | Rejected                                |
| Alternative engines  | AllSpark                                             | Real-shape operator benchmark                      | Not integrated                          |
| Alternative engines  | ExLlama, TensorRT-LLM, Machete                       | Architecture/API survey                            | Not implemented; see Section 5          |

## 3. Deployability First: Weight-Only Quantization and Bundles

### 3.1 Why Marlin W4A16/W8A16 was selected

Early candidates included GPTQ/AWQ-style W4A16, torchao INT8 weight-only,
vLLM Marlin and AllSpark, ExLlama-style kernels, TensorRT-LLM, and Machete.
Marlin was the pragmatic first production backend because it:

- supports Ada/SM89, including RTX 4090;
- provides mature W4/W8 packed layouts and CUDA kernels;
- can replace PyTorch Linear modules without converting the whole MoT model
  into an LLM serving engine; and
- allows W4 and W8 modules in one model.

Machete requires SM90/Hopper and is not a 4090 backend. ExLlamaV2/V3 performs
well on consumer-GPU LLM inference, but its runtime and model assumptions did
not provide a straightforward compatible Linear replacement. TensorRT-LLM has
credible INT4/INT8 support, but exporting the complete MoT, vision, VAE, and
policy-server path would be a much larger project than replacing a kernel. It
was deferred rather than partially integrated.

### 3.2 Four deployable precision strategies

| Strategy        | Precision map                               | Intended role                   |
| --------------- | ------------------------------------------- | ------------------------------- |
| `full_w8`       | All target Linear modules use W8A16         | Calibration-free, quality-first |
| `full_w4`       | All target Linear modules use W4A16         | Minimum model memory            |
| `attention_w8`  | Both attention branches use W8; MLPs use W4 | RoboCasa memory/quality balance |
| `gen_branch_w8` | Generation branch uses W8; the rest uses W4 | Protect generation precision    |

Two independent 50-episode RoboCasa CloseFridge runs produced 96% for Full W8,
92% for Full W4, 96% for Attention W8, and 94% for Gen-branch W8. Their RTX
4090 peak allocations were 19.20, 12.47, 13.93, and 15.84GB. All met the 24GB
goal, but lower bit width did not imply higher task success.

Nano RoboLab made this clearer: at g3/s4, Full W4 reached 26/50, compared with
43/50 for Full W8. Full W4 remains a selectable low-memory artifact, not the
general default.

### 3.3 What calibration does

W4 and mixed strategies use 128 training-split episodes, one frame per
episode, to collect per-input-channel activation maxima for each Linear layer.
An AWQ-style rescaling then balances input and weight channels before weight
quantization without changing the floating-point function being approximated.

Key results were:

- RoboCasa Full W4 eval32 L1 mean fell from 0.0538 to 0.0177, a 67% reduction.
- Most mixed candidates reduced L1 mean by 20%-68%.
- Calibration did not reliably eliminate rare action spikes; Full W4 Linf max
  remained above 1.
- Some precision maps improved average error while worsening Linf.

Calibration was retained, but the conclusion is not that calibrated W4 is
automatically safe. It improves the error distribution; closed-loop rollout
still decides whether a configuration is usable. Full W8 requires no
calibration.

### 3.4 Why the early sensitivity winner was demoted

Early open-loop tests showed that attention was more precision-sensitive than
MLPs. A fine-grained `mixed_best` candidate kept the generation branch and
understanding attention at W8, while quantizing only the understanding MLP to
W4. It had one of the best replay L1 results.

In a matched 50-episode RoboCasa rollout, however, `mixed_best` reached 78%
while Attention W8 reached 90%. Replay ranking did not predict closed-loop
ranking. The product therefore retained four simpler, more thoroughly tested
strategies.

### 3.5 Streaming export and self-contained bundles

Export does not place the complete BF16 checkpoint on the GPU. It reads one
layer, quantizes and packs it, releases the source weight, and moves on. The
RoboCasa release smoke used only about 0.7-1.0GB peak GPU memory during
stream-packing.

Each bundle contains packed weights, residual BF16 tensors, runtime config,
tokenizer, VAE, precision map, source revision, sizes, and SHA256 hashes. A
deployment no longer reads the source DCP/BF16 checkpoint or calibration data.
This fixed the early packaging flaw where a quantized model still depended on
its source checkpoint.

## 4. The Largest First Speedup: Fewer Denoiser Forwards

The first-order denoiser cost is approximately:

```text
num_steps * (2 if guidance > 1 else 1)
```

Guidance 3 runs conditional and unconditional branches, commonly called CFG
or classifier-free guidance. Guidance 1 removes the second branch. Reducing
denoise steps from four to two keeps CFG but evaluates fewer solver points.

### 4.1 Nano

| Guidance / steps | Request p50 |       Banana SR |
| ---------------- | ----------: | --------------: |
| g3/s4            |      4110ms |     43/50 = 86% |
| g3/s2            |  **2403ms** | **45/50 = 90%** |
| g1/s4            |      2431ms |     32/50 = 64% |
| g1/s2            |      1565ms |     40/50 = 80% |

g3/s2 reduced request latency by 41.5% without an observed quality loss.
Guidance 1 was faster per request, but produced more failures. Failed episodes
often ran to the horizon, making complete evaluation slower, so guidance 1 was
rejected as a deployment default.

### 4.2 Edge

Edge BF16 itself improved from 21/50 at four steps to 34/50 at two steps while
request latency fell from 1042ms to 582ms. This is a checkpoint/sampler
interaction, not a quality benefit caused by quantization. UniPC is a numerical
solver: more steps mean finer integration, not guaranteed policy refinement.
If the learned vector field and inference schedule are imperfectly matched,
extra solver points can move an action away from a better trajectory.

RoboLab therefore uses g3/s2. RoboCasa results were noisier, so g3/s4 remains
its conservative default.

## 5. Why W4/W8 Saves Memory but May Not Run Faster

### 5.1 Real shapes matter more than generic LLM benchmarks

Real-shape RTX 4090 GEMM measurements found:

- Nano's call-weighted W4/W8 kernels were about 1.06x/1.04x faster than BF16.
- Edge's call-weighted W4/W8 kernels were 4%/6% slower than BF16.
- Large generation MLP shapes amortized unpacking and launch overhead.
- Small condition projections often favored BF16.

Nsight Compute also showed that representative Marlin kernels were not simply
maxing out Tensor Cores. They used 255 registers per thread and about 101KB of
shared memory per block, with only about 8%-16% occupancy. Most scheduler
cycles had no eligible warp. Tile size, register/shared-memory pressure, and
latency hiding were the practical limits.

### 5.2 Other W4/W8 backends considered

| Backend                  | Main result                                                                                                          | Decision                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| torchao INT8 weight-only | W8-like error, but policy requests reached tens of seconds on H100; representative shapes were 3-4x slower than BF16 | Accuracy reference only |
| vLLM AllSpark W8A16      | Close to BF16 on weighted 4090 shapes, with no clear global win over Marlin                                          | Not integrated          |
| Machete                  | Requires SM90                                                                                                        | Not applicable to 4090  |
| ExLlamaV2/V3             | Surveyed as consumer-GPU evidence, but no directly compatible module/layout path was established                     | Reference only          |
| TensorRT-LLM/ModelOpt    | Credible potential, but requires substantial whole-model export work                                                 | Deferred                |
| PyTorch INT8 `_int_mm`   | Did not support the token-10 shape and dynamic W8A8 was slow                                                         | Rejected                |

The project deliberately avoided adding backends for their own sake. A path
had to show a clear advantage on Cosmos shapes before entering the full model.

## 6. FP8 W8A8: The First Clear Large-GEMM Speedup

RTX 4090 has FP8 Tensor Cores, but not Blackwell's native FP4 Tensor Cores.
FP8 was therefore the useful activation-quantization target for this hardware.

### 6.1 Measure the kernel before integrating the model

Early RoboCasa top-shape benchmarks found:

- FP8 GEMM kernel-only was about 39% faster than Marlin W4.
- Adding eager dynamic activation quantization made the full operator slower
  than Marlin.
- Static activation scales saved 30.7% on the large-MLP subset, but the rough
  end-to-end ceiling was only 14.5%.

The early PyTorch `_scaled_mm` static-scale route was not productized. It would
have added calibration and saturation handling without enough demonstrated
end-to-end return.

### 6.2 Final implementation

The later path reused mature vLLM CUTLASS FP8 operations:

- weights use static per-output-channel E4M3 scales;
- activations use dynamic per-token E4M3 scales;
- outputs return to BF16;
- `gen_branch_w8a8` applies FP8 only to the generation branch and keeps the
  rest in calibrated W4; and
- `full_w8a8` applies FP8 to every target Linear module.

Nano GenW8A8 reduced request p50 from 1674.5ms for GenW8A16 to 1340.0ms, a
20.0% reduction. Full W8A8 reached 1329.9ms, only 10.1ms faster while using
3.57GB more reserved memory. Mixed GenW8A8 was the better Nano starting point.

Edge GenW8A8 reduced request p50 from 573.5ms to 502.8ms, a 12.3% reduction.
Full W8A8 was slower at 510.6ms and had larger replay error, so Edge also chose
GenW8A8.

In 50-episode rollout, Nano GenW8A8 reached 47/50 and Edge reached 39/50. Both
passed the first closed-loop gate.

### 6.3 Equalization and W4A8

Input-channel equalization from DROID train128 statistics at `alpha=0.5` did
not jointly improve mean error, tail error, and latency. The simpler dynamic
FP8 path was retained.

W4A8 was not integrated. RTX 4090 has no native FP4 Tensor Core path, and the
current stack lacked a mature fused kernel that could dequantize INT4 weights
and feed an FP8 GEMM with a clear advantage. The operator study did not justify
hand-writing and maintaining such a backend.

## 7. Compiling the Graph: From Failure to a Useful Path

### 7.1 Why the first attempt failed

The first direct `torch.compile` attempt on Marlin models failed in fullgraph
mode because the Marlin custom op had no fake-tensor/meta implementation.
Allowing graph breaks made it run, but steady generate latency moved from about
1031ms to 1053ms and the first request took about 20 seconds. That path was
rejected.

### 7.2 Why the later path worked

The FP8 runtime exposed quantized Linear work through traceable custom-op
boundaries and compiled complete MoT language blocks. Inductor still cannot
rewrite the inside of a CUTLASS FP8 GEMM, but it can optimize surrounding
normalization, residual, and pointwise work while removing Python dispatch.

| Model | Eager p50 | Compiled p50 | Reduction |
| ----- | --------: | -----------: | --------: |
| Edge  |   502.8ms |      441.4ms |     12.2% |
| Nano  |  1322.7ms |     1130.9ms |     14.5% |

Cold compilation took about 12.5 seconds for Edge and 14.9 seconds for Nano,
so production must issue a warmup request.

### 7.3 Compile and graph variants that were rejected

| Candidate                       | Result                                                                 | Decision |
| ------------------------------- | ---------------------------------------------------------------------- | -------- |
| Static-shape compile            | New prompt lengths caused 6-7 second recompiles                        | Reject   |
| Language plus all VFM heads     | Same steady latency, about twice the cold compile, larger action error | Reject   |
| Inductor CUDA Graphs            | Capture overhead with no steady gain                                   | Disabled |
| CUDA Graph with repeated prompt | Edge 433.6ms, no gain over compile alone                               | Reject   |
| MLP-only compile                | 1%-2% gain and worse parity                                            | Reject   |
| Attention-only compile          | About 6% gain and worse parity                                         | Reject   |
| `force_same_precision`          | Did not improve parity                                                 | Reject   |

CUDA Graph support exists. It was not selected because variable prompt lengths,
CPU preprocessing, tokenization, sampler state, and action transfer make
whole-request capture a poor fit for this policy server.

## 8. Reusing FP8 Activation Quantization

Generation Q/K/V projections read the same activation. Nano gated-MLP gate/up
projections do as well. The original path quantized that activation separately
for every Linear module.

`FP8_PROJECTION_FUSION=shared` quantizes it once and passes the result to
multiple GEMMs. It does not concatenate weights or change GEMM tiling, so all
32 replay action tensors were bit-identical.

- Edge request p50: 436.8 to 428.6ms, a 1.9% reduction.
- Nano request p50: 1144.3 to 1119.9ms, a 2.1% reduction.

Another prototype concatenated projection weights along the output dimension.
It won on some isolated operators, but changed CUTLASS accumulation tiling,
raised Nano reserved memory to 19GB, and did not consistently beat shared
quantization end to end. It was not exposed to deployment users.

## 9. Attention and the Condition K/V Cache

### 9.1 Profile before optimizing

In the early RoboCasa profile, attention accounted for only about 4.4% of GPU
kernel time. Optimizing attention alone could not produce a large overall win.
After FP8 and compilation reduced GEMM and dispatch time, long generation
attention became a worthwhile secondary target.

The real generation shape has Q length about 3093 and KV length about 3175.
Understanding causal attention has only about 82 tokens. One backend is not
best for both.

### 9.2 Shape-aware SageAttention

| Shape          | FlashAttention2 | Sage FP8-PV | Result    |
| -------------- | --------------: | ----------: | --------- |
| Edge long Gen  |         0.706ms |     0.324ms | Sage wins |
| Nano long Gen  |         1.220ms |     0.551ms | Sage wins |
| Edge short Und |         0.084ms |     0.245ms | FA2 wins  |
| Nano short Und |         0.083ms |     0.231ms | FA2 wins  |

The policy therefore selects Sage only on SM89 for dense, non-causal attention
with Q length at least 512. All other cases remain on FlashAttention2.

The first Edge FP8-PV Sage rollout reached 34/50, compared with a 37/50 FA2
baseline at that stage. Its point estimate was unfavorable, so it was not
promoted. A higher-fidelity INT8-QK + FP16-PV/FP32 candidate was then tested:

- the operator took 0.463ms, still faster than FA2 at 0.706ms;
- replay L1 and Linf p95 were about 13% and 14% lower than FP8-PV;
- Edge request p50 fell from 349.7ms to 331.4ms; and
- paired rollout reached 40/50 versus 39/50 for matched FA2, with 6/5 paired
  wins/losses.

Edge therefore uses FP16-PV Sage. A faster FP16+FP32 variant encountered an
unsupported WARPQ=16 shape in the complete graph and was rejected. FlashInfer
BF16 took 0.725ms and did not beat FA2, so it was not integrated.

### 9.3 Condition K/V cache

Understanding tokens do not change across denoising steps within one request.
Nano stores separate conditional and unconditional understanding K/V after
the first step, then recomputes only the generation path on the second step.
The cache lasts one request and never crosses robot observations.

Nano Sage plus cache reached 987.2ms request p50, or 958.5ms after the sparse
transform, with 49/50 Banana success. The cache was numerically correct on
Edge but provided no repeatable latency reduction, so it is enabled only for
Nano.

### 9.4 Why general DiT block caching was deferred

PAB, TeaCache, SmoothCache, and robot-policy-specific BAC were surveyed. They
reuse attention or block outputs across diffusion timesteps and may offer
larger gains, but the current sampler has only two steps, leaving little reuse
headroom. They also change actions. Cosmos Lite retained the exact condition
K/V cache and did not implement approximate block caching in this phase.

## 10. An RTX 4090-Specific Triton FP8 GEMM

The vLLM CUTLASS FP8 kernel used large tiles on dominant Edge shapes. Nsight
Compute showed high register/shared-memory pressure and few eligible warps.
Cosmos Lite therefore added a Triton SM89 kernel for four validated Edge
generation shapes, with automatic CUTLASS fallback everywhere else.

| Edge shape M x K x N | CUTLASS |  Triton | Reduction |
| -------------------- | ------: | ------: | --------: |
| 3093 x 2048 x 9216   | 0.569ms | 0.407ms |     28.6% |
| 3093 x 2048 x 2048   | 0.165ms | 0.111ms |     32.8% |
| 3093 x 2048 x 1024   | 0.245ms | 0.075ms |     69.6% |
| 3093 x 9216 x 2048   | 0.629ms | 0.532ms |     15.4% |

Edge request p50 improved by about 8.4%. Rollout reached 39/50, compared with
37/50 for the matched FA2/CUTLASS control, and passed the gate.

Nano kernels also showed 11%-23% operator reductions and an 8.7% request-p50
reduction, but rollout moved from 49/50 to 47/50. With a 98% control and no
clear quality margin, Nano shapes stay outside the Triton allowlist and remain
on CUTLASS.

## 11. Request Data Path: The Last 30ms Was on the CPU

Fine-grained profiling split a request into sample construction, batch
construction, CUDA generation, action transfer, and postprocessing. The Edge
legacy path spent 36.64ms building a sample even though only the first frame
was observed; the following 32 frames were known zero placeholders.

The new path resizes the observed frame, then allocates zeros directly at the
target size:

| Edge data path   | Build sample | CUDA generate | WebSocket request |
| ---------------- | -----------: | ------------: | ----------------: |
| Resize 33 frames |      36.64ms |      338.35ms |          382.84ms |
| Resize 1 frame   |   **2.31ms** |      339.87ms |      **349.70ms** |

Replay32 actions were exactly identical element by element. Nano similarly
moved from 987.2ms to 958.5ms. The optimization is enabled by default and can
be rolled back with `SPARSE_VIDEO_TRANSFORM=0`; custom transform types keep the
legacy path.

## 12. Other Candidates Rejected or Demoted

| Method                      | Main result                                                                     | Decision                       |
| --------------------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| Guidance 1                  | Faster requests, but substantially more Nano/Edge closed-loop failures          | Do not recommend               |
| Action chunk 16/8           | Lower work proxy but worse measured latency; also changes control semantics     | Reject                         |
| Camera pre-resize to 192    | Model still consumed its 256 bucket; no material gain                           | Reject                         |
| Wrist-only or fewer views   | Changed the input distribution without enough speedup                           | Reject                         |
| CFG batching                | Edge 517.4 to 555.5ms; FLOPs were unchanged and attention became less favorable | Reject                         |
| Edge condition cache        | Correct but no repeatable gain                                                  | Keep off                       |
| Full W8A8                   | Supported; only 10.1ms faster than Nano GenW8A8 while using 3.57GB more         | Keep optional, not default     |
| Nano Triton FP8             | Faster kernels and replay, but SR point estimate moved from 98% to 94%          | Disable                        |
| Edge Sage FP8-PV            | Fastest attention, but early SR point estimate was unfavorable                  | Use FP16-PV instead            |
| FlashInfer                  | Long-attention operator did not beat FA2                                        | Do not integrate               |
| General W4/W8 shape routing | Small-shape opportunity existed, but large Gen shapes dominated the request     | Avoid extra routing complexity |
| Multi-environment batching  | Could improve simulator throughput, but a real robot runs one rollout           | Low priority, not implemented  |

## 13. Final Tradeoff and Recommended Profiles

The release separates the model artifact from runtime sampling. Nano recommends
GenW8A8 for the best tradeoff and W8A16 as the calibration-free 24GB fallback.
Edge recommends GenW8A8 for latency and NVIDIA BF16 as the official
quality-first path; Edge W8A16 remains an optional compressed artifact.
Guidance, denoise steps, and shift can be changed without rebuilding any of
these checkpoints. Benchmark tables state their sampler explicitly.

### 13.1 Current Fast Model And Runtime Profiles

Edge:

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
FP8_PROJECTION_FUSION=shared \
FP8_GEMM_BACKEND=triton_sm89 \
SAGE_ATTENTION=1 \
SAGE_PV=fp16_fp32 \
examples/robolab_quant/pipeline.sh serve
```

Nano:

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
FP8_PROJECTION_FUSION=shared \
SAGE_ATTENTION=1 \
CONDITION_KV_CACHE=1 \
examples/robolab_quant/pipeline.sh serve
```

Both require one warmup request that is excluded from steady-state latency.
The sparse transform is enabled by default. Each optimization has an
independent rollback boundary and does not require repacking weights.

### 13.2 Why this is a reasonable point to stabilize

Edge CUDA generation is now about 330ms and Nano about 948ms. Sample building,
action transfer, and postprocessing take only a few milliseconds. Most
remaining time is real model computation. More ordinary Python cleanup or
small-operator replacement is unlikely to produce another large gain.

Meaningful further reductions would require higher-investment work: deeper FP8
GEMM plus norm/residual fusion, token pruning, early exit, model distillation,
or quality-gated approximate block caching. These are no longer low-risk,
training-free improvements and should be evaluated as separate kernel or
research projects.

## 14. Reproducibility Records and References

Detailed records in this repository:

- [Primary RoboLab benchmark](benchmarks/robolab.md)
- [RoboLab ablations](benchmarks/robolab_ablations.md)
- [RoboCasa365 benchmark](../examples/robocasa365_quant/BENCHMARKS.md)
- [FP8 W8A8 experiment](experiments/fp8_w8a8.md)
- [Graph optimization experiment](experiments/graph_optimization.md)
- [RTX 4090 SM89 optimization](experiments/rtx4090_sm89.md)

External technical references:

- NVIDIA Cosmos Framework: <https://github.com/NVIDIA/Cosmos-Framework>
- NVIDIA Cosmos3 Nano Policy DROID: <https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID>
- NVIDIA Cosmos3 Edge Policy DROID: <https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>
- AWQ: <https://arxiv.org/abs/2306.00978>
- MARLIN: <https://arxiv.org/abs/2408.11743>
- UniPC: <https://arxiv.org/abs/2302.04867>
- Classifier-free guidance: <https://arxiv.org/abs/2207.12598>
- vLLM FP8 quantization: <https://docs.vllm.ai/en/latest/features/quantization/fp8/>
- PyTorch compile for diffusion models: <https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/>
- Triton: <https://github.com/triton-lang/triton>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- FlashInfer: <https://github.com/flashinfer-ai/flashinfer>
- PAB: <https://arxiv.org/abs/2408.12588>
- TeaCache: <https://liewfeng.github.io/TeaCache/>
- SmoothCache: <https://openaccess.thecvf.com/content/CVPR2025W/eLVM/papers/Liu_SmoothCache_A_Universal_Inference_Acceleration_Technique_for_Diffusion_Transformers_CVPRW_2025_paper.pdf>
- Block-wise Adaptive Caching for diffusion policies: <https://block-wise-adaptive-caching.github.io/>
