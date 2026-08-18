<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos3 Edge RoboLab Benchmarks

This record covers Cosmos3 Edge Policy DROID on RTX 4090. It uses the same
RoboLab observation, action, calibration, replay, and paired-rollout protocol
as the [Cosmos3 Nano benchmark](NANO_BENCHMARKS.md), while keeping model-family
results separate.

## v0.3 Release Profiles

The public Edge release retains W8A16 as the calibration-free fallback and
GenW8A8 as the recommended RTX 4090 profile. In GenW8A8, only the W4A16
remainder uses DROID training calibration; the FP8 generation branch uses
dynamic per-token activation scales.

| Profile     | Preset                                | Peak reserved | Request p50 |       Banana SR |
| ----------- | ------------------------------------- | ------------: | ----------: | --------------: |
| W8A16       | `configs/edge_w8.yaml`                |        8.71GB |     576.0ms |     36/50 = 72% |
| **GenW8A8** | `configs/edge_genw8a8_fast_4090.yaml` |        8.79GB | **331.4ms** | **40/50 = 80%** |

### 128-Request Stability Gate

Each profile replayed the same 32 real DROID training requests four times in
one server process. The first lazy-initialization request is excluded. This is
a shared-host longevity gate rather than a replacement for the idle-GPU
headline latency above.

| Profile | Requests | Request p50/p95 | Repeat p50, 0/1/2/3 | Last/peak reserved | Result |
| ------- | -------: | --------------: | ------------------: | -----------------: | ------ |
| W8A16   |      128 |       595/611ms |   594/595/594/598ms |        6.35/8.71GB | Pass   |
| GenW8A8 |      128 |       408/420ms |   401/407/411/411ms |        5.70/8.79GB | Pass   |

Neither run crashed, exhausted memory, returned non-finite actions, selected a
fallback backend, or showed increasing CUDA reserved memory. These captures
have no matched reference responses, so paired rollout remains the quality
gate.

## Benchmark Summary

### Quantization Comparison

This is the primary same-protocol model comparison: one RTX 4090, batch size
one, guidance 3.0, two UniPC denoise steps, seed 0, and 50 paired
`BananaInBowlTask` initial states. Request latency includes policy-server
preprocessing, generation, and response handling; it excludes IsaacSim.

| Quantized model                                                                             | Peak VRAM (GB) | Request median (ms) | Banana SR |
| ------------------------------------------------------------------------------------------- | -------------: | ------------------: | --------: |
| [W8A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W8A16)               |       **8.71** |                 576 |       72% |
| [W4A16](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16)               |           8.87 |             **563** |       74% |
| [W4A16-AttnW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-AttnW8) |           8.74 |                 571 |       58% |
| [W4A16-GenW8](https://huggingface.co/XXXXyu/Cosmos3-Edge-Policy-DROID-Marlin-W4A16-GenW8)   |           8.79 |                 570 |   **80%** |

### Sampler Comparison

This table fixes the model to BF16 and changes only denoise steps. The same 50
initial states were used for both rows.

| Denoise steps | Guidance | Request median (ms) | Banana SR |
| ------------: | -------: | ------------------: | --------: |
|             4 |      3.0 |               1,042 |       42% |
|         **2** |  **3.0** |             **582** |   **68%** |

### Rollout Comparison

[![Watch the RoboLab Edge quantization comparison](../../docs/assets/robolab_edge_quant_comparison_poster.png)](../../docs/assets/robolab_edge_quant_comparison.mp4)

The displayed episodes are a documented cherry-picked subset; all success
rates above use the complete paired 50-rollout evaluation. W4A16 shows its two
fastest successes and one failure; the other settings show their three fastest
successes. All panels use guidance 3 / two denoise steps.

## Status

The BF16 and quantized serving paths, self-contained bundle validation,
32-request replay matrix, and complete paired 50-episode Banana matrix are
validated.

## Measurement Rules

- Base policy: `nvidia/Cosmos3-Edge-Policy-DROID` at revision
  `3ea407af3e156c0af3b4bb6edd85842cc9a58777`.
- GPU: RTX 4090 24GB; batch size one; deterministic model seed 0.
- Task protocol: RoboLab `BananaInBowlTask`, process seed 0, one environment,
  50 paired episodes, and a 750-step horizon.
- Client observation: one 640x540 RGB composition from three RoboLab views.
- Model spatial bucket: 736x544 W x H.
- Policy output: one 32x8 DROID action chunk, matching the source
  `checkpoint.json` policy metadata and the RoboLab client contract. The
  NVIDIA model card's PBR section currently describes a different 16-step
  request, so it is not the protocol used for these results.
- Retained sampler: guidance 3.0, two UniPC denoise steps, shift 5.0. The
  four-step results below are preserved as the original comparison control.
- Quantization: packed vLLM Marlin weight-only W4A16/W8A16; activations remain
  BF16.

Request latency includes server preprocessing, generation, and response
handling, but excludes RoboLab and IsaacSim. The first request is reported
separately because Cosmos3 Edge loads its vision encoder lazily. Tables use
steady-state requests after that warm-up.

## Artifact Matrix

All four bundles passed full SHA256 verification and packed-tensor loading.
They include the Edge processor, vision encoder, Wan VAE, residual weights,
packed weights, immutable source provenance, and a portable runtime config.

| Strategy        | W4/W8 modules |  Bundle bytes | Post-load allocated |
| --------------- | ------------: | ------------: | ------------------: |
| `full_w8`       |         0/336 | 7,747,897,282 |              5.34GB |
| `full_w4`       |         336/0 | 6,382,980,938 |              3.98GB |
| `attention_w8`  |       112/224 | 6,723,995,930 |              4.33GB |
| `gen_branch_w8` |       168/168 | 7,065,438,438 |              4.67GB |

The 4B Edge BF16 policy already fits on a 24GB RTX 4090. Quantization reduces
model-resident memory and artifact size, but its batch-one latency benefit is
small; closed-loop quality is therefore the primary strategy gate.

## Training Calibration

W4 and mixed bundles reuse the general DROID training calibration protocol
from the Nano release:

```text
dataset: nvidia/Cosmos3-DROID
revision: 5c11a20accb11497270a5247a7f1e66ad04c956c
split: train/success
samples: 128 frames from 128 distinct successful episodes
selection: one deterministic central-80% frame per episode
views: wrist + left/right exterior
method: per-Linear input-channel amax, W4 alpha 0.5
```

These samples are not Banana-specific and do not update policy parameters.
Full W8 is calibration-free.

## BF16 Serving Baseline

The official Edge checkpoint completed real RoboLab-format requests on an RTX
4090 with both retained sampler settings.

| Metric                  |        Result |
| ----------------------- | ------------: |
| Post-load allocated     |        8.16GB |
| Process peak reserved   |        9.20GB |
| Steady request p50/p95  | 1,042/1,047ms |
| Steady generate p50/p95 | 1,003/1,007ms |

The first request took about 2.0 seconds because it included lazy vision
encoder initialization. The policy returned finite 32x8 actions.

The same 32-request replay protocol gives the sampler comparison below. The
two-step first request took 1,506ms; it is excluded from the steady-state row.

| Guidance | Steps | Request p50/p95 | Generate p50/p95 |
| -------: | ----: | --------------: | ---------------: |
|      3.0 |     4 |   1,042/1,047ms |    1,003/1,007ms |
|      3.0 |     2 |       582/584ms |        536/538ms |

## Quantization Replay

All rows use the same 32 captured inputs, guidance 3.0, and four denoise steps.
Action error is measured against the matched BF16 response for each request.

| Strategy        | Request p50/p95 | Generate p50/p95 | Last alloc/reserved | Peak reserved | L1 mean | Linf p95 |
| --------------- | --------------: | ---------------: | ------------------: | ------------: | ------: | -------: |
| BF16            |   1,042/1,047ms |    1,003/1,007ms |         8.33/9.20GB |        9.20GB |       0 |        0 |
| `full_w8`       |   1,017/1,032ms |        977/984ms |         5.51/6.35GB |        8.71GB | 0.01864 |   0.4093 |
| `full_w4`       |   1,040/1,047ms |      997/1,001ms |         4.15/5.02GB |        8.87GB | 0.03214 |   0.5400 |
| `attention_w8`  |   1,043/1,051ms |    1,002/1,006ms |         4.50/5.31GB |        8.74GB | 0.03133 |   0.3397 |
| `gen_branch_w8` |     988/1,009ms |        946/955ms |         4.84/5.70GB |        8.79GB | 0.02352 |   0.4404 |

`L1 mean` is the mean absolute action difference over every request, action
step, and dimension. `Linf p95` first takes the largest absolute element error
per request, then reports the 95th percentile across requests. These metrics
are diagnostics, not success-rate proxies.

## Sampler Replay

The sampler latency matrix fixes quantization to full W8 unless noted. Altered
samplers are not compared to saved g3/s4 actions because that would conflate a
deliberate sampling change with quantization error.

| Strategy        | Guidance | Steps | Request p50/p95 | Generate p50/p95 |
| --------------- | -------: | ----: | --------------: | ---------------: |
| BF16            |      3.0 |     2 |       582/584ms |        536/538ms |
| `full_w8`       |      3.0 |     4 |   1,017/1,032ms |        977/984ms |
| `full_w8`       |      3.0 |     2 |       576/603ms |        529/553ms |
| `full_w8`       |      1.0 |     4 |       576/615ms |        531/549ms |
| `full_w8`       |      1.0 |     2 |   **367/399ms** |    **312/334ms** |
| `full_w4`       |      3.0 |     2 |   **563/565ms** |    **523/526ms** |
| `attention_w8`  |      3.0 |     2 |       571/613ms |        527/551ms |
| `gen_branch_w8` |      3.0 |     2 |       570/610ms |        522/540ms |

Two denoise steps approximately halve generation latency while retaining
classifier-free guidance. Guidance 1 removes the unconditional CFG pass. Both
are behavioral changes and require the paired closed-loop gate below.

## RTX 4090 GEMM Kernel Benchmark

The runtime shape recorder captured one real DROID request at guidance 3 / two
steps for each model family. `M` is flattened input tokens, `K` is input
features, and `N` is output features. Calls are counts per request after
grouping identical shapes across layers. Each number below is the mean of two
independent runs with 20 warm-up and 100 timed iterations per shape on one RTX
4090 GPU. Speedup is BF16 latency divided by packed-kernel latency; values
below 1.0 mean quantization is slower.

These are isolated Linear kernel measurements. They exclude quantization and
packing, model wrappers, attention, VAE encode, sampling updates, and request
handling.

### Edge 4B

|    M |    K |    N | Calls | BF16 (ms) | W4 (ms) | W4 speedup | W8 (ms) | W8 speedup |
| ---: | ---: | ---: | ----: | --------: | ------: | ---------: | ------: | ---------: |
| 3093 | 2048 | 1024 |   224 |    0.1009 |  0.1054 |      0.96x |  0.1086 |      0.93x |
| 3093 | 2048 | 2048 |   224 |    0.1867 |  0.1749 |      1.07x |  0.1714 |      1.09x |
| 3093 | 9216 | 2048 |   112 |    0.7268 |  0.7015 |      1.04x |  0.7179 |      1.01x |
| 3093 | 2048 | 9216 |   112 |    0.7008 |  0.7482 |      0.94x |  0.7664 |      0.91x |
|   19 | 2048 | 1024 |   112 |    0.0181 |  0.0313 |      0.58x |  0.0307 |      0.59x |
|  107 | 2048 | 1024 |   112 |    0.0171 |  0.0366 |      0.47x |  0.0352 |      0.49x |
|   19 | 2048 | 2048 |   112 |    0.0174 |  0.0319 |      0.55x |  0.0303 |      0.58x |
|  107 | 2048 | 2048 |   112 |    0.0179 |  0.0365 |      0.49x |  0.0346 |      0.52x |
|   19 | 9216 | 2048 |    56 |    0.0188 |  0.0269 |      0.70x |  0.0259 |      0.73x |
|  107 | 9216 | 2048 |    56 |    0.0333 |  0.0449 |      0.74x |  0.0460 |      0.72x |
|   19 | 2048 | 9216 |    56 |    0.0197 |  0.0266 |      0.74x |  0.0258 |      0.76x |
|  107 | 2048 | 9216 |    56 |    0.0372 |  0.0509 |      0.73x |  0.0524 |      0.71x |

Call-count-weighted latency is 0.1773ms for BF16, 0.1851ms for W4, and
0.1875ms for W8. Thus packed W4 and W8 are 4.0% and 6.0% slower than BF16 on
this Edge shape mix. The large generation projections are close to parity,
but Marlin launch and dequantization overhead dominate the frequent small-M
condition projections.

### Nano 16B

|    M |     K |     N | Calls | BF16 (ms) | W4 (ms) | W4 speedup | W8 (ms) | W8 speedup |
| ---: | ----: | ----: | ----: | --------: | ------: | ---------: | ------: | ---------: |
| 3093 |  4096 | 12288 |   288 |    1.9234 |  1.8602 |      1.03x |  1.9058 |      1.01x |
| 3093 |  4096 |  1024 |   288 |    0.1753 |  0.1721 |      1.02x |  0.1755 |      1.00x |
| 3093 |  4096 |  4096 |   288 |    0.6545 |  0.6278 |      1.04x |  0.6406 |      1.02x |
|   10 |  4096 | 12288 |   144 |    0.1165 |  0.0258 |      4.51x |  0.0253 |      4.60x |
|   95 |  4096 | 12288 |   144 |    0.1232 |  0.0750 |      1.64x |  0.0773 |      1.59x |
| 3093 | 12288 |  4096 |   144 |    1.9146 |  1.8224 |      1.05x |  1.8617 |      1.03x |
|   10 |  4096 |  1024 |   144 |    0.0168 |  0.0311 |      0.54x |  0.0298 |      0.56x |
|   95 |  4096 |  1024 |   144 |    0.0162 |  0.0363 |      0.45x |  0.0343 |      0.47x |
|   10 |  4096 |  4096 |   144 |    0.0185 |  0.0283 |      0.65x |  0.0269 |      0.69x |
|   95 |  4096 |  4096 |   144 |    0.0302 |  0.0336 |      0.90x |  0.0340 |      0.89x |
|   10 | 12288 |  4096 |    72 |    0.1230 |  0.0261 |      4.71x |  0.0254 |      4.85x |
|   95 | 12288 |  4096 |    72 |    0.1360 |  0.0766 |      1.78x |  0.0782 |      1.74x |

Call-count-weighted latency is 0.5623ms for BF16, 0.5303ms for W4, and
0.5418ms for W8. Packed W4 and W8 are therefore 1.06x and 1.04x faster on the
Nano mix. Nano's wider 4096/12288 projections amortize Marlin overhead, while
the narrow K/V projections still favor BF16.

## Paired Banana Rollouts

Each row uses 50 episodes and the same deterministic initial-state order. All
completed HDF5 initial-state hashes match by run. Confidence intervals are
Wilson 95% intervals. A paired win means the candidate succeeded when the
reference failed; a paired loss means the reverse. McNemar p-values are exact,
two-sided, and not adjusted for multiple comparisons.

### Quantization At Guidance 3 / Four Steps

| Strategy        |     Success (Wilson 95% CI) | Successful step median | Paired wins/losses vs BF16 | McNemar p |
| --------------- | --------------------------: | ---------------------: | -------------------------: | --------: |
| BF16            | 21/50 = 0.42 [0.294, 0.558] |                    316 |                  reference |         - |
| `full_w8`       | 29/50 = 0.58 [0.442, 0.706] |                    350 |                      18/10 |     0.185 |
| `full_w4`       | 20/50 = 0.40 [0.276, 0.538] |                  337.5 |                      15/16 |     1.000 |
| `attention_w8`  | 25/50 = 0.50 [0.366, 0.634] |                    286 |                      14/10 |     0.541 |
| `gen_branch_w8` | 22/50 = 0.44 [0.312, 0.577] |                  361.5 |                      11/10 |     1.000 |

No quantized strategy is distinguishable from BF16 with 50 episodes under the
four-step protocol. The apparent full-W8 improvement is not statistically
significant.

### Full-W8 Sampler Gate

| Guidance | Steps |     Success (Wilson 95% CI) | Successful step median | Paired wins/losses vs g3/s4 | McNemar p | Request p50/p95 |
| -------: | ----: | --------------------------: | ---------------------: | --------------------------: | --------: | --------------: |
|      3.0 |     4 | 29/50 = 0.58 [0.442, 0.706] |                    350 |                   reference |         - |   1,017/1,032ms |
|      3.0 |     2 | 36/50 = 0.72 [0.583, 0.825] |                  196.5 |                        15/8 |     0.210 |       576/603ms |
|      1.0 |     4 |  4/50 = 0.08 [0.032, 0.188] |                  541.5 |                        1/26 |  4.17e-07 |       576/615ms |
|      1.0 |     2 | 11/50 = 0.22 [0.128, 0.352] |                    264 |                        6/24 |   0.00143 |       367/399ms |

Guidance 3 / two steps reduces request latency by 1.77x without an observed
quality loss. Removing classifier-free guidance causes a large, statistically
significant failure increase and is rejected despite lower request latency.

### BF16 Denoise-Step Gate

The official BF16 checkpoint was rerun with the same guidance 3 / two-step
sampler and the same 50 initial states.

| Precision | Steps |     Success (Wilson 95% CI) | Successful step median | Paired wins/losses vs BF16 s4 | McNemar p | Request p50/p95 |
| --------- | ----: | --------------------------: | ---------------------: | ----------------------------: | --------: | --------------: |
| BF16      |     4 | 21/50 = 0.42 [0.294, 0.558] |                    316 |                     reference |         - |   1,042/1,047ms |
| BF16      |     2 | 34/50 = 0.68 [0.542, 0.792] |                    208 |                          19/6 |    0.0146 |       582/584ms |
| `full_w8` |     2 | 36/50 = 0.72 [0.583, 0.825] |                  196.5 |                          20/5 |   0.00408 |       576/603ms |

BF16 itself gains 26 percentage points and reaches tasks sooner at two steps.
At that same sampler setting, full W8 and BF16 are not distinguishable: W8 has
12 paired wins and 10 losses versus BF16 (`p=0.832`). The two-step gain is
therefore primarily a checkpoint/sampler effect, not a benefit introduced by
weight quantization.

### Quantization At Guidance 3 / Two Steps

| Strategy        |     Success (Wilson 95% CI) | Successful step median | Paired wins/losses vs full-W8 | McNemar p | Request p50/p95 | Post-load allocated |
| --------------- | --------------------------: | ---------------------: | ----------------------------: | --------: | --------------: | ------------------: |
| `full_w8`       | 36/50 = 0.72 [0.583, 0.825] |                  196.5 |                     reference |         - |       576/603ms |              5.34GB |
| `full_w4`       | 37/50 = 0.74 [0.604, 0.841] |                    180 |                         11/10 |     1.000 |       563/565ms |              3.98GB |
| `attention_w8`  | 29/50 = 0.58 [0.442, 0.706] |                    308 |                          9/16 |     0.230 |       571/613ms |              4.33GB |
| `gen_branch_w8` | 40/50 = 0.80 [0.670, 0.888] |                  220.5 |                          11/7 |     0.481 |       570/610ms |              4.67GB |

Full W4 and GenW8 are not distinguishable from full W8 on these 50 Banana
episodes. W4 uses the least steady model allocation; full W8 remains the only
calibration-free strategy.

### Denoise-Step Interaction

| Strategy        | Four-step success | Two-step success | Paired wins/losses | McNemar p |
| --------------- | ----------------: | ---------------: | -----------------: | --------: |
| BF16            |             21/50 |            34/50 |               19/6 |    0.0146 |
| `full_w8`       |             29/50 |            36/50 |               15/8 |     0.210 |
| `full_w4`       |             20/50 |            37/50 |               20/3 |  0.000488 |
| `attention_w8`  |             25/50 |            29/50 |              14/10 |     0.541 |
| `gen_branch_w8` |             22/50 |            40/50 |               24/6 |   0.00143 |

Two steps significantly improve BF16, full W4, and GenW8 on this protocol.
Denoise steps and weight precision are separate runtime controls, but their
closed-loop quality effects are not orthogonal. More denoise steps are not
automatically a quality rollback for this Edge policy.

### Why Two Steps Can Beat Four

UniPC is a numerical solver, not an iterative policy-refinement guarantee.
More solver evaluations reduce integration error only when the learned vector
field and inference schedule are well matched. They do not guarantee a better
action under behavior-cloning or closed-loop task metrics.

With shift 5, this implementation evaluates RF timesteps `[999, 833]` at two
steps and `[999, 937, 833, 624]` at four steps before the terminal update to
zero. Thus four steps do not simply refine the same two transitions: they add
two model evaluations and alter the multistep solver history. A checkpoint can
be better calibrated to the shorter trajectory; later correction can move an
action away from its training-data manifold or compound vector-field error.

The BF16 result is the key control. Its significant 42% to 68% increase shows
that quantization error is not required for the effect. A 32-request action
diagnostic also rejects a simple action-magnitude explanation: BF16 g3/s2 and
g3/s4 have absolute means 1.026 and 1.029 and chunk cosine similarity 0.999.
Their mean absolute difference is 0.0387, however, so small action changes can
still accumulate into different closed-loop trajectories. Two-step actions
have 16% larger mean adjacent-step variation (0.0299 versus 0.0257), but that
correlation does not establish causality.

The defensible conclusion is task- and checkpoint-specific: two steps are a
validated better operating point for Edge Banana under this protocol, not a
general statement that fewer diffusion steps improve robot policies. Broader
tasks and seeds are required before changing a universal default.

Concurrent multi-lane rollout wall time is intentionally excluded from the
performance tables. Isolated replay provides the policy latency and memory
measurements above.

The result table can be reproduced after rollout with the RoboLab Python
(which provides `h5py`). The first setting is the paired baseline; reporting
stops if any run is missing or any HDF5 initial-state hash differs.

```bash
ROBOLAB_PYTHON=/path/to/RoboLab/.venv/bin/python
$ROBOLAB_PYTHON examples/robolab_quant/summarize_rollouts.py \
  bf16=/path/to/bf16/BananaInBowlTask \
  full_w8=/path/to/full_w8/BananaInBowlTask
```

## Current Conclusion

Cosmos Lite can build, validate, directly load, and serve all four Cosmos3
Edge bundle strategies without the source checkpoint. Edge BF16 and every
quantized variant fit comfortably on a 24GB RTX 4090. Quantization saves up to
4.0GB of steady model allocation relative to BF16. Guidance 3 / two steps is
the retained accelerated sampler; guidance 1 is rejected.

Use full W8 with guidance 3 / two steps as the calibration-free deployment
starting point. Full W4 is the smallest bundle and has the lowest steady model
allocation. GenW8 has the highest observed Banana point estimate, but it is not
significantly different from full W8 and has no Edge cross-task evidence.
Retain all four strategies as selectable artifacts and validate a new task with
paired rollouts before promoting a mixed or W4 strategy.
