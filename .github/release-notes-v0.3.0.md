# Cosmos Lite v0.3.0

Cosmos Lite v0.3.0 turns the earlier quantization experiments into a focused,
config-driven runtime for Cosmos3 Nano and Edge DROID policies. The release
targets efficient on-device inference and high-throughput RoboLab simulation
on RTX 4090 GPUs.

## Release Profiles

| Model    | Artifact | VRAM (GB) | Request p50 (ms) | RoboLab-120 SR (%) |
| -------- | -------- | --------: | ---------------: | -----------------: |
| Edge 4B  | BF16     |      9.20 |            582.0 |              20.92 |
| Edge 4B  | GenW8A8  |      8.79 |            331.4 |              19.25 |
| Nano 16B | W8A16    |     21.42 |          2,403.0 |              31.50 |
| Nano 16B | GenW8A8  |     15.51 |            958.5 |              31.67 |

All rows use guidance 3, two UniPC denoise steps, shift 5, and 1,200
closed-loop rollouts over all 120 RoboLab tasks. Request latency is measured on
one RTX 4090 and excludes simulator work.

## Highlights

- Self-contained Nano W8A16 and Nano/Edge GenW8A8 bundles with strict hashes,
  tensor validation, source provenance, and no deployment-time BF16 dependency.
- Release YAMLs that separate checkpoint precision from runtime sampling and
  fail explicitly when the tested backend is unavailable.
- RTX 4090 FP8 generation kernels, compiled language blocks, shared projection
  fusion, SageAttention, condition K/V caching, sparse video transforms, and a
  reduced request path.
- Shared-policy-server RoboLab rollout with measured 8-GPU topologies.
- Bounded, training-ready trajectory collection with three camera streams,
  aligned state/action rows, reproduction sidecars, and a strict validator.

## Start Here

<!-- markdown-link-check-disable -->
- [Quickstart](https://github.com/xxxxyu/cosmos-lite#quickstart)
- [RoboLab deployment guide](https://github.com/xxxxyu/cosmos-lite/blob/main/examples/robolab_quant/README.md)
- [RoboLab-120 benchmark](https://github.com/xxxxyu/cosmos-lite/blob/main/docs/benchmarks/robolab.md)
- [Runtime architecture](https://github.com/xxxxyu/cosmos-lite/blob/main/docs/runtime_architecture.md)
- [Optimization report](https://github.com/xxxxyu/cosmos-lite/blob/main/docs/cosmos_lite_optimization_report.md)
<!-- markdown-link-check-enable -->

Cosmos Lite is an unofficial community extension of NVIDIA Cosmos Framework.
Review the upstream model licenses and the repository safety guidance before
deployment.
