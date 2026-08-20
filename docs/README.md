<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite Documentation

Start with the path that matches your task.

## Use Cosmos Lite

- [RoboLab deployment guide](../examples/robolab_quant/README.md): install,
  download, validate, serve, replay, and roll out Nano or Edge.
- [RoboLab data generation](../examples/robolab_quant/DATA_GENERATION.md):
  collect successful, training-ready Cosmos-DROID trajectories in LeRobot v3.
- [RoboLab-120 benchmark](benchmarks/robolab.md): primary success, latency,
  memory, protocol, and evaluation-topology results.
- [Building model artifacts](model_build.md): create self-contained W8A16 or
  GenW8A8 bundles from NVIDIA's public DROID policies.
- [RoboCasa365 integration](../examples/robocasa365_quant/README.md): secondary
  support for a user-provided fine-tuned Nano checkpoint.

## Understand The Runtime

- [Runtime architecture](runtime_architecture.md): artifact, startup, request,
  replay, rollout, and fallback behavior.
- [Optimization report](cosmos_lite_optimization_report.md): complete English
  account of explored and retained techniques.
- [优化技术报告（中文）](cosmos_lite_optimization_report_zh.md).

## Reproduce Experiments

- [RoboLab ablations](benchmarks/robolab_ablations.md): sampler and historical
  quantization comparisons.
- [FP8 W8A8](experiments/fp8_w8a8.md): activation quantization and quality gates.
- [Compile and CUDA Graphs](experiments/graph_optimization.md).
- [RTX 4090 SM89 optimization](experiments/rtx4090_sm89.md).

Files outside this index are inherited NVIDIA Cosmos Framework documentation.
