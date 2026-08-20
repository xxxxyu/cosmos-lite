# Change Log

## Cosmos Lite 0.3.0 (August 2026)

- Add config-driven RoboLab serving for Cosmos3 Nano and Edge DROID policies.
- Add self-contained Nano W8A16 and Nano/Edge GenW8A8 bundle validation and
  loading, including strong file-hash and tensor checks.
- Add the optimized RTX 4090 runtime: FP8 generation GEMMs, compiled language
  blocks, shared projections, SageAttention, condition caching, sparse input
  transforms, and reduced request dispatch.
- Add reproducible RoboLab-120 evaluation for the four release profiles and
  shared-server, multi-environment rollout topologies.
- Add bounded RoboLab recording and success-only Cosmos-DROID trajectory export
  in the LeRobot v3 schema.
- Add pinned RoboLab integration patches, release YAMLs, deployment guides,
  benchmark reports, and English and Chinese optimization reports.

The promoted profiles are Nano GenW8A8, Nano W8A16, Edge GenW8A8, and Edge
BF16. Older W4A16 and mixed weight-only artifacts remain experimental and are
not release recommendations.

## Upstream Cosmos Framework

The entries below are inherited from NVIDIA Cosmos Framework.

## 1.2.2 (May 14, 2026)

- New features
  - Add action policy closed-loop evaluation.

## 1.2.1 (May 08, 2026)

- New features
  - Add [action policy post-training (SFT)](./docs/training.md).

## 1.2.0 (May 05, 2026)

- New features
  - Add action modalities (Forward Dynamics, Inverse Dynamics, Policy) for Cosmos3-Nano model.
  - Upgrade Cosmos3-Nano checkpoint to improve T2V, I2V quality.

## 1.1.1 (May 01, 2026)

- New features
  - Add DCP checkpoint conversion/inference.

## 1.1.0 (April 29, 2026)

- New features
  - Add [Post-Training (Supervised Fine-Tuning)](./docs/training.md).
