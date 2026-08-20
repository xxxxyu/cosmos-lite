<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite Robot Policy Release Checklist

A release tag is valid only after all required rows pass from a fresh checkout.
The formal RoboLab model matrix contains Nano W8A16, Nano GenW8A8, Edge BF16,
and Edge GenW8A8. Edge W8A16 and other development strategies do not block a
release. Model artifacts and runtime sampler settings are gated separately.

| Gate                                                   | RoboCasa365 |  Nano W8 | Edge BF16 |  GenW8A8 |
| ------------------------------------------------------ | ----------: | -------: | --------: | -------: |
| Locked base runtime installs without external packages |    required | required |  required | required |
| Optional Sage SM89 build is explicit and reproducible  |         n/a |      n/a |       n/a | required |
| Deployment doctor accepts the exact YAML and artifact  |    required | required |  required | required |
| Strong quantized-bundle hash/tensor validation         |    required | required |       n/a | required |
| Config-driven server reaches ready state below 24GB    |    required | required |  required | required |
| Request-replay parity, latency, and memory gate        |    required | required |  required | required |
| Repeated-request stability and bounded-memory gate     |    required | required |  required | required |
| Closed-loop simulator smoke                            |    required | required |  required | required |
| Documented benchmark under the release protocol        |    required | required |  required | required |
| Sampler defaults and validated overrides documented    |    required | required |  required | required |

Release procedure:

1. Start from a clean worktree at the proposed commit.
2. Run `examples/quantized_robot_policy/setup.sh` with no external
   `PYTHONPATH` or `LD_LIBRARY_PATH` overrides.
3. Run the deployment doctor and strong validation for every formal bundle.
4. Run request replay, repeated-request stability, and one closed-loop smoke
   rollout for every formal configuration.
5. Confirm self-contained quantized runs do not access source checkpoints or
   calibration data at runtime, and confirm every run has no unexpected
   network fallback, OOM, or simulator/server crash.
6. Confirm every run retained `resolved_deployment_config.json`, model-artifact
   identity, sampler settings, and request profile. Quantized runs must also
   retain the bundle manifest hash.
7. Run targeted unit tests and `git diff --check`.
8. Run `python ci/check_public_release.py` and audit the complete reachable Git
   history for credentials, private paths, host names, and unavailable LFS
   objects.
9. Confirm policy-server ports are loopback-only or protected by an
   authenticated encrypted proxy and firewall.
10. Run the approved small stratified RoboLab rollout gate. Do not start a
    larger cluster benchmark without a separate resource review.
11. Update benchmark documents with release-environment results.
12. Create one annotated tag at the tested commit and never move it.
