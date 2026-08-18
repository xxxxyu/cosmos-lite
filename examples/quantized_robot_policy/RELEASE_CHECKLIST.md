<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Cosmos Lite Robot Policy Release Checklist

A release tag is valid only after all required rows pass from a fresh checkout.
The RoboLab BF16 baseline is intentionally not part of this release gate. The
formal RoboLab matrix contains Nano W8A16, Nano GenW8A8, Edge W8A16, and Edge
GenW8A8; development strategies do not block a release.

| Gate                                                   | RoboCasa365 | RoboLab W8 | RoboLab GenW8A8 |
| ------------------------------------------------------ | ----------: | ---------: | --------------: |
| Locked base runtime installs without external packages |    required |   required |        required |
| Optional Sage SM89 build is explicit and reproducible  |         n/a |        n/a |        required |
| Deployment doctor accepts the exact YAML and bundle    |    required |   required |        required |
| Strong bundle hash/tensor validation                   |    required |   required |        required |
| Config-driven server reaches ready state below 24GB    |    required |   required |        required |
| Request-replay parity, latency, and memory gate        |    required |   required |        required |
| Repeated-request stability and bounded-memory gate     |    required |   required |        required |
| Closed-loop simulator smoke                            |    required |   required |        required |
| Documented benchmark under the release protocol        |    required |   required |        required |

Release procedure:

1. Start from a clean worktree at the proposed commit.
2. Run `examples/quantized_robot_policy/setup.sh` with no external
   `PYTHONPATH` or `LD_LIBRARY_PATH` overrides.
3. Run the deployment doctor and strong validation for every formal bundle.
4. Run request replay, repeated-request stability, and one closed-loop smoke
   rollout for every formal configuration.
5. Confirm logs contain no source-checkpoint access, network fallback, OOM, or
   simulator/server crash.
6. Confirm every run retained `resolved_deployment_config.json`, the bundle
   manifest hash, and request profile.
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
