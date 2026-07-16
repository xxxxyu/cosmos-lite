<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Security Policy

## Reporting A Vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's
**Security** tab to submit a private vulnerability report through a Security
Advisory. If private reporting is not available, email
`xiangyu.sdlc@foxmail.com` with the subject `Cosmos Lite security report`.
Include the affected commit or tag, environment, impact, and minimal
reproduction steps. Do not send access tokens, proprietary model weights, or
sensitive robot data unless the maintainer explicitly requests a secure
transfer method.

Reports are handled on a best-effort basis. The maintainer will coordinate
disclosure after the impact is understood and a fix or mitigation is available.

If the issue is in unmodified NVIDIA Cosmos Framework code, also follow the
[NVIDIA vulnerability reporting process](https://www.nvidia.com/en-us/security/).
Reporting it here helps determine whether this fork needs a coordinated update.

## Deployment Boundary

The policy servers are research software. They default to loopback and have no
built-in authentication, authorization, or transport encryption. Do not bind
them to an untrusted network. Use an SSH tunnel or authenticated encrypted
proxy, restrict firewall access, and treat request captures and policy outputs
as sensitive operational data.

Simulator validation is not a real-robot safety case. Any hardware deployment
must independently enforce E-stop, watchdog, workspace and joint limits,
action magnitude and rate limits, stale-command rejection, collision handling,
and operator supervision. Never rely on the model server as the sole safety
boundary.

## Supported Versions

| Version                 | Security updates |
| ----------------------- | ---------------- |
| Latest release          | Supported        |
| `main`                  | Supported        |
| Older release snapshots | Not supported    |

Published tags are immutable and retained for reproducibility. Older tags may
remain vulnerable; upgrade to a fixed release rather than expecting an existing
tag to move.
