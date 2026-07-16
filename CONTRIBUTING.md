<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Contributing

Cosmos Lite is an independent, community-maintained extension of
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework). It is
maintained primarily by one developer, so focused issues and small,
well-tested pull requests are the easiest to review.

## Where To Report An Issue

- Open a Cosmos Lite issue for quantization, packed inference, self-contained
  bundles, RTX 4090 deployment, or the RoboLab and RoboCasa365 pipelines added
  by this repository.
- Report a bug that reproduces on an unmodified NVIDIA checkout to
  [NVIDIA Cosmos Framework](https://github.com/NVIDIA/Cosmos-Framework/issues).
- Follow [SECURITY.md](SECURITY.md) for vulnerabilities. Do not include secrets
  or sensitive deployment details in a public issue.

Search existing issues before opening a new one. A useful report includes the
commit or tag, GPU, driver, CUDA version, exact command, relevant logs, and the
quantization strategy and bundle manifest hash when applicable.

## Development Setup

Install `uv` and [`just`](https://just.systems/), then install the repository
hooks:

```bash
uv tool install -U rust-just
just pre-commit
```

The quantized policy runtime is intentionally isolated from simulator and
training environments:

```bash
CUDA_VISIBLE_DEVICES=0 examples/quantized_robot_policy/setup.sh
```

See the [runtime guide](examples/quantized_robot_policy/README.md) for the
supported CUDA and Python environment.

## Validation

Run checks that match the change:

```bash
just lint
python ci/check_public_release.py
git diff --check
```

For quantization or policy-server changes, also run the targeted tests:

```bash
examples/quantized_robot_policy/.venv/bin/python -m pytest -q \
  cosmos_framework/scripts/action_policy_server_robocasa365_quant_test.py \
  cosmos_framework/scripts/action_policy_server_robolab_test.py \
  cosmos_framework/scripts/export_robolab_train_calibration_requests_test.py \
  cosmos_framework/scripts/robocasa365_quant_pipeline_test.py \
  cosmos_framework/scripts/robolab_quant_bundle_test.py \
  cosmos_framework/scripts/robolab_quant_pipeline_test.py
```

GPU behavior, memory, latency, or model-output changes require a direct-load
smoke test on a supported NVIDIA GPU. Update the relevant benchmark document
only when the protocol, hardware, sample count, and raw result location are
recorded. Do not present replay parity as a closed-loop success rate.

The complete NVIDIA training suite can require multiple GPUs and is not a
default requirement for a Cosmos Lite pull request. Run broader upstream tests
when a change modifies shared Cosmos Framework behavior.

## Pull Requests

Keep each pull request focused and explain:

- the problem and the ownership boundary between Cosmos Lite and upstream;
- the implementation and compatibility impact;
- the checks and hardware validation performed;
- memory, latency, parity, or rollout changes, when relevant;
- any documentation, artifact schema, or migration impact.

Do not commit model weights, access tokens, private datasets, machine-specific
paths, internal host names, or generated environments. Published bundles belong
in an external model repository and must include provenance and integrity
metadata.

Review and merge are best-effort. A pull request may be declined when it is too
broad to validate, duplicates upstream work, weakens artifact or deployment
safety, or cannot be maintained in this repository.

## Signing Your Work

All commits must include a Developer Certificate of Origin sign-off:

```bash
git commit --signoff -m "Describe the change"
```

The sign-off certifies that you have the right to submit the contribution under
the applicable license. Read the full
[Developer Certificate of Origin 1.1](https://developercertificate.org/).
