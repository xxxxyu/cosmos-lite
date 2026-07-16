# Pull Request

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

## Summary

<!-- What problem does this solve, and why does it belong in Cosmos Lite? -->

## Validation

<!-- List exact checks, GPUs, bundles, and evaluation protocols used. -->

- [ ] `just lint`
- [ ] `python ci/check_public_release.py`
- [ ] Relevant unit tests
- [ ] GPU smoke or benchmark, when behavior or performance changes

## Compatibility And Risk

<!-- Note schema, checkpoint, dependency, memory, latency, and rollout impact. -->

- [ ] Documentation and benchmark tables are updated where required.
- [ ] No credentials, private paths, proprietary data, or model weights are included.
- [ ] Commits include a DCO sign-off (`git commit --signoff`).
