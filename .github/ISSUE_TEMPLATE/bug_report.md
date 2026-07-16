---
name: Bug Report
about: Report a reproducible bug or unexpected behavior
title: "[BUG] <short description>"
labels: 'bug'
assignees: ''

---

## Bug Description

<!-- Clear and concise description of the bug. What did you expect? What happened? -->

## Reproduction Steps

```bash
# Minimal command or script to reproduce
```

**Reproducibility:**

- [ ] Always
- [ ] Intermittently (~___% of the time)
- [ ] Only once

## Expected vs. Actual Behavior

|              | Description                 |
| ------------ | --------------------------- |
| **Expected** | What you expected to happen |
| **Actual**   | What actually happened      |

## Outputs

<details>
<summary>Error / Stack Trace</summary>

<!-- Attach or paste error / stack trace -->

</details>

<details>
<summary>Log Files</summary>

<!-- Attach or paste logs from the output directory -->

</details>

## System Information

| Field                        | Value                                       |
| ---------------------------- | ------------------------------------------- |
| **Environment**              | <!-- e.g. UV, Docker -->                    |
| **Hardware**                 | <!-- e.g. RTX 4090 24GB -->                 |
| **OS**                       | <!-- e.g. Ubuntu 22.04 / 24.04 -->          |
| **GPU Driver**               | <!-- e.g. 580.95.05 -->                     |
| **CUDA Version**             | <!-- e.g. 12.8.1 -->                        |
| **Python Version**           | <!-- e.g. 3.13.3 -->                        |
| **Package Version / Commit** | <!-- e.g. v1.2.3 or git SHA -->             |

## Additional Context

<!-- Workarounds tried, related issues, etc. -->

## Artifact And Policy Configuration

<!-- For quantization issues: strategy, bundle schema/manifest hash, sampler
guidance/steps, model source revision, task, and whether validation passed. Do
not attach proprietary captures, model weights, tokens, or private paths. -->
