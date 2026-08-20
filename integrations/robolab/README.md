<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# RoboLab Integration

Cosmos Lite owns policy inference. RoboLab owns simulation, observations, and
dataset files. This directory carries version-pinned patches for bounded
recording and training-ready Cosmos-DROID trajectory export. Simulator internals
remain outside the Cosmos Lite runtime.

## Supported Base

- Repository: `https://github.com/NVlabs/RoboLab`
- Release: `v0.3.1`
- Base commit: `9db0aaf`
- Patches, in order:
  1. `0001-bounded-cpu-streaming-image-recorder.patch`
  2. `0002-Add-training-ready-Cosmos-DROID-trajectory-export.patch`

Apply it only to a clean checkout at the pinned commit:

```bash
git -C "$ROBOLAB_DIR" switch --detach 9db0aaf
git -C "$ROBOLAB_DIR" am \
  /path/to/cosmos-lite/integrations/robolab/0001-bounded-cpu-streaming-image-recorder.patch \
  /path/to/cosmos-lite/integrations/robolab/0002-Add-training-ready-Cosmos-DROID-trajectory-export.patch
```

`git am` stops when the target source no longer matches. Adapting the patch to
another RoboLab release requires a recorder API review plus the unit, schema,
full-horizon, and capacity tests.

Patch 1 fixes CUDA-resident, unbounded HDF5 recording. Patch 2 writes successful
rollouts in the LeRobot v3 schema consumed by Cosmos-DROID training, while
retaining HDF5 and JSON as reproduction sidecars. Normal evaluation does not
require either patch.

See [RoboLab Data Generation](../../examples/robolab_quant/DATA_GENERATION.md)
for installation, collection, validation, and measured overhead.
