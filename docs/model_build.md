<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Building Quantized DROID Artifacts

Prebuilt bundles are preferred for deployment. Build locally when provenance,
custom calibration, or a new source revision requires it. The output is a
self-contained bundle; source weights and calibration data are not needed at
serve time.

## W8A16

W8A16 is calibration-free. Export streams source weights and never places the
complete BF16 policy on GPU.

```bash
HF_TOKEN=... \
MODEL_FAMILY=cosmos3_nano \
STRATEGY=full_w8 \
ASSET_DIR=/data/cosmos_lite/sources/nano \
BUNDLE_DIR=/data/cosmos_lite/nano_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

Set `MODEL_FAMILY=cosmos3_edge` for Edge. The build refuses to overwrite a
completed bundle. Source revisions are pinned in `sources.json` and copied to
the bundle manifest.

## GenW8A8

GenW8A8 needs per-linear input-channel statistics for its W4A16 layers. The
release calibration set contains 128 distinct successful episodes from
revision `5c11a20accb11497270a5247a7f1e66ad04c956c` of
`nvidia/Cosmos3-DROID`:

- one deterministic frame from the central 80% of each episode;
- wrist plus left and right exterior RGB views;
- guidance 3, four UniPC steps, seed 0 during collection;
- per-linear input-channel maximum and W4 equalization alpha 0.5.

Calibration does not update model parameters and does not use RoboLab
evaluation episodes.

```bash
HF_TOKEN=... \
MODEL_FAMILY=cosmos3_nano \
STRATEGY=gen_branch_w8a8 \
CALIBRATION_STATS=/path/to/droid_train128_input_amax.pt \
ASSET_DIR=/data/cosmos_lite/sources/nano \
BUNDLE_DIR=/data/cosmos_lite/nano_genw8a8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

The action-generation branch uses dynamic per-token FP8 activation scales.
The calibration statistics apply to the W4A16 remainder; GenW8A8 is not a
full-model FP8 artifact.

## Validate And Publish

Validate after every build or transfer:

```bash
BUNDLE_DIR=/data/cosmos_lite/nano_genw8a8 \
STRATEGY=gen_branch_w8a8 \
examples/robolab_quant/pipeline.sh validate
```

Validation checks schema, strategy, precision map, packed tensor payloads,
file sizes, and SHA256 hashes. Never modify a completed bundle in place. Build
a new immutable directory and retain its manifest hash with deployment records.
