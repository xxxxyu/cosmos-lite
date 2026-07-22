<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# Rollout Media

This directory contains the public README media for the validated robot-policy
pipelines.

## RoboLab Comparison

`robolab_quant_comparison.mp4` uses the same four quantized configurations as
the paired 50-episode Banana benchmark. Ten episodes were recorded per
configuration at guidance 3 and four denoise steps. The video uses the three
fastest successful episodes for W8A16, AttnW8, and GenW8; W4A16 uses its two
fastest successes and one failure. Episodes retain their original frame rate
and duration. The GitHub copy prepends a one-second static cover so repository
video hosting can extract a useful first-frame thumbnail. The video subset is
illustrative; benchmark success rates use all 50 paired rollouts.

The BF16 panel is an upstream visual reference and is not included in the
paired quantization statistics.

## RoboLab Edge Comparison

`robolab_edge_quant_comparison.mp4` compares BF16 and the same four quantized
strategies using guidance 3 and two denoise steps. Every panel uses three
episodes selected from its 50 paired runs: the fastest three successes, except
W4A16, which shows its fastest two successes and one failure. Unlike the Nano
video, the BF16 panel is replayed from this evaluation's own closed-loop run.
The MP4 begins with a one-second static cover for GitHub previews.

## RoboCasa365 Reference

`robocasa_closefridge_bf16_reference.mp4` contains the three fastest successful
episodes selected from the step-8000 BF16 CloseFridge reference evaluation.
The clips retain their original 20 fps timing. This video verifies the task
and observation presentation only; it is not evidence for quantized-policy
quality. Quantized results are reported separately in
[`examples/robocasa365_quant/BENCHMARKS.md`](../../examples/robocasa365_quant/BENCHMARKS.md).
