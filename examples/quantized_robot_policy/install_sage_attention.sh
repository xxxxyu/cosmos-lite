#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${COSMOS_QUANT_PYTHON:-$script_dir/.venv/bin/python}"
uv_bin="${UV_BIN:-uv}"
sage_version="${SAGEATTENTION_VERSION:-2.2.0}"

[[ -x "$python_bin" ]] || {
  echo "quant runtime is missing; run $script_dir/setup.sh first" >&2
  exit 2
}
cuda_home="${CUDA_HOME:-}"
if [[ -z "$cuda_home" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.4; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      cuda_home="$candidate"
      break
    fi
  done
fi
if [[ -z "$cuda_home" || ! -x "$cuda_home/bin/nvcc" ]]; then
  echo "nvcc is required; set CUDA_HOME to a CUDA toolkit installation" >&2
  exit 2
fi
export CUDA_HOME="$cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"

if [[ -n "${SAGEATTENTION_SOURCE_DIR:-}" ]]; then
  source_spec="$SAGEATTENTION_SOURCE_DIR"
else
  source_spec="sageattention==$sage_version"
fi

TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS="${MAX_JOBS:-8}" \
  "$uv_bin" pip install \
    --python "$python_bin" \
    --no-build-isolation \
    "$source_spec"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$python_bin" -c \
  'from sageattention.core import SM89_ENABLED; assert SM89_ENABLED, "SageAttention SM89 kernel was not built"'

echo "SageAttention $sage_version SM89 backend is ready."
