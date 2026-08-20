#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
uv_bin="${UV_BIN:-uv}"
install_sage="${INSTALL_SAGE_ATTENTION:-0}"

case "${1:-}" in
  --with-sage)
    install_sage=1
    ;;
  --help|-h)
    echo "Usage: $0 [--with-sage]"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

command -v "$uv_bin" >/dev/null 2>&1 || {
  echo "uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 2
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required on the policy server host" >&2
  exit 2
}

sync_args=(
  --project "$script_dir"
  --frozen
  --no-dev
)
if [[ "$install_sage" == "1" ]] && "$script_dir/install_sage_attention.sh" --check; then
  sync_args+=(--inexact)
fi
"$uv_bin" sync "${sync_args[@]}"

python_bin="$script_dir/.venv/bin/python"
COSMOS_TRAINING=0 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$python_bin" "$script_dir/check_runtime.py" \
    --require-cuda \
    --repo-root "$repo_root"

if [[ "$install_sage" == "1" ]]; then
  COSMOS_QUANT_PYTHON="$python_bin" "$script_dir/install_sage_attention.sh"
fi

printf '\nRuntime ready. Use: %s\n' "$python_bin"
