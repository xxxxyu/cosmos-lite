#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${COSMOS_QUANT_PYTHON:-$script_dir/.venv/bin/python}"
uv_bin="${UV_BIN:-uv}"
sage_version="${SAGEATTENTION_VERSION:-2.2.0}"
sage_commit="${SAGEATTENTION_COMMIT:-eb615cf6cf4d221338033340ee2de1c37fbdba4a}"
check_only=0

case "${1:-}" in
  --check)
    check_only=1
    ;;
  "")
    ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

[[ -x "$python_bin" ]] || {
  if [[ "$check_only" == "1" ]]; then
    exit 1
  fi
  echo "quant runtime is missing; run $script_dir/setup.sh first" >&2
  exit 2
}

if [[ -n "${SAGEATTENTION_SOURCE_DIR:-}" ]]; then
  source_spec="$SAGEATTENTION_SOURCE_DIR"
else
  source_spec="https://github.com/thu-ml/SageAttention/archive/${sage_commit}.tar.gz"
fi
build_record="$script_dir/.venv/cosmos_lite_sageattention_build.json"

if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$python_bin" -c \
  'import importlib.metadata,json,pathlib,sys; record=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert record["source"] == sys.argv[2]; assert record["sageattention"] == sys.argv[3]; assert record["torch_cuda_arch_list"] == "8.9"; from sageattention.core import SM89_ENABLED; assert SM89_ENABLED; assert importlib.metadata.version("sageattention") == sys.argv[3]' \
  "$build_record" "$source_spec" "$sage_version" >/dev/null 2>&1; then
  if [[ "$check_only" == "0" ]]; then
    echo "SageAttention $sage_version SM89 backend is already ready."
  fi
  exit 0
fi
if [[ "$check_only" == "1" ]]; then
  exit 1
fi

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

TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS="${MAX_JOBS:-8}" \
  "$uv_bin" pip install \
    --python "$python_bin" \
    --no-build-isolation \
    "$source_spec"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$python_bin" -c \
  'from sageattention.core import SM89_ENABLED; assert SM89_ENABLED, "SageAttention SM89 kernel was not built"'

"$python_bin" -c \
  'import importlib.metadata,json,pathlib,subprocess,sys,torch; pathlib.Path(sys.argv[1]).write_text(json.dumps({"cuda_home": sys.argv[3], "nvcc": subprocess.run([sys.argv[3] + "/bin/nvcc", "--version"], check=True, capture_output=True, text=True).stdout.strip(), "sageattention": importlib.metadata.version("sageattention"), "source": sys.argv[2], "torch": torch.__version__, "torch_cuda": torch.version.cuda, "torch_cuda_arch_list": "8.9"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")' \
  "$build_record" "$source_spec" "$CUDA_HOME"

echo "SageAttention $sage_version SM89 backend is ready."
