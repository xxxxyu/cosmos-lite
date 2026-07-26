#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail
export COSMOS_TRAINING=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
runtime_dir="$repo_root/examples/quantized_robot_policy"
python_bin="${COSMOS_QUANT_PYTHON:-$runtime_dir/.venv/bin/python}"
command_name="${1:-help}"

require_value() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "set $name before running '$command_name'" >&2
    exit 2
  fi
}

require_runtime() {
  if [[ ! -x "$python_bin" ]]; then
    echo "quant runtime is missing; run '$0 setup' first" >&2
    exit 2
  fi
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local pid="$3"
  local timeout="${SERVER_READY_TIMEOUT_SEC:-900}"
  for _ in $(seq 1 "$timeout"); do
    if "$python_bin" -c 'import sys,urllib.request; urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/healthz", timeout=1).read()' "$host" "$port" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "policy server exited before becoming ready; see $RUN_DIR/server.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "policy server was not ready after ${timeout}s; see $RUN_DIR/server.log" >&2
  return 1
}

require_free_port() {
  local host="$1"
  local port="$2"
  if "$python_bin" -c 'import socket,sys; s=socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1); s.close()' "$host" "$port" >/dev/null 2>&1; then
    echo "port $host:$port is already in use; choose another PORT" >&2
    exit 2
  fi
}

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

build_compile_args() {
  compile_args=()
  if [[ "${TORCH_COMPILE:-0}" == "1" ]]; then
    compile_args+=(--use-torch-compile --compiled-region "${COMPILED_REGION:-language}")
    if [[ "${CUDA_GRAPHS:-0}" == "1" ]]; then
      compile_args+=(--use-cuda-graphs)
    fi
    if [[ "${COMPILE_DYNAMIC:-1}" == "1" ]]; then
      compile_args+=(--compile-dynamic)
    else
      compile_args+=(--no-compile-dynamic)
    fi
  fi
  if [[ "${FP8_PROJECTION_FUSION:-none}" != "none" ]]; then
    compile_args+=(--fp8-projection-fusion "${FP8_PROJECTION_FUSION}")
  fi
}

start_server() {
  require_value BUNDLE_DIR
  local policy_gpu="${POLICY_GPU:-0}"
  local host="${HOST:-127.0.0.1}"
  local port="${PORT:-8000}"
  local guidance="${GUIDANCE:-3.0}"
  local num_steps="${NUM_STEPS:-2}"
  build_compile_args
  RUN_DIR="${RUN_DIR:-$PWD/robolab_quant_run_$(date +%Y%m%d_%H%M%S)}"
  export RUN_DIR
  mkdir -p "$RUN_DIR/server"
  require_free_port "$host" "$port"
  CUDA_VISIBLE_DEVICES="$policy_gpu" "$python_bin" -m \
    cosmos_framework.scripts.action_policy_server_robolab \
      --quant-import-dir "$BUNDLE_DIR" \
      --host "$host" \
      --port "$port" \
      --output-dir "$RUN_DIR/server" \
      --profile-jsonl "$RUN_DIR/profile.jsonl" \
      --deterministic-seed \
      --guidance "$guidance" \
      --num-steps "$num_steps" \
      "${compile_args[@]}" \
      >"$RUN_DIR/server.log" 2>&1 &
  server_pid=$!
  wait_for_port "$host" "$port" "$server_pid"
}

case "$command_name" in
  setup)
    exec "$runtime_dir/setup.sh"
    ;;
  build-public)
    require_runtime
    require_value ASSET_DIR
    require_value BUNDLE_DIR
    args=(
      build-public
      --asset-dir "$ASSET_DIR"
      --model-family "${MODEL_FAMILY:-cosmos3_nano}"
      --strategy "${STRATEGY:-full_w8}"
      --output-dir "$BUNDLE_DIR"
      --device cuda:0
    )
    if [[ -n "${CALIBRATION_STATS:-}" ]]; then
      args+=(--calibration-stats "$CALIBRATION_STATS")
    fi
    CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
      cosmos_framework.scripts.robolab_quant_pipeline "${args[@]}"
    "$python_bin" -m cosmos_framework.scripts.robolab_quant_pipeline validate \
      --bundle-dir "$BUNDLE_DIR" \
      --expected-strategy "${STRATEGY:-full_w8}" \
      --check-hashes \
      --check-tensors
    ;;
  validate)
    require_runtime
    require_value BUNDLE_DIR
    args=(validate --bundle-dir "$BUNDLE_DIR" --check-hashes --check-tensors)
    if [[ -n "${STRATEGY:-}" ]]; then
      args+=(--expected-strategy "$STRATEGY")
    fi
    exec "$python_bin" -m cosmos_framework.scripts.robolab_quant_pipeline "${args[@]}"
    ;;
  serve)
    require_runtime
    require_value BUNDLE_DIR
    require_free_port 127.0.0.1 "${PORT:-8000}"
    build_compile_args
    exec env CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
      cosmos_framework.scripts.action_policy_server_robolab \
        --quant-import-dir "$BUNDLE_DIR" \
        --host "${HOST:-127.0.0.1}" \
        --port "${PORT:-8000}" \
        --output-dir "${RUN_DIR:-$PWD/robolab_quant_server}" \
        --profile-jsonl "${RUN_DIR:-$PWD/robolab_quant_server}/profile.jsonl" \
        --deterministic-seed \
        --guidance "${GUIDANCE:-3.0}" \
        --num-steps "${NUM_STEPS:-2}" \
        "${compile_args[@]}"
    ;;
  replay)
    require_runtime
    require_value CAPTURE_DIR
    start_server
    replay_args=(
      --capture-dir "$CAPTURE_DIR"
      --output-dir "$RUN_DIR/replay"
      --host "${HOST:-127.0.0.1}"
      --port "${PORT:-8000}"
      --limit "${REPLAY_LIMIT:-32}"
    )
    if [[ -n "${REFERENCE_DIR:-}" ]]; then
      replay_args+=(--reference-dir "$REFERENCE_DIR")
    fi
    "$python_bin" -m cosmos_framework.scripts.robolab_policy_replay "${replay_args[@]}"
    echo "RoboLab replay complete: $RUN_DIR"
    ;;
  rollout)
    require_runtime
    require_value ROBOLAB_DIR
    require_value ROBOLAB_PYTHON
    if [[ "${POLICY_GPU:-0}" == "${SIM_GPU:-1}" ]]; then
      echo "POLICY_GPU and SIM_GPU must be different physical GPUs" >&2
      exit 2
    fi
    start_server
    output_name="${OUTPUT_NAME:-cosmos3_quant_$(date +%Y%m%d_%H%M%S)}"
    (
      cd "$ROBOLAB_DIR"
      CUDA_VISIBLE_DEVICES="${SIM_GPU:-1}" "$ROBOLAB_PYTHON" policies/cosmos3/run.py \
        --task "${TASK:-BananaInBowlTask}" \
        --remote-host "${HOST:-127.0.0.1}" \
        --remote-port "${PORT:-8000}" \
        --num-envs "${NUM_ENVS:-1}" \
        --num-runs "${NUM_RUNS:-1}" \
        --device cuda:0 \
        --headless \
        --video-mode "${VIDEO_MODE:-none}" \
        --output-folder-name "$output_name"
    ) >"$RUN_DIR/rollout.log" 2>&1
    echo "RoboLab rollout complete: $RUN_DIR"
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: examples/robolab_quant/pipeline.sh COMMAND

Commands:
  setup         install and validate the locked minimal policy runtime
  build-public  download official DROID inputs and build a self-contained bundle
  validate      verify bundle hashes, packed tensors, and precision map
  serve         run the OpenPI WebSocket policy server
  replay        start the server and replay captured OpenPI requests
  rollout       start the server and run a RoboLab task on a separate GPU

Required environment variables by command:
  build-public: ASSET_DIR, BUNDLE_DIR; optional MODEL_FAMILY, STRATEGY, CALIBRATION_STATS, POLICY_GPU
  validate:     BUNDLE_DIR; optional STRATEGY
  serve:        BUNDLE_DIR; optional POLICY_GPU, HOST, PORT, GUIDANCE, NUM_STEPS,
                TORCH_COMPILE, COMPILED_REGION, COMPILE_DYNAMIC, CUDA_GRAPHS,
                FP8_PROJECTION_FUSION
  replay:       BUNDLE_DIR, CAPTURE_DIR; optional REPLAY_LIMIT and server variables
  rollout:      BUNDLE_DIR, ROBOLAB_DIR, ROBOLAB_PYTHON; optional SIM_GPU, TASK,
                NUM_ENVS, NUM_RUNS, VIDEO_MODE and server variables
EOF
    ;;
  *)
    echo "unknown command: $command_name" >&2
    exit 2
    ;;
esac
