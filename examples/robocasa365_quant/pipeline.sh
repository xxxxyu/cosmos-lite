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
    if "$python_bin" -c 'import socket,sys; s=socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1); s.close()' "$host" "$port" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "policy server exited before becoming ready; inspect the server log" >&2
      return 1
    fi
    sleep 1
  done
  echo "policy server was not ready after ${timeout}s" >&2
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
build_dir=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$build_dir" && -d "$build_dir" ]]; then
    rm -rf "$build_dir"
  fi
}
trap cleanup EXIT

start_server() {
  require_value BUNDLE_DIR
  local host="${HOST:-127.0.0.1}"
  local port="${PORT:-5577}"
  RUN_DIR="${RUN_DIR:-$PWD/robocasa365_quant_run_$(date +%Y%m%d_%H%M%S)}"
  export RUN_DIR
  mkdir -p "$RUN_DIR/server"
  require_free_port "$host" "$port"
  export COSMOS3_PROFILE_JSONL="$RUN_DIR/profile_events.jsonl"
  CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
    cosmos_framework.scripts.action_policy_server_robocasa365_quant \
      --quant-import-dir "$BUNDLE_DIR" \
      --host "$host" \
      --port "$port" \
      --output-dir "$RUN_DIR/server" \
      --served-action-steps "${SERVED_ACTION_STEPS:-8}" \
      --deterministic-seed \
      --guidance "${GUIDANCE:-3.0}" \
      --num-steps "${NUM_STEPS:-4}" \
      --no-guardrails \
      --no-torch-compile \
      >"$RUN_DIR/server.log" 2>&1 &
  server_pid=$!
  wait_for_port "$host" "$port" "$server_pid"
}

case "$command_name" in
  setup)
    exec "$runtime_dir/setup.sh"
    ;;
  build)
    require_runtime
    for name in BF16_CHECKPOINT CONFIG_FILE TOKENIZER_DIR VAE_PATH BUNDLE_DIR; do
      require_value "$name"
    done
    strategy="${STRATEGY:-attention_w8}"
    plan_file="$script_dir/configs/$strategy.json"
    [[ -f "$plan_file" ]] || { echo "unsupported STRATEGY=$strategy" >&2; exit 2; }
    if [[ "$strategy" != "full_w8" ]]; then
      require_value CALIBRATION_CAPTURE_DIR
    fi
    if [[ -e "$BUNDLE_DIR" ]]; then
      echo "refusing to overwrite BUNDLE_DIR=$BUNDLE_DIR" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$BUNDLE_DIR")"
    build_dir="$(mktemp -d "$(dirname "$BUNDLE_DIR")/.${strategy}.build.XXXXXX")"
    build_log="${BUILD_LOG:-$BUNDLE_DIR.build.log}"
    packed_dir="$build_dir/packed"
    calibration_stats=""
    stream_phase="1/2"
    bundle_phase="2/2"
    : >"$build_log"

    if [[ "$strategy" != "full_w8" ]]; then
      stream_phase="3/4"
      bundle_phase="4/4"
      bootstrap_dir="$build_dir/bootstrap_full_w8"
      calibration_stats="$build_dir/input_amax.pt"
      echo "[1/4] stream-packing temporary full_w8 calibration model" | tee -a "$build_log"
      CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
        cosmos_framework.scripts.robocasa365_quant_pipeline \
          stream-export-packed \
          --strategy full_w8 \
          --checkpoint-path "$BF16_CHECKPOINT" \
          --output-dir "$bootstrap_dir" \
          --device cuda:0 \
          --max-cpu-batch-size-gb "${MAX_CPU_BATCH_SIZE_GB:-1.0}" \
          >>"$build_log" 2>&1

      runtime_output="$build_dir/calibration_runtime"
      host="127.0.0.1"
      port="${PORT:-5577}"
      require_free_port "$host" "$port"
      echo "[2/4] collecting training-capture input statistics" | tee -a "$build_log"
      CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
        cosmos_framework.scripts.action_policy_server_robocasa365_quant \
          --checkpoint-path "$BF16_CHECKPOINT" \
          --config-file "$CONFIG_FILE" \
          --quant-import-dir "$bootstrap_dir" \
          --allow-legacy-quant-artifact \
          --quant-calib-capture-dir "$CALIBRATION_CAPTURE_DIR" \
          --quant-calib-limit "${CALIBRATION_LIMIT:-128}" \
          --quant-calibration-stats-output "$calibration_stats" \
          --output-dir "$runtime_output" \
          --host "$host" \
          --port "$port" \
          --served-action-steps "${SERVED_ACTION_STEPS:-8}" \
          --no-guardrails \
          --deterministic-seed \
          --no-torch-compile \
          >>"$build_log" 2>&1 &
      server_pid=$!
      wait_for_port "$host" "$port" "$server_pid" || {
        echo "calibration failed; see $build_log" >&2
        exit 1
      }
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
      server_pid=""
      [[ -s "$calibration_stats" ]] || {
        echo "calibration did not produce $calibration_stats; see $build_log" >&2
        exit 1
      }
    fi

    echo "[$stream_phase] stream-packing $strategy" | tee -a "$build_log"
    stream_args=(
      stream-export-packed
      --strategy "$strategy"
      --checkpoint-path "$BF16_CHECKPOINT"
      --output-dir "$packed_dir"
      --device cuda:0
      --calibration-alpha "${CALIBRATION_ALPHA:-0.5}"
      --max-cpu-batch-size-gb "${MAX_CPU_BATCH_SIZE_GB:-1.0}"
    )
    if [[ -n "$calibration_stats" ]]; then
      stream_args+=(--calibration-stats "$calibration_stats")
    fi
    CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
      cosmos_framework.scripts.robocasa365_quant_pipeline "${stream_args[@]}" \
      >>"$build_log" 2>&1

    "$python_bin" -m cosmos_framework.scripts.robocasa365_quant_pipeline \
      write-artifact-metadata \
      --strategy "$strategy" \
      --quant-artifact-dir "$packed_dir" \
      --checkpoint-path "$BF16_CHECKPOINT" \
      --config-file "$CONFIG_FILE" \
      --calib-capture-dir "${CALIBRATION_CAPTURE_DIR:-}" \
      --calib-limit "${CALIBRATION_LIMIT:-128}" \
      --calib-alpha "${CALIBRATION_ALPHA:-0.5}"
    echo "[$bundle_phase] building and validating self-contained bundle" | tee -a "$build_log"
    "$python_bin" -m cosmos_framework.scripts.robocasa365_quant_pipeline \
      build-self-contained-bundle \
      --strategy "$strategy" \
      --quant-artifact-dir "$packed_dir" \
      --checkpoint-path "$BF16_CHECKPOINT" \
      --config-file "$CONFIG_FILE" \
      --tokenizer-dir "$TOKENIZER_DIR" \
      --vae-path "$VAE_PATH" \
      --output-dir "$BUNDLE_DIR"
    "$python_bin" -m cosmos_framework.scripts.robocasa365_quant_pipeline \
      validate-artifact \
      --quant-artifact-dir "$BUNDLE_DIR" \
      --strategy "$strategy" \
      --require-self-contained \
      --check-tensors
    echo "RoboCasa365 bundle ready: $BUNDLE_DIR"
    ;;
  validate)
    require_runtime
    require_value BUNDLE_DIR
    args=(validate-artifact --quant-artifact-dir "$BUNDLE_DIR" --require-self-contained --check-tensors)
    if [[ -n "${STRATEGY:-}" ]]; then
      args+=(--strategy "$STRATEGY")
    fi
    exec "$python_bin" -m cosmos_framework.scripts.robocasa365_quant_pipeline "${args[@]}"
    ;;
  serve)
    require_runtime
    require_value BUNDLE_DIR
    require_free_port 127.0.0.1 "${PORT:-5577}"
    exec env CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}" "$python_bin" -m \
      cosmos_framework.scripts.action_policy_server_robocasa365_quant \
        --quant-import-dir "$BUNDLE_DIR" \
        --host "${HOST:-127.0.0.1}" \
        --port "${PORT:-5577}" \
        --output-dir "${RUN_DIR:-$PWD/robocasa365_quant_server}" \
        --served-action-steps "${SERVED_ACTION_STEPS:-8}" \
        --deterministic-seed \
        --guidance "${GUIDANCE:-3.0}" \
        --num-steps "${NUM_STEPS:-4}" \
        --no-guardrails \
        --no-torch-compile
    ;;
  replay)
    require_runtime
    require_value CAPTURE_DIR
    start_server
    "$python_bin" -m cosmos_framework.scripts.replay_policy_requests \
      --capture-dir "$CAPTURE_DIR" \
      --output-dir "$RUN_DIR/replay" \
      --host "${HOST:-127.0.0.1}" \
      --port "${PORT:-5577}" \
      --limit "${REPLAY_LIMIT:-32}"
    "$python_bin" -m cosmos_framework.scripts.summarize_profile_events \
      --run-dir "$RUN_DIR" >"$RUN_DIR/profile_summary.json"
    echo "RoboCasa365 replay complete: $RUN_DIR"
    ;;
  rollout)
    require_runtime
    require_value ROBOCASA365_PYTHON
    require_value ROBOCASA365_ROLLOUT_SCRIPT
    start_server
    rollout_script="$(realpath "$ROBOCASA365_ROLLOUT_SCRIPT")"
    robocasa365_root="${ROBOCASA365_ROOT:-$(cd "$(dirname "$rollout_script")/../.." && pwd)}"
    (
      cd "$robocasa365_root"
      sim_env_file="${ROBOCASA365_ENV_FILE:-$robocasa365_root/rldx/eval/sim/robocasa365/robocasa365_uv/env.sh}"
      if [[ -f "$sim_env_file" ]]; then
        # The simulator owns EGL/MuJoCo configuration; keep it out of the policy runtime.
        source "$sim_env_file"
      fi
      robocasa365_source="${ROBOCASA365_SOURCE:-$robocasa365_root/external_dependencies/robocasa365}"
      sim_pythonpath="$robocasa365_root"
      if [[ -d "$robocasa365_source/robocasa" ]]; then
        sim_pythonpath="$sim_pythonpath:$robocasa365_source"
      fi
      RLDX_SKIP_HF_REGISTRATION=1 \
        PYTHONPATH="$sim_pythonpath${PYTHONPATH:+:$PYTHONPATH}" \
        "$ROBOCASA365_PYTHON" "$rollout_script" \
          --n_episodes "${N_EPISODES:-50}" \
          --policy_client_host "${HOST:-127.0.0.1}" \
          --policy_client_port "${PORT:-5577}" \
          --max_episode_steps "${MAX_EPISODE_STEPS:-1200}" \
          --env_name "${ENV_NAME:-robocasa/CloseFridge}" \
          --n_action_steps "${SERVED_ACTION_STEPS:-8}" \
          --n_envs "${N_ENVS:-5}" \
          --robocasa_split "${ROBOCASA_SPLIT:-target}" \
          --disable-video
    ) >"$RUN_DIR/rollout.log" 2>&1
    echo "RoboCasa365 rollout complete: $RUN_DIR"
    ;;
  help|--help|-h)
    cat <<'EOF'
Usage: examples/robocasa365_quant/pipeline.sh COMMAND

Commands:
  setup     install and validate the locked minimal policy runtime
  build     quantize a fine-tuned BF16 DCP and build a self-contained bundle
  validate  verify bundle hashes, packed tensors, and precision map
  serve     run the ZMQ policy server
  replay    start the server and replay captured RoboCasa requests
  rollout   start the server and run the external RLDX evaluator

Required environment variables by command:
  build:    BF16_CHECKPOINT, CONFIG_FILE, TOKENIZER_DIR, VAE_PATH, BUNDLE_DIR;
            CALIBRATION_CAPTURE_DIR is also required for W4/mixed strategies;
            optional STRATEGY, POLICY_GPU, MAX_CPU_BATCH_SIZE_GB
  validate: BUNDLE_DIR; optional STRATEGY
  serve:    BUNDLE_DIR; optional POLICY_GPU, HOST, PORT, GUIDANCE, NUM_STEPS
  replay:   BUNDLE_DIR, CAPTURE_DIR; optional REPLAY_LIMIT and server variables
  rollout:  BUNDLE_DIR, ROBOCASA365_PYTHON, ROBOCASA365_ROLLOUT_SCRIPT;
            optional ROBOCASA365_ROOT, ROBOCASA365_SOURCE,
            ROBOCASA365_ENV_FILE, N_EPISODES, N_ENVS, MAX_EPISODE_STEPS
            and server variables
EOF
    ;;
  *)
    echo "unknown command: $command_name" >&2
    exit 2
    ;;
esac
