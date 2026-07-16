#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail
export COSMOS_TRAINING=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

export COSMOS_REPO="${COSMOS_REPO:-$repo_root}"
export COSMOS_PYTHON="${COSMOS_PYTHON:-$COSMOS_REPO/examples/quantized_robot_policy/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5577}"
export REPLAY_LIMIT="${REPLAY_LIMIT:-8}"
export SERVED_ACTION_STEPS="${SERVED_ACTION_STEPS:-8}"
export SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
export SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-60}"
export PROFILE_TOOL="${PROFILE_TOOL:-none}" # none | nsys | ncu-marlin
export LINEAR_SHAPE_PROFILE="${LINEAR_SHAPE_PROFILE:-0}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export NSYS_BIN="${NSYS_BIN:-nsys}"
export NCU_BIN="${NCU_BIN:-ncu}"
export NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*marlin::Marlin.*}"
export RUN_DIR="${RUN_DIR:-/tmp/cosmos3_robocasa365_quant_profile_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$COSMOS_PYTHON" ]]; then
  echo "quant runtime is missing; run examples/quantized_robot_policy/setup.sh" >&2
  exit 2
fi

: "${QUANT_BUNDLE_DIR:?set QUANT_BUNDLE_DIR to a self-contained schema-v2 quant bundle}"
: "${REPLAY_CAPTURE_DIR:?set REPLAY_CAPTURE_DIR to the captured replay request directory}"

mkdir -p "$RUN_DIR/server" "$RUN_DIR/replay"
export COSMOS3_PROFILE_JSONL="$RUN_DIR/profile_events.jsonl"
if [[ "$LINEAR_SHAPE_PROFILE" == "1" ]]; then
  export COSMOS3_LINEAR_SHAPES_JSONL="${COSMOS3_LINEAR_SHAPES_JSONL:-$RUN_DIR/linear_shapes.jsonl}"
fi

validate_args=(--quant-artifact-dir "$QUANT_BUNDLE_DIR" --require-self-contained)
if [[ -n "${STRATEGY:-}" ]]; then
  validate_args+=(--strategy "$STRATEGY")
fi
"$COSMOS_PYTHON" -m cosmos_framework.scripts.robocasa365_quant_pipeline validate-artifact "${validate_args[@]}"

server_cmd=(
  "$COSMOS_PYTHON" -m cosmos_framework.scripts.action_policy_server_robocasa365_quant
  --output-dir "$RUN_DIR/server"
  --host "$HOST"
  --port "$PORT"
  --served-action-steps "$SERVED_ACTION_STEPS"
  --no-guardrails
  --quant-import-dir "$QUANT_BUNDLE_DIR"
)
if [[ "$TORCH_COMPILE" == "1" ]]; then
  server_cmd+=(--torch-compile)
else
  server_cmd+=(--no-torch-compile)
fi

case "$PROFILE_TOOL" in
  none)
    profiled_cmd=("${server_cmd[@]}")
    ;;
  nsys)
    profiled_cmd=(
      "$NSYS_BIN" profile
      --force-overwrite=true
      --sample=none
      --cpuctxsw=none
      --trace=cuda,nvtx,cublas,osrt
      -o "$RUN_DIR/nsys_server"
      "${server_cmd[@]}"
    )
    ;;
  ncu-marlin)
    profiled_cmd=(
      "$NCU_BIN"
      --target-processes all
      --kernel-name-base demangled
      --kernel-name "$NCU_KERNEL_NAME"
      --launch-skip "${NCU_LAUNCH_SKIP:-32}"
      --launch-count "${NCU_LAUNCH_COUNT:-8}"
      --set "${NCU_SET:-basic}"
      --force-overwrite
      --export "$RUN_DIR/ncu_marlin"
      "${server_cmd[@]}"
    )
    ;;
  *)
    echo "Unsupported PROFILE_TOOL=$PROFILE_TOOL; use none, nsys, or ncu-marlin" >&2
    exit 2
    ;;
esac

"${profiled_cmd[@]}" >"$RUN_DIR/server.log" 2>&1 &
server_pid=$!

send_kill() {
  "$COSMOS_PYTHON" -c 'import os, msgpack, zmq; ctx=zmq.Context(); sock=ctx.socket(zmq.REQ); sock.setsockopt(zmq.LINGER, 0); sock.setsockopt(zmq.RCVTIMEO, 3000); sock.connect("tcp://{}:{}".format(os.environ["HOST"], int(os.environ["PORT"]))); sock.send(msgpack.packb({"endpoint":"kill"})); sock.recv()' >/dev/null 2>&1 || true
}

cleanup() {
  if kill -0 "$server_pid" >/dev/null 2>&1; then
    send_kill
    for _ in $(seq 1 "$SERVER_SHUTDOWN_TIMEOUT_SEC"); do
      if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
  wait "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 "$SERVER_READY_TIMEOUT_SEC"); do
  if "$COSMOS_PYTHON" -c 'import os, socket; s=socket.create_connection((os.environ["HOST"], int(os.environ["PORT"])), timeout=1); s.close()' >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    echo "server exited before becoming ready; see $RUN_DIR/server.log" >&2
    exit 1
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "server did not become ready within ${SERVER_READY_TIMEOUT_SEC}s; see $RUN_DIR/server.log" >&2
  exit 1
fi

"$COSMOS_PYTHON" "$COSMOS_REPO/cosmos_framework/scripts/replay_policy_requests.py" \
  --capture-dir "$REPLAY_CAPTURE_DIR" \
  --output-dir "$RUN_DIR/replay" \
  --host "$HOST" \
  --port "$PORT" \
  --limit "$REPLAY_LIMIT"

send_kill
for _ in $(seq 1 "$SERVER_SHUTDOWN_TIMEOUT_SEC"); do
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if kill -0 "$server_pid" >/dev/null 2>&1; then
  kill "$server_pid" >/dev/null 2>&1 || true
fi
wait "$server_pid" >/dev/null 2>&1 || true
trap - EXIT

if [[ -s "$COSMOS3_PROFILE_JSONL" ]]; then
  "$COSMOS_PYTHON" -m cosmos_framework.scripts.summarize_profile_events \
    --run-dir "$RUN_DIR" \
    >"$RUN_DIR/profile_summary.json"
fi

if [[ "$PROFILE_TOOL" == "nsys" && -s "$RUN_DIR/nsys_server.nsys-rep" ]]; then
  "$NSYS_BIN" stats \
    --force-export=true \
    --format csv \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    "$RUN_DIR/nsys_server.nsys-rep" \
    >"$RUN_DIR/nsys_stats.csv"
  "$COSMOS_PYTHON" -m cosmos_framework.scripts.summarize_nsys_stats \
    --stats-csv "$RUN_DIR/nsys_stats.csv" \
    --output-json "$RUN_DIR/nsys_kernel_summary.json" \
    --output-md "$RUN_DIR/nsys_kernel_summary.md" \
    >"$RUN_DIR/nsys_kernel_summary.stdout.json"
fi

echo "profile replay complete: $RUN_DIR"
