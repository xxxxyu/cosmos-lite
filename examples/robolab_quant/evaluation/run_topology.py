"""Launch independent vector simulator lanes against explicit policy endpoints.

Servers are provisioned separately. No simulator GPU is shared or preempted.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate(cfg):
    lanes = cfg["lanes"]
    if not lanes or len({x["name"] for x in lanes}) != len(lanes):
        raise ValueError("Lane names must be unique and nonempty")
    if len({x["sim_gpu"] for x in lanes}) != len(lanes):
        raise ValueError("Simulator GPUs must be unique")
    for lane in lanes:
        if lane["num_envs"] < 1 or lane["num_runs"] < 1:
            raise ValueError("Positive environment/run counts required")
        if lane["sim_gpu"] in cfg.get("policy_gpus", []):
            raise ValueError("Policy and simulation GPUs must be disjoint")
        if Path(lane["name"]).name != lane["name"] or lane["name"] in (".", ".."):
            raise ValueError("Unsafe lane name")


def run(cfg, output):
    validate(cfg)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    (output / "source_hashes.json").write_text(
        json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}, indent=2)
    )
    robolab = Path(cfg["robolab_repo"])
    tracked = [
        robolab / p
        for p in (
            "policies/cosmos3/client.py",
            "policies/cosmos3/run.py",
            "robolab/eval/episode.py",
            "robolab/eval/runner.py",
            "robolab/eval/summarize.py",
        )
    ]
    tracked.extend((robolab / "robolab/tasks").rglob("*.py"))
    (output / "robolab_hashes.json").write_text(
        json.dumps(
            {str(p.relative_to(robolab)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.is_file()},
            indent=2,
        )
    )
    occupancy = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = {int(x.split(",")[0]): int(x.split(",")[1]) for x in occupancy.splitlines()}
    (output / "gpu_before.txt").write_text(occupancy)
    for lane in cfg["lanes"]:
        if used[lane["sim_gpu"]] > 512:
            raise RuntimeError(f"Simulator GPU {lane['sim_gpu']} occupied")
        urllib.request.urlopen(f"http://{lane['host']}:{lane['port']}/healthz", timeout=10).close()
    started = time.monotonic()

    def launch(lane):
        out = output / lane["name"]
        out.mkdir()
        state = dict(status="starting", lane=lane["name"], planned=lane["num_envs"] * lane["num_runs"])

        def publish(**kw):
            state.update(kw, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            tmp = out / "status.tmp"
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(out / "status.json")

        env = dict(
            os.environ,
            ROBOLAB_REPO=cfg["robolab_repo"],
            LD_LIBRARY_PATH=cfg.get("library_path", "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"),
            ASYNC_TRACE=str(out / "requests.jsonl"),
            ASYNC_MODE=lane["mode"],
            ASYNC_HORIZON=str(cfg.get("horizon", 16)),
            ASYNC_DELAY=str(cfg.get("delay_s", 0)),
            ASYNC_JITTER=str(cfg.get("jitter_s", 0)),
            ASYNC_SEED_OFFSET=str(cfg.get("seed_offset", 3000)),
        )
        env.update(cfg.get("sim_env", {}))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        cmd = [
            cfg["sim_python"],
            str(HERE / "run_vector_async.py"),
            "--task",
            lane["task"],
            "--remote-host",
            lane["host"],
            "--remote-port",
            str(lane["port"]),
            "--num-envs",
            str(lane["num_envs"]),
            "--num-runs",
            str(lane["num_runs"]),
            "--device",
            f"cuda:{lane['sim_gpu']}",
            "--headless",
            "--video-mode",
            "none",
            "--output-folder-name",
            str(out),
            "--kit_args",
            f"--/renderer/multiGpu/enabled=false --/renderer/activeGpu={lane['sim_gpu']}",
        ]
        (out / "command.json").write_text(json.dumps(cmd, indent=2))
        begin = time.monotonic()
        with (out / "rollout.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=cfg["robolab_repo"], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                publish(status="running", pid=proc.pid, elapsed_s=time.monotonic() - begin)
                with (out / "rollout.log").open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 65536))
                    tail = handle.read().decode("utf-8", errors="replace")
                if any(
                    x in tail
                    for x in (
                        "No device could be created",
                        "no suitable CUDA GPU was found",
                        "GPU Foundation is not initialized!",
                    )
                ):
                    raise RuntimeError("Simulator GPU initialization failed; inspect rollout.log")
                if time.monotonic() - begin > cfg.get("timeout_s", 3600):
                    raise TimeoutError(lane["name"])
                time.sleep(5)
            if proc.returncode:
                raise RuntimeError(f"Simulator exit {proc.returncode}")
            rows = [json.loads(x) for x in (out / "episode_results.jsonl").read_text().splitlines()]
            seeds = [json.loads(x) for x in (out / "seed_manifest.jsonl").read_text().splitlines()]
            expected = {(i, j) for i in range(lane["num_runs"]) for j in range(lane["num_envs"])}
            # Keep batch run and environment index separate; episode is flattened.
            actual = {(r["run"], r["env_id"]) for r in rows}
            if (
                len(rows) != len(expected)
                or len(seeds) != len(expected)
                or actual != expected
                or {(r["run"], r["env_id"]) for r in seeds} != expected
            ):
                raise ValueError("Episode/seed coverage mismatch")
            trace = {}
            with (out / "requests.jsonl").open() as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (row["run"], row["env_id"])
                    if key in expected:
                        trace.setdefault(key, []).append(row)
            for key in expected:
                events = trace.get(key, [])
                ticks = [r["sim_tick"] for r in events if r["event"] == "action"]
                sent = [r["request_id"] for r in events if r["event"] == "request"]
                received = [r["request_id"] for r in events if r["event"] == "response"]
                if not ticks or ticks != list(range(len(ticks))) or sorted(sent) != sorted(received):
                    raise ValueError(f"Trace continuity/request pairing failed: {key}")
            elapsed = time.monotonic() - begin
            publish(
                status="completed",
                episodes=len(rows),
                successes=sum(r["success"] for r in rows),
                elapsed_s=elapsed,
                episodes_per_hour=3600 * len(rows) / elapsed,
            )
            return state
        except BaseException as exc:
            publish(status="failed", error=repr(exc))
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    results = []
    with ThreadPoolExecutor(max_workers=len(cfg["lanes"])) as pool:
        futures = {pool.submit(launch, l): l for l in cfg["lanes"]}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(dict(lane=futures[future]["name"], status="failed", error=repr(exc)))
    elapsed = time.monotonic() - started
    summary = dict(
        lanes=results,
        elapsed_s=elapsed,
        episodes=sum(x.get("episodes", 0) for x in results),
        status="completed" if all(x["status"] == "completed" for x in results) else "failed",
    )
    summary["episodes_per_hour"] = summary["episodes"] * 3600 / elapsed
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.exit(0 if run(json.loads(args.config.read_text()), args.output.resolve())["status"] == "completed" else 1)
