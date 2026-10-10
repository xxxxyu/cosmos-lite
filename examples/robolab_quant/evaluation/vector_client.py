"""Independent async state for each RoboLab vector environment."""

import json
import os
import threading
import time

import numpy as np
from chunk_prefetch import ChunkPrefetch
from latency_injection import injected_delay


class VectorClient:
    def __init__(self, backend_factory, *args, **kwargs):
        self.backend_factory = backend_factory
        self.args = args
        self.kwargs = dict(kwargs, enable_visualization=False)
        self.states = {}
        self.run = 0
        self.log = open(os.environ["ASYNC_TRACE"], "a", buffering=1)
        self.lock = threading.Lock()

    def emit(self, env_id, row):
        with self.lock:
            self.log.write(json.dumps(dict(row, env_id=env_id, run=self.run)) + "\n")

    def state(self, env_id):
        if env_id not in self.states:
            client = self.backend_factory(*self.args, **self.kwargs)
            state = {"client": client, "request_index": 0}

            def query(request):
                start = time.perf_counter()
                result = client._infer_with_retry(request)
                self.emit(env_id, dict(event="transport", transport_latency_s=time.perf_counter() - start))
                # Batch seed + env index identifies a stream; this is not a claim
                # that vector reset equals a serial run with seed+env_id.
                seed = int(os.environ["ASYNC_BATCH_SEED"]) + env_id
                delay = injected_delay(
                    float(os.environ.get("ASYNC_DELAY", "0")),
                    float(os.environ.get("ASYNC_JITTER", "0")),
                    seed,
                    state["request_index"],
                )
                state["request_index"] += 1
                self.emit(env_id, dict(event="injected_delay", seed=seed, delay_s=delay))
                time.sleep(delay)
                return client._postprocess_chunk(client._unpack_response(result))

            state["scheduler"] = ChunkPrefetch(
                query,
                horizon=int(os.environ.get("ASYNC_HORIZON", "16")),
                mode=os.environ.get("ASYNC_MODE", "sync"),
                emit=lambda row: self.emit(env_id, row),
            )
            self.states[env_id] = state
        return self.states[env_id]

    def infer(self, obs, instruction, *, env_id=0):
        state = self.state(int(env_id))
        client = state["client"]
        start = time.perf_counter()

        def prepare():
            request = client._pack_request(client._extract_observation(obs, env_id=env_id), instruction)
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in request.items()}

        action = state["scheduler"].next(prepare)
        return {
            "action": action,
            "viz": None,
            "_cosmos_timing": {"client_total_s": time.perf_counter() - start, "query_s": None, "refreshed": False},
        }

    def reset(self, *, env_id=None):
        for key, state in self.states.items():
            if env_id is None or key == env_id:
                state["scheduler"].reset()
                state["request_index"] = 0
                state["client"].reset()
        if env_id is None:
            self.run += 1

    def close(self):
        for state in self.states.values():
            state["scheduler"].close()
        self.log.close()
