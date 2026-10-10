"""One-inflight chunk scheduling indexed by completed simulator control steps."""

import time
from concurrent.futures import ThreadPoolExecutor


class ChunkPrefetch:
    def __init__(self, query, horizon=32, asynchronous=False, emit=lambda x: None, mode=None):
        self.query = query
        self.horizon = horizon
        self.mode = mode or ("timestamp" if asynchronous else "sync")
        self.asynchronous = self.mode != "sync"
        self.emit = emit
        if self.mode not in ("sync", "async", "timestamp", "blend"):
            raise ValueError(self.mode)
        self.observation_origin = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.tick = 0
        self.episode = 0
        self.chunk = None
        self.origin = 0
        self.request_id = 0

    def _start(self, request):
        assert self.future is None, "single-inflight invariant"
        origin = self.tick
        episode = self.episode
        rid = self.request_id
        self.request_id += 1
        sent = time.perf_counter()
        self.emit(dict(event="request", episode=episode, request_id=rid, sim_tick=origin, wall_s=sent))

        def work():
            chunk = self.query(request)
            returned = time.perf_counter()
            self.emit(
                dict(
                    event="response",
                    episode=episode,
                    request_id=rid,
                    request_sim_tick=origin,
                    return_sim_tick=self.tick,
                    wall_s=returned,
                    request_latency_s=returned - sent,
                )
            )
            return chunk, origin, episode, rid

        self.future = self.pool.submit(work)

    def _accept(self):
        chunk, origin, episode, rid = self.future.result()
        self.future = None
        age = self.tick - origin
        skipped = age if self.mode in ("timestamp", "blend") else 0
        valid = episode == self.episode and skipped < min(self.horizon, len(chunk))
        self.emit(
            dict(
                event="accept" if valid else "discard",
                episode=self.episode,
                request_id=rid,
                sim_tick=self.tick,
                wall_s=time.perf_counter(),
                expired_prefix_steps=skipped,
                observation_age_steps=age,
                mode=self.mode,
            )
        )
        if valid:
            incoming = [action.copy() for action in chunk[skipped : self.horizon]]
            if self.mode == "blend" and self.chunk is not None:
                remaining = self.chunk[self.tick - self.origin :]
                overlap = min(4, len(remaining), len(incoming))
                for offset in range(overlap):
                    weight = (offset + 1) / (overlap + 1)
                    for joint in range(len(incoming[offset]) - 1):
                        incoming[offset][joint] = (1 - weight) * remaining[offset][joint] + weight * incoming[offset][
                            joint
                        ]
            self.chunk = incoming
            self.origin = self.tick
            self.observation_origin = origin
        return valid

    def next(self, prepare):
        start = time.perf_counter()
        if self.future is not None and self.future.done():
            self._accept()
        remaining = 0 if self.chunk is None else len(self.chunk) - (self.tick - self.origin)
        if remaining <= 0:
            self.emit(dict(event="starvation", episode=self.episode, sim_tick=self.tick, wall_s=start))
            if self.future is None:
                self._start(prepare())
            if not self._accept():
                self._start(prepare())
                assert self._accept()
        remaining = len(self.chunk) - (self.tick - self.origin)
        if self.asynchronous and self.future is None and remaining <= max(1, self.horizon // 2):
            self._start(prepare())
        action = self.chunk[self.tick - self.origin].copy()
        self.emit(
            dict(
                event="action",
                episode=self.episode,
                sim_tick=self.tick,
                wall_s=time.perf_counter(),
                observation_age_steps=self.tick - self.observation_origin,
                wait_and_prepare_s=time.perf_counter() - start,
                gripper=float(action[-1]),
            )
        )
        self.tick += 1
        return action

    def reset(self):
        # Drain the sole connection before a new episode; never carry a stale result.
        if self.future is not None:
            self.future.result()
            self.future = None
        self.episode += 1
        self.tick = 0
        self.chunk = None
        self.emit(dict(event="reset", episode=self.episode, wall_s=time.perf_counter()))

    def close(self):
        self.reset()
        self.pool.shutdown(wait=True)
