import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from run_topology import validate
from vector_client import VectorClient


class Backend:
    instances = []

    def __init__(self, **kwargs):
        self.requests = 0
        self.instances.append(self)

    def _extract_observation(self, obs, *, env_id):
        return obs[env_id]

    def _pack_request(self, obs, instruction):
        return {"value": obs}

    def _infer_with_retry(self, request):
        self.requests += 1
        return np.tile(request["value"], (16, 1))

    def _unpack_response(self, response):
        return response

    def _postprocess_chunk(self, response):
        return response

    def reset(self):
        pass


class VectorTests(unittest.TestCase):
    def test_env_queues_and_partial_reset_do_not_cross(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "ASYNC_TRACE": str(Path(tmp) / "trace.jsonl"),
                    "ASYNC_BATCH_SEED": "9000",
                    "ASYNC_MODE": "sync",
                    "ASYNC_DELAY": "0",
                    "ASYNC_JITTER": "0",
                    "ASYNC_HORIZON": "16",
                },
            ),
        ):
            client = VectorClient(Backend)
            obs = np.array([[1.0, 0.0], [8.0, 1.0]])
            for _ in range(3):
                np.testing.assert_array_equal(client.infer(obs, "test", env_id=1)["action"], obs[1])
            np.testing.assert_array_equal(client.infer(obs, "test", env_id=0)["action"], obs[0])
            self.assertEqual(client.states[1]["scheduler"].tick, 3)
            client.reset(env_id=0)
            self.assertEqual(client.states[1]["scheduler"].tick, 3)
            self.assertEqual(client.states[0]["scheduler"].tick, 0)
            self.assertIsNot(client.states[0]["client"], client.states[1]["client"])
            client.close()
            rows = [json.loads(x) for x in (Path(tmp) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual({r["env_id"] for r in rows}, {0, 1})

    def test_topology_rejects_gpu_overlap(self):
        lane = dict(name="a", sim_gpu=0, num_envs=2, num_runs=1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate(dict(policy_gpus=[0], lanes=[lane]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate(dict(policy_gpus=[1], lanes=[lane, dict(lane, name="b")]))


if __name__ == "__main__":
    unittest.main()
