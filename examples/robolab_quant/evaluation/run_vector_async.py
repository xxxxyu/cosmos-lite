"""RoboLab adapter: one connection, scheduler and delay stream per vector environment."""

import json
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["ROBOLAB_REPO"])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
if os.environ.get("ROBOLAB_WARP_PATH"):
    sys.path.insert(0, os.environ["ROBOLAB_WARP_PATH"])
    import warp

    print("Private Warp", warp.__version__, warp.__file__, flush=True)
from policies.cosmos3 import client as module

Original = module.Cosmos3Client


from vector_client import VectorClient


def make_client(*args, **kwargs):
    return VectorClient(Original, *args, **kwargs)


module.Cosmos3Client = make_client
from robolab.eval import runner

original_run = runner.run_evaluation


def seeded_evaluation(*args, **kwargs):
    from robolab.eval import episode

    original_episode = episode.run_episode

    def paired(**kw):
        seed = int(os.environ.get("ASYNC_SEED_OFFSET", "3000")) + int(kw["episode"])
        os.environ["ASYNC_BATCH_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        kw["env"].seed(seed)
        with Path(os.environ["ASYNC_TRACE"]).with_name("seed_manifest.jsonl").open("a") as f:
            for env_id in range(kw["env"].num_envs):
                f.write(
                    json.dumps(
                        dict(
                            run=kw["episode"],
                            env_id=env_id,
                            batch_seed=seed,
                            delay_seed=seed + env_id,
                            protocol="vector-v1",
                        )
                    )
                    + "\n"
                )
        return original_episode(**kw)

    episode.run_episode = paired
    return original_run(*args, **kwargs)


runner.run_evaluation = seeded_evaluation
runpy.run_path(str(REPO / "policies/cosmos3/run.py"), run_name="__main__")
