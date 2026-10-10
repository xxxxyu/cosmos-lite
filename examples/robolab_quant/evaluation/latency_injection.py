import random


def injected_delay(base, jitter, seed, request):
    if base < 0 or jitter < 0:
        raise ValueError("Delay and jitter must be nonnegative")
    return base + random.Random(f"{seed}:{request}").uniform(0, jitter)
