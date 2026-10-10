import pathlib
import sys
import threading
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from chunk_prefetch import ChunkPrefetch


class Action(list):
    def copy(self):
        return Action(self)


class Tests(unittest.TestCase):
    def test_single_inflight_and_discrete_gripper(self):
        gate = threading.Event()
        calls = []

        def query(r):
            calls.append(r)
            if len(calls) > 1:
                gate.wait(2)
            return [Action([i, i % 2]) for i in range(32)]

        s = ChunkPrefetch(query, horizon=16, asynchronous=True)
        try:
            for _ in range(12):
                self.assertIn(s.next(lambda: "x")[-1], (0, 1))
            self.assertIsNotNone(s.future)
            self.assertEqual(s.request_id, 2)
        finally:
            gate.set()
            s.close()

    def test_expired_prefix(self):
        events = []
        s = ChunkPrefetch(lambda _: [Action([i, 0]) for i in range(32)], 16, True, events.append)
        try:
            s._start(None)
            s.tick = 5
            s._accept()
            self.assertEqual(s.next(lambda: None)[0], 5)
            s._start(None)
            s.tick = 40
            self.assertFalse(s._accept())
            self.assertEqual(events[-1]["event"], "discard")
        finally:
            s.close()

    def test_reset_discards_pending(self):
        s = ChunkPrefetch(lambda _: [Action([99, 1]) for i in range(32)], 16, True)
        s._start(None)
        s.reset()
        self.assertIsNone(s.future)
        self.assertIsNone(s.chunk)
        self.assertEqual(s.tick, 0)
        self.assertEqual(s.episode, 1)
        s.close()

    def test_explicit_modes_have_distinct_time_alignment(self):
        for mode, expected in [("async", 0), ("timestamp", 5)]:
            scheduler = ChunkPrefetch(lambda _: [Action([index, 1]) for index in range(16)], 16, mode=mode)
            try:
                scheduler._start(None)
                scheduler.tick = 5
                scheduler._accept()
                self.assertEqual(scheduler.next(lambda: None)[0], expected)
            finally:
                scheduler.close()

    def test_blend_aligns_suffix_and_preserves_gripper(self):
        scheduler = ChunkPrefetch(lambda _: [Action([index, 1]) for index in range(16)], 16, mode="blend")
        try:
            scheduler.chunk = [Action([100, 0]) for index in range(16)]
            scheduler._start(None)
            scheduler.tick = 5
            scheduler._accept()
            self.assertAlmostEqual(scheduler.chunk[0][0], 81)
            self.assertEqual(scheduler.chunk[0][-1], 1)
            self.assertEqual(len(scheduler.chunk), 11)
            self.assertEqual(scheduler.chunk[4][0], 9)
        finally:
            scheduler.close()


if __name__ == "__main__":
    unittest.main()
