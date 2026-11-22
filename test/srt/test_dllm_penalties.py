"""CPU tests for dLLM penalties (python/sglang/srt/dllm/penalties.py).

  python -m pytest test/srt/test_dllm_penalties.py   (or run directly)
"""

import importlib.util
import os
import sys
import unittest

import torch

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python", "sglang", "srt", "dllm", "penalties.py")
_spec = importlib.util.spec_from_file_location("dllm_penalties", _PATH)
penalties = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(penalties)
DllmPenalties = penalties.DllmPenalties


class TestDllmPenalties(unittest.TestCase):
    def logits(self):
        # Two block positions over a 6-token vocabulary.
        return torch.tensor([[2.0, -2.0, 1.0, 0.5, 0.0, 3.0], [2.0, -2.0, 1.0, 0.5, 0.0, 3.0]])

    def test_no_penalty_means_no_work(self):
        self.assertIsNone(DllmPenalties.from_token_lists([[1, 2]], [(1.0, 0.0, 0.0)], "cpu"))
        self.assertIsNone(DllmPenalties.from_token_lists([[]], [(2.0, 0.0, 0.0)], "cpu"))

    def test_repetition_divides_positive_and_multiplies_negative(self):
        p = DllmPenalties.from_token_lists([[0, 1, 0]], [(2.0, 0.0, 0.0)], "cpu")
        x = self.logits()
        p.apply(0, x)
        self.assertEqual(x[0].tolist(), [1.0, -4.0, 1.0, 0.5, 0.0, 3.0])
        self.assertTrue(torch.equal(x[0], x[1]), "every position in the block is penalized")

    def test_frequency_counts_occurrences_and_presence_counts_once(self):
        p = DllmPenalties.from_token_lists([[5, 5, 5, 2]], [(1.0, 0.5, 1.0)], "cpu")
        x = self.logits()
        p.apply(0, x)
        self.assertAlmostEqual(x[0, 5].item(), 3.0 - 0.5 * 3 - 1.0)
        self.assertAlmostEqual(x[0, 2].item(), 1.0 - 0.5 * 1 - 1.0)
        self.assertEqual(x[0, 3].item(), 0.5, "unseen tokens are untouched")

    def test_a_looping_token_stops_being_the_argmax(self):
        # The loop: token 5 keeps winning. After enough repeats, it no longer does.
        p = DllmPenalties.from_token_lists([[5] * 6], [(1.0, 0.75, 2.0)], "cpu")
        x = self.logits()
        p.apply(0, x)
        self.assertNotEqual(int(x[0].argmax()), 5)

    def test_rows_are_per_request(self):
        p = DllmPenalties.from_token_lists([[5], [0]], [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], "cpu")
        a, b = self.logits(), self.logits()
        p.apply(0, a)
        p.apply(1, b)
        self.assertTrue(torch.equal(a, self.logits()), "request 0 asked for nothing")
        self.assertEqual(b[0, 0].item(), 1.0)

    def test_bfloat16_logits_keep_their_dtype(self):
        p = DllmPenalties.from_token_lists([[5]], [(2.0, 0.0, 0.0)], "cpu")
        x = self.logits().to(torch.bfloat16)
        p.apply(0, x)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(x[0, 5].item(), 1.5)


if __name__ == "__main__":
    unittest.main()
