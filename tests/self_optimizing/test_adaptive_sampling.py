"""Adaptive sampling tests."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.adaptive_sampling import (
    AdaptiveSamplingConfig,
    adaptive_temperature,
    compute_confidence_from_hits,
    select_best_candidate,
    sequence_logprob,
)


class TestConfidence(unittest.TestCase):
    def test_empty_list_returns_zero(self):
        self.assertEqual(compute_confidence_from_hits([]), 0.0)

    def test_all_hits_returns_one(self):
        self.assertEqual(compute_confidence_from_hits([True, True, True]), 1.0)

    def test_all_misses_returns_zero(self):
        self.assertEqual(compute_confidence_from_hits([False, False]), 0.0)

    def test_mixed(self):
        self.assertAlmostEqual(
            compute_confidence_from_hits([True, False, True, False]), 0.5
        )


class TestAdaptiveTemperature(unittest.TestCase):
    def test_confident_library_gives_base_temp(self):
        cfg = AdaptiveSamplingConfig(temperature_base=0.0, temperature_boost=0.8)
        self.assertEqual(adaptive_temperature(1.0, config=cfg), 0.0)

    def test_full_miss_gives_max_temp(self):
        cfg = AdaptiveSamplingConfig(temperature_base=0.0, temperature_boost=0.8)
        self.assertAlmostEqual(adaptive_temperature(0.0, config=cfg), 0.8)

    def test_mid_confidence_interpolates(self):
        cfg = AdaptiveSamplingConfig(temperature_base=0.1, temperature_boost=0.6)
        # confidence=0.5 → 0.1 + 0.5 * 0.6 = 0.4
        self.assertAlmostEqual(adaptive_temperature(0.5, config=cfg), 0.4)


class TestSelectBest(unittest.TestCase):
    def test_picks_highest_logprob_without_verifier(self):
        candidates = [
            {"text": "a", "logprob": -3.0},
            {"text": "b", "logprob": -1.0},
            {"text": "c", "logprob": -5.0},
        ]
        best = select_best_candidate(candidates)
        self.assertEqual(best["text"], "b")

    def test_verifier_overrides_logprob(self):
        candidates = [
            {"text": "bad", "logprob": -0.1},
            {"text": "good", "logprob": -5.0},
            {"text": "ok", "logprob": -2.0},
        ]
        best = select_best_candidate(
            candidates,
            verifier_fn=lambda t: {"bad": 0.0, "good": 1.0, "ok": 0.5}[t],
        )
        self.assertEqual(best["text"], "good")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_best_candidate([])


class TestSequenceLogprob(unittest.TestCase):
    def test_mean_of_picked_logprobs(self):
        # 2 timesteps, vocab 3. Model assigns certain logprobs.
        logprobs = torch.tensor([
            [-0.1, -0.5, -2.0],   # token 0 gets -0.1
            [-0.3, -0.4, -1.5],   # token 1 gets -0.4
        ])
        generated = torch.tensor([0, 1])
        lp = sequence_logprob(logprobs, generated)
        self.assertAlmostEqual(lp, (-0.1 + -0.4) / 2.0, places=5)

    def test_empty_sequence_returns_zero(self):
        lp = sequence_logprob(torch.zeros(0, 5), torch.zeros(0, dtype=torch.long))
        self.assertEqual(lp, 0.0)


if __name__ == "__main__":
    unittest.main()
