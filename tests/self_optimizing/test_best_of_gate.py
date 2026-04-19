"""Best-of-N over gate tests."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing.best_of_gate import (
    BestOfGateConfig,
    BestOfGateResult,
    select_best_over_gates,
)


class TestSelectBestOverGates(unittest.TestCase):
    def test_verifier_picks_best_score(self):
        candidates = [
            {"text": "a", "gate": 0.0, "logprob": -2.0},
            {"text": "b", "gate": 0.02, "logprob": -0.5},
            {"text": "c", "gate": 0.05, "logprob": -5.0},
        ]
        scores = {"a": 0.3, "b": 0.9, "c": 0.6}
        r = select_best_over_gates(candidates, verifier_fn=lambda t: scores[t])
        self.assertEqual(r.selected_text, "b")
        self.assertEqual(r.selected_gate, 0.02)

    def test_logprob_fallback_when_no_verifier(self):
        candidates = [
            {"text": "a", "gate": 0.0, "logprob": -2.0},
            {"text": "b", "gate": 0.05, "logprob": -0.3},
        ]
        r = select_best_over_gates(candidates)
        self.assertEqual(r.selected_text, "b")

    def test_tie_prefers_lower_gate(self):
        candidates = [
            {"text": "a", "gate": 0.05, "logprob": -1.0},
            {"text": "b", "gate": 0.00, "logprob": -1.0},
            {"text": "c", "gate": 0.02, "logprob": -1.0},
        ]
        r = select_best_over_gates(candidates)
        self.assertEqual(r.selected_gate, 0.00)
        self.assertEqual(r.selected_text, "b")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_best_over_gates([])

    def test_config_defaults(self):
        cfg = BestOfGateConfig()
        self.assertIn(0.0, cfg.gate_values)
        self.assertTrue(cfg.force_include_baseline)


if __name__ == "__main__":
    unittest.main()
