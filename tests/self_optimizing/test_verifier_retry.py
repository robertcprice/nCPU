"""Verifier-retry loop tests."""

from __future__ import annotations

import unittest

from ncpu.self_optimizing.verifier_retry import (
    RetryConfig,
    RetryStrategy,
    retry_until_verified,
)


class TestRetryLoop(unittest.TestCase):
    def test_first_attempt_passes(self):
        calls = []
        def gen(s):
            calls.append(s.label)
            return "solution"
        def ver(t):
            return (True, 1.0, None)
        result = retry_until_verified(generate_fn=gen, verify_fn=ver)
        self.assertTrue(result.final_passed)
        self.assertEqual(result.total_attempts, 1)
        self.assertEqual(result.winning_attempt_index, 0)
        self.assertEqual(calls, ["baseline-greedy"])

    def test_retries_until_pass(self):
        # Fail first 2 attempts, pass the 3rd.
        attempt_count = [0]
        def gen(s):
            attempt_count[0] += 1
            return f"attempt-{attempt_count[0]}"
        def ver(t):
            if "3" in t:
                return (True, 1.0, None)
            return (False, 0.1, "syntax error")
        result = retry_until_verified(generate_fn=gen, verify_fn=ver)
        self.assertTrue(result.final_passed)
        self.assertEqual(result.total_attempts, 3)
        self.assertEqual(result.winning_attempt_index, 2)

    def test_none_pass_returns_best_score(self):
        # All fail, but with different scores. Should return highest score.
        scores = [0.1, 0.5, 0.3, 0.2, 0.4]
        idx = [0]
        def gen(s):
            out = f"a{idx[0]}"
            idx[0] += 1
            return out
        def ver(t):
            i = int(t[1:])
            return (False, scores[i], "fail")
        result = retry_until_verified(generate_fn=gen, verify_fn=ver)
        self.assertFalse(result.final_passed)
        self.assertEqual(result.total_attempts, 5)
        # Best score is index 1 (0.5). Final text should be "a1".
        self.assertEqual(result.final_text, "a1")

    def test_custom_strategies(self):
        cfg = RetryConfig(strategies=[
            RetryStrategy(gate=0.0, temperature=0.0, label="only-one"),
        ])
        def gen(s): return "x"
        def ver(t): return (False, 0.5, None)
        result = retry_until_verified(generate_fn=gen, verify_fn=ver, config=cfg)
        self.assertEqual(result.total_attempts, 1)

    def test_stop_on_first_pass_false(self):
        cfg = RetryConfig(stop_on_first_pass=False)
        count = [0]
        def gen(s):
            count[0] += 1
            return f"a{count[0]}"
        def ver(t): return (True, 1.0, None)
        result = retry_until_verified(generate_fn=gen, verify_fn=ver, config=cfg)
        # Should run all 5 strategies even though first passes.
        self.assertEqual(result.total_attempts, 5)


if __name__ == "__main__":
    unittest.main()
