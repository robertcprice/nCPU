# HumanEval-lite Results

Generated 2026-04-18T20:57:19Z — 30 problems, total runtime 338.3s.

## Summary

- **Synthesis success rate**: 17/30 (56.7%)
- **Pass@1 (code runs + passes all test cases)**: **15/30 (50.0%)**

## Per-problem results

| # | problem | synth | pass@1 | ms | method | notes |
|--:|---------|:-----:|:------:|---:|--------|-------|
| 1 | add_two | ✓ | ✓ | 215 | search_scalar_expr |  |
| 2 | double | ✓ | ✓ | 6 | enumerative |  |
| 3 | triple | ✓ | ✓ | 5 | enumerative |  |
| 4 | square | ✓ | ✓ | 6 | enumerative |  |
| 5 | increment | ✓ | ✓ | 5 | enumerative |  |
| 6 | decrement | ✓ | ✓ | 5 | enumerative |  |
| 7 | abs_value | ✗ | ✗ | 24996 | TIMEOUT | synth: TIMEOUT |
| 8 | negate | ✓ | ✓ | 12 | enumerative |  |
| 9 | max_of_two | ✓ | ✗ | 18 | search_max2_formula | exec-error: SyntaxError("unmatched '}'", ('<string>', 4, 9,  |
| 10 | min_of_two | ✗ | ✗ | 24998 | TIMEOUT | synth: TIMEOUT |
| 11 | sign | ✗ | ✗ | 25004 | TIMEOUT | synth: TIMEOUT |
| 12 | is_even | ✓ | ✓ | 12608 | synth_gradient |  |
| 13 | is_odd | ✗ | ✗ | 25004 | TIMEOUT | synth: TIMEOUT |
| 14 | sum_three | ✓ | ✓ | 25 | search_scalar_expr |  |
| 15 | abs_diff | ✗ | ✗ | 25003 | TIMEOUT | synth: TIMEOUT |
| 16 | multiply_add_one | ✓ | ✓ | 5 | search_polynomial_quadratic |  |
| 17 | sum_squares | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 18 | clamp_0_10 | ✗ | ✗ | 25017 | TIMEOUT | synth: TIMEOUT |
| 19 | max_of_three | ✗ | ✗ | 25003 | TIMEOUT | synth: TIMEOUT |
| 20 | double_plus_three | ✓ | ✓ | 4 | search_polynomial_quadratic |  |
| 21 | pythagorean_sum | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 22 | mod_5 | ✓ | ✓ | 5 | enumerative |  |
| 23 | safe_div_or_neg | ✗ | ✗ | 25006 | TIMEOUT | synth: TIMEOUT |
| 24 | polynomial_2ax_plus_b | ✗ | ✗ | 25003 | TIMEOUT | synth: TIMEOUT |
| 25 | sum_abs_diffs | ✓ | ✗ | 313 | search_two_branch | wrong: got 0.3, expected 15 |
| 26 | average_two | ✓ | ✓ | 19 | search_scalar_expr |  |
| 27 | count_down_3 | ✓ | ✓ | 5 | enumerative |  |
| 28 | scaled_add | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 29 | is_positive | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 30 | offset_times_sign | ✓ | ✓ | 8 | enumerative |  |
