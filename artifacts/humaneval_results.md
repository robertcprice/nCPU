# HumanEval-lite Results

Generated 2026-04-18T21:03:49Z — 30 problems, total runtime 371.9s.

## Summary

- **Synthesis success rate**: 16/30 (53.3%)
- **Pass@1 (code runs + passes all test cases)**: **15/30 (50.0%)**

## Per-problem results

| # | problem | synth | pass@1 | ms | method | notes |
|--:|---------|:-----:|:------:|---:|--------|-------|
| 1 | add_two | ✓ | ✓ | 164 | enumerative |  |
| 2 | double | ✓ | ✓ | 6 | enumerative |  |
| 3 | triple | ✓ | ✓ | 5 | enumerative |  |
| 4 | square | ✓ | ✓ | 5 | enumerative |  |
| 5 | increment | ✓ | ✓ | 5 | enumerative |  |
| 6 | decrement | ✓ | ✓ | 5 | enumerative |  |
| 7 | abs_value | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 8 | negate | ✓ | ✓ | 10 | enumerative |  |
| 9 | max_of_two | ✗ | ✗ | 25006 | TIMEOUT | synth: TIMEOUT |
| 10 | min_of_two | ✗ | ✗ | 25013 | TIMEOUT | synth: TIMEOUT |
| 11 | sign | ✗ | ✗ | 25012 | TIMEOUT | synth: TIMEOUT |
| 12 | is_even | ✓ | ✓ | 21065 | synth_gradient |  |
| 13 | is_odd | ✗ | ✗ | 25055 | TIMEOUT | synth: TIMEOUT |
| 14 | sum_three | ✓ | ✓ | 19 | enumerative |  |
| 15 | abs_diff | ✗ | ✗ | 25021 | TIMEOUT | synth: TIMEOUT |
| 16 | multiply_add_one | ✓ | ✓ | 9 | search_polynomial_quadratic |  |
| 17 | sum_squares | ✗ | ✗ | 25014 | TIMEOUT | synth: TIMEOUT |
| 18 | clamp_0_10 | ✗ | ✗ | 25038 | TIMEOUT | synth: TIMEOUT |
| 19 | max_of_three | ✗ | ✗ | 25006 | TIMEOUT | synth: TIMEOUT |
| 20 | double_plus_three | ✓ | ✓ | 9 | search_polynomial_quadratic |  |
| 21 | pythagorean_sum | ✗ | ✗ | 25005 | TIMEOUT | synth: TIMEOUT |
| 22 | mod_5 | ✓ | ✓ | 11 | enumerative |  |
| 23 | safe_div_or_neg | ✗ | ✗ | 25017 | TIMEOUT | synth: TIMEOUT |
| 24 | polynomial_2ax_plus_b | ✗ | ✗ | 25007 | TIMEOUT | synth: TIMEOUT |
| 25 | sum_abs_diffs | ✓ | ✗ | 332 | search_two_branch | wrong: got 0.3, expected 15 |
| 26 | average_two | ✓ | ✓ | 27 | search_scalar_expr |  |
| 27 | count_down_3 | ✓ | ✓ | 10 | enumerative |  |
| 28 | scaled_add | ✗ | ✗ | 25008 | TIMEOUT | synth: TIMEOUT |
| 29 | is_positive | ✗ | ✗ | 25010 | TIMEOUT | synth: TIMEOUT |
| 30 | offset_times_sign | ✓ | ✓ | 9 | enumerative |  |
