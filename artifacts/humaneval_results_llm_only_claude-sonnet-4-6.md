# HumanEval-lite LLM-only Results

Generated 2026-04-18T22:48:14Z — 30 problems, 40.7s total, model=claude-sonnet-4-6.

## Summary

- **Pass@1**: **27/30 (90.0%)**

## Per-problem

| # | problem | pass | ms | notes |
|--:|---------|:----:|---:|-------|
| 1 | add_two | ✓ | 1430 |  |
| 2 | double | ✓ | 769 |  |
| 3 | triple | ✓ | 686 |  |
| 4 | square | ✓ | 806 |  |
| 5 | increment | ✓ | 940 |  |
| 6 | decrement | ✓ | 836 |  |
| 7 | abs_value | ✓ | 1059 |  |
| 8 | negate | ✓ | 1452 |  |
| 9 | max_of_two | ✓ | 1609 |  |
| 10 | min_of_two | ✓ | 1297 |  |
| 11 | sign | ✓ | 1243 |  |
| 12 | is_even | ✓ | 1333 |  |
| 13 | is_odd | ✓ | 1127 |  |
| 14 | sum_three | ✓ | 1028 |  |
| 15 | abs_diff | ✓ | 1017 |  |
| 16 | multiply_add_one | ✓ | 1604 |  |
| 17 | sum_squares | ✓ | 1033 |  |
| 18 | clamp_0_10 | ✓ | 1128 |  |
| 19 | max_of_three | ✓ | 1292 |  |
| 20 | double_plus_three | ✓ | 1125 |  |
| 21 | pythagorean_sum | ✓ | 1101 |  |
| 22 | mod_5 | ✓ | 1002 |  |
| 23 | safe_div_or_neg | ✓ | 1493 |  |
| 24 | polynomial_2ax_plus_b | ✗ | 1376 | wrong [1, 10, 5]→25 exp 15 |
| 25 | sum_abs_diffs | ✗ | 1152 | wrong [10, 5, 0]→20 exp 15 |
| 26 | average_two | ✓ | 1099 |  |
| 27 | count_down_3 | ✓ | 955 |  |
| 28 | scaled_add | ✗ | 7850 | exec: SyntaxError("invalid character '≠' (U+2260)", ('<string>', 5, 34, '- scale |
| 29 | is_positive | ✓ | 1056 |  |
| 30 | offset_times_sign | ✓ | 820 |  |
