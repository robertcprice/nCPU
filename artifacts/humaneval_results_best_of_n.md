# HumanEval-lite Best-of-N Results

Generated 2026-04-18T22:59:31Z — 30 problems, k=5, model=claude-haiku-4-5-20251001, 28.9s total.

## Summary

- **Pass@1 (best-of-5)**: **27/30 (90.0%)**
- Cache hits (0 ms): 0
- Total tokens: ~7788

### Winning-sample distribution

Which sample-index produced the winner when pass@1=True?

| sample | count |
|-------:|------:|
| 0 (T=0.0) | 27 |

## Per-problem

| # | problem | pass | winner | ms | notes |
|--:|---------|:----:|:------:|---:|-------|
| 1 | add_two | ✓ | 0 | 838 |  |
| 2 | double | ✓ | 0 | 766 |  |
| 3 | triple | ✓ | 0 | 881 |  |
| 4 | square | ✓ | 0 | 644 |  |
| 5 | increment | ✓ | 0 | 684 |  |
| 6 | decrement | ✓ | 0 | 1125 |  |
| 7 | abs_value | ✓ | 0 | 918 |  |
| 8 | negate | ✓ | 0 | 864 |  |
| 9 | max_of_two | ✓ | 0 | 1302 |  |
| 10 | min_of_two | ✓ | 0 | 691 |  |
| 11 | sign | ✓ | 0 | 1204 |  |
| 12 | is_even | ✓ | 0 | 1296 |  |
| 13 | is_odd | ✓ | 0 | 1074 |  |
| 14 | sum_three | ✓ | 0 | 1703 |  |
| 15 | abs_diff | ✓ | 0 | 866 |  |
| 16 | multiply_add_one | ✓ | 0 | 924 |  |
| 17 | sum_squares | ✓ | 0 | 832 |  |
| 18 | clamp_0_10 | ✓ | 0 | 757 |  |
| 19 | max_of_three | ✓ | 0 | 649 |  |
| 20 | double_plus_three | ✓ | 0 | 937 |  |
| 21 | pythagorean_sum | ✓ | 0 | 1289 |  |
| 22 | mod_5 | ✓ | 0 | 1115 |  |
| 23 | safe_div_or_neg | ✓ | 0 | 1113 |  |
| 24 | polynomial_2ax_plus_b | ✗ | - | 693 | wrong [1, 10, 5]→25 exp 15 |
| 25 | sum_abs_diffs | ✗ | - | 1084 | wrong [10, 5, 0]→20 exp 15 |
| 26 | average_two | ✓ | 0 | 1017 |  |
| 27 | count_down_3 | ✓ | 0 | 695 |  |
| 28 | scaled_add | ✗ | - | 1310 | wrong [2, 3]→13 exp 7 |
| 29 | is_positive | ✓ | 0 | 726 |  |
| 30 | offset_times_sign | ✓ | 0 | 839 |  |
