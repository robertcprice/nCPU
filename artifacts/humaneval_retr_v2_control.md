# HumanEval-lite Agent Loop Results

Generated 2026-04-19T01:37:32Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 39.7s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **29/30 (96.7%)**
- Path distribution:
  - miss: 1
  - sample: 29
- Total tokens: ~7039

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | add_two | ✓ | sample | s0 (T=0.0) | 3094 |  |
| 2 | double | ✓ | sample | s0 (T=0.0) | 1907 |  |
| 3 | triple | ✓ | sample | s0 (T=0.0) | 1417 |  |
| 4 | square | ✓ | sample | s0 (T=0.0) | 1279 |  |
| 5 | increment | ✓ | sample | s0 (T=0.0) | 652 |  |
| 6 | decrement | ✓ | sample | s0 (T=0.0) | 598 |  |
| 7 | abs_value | ✓ | sample | s0 (T=0.0) | 945 |  |
| 8 | negate | ✓ | sample | s0 (T=0.0) | 754 |  |
| 9 | max_of_two | ✓ | sample | s0 (T=0.0) | 782 |  |
| 10 | min_of_two | ✓ | sample | s0 (T=0.0) | 566 |  |
| 11 | sign | ✓ | sample | s0 (T=0.0) | 759 |  |
| 12 | is_even | ✓ | sample | s0 (T=0.0) | 1107 |  |
| 13 | is_odd | ✓ | sample | s0 (T=0.0) | 869 |  |
| 14 | sum_three | ✓ | sample | s0 (T=0.0) | 1180 |  |
| 15 | abs_diff | ✓ | sample | s0 (T=0.0) | 659 |  |
| 16 | multiply_add_one | ✓ | sample | s0 (T=0.0) | 1014 |  |
| 17 | sum_squares | ✓ | sample | s0 (T=0.0) | 590 |  |
| 18 | clamp_0_10 | ✓ | sample | s0 (T=0.0) | 888 |  |
| 19 | max_of_three | ✓ | sample | s0 (T=0.0) | 766 |  |
| 20 | double_plus_three | ✓ | sample | s0 (T=0.0) | 946 |  |
| 21 | pythagorean_sum | ✓ | sample | s0 (T=0.0) | 1145 |  |
| 22 | mod_5 | ✓ | sample | s0 (T=0.0) | 651 |  |
| 23 | safe_div_or_neg | ✓ | sample | s0 (T=0.0) | 852 |  |
| 24 | polynomial_2ax_plus_b | ✓ | sample | s0 (T=0.0) | 660 |  |
| 25 | sum_abs_three | ✓ | sample | s0 (T=0.0) | 697 |  |
| 26 | average_two | ✓ | sample | s0 (T=0.0) | 792 |  |
| 27 | count_down_3 | ✓ | sample | s0 (T=0.0) | 691 |  |
| 28 | scaled_add | ✗ | miss | k=3,r=2 | 8923 | no-code-found |
| 29 | is_positive | ✓ | sample | s0 (T=0.0) | 814 |  |
| 30 | offset_times_sign | ✓ | sample | s0 (T=0.0) | 3711 |  |
