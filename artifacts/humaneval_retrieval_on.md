# HumanEval-lite Agent Loop Results

Generated 2026-04-19T01:34:50Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 41.9s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **29/30 (96.7%)**
- Path distribution:
  - miss: 1
  - sample: 29
- Total tokens: ~7039

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | add_two | ✓ | sample | s0 (T=0.0) | 1607 |  |
| 2 | double | ✓ | sample | s0 (T=0.0) | 967 |  |
| 3 | triple | ✓ | sample | s0 (T=0.0) | 780 |  |
| 4 | square | ✓ | sample | s0 (T=0.0) | 712 |  |
| 5 | increment | ✓ | sample | s0 (T=0.0) | 1119 |  |
| 6 | decrement | ✓ | sample | s0 (T=0.0) | 799 |  |
| 7 | abs_value | ✓ | sample | s0 (T=0.0) | 731 |  |
| 8 | negate | ✓ | sample | s0 (T=0.0) | 613 |  |
| 9 | max_of_two | ✓ | sample | s0 (T=0.0) | 688 |  |
| 10 | min_of_two | ✓ | sample | s0 (T=0.0) | 671 |  |
| 11 | sign | ✓ | sample | s0 (T=0.0) | 703 |  |
| 12 | is_even | ✓ | sample | s0 (T=0.0) | 648 |  |
| 13 | is_odd | ✓ | sample | s0 (T=0.0) | 1086 |  |
| 14 | sum_three | ✓ | sample | s0 (T=0.0) | 764 |  |
| 15 | abs_diff | ✓ | sample | s0 (T=0.0) | 789 |  |
| 16 | multiply_add_one | ✓ | sample | s0 (T=0.0) | 1769 |  |
| 17 | sum_squares | ✓ | sample | s0 (T=0.0) | 885 |  |
| 18 | clamp_0_10 | ✓ | sample | s0 (T=0.0) | 5340 |  |
| 19 | max_of_three | ✓ | sample | s0 (T=0.0) | 798 |  |
| 20 | double_plus_three | ✓ | sample | s0 (T=0.0) | 1339 |  |
| 21 | pythagorean_sum | ✓ | sample | s0 (T=0.0) | 883 |  |
| 22 | mod_5 | ✓ | sample | s0 (T=0.0) | 836 |  |
| 23 | safe_div_or_neg | ✓ | sample | s0 (T=0.0) | 1477 |  |
| 24 | polynomial_2ax_plus_b | ✓ | sample | s0 (T=0.0) | 1684 |  |
| 25 | sum_abs_three | ✓ | sample | s0 (T=0.0) | 750 |  |
| 26 | average_two | ✓ | sample | s0 (T=0.0) | 815 |  |
| 27 | count_down_3 | ✓ | sample | s0 (T=0.0) | 789 |  |
| 28 | scaled_add | ✗ | miss | k=3,r=2 | 9561 | no-code-found |
| 29 | is_positive | ✓ | sample | s0 (T=0.0) | 1491 |  |
| 30 | offset_times_sign | ✓ | sample | s0 (T=0.0) | 826 |  |
