# HumanEval-lite Agent Loop Results

Generated 2026-04-19T01:37:25Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 38.1s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **29/30 (96.7%)**
- Path distribution:
  - miss: 1
  - sample: 29
- Total tokens: ~10173

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | add_two | ✓ | sample | s0 (T=0.0) | 845 |  |
| 2 | double | ✓ | sample | s0 (T=0.0) | 776 |  |
| 3 | triple | ✓ | sample | s0 (T=0.0) | 602 |  |
| 4 | square | ✓ | sample | s0 (T=0.0) | 772 |  |
| 5 | increment | ✓ | sample | s0 (T=0.0) | 1971 |  |
| 6 | decrement | ✓ | sample | s0 (T=0.0) | 619 |  |
| 7 | abs_value | ✓ | sample | s0 (T=0.0) | 729 |  |
| 8 | negate | ✓ | sample | s0 (T=0.0) | 1006 |  |
| 9 | max_of_two | ✓ | sample | s0 (T=0.0) | 675 |  |
| 10 | min_of_two | ✓ | sample | s0 (T=0.0) | 2171 |  |
| 11 | sign | ✓ | sample | s0 (T=0.0) | 1029 |  |
| 12 | is_even | ✓ | sample | s0 (T=0.0) | 699 |  |
| 13 | is_odd | ✓ | sample | s0 (T=0.0) | 880 |  |
| 14 | sum_three | ✓ | sample | s0 (T=0.0) | 651 |  |
| 15 | abs_diff | ✓ | sample | s0 (T=0.0) | 1881 |  |
| 16 | multiply_add_one | ✓ | sample | s0 (T=0.0) | 847 |  |
| 17 | sum_squares | ✓ | sample | s0 (T=0.0) | 717 |  |
| 18 | clamp_0_10 | ✓ | sample | s0 (T=0.0) | 3545 |  |
| 19 | max_of_three | ✓ | sample | s0 (T=0.0) | 907 |  |
| 20 | double_plus_three | ✓ | sample | s0 (T=0.0) | 695 |  |
| 21 | pythagorean_sum | ✓ | sample | s0 (T=0.0) | 896 |  |
| 22 | mod_5 | ✓ | sample | s0 (T=0.0) | 944 |  |
| 23 | safe_div_or_neg | ✓ | sample | s0 (T=0.0) | 1069 |  |
| 24 | polynomial_2ax_plus_b | ✓ | sample | s0 (T=0.0) | 813 |  |
| 25 | sum_abs_three | ✓ | sample | s0 (T=0.0) | 669 |  |
| 26 | average_two | ✓ | sample | s0 (T=0.0) | 783 |  |
| 27 | count_down_3 | ✓ | sample | s0 (T=0.0) | 3836 |  |
| 28 | scaled_add | ✗ | miss | k=3,r=2 | 5374 | scaled_add(4, 1) returned 9, expected 5 |
| 29 | is_positive | ✓ | sample | s0 (T=0.0) | 1088 |  |
| 30 | offset_times_sign | ✓ | sample | s0 (T=0.0) | 636 |  |
