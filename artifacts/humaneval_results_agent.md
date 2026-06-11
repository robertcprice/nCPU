# HumanEval-lite Agent Loop Results

Generated 2026-04-18T23:04:38Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 34.1s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **29/30 (96.7%)**
- Path distribution:
  - miss: 1
  - sample: 29
- Total tokens: ~6729

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | add_two | ✓ | sample | s0 (T=0.0) | 1018 |  |
| 2 | double | ✓ | sample | s0 (T=0.0) | 622 |  |
| 3 | triple | ✓ | sample | s0 (T=0.0) | 710 |  |
| 4 | square | ✓ | sample | s0 (T=0.0) | 799 |  |
| 5 | increment | ✓ | sample | s0 (T=0.0) | 650 |  |
| 6 | decrement | ✓ | sample | s0 (T=0.0) | 581 |  |
| 7 | abs_value | ✓ | sample | s0 (T=0.0) | 957 |  |
| 8 | negate | ✓ | sample | s0 (T=0.0) | 1078 |  |
| 9 | max_of_two | ✓ | sample | s0 (T=0.0) | 1007 |  |
| 10 | min_of_two | ✓ | sample | s0 (T=0.0) | 1392 |  |
| 11 | sign | ✓ | sample | s0 (T=0.0) | 861 |  |
| 12 | is_even | ✓ | sample | s0 (T=0.0) | 753 |  |
| 13 | is_odd | ✓ | sample | s0 (T=0.0) | 852 |  |
| 14 | sum_three | ✓ | sample | s0 (T=0.0) | 1164 |  |
| 15 | abs_diff | ✓ | sample | s0 (T=0.0) | 1130 |  |
| 16 | multiply_add_one | ✓ | sample | s0 (T=0.0) | 683 |  |
| 17 | sum_squares | ✓ | sample | s0 (T=0.0) | 623 |  |
| 18 | clamp_0_10 | ✓ | sample | s0 (T=0.0) | 696 |  |
| 19 | max_of_three | ✓ | sample | s0 (T=0.0) | 807 |  |
| 20 | double_plus_three | ✓ | sample | s0 (T=0.0) | 1938 |  |
| 21 | pythagorean_sum | ✓ | sample | s0 (T=0.0) | 911 |  |
| 22 | mod_5 | ✓ | sample | s0 (T=0.0) | 915 |  |
| 23 | safe_div_or_neg | ✓ | sample | s0 (T=0.0) | 899 |  |
| 24 | polynomial_2ax_plus_b | ✓ | sample | s0 (T=0.0) | 1258 |  |
| 25 | sum_abs_three | ✓ | sample | s0 (T=0.0) | 1027 |  |
| 26 | average_two | ✓ | sample | s0 (T=0.0) | 616 |  |
| 27 | count_down_3 | ✓ | sample | s0 (T=0.0) | 762 |  |
| 28 | scaled_add | ✗ | miss | k=3,r=2 | 7976 | no-code-found |
| 29 | is_positive | ✓ | sample | s0 (T=0.0) | 782 |  |
| 30 | offset_times_sign | ✓ | sample | s0 (T=0.0) | 583 |  |
