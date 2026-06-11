# HumanEval-lite Agent Loop Results

Generated 2026-04-19T01:33:18Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 32.2s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **29/30 (96.7%)**
- Path distribution:
  - miss: 1
  - sample: 29
- Total tokens: ~6429

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | add_two | ✓ | sample | s0 (T=0.0) | 821 |  |
| 2 | double | ✓ | sample | s0 (T=0.0) | 670 |  |
| 3 | triple | ✓ | sample | s0 (T=0.0) | 583 |  |
| 4 | square | ✓ | sample | s0 (T=0.0) | 1234 |  |
| 5 | increment | ✓ | sample | s0 (T=0.0) | 759 |  |
| 6 | decrement | ✓ | sample | s0 (T=0.0) | 885 |  |
| 7 | abs_value | ✓ | sample | s0 (T=0.0) | 752 |  |
| 8 | negate | ✓ | sample | s0 (T=0.0) | 572 |  |
| 9 | max_of_two | ✓ | sample | s0 (T=0.0) | 699 |  |
| 10 | min_of_two | ✓ | sample | s0 (T=0.0) | 1538 |  |
| 11 | sign | ✓ | sample | s0 (T=0.0) | 927 |  |
| 12 | is_even | ✓ | sample | s0 (T=0.0) | 730 |  |
| 13 | is_odd | ✓ | sample | s0 (T=0.0) | 672 |  |
| 14 | sum_three | ✓ | sample | s0 (T=0.0) | 740 |  |
| 15 | abs_diff | ✓ | sample | s0 (T=0.0) | 660 |  |
| 16 | multiply_add_one | ✓ | sample | s0 (T=0.0) | 638 |  |
| 17 | sum_squares | ✓ | sample | s0 (T=0.0) | 1788 |  |
| 18 | clamp_0_10 | ✓ | sample | s0 (T=0.0) | 749 |  |
| 19 | max_of_three | ✓ | sample | s0 (T=0.0) | 769 |  |
| 20 | double_plus_three | ✓ | sample | s0 (T=0.0) | 1968 |  |
| 21 | pythagorean_sum | ✓ | sample | s0 (T=0.0) | 703 |  |
| 22 | mod_5 | ✓ | sample | s0 (T=0.0) | 1054 |  |
| 23 | safe_div_or_neg | ✓ | sample | s0 (T=0.0) | 1295 |  |
| 24 | polynomial_2ax_plus_b | ✓ | sample | s0 (T=0.0) | 656 |  |
| 25 | sum_abs_three | ✓ | sample | s0 (T=0.0) | 695 |  |
| 26 | average_two | ✓ | sample | s0 (T=0.0) | 1017 |  |
| 27 | count_down_3 | ✓ | sample | s0 (T=0.0) | 1008 |  |
| 28 | scaled_add | ✗ | miss | k=3,r=2 | 6186 | scaled_add(2, 3) returned 8, expected 7 |
| 29 | is_positive | ✓ | sample | s0 (T=0.0) | 827 |  |
| 30 | offset_times_sign | ✓ | sample | s0 (T=0.0) | 623 |  |
