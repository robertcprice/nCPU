# HumanEval-lite Agent Loop Results

Generated 2026-04-19T01:36:30Z — 30 problems, k=3, max_retries=2, model=claude-haiku-4-5-20251001, 33.9s total.

## Summary

- **Pass@1 (cache → best-of-3 → 2 retries)**: **30/30 (100.0%)**
- Path distribution:
  - retry: 1
  - sample: 29
- Total tokens: ~6084

## Per-problem

| # | problem | pass | path | detail | ms | notes |
|--:|---------|:----:|------|--------|---:|-------|
| 1 | sum_to_n | ✓ | sample | s0 (T=0.0) | 1456 |  |
| 2 | factorial | ✓ | sample | s0 (T=0.0) | 789 |  |
| 3 | fibonacci | ✓ | sample | s0 (T=0.0) | 2956 |  |
| 4 | gcd | ✓ | sample | s0 (T=0.0) | 1120 |  |
| 5 | is_prime_simple | ✓ | sample | s0 (T=0.0) | 1104 |  |
| 6 | digit_sum | ✓ | sample | s0 (T=0.0) | 696 |  |
| 7 | digit_count | ✓ | sample | s0 (T=0.0) | 892 |  |
| 8 | reverse_digits | ✓ | sample | s0 (T=0.0) | 808 |  |
| 9 | power_two | ✓ | sample | s0 (T=0.0) | 876 |  |
| 10 | collatz_step | ✓ | sample | s0 (T=0.0) | 1470 |  |
| 11 | is_multiple_of_3 | ✓ | sample | s0 (T=0.0) | 669 |  |
| 12 | clamp_to_range | ✓ | sample | s0 (T=0.0) | 819 |  |
| 13 | nth_odd | ✓ | sample | s0 (T=0.0) | 686 |  |
| 14 | nth_even | ✓ | sample | s0 (T=0.0) | 909 |  |
| 15 | days_in_month | ✓ | sample | s0 (T=0.0) | 1372 |  |
| 16 | grade_letter | ✓ | sample | s0 (T=0.0) | 671 |  |
| 17 | temperature_f_to_c | ✓ | sample | s0 (T=0.0) | 626 |  |
| 18 | abs_sum | ✓ | sample | s0 (T=0.0) | 818 |  |
| 19 | linear_interpolate | ✓ | sample | s0 (T=0.0) | 1303 |  |
| 20 | bit_count_tiny | ✓ | sample | s0 (T=0.0) | 815 |  |
| 21 | quadratic_ax2 | ✓ | sample | s0 (T=0.0) | 1118 |  |
| 22 | time_in_minutes | ✓ | sample | s0 (T=0.0) | 581 |  |
| 23 | celsius_to_fahrenheit | ✓ | sample | s0 (T=0.0) | 819 |  |
| 24 | is_leap_year_simple | ✓ | sample | s0 (T=0.0) | 850 |  |
| 25 | cube | ✓ | sample | s0 (T=0.0) | 1180 |  |
| 26 | rectangle_perimeter | ✓ | sample | s0 (T=0.0) | 694 |  |
| 27 | circle_area_x100 | ✓ | retry | r1 | 4612 |  |
| 28 | diff_from_100 | ✓ | sample | s0 (T=0.0) | 606 |  |
| 29 | triangular | ✓ | sample | s0 (T=0.0) | 1380 |  |
| 30 | difference_of_squares | ✓ | sample | s0 (T=0.0) | 1170 |  |
