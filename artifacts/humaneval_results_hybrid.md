# HumanEval-lite Hybrid Results

Generated 2026-04-18T22:26:51Z — 30 problems, 381.4s total.

## Summary

- **Pass@1**: **27/30 (90.0%)**
- Solved by nsynth: 15
- Solved by LLM fallback: 12
- Missed by both: 3

## Per-problem

| # | problem | path | pass | ms | method | notes |
|--:|---------|:----:|:----:|---:|--------|-------|
| 1 | add_two | nsynth | ✓ | 155 | enumerative |  |
| 2 | double | nsynth | ✓ | 16 | enumerative |  |
| 3 | triple | nsynth | ✓ | 12 | enumerative |  |
| 4 | square | nsynth | ✓ | 9 | enumerative |  |
| 5 | increment | nsynth | ✓ | 11 | enumerative |  |
| 6 | decrement | nsynth | ✓ | 10 | enumerative |  |
| 7 | abs_value | llm | ✓ | 25669 | llm:claude-haiku-4-5-20251001 |  |
| 8 | negate | nsynth | ✓ | 17 | enumerative |  |
| 9 | max_of_two | llm | ✓ | 26680 | llm:claude-haiku-4-5-20251001 |  |
| 10 | min_of_two | llm | ✓ | 25728 | llm:claude-haiku-4-5-20251001 |  |
| 11 | sign | llm | ✓ | 25795 | llm:claude-haiku-4-5-20251001 |  |
| 12 | is_even | nsynth | ✓ | 17978 | synth_gradient |  |
| 13 | is_odd | llm | ✓ | 25582 | llm:claude-haiku-4-5-20251001 |  |
| 14 | sum_three | nsynth | ✓ | 19 | enumerative |  |
| 15 | abs_diff | llm | ✓ | 25805 | llm:claude-haiku-4-5-20251001 |  |
| 16 | multiply_add_one | nsynth | ✓ | 11 | search_polynomial_quadratic |  |
| 17 | sum_squares | llm | ✓ | 25732 | llm:claude-haiku-4-5-20251001 |  |
| 18 | clamp_0_10 | llm | ✓ | 25693 | llm:claude-haiku-4-5-20251001 |  |
| 19 | max_of_three | llm | ✓ | 25979 | llm:claude-haiku-4-5-20251001 |  |
| 20 | double_plus_three | nsynth | ✓ | 11 | search_polynomial_quadratic |  |
| 21 | pythagorean_sum | llm | ✓ | 25701 | llm:claude-haiku-4-5-20251001 |  |
| 22 | mod_5 | nsynth | ✓ | 15 | enumerative |  |
| 23 | safe_div_or_neg | llm | ✓ | 26326 | llm:claude-haiku-4-5-20251001 |  |
| 24 | polynomial_2ax_plus_b | miss | ✗ | 25895 | nsynth:TIMEOUT|llm:llm:claude-haiku-4-5- | wrong on [1, 10, 5]: got 25, expected 15 |
| 25 | sum_abs_diffs | miss | ✗ | 1091 | nsynth:search_two_branch|llm:llm:claude- | wrong on [10, 5, 0]: got 20, expected 15 |
| 26 | average_two | nsynth | ✓ | 30 | search_scalar_expr |  |
| 27 | count_down_3 | nsynth | ✓ | 12 | enumerative |  |
| 28 | scaled_add | miss | ✗ | 25742 | nsynth:TIMEOUT|llm:llm:claude-haiku-4-5- | wrong on [2, 3]: got 13, expected 7 |
| 29 | is_positive | llm | ✓ | 25635 | llm:claude-haiku-4-5-20251001 |  |
| 30 | offset_times_sign | nsynth | ✓ | 13 | enumerative |  |
