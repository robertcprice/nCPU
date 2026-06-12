# Prior Net Phase A — measured result (stage A4)

Generated 2026-06-11T21:48:43. Fresh isolated banks; solved-cache disabled.

- Coverage OFF: **105/105**, ON: **105/105**
- Zero-search solves (prior proposal verified verbatim): **0**
- Warm-refine solves (proposal + ≤120 Adam steps): **0**
- Total bench wall: OFF 57s → ON 61s (delta +4s)
- No problem in the full bench reached the universal-array fallback (search stages pre-empt it) — see the direct head-to-head below.

## Direct fallback head-to-head (search stages bypassed)

The 16 universal-array problems, run straight through `synthesize_universal_array_fallback` with fresh banks:

- Solved: OFF 16/16, ON 16/16
- Zero-search (proposal verified verbatim): **2** — longest_increasing_run_v0, count_peaks_v0
- Warm-refine wins: **0**
- Wall: OFF 84s → ON 117s (+33s)

| problem | OFF s | ON s | Δs | ON method |
|---|---|---|---|---|
| max_pair_diff_v0 | 9.53 | 16.769 | +7.239 | univ_arr_gradient |
| second_max_v0 | 6.619 | 9.394 | +2.775 | univ_arr_gradient |
| array_range_v0 | 0.002 | 3.007 | +3.005 | univ_arr_gradient |
| max_consecutive_sum_v0 | 6.074 | 4.871 | -1.203 | univ_arr_gradient |
| min_consecutive_sum_v0 | 5.127 | 6.27 | +1.143 | univ_arr_gradient |
| max_stock_profit_v0 | 5.714 | 6.181 | +0.467 | univ_arr_gradient |
| is_sorted_v0 | 2.713 | 4.778 | +2.065 | univ_arr_gradient |
| longest_increasing_run_v0 | 9.036 | 1.564 | -7.472 | prior_net |
| longest_plateau_v0 | 9.109 | 9.116 | +0.007 | univ_arr_gradient |
| prefix_max_sum_v0 | 2.646 | 9.125 | +6.479 | univ_arr_gradient |
| max_abs_v0 | 2.176 | 8.517 | +6.341 | univ_arr_gradient |
| min_positive_v0 | 5.828 | 7.118 | +1.29 | univ_arr_gradient |
| count_peaks_v0 | 7.119 | 3.26 | -3.859 | prior_net |
| alternating_sum_v0 | 1.972 | 5.512 | +3.54 | univ_arr_gradient |
| prefix_sum_k_v0 | 4.062 | 8.531 | +4.469 | univ_arr_gradient |
| is_palindrome_arr_v0 | 5.562 | 12.725 | +7.163 | univ_arr_gradient |

## Method shifts (OFF -> ON)

| problem | method OFF | method ON | s OFF | s ON | Δs |
|---|---|---|---|---|---|

