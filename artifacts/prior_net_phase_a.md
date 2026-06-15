# Prior Net Phase A — measured result (stage A4)

## v1 (confidence gate + persistent async server)

- Gate: signal `mean_logp`, tau -0.2473 (rule: hit_recall); model `prior_net_v0.pt`
- Proposer cost: server startup 2.55s (async, off the solve path) + 4.2ms/request median, vs v0 one-shot 0.68s/problem
Generated 2026-06-14T21:41:34. Fresh isolated banks; solved-cache disabled.

- Coverage OFF: **105/105**, ON: **105/105**
- Zero-search solves (prior proposal verified verbatim): **0**
- Warm-refine solves (proposal + ≤120 Adam steps): **0**
- Total bench wall: OFF 60s → ON 46s (delta -14s)
- No problem in the full bench reached the universal-array fallback (search stages pre-empt it) — see the direct head-to-head below.

## Direct fallback head-to-head (search stages bypassed)

The 16 universal-array problems, run straight through `synthesize_universal_array_fallback` with fresh banks:

- Solved: OFF 16/16, ON 16/16
- Zero-search (proposal verified verbatim): **2** — longest_increasing_run_v0, count_peaks_v0
- Warm-refine wins: **0**
- Wall: OFF 84s → ON 63s (-21s)

| problem | OFF s | ON s | Δs | ON method |
|---|---|---|---|---|
| max_pair_diff_v0 | 8.088 | 6.7 | -1.388 | univ_arr_gradient |
| second_max_v0 | 7.77 | 5.329 | -2.441 | univ_arr_gradient |
| array_range_v0 | 0.001 | 0.007 | +0.006 | univ_arr_gradient |
| max_consecutive_sum_v0 | 5.474 | 4.156 | -1.318 | univ_arr_gradient |
| min_consecutive_sum_v0 | 6.685 | 5.274 | -1.411 | univ_arr_gradient |
| max_stock_profit_v0 | 6.986 | 6.11 | -0.876 | univ_arr_gradient |
| is_sorted_v0 | 3.541 | 3.833 | +0.292 | univ_arr_gradient |
| longest_increasing_run_v0 | 6.384 | 0.026 | -6.358 | prior_net |
| longest_plateau_v0 | 7.35 | 7.61 | +0.26 | univ_arr_gradient |
| prefix_max_sum_v0 | 2.869 | 2.441 | -0.428 | univ_arr_gradient |
| max_abs_v0 | 2.82 | 2.988 | +0.168 | univ_arr_gradient |
| min_positive_v0 | 5.368 | 4.987 | -0.381 | univ_arr_gradient |
| count_peaks_v0 | 7.301 | 0.015 | -7.286 | prior_net |
| alternating_sum_v0 | 2.446 | 2.686 | +0.24 | univ_arr_gradient |
| prefix_sum_k_v0 | 4.995 | 4.859 | -0.136 | univ_arr_gradient |
| is_palindrome_arr_v0 | 5.516 | 5.656 | +0.14 | univ_arr_gradient |

## Method shifts (OFF -> ON)

| problem | method OFF | method ON | s OFF | s ON | Δs |
|---|---|---|---|---|---|

## v0 history (2026-06-11 — one-shot subprocess, ungated)

- Coverage OFF 105/105, ON 105/105; full bench never reached the fallback (search stages pre-empt).
- Direct fallback head-to-head: 2 zero-search wins (longest_increasing_run_v0, count_peaks_v0); wall OFF 83.5s -> ON 116.9s (+33.3s). Net negative: each miss paid the ~7s torch import + model load in a fresh subprocess, plus K=4 warm refines.

