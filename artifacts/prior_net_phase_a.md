# Prior Net Phase A — measured result (stage A4)

## v1 (confidence gate + persistent async server)

- Gate: signal `mean_logp`, tau -0.2473 (rule: hit_recall); model `prior_net_v0.pt`
- Proposer cost: server startup 2.55s (async, off the solve path) + 4.2ms/request median, vs v0 one-shot 0.68s/problem
Generated 2026-06-12T01:31:42. Fresh isolated banks; solved-cache disabled.

- Coverage OFF: **105/105**, ON: **105/105**
- Zero-search solves (prior proposal verified verbatim): **0**
- Warm-refine solves (proposal + ≤120 Adam steps): **0**
- Total bench wall: OFF 21s → ON 18s (delta -3s)
- No problem in the full bench reached the universal-array fallback (search stages pre-empt it) — see the direct head-to-head below.

## Direct fallback head-to-head (search stages bypassed)

The 16 universal-array problems, run straight through `synthesize_universal_array_fallback` with fresh banks:

- Solved: OFF 16/16, ON 16/16
- Zero-search (proposal verified verbatim): **2** — longest_increasing_run_v0, count_peaks_v0
- Warm-refine wins: **0**
- Wall: OFF 36s → ON 32s (-5s)

| problem | OFF s | ON s | Δs | ON method |
|---|---|---|---|---|
| max_pair_diff_v0 | 3.115 | 3.019 | -0.096 | univ_arr_gradient |
| second_max_v0 | 2.789 | 2.361 | -0.428 | univ_arr_gradient |
| array_range_v0 | 0.001 | 0.009 | +0.008 | univ_arr_gradient |
| max_consecutive_sum_v0 | 2.501 | 2.028 | -0.473 | univ_arr_gradient |
| min_consecutive_sum_v0 | 2.98 | 2.426 | -0.554 | univ_arr_gradient |
| max_stock_profit_v0 | 2.911 | 2.806 | -0.105 | univ_arr_gradient |
| is_sorted_v0 | 1.563 | 2.042 | +0.479 | univ_arr_gradient |
| longest_increasing_run_v0 | 3.152 | 0.008 | -3.144 | prior_net |
| longest_plateau_v0 | 3.424 | 4.555 | +1.131 | univ_arr_gradient |
| prefix_max_sum_v0 | 1.224 | 1.678 | +0.454 | univ_arr_gradient |
| max_abs_v0 | 1.193 | 1.436 | +0.243 | univ_arr_gradient |
| min_positive_v0 | 2.212 | 2.615 | +0.403 | univ_arr_gradient |
| count_peaks_v0 | 3.018 | 0.023 | -2.995 | prior_net |
| alternating_sum_v0 | 1.082 | 1.137 | +0.055 | univ_arr_gradient |
| prefix_sum_k_v0 | 2.085 | 2.391 | +0.306 | univ_arr_gradient |
| is_palindrome_arr_v0 | 2.781 | 3.05 | +0.269 | univ_arr_gradient |

## Method shifts (OFF -> ON)

| problem | method OFF | method ON | s OFF | s ON | Δs |
|---|---|---|---|---|---|

## v0 history (2026-06-11 — one-shot subprocess, ungated)

- Coverage OFF 105/105, ON 105/105; full bench never reached the fallback (search stages pre-empt).
- Direct fallback head-to-head: 2 zero-search wins (longest_increasing_run_v0, count_peaks_v0); wall OFF 83.5s -> ON 116.9s (+33.3s). Net negative: each miss paid the ~7s torch import + model load in a fresh subprocess, plus K=4 warm refines.

