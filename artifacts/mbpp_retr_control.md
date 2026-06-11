# MBPP Results — mode: agent

Generated 2026-04-19T02:47:35Z — 46 problems, model=claude-haiku-4-5-20251001, 148.8s total.

## Summary

- **Pass@1**: **38/46 (82.6%)**
- path `miss`: 8
- path `sample`: 38
- Total tokens: ~31651

## Sample failures (first 15)

| task_id | path | error |
|--:|------|-------|
| 143 | miss | assertion failed: assert find_lists(([9, 8, 7, 6, 5, 4, 3, 2, 1])) == 1 |
| 160 | miss | assert find_solution(2, 3, 7) == (2, 1) raised NotFoundError("Error code: 404 -  |
| 164 | miss | assertion failed: assert are_equivalent(2, 4) == False |
| 229 | miss | assertion failed: assert re_arrange_array([-1, 2, -3, 4, 5, 6, -7, 8, 9], 9) ==  |
| 232 | miss | no fn set |
| 235 | miss | assertion failed: assert even_bit_set_number(10) == 10 |
| 247 | miss | assertion failed: assert lps("TENS FOR TENS") == 5 |
| 260 | miss | assertion failed: assert newman_prime(3) == 7 |
