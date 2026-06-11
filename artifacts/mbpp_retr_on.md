# MBPP Results — mode: agent

Generated 2026-04-19T02:47:37Z — 46 problems, model=claude-haiku-4-5-20251001, 147.7s total.

## Summary

- **Pass@1**: **39/46 (84.8%)**
- path `miss`: 7
- path `retry`: 1
- path `sample`: 38
- Total tokens: ~39207

## Sample failures (first 15)

| task_id | path | error |
|--:|------|-------|
| 164 | miss | assertion failed: assert are_equivalent(23, 47) == True |
| 228 | miss | no-code-found |
| 229 | miss | assertion failed: assert re_arrange_array([-1, 2, -3, 4, 5, 6, -7, 8, 9], 9) ==  |
| 232 | miss | no fn set |
| 235 | miss | assertion failed: assert even_bit_set_number(20) == 30 |
| 247 | miss | assertion failed: assert lps("TENS FOR TENS") == 5 |
| 260 | miss | assertion failed: assert newman_prime(3) == 7 |
