# MBPP Results — mode: llm

Generated 2026-04-19T01:21:00Z — 74 problems, model=claude-haiku-4-5-20251001, 77.6s total.

## Summary

- **Pass@1**: **67/74 (90.5%)**
- path `miss`: 7
- path `sample`: 67
- Total tokens: ~15869

## Sample failures (first 15)

| task_id | path | error |
|--:|------|-------|
| 83 | miss | assertion failed: assert get_Char("abc") == "f" |
| 84 | miss | assertion failed: assert sequence(10) == 6 |
| 87 | miss | assertion failed: assert merge_dictionaries_three({ "R": "Red", "B": "Black", "P |
| 111 | miss | no fn set |
| 125 | miss | assertion failed: assert find_length("10111") == 1 |
| 138 | miss | assertion failed: assert is_Sum_Of_Powers_Of_Two(10) == True |
| 140 | miss | no fn set |
