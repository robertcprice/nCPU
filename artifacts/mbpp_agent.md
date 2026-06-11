# MBPP Results — mode: agent

Generated 2026-04-19T01:21:52Z — 74 problems, model=claude-haiku-4-5-20251001, 129.2s total.

## Summary

- **Pass@1**: **71/74 (95.9%)**
- path `miss`: 3
- path `retry`: 2
- path `sample`: 69
- Total tokens: ~24891

## Sample failures (first 15)

| task_id | path | error |
|--:|------|-------|
| 111 | miss | no fn set |
| 125 | miss | assertion failed: assert find_length("11000010001") == 6 |
| 140 | miss | no fn set |
