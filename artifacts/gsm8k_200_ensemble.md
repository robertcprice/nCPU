# GSM8K Results — mode: agent

Generated 2026-04-19T16:02:09Z — 200 problems, model=claude-haiku-4-5-20251001, 1353.1s total.

## Summary

- **Accuracy**: **190/200 (95.0%)**
- path `ensemble`: 190
- path `ensemble-miss`: 10
- Total tokens: ~1352055

## Sample failures (first 15)

| # | path | predicted | ground truth | extra |
|--:|------|----------:|-------------:|-------|
| 2 | ensemble-miss | 1.0 | 70000.0 | votes=2/3 [cot,tool] |
| 24 | ensemble-miss | 19.0 | 26.0 | votes=2/3 [cot,tool] |
| 37 | ensemble-miss | 0.0 | 2.0 | votes=2/3 [pot,tool] |
| 61 | ensemble-miss | 0.0 | 1430.0 | votes=2/3 [cot,tool] |
| 73 | ensemble-miss | 0.0 | 255.0 | votes=2/3 [cot,tool] |
| 87 | ensemble-miss | 0.0 | 9360.0 | votes=2/3 [cot,tool] |
| 119 | ensemble-miss | 99076.92307692308 | 95200.0 | votes=2/3 [pot,tool] |
| 162 | ensemble-miss | 0.0 | 92.0 | votes=2/3 [cot,tool] |
| 174 | ensemble-miss | 5700.0 | 95.0 | votes=2/3 [cot,tool] |
| 187 | ensemble-miss | 106.12 | 106.0 | votes=1/2 [escalate] |
