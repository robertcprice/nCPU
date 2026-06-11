# Diversity Pareto Sweep

Offset: 60, Limit: 10, generated 2026-04-18T19:34:22Z

K=0 is the reference (no cap, full rank). The other K values cap
top-K with the diversity pass active. Lower win_pct + lower mean_ms
means the cap is too tight; ~equal win_pct with lower mean_ms is a
Pareto improvement.

| K | wins | attempts | win_pct | mean_ms |
|--:|-----:|---------:|--------:|--------:|
| 4 | 1 | 5 | 20.00 | 2358.60 |
| 8 | 1 | 5 | 20.00 | 18656.00 |
| 16 | 1 | 5 | 20.00 | 13588.80 |
| 32 | 2 | 5 | 40.00 | 8432.00 |
| 48 | 4 | 5 | 80.00 | 16086.80 |
| 64 | 4 | 5 | 80.00 | 7617.00 |
| 0 | 4 | 5 | 80.00 | 7749.60 |
