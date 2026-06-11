# Retrain Impact

Every `bootstrap_train` run appends one row. `mean_rank` is the average 0-based position of the correct teacher in the ranker's output, measured against the full cache for each verified (problem, its_code) pair. Lower = better ranker.

| date (UTC) | scored | mean_rank_before | mean_rank_after | Δ |
|------------|-------:|-----------------:|----------------:|--:|
| 2026-04-18 | 94 | 46.46 | 46.44 | -0.02 |
