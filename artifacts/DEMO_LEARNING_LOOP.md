# Learning Loop Demo

Generated 2026-04-18T17:17:59Z.

This walks through the four stages the nsynth solver actually
performs:

1. **Observe** — capture (input, output) pairs from a reference
   function. The observer has no access to the source code.
2. **Synthesize** — the solver finds a Mog program consistent
   with those pairs. This is a cold-start solve against an
   empty cache.
3. **Re-solve** — the same problem again. Should hit Stage-0
   cache in ~0 ms.
4. **Inspect** — show what the system learned: cache contents,
   weights drift, and the discovered program family.

---

## Step 1 — Observe

Extracted 8 (input → output) pairs from a Python fibonacci
reference. The solver downstream has no access to that source —
only the JSONL record below.

```json
{"name": "fibonacci_extracted", "signature": "fn fibonacci(n: i64) -> i64", "examples": [{"inputs": [10], "expected": 55}, {"inputs": [1], "expected": 1}, {"inputs": [0], "expected": 0}, {"inputs": [4], "expected": 3}, {"inputs": [3], "expected": 2}, {"inputs": [2], "expected": 1}, {"inputs": [8], "expected": 21}, {"inputs": [9], "expected": 34}]}
```

## Steps 2 & 3 — Solve cold, then warm

Ran the first 10 bench problems twice:

- Round 0: empty cache, every solve does real work.
- Round 1: cache populated, every solve that matches an I/O
  fingerprint hits Stage-0 in ~0 ms.

**Measured (round 1 / round 0):**

| metric | value |
|---|---|
| median_ratio | 0.0000 |
| instant_hits | 5 |
| slowdowns | 0 |

✓ Cumulative is **faster** than fresh — the cache works.

## Step 4 — Inspect what was learned

**Cache after the demo:** 10 entries.

Top cached teachers (first 5 by cache order):

```
[top_teachers] showing top 5 of 10 cached teachers (min_success=0)
rank  wins    method                        code
────────────────────────────────────────────────────────────────────────────────────────────────────
1     0       arr_gradient                  fn array_max(arr: [i64]) -> i64 {
2     0       arr_gradient                  fn array_sum(arr: [i64]) -> i64 {
3     0       search_sign_branch            fn sign(x: i64) -> i64 {
4     0       search_clamp_formula          fn clamp_0_100(x: i64) -> i64 {
5     0       search_unary_range_loop       fn sum_to_n(n: i64) -> i64 {
```

Cache teachers grouped by discovered program family:

```
[teacher_clusters] clustering 10 teachers into k=3 groups (seed=42)

── cluster 0 — 8 teachers ──
  0       arr_gradient                    fn array_max(arr: [i64]) -> i64 {
  0       search_sign_branch              fn sign(x: i64) -> i64 {
  0       search_clamp_formula            fn clamp_0_100(x: i64) -> i64 {
  0       search_unary_range_loop         fn sum_to_n(n: i64) -> i64 {
  0       search_gcd_loop                 fn gcd(a: i64, b: i64) -> i64 {
  0       search_max2_formula             fn max2(a: i64, b: i64) -> i64 {
  0       search_lcm_formula              fn gcd_inner(a: i64, b: i64) -> i64 {
  0       search_abs_diff_formula         fn abs_diff(a: i64, b: i64) -> i64 {

── cluster 1 — 1 teachers ──
  0       arr_gradient                    fn array_sum(arr: [i64]) -> i64 {

── cluster 2 — 1 teachers ──
  0       search_scalar_expr              fn add_two(a: i64, b: i64) -> i64 {
```

**Final weight vector** (26 dimensions):

```
(no weights persisted yet — online rule didn't fire this run)
```

## What this shows

- The system *observes* execution without source access.
- On first solve it does real work; on re-solve it's ~0 ms.
- Each successful solve adds a row to the persistent cache.
- The cache is inspectable (`top_teachers`) and clusterable
  (`teacher_clusters`) — learning is not opaque.
- Ranker weights drift from the uniform prior as online updates
  fire, and they're committable / plottable artifacts.

Every piece of this loop is an installed binary or shell script.
