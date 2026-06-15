# Structural Array Teachers

Three new search teachers plus the `ArrayFeature` taxonomy that powers them.
All are registered in `SEARCH_CANDIDATES` and preempt the slow native-gradient
path (`search_result_preempts_native_gradient`). A regression test
(`new_teacher_preemption_cases.rs`) keeps the invariant honest for every
future addition.

## The `ArrayFeature` enum (`nsynth/src/solver/search_families.rs`)

A closed 10-variant feature space over `Vec<i64>` that, when combined by
separate-and-conquer DNF induction, is expressive enough to cover the
bulk of unary-array classification problems seen in benchmarks and
curriculum learning:

| Variant | Meaning | Verifier (Rust) |
|---|---|---|
| `Contains(tok)` | array contains `tok` | `arr.contains(tok)` |
| `Adjacent(a, b)` | some window `[a, b]` exists | `arr.windows(2).any(...)` |
| `Sequence(a, b)` | first occurrence of `a` precedes last occurrence of `b` | `position(rposition)` |
| `CountAtLeast(tok, k)` | `tok` appears ≥ k times | `iter().filter().count() >= k` |
| `CountExactly(tok, k)` | `tok` appears exactly k times | `== k` |
| `RunAtLeast(tok, k)` | contiguous run of `tok` of length ≥ k | loop over `arr` |
| `AnyGreater(thr)` | some element > thr | `any(\|x\| x > thr)` |
| `AnyLess(thr)` | some element < thr | `any(\|x\| x < thr)` |
| `AllGreater(thr)` | every element > thr (non-empty) | `!empty && all(...)` |
| `AllLess(thr)` | every element < thr (non-empty) | `!empty && all(...)` |

The list is the **single source of truth** for what a structural-array
teacher may claim. The codegen backend (`code_array_feature_dnf_search` in
`search_codegen.rs`) emits the same predicates as runtime Mog, so a
candidate that verifies in the Rust matcher is also a candidate that
verifies when transpiled.

## The three new teachers

### `search_array_sequence`

Learns a single ordered pair `(a, b)` such that positives have an `a`
strictly before a `b`. Codegen (`code_array_sequence_search`) uses a
`seen_{i}` flag pattern: emit `seen` when `a` is seen, return `1` when `b`
is seen and `seen == 1`. **Repeated tokens** (e.g. `a == b`, or two
consecutive `a`s before `b`) are handled correctly because the check
`if seen_{i} == 1 { return 1; }` runs *before* the `if x == a { seen_{i} = 1; }`
update for the same iteration — see `tests.rs::search_array_sequence_learns_order_constraint`.

### `search_array_feature_dnf`

The full taxonomy. Mines a feature candidate set from every positive array
(constains, adjacent pairs, ordered pairs up to length 32, count-/run-based
features), then runs separate-and-conquer DNF induction with the same
RIPPER-style loop as `search_array_dnf`. Up to MAX_DISJUNCTS clauses; each
clause is a conjunction of `ArrayFeature` predicates; the codegen emits
one init/loop body per feature, then nests `if f{idx} == 1 { ... }` guards
inside one another. Test: `search_array_feature_dnf_learns_count_and_run_features`.

### `search_string_subsequence_class`

String analog of the DNF teacher. Mines character-subsequence candidates
from positives (length 2..=4, exhaustive), filters those that fire on any
negative, then RIPPER-style separate-and-conquer over disjuncts (max 8).
Codegen (`code_string_subsequence_class_search`) walks the input string
once per disjunct, advancing a per-disjunct cursor; success on every
required token in a disjunct returns `1`. Test:
`search_string_subsequence_class_learns_order_constraint`.

## The preemption invariant

```rust
// post_enumerative.rs
| "search_array_sequence"
| "search_array_feature_dnf"
| "search_string_subsequence_class"
| "search_strictly_increasing"
| "search_has_strictly_increasing_run"
| "search_first_index_of" => true,
```

Each new teacher appears in the preemption whitelist so the
`prefer_differentiable` path returns it directly instead of feeding it
into gradient distillation. The companion regression test
(`new_teacher_preemption_cases.rs`) enforces that **both** invariants hold
for each new teacher:

1. The method is in `SEARCH_CANDIDATES` (so the solver can pick it).
2. The method is in `search_result_preempts_native_gradient` (so the
   pipeline returns it without re-distillation).

A phantom entry on either side would silently waste cycles; the test
fails the build before any benchmark regression can sneak in.

## Related teachers (live in `search_catalog_advanced.rs`)

These unary-array teachers were added alongside the `ArrayFeature`
taxonomy. They round out the strictly-monotonicity and positional-query
surfaces and sit naturally between the existing array-reduction
teachers and the DNF teacher.

### `search_strictly_increasing`

Returns `1` iff every adjacent pair satisfies `arr[i] < arr[i-1]` (no
equal neighbours allowed). Codegen uses a single-pass `while` loop with
an early `return 0` on `arr[i] <= arr[i-1]`. Test:
`search_strictly_increasing_learns_strict_inequality`.

### `search_has_strictly_increasing_run`

Returns `1` iff the array contains a strictly increasing run of length
≥ k. The teacher tries k ∈ {2, 3, 4, 5} in order and emits the first k
whose verification pass succeeds. Codegen is a running-counter loop
with the threshold inlined. Test:
`search_has_strictly_increasing_run_learns_run_length`.

### `search_first_index_of`

Returns the first index `i` where `arr[i] == target`, or `-1` if absent.
The teacher tries a fixed set of candidate targets
({0, 1, -1, 2, 3, 5, 7, 10, -2, 100, 42, 13, 17, -5}) and emits the
first that matches every example. The expected output is a free `i64`
(not a 0/1 classifier), so the test uses an `int_arr_problem` builder
and `assert_search_generalizes_problem` to verify on held-out inputs.
Test: `search_first_index_of_learns_target_value`.

All three are added to `SEARCH_CANDIDATES` and the preemption whitelist;
the extended regression test (`new_teacher_preemption_cases.rs`) now
covers all six new teachers.

## Related teachers (live in `search_catalog_advanced.rs`)

These two unary-array teachers were added alongside the `ArrayFeature`
taxonomy because they round out the strictly-monotonicity surface and
sit naturally between `search_is_sorted` (≤) and the DNF teacher
(set-membership DNF):

### `search_strictly_increasing`

Returns `1` iff every adjacent pair satisfies `arr[i] < arr[i-1]` (no
equal neighbours allowed). Codegen uses a single-pass `while` loop with
an early `return 0` on `arr[i] <= arr[i-1]`. Test:
`search_strictly_increasing_learns_strict_inequality`.

### `search_has_strictly_increasing_run`

Returns `1` iff the array contains a strictly increasing run of length
≥ k. The teacher tries k ∈ {2, 3, 4, 5} in order and emits the first k
whose verification pass succeeds. Codegen is a running-counter loop
with the threshold inlined. Test:
`search_has_strictly_increasing_run_learns_run_length`.

Both are added to `SEARCH_CANDIDATES` and the preemption whitelist; the
extended regression test (`new_teacher_preemption_cases.rs`) now covers
all five new teachers.

## End-to-end coverage

- **Unit**: 3 new tests in `solver/tests.rs` (sequence, feature_dnf, subsequence).
- **Routing regression**: 3 new tests in `solver/tests/routing_cases/new_teacher_preemption_cases.rs`.
- **API integration**: `tests/synthesis_api/test_server.py::test_array_feature_dnf_problem_solves_through_api`
  sends a non-trivial `array_feature_dnf` problem through `/synthesize` and
  asserts `method == "search_array_feature_dnf"`.
- **CLI flow**: `nsynth/src/bin/nsynth_codegen.rs` + `lib.rs` register
  the new methods for the CLI.

## Why this taxonomy

The 10 features were chosen so that a DNF over them is a **strict
superset** of the rules expressible by `search_array_member_class`
(set-membership DNF), `search_array_conjunction` (positive-and-negative
membership), and `search_array_dnf` (pure token-set DNF). Every
classification problem solvable by those teachers is also solvable here
— and many more: count thresholds, runs, ordered pairs, numeric
thresholds, any/all-relations to a threshold. The cost is a bigger
search space; the gain is a one-shot learner for almost any
set-or-sequence problem without enumerating teacher variants.
