# Contributing — Adding a New Search Teacher

This guide explains how to add a new search teacher to `nsynth`.
A "search teacher" is a small Rust function in
`nsynth/src/solver/` that, given a `Problem`, either returns a
`SolveResult` containing verified Mog code or returns `None` to
let the pipeline try the next teacher. The teacher is one of
~60 small functions in the `SEARCH_CANDIDATES` table, each of
which encapsulates a particular pattern (sort + count, member-class
DNF, sequence, run-length, etc.).

The end-to-end flow is small enough to describe in one page. The
key invariant is **the preemption contract**: every new teacher
must be both **registered** in `SEARCH_CANDIDATES` (so the
solver can pick it) and **listed** in
`search_result_preempts_native_gradient` (so the
`prefer_differentiable` path returns it directly without
re-distilling). The regression test
`new_teacher_preemption_cases.rs` enforces this for every new
teacher; the build fails if you forget either step.

## Step 1 — Pick the right module

Most teachers live in `search_catalog_advanced.rs` (advanced
array/reduction teachers) or `search_catalog_simple.rs` (basic
scalar/array teachers). For new teachers that fit into a clear
pattern family, just add a new `pub(super) fn search_*(...)` to
the appropriate file. The list:

- `search_catalog_advanced.rs` — moderately complex teachers
  (sort, run-length, binary-search, palindrome, etc.). Add here
  for any new teacher that needs more than a one-liner
  verification.
- `search_catalog_simple.rs` — basic teachers (sum, max,
  first/last, etc.). Add here for any new teacher that just runs
  a Rust helper and emits a short code block.
- `search_catalog_runtime.rs` — pure Rust helpers (e.g.
  `is_sorted_rust`, `first_index_of_rust`). Add a helper here
  if your teacher needs more than the inline closure pattern.
- `search_catalog_codegen.rs` — small `code_*(fn_name)` template
  helpers. Most teachers can use `templ(...)` inline; only add
  here for a codegen pattern that needs to be reused.

## Step 2 — Write the Rust helper

If the teacher needs a Rust verification function, add it to
`search_catalog_runtime.rs`. The helper takes `&[i64]` (or
whatever the input shape) and returns `i64` (the expected output
for each example). For a 0/1 classifier, return `0` or `1`. For
an integer-output teacher, return the integer.

```rust
pub(super) fn my_new_teacher_rust(arr: &[i64]) -> i64 {
    // Verification logic that must agree with the emitted Mog code.
    if arr.is_empty() {
        return 0;  // edge case
    }
    // ...
    1
}
```

## Step 3 — Write the codegen helper

If the teacher needs a non-trivial codegen pattern, add a
`code_my_new_teacher(fn_name, ...) -> String` helper. For simple
teachers, use `templ(...)` inline. The template must use the
Mog syntax — `if`, `while`, `for`, `return`, `fn`, `->`, `let`,
etc. Mog does not have a `set` type; use a table-array if you
need a set-like structure.

```rust
fn code_my_new_teacher(fn_name: &str) -> String {
    templ(
        r#"fn __FN__(arr: [i64]) -> i64 {
    // Mog code that mirrors the Rust helper exactly.
    return 0;
}
"#,
        fn_name,
    )
}
```

## Step 4 — Write the search function

In `search_catalog_advanced.rs` (or `simple.rs`):

```rust
pub(super) fn search_my_new_teacher(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    let param_types = parse_param_types(problem.signature);
    if param_types != [ParamType::ArrayI64] {
        return None;  // wrong shape; let the next teacher try
    }
    if !validate_unary_array(problem, my_new_teacher_rust) {
        return None;  // examples don't match the rule; try next
    }
    verified_result(
        problem,
        code_my_new_teacher(fn_name),
        "search_my_new_teacher",
    )
}
```

`validate_unary_array` runs the Rust helper against every
example and returns `true` iff the outputs match. If `true`, the
emitted code is verified the same way; if both pass, the result
is returned with `success: true`. The `None` paths mean "this
problem is not for me; let another teacher try."

## Step 5 — Register the teacher

Add your teacher to the `SEARCH_CANDIDATES` table in
`nsynth/src/solver/search.rs`:

```rust
SearchCandidate {
    key: "search_my_new_teacher",
    func: search_my_new_teacher,
},
```

The `key` is the string the pipeline reports as `result.method`
and is what the `preemption` test searches for.

## Step 6 — Add to the preemption whitelist

In `nsynth/src/solver/post_enumerative.rs`, add the teacher to
the `search_result_preempts_native_gradient` match:

```rust
| "search_my_new_teacher" => true,
```

This is the second invariant — without it, the
`prefer_differentiable` path would silently re-distill the
problem through the slow gradient solver instead of returning
your verified code directly.

## Step 7 — Add the regression test entry

In `nsynth/src/solver/tests/routing_cases/new_teacher_preemption_cases.rs`,
add your teacher to the `every_preempting_method_has_a_registered_search_candidate`
list. Then add a dedicated test function:

```rust
#[test]
fn search_my_new_teacher_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(candidates.contains(&"search_my_new_teacher"), "...");
    let fake = make_solve_result("search_my_new_teacher", "");
    assert!(search_result_preempts_native_gradient(&fake), "...");
}
```

## Step 8 — Add unit tests

In `nsynth/src/solver/tests.rs`, add a test that builds a
problem your teacher should solve, runs
`solve_problem_search_only(&problem)`, and asserts the method
is `search_my_new_teacher`. Mirror the existing
`search_*_learns_*` tests.

For the int-output teachers (`search_first_index_of`,
`search_last_index_of`), use the `int_arr_problem` helper to
build a problem with a free i64 expected output:

```rust
fn int_arr_problem(
    name: &'static str,
    signature: &'static str,
    rows: &[(&[i64], i64)],
) -> Problem {
    Problem {
        name: name.to_string(),
        category: "array_index",
        description: "",
        signature,
        examples: rows.iter().map(|(arr, label)| Example {
            inputs: vec![Value::Array(arr.to_vec())],
            expected: Value::Int(*label),
        }).collect(),
        holdouts: vec![],
        reference_code: "",
    }
}
```

For 0/1 classifiers, use `arr_class_problem` (already in
`tests.rs`).

## Step 9 — Optionally add a benchmark factory

If the teacher is a useful general-purpose tool, add a factory
to `nsynth/src/benchmark.rs` and wire it into `FACTORIES`. This
makes the new teacher part of the full benchmark sweep. The
multi-variant factories use `variant % cycle_size` to pick
parameters per variant.

If your factory is multi-variant, you'll also need to:

- Use `Box::leak` to convert dynamic strings to `&'static str`
  (the `Problem` struct's `signature` / `description` /
  `reference_code` fields require it).
- Update the `legacy_only_entrypoint_still_solves_full_benchmark`
  test to filter your factory prefix out of the legacy sweep
  (the legacy fallback intentionally bypasses the new teachers).

## Step 10 — Optionally add CLI/API integration tests

- Add a case to `nsynth/scripts/cli_smoke.py` so the release
  binary is exercised end-to-end.
- Add an integration test in
  `tests/synthesis_api/test_server.py` that sends a problem to
  `POST /synthesize` and asserts the response's `method` field
  matches the expected teacher.

## The two-command check

After adding a teacher, run:

```bash
cd nsynth && cargo test --release --lib --no-fail-fast -- \
  new_teacher_preemption_cases \
  search_my_new_teacher_learns_my_thing
```

If both pass, the teacher is wired correctly. The first test
catches a missing preemption whitelist entry; the second
catches a broken codegen or a wrong signature.

## Common pitfalls

- **Forgetting the preemption entry** — the teacher works
  (because the search pipeline still tries it) but the
  `prefer_differentiable` path re-distills instead of
  returning the verified code. The build still passes; only
  the regression test catches it.

- **Forgetting the regression test** — easy to miss, but the
  test is a one-liner and lives in
  `routing_cases/new_teacher_preemption_cases.rs`.

- **Codegen for empty arrays** — many teachers (e.g.
  `search_count_distinct`) need an explicit
  `if arr.len == 0 { return 0; }` because Mog loops don't
  execute on empty arrays but the `count` initialiser defaults
  to 1.

- **Wire-protocol mismatch** — the synthesis API coerces
  `expected: bool` to `expected: 0/1 int` on the way through.
  If your teacher takes the i64 0/1 predicate lane (the
  common case), no change is needed. If it expects a true bool
  expected, the rust CLI parser also handles bool. See
  `value_from_json` in `nsynth/src/bin/nsynth_serve.rs`.

- **Search teachers that crash on empty input** — if your
  Rust helper panics on empty arrays, *fix the helper* (don't
  use `unwrap()` on `arr[0]` without an empty check). The
  search pipeline tries every search teacher on every example
  and a panic in one teacher aborts the whole run.
