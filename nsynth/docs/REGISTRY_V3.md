# Verified-Skill Registry — ProgramV3 and Library Format 3

The verified-skill registry (`tools/registry/`) gained a third program
shape that supports **per-data-point running state** with a guard-reset
trigger, and the library JSON was lifted to format 3.

## Why a V3

| Format | Pipeline | When you need it |
|---|---|---|
| V1 | `acc = init; for x: acc = reduce(acc, transform(x)); post-scale; +offset` | Whole-array reductions (sum, max, etc.) |
| V2 | V1 + per-point `combine` (field select / sum / prod / diff) + `guard` (skip point) | Multi-field aggregations with skip semantics |
| V3 | V2 + per-point `state_init` + `update_transform` + `update_reduce` + `reset_guard` | Running counters, running max with reset, prefix-stateful reductions |

V3 is the **per-data-point running state** variant. Every input point
contributes a per-call output (`List[float]` of length `n_points`),
instead of collapsing to one final scalar. The state can be initialized
fresh, updated by an affine of the transformed point, and reset by a
guard condition.

## The codegen surface

```python
# executor.py
class ProgramV3(NamedTuple):
    arity: int
    combine_idx: int           # field select / sum / prod / diff
    guard_idx: int             # 0=pass, 1=lt, 2=le, 3=eq, 4=ge, 5=gt, 6=ne
    guard_threshold: float
    reset_guard_idx: int       # 0=never, 1=lt, 2=le, ...
    reset_threshold: float
    state_init_idx: int        # 0=zero, 1=identity (multiply), 2=-inf
    update_transform_idx: int  # 0..3 (x, -x, abs(x), x*x) applied to v
    update_reduce_idx: int     # 0=set, 1=add, 2=max, 3=min
    post_scale_idx: int        # 0=none, 1=/max(n,1), 2=exp(clamp(s,-30,30))
    output_idx: int            # 0=state, 1=delta, 2=state-init
    offset: float
```

`execute_program_v3(p, data, n_steps) -> List[float]` mirrors the Rust
`execute_program_v3` in `kernels/npcot_wasm/src/lib.rs` line-for-line.
`execute_program_v3_final(p, data, n_steps) -> float` is the last-element
wrapper that lets V3 drop into any V2-shaped verifier by lifting
`ProgramV2.from_v1` → `V2.from_v1` → `V3.from_v2`.

## Library format 3

`library.json` schema is now:

```json
{
  "format": 3,
  "entries": [
    {
      "task_name": "running_counter",
      "author": "dana",
      "examples": [{"data": [...], "n_points": 4, "targets": [...]}],
      "program_v3": {"arity": 1, "combine_idx": 0, ...}
    }
  ]
}
```

V3 entries have no `program` or `program_v2` field. Older formats (1, 2)
are still accepted on read; the server writes format 3 once any V3 entry
is submitted. `V3_FIELDS` is the canonical field-set; the server rejects
submissions with extra or missing fields.

## Tests

`tests/registry/test_registry.py` (25 tests, all green):

- `test_v3_running_counter_replay` — counter 1, 2, 3, 4 on `[5, -2, 0, 9]`.
- `test_v3_running_max_with_reset` — state goes 5, 5, -10 (reset on `< -8`), 1, 1.
- `test_v3_lift_matches_v2_final_output` — V3.from_v2(v2) agrees with v2 on the final output.
- `test_verify_accepts_v3_trace_examples` — `verify_program` accepts a V3 trace.
- `test_submit_v3_skill_lifts_library_to_format_3` — V3 submission forces `library.json` to `format: 3`.

## Files

- `tools/registry/executor.py` — `ProgramV3`, `execute_program_v3`, `execute_program_v3_final`, `output_select_v3`, `post_scale_v3`, `verify_program` (V3 branch), `V3_FIELDS`.
- `tools/registry/server.py` — format-3 library writing, V3 submission validation.
- `tests/registry/test_registry.py` — 25 tests including the 4 new V3 tests.
- `tests/registry/test_registry_misses_loop.py` — the loop that mines rejected submissions into a queue (covered separately under the autorearch sources doc).
