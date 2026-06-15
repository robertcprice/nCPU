# Autoresearch Driver — Verified-Skill Sources

The autoresearch cascade now has two production-grade **work-item
sources** that feed it failures from elsewhere in the system:

1. `ncpu.autoresearch.sources.registry` — rejected verified-skill submissions.
2. `ncpu.autoresearch.sources.synthesis_api` — synthesis-API refusals.

Both close the loop: failures in the user-facing surface become input to
the cascade, which can recover a verified Mog program and re-submit it
to where the failure originated.

## The two adapters

### `ncpu/autoresearch/sources/registry.py`

Reads a `registry_misses.jsonl` (one rejected submission per line) and
maps each one to a canonical `WorkItem`:

- `entry_point` from the sanitized skill name.
- `io_pairs` from the registry `examples` (V1/V2 `{data, n_points, target}` → `IoPair(args=[data], expected=target)`).
- `source_benchmark="registry"`.
- A synthesized `def check(candidate)` harness (reusing `prompt_parser.build_work_item`'s harness emitter).

V3 trace examples (`{data, n_points, targets: [...]}`) are **skipped** —
the per-shot cascade runs one call, not a sequence. A missed run that
emits nothing leaves no empty queue file behind.

**Counter shape** (returned by `mine_registry_misses`):

```json
{
  "read": N, "accepted": M, "skipped_v3_trace": K, "skipped_unsupported": L
}
```

The CLI prints the counter so the runner can see why the queue isn't
growing.

### `ncpu/autoresearch/sources/synthesis_api.py`

Reads `refusals.jsonl` — `success: false` responses from
`ncpu.synthesis_api.server`. Same canonical `WorkItem` shape; the
mining rule is intentionally strict (ints / [int] / str inputs, int
expected, no kwargs). Anything else is skipped and counted.

## CLI

```bash
# Mine registry rejections into the driver queue
python -m ncpu.autoresearch.cli mine-registry --misses path/to/registry_misses.jsonl
# → writes path/to/registry_queue.jsonl, prints counters

# Run the cascade once on the mined queue
python -m ncpu.autoresearch.cli run-once --benchmark registry
```

The `--benchmark` argument of `run-once` now accepts `humaneval`, `mbpp`,
or `registry`.

## Tests

| Test | Asserts |
|---|---|
| `tests/autoresearch/test_cli_registry.py` | `mine-registry` CLI end-to-end, counter shape, queue file format |
| `tests/autoresearch/test_registry_source.py` | `mine_registry_misses` direct: accepted/skipped partitioning, V3 traces skipped, invalid shapes skipped |
| `tests/autoresearch/test_synthesis_api_source.py` | `mine_synthesis_api_refusals` direct: counter shape, mixed input/expected types filtered |
| `tests/registry/test_registry_misses_loop.py` | Round-trip a rejected submission through the registry and verify the miss appears in the JSONL with the right shape |

All 19 new tests pass.

## Doc update

`docs/autoresearch_continuous.md` §4 (registry source) was updated to
document the v3-trace skip rule and the `mine-registry` CLI command.

## Files

- `ncpu/autoresearch/sources/__init__.py` (new)
- `ncpu/autoresearch/sources/registry.py` (new, 157 lines)
- `ncpu/autoresearch/sources/synthesis_api.py` (new, 138 lines)
- `ncpu/autoresearch/cli.py` — `cmd_mine_registry` + parser wiring
- `tests/autoresearch/test_cli_registry.py` (new)
- `tests/autoresearch/test_registry_source.py` (new)
- `tests/autoresearch/test_synthesis_api_source.py` (new)
- `tests/registry/test_registry_misses_loop.py` (new)
- `docs/autoresearch_continuous.md` — §4 updated
