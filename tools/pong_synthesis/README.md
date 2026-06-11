# Synthesized Pong — provenance and reproduction

This directory is the complete toolchain behind the `/pong` page on
ncpu.ai (sms-hub/apps/ncpu-site/src/app/pong/), where every game-logic
function was discovered by the nsynth synthesizer rather than written.
ROADMAP.md Rung 1.

## What was the input?

**Small sets of input→output examples — nothing else.** No code, no
pseudo-code, no natural language. For each game rule, a contract was
designed (name, integer signature, 6–16 example pairs), fed to
`mog_synth --problem-json`, and the synthesizer searched for a program
reproducing every pair. Real examples from `pong_rules_final.json`:

| Rule | Training input (literally all of it) | What it found |
|---|---|---|
| `next_pos` | (10,2)→12, (5,−3)→2, (0,4)→4, (100,−7)→93, (42,0)→42, (799,11)→810, (−5,−11)→−16, (600,1)→601 | `return (b + a)` via scalar-expression search |
| `hit_top` | (0)→1, (−1)→1, (−5)→1, (1)→0, (3)→0, (100)→0, (600)→0, (−20)→1 | a gradient-discovered ≤0 test |
| `crossed_left` | (40,30,34)→1, (40,34,34)→1, (40,35,34)→0, (34,30,34)→0, … 14 pairs | a two-branch program using integer division as comparison |
| `sub2` | (7,3)→4, (2,5)→−3, (0,0)→0, (10,−4)→14, … 16 pairs | `a − b` (second attempt — see CEGIS below) |

Two automatic refinement loops sit on top of the seed examples:

1. **CEGIS** (counterexample-guided): after a candidate verifies on its
   training pairs, it is swept over the rule's full reachable game domain
   against a reference. Mismatches become new training examples and the
   synthesizer runs again. This caught two impostors — `sub2` first came
   back as a branchy program fitting only its 6 seed pairs (40,100/160,801
   domain mismatches); with 10 counterexamples folded in, the gradient
   solver produced true subtraction in 0.7 s. An overfit `gte` was caught
   the same way.
2. **Composition fallback**: rules the solver couldn't find directly in
   budget were built as pure wiring of already-synthesized rules — e.g.
   `gte(a,b) = hit_top(sub2(b,a))` (a≥b ⟺ b−a≤0),
   `max2(a,b) = neg(min2(neg(a),neg(b)))`,
   `score_if_out_right(s,x,w) = next_pos(s, exited_right(x,w))`.
   Composed rules are swept over the same domains; the page labels them
   `composed:` and lists which primitives they reuse.

Final tally: **22 rules — 14 synthesized directly, 8 composed** — every
one swept with zero mismatches (51 to 160,801 cases per rule, bounded to
the game's reachable physics: |ball velocity| ≤ 11, field 800×600).

## Files

| File | Role |
|---|---|
| `synth_pong_driver.mjs` | Original CEGIS driver: rule contracts (signature + reference + seed examples), solve → sweep → counterexample loop. Run: `node synth_pong_driver.mjs [rule ...]`, writes `$PONG_OUT`. |
| `solve_remaining.py` | Batch direct-solve pass used for the small arithmetic/predicate contracts. |
| `finalize_pong_rules.mjs` | Merges solved shards, transpiles Mog→TypeScript (`mog_synth --transpile typescript`), domain-sweeps every rule (CEGIS retry on failure), verifies the 8 compositions, and emits the site's `synthesized.ts` (RULES manifest + function bodies). |
| `pong_rules_final.json` | The verified artifact: per rule — method, Mog source, TypeScript, the exact training examples, and the domain-case count it was swept over. |

## Reproduce

```bash
cd nsynth && cargo build --release   # provides mog_synth + --transpile
node tools/pong_synthesis/finalize_pong_rules.mjs
# rewrites apps/ncpu-site/src/app/pong/synthesized.ts; fails loudly on any
# domain mismatch. Solver memory banks (~/.nsynth_solved_programs.json,
# ~/.nsynth_rejected_programs.tsv) make reruns near-instant.
```

Transpiler fixes that fell out of this work (both committed with tests):
truncating integer division in TypeScript output (`Math.trunc`) and
inline single-line `if cond { stmt }` handling.
