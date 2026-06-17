# nCPU Understanding — Master Roadmap

> **Vision.** A comprehension/reasoning/generation system that is *categorically*
> better than an LLM on the axes LLMs are worst at — **soundness, proof,
> auditability, monotone updatability, zero hallucination** — and that reaches
> LLM-scale breadth *without surrendering any of them*, by letting an LLM propose
> and a verified gate dispose. "Anything, soundly. Generate anything it can stand
> behind."

This is the durable plan. Every phase states **Goal · Design · Soundness ·
Deliverables · Verify · Risk**. Done phases are summarized; future phases are
specified to the depth needed to build them. The non-negotiable invariant across
all of it: **nothing is adopted unless it is verified on its spec AND passes the
full regression gate. Growth is monotone. The system never asserts what it cannot
prove.**

---

## Status — shipped (all on `main`, all sound)

| Phase | Commit | What it gave us |
|---|---|---|
| **A · autonomy spine** | `c1cc09a` | persistent learned-component store (reload **+ re-gate** on boot), auto gap detection, self-curriculum, `study()` loop, provenance. Growth survives restart. |
| **B · external validation** | `0cbd296` | FraCaS-style entailment benchmark (40 cases / 9 phenomena), dashboard, bench→study feedback. Soundness bar = zero wrong. |
| **C · functional integration + breadth** | `f8bdcd6` | learned classifiers drive the parser + answerer (domain-bounded — no overfit false-Yes); relative-clause Qs, possessives, PPs; **the loop bites** (97.6%→100%). |
| **D · deeper reasoning + belief revision** | `3b83ad4` | modus ponens/tollens (fallacies refused), actionable belief revision (never holds F and ¬F), metamorphic paraphrase fuzzer. |
| **E · grammar induction** | `40e8a70` | the parser **learns a new construction** (object-fronting/OSV) as a synthesized+gated verified rule; generalizes to unseen words; `no_construction_collision` gate probe. |
| **F · hybrid LLM verifier** | `5ac574d` | **Claude proposes, nCPU verifies.** Live: learns "wizard" from a Claude proposal, gated; a hallucinated proposal is rejected, engine unchanged. The LLM gives breadth and cannot make nCPU unsound. |

**The moat, concretely:** every answer carries a proof; every self-added algorithm
is verified + gated; a stale/poisoned store row is rejected on boot; a learned
classifier answers only within its verified domain; an LLM proposal that would
regress *anything* is rejected. Tests at `94acf6b`: understanding 309/0,
self_improve 25/0, comprehension 6/0, eval 7/0, grammar 8/0, adversarial 60/0,
soundness+metamorphic fuzz green, hybrid 13/0.

---

## Phase G — Open-vocabulary at scale  *(next)*

**Goal.** Turn the one-word-at-a-time hybrid (F) into a **corpus-driven** loop
that grows vocabulary at scale, soundly and cumulatively — the literal path to
"trillion vocab," bounded only by what passes verification.

**Design.**
- `Mind::study_corpus(corpus, proposer, max_words) -> CorpusStudyReport`: scan every
  sentence; collect the *distinct* unknown content words (lexical gaps); ask the
  proposer (Claude) to classify them — **batched** (one request classifies many
  words, returning a list of `MembershipProposal`s) to amortize API cost; route
  each through the verify + gate funnel; persist; loop until dry or `max_words`.
- **Batch gating for scale.** Re-gating per component is O(components × golden). For
  a batch, graft the whole accepted batch onto a candidate, gate **once**; on red,
  **bisect** to find and drop the culprit(s), keeping the sound remainder. Soundness
  is preserved (a poison word still can't survive); cost drops from per-word to
  per-batch.
- **Coverage metric.** `coverage(corpus) -> (known, total, pct)` — % of a corpus's
  content words the engine can parse. Report before/after a study run.
- **Cumulative + bounded reload.** Many persisted components make `Engine::new()`
  reload slow. Keep the re-gate (poison-safety) but batch it; cap with a configurable
  budget; log what was dropped (no silent truncation).

**Soundness.** Every word still passes `solve_problem` verify + the gate (batched,
with bisection so a poison word in a batch is still caught). A learned word stays
domain-bounded. The corpus loop is monotone — coverage only rises, wrong stays 0.

**Deliverables.** `study_corpus`, batched `Proposer::propose_membership_batch`,
`coverage` report, a `vocab` demo (small corpus with several unknown creature/agent
words → Claude classifies the batch → verified+gated+learned → coverage X%→Y% →
restart boots with the grown vocab).

**Verify (default-fail).** A batch with one poison word: the poison is rejected,
the rest learned, base engine sound (bisection works). Coverage strictly rises and
wrong stays 0. Tests hermetic (MockProposer batch). No regression.

**Risk.** Reload cost as components accumulate (mitigate: batch-gate + budget +
later Phase M perf work). Batch-bisection complexity (keep it simple: binary split).

---

## Phase H — Generation  *(after G)*

**Goal.** nCPU **produces** English, not just answers — and only what it can stand
behind. "Generate anything it can stand behind" = sound generation, zero
hallucinated statements.

**Design.**
- Unify comprehension with the existing `speak.py`/inflection generation side into a
  bidirectional surface on `Mind`:
  - `describe(entity) -> Vec<String>`: generate grammatical sentences stating
    **only** facts + sound consequences the world model holds about the entity.
  - `say(meaning) -> String`: realize an arbitrary `Meaning` to a grammatical
    sentence (generalizes the existing `qa::realize`).
  - `answer_prose(question) -> String`: a grounded free-text answer assembled from
    entailed facts (still backed by `why`'s proof).
- **Bidirectional-consistency soundness (the no-hallucination guarantee for
  generation):** for everything it utters, `understand(say(M))` round-trips to `M`.
  It only says what it means and means what it says. A sentence that doesn't
  round-trip is **not emitted** (regenerate or fall back).
- **Sound generation:** `describe`/`answer_prose` never emit a statement the world
  model doesn't entail — generation is bounded by `holds`/`prove`, exactly like
  answering.
- **Hybrid generation (optional, the F pattern for fluency):** let Claude propose
  *fluent phrasing* for an intended meaning; nCPU accepts it **only if**
  `understand(claude_phrasing) == intended_meaning`. Even LLM-fluent prose is then
  sound — re-parsed and checked before it leaves.

**Soundness.** Round-trip check on every generated sentence; generation gated by
world-model entailment; hybrid phrasing re-verified by re-parsing. No utterance the
system can't both parse and prove.

**Deliverables.** `say`/`describe`/`answer_prose`, the round-trip guarantee, a
`generate` demo (read facts → describe an entity in grammatical sentences →
round-trip each → refuse to generate an unentailed claim), optional Claude-phrasing
path gated by re-parse.

**Verify (default-fail).** Every generated sentence round-trips to its source
meaning; nothing unentailed is ever generated; the hybrid phrasing path rejects a
fluent-but-wrong rephrase. No regression.

**Risk.** Round-trip coverage limited by parser breadth (a meaning it can't yet
re-parse can't be generated soundly) — honest limit, shrinks as E/G grow grammar+vocab.

---

## Post-phases — the long arc (specified, not yet scheduled)

### Phase I — World-knowledge ingestion
**Goal.** Know things beyond the conversation. **Design.** A knowledge base of facts
(triples / asserted meanings); `ingest(source)` where Claude proposes candidate
facts and nCPU **verifies consistency** against the existing KB (contradiction
ledger + belief revision from D) before storing. `what_do_you_know` and answering
draw on the KB. **Soundness.** A proposed fact that contradicts known facts is
flagged/revised, never silently stored; provenance tracked. **Why.** Turns a
blank-slate reasoner into one with standing knowledge — the other half of
"trillion-fact" alongside G's "trillion-vocab."

### Phase J — Pragmatics, context, discourse depth
**Goal.** Anaphora beyond recency (binding, reflexives), presupposition, a *sound*
slice of implicature, question-under-discussion, multi-turn context. **Design.**
Extend `discourse.rs` with a discourse-state model; gate each pragmatic inference so
it never over-derives. **Soundness.** Implicature is defeasible — model only the
monotone, cancellable-flagged subset, or keep it as "suggested, not entailed."

### Phase K — Confidence & calibration
**Goal.** Graded confidence beyond 3-valued (yes/no/idk). **Design.** Provenance-
weighted confidence (asserted=certain, derived=conditional, learned-classifier=
domain-bounded, LLM-proposed=until-verified); answers carry a calibrated band.
**Why.** Enables active learning ("ask about the most uncertain thing") and a
richer belief-revision policy. **Soundness.** Confidence never upgrades an unproven
claim to asserted; it annotates, never fabricates.

### Phase L — The moat product (provably-correct NLI / spec verification)
**Goal.** Package the engine as a deployable **hallucination-proof reasoning layer**
for high-stakes domains (legal, medical, compliance, NL-spec verification) where a
sound "I don't know" beats a confident wrong answer. **Design.** A clean API/SDK +
C FFI surface (`ncpu_*`), a domain-loading path (vocab+rules via G/E from a domain
corpus), proof export. **Why.** The defensible, monetizable application of the moat —
nobody else offers proof-backed "I don't know."

### Phase M — Scale & performance
**Goal.** Handle thousands of learned components and large KBs efficiently.
**Design.** Modularize the composed program (don't re-execute one giant string);
incremental + parallel reload; index learned classifiers; cache compiled programs;
profile `Engine::new()`. **Why.** G/I make the component/fact count grow; the
reload/exec path must stay fast and the re-gate must stay bounded.

### Phase N — Continual self-directed learning
**Goal.** The system improves unattended. **Design.** A scheduled `study` loop that
mines gaps from real usage logs, proposes via the LLM, gates, grows — with the
journal as its audit trail and the gate as its safety. **Why.** A genuinely
self-improving system that compounds over time, every gain verified.

### Phase O — Publish / open-source / benchmark at scale
**Goal.** External validation + dissemination. **Design.** Run on real NLI suites
(raw FraCaS once G grows the vocab; a slice of standard NLI); a paper section
(sound, self-extending neuro-symbolic comprehension with verified growth +
hallucination-proof hybrid); open-source the engine + demos. **Why.** Turns the
result into a citable, reusable artifact.

---

## Dependency / sequencing

```
A ─ B ─ C ─ D ─ E ─ F   (done)
                     │
                     ├─► G  open-vocab at scale  ──┐
                     │                              ├─► I  world-knowledge ─┐
                     └─► H  generation ────────────┘                        ├─► L  moat product
                                                          J  pragmatics ────┤
                                                          K  confidence ────┤
                                                          M  scale/perf ────┘  (cross-cutting; pull in as G/I grow counts)
                                                          N  continual learning   (rides A+F+G)
                                                          O  publish/benchmark     (rides B+everything)
```

**Build order recommendation:** G → H (finish the greenlit "vocab + generate" arc),
then **M** (perf, because G/I make counts grow), then **I** (world knowledge), then
**L** (the product) with **J/K** folded in as needed, then **N**, then **O**.

---

## Honest limits (carried forward, shrinking with each phase)

- Self-extension is bounded to what `solve_problem` can synthesize + verify
  (lexical classifiers, membership maps, role-assignment rules over class skeletons).
  Open-ended *novel grammar* induction beyond skeleton-bounded rules remains research-
  grade; E is the first real step.
- Generation soundness is bounded by parser breadth (H): a meaning it can't re-parse
  can't be generated soundly. Shrinks as E/G/J expand grammar+vocab+pragmatics.
- The hybrid (F/G/I) depends on an external LLM for *breadth proposals*; nCPU stays
  sound regardless, but breadth-acquisition availability is gated on the proposer.
- No multimodality in the symbolic core (it would live LLM-side in the hybrid).
