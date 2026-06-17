# nCPU Understanding Layer — Architecture, Learning, and Performance

> A sound, self-extending neuro-symbolic comprehension engine where **every
> lexical/rule decision is a verified synthesized program**, every reasoning step
> is an auditable derivation, and the system **adds new algorithms to itself**
> under a regression gate that keeps growth monotone.

This document is the durable reference for *how the system works, how it learns
concepts, and how it performs*. Numbers are real (re-run, not assumed) and tagged
with the commit at which they were measured.

---

## 1. What this is, in one paragraph

You `read` English sentences into a `Mind`; it builds a three-valued, open-world
model of the world; you `ask` questions and it answers `Yes` / `No` / `I don't
know` from that model, **closed under sound inference**. Underneath, the lexical
facts it needs (which words are nouns, how verbs inflect, subject–verb agreement,
…) are not hand-coded — they are **Mog programs the engine synthesized and
verified** from a small curriculum at startup. On top of that, it can **explain
its reasoning** (proof traces), **reflect on its own knowledge and algorithms**,
reason **hypothetically / counterfactually / abductively**, and **teach itself new
classifiers** from gaps it detects — accepting a self-taught program only if it
passes the full regression gate, and remembering it across restarts.

The guiding invariant everywhere: **soundness over coverage.** The system would
rather say "I don't know" than assert something it cannot prove. It never
manufactures a false entailment, a false cause, a fabricated justification, or a
membership it did not verify.

---

## 2. Architecture

```
            ┌─────────────────────────────────────────────────────────────┐
            │ Mind  (src/understanding/mind.rs)  — the top-level handle     │
            │   read · ask · why · understand · explain_self · gaps ·       │
            │   what_do_you_know · what_would_change_your_mind ·            │
            │   suppose · what_if_not · explain_cause · study · self_improve│
            │   · self_check · contradictions · recognizes_word             │
            └───────────────┬───────────────────────────────┬──────────────┘
                            │                                │
        ┌───────────────────▼──────────┐      ┌──────────────▼───────────────┐
        │ Discourse (discourse.rs)      │      │ Engine (comprehension.rs)     │
        │  coreference, animacy memory, │      │  the SYNTHESIZED programs:     │
        │  mention recency, asserts to  │      │  11 base Mog programs + verb   │
        │  the World                    │      │  wrappers, composed + executed │
        └───────────────┬───────────────┘      │  in-process. No Python.        │
                        │                       └──────────────┬────────────────┘
        ┌───────────────▼───────────────┐                     │ solve_problem (verified)
        │ World model (world_model.rs)   │                     ▼
        │  three-valued open-world truth │      ┌──────────────────────────────┐
        │  facts · taxonomy · orderings  │      │ Self-improvement substrate     │
        │  · attitudes · causals ·       │      │  (src/self_improve/)           │
        │  conditionals · contradictions │      │   gate · journal · store ·     │
        └───────────────┬───────────────┘      │   extend (the learning loop)   │
                        │                       └──────────────────────────────┘
        ┌───────────────▼───────────────┐      ┌──────────────────────────────┐
        │ semantics.rs : parse English   │      │ inference.rs : entailment      │
        │   → Meaning                    │      │   relation · consequences ·    │
        │ qa.rs : answer + proof          │      │   prove → Proof · render_proof │
        └────────────────────────────────┘      └──────────────────────────────┘
```

### 2.1 The Engine — understanding built on synthesized programs

`Engine::new()` synthesizes and composes the base programs **at startup**, each
recovered by a teacher and **verified against its examples** by `solve_problem`
(if synthesis fails, construction panics — the engine is never half-built):

| Component | What it decides | Recovered as |
|---|---|---|
| `noun_animacy` | is a word a noun, and animate(1)/inanimate(2)/non-noun(0) | string→int lexicon (incl. eventive nouns mined from `PROP_PAIRS`) |
| `valid_roles` / `valid_agreement` | selectional restriction & subject–verb agreement | DNF over morpheme tokens |
| `ends_s` | plural/3sg `-s` detector | suffix rule |
| `regular_3sg` / `irregular_3sg` | verb→3sg (walks / has) | suffix transduction + whole-word lexicon |
| `regular_past` / `irregular_past` | verb→past (walked / wrote) | suffix transduction + lexicon |
| `prop_id` / `has_negation` / `valid_argument` | proposition id, negation cue, argument validity | lexicon / contains-rule / DNF |
| `verb_3sg`, `verb_past` (wrappers) | compose irregular-lexicon-first, regular-rule-else | hand-composed over the above |

`Engine.methods: Vec<(component, teacher)>` records **which teacher recovered each
component** — this is the latent metacognition `explain_self` surfaces. `Engine`
derives `Clone`, and cloning is a cheap `String`+`Vec` copy that re-runs **zero**
synthesis — this is what makes candidate engines cheap to build for the gate.

### 2.2 The Meaning representation (`meaning.rs`)

A parsed sentence is a `Meaning`:

- `Event { predicate, agent, patient, recipient, tense, aspect, negated }`
- `IsA`, `HasProperty`, `Comparison { subject, scale, more, than, negated }`
- `Quantified { Every/Some/No, var_category, body }`, `Cardinal { at_least, … }`
- `Attitude { holder, verb, content, negated }` (know / believe / …)
- `Modal { Can/Must/Might/Should, body, negated }`
- `Temporal { Before/After, first, second }`, `Causal { cause, effect }`
- `Or`, `Not`, `YesNoQuestion`, `WhQuestion`, `CountQuestion`, `DegreeQuestion`
- `Unknown(text)` — the honest "I could not parse this"

Terms: `Entity` / `Indefinite` / `Pronoun` / `Restricted { head, clause }` (relative
clauses). Tense `{ Past, Present, Future }`, plus `Aspect { Simple, Progressive,
Perfect }`.

### 2.3 The World model (`world_model.rs`)

A small **model-theoretic, three-valued, open-world** store. `holds(meaning) ->
Option<bool>`: `Some(true)` proven, `Some(false)` proven false, `None` genuinely
unknown (the open-world default — *absence of a fact is not a false fact*). It
holds event facts, a taxonomy (hypernym chains: teacher ⊑ person ⊑ agent),
comparison orderings (with transitive closure), attitudes, causal links, and a
**contradiction ledger**. `comparison_facts()` exposes the ordering edges so the
proof layer can reconstruct transitive chains.

### 2.4 Inference (`inference.rs`) and answering (`qa.rs`)

`relation(premise, hypothesis) -> Entails | Contradicts | Neutral`, built on
`consequences` (the sound things a meaning entails: aspect-reduction,
drop-patient, generalize-agent, taxonomy-hypernym, comparison-converse/transitivity,
…). `qa::world_truth_traced` runs the full truth cascade and returns *both* the
verdict and a `Proof`; `qa::answer` is the user-facing string.

---

## 3. How it reasons — and shows its work

`prove(facts, goal) -> Option<Proof>` does **bounded forward-chaining** and returns
a derivation tree (`Proof { conclusion, rule, premises }`); `render_proof` turns it
into English. `mind.why(question)` answers *and* shows the chain:

```
why("is the report longer than the letter?")
 => "Yes, the report is longer than the letter. the report is longer than the
     letter because you told me the report is longer than the book and you told me
     the book is longer than the letter (by comparison-transitivity)."
```

A leaf is `"you told me …"` (a real asserted fact); each internal step names the
sound rule it used. The system **never emits a proof of a falsehood** and never
fabricates a `because` for an unprovable query.

**Soundness invariants enforced and adversarially verified** (each has a
default-to-fail probe):

- `must ⊨ can` but `can ⊭ must`, and possibility `⊭` actuality (modal monotonicity)
- temporal `before`/`after` are transitive and **asymmetric** (A▸B ⊬ B▸A)
- causal is **non-commutative** (cause ≠ effect; `why` reads the link one-way only)
- factive `know P ⊨ P`, non-factive `believe P ⊭ P`
- negation scope: `not every X P` (wide) ≠ `every X does not P` (narrow)
- comparison transitive **and** asymmetric; cardinal "at least N" monotone
- **modus ponens / tollens** over a stated `if P then Q` (with a named proof chain),
  and the classic fallacies — affirming the consequent, denying the antecedent —
  are **refused** (both answer "I don't know", never yes/no)
- **belief revision**: on an F-vs-¬F clash the world resolves to *one* coherent
  belief (a direct assertion outranks a derived one; two direct → most-recent-wins;
  the superseded belief is retracted and recorded). The store never holds F and ¬F
  both true; revision only ever moves from incoherent to coherent. See
  `mind.revisions()`.

---

## 4. How it reflects on itself (metacognition)

| Surface | What it does |
|---|---|
| `explain_self(topic)` | prints the **actual synthesized Mog source** of the mapped component + the teacher that recovered it — the system explaining its own learned algorithm |
| `what_do_you_know(entity)` | every asserted fact + sound derived consequence about an entity |
| `gaps(question)` | the genuine open-world unknowns blocking an answer |
| `what_would_change_your_mind(q)` | names only flips that **genuinely** move the verdict (tested against the real monotone pipeline); for a proof-backed answer it reports the honest *dependency* ("rests on what you told me …; you cannot un-tell me") rather than a false promise |
| `suppose(assumption, q)` | hypothetical reasoning on a discourse **snapshot** — never leaks into the real world |
| `what_if_not(fact, q)` | counterfactual via asserting the negation in a clone |
| `explain_cause(q)` | abductive best-explanation from a stored causal link / an entailing fact; honest "I don't know why." otherwise |
| `self_check()` | re-runs the regression gate against this mind's engine |
| `contradictions()` | the inconsistencies the world has flagged |

---

## 5. How it learns concepts (the heart of the system)

The synthesizer that nCPU already trusts is the mechanism by which the
comprehension engine **grows new abilities at runtime** — bounded and sound.

### 5.1 The loop

```
detect_gap(input)            a word/construction it cannot handle (Unknown / unknown noun)
      │
      ▼
propose_curriculum(gap)      mine I/O examples from in-repo curriculum data
      │                      (the "curriculum-mined" teacher source)
      ▼
solve_problem(examples)      SYNTHESIZE a Mog program — and VERIFY it on the examples
      │
      ▼
Engine::try_extend           graft the verified program onto a CHEAP candidate clone
      │
      ▼
regression_gate(candidate)   31 curriculum-mined golden cases + a soundness oracle;
      │                      accept iff ok() == (passed == total && sound)
      ▼
journal::record + store      audit the attempt (accept OR reject) and PERSIST on accept
      │
      ▼
live engine ← candidate      adopt ONLY on a green gate → growth is MONOTONE
```

Driven autonomously by `mind.study(corpus, max_rounds)` — detect → propose →
self_improve per sentence, **loop until a round learns nothing new** (loop-until-dry).
`mind.self_improve(LearnRequest)` runs one cycle and replaces the live engine only
on acceptance.

### 5.2 Why this is sound, not hand-waving

1. **Verified before integration.** A learned program is proven on its examples by
   `solve_problem` before it is even a candidate.
2. **Gated before adoption.** The candidate must pass the *full* regression gate.
   The gate is **proven to discriminate** — a mutation test forcing `ok()` to
   always-true is caught, and a verified-but-*shadowing* component (one that
   redefines a base lexicon with wrong labels) is rejected (8/31 golden cases), the
   base engine left byte-for-byte unchanged.
3. **Monotone growth.** Anything that breaks an existing behavior is rejected. The
   engine only ever gains abilities.
4. **Domain-bounded verdicts.** A learned classifier's *generalization* is not
   verified — only its behavior on the training examples is. So membership verdicts
   and parser recognition read the **verified example domain** (`Engine.learned_members`),
   never the synthesized program's extrapolation. For a word it was never trained
   on → `I don't know`, always. *This is what prevents an overfitting suffix-rule
   classifier from emitting a false "Yes" on an unseen word.*
5. **Persisted safely.** Accepted components are written to a JSONL store
   (`~/.ncpu_learned_components.jsonl`, env-overridable, empty-disables). On the next
   `Engine::new()` they are reloaded **and re-gated** — disk is untrusted input, so a
   stale/poisoned row is rejected on boot and the engine stays sound.
6. **Functionally integrated.** A learned `<x>_class` classifier is consulted by the
   parser's noun-head predicate and by the answerer, so a self-taught word actually
   *parses and answers* — not merely `eval`-callable.

### 5.3 The learning, demonstrated end-to-end (`comprehend study` / `grow`)

- **Concept acquisition:** the mind cannot classify creatures → `study` detects the
  gap → mines examples → synthesizes a verified `creature_class` → gate 31/31 →
  adopts it (`self_check` stays green) → now "the dragon flies" parses and "is the
  dragon a creature?" → **Yes**, while unseen non-creatures stay `idk` (no false Yes).
- **Cumulative across restarts:** a brand-new `Engine::new()` ("a restart") on the
  same store **boots with `creature_class` already present**, re-gated green, without
  re-studying.
- **Safe refusal:** a contradictory spec fails synthesis and is declined with the
  base engine intact; every attempt — accept or reject — is in the journal.

---

## 6. The safety substrate (`src/self_improve/`)

| File | Role |
|---|---|
| `gate.rs` | `regression_gate(&engine) -> GateReport`. 31 curriculum-mined golden behavioral cases + a soundness oracle (no-spurious-entailment, fact recall + negation, must-monotonicity, causal non-commutativity). `ok() = passed==total && sound`. **The guard every self-modification passes.** |
| `journal.rs` | append-only JSONL audit of every learning attempt (gap, method, verified, regression_passed, accepted). The system's memory of its own growth. |
| `store.rs` | persistent learned components; reload **+ re-gate** on boot; `members` field carries the verified domain so cross-run answers stay sound. |
| `extend.rs` | `self_extend` — the synthesize→gate→journal→persist funnel; `detect_gap`/`propose_curriculum`/`study` types. |

Plus two property-based fuzzers (deterministic, seeded LCG, no `rand`):
- `tests/soundness_fuzz.rs` — 240 random small worlds asserting asserted-fact
  recall, negation-consistency, no-spurious-entailment. **0 violations.**
- `tests/metamorphic_fuzz.rs` — 180 worlds asserting **paraphrase invariance**:
  active⟺passive and comparative-converse phrasings get the *same* verdict.
  **0 disagreements.**

---

## 7. How it performs (verified)

Measured at **40e8a70** (Phase E; `cargo test --release`, store disabled):

| Suite | Result | What it covers |
|---|---|---|
| `understanding` lib | **306 / 0** | parser, world model, inference, qa, discourse, mind, proofs, reflection, conditionals, belief revision |
| `grammar` lib | **8 / 0** | learned constructions (induction + apply + skeleton match) |
| `self_improve` lib | **25 / 0** | gate (discrimination, shadow-rejection, construction-collision), journal, store reload/poison, extend, construction persist+gate |
| `comprehension` lib | **6 / 0** | engine + curriculum + learned-classifier + token_classes |
| `eval` (FraCaS-style) | **7 / 0** | the entailment benchmark + feedback loop |
| `adversarial_*` | **60 / 0** (12 suites) | soundness traps incl. learned-construction-on-unseen |
| `soundness_fuzz` / `metamorphic_fuzz` | **240 + 180 iters / 0 violations** | property-based soundness + paraphrase invariance |

**Entailment benchmark** (`comprehend bench`) — 40 in-vocabulary cases across 9
FraCaS phenomena (quantifiers, comparatives, attitudes, negation, temporal,
conjunction, cardinality, aspect, taxonomy), golds mixed (Yes 24 / No 5 / Unknown 11):

```
OVERALL  40 correct · 0 idk · 0 WRONG · 100% accuracy · SOUND (WRONG = 0)
```

**The autonomous loop bites** (`bench → study → bench`): the edge-of-competence
`learned` section goes correct 0→1; overall **40 → 41, accuracy 97.6% → 100%, idk
1 → 0, wrong 0 → 0**, monotone and sound throughout — `study` autonomously mined
`creature_class` and closed the measured gap. *Measure → autonomously learn →
measurably improve.*

---

## 8. Commit timeline

| Commit | Milestone |
|---|---|
| `251cacf` | grammatical core (aspect, modality, relatives, passive, plurals, temporal, causal, degree-Q, negation scope) |
| `7745b29` | reasoning + autonomous self-improvement (proof traces, safety substrate, metacognition, self-extension loop) |
| `c1cc09a` | **autonomy spine** — persistence (reload + re-gate), auto gap detection, self-curriculum, `study` loop, provenance |
| `0cbd296` | **external validation** — FraCaS-style benchmark + dashboard + bench→study feedback |
| `f8bdcd6` | **functional integration + breadth** — learned classifiers drive parsing/answering (domain-bounded), relative-clause questions, possessives, PPs; the loop bites |
| `3b83ad4` | **D** — conditional/syllogistic reasoning (modus ponens/tollens, fallacies refused) + actionable belief revision + metamorphic paraphrase fuzzer |
| `40e8a70` | **E** — **grammar induction**: the parser learns a new construction (object-fronting / OSV) as a synthesized + gated verified rule; generalizes to unseen words; survives restart; never corrupts a base parse (`no_construction_collision` gate probe) |
| *(greenlit)* | **F** hybrid LLM verifier · **G** open-vocabulary at scale · **H** generation |

---

## 9. Honest limits (what it cannot yet do)

- **Self-extension now reaches both lexicon AND grammar.** The system teaches
  itself concept classifiers *and* new grammatical constructions (parse rules as
  synthesized, verified programs — object-fronting / OSV demonstrated, generalizing
  across unseen words, gated by `no_construction_collision`, persisted across
  restarts). The base parser (`semantics.rs`) is still hand-written; learned
  constructions are consulted only as a fallback when it returns Unknown, so they
  *fill gaps* and never override a correct parse. Open-ended grammar acquisition
  (many constructions from raw text) is the next scale step (Thrust G).
- **Learned classifiers answer within their verified domain.** This is a soundness
  choice, not a bug: beyond the proven examples the answer is honestly `idk`.
- **The benchmark is in-vocabulary.** It measures *sound coverage* of the 9
  phenomena, not raw-FraCaS breadth — which is gated on vocabulary the autonomy loop
  grows over time.
- **No graded confidence yet** — answers are three-valued (Yes / No / idk). Belief
  revision (Phase D) adds provenance weighting but not a continuous confidence score.

---

## 10. Quick start

```bash
# the demos (each self-contained; set their own temp store/journal)
cargo run --release --bin comprehend -- understand   # comprehension + reasoning + reflection
cargo run --release --bin comprehend -- bench        # FraCaS-style dashboard + feedback loop
cargo run --release --bin comprehend -- study        # autonomous learning, cumulative across "restarts"
cargo run --release --bin comprehend -- grow         # one self-extension cycle (accept + safe reject)

# the test surface
cargo test --release --lib understanding self_improve comprehension eval
cargo test --release --test 'adversarial_*' --test soundness_fuzz
```
