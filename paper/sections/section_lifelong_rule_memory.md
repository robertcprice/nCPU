# Lifelong Rule Memory and Recovered Grammar

## Thesis

The companion section on rule-compressed memory (Section: *Rule-Compressed Memory:
Bounded Storage, Unbounded Reach, Zero Forgetting*) made a claim about a single
rule — pluralization — streamed as `(word → form)` pairs. This section closes
three remaining gaps in that argument, each measured end-to-end against the
LinguaGenesis curriculum with the real `mog_synth` binary:

1. **The grammar is extensible.** We recover three further *agreement* checkers —
   subject–verb **number** agreement, **copular** BE agreement, and a sentence-level
   **past-tense** grammaticality judge — each as a verified Mog program. These are
   not new mechanisms; they are new entries in the same DNF-teacher hypothesis
   class used for 3sg agreement, demonstrating that the recovered grammar *grows*
   by adding verified programs rather than retraining anything.
2. **The library is lifelong and multi-domain.** Three string-transduction domains
   (pluralize, reverse, verb-3sg) are learned *in sequence* into one persistent
   on-disk library, and every earlier domain is re-tested after each new one is
   added. We measure **zero catastrophic forgetting** — and explain precisely why
   that result, while real, is *structural* rather than a hard-won training
   property.
3. **Rule memory holds at the sentence level.** We extend the word-transduction
   memory experiment to a *judgement* task — streaming labeled grammaticality
   decisions over sentences — and show the same storage/reach/forgetting profile,
   including correct verdicts on verbs the memory never saw.

Throughout, we tie back to the **two-pillar memory** of the rule-memory section:
the *positive* pillar (rules + learned biases — compress the regular, unbounded
reach, bounded storage, no decay) and the *negative* pillar (Hamilton
mistake-memory + rejected cache + exceptions — remember the irreducible, never
repeat an error). Every result below is an instance of one or both pillars.

---

## 1. Recovered agreement checkers — verified programs that extend the grammar

The curriculum rule-learning section established a ladder of "teachers", each a
strictly larger hypothesis class, culminating in `search_array_dnf` (disjunction
of conjunctions over feature tokens). All three checkers here are recovered by
that same DNF teacher, over a small perception-computed feature array, in **one
synthesis call each** — first try, no retry loop, well under the ~6-call budget.
The pattern is identical to the section's `task_sentence_3sg_general` recipe:
*perceive the load-bearing features, synthesize a DNF over them, verify on a
held-out split and against the curriculum.*

### 1.1 Subject–verb number agreement

**Target rule.** A sentence is grammatical iff *(singular subject ∧ verb carries
a 3sg suffix) ∨ (plural subject ∧ no 3sg suffix)* — an XOR over two features.

**Recovered program** (`number_agreement.py`, `method=search_array_dnf`):

```
fn number_agreement_ok(arr: [i64]) -> i64 {
    t0 = 0; t1 = 0;
    for x in arr {
        if x == 902 { t0 = 1; }   // subject-is-plural
        if x == 905 { t1 = 1; }   // verb-carries-3sg-suffix
    }
    if t0 == 1 { if t1 == 0 { return 1; } }   // plural ∧ ¬3sg
    if t0 == 0 { if t1 == 1 { return 1; } }   // singular ∧ 3sg
    return 0;
}
```

**The load-bearing perception decision.** We did *not* feed the raw morpheme token
stream to the teacher. The plural noun "dogs" emits a `<+s>` suffix token (id 106)
that **collides** with the verb's own 3sg `<+s>` — so token-presence alone cannot
distinguish a plural subject from a 3sg verb, a noisy positional problem that
would make the DNF teacher time out. The perception layer instead computes two
clean boolean features in Python: `902` (subject is plural) and `905` (verb
carries a 3sg suffix, recomputed against the curriculum's own 3sg forms,
independent of the label). That turns the rule into a 2-feature XOR-shaped DNF,
which the teacher recovers exactly.

**Measured (re-run end-to-end against the live binary, 82 NOUN_FRAMES).**
`train_n=248` (124 pos / 124 neg), `holdout_n=80`, `holdout_accuracy = 1.0`
(80/80).

**Honest reading of the metric.** The feature space is two booleans — only **four**
distinct feature vectors exist — and the stratified split places all four cells in
*both* train and holdout by design. So the headline `1.0` is **not** measuring
extrapolation to unseen feature combinations; it confirms the learned per-cell
mapping reproduces across different surface sentences. The stronger evidence is an
*exhaustive* check: the recovered Mog program agrees with the oracle on **every**
one of the four possible inputs over the full 2-bit feature domain (4/4), so there
is no unseen cell it could get wrong. Independent adversarial harnesses confirmed
robustness: a full-curriculum sweep of all 82 frames × 4 sentence cells scored
328/328; 11/11 perturbations passed (noise tokens, duplicate features,
feature-order reversal, the empty array, and the `<+s>` collision trap — raw token
106 present without 905 — all correct); and the perception layer is correct on all
328 sentences, with the one spelling collision found ("cooks" is both a 3sg verb
and a noun plural) not leaking because perception keys on the last word, always the
verb. **Verdict: the recovered program is genuinely correct over its entire input
domain, not memorized.**

### 1.2 Copular BE agreement — and a recovered shortcut

**Target rule.** Grammatical iff *(singular ∧ "is" ∧ ¬"are") ∨ (plural ∧ "are" ∧
¬"is")*.

**Recovered program** (`copular_agreement.py`, `method=search_array_dnf`):

```
fn copular_agreement_ok(arr: [i64]) -> i64 {
    t0 = 0; t1 = 0;
    for x in arr {
        if x == 902 { t0 = 1; }   // subject-is-plural
        if x == 903 { t1 = 1; }   // contains "is"
    }
    if t0 == 1 { if t1 == 0 { return 1; } }   // plural ∧ ¬is
    if t0 == 0 { if t1 == 1 { return 1; } }   // singular ∧ is
    return 0;
}
```

Sentences "The `<subject>` `<copula>` `<adjective>`." were built from curriculum
word lists only (subjects from `determiners_number.NOUN_FRAMES`, adjectives from
`basic_clause_frames.COPULAR_FRAMES`); the label is a structural oracle (no
per-sentence hand-authoring), because the curriculum exposes no
`validate_copular_agreement` function. Perception reduces each sentence to a sorted
feature array over `902` (plural), `903` ("is"), `904` ("are").

**Measured (deterministic, two identical runs).** `train_n=984` (492 pos / 492
neg), `holdout_n=328`, `holdout_accuracy = 1.0` (328/328), one synthesis call.

**Honest reading — this one is brittle, and we report it as such.** The narrow
claim (holdout `1.0`) reproduces exactly and is real. But adversarial probing of
the *recovered program* shows it does **not** capture the general agreement rule.
The teacher found a simpler-but-in-distribution-equivalent DNF that inspects only
tokens `902` and `903` and **never references `904` ("are") at all**. Its effective
rule is *(plural ∧ ¬is) ∨ (singular ∧ is)*. Within the checker's dataset this is
provably identical to the target, because every sentence carries exactly one copula
(`is` XOR `are`), so "¬is ⇒ are". But on legitimate out-of-distribution inputs
expressible in the same feature vocabulary the program breaks (2 of 4 OOD cases
wrong against a true oracle):

* `[902]` — plural subject, **no** copula ("The dogs happy") → program returns
  VALID, correct is INVALID.
* `[903, 904]` — singular subject with **both** "is" and "are" → program returns
  VALID, correct is INVALID.

The root cause is structural and confirmed by inspecting the code: all 1 312 rows
collapse to only four `(combo, label)` classes, with zero missing-copula and zero
double-copula rows, so the i.i.d. holdout cannot punish a program that ignores
`904`. **Verdict: the holdout number is honest within its stated scope, but it
measures memorization of four classes, not generalization of the agreement rule.**
To make the recovered program robust, the checker's data must add missing-copula
and double-copula negatives and draw OOD holdouts that force the DNF to consult
token `904`. This is a perception/data-coverage fix of exactly the kind the
curriculum section names as the research boundary, not a failure of the synthesis
machinery — the machinery faithfully recovers the *simplest rule the data
admits*, which is the point and the pitfall.

### 1.3 Sentence-level past-tense grammaticality

**Target rule (within declared scope).** Grammatical iff *`<+ied>` ∨ (`<+ed>` ∧ ¬
consonant-y-stem)*.

**Recovered program** (`past_sentence.py`, `method=search_array_dnf`):

```
fn valid_past_sentence(arr: [i64]) -> i64 {
    t0 = 0; t1 = 0; t2 = 0;
    for x in arr {
        if x == 102 { t0 = 1; }   // <+ied>
        if x == 105 { t1 = 1; }   // <+ed>
        if x == 900 { t2 = 1; }   // consonant-y stem feature
    }
    if t1 == 1 { if t2 == 0 { return 1; } }   // <+ed> ∧ ¬y-stem
    if t0 == 1 { return 1; }                  // <+ied>
    return 0;
}
```

Clause frames ("The student studied.") are built from the curriculum: subjects
from `Stage5TenseAspectGenerator`'s `grammar.tense.past` examples, verbs and
correct past forms from `morphology_productivity.REGULAR_VERBS`. Each verb yields
three rows — correct past (positive), over-regularized wrong suffix
(*studyed* / *walkd*, negative), bare stem (negative). Perception keeps only the
past-suffix tokens (`<+ied>=102`, `<+ed>=105`, `<+d>=107`) plus the synthetic
consonant-y-stem feature (`900`); determiner/subject/period tokens are dropped as
noise. Stratified by `(feature-combo, label)`: 355 train / 116 holdout, all six
distinct feature combos in both splits, 0 label conflicts (fully separable).

**Measured (deterministic).** `holdout_accuracy = 1.0` (116/116). A Python replica
of the recovered program is bit-identical to the Mog binary on 41 edge/random
arrays. Evaluated on a *fresh, independent, in-scope* grid — 157 verbs × 22 real
curriculum subject frames × 3 error-types = **10 362 cases not in the holdout** —
the program scores **100%** (pos 3454/3454, wrong 3454/3454, bare 3454/3454).
This is genuine generalization within scope, not memorization of the six-combo
holdout.

**Honest scope and caveats.** The claim reproduces *within the rule's declared
scope* — regular and consonant-y verbs whose past tokenizes into the 100–108
suffix band, which the module docstring states explicitly. Two caveats sit
*outside* that scope:

* On a broader fresh adversarial set mixing out-of-scope verbs the program scores
  42/45 = **0.933**. All three failures are grammatical e-stem pasts
  (*danced / hoped / smiled*) that the morpheme tokenizer stores as **atomic
  whole-word IDs**, so the encoder emits `[]` and the rule false-rejects. This is a
  *perception-layer* artifact upstream of the rule, not a rule error.
* The original target named a third disjunct, `<+d>` (for e-stems). The curriculum's
  `REGULAR_VERBS` contains **no** e-stem verb whose correct past tokenizes to
  `<+d>=107` — "liked" renders as `lik+<+ed>=105`, not `like+<+d>=107`. So the
  `<+d>` token only ever appears as the *wrong* "walkd" error and is correctly a
  negative feature here. The e-stem branch is **not exercisable** from this
  lexicon; we report this honestly rather than claim a general past-tense judge.
  Irregular pasts (*went / ate / ran*) are correctly out of scope.

**Verdict: a sound, well-scoped regular-past DNF that generalizes to thousands of
unseen in-scope cases — not a general past-tense grammaticality checker.**

### 1.4 What these three add to the grammar

All three are *new verified programs in the same hypothesis class*, recovered by the
same teacher, over the same perception→synthesis→verification loop. They extend the
recovered grammar from inflection (3sg, plural, gerund, past) into **agreement**
(number, copular BE, past-tense well-formedness) — and each is inspectable,
executable code, deployable as a constraint exactly like the existing 3sg and
pluralize rules. The honest line across all three is the section's research
boundary made concrete: *the rule layer is exact within the structure perception
makes visible* — number agreement is exact over its full domain; copular
agreement is exact only over the data's four-combo structure and needs richer
negatives; past-tense is exact within the suffix-band scope and limited only by an
upstream tokenizer artifact. The difficulty is never the search; it is *what
features perception exposes*.

---

## 2. A lifelong, multi-domain rule library with zero catastrophic forgetting

`lifelong_library_experiment.py` learns three string-transduction domains **in
sequence** into **one** persistent on-disk memory
(`lifelong_library.json`), then re-tests every earlier domain after each new one is
added. Each domain's rule is a verified Mog program synthesized via string-output
`--problem-json` (`fn <name>(s: string) -> string`).

**Domains** (curriculum-sourced, stratified 75/25 with zero train/holdout
overlap):

* **D1 pluralize** — 60 train / 20 holdout, balanced across `-s` / `-es` / `-ies`
  buckets (irregulars excluded as irreducible).
* **D2 reverse** — 44 train / 16 holdout.
* **D3 verb-3sg** — 48 train / 16 holdout, balanced across sibilant / consonant-y /
  regular classes (oracle = curriculum `_correct_3sg_form`).

**Measured (re-run from scratch twice, fully deterministic, identical both times).**

| Domain | train acc. | holdout acc. | size |
|---|---|---|---|
| pluralize | 100.0% | 100.0% | 307 B |
| reverse | 100.0% | 100.0% | 60 B |
| verb-3sg | 100.0% | 100.0% | 536 B |
| **library total** | | | **903 B** |

Zero-forgetting check: each domain learned@100.0% → final@100.0%, **drift +0.0%**;
the accuracy-matrix lower triangle stays flat at 100% (after D2, pluralize holdout
still 100.0%; after D3, both pluralize and reverse still 100.0%). Catastrophic
forgetting observed: **no**. Budget: 3 synthesis calls (one per domain), all
succeeded on the first try.

**The programs are general rules, not lookup tables** — this is what makes the
result meaningful. pluralize branches on `ends_with h/s/x → +es`, `ends_with y →
slice(0,len-1)+"ies"`, else `+s`; reverse is `s.reverse()`; verb-3sg discovered the
vowel-y sub-rule ("ay"/"oy" stay `+s`; ch/sh/s/x/z → `+es`; consonant-y → `ies`;
else `+s`). They generalize to unseen holdout words
(*abolish→abolishes*, *apply→applies*, *blesses*), and the synthesized programs were
checked against an independent from-scratch Python oracle on the holdouts
(20/20, 16/16, 16/16 match), so the 100% is not an artifact of a quirky curriculum
oracle. The stratified holdouts place all three orthographic sub-rules in *both*
splits, so 100% holdout is a real generalization test, not a degenerate easy case.
The re-evaluation is live, not cached: corrupting the in-memory pluralize program
(`return s+"s"` → `s+"ZZZ"`) drops its matrix accuracy 100% → 50%, proving the
lower-triangle genuinely re-runs each earlier program after later domains are
added.

**Honest reading — zero forgetting here is *structural*, by construction.** Each
domain is an immutable dict entry keyed by name; adding a new key cannot mutate an
existing one, so drift is *guaranteed* to be 0. There is **no neural
sequential-training baseline** in this experiment, so the result makes no
comparative numeric claim that could be unfairly computed — it demonstrates that a
**verified-program memory is immune to overwrite**, which is the intended point and
a weaker statement than "a network retained accuracy after sequential training."
This is precisely the *Persistence without drift* property named in the rule-memory
section: a rule is discrete code, it does not decay and cannot be evicted, so there
is no shared weight matrix for domain *N* to overwrite in domains 1..*N*−1. The
experiment empirically confirms that structural guarantee; it does not pretend to
have beaten a neural lifelong learner at its own game.

This is the **positive pillar** of the two-pillar memory, instantiated across
domains: *N* verified primitives stored once, retrieved forever, composing into the
larger library (903 bytes for three full grammatical sub-systems) with no
interference between them.

---

## 3. Rule memory over sentences (Tier C) — the judgement-task analog

The rule-memory section streamed word→form *transductions*. We now stream labeled
*grammaticality judgements* over whole sentences and ask the same question: does a
rule memory beat a retrieval store and a context window on storage, recall, and
generalization?

**Setup** (`sentence_memory_experiment.py`, mirroring the Tier-A harness for a
judgement task). A sentence-level checker
(`checkers/sentence_3sg_grammatical.py`) first recovers subject–verb 3sg-agreement
grammaticality as a verified DNF — *valid iff `<+es>` ∨ (`<+ies>` ∧ ¬sibilant) ∨
(`<+s>` ∧ ¬sibilant)* — over the morpheme suffix-band tokens plus one computed
stem-is-sibilant feature (901). Measured: `holdout_accuracy = 1.0` (50/50), one
synthesis call. Then we stream **992** grammaticality-labeled 3sg sentences
(496 pos / 496 neg) chunk-by-chunk, and hold out **256** sentences from a withheld
20% verb block (every 5th regular verb) — never shown to any memory. Three memories
are compared, exactly as in the word experiment:

* **RULE** (ours): the synthesized DNF classifier + a tiny exception table; storage
  = `|code| + |exceptions|`. Re-synthesized only on a mis-prediction.
* **INSTANCE** (RAG): store every sentence→label.
* **WINDOW-150** (LLM): keep only the last 150 sentences.

**Measured (`--stream 1000 --window 150 --chunk 50`, seed 42; regenerated CSV is
byte-identical to the artifact).**

| | RULE (ours) | INSTANCE (RAG) | WINDOW-150 (LLM) |
|---|---|---|---|
| storage | **474 B — flat from chunk 1, 0 exceptions, 1 synth** | 17 792 B — linear (**37.5×**) | ~2 917 B (bounded) |
| recall on all 992 seen | **100.0%** | 100.0% | **15.1%** (forgot the rest) |
| generalize to 256 unseen-verb | **100.0%** | **0.0%** | **0.0%** |

The rule store is **37.5× smaller and flat**, never forgets (100% on sentences long
past any window), and answers an unbounded set of sentences — *any verb, seen or
not* — at 100%, because it judges the suffix+sibilant feature cell, not the verb
identity. The instance store grows linearly and generalizes 0%; the window forgets
85% of what it saw and generalizes 0%. An independent audit found **zero leakage**:
no shared sentences, verb bases, or surface 3sg forms between the 992 stream and the
256 unseen set; a fresh from-scratch synthesis scored 992/992 seen and 256/256
unseen at 474 bytes with 0 exceptions.

**Both pillars are visible in one run.** The *positive* pillar is the 474-byte
flat-storage rule itself. The *negative* pillar — **Hamilton mistake-memory** — is
the mistake log: **50 distinct misjudgements**, all occurring *before the rule was
synthesized* (chunk 1), and **0 repeats** thereafter. Once the rule was recovered,
the cumulative count of distinct mistakes converged to that finite bound and the
repeat-mistake rate was zero — the exact "monotonically self-improving, never
repeats a mistake" property the rule-memory section attributes to the negative
pillar, now demonstrated at the sentence level.

**Honest reading.** INSTANCE's seen=100% is its hard-coded best case (it literally
stores every sentence) but costs 17 792 bytes; INSTANCE/WINDOW unseen=0% is
legitimate, since they key on strings never stored. The 100% unseen for RULE is
genuine generalization via abstracted suffix+sibilant features, not leakage. The
task's eight feature cells are label-unambiguous, so it is cleanly separable —
slightly easier than the framing implies — but this *inflates no reported number*:
the rule legitimately reaches 100% with 0 exceptions because the regular part of the
judgement stream is genuinely a function of the features perception exposes.

---

## 4. Synthesis — what these results establish together

Read together with the rule-memory section, these three results complete a single
argument:

1. **The grammar is an extensible library of verified programs.** Three new
   agreement checkers (number, copular, past-tense) join the existing inflection
   rules, recovered by the same DNF teacher in one synthesis call each. Two of them
   (number, past) are exact and generalize within their declared scope; the third
   (copular) is exact only over its data's structure and honestly exposes the
   research boundary — the recovered rule is only as general as the negatives the
   data provides. The limit is always *perception*, never the search.
2. **A program memory is lifelong by construction.** Three domains learned in
   sequence into 903 bytes with +0.0% drift is *structural* zero-forgetting — a
   verified-program library cannot overwrite itself because there is no shared
   weight matrix. We report this as the modest, exact statement it is, not as a
   defeated neural baseline.
3. **The bounded-storage / unbounded-reach / zero-forgetting profile holds at the
   sentence level.** A 474-byte rule judges 992 seen + 256 unseen-verb sentences at
   100% while RAG grows 37.5× and a window forgets 85%, and the Hamilton
   mistake-memory caps distinct errors at 50 with 0 repeats.

The two pillars are both present and both measured: the *positive* pillar (rules +
the lifelong library — compress the regular, bounded storage, unbounded reach, no
decay) and the *negative* pillar (Hamilton mistakes + exceptions — remember the
irreducible, never repeat an error). Their sum is a memory that grows only with the
genuinely new and never unlearns — now demonstrated not for one rule but for an
extensible, multi-domain, sentence-level grammar.

## Reproduction

```bash
# Agreement checkers (each: one mog_synth call, prints holdout accuracy)
python3 nsynth/scripts/checkers/number_agreement.py
python3 nsynth/scripts/checkers/copular_agreement.py
python3 nsynth/scripts/checkers/past_sentence.py

# Lifelong multi-domain library (zero-forgetting matrix + lifelong_library.json)
python3 nsynth/scripts/lifelong_library_experiment.py

# Sentence-level rule memory (Tier C); 20-checkpoint CSV
python3 nsynth/scripts/sentence_memory_experiment.py --stream 1000 --window 150 --chunk 50
```

All words and labels come from the LinguaGenesis curriculum
(`/Users/bobbyprice/projects/linguigenesis`); nothing is hand-authored. The
checkers and experiments are deterministic — every number above reproduced exactly
on independent re-runs against the live `nsynth/target/release/mog_synth` binary.
Artifacts: `nsynth/scripts/lifelong_library.json`,
`nsynth/scripts/lifelong_library_results.csv`,
`nsynth/scripts/sentence_memory_results.csv`.
