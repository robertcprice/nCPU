# LinguaGenesis × nsynth — rule-learning bridge

A proof that nsynth (the program synthesizer) **learns the LinguaGenesis
curriculum's actual rules** as verified Mog programs — from the curriculum's own
words and labels, nothing hand-authored. Spans **morphology → grammar →
reasoning** (see `paper/sections/section_curriculum_rule_learning.md`).

## Rule-class ladder (the teacher family)

Each rule needs a particular program *shape*; each teacher is a strictly larger
hypothesis class, all exact + verifying:

| Bridge task | Teacher | Rule shape | Recovered |
|---|---|---|---|
| `verb_3sg_es` | `search_suffix_class` | OR of `ends_with` | `-es` after `ch/sh/s/x/z/o` |
| `pluralize_gen` | `morph_transduce` | suffix decision-list + exceptions | full plural incl. `y→ies` |
| `sentence_3sg` | `search_array_member_class` | OR of membership | "has valid 3sg token" |
| `sentence_gerund` | `search_array_conjunction` | one AND/NOT | `is` AND `<+ing>` |
| `sentence_past` | `search_array_dnf` | DNF + stem feature | `(<+ied>∧y) ∨ (<+ed>∧¬y)` |
| `formal_logic` | `search_array_dnf` | DNF | modus ponens/tollens validity |
| `sentence_full` | — (declines) | — | needs lexical features (honest ceiling) |

## What "a code synthesizer learns language" means here

LinguaGenesis defines each curriculum stage by a *validator* (a rule). nsynth's
job is to re-derive that validator as an executable program from labeled
examples. The language model (AR/diffusion) learns to *produce* language; nsynth
learns to *encode/verify the rule*. Verified rules can then constrain the model
(the KVRM registry idea) so it can't emit forms that violate them.

## The loop

```
curriculum rule ─labels─► (word/sentence, 0/1) ──► nsynth search ──► verified Mog program
(+ curriculum lexicon)        (rule never shown)                          (the rule, recovered)
```

Everything comes from `linguigenesis/v2/curriculum/morphology_productivity.py`:
the word list (`REGULAR_VERBS`), the oracle (`_correct_3sg_form`), and the
sentence labels (`Stage3...Generator().generate()`). The bridge
(`scripts/linguagenesis_bridge.py`) authors no data of its own.

## Two tasks, two honest outcomes

```bash
cd nsynth
# A: word-level — SOLVED
python3 scripts/linguagenesis_bridge.py --task verb_3sg_es  | ./target/release/mog_synth --problem-json -
# B: sentence-level — UNSOLVED (the frontier)
python3 scripts/linguagenesis_bridge.py --task sentence_3sg | ./target/release/mog_synth --problem-json -
```

**A — verb 3sg `-es` (word-level).** Oracle = the curriculum's own rule
(`morphology_productivity.py:395`, literally `base.endswith(("ch","sh","ss","x","z","o"))`).
nsynth recovers:

```
fn verb_takes_es_3sg(s: string) -> i64 {
    if s.ends_with("h") { return 1; }   // covers -ch, -sh
    if s.ends_with("s") { return 1; }   // covers -ss
    if s.ends_with("x") { return 1; }   // covers -x
    return 0;
}
```

Correct on all 63 curriculum verbs (47 train + 16 holdout). Note it is a
**subset** of the full rule: the curriculum's verb lexicon only contains
`-ch/-sh/-ss/-x` sibilants — no `-z`, no `-o`, and no non-sibilant `-h/-s/-x`
negative. So `h|s|x` is the simplest rule consistent with the curriculum's
*actual data*. nsynth learns exactly what the data supports — not arbitrary, but
not the complete English rule either. Recovering the rest needs richer examples,
which is precisely what LinguaGenesis's validator-in-the-loop / mistake-replay is
built to generate. **The learner is only as complete as the curriculum's data.**

**B — sentence grammaticality (sentence-level).** Examples straight from
`generate()`: `"The dog walks."` (1) vs `"The dog walk."` (0). nsynth returns
`diff_gradient_unsupported` — no single-string suffix separates grammatical from
ungrammatical sentences (positives end in `walks.` / `started.` / `talking.`).
This is the real frontier: the learner must tokenize the sentence, find the verb,
and check *its* morphology. That per-token feature extraction is the next
capability to build — not a single `ends_with`.

## The teacher (`search_suffix_class`)

`nsynth/src/solver/search_text_families.rs`. Generalizes the single-literal
teachers (`starts_with_literal`, `contains_literal`) to a *learned disjunction*
of suffixes:

1. mine suffixes (len 1–3) from positive examples,
2. keep only **admissible** suffixes (never fire on a negative → no false positives),
3. greedy set-cover the positives,
4. emit `if s.ends_with(...) return 1` chain, verify.

First teacher whose hypothesis class can express a morphological rule. Regression
test: `search_suffix_class_learns_sibilant_plural_rule` in `src/solver/tests.rs`.

## Known limit (next experiment)

- **Phonology blind spot**: orthographic suffix can't tell `ch`=/k/ (epoch → epochs)
  from `ch`=/tʃ/ (church → churches). The oracle has the same blind spot, so nsynth
  matches it — but real English needs a lexical exception list.
- **Irregulars excluded**: we test the *productive* rule (regulars only). Including
  irregulars (ox→oxen ends in -x but isn't -es) is a harder experiment that shows a
  pure suffix rule can't capture lexical exceptions — the regular/irregular split.

## Three levels, all working

```bash
cd nsynth
python3 scripts/linguagenesis_bridge.py --task verb_3sg_es   | ./target/release/mog_synth --problem-json -  # classify, full rule
python3 scripts/linguagenesis_bridge.py --task sentence_3sg  | ./target/release/mog_synth --problem-json -  # sentence-level
python3 scripts/linguagenesis_bridge.py --task pluralize_gen | ./target/release/mog_synth --problem-json -  # generate
```

**Level 1 — word classification, FULL rule.** `search_suffix_class`. After adding
the missing `-z` verbs (buzz/fizz/whizz) and `-h` hard negatives (laugh/cough) to
the curriculum's `REGULAR_VERBS`, nsynth recovers the complete sibilant set:
`ch / sh / s / x / z / o → -es`. 14 pos / 80 neg, 32/32 holdout.

**Level 2 — sentence-level grammaticality.** `search_array_member_class`. The
curriculum's **morpheme tokenizer** (`v2/tokenizer/morpheme_tokenizer.py`) encodes
a sentence to a token-id array with inflectional suffixes as explicit symbol
tokens (`<+es>`=104, `<+ies>`=101, ...). nsynth learns "grammatical iff the array
carries a valid `-es`-family 3sg token" — a set-membership rule it discovers from
labels alone:

```
for x in arr { if x == 101 { return 1; } if x == 104 { return 1; } } return 0;
```

Scope = sibilant 3sg, where the rule is cleanly membership-shaped (`sentence_3sg`).

**Level 2b — sentence-level CONJUNCTION.** `search_array_conjunction` (task
`sentence_gerund`). The progressive is grammatical iff `... is V-ing` — auxiliary
`is` AND the `<+ing>` suffix. Each negative drops one conjunct (`is V` or `V-ing`),
so neither token alone separates; only the conjunction does. nsynth learns it
(strictly beyond the OR-only member-class):

```
for x in arr { if x == 103 { r0 = 1; } if x == 109 { r1 = 1; } }
if r0 == 1 { if r1 == 1 { return 1; } } return 0;     // <+ing> AND is
```

75 train / 25 holdout. **Honest ceiling:** the *full* Stage-3 set also contains
wrong-stem errors ("copyed" vs "asked", both carry `<+ed>`) that are NOT a function
of the token set at all — separating them needs lexical/positional features, the
genuine next frontier. `sentence_full` is therefore not membership-separable and
nsynth correctly declines.

**Level 3 — generative morphology (PRODUCE the form).** `morph_transduce`
(`src/morph_transduce.rs`, additive — does not touch the i64 `Example`/verify
core). nsynth synthesizes a **suffix decision list with exceptions** (a general,
reusable learner) and verifies it by *executing the program and comparing output
strings*. From curriculum data it recovers the COMPLETE English plural rule,
including the `y→ies` stem change and the vowel+y exceptions (placed above the
general rule by specificity):

```
fn pluralize(s: string) -> string {
    if s.ends_with("ay") { return s + "s"; }              // vowel+y exception
    if s.ends_with("ch") { return s + "es"; }             // sibilant ...
    if s.ends_with("ey") { return s + "s"; }
    if s.ends_with("oy") { return s + "s"; }
    if s.ends_with("y")  { return s.slice(0, s.len - 1) + "ies"; }  // stem change
    return s + "s";                                        // default
}
```

241 train / 36 holdout (stratified so every suffix class is in train). Stem changes
use the new Mog `slice(start,end)` string method. Genuinely lexical plurals
(knife→knives, irregulars) are excluded as not suffix-predictable.

## Scope

Covers LinguaGenesis Stages 1–10 (grammar/morphology/semantic-type/logic — the
program-synthesizable layers). Stages 11–22 (pragmatics, social, relational) are
soft/social and not I/O-synthesizable. The remaining sentence-level ceiling
(wrong-stem lexical errors) needs lexical/positional features beyond token sets.
