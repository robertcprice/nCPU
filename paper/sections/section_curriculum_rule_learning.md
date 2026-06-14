# Learning a Language Curriculum's Rules as Verified Programs

## Thesis

Large language models learn grammar *statistically* — implicitly, from scraped
text, with no explicit, inspectable, or guaranteed-correct rule. We demonstrate
the inverse: a **program synthesizer** (nSynth) that learns the rules of an
English curriculum *explicitly*, recovering each as a **small, verified,
executable program**, using the curriculum's own validators as the oracle.

The curriculum is LinguaGenesis, a rules-first English learning system whose 22
stages are each defined by generators (which emit labeled examples) and
validators (which decide grammaticality). nSynth never sees a rule; it sees only
labeled examples and must re-derive the rule as a Mog program that the validator —
and a held-out test set — confirms. Where a large model would *approximate*
subject–verb agreement, nSynth *recovers it exactly* and the result *runs*.

## Architecture: perception → synthesis → verification

Every task follows one shape, identical from morphology to reasoning:

```
curriculum rule (oracle)
   │  generates labeled examples
   ▼
perception layer  ──►  structured encoding  ──►  nSynth search  ──►  verified Mog program
(morpheme tokenizer /                                 │                     ▲
 argument parser)                                     └── validator + holdout ┘
```

The **perception layer** turns surface text into the structure a rule operates
over — the morpheme tokenizer exposes inflectional suffixes as explicit symbol
tokens (`<+es>`, `<+ing>`, …); a logic parser exposes an argument's abstract form
(which proposition is asserted/concluded, negated or not). nSynth then learns the
**rule over that structure** and verifies it by executing the synthesized program
and comparing outputs. This is the same division of labor a person uses: perceive
the morphemes, then apply the rule.

## A ladder of rule classes

Each English rule needs a particular *shape* of program; nSynth grew a small
family of exact, verifying "teachers", each a strictly larger hypothesis class:

| Teacher | Hypothesis class | Recovered rule |
|---|---|---|
| `search_suffix_class` | disjunction of `ends_with` | English `-es` plural/3sg (`ch/sh/s/x/z/o`) |
| `morph_transduce` | suffix **decision list with exceptions** | full plural incl. `y→ies` stem change + vowel exceptions |
| `search_array_member_class` | disjunction of membership | "carries a valid 3sg suffix token" |
| `search_array_conjunction` | one conjunction (AND/NOT) | gerund aux: "`is` AND `<+ing>`" |
| `search_array_dnf` | **DNF** (disjunction of conjunctions) | modus ponens/tollens validity; past tense w/ stem feature |

The progression is the point: morphology needs *conditioned rules with
exceptions* (the `y→ies`-unless-vowel structure that is how English actually
works); auxiliary agreement needs a *conjunction*; logical validity needs a *DNF*.
Each teacher emits a fully verified program (admissible features never fire on a
negative; the program generalizes by construction) and is returned directly
rather than distilled by gradient.

### Morphology — generative, with stem changes

From the curriculum's own verb/noun lexicon, `morph_transduce` recovers the
**complete English pluralization rule** as a string→string program, including the
`y→ies` stem change and the vowel+y exceptions, ordered by specificity so longer
exception rules override the general rule:

```
fn pluralize(s: string) -> string {
    if s.ends_with("ay") { return s + "s"; }                       // vowel+y exception
    if s.ends_with("ch") { return s + "es"; }                      // sibilant
    if s.ends_with("y")  { return s.slice(0, s.len - 1) + "ies"; } // stem change
    return s + "s";                                                // default
}
```

This is a general decision-list learner (greedy max-coverage pure rules, emitted
specific-first, then pruned), reusable for any conditioned-rule task.

### Reasoning — logical inference validity

The same machinery climbs to Stage 8 (Formal Logic). A parser extracts each
argument's abstract form into feature tokens (assert/conclude × proposition ×
negation); nSynth's DNF teacher learns **validity** — recovering that an argument
is valid iff it is modus ponens or modus tollens and invalid iff it affirms the
consequent or denies the antecedent — as a verified program:

```
valid iff (asserts antecedent ∧ ¬negated)   // modus ponens
        ∨ (¬asserts antecedent ∧ negated)    // modus tollens
```

A program synthesizer, learning a rule of *logic*, from labeled arguments, with
the curriculum's validator as oracle.

### Semantics — selectional restriction

The same machinery reaches Stage 7 (compositional semantics). "The teacher writes
the report" is well-formed; "The report writes the teacher" is not — the *agent*
of an action must be animate. With an animacy feature (the curriculum's own
agent/patient lexicon), nSynth learns the **selectional restriction** "valid iff
the subject is animate" as a verified program. Four linguistic levels —
morphology, syntax, reasoning, semantics — recovered by one mechanism.

## Deployment — recovered rules as constraints

Because each rule is an executable program, it deploys directly on held-out
inputs the synthesizer never saw (`scripts/rule_constraint_demo.py`): `pluralize`
*generates* correct forms (church→churches, lady→ladies, including the y→ies stem
change); the 3sg and inference rules *accept or reject* fresh sentences and
arguments. This is the program-synthesis analog of KVRM's verified-registry
guarantee, applied to language: a generator constrained to nSynth-recovered rules
cannot produce a form the rule forbids.

## The honest ceiling — and how features close it

Not everything is a function of the token *set*. Two Stage-3 forms — "asked"
(correct) and "copyed" (wrong) — carry the same `<+ed>` token; they differ only
because *copy is a consonant+y stem* and takes `-ied`. No boolean function of the
token multiset can separate them: this is genuinely lexical/orthographic, and
nSynth correctly *declines* rather than overfitting.

The resolution is a richer **perception layer**, not a bigger search. Exposing one
orthographic feature — "the stem ends in a consonant + y" — lets the DNF teacher
learn the *general* past-tense rule, which then generalizes to unseen verbs:

```
valid past iff (<+ied> ∧ y-stem) ∨ (<+ed> ∧ ¬y-stem)
```

This locates the boundary precisely: the rule layer is exact and small; the
remaining difficulty is *what structure perception makes visible*.

## Why it matters

1. **Verified, not approximate.** Each rule is checked on held-out examples and is
   correct by construction (the disjunctive/decision-list learners never admit a
   feature that fires on a negative). This is the program-synthesis analog of
   KVRM's verified-registry guarantee: a language model emitting only
   nSynth-recovered rules cannot make an agreement error the rule forbids.
2. **Inspectable and runnable.** The recovered grammar is source code a human can
   read and a machine can execute — the opposite of weights.
3. **One mechanism, three levels.** Morphology, syntax, and reasoning are all
   "perceive structure, synthesize the rule, verify against the oracle." The
   ladder of teachers is a ladder of rule *shapes* (disjunction → decision list →
   conjunction → DNF), not a pile of special cases.
4. **A clean research boundary.** What a feature-rule learner cannot do is exactly
   what is lexical/positional, and that boundary is closed by enriching
   perception — a concrete, testable research program, not a mystery.

## Reproduction

`nsynth/scripts/linguagenesis_bridge.py` drives the loop (all words and labels
from the curriculum; nothing hand-authored here):

```bash
for t in verb_3sg_es pluralize_gen sentence_3sg sentence_gerund sentence_past formal_logic; do
  python3 scripts/linguagenesis_bridge.py --task $t | ./target/release/mog_synth --problem-json -
done
```

Regression tests pin each teacher in `nsynth/src/solver/tests.rs` and
`nsynth/src/morph_transduce.rs`.
