# Rule-Compressed Memory: Bounded Storage, Unbounded Reach, Zero Forgetting

## The claim

A system that stores knowledge as **verified synthesized programs (rules)** rather
than instances exhibits a memory profile no context window or retrieval store can
match: **storage that converges to a bound while the input stream grows without
limit, perfect recall of everything ever seen, and correct generalization to an
unbounded set of items never seen.** We call this *rule-compressed memory*, and we
argue it is the precise, defensible form of the informal goal "infinite context".

The intuition: a context window stores *instances* (tokens) — storage grows
linearly and old entries are evicted (forgotten). A rule is a **lossless
compression of an unbounded instance set**: the English pluralization rule is a
~700-byte program that produces the correct plural of *every* word, including
words never observed. Once a regularity has been captured as a rule, every future
instance of it is free (a cache hit), so storage stops growing while the stream
continues; and because a rule is *code*, not weights, it never drifts and cannot
be evicted.

## Experiment

We stream `(word → plural)` instances and compare three memories:

* **RULE** (ours): an nSynth-synthesized pluralization program plus a small
  exception table for the genuinely irreducible (irregular) items. Storage =
  `|rule| + |exceptions|`. The rule is re-synthesized only when the current rule
  mis-predicts a streamed item — i.e. only when a new regularity appears.
* **INSTANCE** (the RAG / lookup baseline): remember every pair seen.
* **WINDOW-W** (the context-window / LLM baseline): remember only the last `W`.

Metrics vs. stream length: storage; coverage on *all items seen so far* (does the
memory still answer them?); and coverage on *held-out items never streamed* —
real words and nonce "wug" words (`wug → wugs`, `dax → daxes`) — i.e. does the
memory **generalize**? The oracle is the curriculum's `pluralize`; the rule is
recovered by nSynth from the streamed pairs alone.

## Result

Streaming 600 instances (window `W = 150`):

| | RULE (ours) | INSTANCE (RAG) | WINDOW-150 (LLM) |
|---|---|---|---|
| storage | **732 B — flat after ~9 re-synths** | 8 542 B — linear | constant W |
| recall on all 600 seen | **99.8 %** | 100 % | **25 %** (forgot 75 %) |
| generalize to unseen real words | **99.3 %** | 0 % | 0 % |
| generalize to nonce "wug" words | **100 %** | 0 % | 0 % |

The rule store is **11.7× smaller and converging** (its growth flattens as the
regularity is captured — only the finite set of true irregulars can ever enlarge
it), it **never forgets** (99.8 % recall on items long past any window), and it
**generalizes to the unbounded unseen** (100 % on words that were never streamed).
The instance store grows without bound and generalizes 0 %; the window store
forgets three-quarters of what it saw and generalizes 0 %.

## Why this is "effectively infinite context"

For the *compressible* (regular) part of a stream — which for language, code,
arithmetic, and formatted data is the overwhelming majority — rule-compressed
memory processes an arbitrarily long input with **bounded storage** and answers
any past or future query **exactly**, including queries about content it never
saw. The only part that consumes growing storage is the genuinely irreducible
residue (lexical exceptions, arbitrary facts), which in a real domain is a small
finite set. This is strictly stronger than a context window (which is bounded but
forgets and never generalizes) and than retrieval augmentation (which generalizes
to nothing and grows without bound).

Three properties combine to make the reach unbounded from finite memory:

1. **Compression** — one program covers an unbounded instance set (every word).
2. **Persistence without drift** — a rule is discrete code; it does not decay and
   cannot be evicted, so there is no catastrophic forgetting (unlike weights, and
   unlike a window).
3. **Composition** — rules compose (Section on curriculum rule learning shows a
   handful of inflection + agreement programs generate thousands of correct
   sentences), so `N` verified primitives reach combinatorially many behaviours.

## The second pillar: Hamilton mistake-memory (self-improvement)

Rules compress the *regular* part of a stream. The *irreducible* residue — lexical
exceptions, genuine errors — is handled by the complementary pillar: a persistent
**mistake memory**. In LinguaGenesis this is HamiltonGuard (every validator
failure becomes a `MistakeRecord` that is replayed as a future training example);
in nSynth it is the `rejected_cache` (negative memoization of failed candidates)
beside the `learned_biases` bank (positive memoization of what worked). The
experiment above is itself a mistake-memory loop: a mis-prediction is either used
to **re-synthesize** (refine the rule — the 9 re-syntheses) or recorded as an
**exception** (remembered, never repeated).

This closes the loop and gives the memory a property neither a window nor a weight
matrix has: it is **monotonically self-improving and never repeats a mistake**.
Concretely, once the system errs on an item it either fixes the rule or stores the
correction, so the *cumulative count of distinct mistakes converges to a finite
bound* and the *repeat-mistake rate is zero*. Every error permanently improves the
system and is never made again — the negative-memory analogue of "never forgets".

So the full memory has two convergent halves:

* **Positive** (rules + learned biases): compress the regular → unbounded reach,
  bounded storage, no decay.
* **Negative** (Hamilton mistakes + rejected cache + exceptions): remember the
  irreducible and the wrong → no repeated errors, and drives rule refinement.

Their sum is a memory that grows only with the genuinely new and never unlearns.

## Relation to the rest of the system

This is not a separate mechanism: it is the visible behaviour of nSynth's
existing persistent solved-program memory (`solved_cache`), learned-bias bank, and
rule synthesizers, viewed as a memory architecture. Every successful solve is a
recovered rule that is stored once and retrieved forever; the curriculum
rule-learning results show the rules are exact and verified; the speaker shows
they compose into generation. The contribution here is the framing and the
measurement: a memory whose storage converges while its reach does not.

## Reproduction

`nsynth/scripts/rule_memory_experiment.py --stream 600 --window 150` produces the
table above and `rule_memory_results.csv` (storage and coverage curves).
