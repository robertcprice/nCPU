# A Program-Native Coding Model (bottom-up, not bolt-on)

Design document for ROADMAP Rung 9. Status: design, with Phase A
immediately buildable from existing components.

## The inversion

Every current code LLM — and NPCoT-as-addition — keeps token prediction
as the substrate and treats programs as text to be emitted. The bottom-up
architecture inverts this:

> **Reasoning happens in program space. Natural language and source code
> are input/output renderings, not the medium of thought.**

The model's core loop is: encode the task → propose a program in a typed
internal IR → execute it (differentiably during training, exactly at
inference) → check against evidence → emit. The emitted code is a
*deterministic transpile* of the internal program — `mog_transpile`
already renders one IR to Python/Rust/TypeScript with zero learned
parameters. Nothing in the output path can hallucinate, because the
output path has no generative freedom.

## Architecture

```
 NL prompt ──► [Text Encoder]──┐
                               ▼
 I/O examples ─► [Example     [Task Latent z]
                  Encoder] ──► (shared space) ──► [Program Prior Net] ──► soft program p̃
                               ▲                        │
              [Library Attention] ◄─── verified         ▼
              (external memory of      skills    [Differentiable Executor]
               discrete programs,                 run p̃ on examples
               cosine over signatures)                  │
                                                        ▼
                                            execution-match loss / verify
                                                        │
                                        discretize (STE/Gumbel) ──► exact verify
                                                        │
                                         [Transpiler — deterministic, 0 params]
                                                        ▼
                                              Python / Rust / TS source
```

Learned components (the entire parameter budget):
1. **Example encoder** — I/O pairs → task latent. Already exists in
   embryo: `mog_synth` meta-learner v5 (TransformerEncoder mapping I/O →
   UniversalProgramDescription, 23/24 on its bench).
2. **Text encoder** — NL → the *same* latent space. Small (100M-class);
   trained on a synthetic NL↔program corpus (below).
3. **Program prior net** — latent → distribution over the typed program
   space (ops, structure, constants). The Gumbel-softmax machinery exists
   in `ncpu/differentiable/` (opcode/register distributions, temperature
   annealing); the gradient solvers in nsynth are the non-amortized
   version of this network.
4. **Confidence gate** — execution-variance → answer/refuse. Exists
   (confidence-aware gating, Section 11.10 of the paper).

Non-learned components (free):
- Executor semantics (the canonical NPCoT executor — v1/v2/v3 — plus the
  Mog register machine), exact at inference, soft during training.
- Transpiler (mog_transpile).
- Library + the three memory banks (solved / biases / rejected) — the
  bank of learned restart biases is literally an amortized proposal
  distribution being collected already.

## Why this can be tiny

A 7B code LLM spends most of its capacity memorizing surface code and
human idiom. Here, surface form costs zero (transpiler), execution costs
zero (interpreter), and correctness is enforced by verification rather
than learned. The model only learns the *mapping from task description to
program-space region* — a far smaller function. Plausible target: a
sub-1B model that, on the program space it covers, beats any LLM at any
size — because its answers are verified, and refuses outside it.

## Training: self-generated, perfect ground truth

The data engine requires no human labels and no scraped code:
1. Sample random programs from the DSL (the `random_bias_init` emergent
   generator already does this for the universal-array space).
2. Execute on random inputs → perfect I/O pairs (this is the repo's
   "perfect ground truth advantage").
3. Render NL descriptions: template-based first ("a function that returns
   the sum of squares of the inputs"), LLM-paraphrased for diversity.
   The teacher LLM contributes *linguistic variety only* — semantics come
   from execution, so this stays bottom-up: no LLM weights, logits, or
   architecture enter the model.
4. Train: encoder(s) → program prior → soft-execute → execution-match
   loss + program-reconstruction loss; discretize with straight-through;
   hard-verify; failures feed the negative bank.
5. Compounding: every real-world solve (autoresearch, MCP server,
   registry) adds (NL, examples, program) triples back into training.

## Phases

- **Phase A (buildable now):** scale the meta-learner into the Program
  Prior Net over the v1/v2/v3 + universal-array spaces. Use as tier-0
  proposer in nsynth: predict program region, verify, fall back to search.
  Metric: % solved with zero search; mean search-steps saved on the
  105-problem bench. Components: meta-learner v5 + synthetic generator.
- **Phase B:** text encoder into the same latent space (synthetic NL
  corpus). Result: NL → verified program with no LLM in the loop. The MCP
  server swaps its extraction step for this encoder.
- **Phase C:** end-to-end differentiable stack with library attention as
  external memory; joint training; confidence gate. This is the native
  model. Evaluate against the matched-size LLM baseline on
  DSL-expressible tasks.
- **Phase D:** grow the DSL (typed values, richer data structures,
  bounded recursion) with the format-versioning discipline; curriculum
  over program size; honest published scoping of what fraction of e.g.
  HumanEval is DSL-expressible at each stage.

## Risks, stated plainly

- **The DSL ceiling becomes the model ceiling.** No internal LLM fallback
  — by design. Mitigation: the external cascade remains for everything
  outside the space; the model's job is to own its space perfectly.
- **NL understanding without web-scale pretraining is the hard part.**
  Synthetic-paraphrase corpora may not cover real user phrasing.
  Mitigation: the MCP/agent deployments collect real (prompt, examples,
  program) triples continuously — the data engine targets exactly the
  distribution that matters.
- **Amortization may not generalize** beyond training program sizes.
  Mitigation: the prior proposes, search + verification dispose — worst
  case degrades to today's (already 100%-coverage) search, never below.

## JEPA integration — imagination before execution

The repo's JEPA machinery (ncpu/jepa_neural_cpu, ncpu/world_model,
kernels/rust_metal/src/jepa) predicts machine-state transitions in latent
space. In the native model it becomes the **executor's fast approximate
twin**:

- **Latent speculation:** before running a candidate program exactly, a
  JEPA predictor estimates its execution outcome embedding from (task
  latent, program embedding). Candidates whose predicted outcome is far
  from the target are pruned without execution — thousands of candidates
  scored for the cost of one forward pass. The exact executor only runs
  finalists; verification stays exact, so soundness is untouched (JEPA
  prunes, never approves).
- **Stateful traces (format v3):** JEPA's native job is next-state
  prediction — exactly the (state, input) → state' shape of v3 skills.
  Train the predictor on traces of soft-program execution; prediction
  error doubles as an anomaly detector flagging library entries whose
  behavior drifts from their fingerprint.
- **Perfect ground truth:** JEPA training data is free and unlimited —
  the exact executor generates as many (state, program, next-state)
  triples as wanted. This is the paper's bottom-up JEPA thesis applied to
  the synthesis stack.

## Recursive (tree-structured) networks — programs are trees

Programs are ASTs; recursive networks are their natural geometry, and the
Pong build surfaced exactly the phenomenon they model:

- **Recursive program decoder:** instead of emitting a flat op-vector,
  the Program Prior Net grows the AST recursively — each node's
  distribution conditioned on its parent's embedding. Nested structure
  (if-inside-while, composed calls) gets first-class treatment instead of
  fixed slots; this is the principled path past the hardcoded
  `N_ARR_BODY = 4` body-slot architecture.
- **Compositional library embeddings:** a recursive encoder makes a
  composed skill's embedding a *function of its parts* —
  embed(gte) = g(embed(hit_top), embed(sub2)) for the Pong composition
  gte(a,b) = hit_top(sub2(b,a)). Retrieval then generalizes
  compositionally: a task near "comparison" can surface a composition of
  cached pieces that was never explicitly taught. The 8 composed Pong
  rules are the hand-built existence proof; the recursive encoder
  automates discovering such reuse.
- **Bounded recursion in the DSL (Phase D)** pairs with this decoder:
  self-similar programs handled by a self-similar network.

## Relation to the rest of nCPU

This is where the two halves of the project meet: the neural computer
supplies differentiable execution substrates (neural ALU, soft truth
tables, JEPA dynamics for latent speculation); the synthesis stack
supplies the program space, verification discipline, and memory banks.
The native model is both pillars in one artifact.
