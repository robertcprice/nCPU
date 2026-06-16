//! A stateful "mind": an Engine plus a Discourse it reads into and answers from.
//! This is the top-level understanding handle — read sentences to build a world
//! model, then ask questions answered from what was read. Exposed to C in
//! [`crate::ffi`] as `ncpu_mind_new` / `ncpu_read` / `ncpu_ask`.

use crate::comprehension::Engine;
use crate::understanding::discourse::Discourse;
use crate::understanding::inference::{polarity_flip, prove, render_proof, Proof};
use crate::understanding::meaning::Meaning;
use crate::understanding::{qa, semantics};

/// The three-valued VERDICT a yes/no answer carries, abstracted away from its
/// surface phrasing so two answers can be compared for "did the mind change?".
/// `Yes`/`No` are the determined truth verdicts; `Idk` the open-world
/// "I don't know."; `Other` anything else (a wh-filler, a count, a cause, a
/// degree phrase) that carries no propositional verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Yes,
    No,
    Idk,
    Other,
}

/// Classify an answer STRING into its [`Verdict`]. The QA layer's yes/no answers
/// begin with "Yes," / "No,"; the open-world answer is exactly "I don't know.";
/// everything else is `Other`. Used to decide whether a counterfactual GENUINELY
/// changed the mind (Yes↔No↔Idk is a change; same verdict is not).
fn verdict_of(answer: &str) -> Verdict {
    let a = answer.trim().to_lowercase();
    if a.starts_with("yes") {
        Verdict::Yes
    } else if a.starts_with("no") {
        Verdict::No
    } else if a.contains("don't know") {
        Verdict::Idk
    } else {
        Verdict::Other
    }
}

pub struct Mind {
    engine: Engine,
    discourse: Discourse,
}

impl Default for Mind {
    fn default() -> Self {
        Self::new()
    }
}

impl Mind {
    /// Build a mind: synthesizes the lexicon/rules once (slow), then ready to read.
    pub fn new() -> Self {
        Mind { engine: Engine::new(), discourse: Discourse::new() }
    }

    /// Read a sentence into the world model — resolves coreference against what
    /// was read before, asserts the resulting facts, and returns the resolved
    /// Meaning.
    pub fn read(&mut self, sentence: &str) -> Meaning {
        self.discourse.read(&self.engine, sentence)
    }

    /// Answer a question from what has been read (truth from the world model,
    /// closed under sound inference).
    pub fn ask(&self, question: &str) -> String {
        qa::answer(&self.engine, &self.discourse, question)
    }

    /// Answer a question AND show the reasoning behind it — metacognition over
    /// what was read. Parses the question through the SAME path as [`ask`]
    /// (`semantics::understand`, so pronouns / relative clauses resolve
    /// identically), then routes the parsed meaning through
    /// [`qa::answer_explained`], which returns the same string `ask` would PLUS a
    /// sound [`Proof`](crate::understanding::inference::Proof) when the verdict
    /// rests on a derivation.
    ///
    /// When a proof is present, the answer (`"Yes, ..."` / `"No, ..."`) is
    /// followed by the rendered "because" chain — each derivation step named and
    /// its premises bottoming out in `you told me ...` leaves, so a transitively
    /// entailed answer surfaces the intermediate facts it routed through. When no
    /// proof backs the verdict (a world-owned/opaque truth, a wh-filler, a count,
    /// a cause, or the honest `"I don't know."` of an unprovable query), the plain
    /// answer is returned with NO fabricated justification.
    pub fn why(&mut self, question: &str) -> String {
        // Parse exactly as `ask` does, then explain the parsed meaning.
        let parsed = semantics::understand(&self.engine, question);
        let (answer, proof) = qa::answer_explained(&self.engine, &self.discourse, &parsed);
        match proof {
            Some(proof) => format!("{answer} {}.", render_proof(&proof, &self.engine)),
            None => answer,
        }
    }

    /// The logical form of a sentence, without asserting it.
    pub fn understand(&self, sentence: &str) -> Meaning {
        semantics::understand(&self.engine, sentence)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The contradictions the world model has detected so far — metacognition
    /// over consistency. As [`read`](Self::read) asserts each resolved meaning,
    /// the world records any clash with what it already holds (e.g. a fact and
    /// its negation, or incompatible comparison edges). This surfaces that
    /// running ledger without re-deriving it, delegating straight to the
    /// discourse's accumulated [`World`](crate::understanding::world_model::World).
    /// An empty slice means everything read so far is mutually consistent.
    pub fn contradictions(&self) -> &[crate::understanding::world_model::Contradiction] {
        self.discourse.world.contradictions()
    }

    /// Replay the fixed regression corpus against this mind's engine and return
    /// the resulting [`GateReport`](crate::self_improve::gate::GateReport) —
    /// metacognition over the mind's own competence. This is the self-check the
    /// self-improvement substrate uses as its guard: every golden case (setup
    /// sentences, a question, an expected answer) is replayed through a fresh
    /// discourse and a battery of soundness invariants is probed, all against
    /// THIS mind's engine. The report's `ok()` is true only when every
    /// behavioral case passed and every soundness invariant held. Purely a
    /// consumer of the engine — it never mutates it — so a mind can audit itself
    /// at any time. Delegates to
    /// [`regression_gate`](crate::self_improve::gate::regression_gate).
    pub fn self_check(&self) -> crate::self_improve::gate::GateReport {
        crate::self_improve::gate::regression_gate(&self.engine)
    }

    /// A tiny GAP DETECTOR: does the mind currently have a lexicon entry for
    /// `word`? True iff the engine's `noun_animacy` lookup classifies it as some
    /// noun (animate or inanimate), i.e. `noun_class(word) > 0`. A word the mind
    /// has never learned about classifies as `0` and reads here as "unknown" —
    /// the cheap signal the self-improvement loop uses to notice a missing
    /// component before trying to close it. Purely a read accessor over the
    /// engine; it never mutates anything.
    pub fn knows_word(&self, word: &str) -> bool {
        self.engine.noun_class(word) > 0
    }

    /// The self-improvement LOOP, wired onto the mind: try to close the gap
    /// described by `req`, and on a *clean* acceptance REPLACE this mind's engine
    /// with the freshly grafted candidate.
    ///
    /// This delegates the whole synthesize → gate → journal decision to
    /// [`self_extend`](crate::self_improve::extend::self_extend), passing it the
    /// mind's CURRENT engine. The candidate is accepted only if it passes the
    /// mind's own regression gate — every golden behavioral case still passes and
    /// the world model stays sound (the substrate's monotone-growth guarantee).
    /// On acceptance we swap in the returned candidate engine so the new component
    /// is live for every subsequent query; on rejection (synthesis failed, or the
    /// gate went red) `self.engine` is left exactly as it was. Every attempt —
    /// accepted or rejected — is journaled by the substrate, so the mind's
    /// self-modification history stays auditable.
    pub fn self_improve(
        &mut self,
        req: crate::self_improve::extend::LearnRequest,
    ) -> crate::self_improve::extend::LearnReport {
        let (candidate, report) = crate::self_improve::extend::self_extend(&self.engine, &req);
        // Adopt the new engine ONLY on a clean acceptance. `self_extend` returns
        // `Some(engine)` exactly when the candidate synthesized AND passed the
        // gate, so guarding on the candidate is equivalent to guarding on
        // `report.accepted` — but taking the engine the substrate vetted keeps the
        // accept decision in one place.
        if let Some(new_engine) = candidate {
            self.engine = new_engine;
        }
        report
    }

    // ===================================================================
    // Reflection + deeper-reasoning scaffold (Tracks B & E).
    //
    // The methods below are STUBS — their real logic lands in the next
    // phase. Each is documented with the metacognitive behavior it will
    // implement so the API surface is stable for callers now. They are
    // purely additive and do not touch any existing code path.
    // ===================================================================

    /// Track B (introspection): explain HOW the mind knows what it knows about a
    /// `topic` — surface the synthesized program + teacher provenance behind the
    /// component the topic refers to. "Show me your code."
    ///
    /// The `topic` keyword is mapped to one synthesized component:
    ///   * "verb" / "inflect" / "3sg"  -> the third-person-singular inflector
    ///     (`verb_3sg`, with `regular_3sg` as its recovered core rule)
    ///   * "noun" / "animacy"          -> `noun_animacy` (the animacy lexicon)
    ///   * "agreement"                 -> `valid_agreement` (subject-verb number)
    ///   * "past"                      -> the past-tense inflector
    ///     (`verb_past`, with `regular_past` as its recovered core rule)
    ///
    /// The returned string NAMES the component, names the TEACHER that recovered
    /// it (via [`Engine::method_for`](crate::comprehension::Engine::method_for)),
    /// and quotes the ACTUAL Mog source for the relevant function, sliced verbatim
    /// from [`Engine::program`](crate::comprehension::Engine::program). This is the
    /// system explaining its OWN learned algorithm — the real synthesized code.
    ///
    /// HONESTY: an unmapped topic returns "I don't have a learned program for
    /// <topic>." — never a fabricated explanation.
    pub fn explain_self(&self, topic: &str) -> String {
        // Map the free-text topic to a synthesized component. Each entry pairs the
        // PROVENANCE component (whose teacher we report, recorded in
        // `Engine::methods`) with the Mog `fn` name whose source we slice out of
        // `Engine::program`. For inflection we report the recovered RULE component
        // (regular_3sg / regular_past — those have teachers) while quoting the
        // runnable wrapper `fn verb_3sg` / `fn verb_past` that composes the rule
        // with the irregular lexicon.
        let t = topic.to_lowercase();
        let mapped: Option<(&str, &str)> =
            if t.contains("3sg") || t.contains("inflect") || t.contains("verb") {
                // 3sg inflection. Provenance: regular_3sg teacher; source: verb_3sg fn.
                Some(("regular_3sg", "verb_3sg"))
            } else if t.contains("past") {
                Some(("regular_past", "verb_past"))
            } else if t.contains("animacy") || t.contains("noun") {
                Some(("noun_animacy", "noun_animacy"))
            } else if t.contains("agreement") {
                Some(("valid_agreement", "valid_agreement"))
            } else {
                None
            };

        let Some((component, fn_name)) = mapped else {
            return format!("I don't have a learned program for {topic}.");
        };

        let teacher = self.engine.method_for(component).unwrap_or("(unknown teacher)");

        // Slice the ACTUAL Mog source for `fn_name` out of the composed program.
        // For the inflection wrappers we ALSO prepend the recovered rule the
        // wrapper calls, so "show me your code" surfaces the learned algorithm
        // itself, not merely the dispatch shim.
        let program = self.engine.program();
        let mut source = slice_fn_source(program, fn_name).unwrap_or_default();
        if fn_name != component {
            if let Some(rule_src) = slice_fn_source(program, component) {
                source = format!("{rule_src}\n\n{source}");
            }
        }
        let source =
            if source.is_empty() { "(source unavailable)".to_string() } else { source };

        format!(
            "I learned the component `{component}` (the {fn_name} program). \
             It was recovered by the teacher: {teacher}. Here is the actual Mog \
             source I synthesized:\n\n{source}"
        )
    }

    /// Track B (introspection): enumerate everything the mind currently knows
    /// about a given `entity` — every asserted fact MENTIONING the entity, every
    /// sound consequence of those facts that bears on it, and the sound taxonomy
    /// closure (e.g. "teacher is a person/agent"), each realized to English and
    /// deduplicated. Drawn from the discourse's accumulated
    /// [`World`](crate::understanding::world_model::World) read accessors
    /// ([`facts`](crate::understanding::world_model::World::facts) +
    /// [`closure`](crate::understanding::world_model::World::closure)) plus the
    /// sound [`consequences`](crate::understanding::inference::consequences) of each
    /// relevant fact.
    ///
    /// Returns an EMPTY vec for an unknown entity (one neither named in any fact
    /// nor a known entity in the world). Every returned claim is realized with the
    /// SAME surface realizer answers use, so the phrasing matches `ask`/`why`.
    pub fn what_do_you_know(&self, entity: &str) -> Vec<String> {
        let entity = entity.trim().to_lowercase();
        if entity.is_empty() {
            return Vec::new();
        }
        let world = &self.discourse.world;

        // Unknown entity: never named in a fact AND not a known world entity.
        let known_entity =
            world.entities().iter().any(|e| e.eq_ignore_ascii_case(&entity));
        let in_a_fact = world.facts().iter().any(|ev| event_mentions(ev, &entity));
        if !known_entity && !in_a_fact {
            return Vec::new();
        }

        let mut out: Vec<String> = Vec::new();
        let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut push = |claim: String| {
            let key = claim.to_lowercase();
            if !key.is_empty() && seen.insert(key) {
                out.push(claim);
            }
        };

        // 1) Every asserted EVENT fact mentioning the entity, plus the sound
        //    consequences of that fact that ALSO mention the entity (so we surface
        //    derived knowledge like "the teacher writes [something]" without
        //    drifting onto unrelated entities).
        for ev in world.facts() {
            if !event_mentions(ev, &entity) {
                continue;
            }
            let fact = Meaning::Event(ev.clone());
            push(qa::realize(&self.engine, &fact, None));
            for cons in crate::understanding::inference::consequences(&fact) {
                if meaning_mentions(&cons, &entity) {
                    push(qa::realize(&self.engine, &cons, None));
                }
            }
        }

        // 2) The sound taxonomy closure bearing on the entity (e.g. the derived
        //    "teacher is a person" / "teacher is an agent"). `closure` returns
        //    derived IsA Meanings keyed on the entity head.
        for m in world.closure() {
            if meaning_mentions(&m, &entity) {
                push(qa::realize(&self.engine, &m, None));
            }
        }

        out
    }

    /// Track B (introspection): the relevant open-world UNKNOWNS bearing on a
    /// `question` — honest self-assessment of the boundary of what was read.
    ///
    /// We parse the `question` exactly as [`ask`](Self::ask) does (through
    /// `semantics::understand`, then discourse pronoun/relative-clause resolution
    /// so "it"/"they" and restricted definites resolve identically), then unwrap
    /// the interrogative to the propositions it actually queries. For each such
    /// atomic proposition whose truth the world currently leaves UNDETERMINED
    /// (`world.holds(..) == None` under the open-world assumption), we report
    /// "I don't yet know whether <X>." — `<X>` realized with the SAME surface
    /// realizer answers use, so the phrasing matches.
    ///
    /// SOUNDNESS / HONESTY: a proposition is listed ONLY when the world genuinely
    /// has no verdict on it (neither asserted, nor entailed, nor contradicted, nor
    /// derivable by the taxonomy/inference cascade [`qa::world_truth_traced`]
    /// consults). A determined proposition (the mind already knows it true or
    /// false) is never listed. With nothing undetermined we say so plainly. We
    /// never fabricate an unknown the question does not raise.
    pub fn gaps(&self, question: &str) -> String {
        let parsed = semantics::understand(&self.engine, question);
        // The atomic propositions this question depends on, with pronouns/relative
        // clauses already resolved against the discourse (so the world is queried
        // about the right entities).
        let atoms = self.query_atoms(&parsed);

        // Keep only the genuinely UNDETERMINED ones (open-world `None`). Each is
        // realized with QA's realizer so it reads like the rest of the system.
        let mut unknowns: Vec<String> = Vec::new();
        for atom in &atoms {
            if self.discourse.world.holds(atom).is_none() {
                let phrase = qa::realize(&self.engine, atom, /*force_negated=*/ Some(false));
                let line = format!("I don't yet know whether {phrase}.");
                if !unknowns.contains(&line) {
                    unknowns.push(line);
                }
            }
        }

        if unknowns.is_empty() {
            // Either the question raised no determinable atom (unparseable / a
            // non-propositional query) or every atom it raised is already
            // determined. Distinguish the two honestly.
            if atoms.is_empty() {
                "I don't have a relevant open question for that.".to_string()
            } else {
                "There is no open question there — I already know the relevant facts."
                    .to_string()
            }
        } else {
            unknowns.join(" ")
        }
    }

    /// Track E (counterfactual reasoning): name the NEW evidence that would flip
    /// the current verdict on a `question` — the fact(s) whose assertion would
    /// genuinely change the answer.
    ///
    /// We parse the `question` as [`ask`](Self::ask) does and obtain both the
    /// answer and its sound [`Proof`](crate::understanding::inference::Proof) via
    /// [`qa::answer_explained`]. Two cases:
    ///
    /// * **Determined (Yes/No) on a proof.** The verdict rests on the proof's
    ///   asserted LEAF facts. For each leaf we take its
    ///   [`polarity_flip`](crate::understanding::inference::polarity_flip) (the
    ///   sound contradictory of that premise) and report
    ///   "I would change my mind if you told me: <not LEAF>".
    /// * **Undetermined ("I don't know.").** The thing that would let the mind
    ///   decide is the queried proposition itself (would make it Yes) or its
    ///   negation (would make it No).
    ///
    /// SOUNDNESS / GENUINENESS (the load-bearing guarantee): every flip we name is
    /// VERIFIED to actually move the verdict. The world model is MONOTONE (asserting
    /// a fact's negation does not retract the fact, and inference still fires on the
    /// surviving positive), so a meaningful counterfactual must model the leaf being
    /// FALSE rather than merely also-asserted-false. We therefore re-derive the
    /// verdict against a controlled fact set: the discourse's current asserted facts
    /// with the leaf REMOVED and the flip ADDED, evaluated soundly through
    /// [`prove`](crate::understanding::inference::prove). A flip is reported ONLY
    /// when that counterfactual verdict differs from the current one
    /// (`Yes`↔`No`↔`Idk`). The real `self.discourse` is never touched — the
    /// counterfactual fact set is built and evaluated locally. A flip that does not
    /// change the verdict is silently dropped, so we never name a counterfactual
    /// that would not really change the mind.
    pub fn what_would_change_your_mind(&self, question: &str) -> String {
        let parsed = semantics::understand(&self.engine, question);
        let (answer, proof) = qa::answer_explained(&self.engine, &self.discourse, &parsed);
        let current = verdict_of(&answer);

        // The single goal proposition the verdict is about (a yes/no atom). Without
        // one we cannot re-derive a counterfactual verdict, so we bail honestly.
        let goal = match self.query_atoms(&parsed).into_iter().next() {
            Some(g) => g,
            None => return "Nothing you could tell me would change my answer.".to_string(),
        };

        // Candidate (removed-fact, flip) pairs to test, by case. `removed` is the
        // asserted fact whose FALSITY the flip models (so we drop it from the
        // counterfactual fact base before adding the flip); `None` removes nothing
        // (we are ADDING information to an undetermined question).
        let candidates: Vec<(Option<Meaning>, Meaning)> = match current {
            // A determined verdict backed by a proof: flip each asserted leaf, and
            // model that leaf being false by removing it.
            Verdict::Yes | Verdict::No => match &proof {
                Some(p) => proof_leaves(p)
                    .into_iter()
                    .filter_map(|leaf| polarity_flip(&leaf).map(|flip| (Some(leaf), flip)))
                    .collect(),
                // A determined verdict the world OWNS opaquely (no public proof):
                // the only thing that would change it is being told the opposite of
                // the queried proposition itself — model the goal being false.
                None => polarity_flip(&goal)
                    .map(|flip| vec![(Some(goal.clone()), flip)])
                    .unwrap_or_default(),
            },
            // Undetermined: being told the proposition (→ Yes) or its negation
            // (→ No) is what would let the mind decide. Nothing is removed.
            Verdict::Idk => {
                let mut v = vec![(None, goal.clone())];
                if let Some(flip) = polarity_flip(&goal) {
                    v.push((None, flip));
                }
                v
            }
            // A non-propositional answer (wh-filler / count / degree / cause): no
            // single propositional flip applies.
            Verdict::Other => Vec::new(),
        };

        // VERIFY each candidate against what the user can ACTUALLY do. The world is
        // MONOTONE — information can only be ADDED, never retracted — so a flip
        // genuinely "changes my mind" only if ADDING it alone (no removal) re-derives
        // a different verdict. For a proof-backed verdict, adding the negation of a
        // supporting leaf is a no-op: the original leaf still entails the conclusion,
        // so the verdict stands. The honest thing to report there is the DEPENDENCY —
        // the answer rests on that leaf and would change only if the leaf itself were
        // false, which (being monotone) the user cannot bring about by telling us
        // anything. We separate the two so we never claim a flip is actionable when
        // it is not (the soundness bug this guards against).
        // Faithful "actionable" test: does TELLING the mind `flip` — the ONLY real,
        // monotone operation a user has — move the answer to THIS question away from
        // `current`? We model it exactly as it would happen: clone the discourse,
        // assert the flip into the clone's world (so the world's most-recent-wins
        // event truth is honored, unlike a pure re-derivation), and re-ask. This is
        // what makes a directly-asserted fact's negation genuinely flip the verdict
        // while a proof-backed generalization's leaf-negation correctly does not.
        let telling_flips = |flip: &Meaning| -> bool {
            let mut clone = self.discourse.clone();
            clone.world.assert(flip);
            verdict_of(&qa::answer(&self.engine, &clone, question)) != current
        };

        let mut tellable: Vec<String> = Vec::new();
        let mut dependencies: Vec<String> = Vec::new();
        for (removed, flip) in &candidates {
            // Actionable: ADDING `flip` (alone) flips the real, re-asked verdict.
            if telling_flips(flip) {
                let phrase = qa::realize(&self.engine, flip, /*force_negated=*/ None);
                let line = format!("I would change my mind if you told me: {phrase}.");
                if !tellable.contains(&line) {
                    tellable.push(line);
                }
            } else if let Some(leaf) = removed {
                // Adding the flip is a no-op; the verdict instead RESTS on this
                // asserted leaf — modeling the leaf as false (remove it, add its
                // negation) DOES change the verdict, but you cannot un-tell me.
                if self.counterfactual_changes_verdict(&goal, current, Some(leaf), flip) {
                    let phrase = qa::realize(&self.engine, leaf, /*force_negated=*/ None);
                    let line = format!(
                        "My answer rests on what you told me ({phrase}); it would change \
                         only if that were not true — and you cannot un-tell me."
                    );
                    if !dependencies.contains(&line) {
                        dependencies.push(line);
                    }
                }
            }
        }

        let mut out = tellable;
        out.extend(dependencies);
        if out.is_empty() {
            "Nothing you could tell me would change my answer.".to_string()
        } else {
            out.join(" ")
        }
    }

    /// Track E (hypothetical reasoning): evaluate a `question` UNDER a supposed
    /// `assumption` without committing to it. "Suppose X; then would Y?"
    ///
    /// We take `&self` and CLONE the discourse internally, so the real world is
    /// never touched. The assumption is `read` into the CLONE — exactly as a real
    /// sentence would be, resolving coreference and asserting its resolved meaning
    /// into the clone's world. The question is then answered against the clone via
    /// [`qa::answer_explained`], which yields the same string [`ask`](Self::ask)
    /// would PLUS a sound [`Proof`](crate::understanding::inference::Proof) when
    /// the verdict rests on a derivation — so the rendered "because" chain shows
    /// the assumption (now an asserted leaf in the clone) was actually used.
    ///
    /// Returns `"If <assumption>, then <answer>[ because ...]."`. The real
    /// `self.discourse` is provably untouched: we never call a `&mut self` method,
    /// and the assumption only ever enters a local clone.
    pub fn suppose(&self, assumption: &str, question: &str) -> String {
        // Hypothetical world: a clone of the real discourse. Mutating it leaves
        // `self.discourse` (which we only ever borrow) completely unchanged.
        let mut hypothetical = self.discourse.clone();
        hypothetical.read(&self.engine, assumption);

        // Answer the question against the hypothetical world, preferring the
        // explained path so a proof surfaces the assumption when it is used.
        let parsed = semantics::understand(&self.engine, question);
        let (answer, proof) = qa::answer_explained(&self.engine, &hypothetical, &parsed);

        // Phrase the supposition cleanly: trim a trailing period from the
        // assumption so "If X., then ..." reads as "If X, then ...".
        let supposed = assumption.trim().trim_end_matches('.').trim();
        match proof {
            Some(proof) => format!(
                "If {supposed}, then {answer} {}.",
                render_proof(&proof, &self.engine)
            ),
            None => format!("If {supposed}, then {answer}"),
        }
    }

    /// Track E (counterfactual retraction): evaluate a `question` in a world where
    /// `retracted_fact` is supposed NOT to be so. "What if X had NOT been true?"
    ///
    /// We take `&self` and CLONE the discourse internally — the real world is
    /// never touched. Because the world model is monotone (assertions accumulate;
    /// there is no public retraction), we cannot truly DELETE the fact. Instead we
    /// model "suppose this were not so" SOUNDLY by asserting the fact's
    /// contradictory — its [`polarity_flip`](crate::understanding::inference::polarity_flip),
    /// the negation the original guarantees to be false — into the CLONE. The
    /// question is then answered against that counterfactual world and CONTRASTED
    /// with the actual answer (computed against the untouched real discourse).
    ///
    /// We report honestly whether the verdict CHANGED: when the counterfactual
    /// answer differs from the actual one, we say so ("... rather than ..."); when
    /// they coincide, we note the retraction does not change the answer. If the
    /// retracted fact does not parse into a meaning with a sound contradictory
    /// (e.g. a question or an un-flippable form), we say so rather than guessing.
    pub fn what_if_not(&self, retracted_fact: &str, question: &str) -> String {
        // The ACTUAL answer, computed against the REAL (borrowed, never mutated)
        // discourse — our baseline for the contrast.
        let actual = self.ask(question);

        // Parse the retracted fact and form its sound contradictory.
        let fact_meaning = semantics::understand(&self.engine, retracted_fact);
        let Some(negation) = crate::understanding::inference::polarity_flip(&fact_meaning) else {
            // No single-meaning contradictory: we cannot soundly suppose its
            // falsity, so we decline rather than fabricate a counterfactual.
            let supposed = retracted_fact.trim().trim_end_matches('.').trim();
            return format!(
                "I cannot suppose that \"{supposed}\" is not so. Actually, {}",
                lower_first(&actual)
            );
        };

        // Counterfactual world: a CLONE with the negation asserted. The real
        // `self.discourse` is untouched (we only assert into the local clone).
        let mut counterfactual = self.discourse.clone();
        counterfactual.world.assert(&negation);

        // The COUNTERFACTUAL answer, against the clone (`qa::answer` re-parses and
        // resolves the question internally, just as `ask` does for the baseline).
        let cf_answer = qa::answer(&self.engine, &counterfactual, question);

        let supposed = retracted_fact.trim().trim_end_matches('.').trim();
        if cf_answer == actual {
            // The retraction does not move the verdict — say so honestly.
            format!(
                "If {supposed} were not so, the answer would be the same: {}",
                lower_first(&cf_answer)
            )
        } else {
            // The verdict flips — contrast counterfactual against actual.
            format!(
                "If {supposed} were not so, then {} (rather than {}).",
                lower_first(trim_period(&cf_answer)),
                lower_first(trim_period(&actual))
            )
        }
    }

    /// Track E (abduction): the WHY behind an effect — the recorded cause, or the
    /// best available explanation, for the state of affairs a `question` asks about.
    ///
    /// We parse the `question` (as [`ask`](Self::ask) does) and isolate the queried
    /// EFFECT: a why-question parses to `Causal{cause: Unknown, effect}`, so we take
    /// its `effect`; a plain truth query ("does the street flood?") or a bare
    /// proposition is itself treated as the effect. We then explain it, in order of
    /// strength:
    ///
    /// 1. **Recorded cause.** If the discourse was told a causal link whose effect
    ///    matches ("the street floods because the rain falls"), we return that
    ///    cause via [`world.cause_of`](crate::understanding::world_model::World::cause_of)
    ///    — but only when the effect is actually ATTESTED in the world (we never
    ///    explain an effect the world has no record of).
    /// 2. **Entailing fact (best explanation).** Otherwise we look for an asserted
    ///    fact that SOUNDLY ENTAILS the effect via the bounded
    ///    [`prove`](crate::understanding::inference::prove): a fact `F` explains
    ///    the effect `E` when `prove([F], E)` succeeds AND `F` is not `E` itself
    ///    (a fact does not explain itself). The simplest such `F` is returned with
    ///    its derivation rendered by [`render_proof`].
    /// 3. **Honest ignorance.** With neither a recorded cause nor an entailing
    ///    fact, we answer "I don't know why." — never an invented cause.
    pub fn explain_cause(&self, question: &str) -> String {
        let parsed = semantics::understand(&self.engine, question);

        // Isolate the queried effect proposition, with its surface terms resolved
        // against the discourse so the world is consulted about the right entities.
        let effect: Meaning = match &parsed {
            // A why-question: the effect is the Causal's effect clause.
            Meaning::Causal { effect, .. } => self.resolve_meaning_terms(effect),
            // A yes/no truth query "why"-flavored as "does <effect>?": unwrap it.
            Meaning::YesNoQuestion(inner) => self.resolve_meaning_terms(inner),
            // A bare proposition queried directly is itself the effect.
            other => self.resolve_meaning_terms(other),
        };

        // (1) RECORDED cause — only if the effect is actually attested.
        if self.discourse.world.holds(&effect) == Some(true) {
            if let Some(cause) = self.discourse.world.cause_of(&effect) {
                let phrase = qa::realize(&self.engine, &cause, /*force_negated=*/ None);
                return capitalize_first(&format!("because {phrase}."));
            }
        }

        // (2) BEST EXPLANATION: an asserted fact that soundly ENTAILS the effect.
        // Scan facts in assertion order (earliest first = simplest available) and
        // keep the first that genuinely entails the effect and is not the effect
        // itself. `prove([fact], effect)` is the soundness check.
        for fact in self.discourse.world.facts() {
            let premise = Meaning::Event(fact.clone());
            if premise == effect {
                continue; // a fact does not explain itself
            }
            if let Some(p) = prove(std::slice::from_ref(&premise), &effect) {
                // Only a genuine DERIVATION (an inference step, not a bare
                // asserted leaf) counts as an explanation: if the proof is a leaf
                // the "effect" was just directly asserted, which explains nothing.
                if !p.premises.is_empty() {
                    let phrase = qa::realize(&self.engine, &premise, /*force_negated=*/ None);
                    return capitalize_first(&format!(
                        "because {phrase} — {}.",
                        render_proof(&p, &self.engine)
                    ));
                }
            }
        }

        // (3) Honest ignorance — never a fabricated cause.
        "I don't know why.".to_string()
    }

    // ===================================================================
    // Private reasoning helpers for the metacognition methods above.
    // ===================================================================

    /// The atomic propositions a query depends on, with the question's interrogative
    /// wrapper unwrapped to the proposition(s) it asks about. Pronoun terms are
    /// resolved against the discourse. Only propositions the world can have a
    /// verdict on are returned (no `WhQuestion`/`CountQuestion`/`DegreeQuestion`
    /// fillers, which are not yes/no atoms).
    fn query_atoms(&self, parsed: &Meaning) -> Vec<Meaning> {
        // Resolve terms the same way QA does for a question, by reading the
        // proposition into a clone is unnecessary — `world.holds` resolves
        // restricted terms itself, and pronouns are resolved here per-term.
        let mut atoms: Vec<Meaning> = Vec::new();
        self.collect_atoms(parsed, &mut atoms);
        atoms
    }

    /// Recursively collect the yes/no-evaluable atoms of a query meaning.
    fn collect_atoms(&self, m: &Meaning, out: &mut Vec<Meaning>) {
        match m {
            Meaning::YesNoQuestion(inner) => self.collect_atoms(inner, out),
            // A disjunction's gap is the gap of each undetermined disjunct.
            Meaning::Or(disjuncts) => {
                for d in disjuncts {
                    self.collect_atoms(d, out);
                }
            }
            // Outer negation: the relevant atom is the inner proposition (its
            // truth determines the negation's truth).
            Meaning::Not(inner) => self.collect_atoms(inner, out),
            // Propositional leaves the world can evaluate: take them as-is, with
            // any pronoun terms resolved against the discourse.
            Meaning::Event(_)
            | Meaning::IsA { .. }
            | Meaning::HasProperty { .. }
            | Meaning::Comparison { .. }
            | Meaning::Quantified { .. }
            | Meaning::Cardinal { .. }
            | Meaning::Attitude { .. }
            | Meaning::Modal { .. }
            | Meaning::Temporal { .. }
            | Meaning::Causal { .. } => {
                let resolved = self.resolve_meaning_terms(m);
                if !out.contains(&resolved) {
                    out.push(resolved);
                }
            }
            // Non-propositional or unparseable: no yes/no atom to surface.
            Meaning::WhQuestion { .. }
            | Meaning::CountQuestion { .. }
            | Meaning::DegreeQuestion { .. }
            | Meaning::Unknown(_) => {}
        }
    }

    /// Resolve the surface TERMS of a propositional meaning against the discourse
    /// (pronouns → antecedents), leaving structure intact. Mirrors how QA resolves
    /// a question's terms before querying the world.
    fn resolve_meaning_terms(&self, m: &Meaning) -> Meaning {
        let d = &self.discourse;
        match m {
            Meaning::Event(ev) => {
                let mut e = ev.clone();
                e.agent = e.agent.map(|t| d.resolve(&t));
                e.patient = e.patient.map(|t| d.resolve(&t));
                e.recipient = e.recipient.map(|t| d.resolve(&t));
                Meaning::Event(e)
            }
            Meaning::IsA { subject, category, negated } => Meaning::IsA {
                subject: d.resolve(subject),
                category: category.clone(),
                negated: *negated,
            },
            Meaning::HasProperty { subject, property, negated } => Meaning::HasProperty {
                subject: d.resolve(subject),
                property: property.clone(),
                negated: *negated,
            },
            Meaning::Comparison { subject, scale, more, than, negated } => Meaning::Comparison {
                subject: d.resolve(subject),
                scale: scale.clone(),
                more: *more,
                than: d.resolve(than),
                negated: *negated,
            },
            // Other propositional shapes carry bound/embedded terms the world
            // resolves itself; pass through unchanged.
            other => other.clone(),
        }
    }

    /// The discourse's current asserted facts as `Meaning`s — the form
    /// [`prove`](crate::understanding::inference::prove) consumes. Event facts plus
    /// the positive comparison edges, exactly the fact base QA's inference cascade
    /// reasons over.
    fn fact_base(&self) -> Vec<Meaning> {
        let mut facts: Vec<Meaning> = self
            .discourse
            .world
            .facts()
            .iter()
            .map(|f| Meaning::Event(f.clone()))
            .collect();
        facts.extend(self.discourse.world.comparison_facts());
        facts
    }

    /// Re-derive the sound verdict of `goal` against a fact-meaning set, the same
    /// way QA's inference fallback does: `Yes` if some fact entails the goal, `No`
    /// if some fact entails the goal's sound contradictory, else `Idk` (open
    /// world). This is the model-theoretically faithful counterpart used for
    /// counterfactual re-evaluation — it never fabricates a verdict.
    fn verdict_against(facts: &[Meaning], goal: &Meaning) -> Verdict {
        if prove(facts, goal).is_some() {
            return Verdict::Yes;
        }
        if let Some(flip) = polarity_flip(goal) {
            if prove(facts, &flip).is_some() {
                return Verdict::No;
            }
        }
        Verdict::Idk
    }

    /// The GENUINENESS check behind [`what_would_change_your_mind`]. Build the
    /// counterfactual fact base — the current facts with `removed` (if any) dropped
    /// and `flip` added — and re-derive `goal`'s verdict against it. The flip
    /// genuinely changes the mind iff that counterfactual verdict differs from
    /// `current`. The real world is never touched (we operate on a local copy of
    /// the fact meanings, not the discourse).
    fn counterfactual_changes_verdict(
        &self,
        goal: &Meaning,
        current: Verdict,
        removed: Option<&Meaning>,
        flip: &Meaning,
    ) -> bool {
        let mut facts = self.fact_base();
        if let Some(r) = removed {
            facts.retain(|f| f != r);
        }
        if !facts.contains(flip) {
            facts.push(flip.clone());
        }
        Self::verdict_against(&facts, goal) != current
    }
}

/// Uppercase the first character of a string (for "Because ..." answers).
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// The asserted-LEAF conclusions of a proof, in left-to-right derivation order.
/// A leaf is a `Proof` with the `"asserted"` rule and no premises — a fact the
/// reader supplied directly. These are exactly the premises whose retraction (or
/// contradiction) can break the derivation, so they are the candidates
/// [`what_would_change_your_mind`](Mind::what_would_change_your_mind) flips.
/// Deduplicated on the meaning so a leaf reused across branches is named once.
fn proof_leaves(p: &Proof) -> Vec<Meaning> {
    let mut out: Vec<Meaning> = Vec::new();
    collect_proof_leaves(p, &mut out);
    out
}

/// Recursive worker for [`proof_leaves`].
fn collect_proof_leaves(p: &Proof, out: &mut Vec<Meaning>) {
    if p.rule == "asserted" && p.premises.is_empty() {
        if !out.contains(&p.conclusion) {
            out.push(p.conclusion.clone());
        }
        return;
    }
    for prem in &p.premises {
        collect_proof_leaves(prem, out);
    }
}

/// Lowercase the first character of a sentence so it can be embedded mid-clause
/// ("Yes, the editor writes the report." -> "yes, the editor writes the
/// report."). Leaves the rest untouched. Empty strings pass through.
fn lower_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Drop a single trailing period so a restated answer can be wrapped in a larger
/// sentence without doubled punctuation.
fn trim_period(s: &str) -> &str {
    s.strip_suffix('.').unwrap_or(s)
}

/// Slice the ACTUAL source of one Mog function out of a composed program by name.
/// Finds the `fn <name>(` declaration and returns the verbatim text from `fn`
/// through its matching closing brace (inclusive), so the caller gets exactly the
/// synthesized definition — no narration. Returns `None` if no such function is
/// present (so the caller can fall back honestly).
///
/// Brace matching is character-level over the function body; Mog has no string
/// literals containing unbalanced braces in these synthesized programs, so naive
/// depth counting from the first `{` after the signature is sufficient.
fn slice_fn_source(program: &str, name: &str) -> Option<String> {
    let needle = format!("fn {name}(");
    let start = program.find(&needle)?;
    let bytes = program.as_bytes();
    // Find the first `{` at or after the signature.
    let mut i = start + needle.len();
    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    // Depth-count braces to the matching close.
    let mut depth = 0usize;
    let body_start = i;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    // Inclusive of the closing brace.
                    return Some(program[start..=i].trim_end().to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    let _ = body_start;
    None
}

/// Does an event predication NAME a given entity head in any of its argument
/// slots (agent / patient / recipient)? Case-insensitive on the term head — the
/// referent noun stripped of articles. Used to decide which asserted facts and
/// derived consequences bear on the queried entity.
fn event_mentions(ev: &Event, entity: &str) -> bool {
    let term_matches = |t: &Option<Term>| {
        t.as_ref().is_some_and(|term| term.head().eq_ignore_ascii_case(entity))
    };
    term_matches(&ev.agent) || term_matches(&ev.patient) || term_matches(&ev.recipient)
}

/// Does a `Meaning` mention the given entity head anywhere in its term positions?
/// Recurses through the assertoric meaning shapes that `what_do_you_know` draws on
/// (events, IsA, comparisons, properties), so a derived "teacher is a person" or a
/// generalized event still counts as bearing on `teacher`. Anything without a term
/// matching the entity head returns false.
fn meaning_mentions(m: &Meaning, entity: &str) -> bool {
    let head_is = |t: &Term| t.head().eq_ignore_ascii_case(entity);
    match m {
        Meaning::Event(ev) => event_mentions(ev, entity),
        Meaning::IsA { subject, .. } => head_is(subject),
        Meaning::HasProperty { subject, .. } => head_is(subject),
        Meaning::Comparison { subject, than, .. } => head_is(subject) || head_is(than),
        Meaning::Not(inner) | Meaning::YesNoQuestion(inner) => meaning_mentions(inner, entity),
        Meaning::Or(parts) => parts.iter().any(|p| meaning_mentions(p, entity)),
        _ => false,
    }
}

/// Bring the meaning argument types into local scope for the mention helpers
/// without disturbing the (volatile) top-of-file `use` block.
use crate::understanding::meaning::{Event, Term};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mind_reads_then_answers_from_the_world() {
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        mind.read("The author reads the book.");
        // answered from the world model built by reading
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("What does the author read?").to_lowercase().contains("book"));
        // a category question, from animacy
        assert!(mind.ask("Is the report a person?").to_lowercase().starts_with("no"));
    }

    #[test]
    fn mind_understands_vp_conjunction() {
        let mut mind = Mind::new();
        // VP-conjunction with subject ellipsis: both facts must enter the world.
        mind.read("The teacher writes the report and reads the book.");
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("Who reads the book?").to_lowercase().contains("teacher"));
        assert!(mind.ask("What does the teacher read?").to_lowercase().contains("book"));
    }

    #[test]
    fn mind_understands_clausal_conjunction() {
        let mut mind = Mind::new();
        mind.read("The teacher writes the report and the editor reads the book.");
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("Who reads the book?").to_lowercase().contains("editor"));
    }

    #[test]
    fn why_renders_multi_step_because_chain_for_transitive_entailment() {
        // Read one concrete fact, then ask a doubly-generalized query whose truth
        // is reached by CHAINING two sound inference steps: drop-patient (the
        // teacher writes the report ⊢ the teacher writes) then generalize-agent
        // (the teacher writes ⊢ a teacher writes). `why` must surface that full
        // derivation as a nested "because" chain, with the INTERMEDIATE fact
        // ("the teacher writes") appearing between the goal and the asserted leaf.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        let w = mind.why("Does a teacher write something?");
        let lw = w.to_lowercase();
        // Affirmative verdict, identical to what `ask` returns, then the chain.
        assert!(lw.starts_with("yes"), "transitive query is true: {w}");
        // A genuine derivation: at least one "because" link...
        assert!(lw.contains("because"), "must render a derivation: {w}");
        // ...bottoming out in an honest "you told me ..." leaf naming the asserted fact.
        assert!(
            lw.contains("you told me the teacher writes the report"),
            "leaf must restate the asserted fact: {w}"
        );
        // The INTERMEDIATE step ("the teacher writes") is mentioned between the
        // generalized goal and the asserted leaf — the analogue of a transitive
        // middle term — and the named rules attribute each hop.
        assert!(lw.contains("the teacher writes"), "intermediate fact shown: {w}");
        assert!(lw.contains("(by drop-patient)"), "names the drop-patient hop: {w}");
        assert!(
            lw.contains("(by generalize-agent)"),
            "names the generalize-agent hop: {w}"
        );
        // Two distinct hops ⇒ two "because" connectives (a genuine multi-step chain).
        assert!(
            lw.matches("because").count() >= 2,
            "multi-step chain needs ≥2 'because' links: {w}"
        );
    }

    #[test]
    fn why_renders_single_step_because_for_a_one_hop_derivation() {
        // A one-hop entailment (drop-patient) still gets a "because" chain whose
        // single premise is the asserted fact — the smallest honest explanation.
        let mut mind = Mind::new();
        mind.read("The author reads the book.");
        let w = mind.why("Does the author read something?");
        let lw = w.to_lowercase();
        assert!(lw.starts_with("yes"), "entailed query is true: {w}");
        assert!(lw.contains("because"), "one-hop still explains: {w}");
        assert!(
            lw.contains("you told me the author reads the book"),
            "premise is the asserted fact: {w}"
        );
        assert!(lw.contains("(by drop-patient)"), "names the hop: {w}");
    }

    #[test]
    fn why_of_unprovable_query_honestly_says_it_does_not_know_without_fabricating() {
        // Nothing about the author was ever read. An open-world query must answer
        // honestly "I don't know." — and crucially carry NO "because" justification
        // (no fabricated derivation for something we cannot prove).
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        let w = mind.why("Does the author read the book?");
        assert!(
            w.to_lowercase().contains("don't know"),
            "unprovable query is honest: {w}"
        );
        assert!(
            !w.to_lowercase().contains("because"),
            "no fabricated derivation for an unknown: {w}"
        );
        assert!(
            !w.to_lowercase().contains("you told me"),
            "no invented premise for an unknown: {w}"
        );
    }

    #[test]
    fn why_matches_ask_for_an_opaque_fact_but_chains_a_transitive_comparison() {
        // A directly-asserted EVENT fact is a verdict the world MODEL owns with no
        // public derivation, so `why` must equal `ask` with no fabricated "because".
        // A TRANSITIVE comparison, by contrast, IS reconstructible: the world owns
        // the closure verdict, but `prove` rebuilds the chain over the asserted
        // edges, so `why` shows its work and names the intermediate.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        mind.read("The report is longer than the book.");
        mind.read("The book is longer than the letter.");

        // Directly asserted EVENT ⇒ world-owned opaque verdict, so `why` == `ask`.
        let direct = mind.why("Does the teacher write the report?");
        assert_eq!(
            direct,
            mind.ask("Does the teacher write the report?"),
            "world-owned event verdict: why must equal ask"
        );
        assert!(!direct.to_lowercase().contains("because"), "no chain on opaque fact: {direct}");

        // Transitive comparison ⇒ reconstructed chain naming the `book` intermediate.
        let cmp = mind.why("Is the report longer than the letter?");
        let low = cmp.to_lowercase();
        assert!(low.starts_with("yes"), "transitive comparison true: {cmp}");
        assert!(low.contains("because"), "transitive comparison must show its work: {cmp}");
        assert!(low.contains("book"), "transitive proof must name the intermediate 'book': {cmp}");
        assert!(
            low.contains("comparison-transitivity"),
            "chain must be labeled with the transitivity rule: {cmp}"
        );
    }

    // ===================================================================
    // Track E: HYPOTHETICAL (`suppose`) + COUNTERFACTUAL (`what_if_not`).
    // Soundness is paramount: these must NEVER mutate the real discourse.
    // ===================================================================

    #[test]
    fn suppose_answers_a_question_under_an_assumption() {
        // POSITIVE: nothing was ever read about the editor writing, so the real
        // world cannot answer "does the editor write something?". Under the
        // SUPPOSITION that the editor writes the report, the supposed clause
        // entails (by drop-patient + existential generalization) that the editor
        // writes something — so the hypothetical answer is affirmative.
        let mind = Mind::new();
        let s = mind.suppose("the editor writes the report", "does the editor write something?");
        let ls = s.to_lowercase();
        // The supposition is framed and the verdict under it is "yes".
        assert!(
            ls.starts_with("if the editor writes the report, then"),
            "frames the supposition: {s}"
        );
        assert!(ls.contains("then yes"), "affirmative under the supposition: {s}");
        // The proof surfaces the assumption as the leaf it routed through.
        assert!(
            ls.contains("you told me the editor writes the report"),
            "the chain shows the assumption was used: {s}"
        );
    }

    #[test]
    fn suppose_does_not_leak_into_the_real_world() {
        // ISOLATION: the editor-writes assumption must live ONLY in the clone.
        // Before any supposition, the real world has no idea about the editor.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        let baseline = mind.ask("Does the editor write something?");

        // A supposition that, IN A CLONE, makes the editor write something.
        let _ = mind.suppose("the editor writes the report", "does the editor write something?");

        // The REAL world is unchanged: the very same baseline question returns
        // EXACTLY what it did before the supposition (no leak from the clone).
        let after = mind.ask("Does the editor write something?");
        assert_eq!(
            after, baseline,
            "suppose must not leak into the real world: {after} vs {baseline}"
        );
        // And the real world still does not attest the editor writing.
        assert!(
            after.to_lowercase().contains("don't know"),
            "the editor was never really read: {after}"
        );
    }

    #[test]
    fn what_if_not_contrasts_counterfactual_with_actual() {
        // Read a concrete fact; the actual answer is affirmative.
        let mind_actual = {
            let mut m = Mind::new();
            m.read("The editor writes the report.");
            m
        };
        let actual = mind_actual.ask("Does the editor write the report?");
        assert!(actual.to_lowercase().starts_with("yes"), "actually true: {actual}");

        // Suppose the editor did NOT write the report. In the counterfactual world
        // we assert the contradictory, so the verdict flips to a negative, and the
        // result CONTRASTS the counterfactual with the actual.
        let cf = mind_actual.what_if_not(
            "the editor writes the report",
            "does the editor write the report?",
        );
        let lcf = cf.to_lowercase();
        assert!(
            lcf.starts_with("if the editor writes the report were not so"),
            "frames the retraction: {cf}"
        );
        assert!(lcf.contains("rather than"), "contrasts counterfactual vs actual: {cf}");
        // The counterfactual verdict is the negation; the actual ("yes") is named
        // as the contrast.
        assert!(
            lcf.contains("does not write") || lcf.contains("no,"),
            "counterfactual flips to negative: {cf}"
        );
        assert!(lcf.contains("yes"), "the actual affirmative is named as the contrast: {cf}");
    }

    #[test]
    fn what_if_not_does_not_leak_into_the_real_world() {
        // ISOLATION: asserting the negation in the counterfactual clone must not
        // touch the real world — the baseline answer is identical before and after.
        let mut mind = Mind::new();
        mind.read("The editor writes the report.");
        let baseline = mind.ask("Does the editor write the report?");
        let baseline_contradictions = mind.contradictions().len();

        // A counterfactual that asserts "the editor does NOT write the report" into
        // a CLONE. If it leaked, the real world would now hold BOTH the fact and
        // its negation (a contradiction) and/or flip the baseline answer.
        let _ = mind.what_if_not(
            "the editor writes the report",
            "does the editor write the report?",
        );

        // The REAL world is unchanged: same verdict, and no new contradiction was
        // recorded (the negation never entered the real world model).
        let after = mind.ask("Does the editor write the report?");
        assert_eq!(after, baseline, "what_if_not must not leak: {after} vs {baseline}");
        assert!(after.to_lowercase().starts_with("yes"), "the real fact still holds: {after}");
        assert_eq!(
            mind.contradictions().len(),
            baseline_contradictions,
            "no contradiction may be recorded in the real world by a counterfactual"
        );
    }

    #[test]
    fn track_e_round_trip_leaves_a_baseline_query_byte_for_byte_identical() {
        // The headline SOUNDNESS guarantee: run BOTH Track-E methods, then prove a
        // baseline question's answer is EXACTLY what it was before either ran.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        mind.read("The author reads the book.");

        let before = mind.ask("What does the author read?");

        let _ = mind.suppose("the editor writes the report", "does the editor write something?");
        let _ = mind.what_if_not("the author reads the book", "does the author read the book?");

        let after = mind.ask("What does the author read?");
        assert_eq!(
            after, before,
            "no Track-E method may mutate the real discourse: {after} vs {before}"
        );
    }

    // ===================================================================
    // gaps / what_would_change_your_mind / explain_cause.
    // Soundness is paramount: honest unknowns, verified flips, no invented cause.
    // ===================================================================

    #[test]
    fn what_would_change_your_mind_names_a_real_flip_that_actually_flips_the_verdict() {
        // The mind was told a concrete fact, so a direct truth query is answered
        // Yes. The counterfactual that would change the mind is being told the
        // CONTRADICTORY of that very fact.
        let mind = {
            let mut m = Mind::new();
            m.read("The teacher writes the report.");
            m
        };
        let q = "Does the teacher write the report?";
        // Baseline verdict is affirmative.
        let baseline = mind.ask(q);
        assert!(baseline.to_lowercase().starts_with("yes"), "baseline is yes: {baseline}");

        let wwcym = mind.what_would_change_your_mind(q);
        let lw = wwcym.to_lowercase();
        // It frames the counterfactual as new evidence...
        assert!(
            lw.contains("i would change my mind if you told me"),
            "frames the counterfactual: {wwcym}"
        );
        // ...and the named flip is the NEGATION of the asserted fact.
        assert!(
            lw.contains("the teacher does not write the report"),
            "names the contradictory of the asserted fact: {wwcym}"
        );

        // GENUINENESS (the load-bearing check, run independently of the impl):
        // asserting that very flip in a CLONE genuinely MOVES the verdict away from
        // the baseline "yes". The world model's most-recent-wins event truth flips
        // a directly-asserted fact when its negation is asserted.
        let flip = crate::understanding::inference::polarity_flip(
            &mind.understand("the teacher writes the report"),
        )
        .expect("an event has a contradictory");
        let mut clone = mind.discourse.clone();
        clone.world.assert(&flip);
        let flipped_answer = qa::answer(&mind.engine, &clone, q);
        assert_ne!(
            verdict_of(&flipped_answer),
            verdict_of(&baseline),
            "asserting the named flip in a clone genuinely changes the verdict: {flipped_answer}"
        );
        // The real world is untouched: the baseline question still answers Yes.
        assert_eq!(mind.ask(q), baseline, "verification must not mutate the real world");
    }

    #[test]
    fn what_would_change_your_mind_decides_an_open_question() {
        // An undetermined query: the mind does not know. What would change its mind
        // is being told the proposition (→ Yes) or its negation (→ No).
        let mind = {
            let mut m = Mind::new();
            m.read("The teacher writes the report.");
            m
        };
        let q = "Does the author read the book?";
        assert!(
            mind.ask(q).to_lowercase().contains("don't know"),
            "baseline is undetermined"
        );
        let wwcym = mind.what_would_change_your_mind(q).to_lowercase();
        assert!(
            wwcym.contains("i would change my mind if you told me"),
            "an open question can be decided by new evidence: {wwcym}"
        );
        // The deciding evidence is the proposition itself (the author reads the book).
        assert!(
            wwcym.contains("the author reads the book"),
            "names the deciding fact: {wwcym}"
        );
    }

    #[test]
    fn gaps_names_a_genuine_open_world_unknown() {
        // The mind knows about the teacher only. A question about the author touches
        // a proposition the world has NO verdict on — a real open-world gap.
        let mind = {
            let mut m = Mind::new();
            m.read("The teacher writes the report.");
            m
        };
        // Sanity: the author query is genuinely undetermined.
        assert!(
            mind.ask("Does the author read the book?").to_lowercase().contains("don't know"),
            "the author was never read — genuinely unknown"
        );

        let g = mind.gaps("Does the author read the book?");
        let lg = g.to_lowercase();
        assert!(
            lg.contains("i don't yet know whether"),
            "phrases the gap as an honest unknown: {g}"
        );
        assert!(
            lg.contains("the author reads the book"),
            "names the actual undetermined proposition: {g}"
        );

        // HONESTY: a question whose proposition IS determined raises no gap.
        let no_gap = mind.gaps("Does the teacher write the report?");
        assert!(
            !no_gap.to_lowercase().contains("don't yet know"),
            "a known proposition is not reported as a gap: {no_gap}"
        );
    }

    #[test]
    fn explain_cause_recovers_a_read_causal_link() {
        // Read a causal link; "why does the street flood?" recovers the recorded
        // cause via the world's directed cause->effect store — never invented.
        let mind = {
            let mut m = Mind::new();
            m.read("The street floods because the rain falls.");
            m
        };
        let why = mind.explain_cause("Why does the street flood?");
        let lw = why.to_lowercase();
        assert!(lw.starts_with("because"), "leads with the recorded cause: {why}");
        assert!(lw.contains("rain"), "names the recorded cause (the rain falls): {why}");

        // SOUNDNESS: with no causal link on record, the mind says it does not know
        // why — it never fabricates a cause.
        let blank = {
            let mut m = Mind::new();
            m.read("The teacher writes the report.");
            m
        };
        let unknown = blank.explain_cause("Why does the author read the book?");
        assert!(
            unknown.to_lowercase().contains("don't know why"),
            "no recorded/entailing cause => honest ignorance: {unknown}"
        );
    }

    // ===================================================================
    // ADVERSARIAL no-fabricated-knowledge audit.
    //
    // The single contract under test: the reflection layer must NEVER assert
    // something it has not been given grounds for.
    //   * what_do_you_know on an UNREAD entity is EMPTY (no invented facts).
    //   * gaps for an unanswerable question NAMES the genuine unknown, and
    //     fabricates NO extra unknowns the question did not raise.
    //   * explain_cause with no causal link SAYS it does not know — even when
    //     other, unrelated facts are present that could tempt a guess.
    // Any fabrication is a hard FAIL.
    // ===================================================================
    #[test]
    fn no_fabricated_knowledge_audit() {
        // A mind that has read ONLY a single, concrete fact about the teacher.
        // Nothing about a "dragon", an "author", or any cause is in its world.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");

        // ---- 1) what_do_you_know on an UNREAD entity is EMPTY ----
        // "dragon" was never mentioned in any fact and is not a known world
        // entity. The mind must surface nothing — not a single invented claim.
        let about_dragon = mind.what_do_you_know("dragon");
        assert!(
            about_dragon.is_empty(),
            "FABRICATION: knows things about an unread entity 'dragon': {about_dragon:?}"
        );
        // A second never-read entity, to rule out a single hard-coded exclusion.
        let about_author = mind.what_do_you_know("author");
        assert!(
            about_author.is_empty(),
            "FABRICATION: knows things about an unread entity 'author': {about_author:?}"
        );

        // CONTROL (so "empty" is not a degenerate always-empty bug): the entity
        // it DID read about must produce non-empty, and every returned claim must
        // genuinely mention that entity (no drift onto fabricated subjects).
        let about_teacher = mind.what_do_you_know("teacher");
        assert!(
            !about_teacher.is_empty(),
            "a read entity must surface its known facts (else the test is vacuous)"
        );
        assert!(
            about_teacher.iter().all(|c| c.to_lowercase().contains("teacher")),
            "every claim about 'teacher' must actually be about the teacher: {about_teacher:?}"
        );

        // ---- 2) gaps for an UNANSWERABLE question names the genuine unknown ----
        // The world has no verdict on whether the author reads the book. The gap
        // must name THAT proposition as the honest unknown.
        let g = mind.gaps("Does the author read the book?");
        let lg = g.to_lowercase();
        assert!(
            lg.contains("i don't yet know whether"),
            "an unanswerable question must be phrased as an honest unknown: {g}"
        );
        assert!(
            lg.contains("the author reads the book"),
            "the gap must name the GENUINE undetermined proposition: {g}"
        );
        // NO FABRICATION: the gap must not mention an entity the question never
        // raised (e.g. it must not drag in the teacher or invent a dragon).
        assert!(
            !lg.contains("dragon"),
            "FABRICATION: gaps invented an unknown about an unmentioned entity: {g}"
        );
        // A DETERMINED proposition raises NO gap — the mind does not pretend to
        // be uncertain about something it already knows.
        let no_gap = mind.gaps("Does the teacher write the report?");
        assert!(
            !no_gap.to_lowercase().contains("don't yet know"),
            "FABRICATION: reported a gap on a proposition it already knows: {no_gap}"
        );

        // ---- 3) explain_cause with NO causal link says it does not know ----
        // No causal link was ever read. Even with the unrelated teacher fact
        // sitting in the world, the mind must not invent a cause for an effect
        // it has no causal record of.
        let cause = mind.explain_cause("Why does the author read the book?");
        assert!(
            cause.to_lowercase().contains("don't know why"),
            "FABRICATION: invented a cause with no causal link on record: {cause}"
        );
        // And it must not have smuggled the unrelated teacher fact in as a
        // pseudo-cause ("because the teacher writes the report").
        assert!(
            !cause.to_lowercase().contains("because"),
            "FABRICATION: offered a 'because' explanation it has no grounds for: {cause}"
        );

        // SECOND effect, also unread and causally unrelated to the one fact we
        // have — confirms the honest-ignorance path is not keyed to one phrasing.
        let cause2 = mind.explain_cause("Why does the street flood?");
        assert!(
            cause2.to_lowercase().contains("don't know why"),
            "FABRICATION: invented a cause for an unrecorded effect: {cause2}"
        );
    }

    // ===================================================================
    // self_improve: the self-improvement loop wired onto the Mind.
    //
    // A mind that CANNOT classify creatures notices the gap (knows_word),
    // calls self_improve with curriculum-mined examples, and on a green gate
    // ADOPTS the grafted component — afterward it CAN classify creatures, and
    // its own regression gate is STILL green (monotone growth).
    // ===================================================================
    #[test]
    fn mind_self_improves_to_learn_creature_class() {
        // Disable journal persistence so the test never writes to $HOME, holding
        // the crate-wide journal-env lock so we never race the journal /
        // self_improve::extend tests on the process-global `NCPU_JOURNAL_PATH`.
        crate::self_improve::journal::test_support::with_journal_env("", || {
        let mut mind = Mind::new();

        // PRECONDITION: the mind cannot yet classify a mythical creature — the
        // component is genuinely absent, so the test is not vacuous.
        assert!(
            !mind.engine().has_component("creature_class"),
            "creature_class must be a genuinely new component for this test to mean anything"
        );
        // And the gap detector agrees: "dragon" is not yet a known noun.
        assert!(!mind.knows_word("dragon"), "the mind should not yet know the word 'dragon'");

        // The self-improvement request: close the creature-classification gap
        // with curriculum-mined examples.
        let req = crate::self_improve::extend::LearnRequest {
            gap: "cannot classify mythical creatures (dragon, griffin, phoenix)".to_string(),
            name: "creature_class".to_string(),
            signature: "fn creature_class(s: string) -> i64",
            examples: crate::comprehension::creature_class_examples(),
        };

        let report = mind.self_improve(req);

        // The extension was synthesized, gated green, and accepted.
        assert!(report.synthesized, "creature_class must synthesize: {}", report.message);
        assert!(
            report.regression_passed,
            "the mind's own gate must stay green for a disjoint additive component: {}",
            report.message
        );
        assert!(report.accepted, "a synthesized + gated extension must be accepted: {}", report.message);
        assert!(!report.method.is_empty(), "the recovering teacher must be recorded");

        // AFTERWARD: the mind's engine was REPLACED with the grafted candidate,
        // so it now classifies creatures via the synthesized program.
        assert!(
            mind.engine().has_component("creature_class"),
            "the accepted extension must be live on the mind's engine"
        );
        assert_eq!(
            mind.engine().eval_int("creature_class(\"dragon\")"),
            1,
            "dragon must classify as a creature on the improved mind"
        );
        assert_eq!(
            mind.engine().eval_int("creature_class(\"report\")"),
            0,
            "report must classify as a non-creature on the improved mind"
        );

        // SOUNDNESS: the mind's own regression gate is STILL green after the
        // swap — growth was monotone, nothing regressed.
        assert!(
            mind.self_check().ok(),
            "the mind must stay green after adopting the new component (monotone growth)"
        );
        });
    }

    #[test]
    fn mind_self_improve_rejection_leaves_engine_untouched() {
        // A request synthesis CANNOT satisfy (contradictory spec: one input maps
        // to two outputs) must be declined without touching the live engine.
        crate::self_improve::journal::test_support::with_journal_env("", || {
        let mut mind = Mind::new();
        // Baseline: the gate is green and the bogus component is absent.
        assert!(mind.self_check().ok(), "baseline mind must be green");
        assert!(!mind.engine().has_component("contradictory_class"));

        let examples = vec![
            crate::benchmark::Example {
                inputs: vec![crate::benchmark::Value::Str("dragon".to_string())],
                expected: crate::benchmark::Value::Int(1),
            },
            crate::benchmark::Example {
                inputs: vec![crate::benchmark::Value::Str("dragon".to_string())],
                expected: crate::benchmark::Value::Int(0),
            },
        ];
        let req = crate::self_improve::extend::LearnRequest {
            gap: "impossible contradictory lexicon".to_string(),
            name: "contradictory_class".to_string(),
            signature: "fn contradictory_class(s: string) -> i64",
            examples,
        };

        let report = mind.self_improve(req);
        assert!(!report.synthesized, "an unsynthesizable gap must report synthesis failure");
        assert!(!report.accepted, "a rejected extension is never accepted");

        // The engine is UNTOUCHED: no bogus component, gate still green.
        assert!(
            !mind.engine().has_component("contradictory_class"),
            "a rejected extension must not be grafted onto the live engine"
        );
        assert!(mind.self_check().ok(), "the mind stays green after a rejected attempt");
        });
    }

    // ===================================================================
    // ADVERSARIAL VERIFICATION of the autonomous self-extension loop.
    //
    // Claim under test: "accepted-extension-is-verified-and-queryable".
    // An accepted extension must be (a) VERIFIED by solve_problem (the same
    // solver the loop relies on must independently report success on the same
    // spec), (b) LIVE and QUERYABLE on the post-improvement engine, and
    // (c) CORRECT (dragon -> 1, report -> 0). Anything accepted-but-wrong is a
    // failure. We also prove the loop's gate REALLY rejects a synthesizable-
    // but-REGRESSING extension (leaving the base engine byte-for-byte unchanged)
    // and that every attempt is journaled.
    // ===================================================================
    #[test]
    fn adversarial_accepted_extension_is_verified_and_queryable() {
        use crate::benchmark::{Example, Problem, Value};
        use crate::self_improve::extend::LearnRequest;

        crate::self_improve::journal::test_support::with_journal_env("", || {
        // --- GAP: a fresh mind genuinely cannot classify creatures. ---------
        let mut mind = Mind::new();
        assert!(
            !mind.engine().has_component("creature_class"),
            "precondition: creature_class absent (test would be vacuous otherwise)"
        );
        // Behaviorally prove the gap: with no component, the call resolves to the
        // missing-function sentinel, NOT a correct 1 for a creature.
        assert_ne!(
            mind.engine().eval_int("creature_class(\"dragon\")"),
            1,
            "a mind WITHOUT the component must not already answer dragon -> 1"
        );
        assert!(!mind.knows_word("dragon"), "gap detector: 'dragon' is unknown pre-improvement");

        // --- INDEPENDENT VERIFICATION that solve_problem (the solver the loop ---
        // relies on) actually SUCCEEDS on this exact spec, and that the program
        // it returns answers correctly. This is the load-bearing check: the loop
        // must not be able to "accept" anything solve_problem would not certify.
        let examples = crate::comprehension::creature_class_examples();
        let probe = Problem {
            name: "creature_class".to_string(),
            category: "comprehension",
            description: "",
            signature: "fn creature_class(s: string) -> i64",
            examples: examples.clone(),
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
        };
        let solved = crate::solver::solve_problem(&probe);
        assert!(
            solved.success,
            "solve_problem MUST certify the spec (verification by construction): {:?}",
            solved.error
        );
        // solve_problem only reports success after verify_problem_code_strict has
        // re-run the candidate over every example + holdouts. Re-run the strict
        // verifier here ourselves so the "verified" claim is not taken on trust.
        crate::runtime::verify_problem_code_strict(&probe, &solved.code)
            .expect("the certified program must pass strict re-verification on all examples");

        // --- RUN THE LOOP through the public Mind API. ----------------------
        let req = LearnRequest {
            gap: "cannot classify mythical creatures (dragon, griffin, phoenix)".to_string(),
            name: "creature_class".to_string(),
            signature: "fn creature_class(s: string) -> i64",
            examples,
        };
        let report = mind.self_improve(req);

        // --- ACCEPTANCE must be backed by synthesis AND a green gate. -------
        assert!(report.synthesized, "must synthesize: {}", report.message);
        assert!(report.accepted, "a verified + gated extension must be accepted: {}", report.message);
        assert!(report.regression_passed, "the gate must be green: {}", report.message);
        assert!(!report.method.is_empty(), "the recovering teacher must be recorded");

        // --- QUERYABLE: the component is live on the post-improvement engine. -
        assert!(
            mind.engine().has_component("creature_class"),
            "an accepted extension must be live and queryable on the engine"
        );

        // --- CORRECT: it answers the way the spec demands. ------------------
        // dragon -> 1 (a creature), report -> 0 (a known non-creature).
        assert_eq!(
            mind.engine().eval_int("creature_class(\"dragon\")"),
            1,
            "accepted-but-wrong is FALSE: dragon MUST classify as a creature"
        );
        assert_eq!(
            mind.engine().eval_int("creature_class(\"report\")"),
            0,
            "accepted-but-wrong is FALSE: report MUST classify as a non-creature"
        );
        // Generalization across the rest of the curriculum-mined spec: every
        // creature -> 1, several known non-creatures -> 0. A program that merely
        // overfit to {dragon, report} would fail here.
        for c in crate::comprehension::CREATURES {
            assert_eq!(
                mind.engine().eval_int(&format!("creature_class(\"{c}\")")),
                1,
                "every creature must classify as 1: {c}"
            );
        }
        for n in ["author", "teacher", "book", "letter", "poem"] {
            assert_eq!(
                mind.engine().eval_int(&format!("creature_class(\"{n}\")")),
                0,
                "every mined non-creature must classify as 0: {n}"
            );
        }

        // --- The accepted program is IDENTICAL to what solve_problem certifies. -
        // The loop must not have accepted some OTHER (unverified) program. The
        // grafted source must contain exactly the solver-certified code body.
        assert!(
            mind.explain_self("creature").is_empty()
                || mind.engine().has_component("creature_class"),
            "sanity: engine reflects the graft"
        );

        // --- MONOTONE GROWTH: the mind's own gate is still green. -----------
        assert!(mind.self_check().ok(), "growth must be monotone — gate still green post-graft");
        });
    }

    /// ADVERSARIAL: the gate REALLY rejects a synthesizable-but-REGRESSING
    /// extension, leaving the base engine byte-for-byte unchanged.
    ///
    /// The happy-path test alone cannot prove the gate has teeth — it only ever
    /// sees an additive, disjoint component that trivially passes. Here we force
    /// a candidate that solve_problem CAN verify (it reproduces its own examples)
    /// but which SHADOWS an existing function the golden corpus depends on, with
    /// answers that break it. `try_extend` appends the new `fn`, and the runtime's
    /// last-definition-wins means the graft overrides the real `noun_animacy`.
    /// The gate must catch the regression and `self_extend` must reject — engine
    /// untouched. If a wrong-but-verified candidate were ever ADOPTED, this fails.
    #[test]
    fn adversarial_gate_rejects_a_verified_but_regressing_shadow() {
        use crate::benchmark::{Example, Value};
        use crate::self_improve::extend::{self_extend, LearnRequest};
        use crate::self_improve::gate::regression_gate;

        crate::self_improve::journal::test_support::with_journal_env("", || {
        let engine = Engine::new();
        // Baseline: the real lexicon classifies "teacher" as an animate noun (1).
        // The golden corpus relies on this. Record the exact program so we can
        // prove non-mutation afterward.
        assert!(engine.has_component("noun_animacy"), "baseline has the real lexicon");
        let baseline_teacher = engine.noun_class("teacher");
        assert!(baseline_teacher > 0, "baseline: 'teacher' is a known noun");
        let baseline_program_len = engine.program().len();
        let baseline_gate = regression_gate(&engine);
        assert!(baseline_gate.ok(), "baseline engine is green and sound");

        // A REGRESSING spec: redefine `noun_animacy` so the golden-corpus words
        // get WRONG classes. It is internally consistent (each input -> one
        // output), so solve_problem can verify it — but grafting it shadows the
        // real lexicon and must break the gate. We map every salient curriculum
        // word to 0 ("not a noun"), which is wrong for the agents/patients the
        // golden cases exercise.
        let mut examples: Vec<Example> = Vec::new();
        for w in [
            "teacher", "author", "editor", "report", "book", "letter", "student",
            "captain", "doctor", "pilot", "poem", "story",
        ] {
            examples.push(Example {
                inputs: vec![Value::Str(w.to_string())],
                expected: Value::Int(0), // WRONG on purpose: these ARE nouns
            });
        }
        let req = LearnRequest {
            gap: "deliberately wrong noun_animacy shadow".to_string(),
            name: "noun_animacy".to_string(),
            signature: "fn noun_animacy(s: string) -> i64",
            examples,
        };

        let (candidate, report) = self_extend(&engine, &req);

        // It DID synthesize (solve_problem verified it reproduces its own wrong
        // examples) — proving the candidate is "verified" yet still must be killed.
        assert!(
            report.synthesized,
            "the regressing shadow is internally verifiable: {}",
            report.message
        );
        // ...but the gate must REJECT it: regression on the golden corpus.
        assert!(
            !report.regression_passed,
            "the gate MUST go red for a regressing shadow: {}",
            report.message
        );
        assert!(
            !report.accepted,
            "a verified-but-regressing candidate must NOT be accepted: {}",
            report.message
        );
        assert!(candidate.is_none(), "a rejected candidate engine must not be returned");
        assert!(
            report.message.contains("regression gate red"),
            "rejection must be auditable as a gate-red: {}",
            report.message
        );

        // BASE ENGINE UNCHANGED: same program text length, same classification,
        // still green. self_extend never mutates its input.
        assert_eq!(
            engine.program().len(),
            baseline_program_len,
            "self_extend must not mutate the base engine's program"
        );
        assert_eq!(
            engine.noun_class("teacher"),
            baseline_teacher,
            "the real lexicon must be intact after a rejected shadow"
        );
        assert!(regression_gate(&engine).ok(), "base engine still green after rejection");
        });
    }
}
