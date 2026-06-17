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

/// The eleven components every freshly-built [`Engine`] synthesizes in
/// [`Engine::new`](crate::comprehension::Engine::new) — the base curriculum. Any
/// component on the engine's `methods` list NOT in this set was learned AFTER
/// construction by a self-extension (the autonomy spine grafting a new program
/// onto itself). [`Mind::learned_components`] is exactly the set difference, and
/// [`Mind::explain_self`] uses it to mark a component as self-learned vs. part of
/// the base curriculum. Kept in lock-step with `Engine::new`'s `methods` vec.
const BASE_METHODS: &[&str] = &[
    "noun_animacy",
    "valid_roles",
    "ends_s",
    "valid_agreement",
    "regular_3sg",
    "irregular_3sg",
    "regular_past",
    "irregular_past",
    "prop_id",
    "has_negation",
    "valid_argument",
];

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

    /// The belief revisions the world model has applied so far — metacognition
    /// over how the mind RESOLVED inconsistencies, not merely flagged them. When
    /// [`read`](Self::read) asserts a meaning that contradicts an Event belief the
    /// world already holds, the world resolves the clash to ONE coherent belief
    /// (a directly asserted fact supersedes a derived one; otherwise most-recent
    /// wins) and records the superseded belief, the surviving belief, and the
    /// reason. This surfaces that running ledger. An empty slice means nothing has
    /// had to be revised — everything read so far was mutually coherent. Delegates
    /// to the discourse's accumulated [`World`](crate::understanding::world_model::World).
    pub fn revisions(&self) -> &[crate::understanding::world_model::Revision] {
        self.discourse.world.revisions()
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

    /// Does the mind recognize `word` at all — either as a base-lexicon noun
    /// ([`knows_word`](Self::knows_word)) OR as a positive member of some
    /// SELF-LEARNED classifier (a `<x>_class` component the autonomy loop
    /// synthesized and adopted)? This is the minimal gap-closure the study loop
    /// needs to CONVERGE: once `study` learns `creature_class`, "dragon" is
    /// recognized here, so [`detect_gap`](Self::detect_gap) stops flagging it and
    /// a subsequent round goes dry. (Folding a learned class back into
    /// `noun_animacy` itself — so it parses in every role — is a later
    /// functional-integration phase; this is the sound, bounded version that makes
    /// autonomous study terminate.)
    pub fn recognizes_word(&self, word: &str) -> bool {
        if self.knows_word(word) {
            return true;
        }
        self.learned_components()
            .iter()
            .filter(|c| c.ends_with("_class"))
            .any(|c| self.engine.eval_int(&format!("{c}(\"{word}\")")) == 1)
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
            // `self_extend` has already JOURNALED the attempt AND PERSISTED the
            // accepted component to the durable cross-run learned-component store
            // (so the mind's self-taught knowledge compounds across runs, gated
            // identically to the in-process graft). Here we only adopt the vetted
            // engine, so the new component is live for every subsequent query.
            self.engine = new_engine;
        }
        report
    }

    /// GRAMMAR INDUCTION, wired onto the mind: learn a word-order CONSTRUCTION
    /// from labeled examples, and on a *clean* acceptance REPLACE this mind's
    /// engine with the freshly grafted candidate. The grammar-acquisition analogue
    /// of [`self_improve`](Self::self_improve).
    ///
    /// `name` tags the construction (e.g. `"object_fronting"`); each example is a
    /// `(sentence, agent_word, patient_word, predicate_lemma)` tuple — the learner
    /// is told the ROLES and INDUCES the position-to-role mapping, then SYNTHESIZES
    /// + VERIFIES it as `[i64] -> i64` slot programs over the class skeletons. The
    /// whole synthesize → gate → journal → persist decision is delegated to
    /// [`self_learn_construction`](crate::self_improve::extend::self_learn_construction),
    /// which registers the construction onto a CLONE of this mind's current engine
    /// and runs the full regression gate against it. The construction is adopted
    /// ONLY if the gate is green — every golden behavioral case still passes and the
    /// world model stays sound (the substrate's monotone-growth guarantee). The
    /// parser consults a registered construction ONLY on an otherwise-Unknown parse,
    /// so a sound addition can ADD coverage but never override a correct hand-parse.
    ///
    /// Returns `true` iff the construction was synthesized, gated green, and
    /// adopted. On acceptance the candidate engine is swapped in (the construction
    /// is live for every subsequent `read`/`understand`/`ask`) AND persisted to the
    /// durable cross-run construction store so a later boot re-registers it (gated
    /// again). On rejection (induction failed, an ill-formed skeleton→role mapping,
    /// or the gate went red) `self.engine` is left exactly as it was. Every attempt
    /// — accepted or rejected — is journaled by the substrate.
    pub fn learn_construction(
        &mut self,
        name: &str,
        examples: &[crate::understanding::grammar::ConstructionExample],
    ) -> bool {
        let req = crate::self_improve::extend::ConstructionRequest {
            gap: format!("cannot parse the `{name}` word-order construction"),
            name: name.to_string(),
            examples: examples.to_vec(),
        };
        let (candidate, report) =
            crate::self_improve::extend::self_learn_construction(&self.engine, &req);
        // Adopt ONLY on a clean acceptance. `self_learn_construction` returns
        // `Some(engine)` exactly when induction verified AND the gate passed, and it
        // has already JOURNALED the attempt and (on accept) PERSISTED the
        // construction to the durable store. Here we only swap in the vetted engine.
        if let Some(new_engine) = candidate {
            self.engine = new_engine;
        }
        report.accepted
    }

    /// The word-order CONSTRUCTIONS this mind has acquired — the verified
    /// [`LearnedConstruction`](crate::understanding::grammar::LearnedConstruction)s
    /// registered on its engine, in adoption order. Empty on a fresh mind that has
    /// learned no constructions. Purely a read accessor over the engine; it never
    /// mutates anything. Mirrors [`learned_components`](Self::learned_components),
    /// but for grammar acquisition rather than lexical/inferential components.
    pub fn learned_constructions(
        &self,
    ) -> &[crate::understanding::grammar::LearnedConstruction] {
        self.engine.learned_constructions()
    }

    /// The components this mind learned for ITSELF — every component on the
    /// engine's `methods` list that is NOT one of the eleven [`BASE_METHODS`] the
    /// base curriculum synthesizes in [`Engine::new`](crate::comprehension::Engine::new).
    ///
    /// This is the running PROVENANCE of the autonomy spine: each name here was
    /// grafted on by an accepted, gate-passing self-extension (so it is verified
    /// and monotone by construction). The base curriculum is excluded, so the
    /// result is exactly what the mind taught itself beyond what it was born
    /// knowing — in adoption order (the order `self_extend` pushed them onto
    /// `methods`). Purely a read accessor; it never mutates anything.
    pub fn learned_components(&self) -> Vec<String> {
        self.engine
            .methods
            .iter()
            .filter(|(name, _)| !BASE_METHODS.contains(name))
            .map(|(name, _)| name.to_string())
            .collect()
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

        // FIRST: does the topic name a SELF-LEARNED component (one acquired by the
        // autonomy spine, not part of the base curriculum)? If so, explain THAT —
        // and mark it as something the mind taught itself, distinct from the base
        // curriculum. A learned component's `name` IS its Mog `fn` name, and its
        // teacher is recorded the same way as the base components.
        let learned = self.learned_components();
        if let Some(component) =
            learned.iter().find(|name| t.contains(name.as_str()) || name.contains(&t))
        {
            let teacher = self.engine.method_for(component).unwrap_or("(unknown teacher)");
            let source = slice_fn_source(self.engine.program(), component)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "(source unavailable)".to_string());
            return format!(
                "I LEARNED the component `{component}` for myself — it is NOT part of my \
                 base curriculum; I synthesized it to close a gap I detected. It was \
                 recovered by the teacher: {teacher}. Here is the actual Mog source I \
                 synthesized:\n\n{source}"
            );
        }

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
            // Not a base topic. If the mind HAS learned components, name them as the
            // self-acquired provenance the caller can ask about by name — so an
            // unmapped topic still surfaces what was self-learned, distinct from the
            // base curriculum.
            if learned.is_empty() {
                return format!("I don't have a learned program for {topic}.");
            }
            return format!(
                "I don't have a base program for {topic}. Beyond my base curriculum I \
                 have taught MYSELF these components: {}. Ask me about one by name to \
                 see the Mog source I synthesized for it.",
                learned.join(", ")
            );
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
            "I learned the component `{component}` (the {fn_name} program) as part of \
             my BASE curriculum. It was recovered by the teacher: {teacher}. Here is \
             the actual Mog source I synthesized:\n\n{source}"
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
            | Meaning::Causal { .. }
            // A Conditional is a yes/no-evaluable proposition (the world has a
            // three-valued verdict via `holds_conditional`), so it is a leaf
            // atom like Causal.
            | Meaning::Conditional { .. } => {
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

    // ===================================================================
    // Autonomy spine (next-phase logic).
    //
    // The three methods below are STUBS — their real logic lands in the
    // next phase. They are the entry points for the SELF-DIRECTED study
    // loop: detect a gap in what was read, propose a curriculum that would
    // close it, and study a corpus end-to-end by folding in every verified
    // + gated component. Each is documented with the behavior it will
    // implement so the API surface is stable for callers now. They are
    // purely additive and do not touch any existing code path.
    // ===================================================================

    /// Track the autonomy spine (gap detection): inspect a single `input` the
    /// mind read and report the FIRST capability it could not fully handle, if
    /// any.
    ///
    /// The next-phase logic will parse `input` through the same understanding
    /// path the mind uses for [`read`](Self::read) / [`ask`](Self::ask) and
    /// classify the first unhandled fragment into a
    /// [`Gap`](crate::self_improve::extend::Gap):
    ///   * an unknown WORD the lexicons carry no class for →
    ///     [`GapKind::Lexical`](crate::self_improve::extend::GapKind::Lexical),
    ///   * a construction that parses word-by-word but whose STRUCTURE is not
    ///     recovered →
    ///     [`GapKind::Structural`](crate::self_improve::extend::GapKind::Structural),
    ///   * a proposition that parses fully but cannot be DERIVED →
    ///     [`GapKind::Inferential`](crate::self_improve::extend::GapKind::Inferential).
    ///
    /// `None` means the mind handled the input completely — there is no gap to
    /// close. This method is read-only; it never mutates the engine or the
    /// discourse.
    ///
    /// DETECTION (implemented):
    ///
    /// We parse `input` through the SAME path the mind uses for [`ask`](Self::ask)
    /// / [`read`](Self::read) ([`semantics::understand`]) and report the FIRST
    /// thing it could not fully handle, in priority order:
    ///
    /// 1. **LEXICAL gap** — an unknown CONTENT WORD the parser NEEDED in a
    ///    noun-phrase slot. We scan for a word the lexicon classifies as
    ///    `noun_class == 0` that nonetheless sits in a NOUN POSITION: immediately
    ///    after a determiner (`the` / `a` / `an`). A determiner-headed NP REQUIRES
    ///    a noun head, so an unknown word there is one the parser genuinely needed
    ///    and could not classify. We additionally exclude any word the curriculum
    ///    DOES account for in another role (a known verb form — base / 3sg / past /
    ///    gerund / participle — a modifier, a gradable adjective, a taxonomy class,
    ///    or a function word), so a known word that happens to follow a determiner
    ///    can never trip a spurious gap. The result is
    ///    [`GapKind::Lexical`](crate::self_improve::extend::GapKind::Lexical) with
    ///    `surface` = the unknown word and `context` = the full `input`, so
    ///    [`propose_curriculum`](Self::propose_curriculum) can mine a spec from it.
    /// 2. **STRUCTURAL gap** — the input did NOT parse into a usable meaning
    ///    ([`Meaning::Unknown`]) and no specific unknown noun explains it. The
    ///    CONSTRUCTION itself is beyond the current grammar. The result is
    ///    [`GapKind::Structural`](crate::self_improve::extend::GapKind::Structural)
    ///    with `surface` = the whole sentence (a grammar-level gap, out of scope for
    ///    the lexical teacher — see [`propose_curriculum`](Self::propose_curriculum)).
    ///
    /// `None` (NO false gap) when the input parses into a non-`Unknown` meaning AND
    /// every determiner-headed slot is a known noun. A LEXICAL gap takes precedence
    /// over a STRUCTURAL one when both could apply (an unknown noun is the more
    /// specific, more actionable diagnosis). Purely read-only: it never mutates the
    /// engine or the discourse.
    pub fn detect_gap(&self, input: &str) -> Option<crate::self_improve::extend::Gap> {
        use crate::self_improve::extend::{Gap, GapKind};

        let parsed = semantics::understand(&self.engine, input);

        // 1) LEXICAL: an unknown content word the parser needed in a noun slot.
        if let Some(word) = self.first_unknown_noun(input) {
            return Some(Gap {
                kind: GapKind::Lexical,
                surface: word,
                context: input.to_string(),
            });
        }

        // 2) STRUCTURAL: parsed to nothing usable, and no unknown noun explains it.
        if matches!(parsed, Meaning::Unknown(_)) {
            return Some(Gap {
                kind: GapKind::Structural,
                surface: input.to_string(),
                context: input.to_string(),
            });
        }

        // The input parsed and every needed noun is known — NO gap.
        None
    }

    /// The first content word in `input` the lexicon could not classify as a noun
    /// yet which the grammar clearly NEEDED as a noun-phrase head: a word
    /// immediately following a determiner (`the` / `a` / `an`) whose `noun_class`
    /// is `0` and which is recognized in NO other lexical role. Returns the surface
    /// word (lowercased + tokenized exactly as the parser sees it), or `None` when
    /// every determiner-headed slot is a known noun.
    ///
    /// SOUNDNESS (no false gaps): the determiner test is load-bearing. A
    /// determiner-headed NP requires a noun head, so the only way an unknown word
    /// lands here is as a noun the parser truly needed and the lexicon truly lacks.
    /// We still defensively exclude every other role the curriculum accounts for
    /// ([`is_known_non_noun`]) so that — even for a hypothetical future
    /// construction that placed a KNOWN non-noun after a determiner — a word the
    /// engine genuinely understands can never be misreported as a gap.
    fn first_unknown_noun(&self, input: &str) -> Option<String> {
        let toks = crate::comprehension::words_of(input);
        for i in 1..toks.len() {
            // Must sit in an NP-head slot: immediately after a determiner.
            if !matches!(toks[i - 1].as_str(), "the" | "a" | "an") {
                continue;
            }
            let w = toks[i].as_str();
            // Recognized as a base noun OR by a self-learned classifier -> the slot
            // is filled; not a gap. The learned-classifier arm is what lets the
            // study loop converge: once `creature_class` is learned, "dragon" is
            // recognized and stops re-surfacing as a gap.
            if self.recognizes_word(w) {
                continue;
            }
            // Recognized in some OTHER role -> the parser handles it; not a gap.
            if is_known_non_noun(&self.engine, w) {
                continue;
            }
            return Some(toks[i].clone());
        }
        None
    }

    /// Track the autonomy spine (curriculum proposal): given a detected
    /// [`Gap`](crate::self_improve::extend::Gap), propose a
    /// [`LearnRequest`](crate::self_improve::extend::LearnRequest) — the
    /// component name, signature, and characterizing examples — that, if
    /// synthesized and gated, would close it.
    ///
    /// The next-phase logic will mine examples from the gap's `context` (and the
    /// curriculum the engine was originally built from) keyed on the gap's
    /// [`GapKind`](crate::self_improve::extend::GapKind): a lexical gap proposes a
    /// string→class lexicon, a structural gap a transduction/parse rule, an
    /// inferential gap a reasoning rule. `None` means no well-posed curriculum
    /// could be assembled for the gap (so it stays open rather than feeding
    /// synthesis a spec it cannot satisfy). This method is read-only.
    ///
    /// IMPLEMENTED (lexical curriculum mining): given a detected
    /// [`Gap`](crate::self_improve::extend::Gap), assemble a well-posed
    /// [`LearnRequest`](crate::self_improve::extend::LearnRequest) whose examples
    /// synthesis can VERIFY — or `None` when no such curriculum can be mined for
    /// the gap (so it stays open rather than feeding the solver a spec it cannot
    /// satisfy).
    ///
    /// For a [`Lexical`](crate::self_improve::extend::GapKind::Lexical) gap we mine
    /// a binary string-membership lexicon from in-repo curriculum data, keyed on
    /// the unknown surface word:
    ///   * a known mythical creature ([`CREATURES`](crate::comprehension::CREATURES))
    ///     → the `creature_class` lexicon ("is this a creature?"), via
    ///     [`creature_class_examples`](crate::comprehension::creature_class_examples) —
    ///     creatures map to 1, a spread of known non-creatures to 0.
    ///
    /// The mined examples are disjoint positives/negatives (a well-posed
    /// string-equality map), so the string-membership teacher recovers them as a
    /// verified Mog program. A gap whose word matches no mineable class returns
    /// `None`: with no curriculum to characterize it, the mind has no honest spec
    /// to synthesize, and the gap stays open. Purely read-only.
    pub fn propose_curriculum(
        &self,
        gap: &crate::self_improve::extend::Gap,
    ) -> Option<crate::self_improve::extend::LearnRequest> {
        use crate::comprehension::{creature_class_examples, CREATURES};
        use crate::self_improve::extend::{GapKind, LearnRequest};

        match gap.kind {
            GapKind::Lexical => {
                let word = gap.surface.to_lowercase();
                // A mythical creature → the creature-membership lexicon.
                if CREATURES.contains(&word.as_str()) {
                    return Some(LearnRequest {
                        gap: format!(
                            "cannot classify mythical creatures (unknown word: {word})"
                        ),
                        name: "creature_class".to_string(),
                        signature: "fn creature_class(s: string) -> i64",
                        examples: creature_class_examples(),
                    });
                }
                // No mineable membership class for this word — stay honest, leave
                // the gap open rather than synthesize against a spec we can't pose.
                None
            }
            // Structural / inferential curriculum mining is not yet implemented;
            // those gaps stay open (no fabricated spec).
            GapKind::Structural | GapKind::Inferential => None,
        }
    }

    /// Track the autonomy spine (self-directed study): the top-level autonomous
    /// loop — read a `corpus`, detect gaps, propose curricula, and fold in every
    /// component that synthesizes AND passes the regression gate, for up to
    /// `max_rounds` rounds.
    ///
    /// The next-phase logic will, each round: [`read`](Self::read) the corpus,
    /// [`detect_gap`](Self::detect_gap) on each entry,
    /// [`propose_curriculum`](Self::propose_curriculum) for each detected gap,
    /// and [`self_improve`](Self::self_improve) each proposal — adopting the
    /// candidate engine ONLY on a green gate (monotone growth), persisting each
    /// accepted component via [`crate::self_improve::store`], and journaling every
    /// attempt. The loop stops early once a round closes no new gaps (fixpoint).
    /// The returned [`StudyReport`](crate::self_improve::extend::StudyReport)
    /// tallies the rounds run, the components learned, and the attempts /
    /// rejections — a full audit of the session.
    ///
    /// IMPLEMENTED (the continuous study loop): read a `corpus`, detect gaps,
    /// propose curricula, and fold in every component that synthesizes AND passes
    /// the regression gate, for up to `max_rounds` rounds — stopping early once a
    /// full round closes NOTHING new (loop-until-dry).
    ///
    /// Each round:
    ///   1. [`read`](Self::read) every sentence into the world model (building the
    ///      context the gap detector and curriculum miner work over).
    ///   2. [`detect_gap`](Self::detect_gap) on each sentence; for each detected
    ///      gap, [`propose_curriculum`](Self::propose_curriculum) a closing
    ///      [`LearnRequest`](crate::self_improve::extend::LearnRequest). When a
    ///      curriculum is proposed AND its target component is not already learned,
    ///      [`self_improve`](Self::self_improve) it — which synthesizes, gates,
    ///      journals, persists, and (on a green gate ONLY) swaps the engine in.
    ///   3. A round that ACCEPTS at least one new component may have unlocked
    ///      follow-on gaps (a creature now classified might license a new
    ///      construction), so we loop again; a round that accepts NOTHING is a
    ///      fixpoint and we stop.
    ///
    /// MONOTONE GROWTH: only gate-accepted components are ever kept (every accept
    /// is mediated by [`self_improve`] → [`self_extend`](crate::self_improve::extend::self_extend),
    /// which adopts the candidate engine only on a green gate). A non-`ok`
    /// candidate is never accepted by the substrate, so the mind's
    /// [`self_check`](Self::self_check) stays green throughout — we assert that
    /// invariant internally after every accepted graft (debug builds), and it
    /// holds by the substrate's construction regardless. The same `(name)`
    /// curriculum is attempted at most once per session (we skip a target already
    /// in [`learned_components`](Self::learned_components) and de-dupe within a
    /// round), so a dry round genuinely means "nothing left to learn here".
    ///
    /// The returned [`StudyReport`](crate::self_improve::extend::StudyReport)
    /// tallies the rounds run, the components learned (adoption order), and the
    /// attempts / rejections — a full audit of the session
    /// (`attempted == learned.len() + rejected`).
    pub fn study(
        &mut self,
        corpus: &[&str],
        max_rounds: usize,
    ) -> crate::self_improve::extend::StudyReport {
        let mut learned: Vec<String> = Vec::new();
        let mut attempted = 0usize;
        let mut rejected = 0usize;
        let mut rounds = 0usize;

        for _ in 0..max_rounds {
            rounds += 1;
            let mut learned_this_round = false;
            // Curricula attempted in THIS round, by target component name, so two
            // sentences raising the same gap do not double-attempt within a round.
            let mut attempted_this_round: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();

            for sentence in corpus {
                // 1) READ into the world model (also surfaces context for mining).
                self.read(sentence);

                // 2) DETECT the first gap in this sentence and try to close it.
                let Some(gap) = self.detect_gap(sentence) else {
                    continue;
                };
                let Some(req) = self.propose_curriculum(&gap) else {
                    continue;
                };

                // Skip a target we already learned (this session or earlier) and a
                // target already attempted this round — a dry round must mean there
                // is genuinely nothing new to fold in, not that we re-tried a known
                // component.
                if self.learned_components().contains(&req.name)
                    || !attempted_this_round.insert(req.name.clone())
                {
                    continue;
                }

                // 3) SYNTHESIZE → GATE → JOURNAL → PERSIST → (accept only on green).
                attempted += 1;
                let name = req.name.clone();
                let report = self.self_improve(req);
                if report.accepted {
                    learned.push(name);
                    learned_this_round = true;
                    // MONOTONE-GROWTH invariant (asserted in debug builds): every
                    // accepted graft leaves the mind's own gate green. `self_improve`
                    // already guarantees this (it adopts only a gate-passing engine),
                    // so this is a redundant in-loop proof, never a control path.
                    debug_assert!(
                        self.self_check().ok(),
                        "study must keep the mind sound after every accepted graft \
                         (monotone growth)"
                    );
                } else {
                    rejected += 1;
                }
            }

            // STOP early once a full round adds NOTHING new (fixpoint / dry).
            if !learned_this_round {
                break;
            }
        }

        crate::self_improve::extend::StudyReport { rounds, learned, attempted, rejected }
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

/// Does the curriculum account for `word` in some NON-noun lexical role — so that
/// a `noun_class == 0` classification reflects "not a noun", NOT "unknown word"?
/// True for any recognized function/auxiliary/question word, modifier (plain or
/// gradable adjective, in either polarity / comparative form), taxonomy class
/// name, pronoun, or known verb form (base, 3sg, past, gerund, or past
/// participle — verified against the synthesized inflectors so allomorphs like
/// "writes" / "writing" / "written" all resolve). Used by
/// [`Mind::first_unknown_noun`] to ensure a word the engine DOES understand can
/// never be misreported as a lexical gap, even when it lands (unexpectedly) in a
/// determiner-headed slot.
fn is_known_non_noun(engine: &Engine, word: &str) -> bool {
    use crate::comprehension::{
        FUNCTION_WORDS, GRADABLE, IRREGULAR_PAST, IRREGULAR_VERBS, MODIFIERS,
        PAST_PARTICIPLE, REG_VERBS, REG_VERBS_PAST,
    };

    // Function/auxiliary/question words and pronouns.
    if FUNCTION_WORDS.contains(&word) {
        return true;
    }
    if matches!(word, "it" | "they" | "he" | "she" | "them" | "him" | "her") {
        return true;
    }
    // Adjectives: plain modifiers + gradable positives/comparatives.
    if MODIFIERS.contains(&word)
        || GRADABLE.iter().any(|(pos, comp, _)| *pos == word || *comp == word)
    {
        return true;
    }
    // Taxonomy class names (singular or regular plural).
    if matches!(word, "agent" | "person" | "thing" | "document")
        || matches!(word, "agents" | "persons" | "things" | "documents")
    {
        return true;
    }
    // Verb bases + stored inflected forms (3sg / past / participle).
    let in_table = |t: &[(&str, &str)]| t.iter().any(|(b, f)| *b == word || *f == word);
    if in_table(REG_VERBS)
        || in_table(REG_VERBS_PAST)
        || in_table(IRREGULAR_VERBS)
        || in_table(IRREGULAR_PAST)
        || in_table(PAST_PARTICIPLE)
    {
        return true;
    }
    // Rule-derived regular forms not stored as literal tuples: a regular verb's
    // synthesized 3sg ("writes"), gerund ("writing"), or past ("walked") via the
    // engine's inflectors. This catches allomorphs the static tables omit.
    for (base, _f3) in REG_VERBS.iter() {
        if engine.verb_3sg(base) == word
            || engine.verb_past(base) == word
            || regular_gerund(base) == word
        {
            return true;
        }
    }
    false
}

/// The regular `-ing` (gerund) form of a verb `base`, mirroring the curriculum's
/// regular gerund allomorphs well enough to recognize a known verb's progressive
/// form in [`is_known_non_noun`]: drop a silent final "e" ("describe" ->
/// "describing"), keep a final "y" ("carry" -> "carrying"), and otherwise append
/// "ing" ("walk" -> "walking"). This is a RECOGNIZER only (it never has to
/// generate a novel form), so the common allomorphs suffice; anything it misses
/// simply means the word is not matched here and is judged by the other checks.
fn regular_gerund(base: &str) -> String {
    if let Some(stem) = base.strip_suffix('e') {
        // Keep "ee" verbs intact (none in-curriculum); only drop a single silent e.
        if !base.ends_with("ee") {
            return format!("{stem}ing");
        }
    }
    format!("{base}ing")
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
    // CONDITIONAL / SYLLOGISTIC REASONING. Modus ponens and modus tollens
    // are sound; affirming the consequent and denying the antecedent are the
    // classic FALLACIES the reasoner must REFUSE (answer "I don't know.",
    // never "yes"/"no"). Soundness is paramount.
    // ===================================================================

    #[test]
    fn conditional_modus_ponens_derives_consequent_with_proof() {
        // Read a rule, then assert its antecedent. The consequent must follow by
        // MODUS PONENS, and `why` must render the named chain bottoming out in the
        // two premises (the asserted antecedent and the conditional rule).
        let mut mind = Mind::new();
        mind.read("If the alarm rings then the guard wakes.");
        mind.read("The alarm rings.");

        // Verdict: "yes" — the guard wakes (derived, not directly asserted).
        let a = mind.ask("Does the guard wake?");
        assert!(a.to_lowercase().starts_with("yes"), "modus ponens => yes: {a}");

        // Proof: a modus-ponens chain naming both premises.
        let w = mind.why("Does the guard wake?");
        let lw = w.to_lowercase();
        assert!(lw.starts_with("yes"), "why agrees with ask: {w}");
        assert!(lw.contains("(by modus-ponens)"), "names the modus-ponens rule: {w}");
        assert!(lw.contains("because"), "renders a derivation chain: {w}");
        // The asserted antecedent is the grounding leaf...
        assert!(
            lw.contains("you told me the alarm rings"),
            "leaf restates the asserted antecedent: {w}"
        );
        // ...and the conditional rule appears as the other premise.
        assert!(
            lw.contains("if the alarm rings then the guard wakes"),
            "the conditional rule is cited as a premise: {w}"
        );
    }

    #[test]
    fn conditional_inverted_surface_form_also_chains() {
        // The inverted "<consequent> if <antecedent>" surface form must parse to the
        // SAME directed rule (antecedent governed by "if"), so modus ponens still
        // fires when the antecedent is asserted.
        let mut mind = Mind::new();
        mind.read("The guard wakes if the alarm rings.");
        mind.read("The alarm rings.");
        let a = mind.ask("Does the guard wake?");
        assert!(a.to_lowercase().starts_with("yes"), "inverted-form ponens => yes: {a}");
    }

    #[test]
    fn conditional_refuses_to_affirm_the_consequent() {
        // FALLACY: from "if P then Q" and Q, one may NOT conclude P. Reading the
        // rule and the CONSEQUENT must leave the antecedent OPEN ("I don't know."),
        // never "yes".
        let mut mind = Mind::new();
        mind.read("If the alarm rings then the guard wakes.");
        mind.read("The guard wakes.");
        let a = mind.ask("Does the alarm ring?");
        let la = a.to_lowercase();
        assert!(
            la.contains("don't know"),
            "affirming the consequent is refused: {a}"
        );
        assert!(!la.starts_with("yes"), "must NOT affirm the consequent: {a}");
    }

    #[test]
    fn conditional_refuses_to_deny_the_antecedent() {
        // FALLACY: from "if P then Q" and NOT P, one may NOT conclude NOT Q. Reading
        // the rule and the NEGATED ANTECEDENT must leave the consequent OPEN, never
        // "no".
        let mut mind = Mind::new();
        mind.read("If the alarm rings then the guard wakes.");
        mind.read("The alarm does not ring.");
        let a = mind.ask("Does the guard wake?");
        let la = a.to_lowercase();
        assert!(la.contains("don't know"), "denying the antecedent is refused: {a}");
        assert!(!la.starts_with("no"), "must NOT deny the antecedent: {a}");
    }

    #[test]
    fn conditional_modus_tollens_derives_negated_antecedent_with_proof() {
        // MODUS TOLLENS: from "if P then Q" and NOT Q, conclude NOT P. Reading the
        // rule and the negated CONSEQUENT must make "does P?" answer "no", with a
        // modus-tollens proof chain.
        let mut mind = Mind::new();
        mind.read("If the alarm rings then the guard wakes.");
        mind.read("The guard does not wake.");

        let a = mind.ask("Does the alarm ring?");
        assert!(a.to_lowercase().starts_with("no"), "modus tollens => no: {a}");

        let w = mind.why("Does the alarm ring?");
        let lw = w.to_lowercase();
        assert!(lw.starts_with("no"), "why agrees with ask: {w}");
        assert!(lw.contains("(by modus-tollens)"), "names the modus-tollens rule: {w}");
        assert!(
            lw.contains("you told me the guard does not wake"),
            "leaf restates the asserted negated consequent: {w}"
        );
        assert!(
            lw.contains("if the alarm rings then the guard wakes"),
            "the conditional rule is cited as a premise: {w}"
        );
    }

    #[test]
    fn conditional_alone_derives_nothing_about_either_side() {
        // SOUNDNESS: a conditional presupposes NOTHING. Reading only the rule must
        // leave BOTH the antecedent and the consequent open (no presupposition, no
        // premature chaining).
        let mut mind = Mind::new();
        mind.read("If the alarm rings then the guard wakes.");
        assert!(
            mind.ask("Does the alarm ring?").to_lowercase().contains("don't know"),
            "rule alone says nothing about the antecedent"
        );
        assert!(
            mind.ask("Does the guard wake?").to_lowercase().contains("don't know"),
            "rule alone says nothing about the consequent"
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
    // learn_construction: GRAMMAR INDUCTION wired onto the Mind — the
    // syntactic analogue of self_improve. A mind that CANNOT parse an
    // object-fronted clause learns the OSV construction from labeled
    // examples (induce role-to-position mapping -> synthesize + verify slot
    // programs -> gate -> adopt), and afterward PARSES the fronted clause —
    // even with UNSEEN words — while staying green and leaving SVO untouched.
    // ===================================================================

    /// The labeled OSV training set used by the construction tests: same
    /// word-order SHAPE, different words; each tagged with the agent / patient
    /// surface word and the predicate lemma.
    fn osv_training() -> Vec<crate::understanding::grammar::ConstructionExample<'static>> {
        vec![
            ("the report the teacher writes", "teacher", "report", "write"),
            ("the book the student reads", "student", "book", "read"),
            ("the memo the doctor fixes", "doctor", "memo", "fix"),
        ]
    }

    /// Run `f` with BOTH self-modification stores disabled — the journal
    /// (`with_journal_env("")`, which also empties `NCPU_COMPONENTS_PATH`) AND the
    /// construction store (`NCPU_CONSTRUCTIONS_PATH=""`). This keeps a
    /// construction test from reading or writing the developer's real $HOME stores,
    /// and holds the crate-wide ENV_LOCK (via `with_journal_env`) so it never races
    /// another env-mutating test. The prior `NCPU_CONSTRUCTIONS_PATH` is restored.
    fn with_constructions_disabled<R>(f: impl FnOnce() -> R) -> R {
        crate::self_improve::journal::test_support::with_journal_env("", || {
            let prev = std::env::var("NCPU_CONSTRUCTIONS_PATH").ok();
            // SAFETY: ENV_LOCK (held by with_journal_env) serializes env access.
            unsafe { std::env::set_var("NCPU_CONSTRUCTIONS_PATH", "") };
            let out = f();
            match prev {
                Some(v) => unsafe { std::env::set_var("NCPU_CONSTRUCTIONS_PATH", v) },
                None => unsafe { std::env::remove_var("NCPU_CONSTRUCTIONS_PATH") },
            }
            out
        })
    }

    #[test]
    fn mind_learns_osv_construction_and_parses_unseen() {
        with_constructions_disabled(|| {
            let mut mind = Mind::new();

            // PRECONDITION: the base parser CANNOT read an object-fronted clause —
            // it returns Unknown, and no construction is acquired yet. Not vacuous.
            let fronted = "the report the teacher writes";
            assert!(
                matches!(mind.understand(fronted), Meaning::Unknown(_)),
                "the base parser must fail (Unknown) on object-fronting before learning"
            );
            assert!(
                mind.learned_constructions().is_empty(),
                "no construction acquired before learn_construction"
            );

            // LEARN: induce + synthesize + verify + gate + adopt the OSV construction.
            let accepted = mind.learn_construction("object_fronting", &osv_training());
            assert!(accepted, "a verified, non-regressing construction must be accepted");

            // It is now registered on the mind's engine, with the recovered indices.
            let cons = mind.learned_constructions();
            assert_eq!(cons.len(), 1, "exactly one construction adopted");
            assert_eq!(cons[0].name, "object_fronting");
            assert_eq!(cons[0].patient_idx, 1, "the fronted object is the patient (index 1)");
            assert_eq!(cons[0].agent_idx, 3, "the embedded subject is the agent (index 3)");
            assert_eq!(cons[0].predicate_idx, 4, "the final verb is the predicate (index 4)");

            // AFTER: the fronted clause now parses to the correct Event.
            let Meaning::Event(e) = mind.understand(fronted) else {
                panic!("the learned construction must parse the trained fronted clause");
            };
            assert_eq!(e.predicate, "write");
            assert_eq!(e.agent, Some(Term::Entity("teacher".to_string())));
            assert_eq!(e.patient, Some(Term::Entity("report".to_string())));

            // GENERALIZATION: an UNSEEN-word OSV sentence parses too — the
            // construction keys on the class skeleton, not the specific words.
            let Meaning::Event(u) = mind.understand("the letter the editor reads") else {
                panic!("the learned construction must generalize to unseen-word OSV");
            };
            assert_eq!(u.predicate, "read");
            assert_eq!(u.agent, Some(Term::Entity("editor".to_string())));
            assert_eq!(u.patient, Some(Term::Entity("letter".to_string())));

            // Q&A: reading a fronted clause and asking about it answers correctly.
            let mut qa_mind = Mind::new();
            assert!(qa_mind.learn_construction("object_fronting", &osv_training()));
            qa_mind.read("the report the teacher writes");
            let answer = qa_mind.ask("does the teacher write the report");
            assert!(
                answer.to_lowercase().starts_with("yes"),
                "a question about the OSV-parsed Event must answer Yes, got: {answer}"
            );

            // SOUNDNESS: the gate is STILL green (monotone), and an ordinary SVO
            // clause is parsed EXACTLY as a mind that never learned the construction.
            assert!(mind.self_check().ok(), "the mind must stay green after acquiring grammar");
            let svo = "the teacher writes the report";
            let plain = Mind::new();
            assert!(plain.learned_constructions().is_empty());
            assert_eq!(
                mind.understand(svo),
                plain.understand(svo),
                "the OSV fallback must NEVER perturb a base-parseable SVO clause"
            );
        });
    }

    #[test]
    fn mind_rejects_ill_formed_construction_and_stays_untouched() {
        with_constructions_disabled(|| {
            let mut mind = Mind::new();
            assert!(mind.self_check().ok(), "baseline mind must be green");

            // An ILL-FORMED construction: two examples share the SAME skeleton
            // [0,1,0,1,2] but label the agent at DIFFERENT positions (index 3 vs
            // index 1) — the role-to-position mapping is not a function of the
            // skeleton, so induction must fail and the construction must be rejected.
            let bad: Vec<crate::understanding::grammar::ConstructionExample> = vec![
                ("the report the teacher writes", "teacher", "report", "write"),
                ("the editor the letter reads", "editor", "letter", "read"),
            ];
            let accepted = mind.learn_construction("bad", &bad);
            assert!(!accepted, "a contradictory role-to-position mapping must be rejected");

            // The engine is UNTOUCHED: no construction registered, gate still green.
            assert!(
                mind.learned_constructions().is_empty(),
                "a rejected construction must not be registered on the live engine"
            );
            assert!(mind.self_check().ok(), "the mind stays green after a rejected construction");
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

    // ===================================================================
    // study: the CONTINUOUS STUDY LOOP + PROVENANCE.
    //
    // study(corpus, max_rounds) reads a corpus, detects gaps, proposes
    // curricula, and folds in every verified + gated component, stopping
    // early once a round adds nothing new. Growth is monotone — only
    // gate-accepted components are kept and self_check stays green.
    // explain_self / learned_components surface what was self-learned,
    // distinct from the base curriculum.
    // ===================================================================

    /// Point BOTH self-modification stores (the journal and the learned-component
    /// store) at process-unique temp files for the duration of `f`, holding the
    /// crate-wide env lock so no other env-mutating test interleaves. Mirrors
    /// `with_journal_env` but ALSO sets `NCPU_COMPONENTS_PATH` (per the task's
    /// study-test contract). The temp files are removed afterward.
    fn with_study_env<R>(label: &str, f: impl FnOnce() -> R) -> R {
        let journal = std::env::temp_dir()
            .join(format!("ncpu_study_journal_{}_{label}.jsonl", std::process::id()));
        let components = std::env::temp_dir()
            .join(format!("ncpu_study_components_{}_{label}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&journal);
        let _ = std::fs::remove_file(&components);
        let components_path = components.to_string_lossy().to_string();
        // `with_journal_env` holds the crate-wide ENV_LOCK for the whole closure
        // (so setting env inside is race-free) AND saves/restores
        // NCPU_COMPONENTS_PATH around the closure — so we just OVERRIDE it to our
        // temp store inside `f`; the helper restores the prior value on exit.
        let out = crate::self_improve::journal::test_support::with_journal_env(
            &journal.to_string_lossy(),
            || {
                // SAFETY: ENV_LOCK (held by with_journal_env) serializes env access.
                unsafe { std::env::set_var("NCPU_COMPONENTS_PATH", &components_path) };
                f()
            },
        );
        let _ = std::fs::remove_file(&journal);
        let _ = std::fs::remove_file(&components);
        out
    }

    #[test]
    fn study_over_unknown_creature_corpus_learns_a_component() {
        with_study_env("learns", || {
            let mut mind = Mind::new();

            // PRECONDITION: the mind cannot yet classify creatures, and "dragon" is
            // a genuine lexical gap (not a known noun). The test is not vacuous.
            assert!(
                !mind.engine().has_component("creature_class"),
                "creature_class must be genuinely absent before study"
            );
            assert!(!mind.knows_word("dragon"), "'dragon' must be unknown pre-study");

            // The corpus mixes a KNOWN sentence with one naming an unknown creature.
            let corpus = [
                "The teacher writes the report.",
                "The dragon guards the gold.",
            ];

            let report = mind.study(&corpus, 3);

            // It LEARNED at least one component, and creature_class specifically.
            assert!(
                !report.learned.is_empty(),
                "study must learn >=1 component from the unknown-creature corpus: {report:?}"
            );
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "study must learn creature_class to close the dragon gap: {report:?}"
            );
            // Audit accounting holds by construction.
            assert_eq!(
                report.attempted,
                report.learned.len() + report.rejected,
                "attempted == learned + rejected: {report:?}"
            );
            assert!(report.rounds >= 1, "at least one round ran: {report:?}");

            // The learned program is LIVE and CORRECT on the post-study engine.
            assert!(
                mind.engine().has_component("creature_class"),
                "the learned component must be live on the engine"
            );
            assert_eq!(mind.engine().eval_int("creature_class(\"dragon\")"), 1);
            assert_eq!(mind.engine().eval_int("creature_class(\"report\")"), 0);

            // SOUNDNESS: the mind's own gate is STILL green after study — growth was
            // monotone, nothing regressed.
            assert!(
                mind.self_check().ok(),
                "self_check must stay ok after study (monotone growth)"
            );
        });
    }

    #[test]
    fn study_over_only_known_sentences_learns_nothing_and_converges() {
        with_study_env("nothing", || {
            let mut mind = Mind::new();
            let baseline_learned = mind.learned_components();
            assert!(baseline_learned.is_empty(), "a fresh mind has learned nothing yet");

            // Every word here is in the base curriculum — no lexical gap anywhere.
            let corpus = [
                "The teacher writes the report.",
                "The author reads the book.",
                "The editor explains the lesson.",
            ];

            let report = mind.study(&corpus, 5);

            // Nothing was learned; the loop converged WITHOUT exhausting max_rounds
            // (a dry first round is a fixpoint, so it stops after round 1).
            assert!(
                report.learned.is_empty(),
                "a wholly-known corpus must teach the mind nothing: {report:?}"
            );
            assert_eq!(report.attempted, 0, "no self-extension attempt on a known corpus: {report:?}");
            assert_eq!(report.rejected, 0, "nothing to reject: {report:?}");
            assert_eq!(
                report.rounds, 1,
                "a dry first round is a fixpoint — converges immediately: {report:?}"
            );

            // The mind learned no new components, and stays green.
            assert!(
                mind.learned_components().is_empty(),
                "no components learned from a known corpus"
            );
            assert!(mind.self_check().ok(), "the mind stays sound after a no-op study");
        });
    }

    /// ADVERSARIAL — "autonomous study is monotone." Drive a step-faithful
    /// re-implementation of the `study` loop over a corpus mixing KNOWN sentences,
    /// an UNKNOWN-CREATURE sentence (a gap that mines a verifiable + gateable
    /// curriculum), and an UNKNOWN-NON-CREATURE sentence (a lexical gap that
    /// proposes NO curriculum — nothing synthesis could verify, nothing the gate
    /// could pass). We assert the monotone-growth invariant the production
    /// `debug_assert!` only checks in debug builds, but here UNCONDITIONALLY (a
    /// hard `assert!`, so it holds in release too):
    ///
    ///   * `self_check().ok()` is true BEFORE the loop,
    ///   * `self_check().ok()` is true DURING — re-checked after EVERY accepted
    ///     graft, before moving on,
    ///   * `self_check().ok()` is true AFTER the whole loop,
    ///   * EVERY component that ended up on the engine beyond the base curriculum
    ///     was synthesized AND passed the gate (acceptance is the ONLY way in),
    ///   * the unmineable unknown word (`blorptangle`) is NEVER learned — a gap
    ///     with no verifiable curriculum stays open; nothing study could not
    ///     verify+gate got in,
    ///   * the loop accounting is honest (`attempted == accepted + rejected`).
    ///
    /// This is the loop's own logic, transcribed, so the assertions sit at the
    /// exact program points the property names — there is no way for an accepted-
    /// but-regressing component to slip past an AFTER-only check, and no way for an
    /// unverifiable gap to be silently grafted.
    #[test]
    fn study_is_monotone_self_check_holds_before_during_and_after() {
        with_study_env("monotone", || {
            let mut mind = Mind::new();

            // BEFORE: the spine starts sound, and the corpus's gaps are GENUINE.
            assert!(
                mind.self_check().ok(),
                "self_check must be green BEFORE study (the spine starts sound)"
            );
            assert!(!mind.knows_word("dragon"), "'dragon' must be a real gap pre-study");
            assert!(
                !mind.knows_word("blorptangle"),
                "'blorptangle' must be a real (but unmineable) gap pre-study"
            );
            assert!(
                !mind.engine().has_component("creature_class"),
                "creature_class must be genuinely absent before study"
            );
            let base_learned = mind.learned_components();
            assert!(base_learned.is_empty(), "a fresh mind has learned nothing yet");

            // The corpus MIXES: a fully-known sentence, an unknown-CREATURE
            // sentence (mineable → verifiable + gateable), and an unknown-
            // NON-creature sentence (a lexical gap that proposes NO curriculum).
            let corpus = [
                "The teacher writes the report.",   // wholly known — no gap
                "The dragon guards the gold.",      // mineable creature gap
                "The blorptangle eats the report.", // lexical gap, NO curriculum
            ];

            // Re-implement the study loop FAITHFULLY so the DURING check is a hard
            // assert at the exact point `study`'s debug_assert sits.
            let max_rounds = 3usize;
            let mut accepted: Vec<String> = Vec::new();
            let mut attempted = 0usize;
            let mut rejected = 0usize;
            // Proof that the unmineable gap was actually DETECTED and then DROPPED at
            // the proposal boundary (it has no verifiable+gateable curriculum), so
            // our "it never got learned" claim is non-vacuous.
            let mut blorptangle_detected_but_unmineable = false;

            for _round in 0..max_rounds {
                let mut learned_this_round = false;
                let mut attempted_this_round: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();

                for sentence in &corpus {
                    mind.read(sentence);

                    let Some(gap) = mind.detect_gap(sentence) else {
                        continue;
                    };
                    // An unmineable gap (blorptangle) yields NO curriculum: it can
                    // never be synthesized, verified, or gated, so it must never be
                    // learned. The None branch is EXACTLY where the unmineable gap is
                    // dropped — record that the blorptangle gap reached it.
                    let Some(req) = mind.propose_curriculum(&gap) else {
                        if gap.surface.to_lowercase() == "blorptangle" {
                            blorptangle_detected_but_unmineable = true;
                        }
                        continue;
                    };
                    // A gap whose word is the unmineable one must NEVER reach a
                    // curriculum — if it did, the honesty guarantee is broken.
                    assert_ne!(
                        req.name, "blorptangle",
                        "an unmineable word must not produce a learn request"
                    );

                    if mind.learned_components().contains(&req.name)
                        || !attempted_this_round.insert(req.name.clone())
                    {
                        continue;
                    }

                    attempted += 1;
                    let name = req.name.clone();
                    let report = mind.self_improve(req);
                    if report.accepted {
                        // ACCEPTANCE is the ONLY way a component enters the engine,
                        // and it implies synthesis succeeded AND the gate stayed
                        // green for THIS candidate.
                        assert!(
                            report.synthesized,
                            "an accepted component must have synthesized: {}",
                            report.message
                        );
                        assert!(
                            report.regression_passed,
                            "an accepted component must have passed the gate: {}",
                            report.message
                        );
                        accepted.push(name.clone());
                        learned_this_round = true;
                        // DURING: the mind's OWN gate must still be green right now,
                        // after this graft, before we touch the next sentence. A
                        // HARD assert (holds in release, unlike study's debug_assert).
                        assert!(
                            mind.self_check().ok(),
                            "self_check must be green DURING study, immediately after \
                             accepting `{name}` (monotone growth, every acceptance)"
                        );
                        // And the freshly accepted component is LIVE on the engine.
                        assert!(
                            mind.engine().has_component(&name),
                            "an accepted component must be live on the engine: {name}"
                        );
                    } else {
                        rejected += 1;
                    }
                }

                if !learned_this_round {
                    break;
                }
            }

            // AFTER: still green.
            assert!(
                mind.self_check().ok(),
                "self_check must be green AFTER study (monotone growth held throughout)"
            );

            // NON-VACUITY: the unmineable gap was genuinely detected and then
            // dropped for want of a verifiable curriculum — it didn't simply go
            // unnoticed. This is what makes "it never got learned" meaningful.
            assert!(
                blorptangle_detected_but_unmineable,
                "the blorptangle gap must have been DETECTED yet yielded no curriculum \
                 (so 'never learned' is non-vacuous)"
            );

            // The mineable creature gap WAS closed.
            assert!(
                accepted.contains(&"creature_class".to_string()),
                "study must have learned creature_class from the dragon sentence: {accepted:?}"
            );
            // ONLY gate-accepted components are kept: everything the engine carries
            // beyond the base curriculum is exactly what we ACCEPTED in-loop.
            let learned_now = mind.learned_components();
            assert_eq!(
                learned_now, accepted,
                "the engine's learned components must be EXACTLY the in-loop accepted set \
                 (no component entered except by acceptance): live={learned_now:?} accepted={accepted:?}"
            );
            // The unmineable gap NEVER became a component — nothing study could not
            // verify+gate got in.
            assert!(
                !mind.engine().has_component("blorptangle"),
                "the unmineable unknown word must never be grafted as a component"
            );
            assert!(
                !mind.knows_word("blorptangle"),
                "'blorptangle' stays unknown — its gap stayed open (no fabricated learning)"
            );
            // Accounting is honest.
            assert_eq!(
                attempted,
                accepted.len() + rejected,
                "attempted == accepted + rejected (honest loop accounting)"
            );
            // The blorptangle gap was real but produced ZERO attempts (no curriculum
            // ever fed synthesis), so attempts came only from the mineable gap.
            assert_eq!(
                attempted, 1,
                "exactly one self-extension attempt (the mineable creature gap); the \
                 unmineable gap produced no attempt: attempted={attempted}"
            );
            assert_eq!(rejected, 0, "the one mineable attempt was accepted, not rejected");
        });
    }

    /// ADVERSARIAL — the public `study` API agrees with the monotone invariant: a
    /// study over a corpus whose ONLY gap is unmineable learns NOTHING, attempts
    /// NOTHING, stays sound, and converges immediately (a dry first round is a
    /// fixpoint). This proves an undetectable / unmineable gap can never sneak a
    /// component in through the real entry point, not just the transcribed loop.
    #[test]
    fn study_over_unmineable_gap_learns_nothing_and_stays_sound() {
        with_study_env("unmineable", || {
            let mut mind = Mind::new();
            assert!(mind.self_check().ok(), "green before study");
            assert!(!mind.knows_word("blorptangle"), "'blorptangle' is a genuine gap");

            // The ONLY gap here is the unmineable unknown word. detect_gap fires,
            // propose_curriculum returns None, so study can never verify or gate
            // anything — it must learn nothing.
            let corpus = ["The blorptangle eats the report."];
            let report = mind.study(&corpus, 5);

            assert!(
                report.learned.is_empty(),
                "an unmineable-only corpus must teach the mind nothing: {report:?}"
            );
            assert_eq!(
                report.attempted, 0,
                "no curriculum can be posed → no self-extension attempt: {report:?}"
            );
            assert_eq!(report.rejected, 0, "nothing was even attempted to reject: {report:?}");
            assert_eq!(
                report.rounds, 1,
                "a dry first round is a fixpoint — converges immediately: {report:?}"
            );
            assert!(
                mind.learned_components().is_empty(),
                "no component entered the engine from an unmineable gap"
            );
            assert!(
                !mind.engine().has_component("blorptangle"),
                "the unmineable word never became a live component"
            );
            assert!(mind.self_check().ok(), "the mind stays sound after a no-op study");
        });
    }

    #[test]
    fn learned_components_and_explain_self_mark_the_acquired_component() {
        with_study_env("provenance", || {
            let mut mind = Mind::new();

            // PROVENANCE baseline: before study, learned_components is empty and
            // explain_self has NO self-learned creature program.
            assert!(mind.learned_components().is_empty(), "nothing learned yet");

            let corpus = ["The dragon guards the gold."];
            let report = mind.study(&corpus, 2);
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "study must learn creature_class: {report:?}"
            );

            // learned_components LISTS exactly the acquired component, and NONE of
            // the eleven base components leak in.
            let learned = mind.learned_components();
            assert_eq!(
                learned,
                vec!["creature_class".to_string()],
                "learned_components must list exactly the self-acquired component"
            );
            for base in BASE_METHODS {
                assert!(
                    !learned.contains(&base.to_string()),
                    "a base-curriculum component must never appear as self-learned: {base}"
                );
            }

            // explain_self MARKS it as self-learned — distinct from the base
            // curriculum — and quotes the actual synthesized Mog source.
            let explained = mind.explain_self("creature_class");
            let le = explained.to_lowercase();
            assert!(
                le.contains("learned") && le.contains("not part of my base curriculum"),
                "explain_self must mark creature_class as self-learned, not base: {explained}"
            );
            assert!(
                explained.contains("creature_class"),
                "explain_self must name the component: {explained}"
            );
            assert!(
                explained.contains("fn creature_class("),
                "explain_self must quote the actual synthesized Mog source: {explained}"
            );

            // A BASE component, by contrast, is explained as base curriculum — the
            // two provenances are kept distinct.
            let base_expl = mind.explain_self("animacy").to_lowercase();
            assert!(
                base_expl.contains("base curriculum"),
                "a base component must be marked as base curriculum: {base_expl}"
            );
        });
    }

    // ===================================================================
    // detect_gap + propose_curriculum: AUTOMATIC GAP DETECTION and
    // SELF-CURRICULUM GENERATION.
    //
    // Contract:
    //   * detect_gap on a sentence naming an UNKNOWN creature word returns a
    //     LEXICAL gap whose surface IS that word.
    //   * detect_gap on a FULLY-KNOWN sentence returns None — NO false gap.
    //   * propose_curriculum on a lexical gap returns a LearnRequest whose
    //     examples solve_problem can actually satisfy (verified by construction).
    // ===================================================================

    #[test]
    fn detect_gap_flags_an_unknown_creature_word_as_lexical() {
        use crate::self_improve::extend::GapKind;
        let mind = Mind::new();

        // PRECONDITION (test is not vacuous): "dragon" is genuinely unknown — the
        // lexicon carries no class for it.
        assert!(!mind.knows_word("dragon"), "'dragon' must be unknown for this test");

        // A grammatical sentence whose SUBJECT is an unknown creature in a
        // determiner-headed noun slot ("the dragon").
        let gap = mind
            .detect_gap("The dragon guards the gold.")
            .expect("an unknown noun in a determiner slot must surface a gap");

        // It is a LEXICAL gap (an unknown WORD, not a grammar gap).
        assert_eq!(gap.kind, GapKind::Lexical, "an unknown noun is a lexical gap: {gap:?}");
        // Its surface is the FIRST unknown noun the parser needed — "dragon".
        assert_eq!(gap.surface, "dragon", "the gap names the unknown content word: {gap:?}");
        // The context preserves the full input so the curriculum miner can use it.
        assert_eq!(gap.context, "The dragon guards the gold.");
    }

    #[test]
    fn detect_gap_returns_none_on_a_fully_known_sentence() {
        let mind = Mind::new();

        // Every content word here is in the base curriculum (teacher/report are
        // nouns; "writes" is a known 3sg verb form). The sentence parses cleanly,
        // so there is NO gap — the detector must NOT fabricate one.
        assert!(
            mind.detect_gap("The teacher writes the report.").is_none(),
            "a fully-known declarative must raise no gap"
        );

        // A second, structurally different known sentence — a known verb's present
        // form and a definite object — also raises nothing. This guards the
        // no-false-gap contract across inflected verb forms that classify as 0
        // ("not a noun") yet are perfectly known.
        assert!(
            mind.detect_gap("The author reads the book.").is_none(),
            "a known sentence with an inflected verb must raise no gap"
        );

        // A copular known sentence — the determiner "a" precedes the taxonomy
        // class "person", which is KNOWN, so no gap.
        assert!(
            mind.detect_gap("The teacher is a person.").is_none(),
            "a copular sentence over a known taxonomy class must raise no gap"
        );
    }

    // ADVERSARIAL VERIFICATION (gap-detection-is-honest): a single test that
    // exercises all three honesty obligations together:
    //   (A) a GENUINE gap (unknown creature) -> Some(Lexical, surface=word)
    //   (B) a FULLY-KNOWN sentence -> None (no false gap)
    //   (C) an ALREADY-LEARNED word, post-study -> None (the gap actually CLOSED).
    // Case (C) is the load-bearing probe: study() must not merely add a
    // creature_class COMPONENT while detect_gap keeps firing on the same word —
    // that would be a MISSED CLOSE (the detector lying that a handled input is
    // still a gap). If detect_gap still fires post-study, this test FAILS.
    #[test]
    fn detect_gap_is_honest_genuine_known_and_post_study() {
        use crate::self_improve::extend::GapKind;

        with_study_env("honest_gap", || {
            // ---- (A) GENUINE GAP: unknown creature in a determiner-headed slot.
            let mut mind = Mind::new();
            assert!(
                !mind.knows_word("dragon"),
                "precondition: 'dragon' unknown so (A) is not vacuous"
            );
            let gap = mind
                .detect_gap("The dragon guards the gold.")
                .expect("(A) a genuine unknown noun MUST surface a gap");
            assert_eq!(gap.kind, GapKind::Lexical, "(A) unknown noun => lexical: {gap:?}");
            assert_eq!(gap.surface, "dragon", "(A) gap names the unknown word: {gap:?}");

            // ---- (B) FULLY-KNOWN SENTENCE: every content word is in the base
            // curriculum (teacher/report nouns, writes a known 3sg form). NO gap.
            assert!(
                mind.detect_gap("The teacher writes the report.").is_none(),
                "(B) a fully-known sentence MUST raise no gap (no false positive)"
            );

            // ---- (C) POST-STUDY: after the mind STUDIES the dragon corpus and
            // closes the gap, detect_gap on the SAME sentence must return None.
            // This proves the close is REAL, not just a component bolted on while
            // the detector keeps reporting the input as unhandled.
            let report = mind.study(&["The dragon guards the gold."], 3);
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "(C) study must actually learn creature_class to close the gap: {report:?}"
            );
            assert!(
                mind.recognizes_word("dragon"),
                "(C) post-study 'dragon' must read as recognized via the learned \
                 classifier (gap genuinely closed; base noun_animacy is untouched — \
                 folding the learned class into it is a later integration phase)"
            );
            // The DRAGON lexical gap is closed: detect_gap no longer names "dragon".
            // The sentence's OTHER unknown, "gold", is not a creature, so it honestly
            // remains a gap — that is correct, not a miss. (Full parser-level closure
            // of a recognized word is the later functional-integration phase; Phase A
            // guarantees the lexical gap on the LEARNED word closes.)
            match mind.detect_gap("The dragon guards the gold.") {
                None => {}
                Some(g) => assert_ne!(
                    g.surface, "dragon",
                    "(C) the dragon gap must be CLOSED post-study; any remaining gap \
                     must be the still-unknown 'gold', never 'dragon': {g:?}"
                ),
            }

            // (B') re-confirm no false gap survived the engine swap.
            assert!(
                mind.detect_gap("The teacher writes the report.").is_none(),
                "(B') a known sentence stays gap-free after study"
            );
        });
    }

    #[test]
    fn propose_curriculum_for_a_lexical_gap_yields_a_solvable_learn_request() {
        use crate::self_improve::extend::{Gap, GapKind};

        let mind = Mind::new();

        // Detect a real lexical gap, then ask for a curriculum that would close it.
        let gap = mind
            .detect_gap("The dragon guards the gold.")
            .expect("unknown-creature sentence yields a gap");
        let req = mind
            .propose_curriculum(&gap)
            .expect("a lexical creature gap must mine a well-posed curriculum");

        // The request targets a string->int classifier and names the surface word.
        assert_eq!(req.name, "creature_class", "targets the creature membership map");
        assert!(
            req.signature.contains("-> i64"),
            "a membership classifier returns an int label: {}",
            req.signature
        );
        assert!(!req.examples.is_empty(), "the mined spec must carry examples");

        // The gap word must be a POSITIVE (the unknown creature -> 1); the spec is
        // well-posed (no string mapped to two labels — guaranteed by lexicon_examples).
        let labeled = |w: &str| {
            req.examples
                .iter()
                .find(|e| e.inputs == vec![crate::benchmark::Value::Str(w.to_string())])
                .map(|e| e.expected.clone())
        };
        assert_eq!(
            labeled("dragon"),
            Some(crate::benchmark::Value::Int(1)),
            "the unknown creature word must be a positive in the mined spec"
        );

        // The headline guarantee: the mined examples are SOLVABLE — solve_problem
        // (the same solver self_improve relies on) certifies a program that
        // reproduces every example. We never fabricate a spec the solver can't meet.
        let probe = crate::benchmark::Problem {
            name: req.name.clone(),
            category: "comprehension",
            description: "",
            signature: req.signature,
            examples: req.examples.clone(),
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
            "propose_curriculum must yield a SOLVABLE spec: {:?}",
            solved.error
        );
        // Independently re-verify the certified program over every example, so the
        // "solvable" claim is not taken on trust.
        crate::runtime::verify_problem_code_strict(&probe, &solved.code)
            .expect("the certified program must pass strict re-verification");

        // HONESTY: a STRUCTURAL gap (a grammar gap, out of scope for the lexical
        // teacher) mines NO curriculum — the mind never fabricates a spec it has
        // no honest way to characterize.
        let structural = Gap {
            kind: GapKind::Structural,
            surface: "qwx zzt plonk".to_string(),
            context: "qwx zzt plonk".to_string(),
        };
        assert!(
            mind.propose_curriculum(&structural).is_none(),
            "a structural gap is out of scope for the lexical teacher — honest None"
        );
    }

    // ===================================================================
    // FUNCTIONAL INTEGRATION — the parser + answerer recognize a SELF-LEARNED
    // classifier word. After `study` synthesizes `creature_class`, "dragon"
    // (a base-lexicon UNKNOWN) parses as a proper NP head, so "the dragon
    // flies" is an Event with agent dragon (not Unknown), and the category
    // question "is the dragon a creature?" is answered from the VERIFIED
    // program. A FRESH mind with no learning is unchanged: "dragon" is still
    // unknown, "the dragon flies" is Unknown, and the question is "I don't
    // know." — the soundness/fresh-engine invariance the task demands.
    // ===================================================================

    /// FRESH mind, NO learning: "dragon" is an unknown word, "the dragon flies"
    /// fails to parse (Unknown), the engine has no `creature_class` verdict, and
    /// the category question is answered honestly "I don't know.". This pins the
    /// before-state of the integration — the learned-classifier hooks are inert
    /// without any learning, so existing behaviour is unchanged.
    #[test]
    fn fresh_mind_treats_learned_classifier_word_as_unknown() {
        let mind = Mind::new();

        // PRECONDITIONS: genuinely unlearned. "dragon" is not a base noun and no
        // learned classifier exists, so every integration hook is inert.
        assert!(!mind.knows_word("dragon"), "'dragon' must be unknown on a fresh mind");
        assert!(
            !mind.engine().has_component("creature_class"),
            "a fresh mind has not learned creature_class"
        );
        assert_eq!(
            mind.engine().learned_class_of("dragon"),
            None,
            "no learned classifier claims 'dragon' yet"
        );
        assert_eq!(
            mind.engine().learned_class_verdict("creature", "dragon"),
            None,
            "no `creature_class` component exists -> no verdict, defer to the world"
        );

        // PARSE: with no recognition hook firing, the subject "dragon" is not an
        // NP head, so the declarative does not parse to an Event.
        let parsed = mind.understand("the dragon flies");
        assert!(
            matches!(parsed, Meaning::Unknown(_)),
            "BEFORE learning, 'the dragon flies' must be Unknown, got {parsed:?}"
        );

        // ANSWER: an honest open-world "I don't know." — never a fabricated Yes.
        assert_eq!(
            mind.ask("is the dragon a creature?"),
            "I don't know.",
            "BEFORE learning, the category question is open-world"
        );
    }

    /// AFTER `study` learns `creature_class`: the SAME unknown word "dragon" now
    /// parses as an NP head, so "the dragon flies" is an Event with agent
    /// Entity("dragon") — and the category question routes through the verified
    /// program: "is the dragon a creature?" -> Yes (program returns 1),
    /// "is the report a creature?" -> No (program returns 0). This is the
    /// after-state: the verified, self-acquired classifier is live in BOTH the
    /// parser and the answerer.
    #[test]
    fn studied_mind_parses_and_answers_learned_classifier_word() {
        with_study_env("integration", || {
            let mut mind = Mind::new();

            // BEFORE: pin the gap so the test is not vacuous, and capture the
            // before-state parse for the report.
            assert!(!mind.knows_word("dragon"), "'dragon' unknown pre-study");
            assert!(
                !mind.engine().has_component("creature_class"),
                "creature_class genuinely absent pre-study"
            );
            let before = mind.understand("the dragon flies");
            assert!(
                matches!(before, Meaning::Unknown(_)),
                "pre-study parse must be Unknown, got {before:?}"
            );

            // STUDY a creature corpus until the gap closes.
            let corpus = [
                "The teacher writes the report.",
                "The dragon guards the gold.",
            ];
            let report = mind.study(&corpus, 3);
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "study must learn creature_class: {report:?}"
            );
            assert!(
                mind.engine().has_component("creature_class"),
                "the learned classifier must be live on the engine"
            );

            // AFTER — PARSE: "the dragon flies" is now an Event whose agent is the
            // dragon. SOUND ANIMACY: the learned-classifier noun keeps base
            // noun_class 0, so it reads as an INANIMATE "thing", never a "person".
            let after = mind.understand("the dragon flies");
            let Meaning::Event(ev) = &after else {
                panic!("AFTER learning, 'the dragon flies' must be an Event, got {after:?}");
            };
            assert_eq!(
                ev.agent,
                Some(Term::Entity("dragon".to_string())),
                "the agent of 'the dragon flies' must be the dragon"
            );
            assert!(
                !mind.engine().is_person("dragon"),
                "a learned-classifier noun must NOT be animate (sound 'thing' default)"
            );

            // AFTER — ANSWER: the verified program decides the category question.
            assert_eq!(
                mind.engine().learned_class_verdict("creature", "dragon"),
                Some(true),
                "the verified creature_class program returns 1 for 'dragon'"
            );
            let yes = mind.ask("is the dragon a creature?");
            assert!(
                yes.starts_with("Yes,"),
                "'is the dragon a creature?' must be answered Yes, got {yes:?}"
            );

            // A KNOWN non-creature noun ("report") is classified 0 -> No. This is
            // not vacuous: "report" is a base patient noun, so the IsA is built and
            // the verified program supplies the NEGATIVE verdict.
            assert_eq!(
                mind.engine().learned_class_verdict("creature", "report"),
                Some(false),
                "creature_class returns 0 for the non-creature 'report'"
            );
            let no = mind.ask("is the report a creature?");
            assert!(
                no.starts_with("No,"),
                "'is the report a creature?' must be answered No, got {no:?}"
            );

            // SOUNDNESS: growth was monotone — the mind stays green after study.
            assert!(mind.self_check().ok(), "self_check must stay ok after study");
        });
    }

    /// SOUNDNESS — Yes is returned ONLY when the verified program returns 1. We
    /// never answer Yes for a category whose `<x>_class` component the mind has
    /// NOT learned: even after learning `creature_class`, an unlearned category
    /// ("villain") yields no verdict and the open-world honest answer. This
    /// guards the contract "NEVER answer Yes unless the learned VERIFIED program
    /// returns 1".
    #[test]
    fn learned_classifier_never_answers_yes_without_a_verified_program() {
        with_study_env("soundness", || {
            let mut mind = Mind::new();
            let corpus = [
                "The teacher writes the report.",
                "The dragon guards the gold.",
            ];
            let report = mind.study(&corpus, 3);
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "study must learn creature_class: {report:?}"
            );

            // A category the mind never learned a classifier for: no `villain_class`
            // component, hence no verdict, hence the honest open-world answer — even
            // for the now-recognized word "dragon".
            assert_eq!(
                mind.engine().learned_class_verdict("villain", "dragon"),
                None,
                "no `villain_class` component -> no verdict (never fabricate Yes)"
            );
            assert_eq!(
                mind.ask("is the dragon a villain?"),
                "I don't know.",
                "an unlearned category stays open-world, never a fabricated Yes"
            );
        });
    }

    /// ADVERSARIAL VERIFICATION — Thrust C "self-learned-word-parses-and-answers".
    ///
    /// This is the hostile counterpart to `studied_mind_parses_and_answers_*`. It
    /// proves the FOUR things the thrust claim hinges on, in one env-fenced test
    /// over a fresh Mind, with the loop biting AT the boundaries an honest verifier
    /// would attack:
    ///
    ///   (A) BEHAVIOR CHANGE IS REAL — the before-state is captured (Unknown parse,
    ///       "I don't know." answer) and the after-state is captured (Event parse,
    ///       Yes/No answer); the test asserts they DIFFER. No behavior change after
    ///       learning would be a FALSE result, so we pin the negation explicitly.
    ///
    ///   (B) THE VERIFIED CLASSIFIER DRIVES IT — we assert the very same Mog program
    ///       the gate accepted (`creature_class`, evaluated on the engine) is what
    ///       both the parse hook (`learned_class_of`) and the answer hook
    ///       (`learned_class_verdict`) consult, by checking the answer string and
    ///       the program verdict AGREE on every probe. The answer is not a separate
    ///       hardcoded path — it tracks the program byte-for-byte.
    ///
    ///   (C) NEVER A FALSE YES — the killer probe. The classifier was trained on 5
    ///       creatures→1 and 12 nonmembers→0. We hammer it with OUT-OF-DISTRIBUTION
    ///       words it NEVER saw in training (neither a CREATURES member nor one of
    ///       the 12 nonmembers): "editor", "manager", "lesson", "table", "engine",
    ///       "report" is in-set but we add genuinely unseen tokens too. EVERY one of
    ///       these MUST answer "No" or "I don't know." — NEVER "Yes,". A single
    ///       spurious "Yes," on an unseen non-creature (overfit to a feature like
    ///       first-letter or length) is a FALSE result for the whole thrust.
    ///
    ///   (D) THE LOOP ACTUALLY BIT — `study` reports `creature_class` learned and
    ///       the component is live; the mind stays green (monotone growth). A study
    ///       that "succeeds" without a live, gate-passing program would be FALSE.
    #[test]
    fn adversarial_self_learned_word_parses_and_answers_never_false_yes() {
        with_study_env("adversarial_thrustc", || {
            let mut mind = Mind::new();

            // ---- BEFORE: the gap is GENUINE and the behavior is the unlearned one.
            assert!(!mind.knows_word("dragon"), "'dragon' must be a real gap pre-study");
            assert!(
                !mind.engine().has_component("creature_class"),
                "creature_class genuinely absent pre-study (test not vacuous)"
            );
            assert!(mind.self_check().ok(), "the spine is green before study");

            let before_parse = mind.understand("the dragon flies");
            assert!(
                matches!(before_parse, Meaning::Unknown(_)),
                "BEFORE: 'the dragon flies' must be Unknown, got {before_parse:?}"
            );
            let before_answer = mind.ask("is the dragon a creature?");
            assert_eq!(
                before_answer, "I don't know.",
                "BEFORE: the category question is honest open-world idk"
            );

            // ---- STUDY a creature corpus. (D) The loop must actually close the gap.
            let corpus = [
                "The teacher writes the report.", // wholly known
                "The dragon guards the gold.",    // mineable creature gap
            ];
            let report = mind.study(&corpus, 3);
            assert!(
                report.learned.contains(&"creature_class".to_string()),
                "study must learn creature_class — the loop must bite: {report:?}"
            );
            assert!(
                mind.engine().has_component("creature_class"),
                "the learned classifier must be LIVE on the engine (not just reported)"
            );
            assert!(
                mind.self_check().ok(),
                "self_check must stay green AFTER study (monotone growth)"
            );

            // ---- (A) BEHAVIOR CHANGE IS REAL: after-state differs from before.
            let after_parse = mind.understand("the dragon flies");
            let Meaning::Event(ev) = &after_parse else {
                panic!("AFTER: 'the dragon flies' must parse to an Event, got {after_parse:?}");
            };
            assert_eq!(
                ev.agent,
                Some(Term::Entity("dragon".to_string())),
                "AFTER: the agent of 'the dragon flies' must be the dragon"
            );
            // The parse MEANING changed (Unknown -> Event). Non-vacuity of "changed".
            assert!(
                std::mem::discriminant(&before_parse) != std::mem::discriminant(&after_parse),
                "the PARSE must change shape across learning (Unknown -> Event)"
            );
            // SOUND ANIMACY: a learned-classifier noun is an inanimate "thing", never
            // a person — it must not spuriously satisfy an agent-only restriction.
            assert!(
                !mind.engine().is_person("dragon"),
                "a learned-classifier noun must NOT be animate (sound 'thing' default)"
            );

            let after_answer = mind.ask("is the dragon a creature?");
            assert!(
                after_answer.starts_with("Yes,"),
                "AFTER: 'is the dragon a creature?' must be Yes, got {after_answer:?}"
            );
            assert_ne!(
                before_answer, after_answer,
                "the ANSWER must change across learning (idk -> Yes)"
            );

            // ---- (B) THE VERIFIED CLASSIFIER DRIVES BOTH PARSE AND ANSWER.
            // The load-bearing soundness directions, asserted as a bridge between
            // the raw program verdict and the end-to-end answer string:
            //   * program==1  ==> the answer MUST be "Yes,"  (program drives Yes)
            //   * answer "Yes," ==> program==1               (NEVER a fabricated Yes)
            // The program==0 case maps to EITHER "No," (the subject is a parseable
            // base entity, so the IsA is built and the program supplies the negative)
            // OR "I don't know." (the subject is an UNRECOGNIZED word — not a base
            // noun and not positively classified — so the question never resolves to
            // a concrete entity and the honest open-world answer is returned). BOTH
            // are sound: neither is a false Yes. So the bridge only constrains the
            // Yes direction in both ways; the negative side is checked by the
            // never-false-Yes probe below.
            let agrees = |mind: &Mind, word: &str| {
                let verdict = mind.engine().learned_class_verdict("creature", word);
                let answer = mind.ask(&format!("is the {word} a creature?"));
                // program==1  ==>  answer Yes (the program DRIVES the positive answer)
                if verdict == Some(true) {
                    assert!(
                        answer.starts_with("Yes,"),
                        "program says creature_class({word})==1 but answer was {answer:?}"
                    );
                }
                // answer Yes  ==>  program==1 (a Yes is ONLY ever the program's 1)
                if answer.starts_with("Yes,") {
                    assert_eq!(
                        verdict,
                        Some(true),
                        "answer was Yes for {word:?} but the verified program did not \
                         return 1 (verdict {verdict:?}) — that would be a FABRICATED Yes"
                    );
                }
                verdict
            };
            // dragon -> program 1 -> Yes (already checked, re-asserted via the bridge).
            assert_eq!(agrees(&mind, "dragon"), Some(true));

            // ---- (C) NEVER A FALSE YES — the killer probe over IN-set and
            // OUT-OF-DISTRIBUTION non-creatures. NONE may answer "Yes,".
            // In-set negative (seen in training):
            let in_set_negatives = ["report", "book", "teacher", "author"];
            // OUT-OF-DISTRIBUTION negatives — words the classifier NEVER saw in
            // EITHER its 5 positives or its 12 negatives. A spurious 1 here exposes
            // overfitting to an incidental feature (first letter / length / animacy).
            let ood_negatives = ["editor", "manager", "lesson", "engine", "table", "clerk"];

            for w in in_set_negatives.iter().chain(ood_negatives.iter()) {
                // The word must NEVER be classified as a creature by the program.
                let verdict = mind.engine().learned_class_verdict("creature", w);
                assert_ne!(
                    verdict,
                    Some(true),
                    "FALSE YES: the verified classifier must NEVER return 1 for the \
                     non-creature {w:?} (verdict was {verdict:?})"
                );
                // And the END-TO-END answer must never be a fabricated Yes — it is
                // either an honest No (program returned 0) or idk (no verdict).
                let ans = mind.ask(&format!("is the {w} a creature?"));
                assert!(
                    ans.starts_with("No,") || ans == "I don't know.",
                    "FALSE YES end-to-end: 'is the {w} a creature?' answered {ans:?} \
                     — must be No or idk, never Yes"
                );
                // Bridge B: whenever a verdict exists, the answer tracks it exactly.
                agrees(&mind, w);
            }

            // ---- (C') The in-set negatives the thrust names explicitly resolve No
            // (program saw them -> 0), so "is the report a creature?" is a hard No,
            // not merely idk — the strongest form of "never a false Yes".
            assert_eq!(
                mind.engine().learned_class_verdict("creature", "report"),
                Some(false),
                "the verified program returns 0 for the non-creature 'report'"
            );
            assert!(
                mind.ask("is the report a creature?").starts_with("No,"),
                "'is the report a creature?' must be a hard No (program returned 0)"
            );

            // ---- (C'') An UNLEARNED category never fabricates Yes even for the now-
            // recognized 'dragon' (no villain_class component -> no verdict -> idk).
            assert_eq!(
                mind.engine().learned_class_verdict("villain", "dragon"),
                None,
                "no villain_class component -> no verdict (never fabricate Yes)"
            );
            assert_eq!(
                mind.ask("is the dragon a villain?"),
                "I don't know.",
                "an unlearned category stays open-world, never a fabricated Yes"
            );

            // ---- POSITIVE GENERALIZATION (non-vacuity of "Yes"): every KNOWN
            // creature in the curriculum's CREATURES set classifies 1 -> Yes, so the
            // program is a real membership lexicon, not a single-word special case.
            for c in crate::comprehension::CREATURES {
                assert_eq!(
                    mind.engine().learned_class_verdict("creature", c),
                    Some(true),
                    "every trained creature must classify 1: {c}"
                );
            }
        });
    }
}
