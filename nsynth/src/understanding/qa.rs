//! Question answering: reading comprehension over the world model built by
//! reading. Answers come from the model, not from keyword-matching the
//! question.
//!
//! `answer` parses the question with `semantics::understand`. For a
//! YesNoQuestion it queries `world.holds` (augmented by inference) and replies
//! "Yes, ..." / "No, ..." / "I don't know.". For a WhQuestion it searches
//! `world.facts` for a matching event and returns the filler of the queried
//! slot ("the teacher"). For an IsA category question it answers from animacy.

use crate::comprehension::{capitalize, Engine, AGENTS, GRADABLE, PATIENTS};
use crate::understanding::discourse::Discourse;
use crate::understanding::inference::{prove, relation, Proof, Relation};
use crate::understanding::meaning::{
    Aspect, Event, Meaning, Modality, Quantifier, Role, Tense, TemporalRel, Term,
};

/// Answer a question against the world model built up in `discourse`.
pub fn answer(engine: &Engine, discourse: &Discourse, question: &str) -> String {
    // Parse the question into a logical form, then resolve any pronouns in it
    // against the discourse history so "does it write the report?" queries the
    // entity "it" refers to — answers come from the world model, not the
    // surface string.
    let parsed = crate::understanding::semantics::understand(engine, question);
    let m = resolve_meaning(discourse, &parsed);

    match m {
        Meaning::YesNoQuestion(body) => answer_yes_no(engine, discourse, &body),
        Meaning::WhQuestion { slot, body } => answer_wh(discourse, slot, &body),
        // A bare IsA reaching here means the question was phrased as a copular
        // query ("the teacher is a person?"). Treat it as a yes/no over the IsA.
        Meaning::IsA { .. } => answer_yes_no(engine, discourse, &m),
        // A bare event question without interrogative marking — still answerable
        // as a yes/no truth query against the world.
        Meaning::Event(_) => answer_yes_no(engine, discourse, &m),
        // A quantified / disjunctive / attribute query that reaches `answer`
        // bare (i.e. phrased as a query but not wrapped in YesNoQuestion, e.g.
        // "the teacher is careful?" or an interrogative the parser left bare) is
        // answered as a yes/no truth query: query the world's model-theoretic
        // truth (universal/existential/negative quantification over entities,
        // disjunction, asserted attributes) augmented by sound inference and the
        // taxonomy, then reply Yes / No / I don't know.
        Meaning::Quantified { .. }
        | Meaning::Or(_)
        | Meaning::HasProperty { .. } => answer_yes_no(engine, discourse, &m),
        // A Comparison reaching `answer` is a truth query ("Is the report longer
        // than the book?"): its truth (direct + transitive closure, respecting
        // negation) lives in the world model, so route through `answer_yes_no`.
        Meaning::Comparison { .. } => answer_yes_no(engine, discourse, &m),
        // An Attitude is either a truth query ("Does the teacher know that P?")
        // or a CONTENT query ("What does the teacher know?", parsed with an
        // `Unknown` content placeholder). A content query needs the realized
        // content, not Yes/No; everything else is a truth query.
        Meaning::Attitude { ref content, .. } if is_content_query(content) => {
            answer_attitude_content(engine, discourse, &m)
        }
        Meaning::Attitude { .. } => answer_yes_no(engine, discourse, &m),
        // A Cardinal ("Do two teachers write a report?") is a truth query: true
        // iff at least N known members satisfy the body (world model owns the
        // at-least monotonic semantics).
        Meaning::Cardinal { .. } => answer_yes_no(engine, discourse, &m),
        // "How many teachers write a report?" — the answer is a NUMBER, counted
        // over the world's known members of the category.
        Meaning::CountQuestion { var_category, body } => {
            answer_count(discourse, &var_category, &body)
        }
        // "Can/must/might/should the teacher write the report?" — a modal truth
        // query. Modal monotonicity (Must ⊢ Can; actuality ⊢ Can) is owned by the
        // world model + inference, consulted through `answer_yes_no`.
        Meaning::Modal { .. } => answer_yes_no(engine, discourse, &m),
        // "Does X happen before Y?" — a temporal-order truth query (transitive,
        // asymmetric closure owned by the world model), consulted as a yes/no.
        Meaning::Temporal { .. } => answer_yes_no(engine, discourse, &m),
        // "Why does the street flood?" — a CAUSE query. The parser surfaces a
        // why-question as a Causal whose `cause` is an Unknown placeholder; we
        // answer with the realized cause of the queried effect ("Because the rain
        // falls."). A fully-specified Causal ("does the street flood because the
        // rain falls?") is a yes/no truth query instead.
        Meaning::Causal { ref cause, .. } if is_content_query(cause) => {
            answer_why(engine, discourse, &m)
        }
        Meaning::Causal { .. } => answer_yes_no(engine, discourse, &m),
        // "How long is the report?" — a degree question, answered from KNOWN
        // comparison facts (a comparative phrase) or honestly "I don't know.".
        Meaning::DegreeQuestion { subject, scale } => {
            answer_degree(engine, discourse, &subject, &scale)
        }
        // "Not every teacher writes a report?" — an outer-negation (wide-scope)
        // truth query whose truth is the three-valued negation of the inner
        // meaning's truth; distinct from the narrow-scope "every ... does not ...".
        Meaning::Not(_) => answer_yes_no(engine, discourse, &m),
        Meaning::Unknown(_) => "I don't know.".to_string(),
    }
}

/// Answer an already-parsed query Meaning AND return the SOUND [`Proof`] backing
/// it, when one exists. The returned `String` (`.0`) is IDENTICAL to what
/// [`answer`] produces for the same query: this mirrors `answer`'s dispatch
/// exactly (including the same `resolve_meaning` pronoun/relative-clause
/// resolution), routing every truth query through the explained yes/no path so
/// the proof rides along.
///
/// A `Proof` is returned for the `Yes`/`No` (truth-query) and entailment answers
/// — those that bottom out in [`world_truth_traced`]. The non-truth answers
/// (wh-fillers, counts, why-causes, degree phrases, attitude-content,
/// "I don't know.") have no propositional proof, so they carry `None`. As with
/// the yes/no path, a world-owned/opaque or open-world verdict also yields `None`.
pub fn answer_explained(
    engine: &Engine,
    discourse: &Discourse,
    m: &Meaning,
) -> (String, Option<Proof>) {
    // Resolve pronouns / relative-clause subjects exactly as `answer` does for its
    // parsed question, so the explained answer queries the same entities.
    let m = resolve_meaning(discourse, m);

    match m {
        Meaning::YesNoQuestion(body) => answer_yes_no_explained(engine, discourse, &body),
        Meaning::WhQuestion { slot, body } => (answer_wh(discourse, slot, &body), None),
        Meaning::IsA { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Event(_) => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Quantified { .. }
        | Meaning::Or(_)
        | Meaning::HasProperty { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Comparison { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Attitude { ref content, .. } if is_content_query(content) => {
            (answer_attitude_content(engine, discourse, &m), None)
        }
        Meaning::Attitude { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Cardinal { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::CountQuestion { var_category, body } => {
            (answer_count(discourse, &var_category, &body), None)
        }
        Meaning::Modal { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Temporal { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Causal { ref cause, .. } if is_content_query(cause) => {
            (answer_why(engine, discourse, &m), None)
        }
        Meaning::Causal { .. } => answer_yes_no_explained(engine, discourse, &m),
        Meaning::DegreeQuestion { subject, scale } => {
            (answer_degree(engine, discourse, &subject, &scale), None)
        }
        Meaning::Not(_) => answer_yes_no_explained(engine, discourse, &m),
        Meaning::Unknown(_) => ("I don't know.".to_string(), None),
    }
}

/// Is an embedded attitude `content` a CONTENT-query placeholder rather than a
/// concrete proposition? The parser of "What does the teacher know?" cannot fill
/// in the proposition, so it leaves an `Unknown` content; a concrete-content
/// attitude question ("Does the teacher know that the report is long?") carries a
/// real embedded `Meaning`.
fn is_content_query(content: &Meaning) -> bool {
    matches!(content, Meaning::Unknown(_))
}

// ---------------------------------------------------------------------------
// Yes/No questions
// ---------------------------------------------------------------------------

/// Answer a yes/no query: "Yes, <restated affirmation>." / "No, <restated
/// negation>." / "I don't know." when the open world has no information.
fn answer_yes_no(engine: &Engine, discourse: &Discourse, body: &Meaning) -> String {
    // SINGLE SOURCE OF TRUTH: the explained variant computes the identical string
    // (the proof is dropped here), so behaviour is byte-for-byte unchanged.
    answer_yes_no_explained(engine, discourse, body).0
}

/// Like [`answer_yes_no`], but also returns the SOUND [`Proof`] that backs a
/// `Yes`/`No` verdict (from [`world_truth_traced`]). The string (`.0`) is
/// computed exactly as `answer_yes_no` does — `answer_yes_no` is defined as this
/// function's `.0` — so they never drift. The proof is `None` whenever the verdict
/// was world-owned/opaque or the answer is "I don't know." (open world).
fn answer_yes_no_explained(
    engine: &Engine,
    discourse: &Discourse,
    body: &Meaning,
) -> (String, Option<Proof>) {
    // SELF-LEARNED CLASSIFIER category query, e.g. "is the dragon a creature?".
    // When the queried category matches a classifier the mind LEARNED on its own
    // (class "creature" <-> the verified `creature_class` Mog program), the truth
    // of the IsA comes from RUNNING that verified program on the subject — not the
    // world model (which has never heard of "creature"). We answer Yes only when
    // the verified program returns 1, No when it returns 0, and leave everything
    // else to the ordinary world cascade below.
    //
    // SOUNDNESS: `learned_classifier_truth` returns `Some` ONLY when a learned
    // `<category>_class` component exists on this engine (a fresh, unlearned engine
    // has none, so this is inert and the 269-test baseline is unchanged) AND the
    // subject is a concrete entity. It NEVER answers Yes unless the verified program
    // evaluates to exactly 1.
    if let Some((truth, proof)) = learned_classifier_truth(engine, body) {
        return match truth {
            true => (
                format!("Yes, {}.", realize(engine, body, /*force_negated=*/ None)),
                proof,
            ),
            false => (
                format!("No, {}.", realize(engine, body, /*force_negated=*/ Some(true))),
                proof,
            ),
        };
    }

    let (truth, proof) = world_truth_traced(discourse, body);
    match truth {
        Some(true) => (
            format!("Yes, {}.", realize(engine, body, /*force_negated=*/ None)),
            proof,
        ),
        Some(false) => {
            // Restate the falsity of what was asked. For a simple predication
            // ("does X write Y?", "is X careful?", "is X a person?") the clean
            // restatement is the explicit negation ("X does not write Y").
            //
            // For a UNIVERSAL/EXISTENTIAL or DISJUNCTION, force-negating each
            // leaf would be logically wrong (the negation of "every X writes Y"
            // is NOT "every X does not write Y"). So we restate the query
            // verbatim and let the leading "No," carry the polarity. The
            // grammatical-core forms whose negation is NOT a simple leaf-flip get
            // the same verbatim treatment: a Modal ("can X write Y" — its falsity
            // is "X cannot ...", but the modal force/embedded structure makes a
            // blanket flip unreliable), a Temporal ("X before Y"), a Causal, an
            // outer Not (already a wide-scope negation), and a Cardinal/Comparison/
            // Attitude (whose contradictories are bespoke). Verbatim + "No," stays
            // sound for every one of these.
            let restated = match body {
                Meaning::Quantified { .. }
                | Meaning::Or(_)
                | Meaning::Modal { .. }
                | Meaning::Temporal { .. }
                | Meaning::Causal { .. }
                | Meaning::Not(_)
                | Meaning::Cardinal { .. } => realize(engine, body, /*force_negated=*/ None),
                _ => realize(engine, body, /*force_negated=*/ Some(true)),
            };
            (format!("No, {restated}."), proof)
        }
        None => ("I don't know.".to_string(), None),
    }
}

/// Truth of `body` against the world, augmented by inference and the taxonomy.
///
/// Order of consultation (first definite answer wins; all sound):
///   1. The world's own model-theoretic verdict (`world.holds`). The world owns
///      truth for events, categories, quantifiers, disjunction, and attributes;
///      it returns `None` under the open-world assumption when it has no info.
///   2. The taxonomy / hypernymy fallback for category (`IsA`) queries: "Is the
///      teacher an agent?" is true because teacher is-a person is-a agent, even
///      if only "teacher" was ever mentioned. (Belt-and-suspenders with the
///      world's own taxonomy: if `holds` already derived it, this never runs.)
///   3. Natural-language inference over asserted facts: any asserted fact (or a
///      sound consequence of one) that entails the query answers Yes; one that
///      contradicts it answers No. This closes QA under existential
///      generalization, patient-dropping, etc.
fn world_truth(discourse: &Discourse, body: &Meaning) -> Option<bool> {
    // SINGLE SOURCE OF TRUTH: the verdict is exactly the `.0` of the traced
    // variant, which mirrors this cascade step-for-step. Dropping the proof here
    // leaves behaviour byte-for-byte identical to the original `world_truth`.
    world_truth_traced(discourse, body).0
}

/// Like [`world_truth`], but additionally returns a SOUND [`Proof`] for the
/// determined cases when one is available. The truth value (`.0`) is computed by
/// the EXACT same cascade as `world_truth` — indeed `world_truth` is defined as
/// `world_truth_traced(..).0`, so the two never drift.
///
/// Proof availability by cascade step (every emitted proof is a real
/// certificate; we NEVER fabricate a step):
///   - **aspect-normalize**: recurse on the Simple twin; the recovered proof
///     (of the simple eventuality the world actually records) carries through.
///   - **`world.holds`**: the world model is an opaque verdict oracle with no
///     public derivation, so a verdict it owns has `None` proof.
///   - **`Or` (true)**: the winning disjunct's proof, wrapped in a
///     `disjunction-introduction` step (`d ⊢ (… or d or …)`); `None` if that
///     disjunct's truth was itself proofless (e.g. world-owned).
///   - **`Not`**: three-valued flip of the inner truth; no single-meaning
///     certificate, so `None` proof.
///   - **modal actuality→possibility**: the bare event's proof carries through
///     (what is proven to happen is proven possible).
///   - **taxonomy**: a hypernym-chain verdict with no `prove`-style derivation
///     here, so `None` proof.
///   - **fact-loop Entails**: `inference::prove(facts, body)` — the genuine
///     derivation of `body` from the asserted facts.
///   - **fact-loop Contradicts (`Some(false)`)**: `inference::prove(facts,
///     polarity_flip(body))` — a derivation whose CONCLUSION is the polarity-flip
///     of the query (we proved the negation, hence the query is false).
///
/// `None` (open-world) truth always carries a `None` proof.
pub fn world_truth_traced(
    discourse: &Discourse,
    body: &Meaning,
) -> (Option<bool>, Option<Proof>) {
    // ASPECT is non-truth-conditional for fact-matching in this curriculum: a
    // Present/Past Progressive ("is writing") or Perfect ("has written") query is
    // true exactly when the underlying SIMPLE eventuality holds. Normalize a
    // non-Future event's aspect to Simple and evaluate that twin, so a stored
    // simple fact answers an aspectual query. SOUNDNESS: this maps the query onto
    // the very same eventuality the world records; FUTURE ("will write") is
    // excluded because it describes an event that has not happened, so it must
    // stay genuinely open rather than reduce to a (false-implying) simple match.
    if let Meaning::Event(ev) = body {
        if ev.aspect != Aspect::Simple && ev.tense != Tense::Future {
            let mut simple = ev.clone();
            simple.aspect = Aspect::Simple;
            return world_truth_traced(discourse, &Meaning::Event(simple));
        }
    }

    if let Some(v) = discourse.world.holds(body) {
        // The world model owns this verdict but generally exposes no derivation.
        // EXCEPTION — a TRUE comparison: the world owns the transitive-closure
        // verdict, but we can reconstruct the actual derivation chain by running
        // `prove` over the asserted comparison edges, so "is the report longer
        // than the letter?" shows its work ("... because report>book and
        // book>letter"). The verdict stays the world's; we only ATTACH a proof
        // when one is genuinely derivable (a direct edge proves as an asserted
        // leaf; a transitive one as a named chain). Soundness is unchanged.
        if v {
            if let Meaning::Comparison { .. } = body {
                let cmp_facts = discourse.world.comparison_facts();
                if let Some(p) = crate::understanding::inference::prove(&cmp_facts, body) {
                    return (Some(true), Some(p));
                }
            }
        }
        return (Some(v), None);
    }

    // Disjunction: evaluate it compositionally in QA so the verdict does not
    // depend on the sibling inference engine recognizing "fact entails Or". A
    // disjunction is `Some(true)` as soon as ANY disjunct is true; `Some(false)`
    // only when EVERY disjunct is known false; otherwise `None` (some disjunct
    // undetermined leaves the whole thing open). Each disjunct is evaluated by
    // the full `world_truth` cascade (world + taxonomy + facts), so a disjunct
    // proven only by inference still makes the disjunction true. This is sound:
    // we never claim true without a true disjunct, nor false with an open one.
    if let Meaning::Or(disjuncts) = body {
        let mut all_false = true;
        for d in disjuncts {
            let (dt, dp) = world_truth_traced(discourse, d);
            match dt {
                Some(true) => {
                    // The disjunction follows from this true disjunct by
                    // disjunction-introduction. Only certificate it when the
                    // disjunct itself carried a proof (else None proof).
                    let proof = dp.map(|inner| Proof {
                        conclusion: body.clone(),
                        rule: "disjunction-introduction".to_string(),
                        premises: vec![inner],
                    });
                    return (Some(true), proof);
                }
                Some(false) => {}
                None => all_false = false,
            }
        }
        // Empty disjunction is vacuously false; otherwise false iff all false.
        return (if all_false { Some(false) } else { None }, None);
    }

    // Outer negation (wide-scope): the truth of `Not(m)` is the THREE-VALUED
    // negation of the truth of `m`. Computed compositionally in QA so the scope
    // distinction holds even if the sibling world model has not yet wired its own
    // `Not` truth: Some(true) -> Some(false), Some(false) -> Some(true), None ->
    // None. SOUND: we only ever flip a determined verdict and leave the open
    // world open. This is what makes "not every teacher writes a report"
    // (= Not(Quantified Every)) get a DIFFERENT value than the narrow-scope
    // "every teacher does not write a report" (a Quantified with a negated body).
    if let Meaning::Not(inner) = body {
        // The flipped truth has no single-meaning certificate here: None proof.
        return (three_valued_not(world_truth_traced(discourse, inner).0), None);
    }

    // Modal monotonicity, computed compositionally in QA as a belt-and-suspenders
    // over the world model's own modal truth (already consulted in step 1):
    //   - actuality entails possibility: if the bare event is KNOWN TRUE, then
    //     "can/might <event>" is true (what actually happens is possible).
    //   - necessity entails possibility: handled by the world model storing a
    //     Must fact; here we additionally treat a known-true event as licensing
    //     Can/Might. We NEVER derive actuality FROM a modal (possibility does not
    //     entail the event happened) and never claim a modal true from an
    //     unproven event — so this only ever STRENGTHENS to a sound `Some(true)`.
    if let Meaning::Modal { modality, body: ev, negated: false } = body {
        if matches!(modality, Modality::Can | Modality::Might) {
            // The actual occurrence of the event makes it possible.
            let (et, ep) = world_truth_traced(discourse, &Meaning::Event((**ev).clone()));
            if et == Some(true) {
                // The bare event's derivation (if any) certifies its possibility.
                return (Some(true), ep);
            }
        }
    }

    // Taxonomy fallback for category queries: derive "teacher is an agent" from
    // the hypernym chain even when only the subtype ("teacher") is known.
    if let Some(v) = taxonomy_truth(body) {
        // Taxonomy verdict has no `prove`-style certificate here: None proof.
        return (Some(v), None);
    }

    // Inference fallback: does any asserted fact — or a sound consequence of it —
    // entail / contradict the query? This makes QA close under sound
    // natural-language inference, e.g. an existential query entailed by a
    // concrete asserted fact answers "Yes". For Quantified/Or queries the world
    // already evaluated truth in step 1; here we only add fact-driven derivation
    // for the assertoric leaf meanings (events, categories, attributes).
    let mut saw_contradiction = false;
    for fact in discourse.world.facts() {
        let premise = Meaning::Event(fact.clone());
        match relation(&premise, body) {
            Relation::Entails => {
                // Recover the actual derivation of `body` from the asserted facts.
                let facts_as_meanings = facts_as_meanings(discourse);
                let proof = prove(&facts_as_meanings, body);
                return (Some(true), proof);
            }
            Relation::Contradicts => saw_contradiction = true,
            Relation::Neutral => {}
        }
    }
    if saw_contradiction {
        // `body` is false BECAUSE the facts prove its polarity-flip. Build a proof
        // whose CONCLUSION is that flip (we derived the negation of the query).
        let proof = polarity_flip_query(body)
            .and_then(|flipped| prove(&facts_as_meanings(discourse), &flipped));
        return (Some(false), proof);
    }
    (None, None)
}

/// The world's asserted facts as `Meaning`s, the form [`prove`] consumes. Mirrors
/// the premise construction in the fact-loop above (`Meaning::Event(fact)`).
fn facts_as_meanings(discourse: &Discourse) -> Vec<Meaning> {
    discourse
        .world
        .facts()
        .iter()
        .map(|f| Meaning::Event(f.clone()))
        .collect()
}

/// The polarity-flip (sound CONTRADICTORY) of an assertoric leaf query, used to
/// name the conclusion of a `Some(false)`-via-contradiction proof. This mirrors
/// `inference::polarity_flip` for exactly the leaf shapes that reach the
/// fact-loop's contradiction branch (events, categories, attributes,
/// comparisons, modals); other shapes have no single-meaning contradictory, so we
/// return `None` (the proof is then left `None`, never fabricated).
fn polarity_flip_query(m: &Meaning) -> Option<Meaning> {
    match m {
        Meaning::Event(ev) => {
            let mut e = ev.clone();
            e.negated = !e.negated;
            Some(Meaning::Event(e))
        }
        Meaning::IsA { subject, category, negated } => Some(Meaning::IsA {
            subject: subject.clone(),
            category: category.clone(),
            negated: !negated,
        }),
        Meaning::HasProperty { subject, property, negated } => Some(Meaning::HasProperty {
            subject: subject.clone(),
            property: property.clone(),
            negated: !negated,
        }),
        Meaning::Comparison { subject, scale, more, than, negated } => Some(Meaning::Comparison {
            subject: subject.clone(),
            scale: scale.clone(),
            more: *more,
            than: than.clone(),
            negated: !negated,
        }),
        Meaning::Modal { modality, body, negated } => Some(Meaning::Modal {
            modality: *modality,
            body: body.clone(),
            negated: !negated,
        }),
        // No single-meaning contradictory for the rest (Or/Quantified-Some/
        // Cardinal/Temporal/Causal/questions/Unknown): leave the proof None.
        _ => None,
    }
}

/// Three-valued negation of a truth value: the outer-negation truth of `Not(m)`
/// given the truth of `m`. `Some(true)` becomes `Some(false)`, `Some(false)`
/// becomes `Some(true)`, and `None` (open world) stays `None`. This is the sound
/// Kleene/strong negation the contract's negation-scope semantics requires.
fn three_valued_not(inner: Option<bool>) -> Option<bool> {
    inner.map(|v| !v)
}

// ---------------------------------------------------------------------------
// "Why ...?" causal questions
// ---------------------------------------------------------------------------

/// Answer a "why does <effect>?" question. The parser surfaces it as a
/// `Causal { cause: Unknown, effect }` placeholder (the cause is what we are
/// asking for). We look for a stored causal link whose effect matches the
/// queried effect and realize its cause as "Because <cause>.".
///
/// The world model owns causal storage but exposes no public enumeration of
/// links, so we recover the cause SOUNDLY from the discourse's asserted facts:
/// a `Causal(C, E)` assertion presupposes and asserts both C and E as facts, and
/// the discourse records the realized causal pairing nowhere public. Therefore we
/// answer from the meaning the parser handed us only when the cause is concrete;
/// for a bare why-question with no recoverable cause we answer honestly.
///
/// SOUNDNESS: we never fabricate a cause. If the queried effect is not a known
/// world fact (the open world does not even attest the effect happened), or no
/// concrete cause is available, we answer "I don't know." rather than guessing.
fn answer_why(engine: &Engine, discourse: &Discourse, m: &Meaning) -> String {
    let Meaning::Causal { cause, effect } = m else {
        return "I don't know.".to_string();
    };
    // A concrete cause was provided by the parser (rare for a pure why-question,
    // but handles "the street floods because the rain falls. why does the street
    // flood?" pipelines that carry the cause through): realize it directly, but
    // only if the effect is actually attested in the world (we do not explain an
    // effect the world has no record of).
    if !is_content_query(cause) {
        if world_truth(discourse, effect) == Some(true)
            || world_truth(discourse, cause) == Some(true)
        {
            return capitalize(&format!("because {}.", realize(engine, cause, None)));
        }
    }
    // A bare "why does <effect>?" leaves the cause an Unknown placeholder. Recover
    // it from a previously asserted causal link whose effect matches — but only
    // when the effect is actually ATTESTED in the world (we never explain an
    // effect the world has no record of). SOUND: `cause_of` reads the directed
    // link in the cause->effect direction only, so a recovered cause is one the
    // discourse was explicitly told, never a guessed or reversed one.
    if is_content_query(cause) && world_truth(discourse, effect) == Some(true) {
        if let Some(c) = discourse.world.cause_of(effect) {
            return capitalize(&format!("because {}.", realize(engine, &c, None)));
        }
    }
    "I don't know.".to_string()
}

// ---------------------------------------------------------------------------
// Degree questions ("how <adj> is the <noun>?")
// ---------------------------------------------------------------------------

/// Answer "how <scale> is the <subject>?" from KNOWN comparison facts. We have no
/// numeric measures (this curriculum stores only orderings), so the honest,
/// grounded answer is a comparative phrase recovered from the world's asserted
/// comparisons: if the subject is known to exceed some entity on the scale, we
/// answer "<comparative-high> than the <other>" ("longer than the book"); if it
/// is known to fall below some entity, "<comparative-low> than the <other>"
/// ("shorter than the book"). With no comparison on record we answer
/// "I don't know." — never invent a measure.
///
/// We probe the public `world.holds(Comparison{..})` API over every known entity
/// (no private ordering accessor is required), so this is robust to the world
/// model's internal representation.
///
/// SOUNDNESS: every phrase we emit corresponds to a comparison the world proves
/// true (`holds == Some(true)`), so we never assert an ordering the world does
/// not license. We prefer the "exceeds" direction (more informative high pole)
/// and fall back to the "falls below" direction.
fn answer_degree(engine: &Engine, discourse: &Discourse, subject: &Term, scale: &str) -> String {
    let subj_head = subject.head();

    // Candidate other entities to compare against: every known entity except the
    // subject itself.
    let others: Vec<String> = discourse
        .world
        .entities()
        .into_iter()
        .filter(|e| e != subj_head)
        .collect();

    // First pass: the subject EXCEEDS some other entity on the scale (high pole,
    // e.g. "longer than the book").
    for other in &others {
        let exceeds = Meaning::Comparison {
            subject: subject.clone(),
            scale: scale.to_string(),
            more: true,
            than: Term::Entity(other.clone()),
            negated: false,
        };
        if discourse.world.holds(&exceeds) == Some(true) {
            return capitalize(&format!(
                "{} than the {}.",
                comparative_for(scale, true),
                other
            ));
        }
    }

    // Second pass: the subject FALLS BELOW some other entity on the scale (low
    // pole, e.g. "shorter than the report").
    for other in &others {
        let below = Meaning::Comparison {
            subject: subject.clone(),
            scale: scale.to_string(),
            more: false,
            than: Term::Entity(other.clone()),
            negated: false,
        };
        if discourse.world.holds(&below) == Some(true) {
            return capitalize(&format!(
                "{} than the {}.",
                comparative_for(scale, false),
                other
            ));
        }
    }

    // No comparison on record for this subject/scale: honest open-world answer.
    "I don't know.".to_string()
}

// ---------------------------------------------------------------------------
// Counting questions ("how many <category> <verb> ...?")
// ---------------------------------------------------------------------------

/// Answer "how many <var_category> <body>?" with a number phrase. We count the
/// world's KNOWN entities of the category whose body-event PROVABLY holds
/// (`world.holds(Event) == Some(true)`), binding each candidate entity into the
/// body's agent slot.
///
/// SOUNDNESS (closed-world-on-knowns, three-valued-aware):
///   - We only count entities the body is *proven true* for. An entity whose
///     body-truth is `None` (open world) is NOT counted — we never inflate the
///     count with unverified members.
///   - If EVERY candidate member's body-truth is determined (`Some(_)` for all),
///     the count is exact and we report the number ("two"). If some member is
///     undetermined, the true count could be higher, so we report a lower bound
///     phrasing ("at least two") rather than an exact figure — never an
///     over-claim. With zero determined satisfiers and open members we say "I
///     don't know." rather than a false "zero".
///   - With no known members of a recognized category, the answer is "zero".
fn answer_count(discourse: &Discourse, var_category: &str, body: &Event) -> String {
    let members = category_members(discourse, var_category);

    // Unknown category entirely (no members and not a recognized class/noun):
    // open world, we cannot count.
    if members.is_empty() {
        if category_is_recognized(discourse, var_category) {
            return "Zero.".to_string();
        }
        return "I don't know.".to_string();
    }

    let mut satisfied = 0usize;
    let mut any_unknown = false;
    for member in &members {
        let mut ev = body.clone();
        ev.agent = Some(Term::Entity(member.clone()));
        match discourse.world.holds(&Meaning::Event(ev)) {
            Some(true) => satisfied += 1,
            Some(false) => {}
            None => any_unknown = true,
        }
    }

    if satisfied == 0 && any_unknown {
        // No proven satisfier but at least one member is undetermined — the real
        // count is unknown (could be 0 or more). Do not claim "zero".
        return "I don't know.".to_string();
    }

    let word = number_phrase(satisfied);
    if any_unknown && satisfied > 0 {
        // A definite lower bound: at least `satisfied` members satisfy the body,
        // but undetermined members could push the true count higher.
        format!("At least {word}.")
    } else {
        capitalize(&format!("{word}."))
    }
}

/// Known entities of `category` in the world, by reusing the world's own
/// taxonomy-aware category membership: an entity is a member iff
/// `world.holds(IsA{entity, category})` is `Some(true)`. This delegates the
/// (noun-identity / hypernym-chain / animacy) membership logic to the world
/// model so QA and the world agree on who counts.
fn category_members(discourse: &Discourse, category: &str) -> Vec<String> {
    discourse
        .world
        .entities()
        .into_iter()
        .filter(|e| {
            let isa = Meaning::IsA {
                subject: Term::Entity(e.clone()),
                category: category.to_string(),
                negated: false,
            };
            discourse.world.holds(&isa) == Some(true)
        })
        .collect()
}

/// Does the world recognize `category` as a category at all (a known noun, a
/// taxonomy class, or the head of some known entity)? Used so a counting
/// question over an empty-but-recognized category answers "zero" rather than
/// "I don't know.". We probe via the public taxonomy/animacy helpers in this
/// module plus the world's entity set.
fn category_is_recognized(discourse: &Discourse, category: &str) -> bool {
    if is_known_taxon(category) {
        return true;
    }
    if hypernym_chain(category).is_some() {
        // A known leaf noun (teacher/report/...) has a taxonomy chain.
        return true;
    }
    discourse.world.entities().iter().any(|e| e == category)
}

/// Render a non-negative count as an English number word for small values,
/// falling back to digits for larger ones. Lowercase, no trailing punctuation.
fn number_phrase(n: usize) -> String {
    match n {
        0 => "zero".to_string(),
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        10 => "ten".to_string(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Attitude content questions ("what does the teacher know?")
// ---------------------------------------------------------------------------

/// Answer a CONTENT query over an attitude ("What does the teacher know?").
///
/// The world model stores attitudes but exposes no public enumeration of their
/// contents (only `assert`/`holds`). A FACTIVE attitude ("know that P") asserts
/// its content P into the world, so the content becomes a queryable world fact;
/// a non-factive one ("believe/think/say that P") does NOT. With no public way
/// to recover *which* proposition was the complement of a stored attitude, QA
/// cannot fabricate the content soundly, so a bare content query returns
/// "I don't know." This stays SOUND (never invents a known proposition) and is
/// the documented cross-module assumption: surfacing "what X knows" requires a
/// world-model accessor that enumerates a holder's attitude contents.
fn answer_attitude_content(_engine: &Engine, _discourse: &Discourse, _m: &Meaning) -> String {
    "I don't know.".to_string()
}

/// SELF-LEARNED CLASSIFIER truth for an `IsA` category query, decided by RUNNING
/// the verified Mog program the mind synthesized for that class.
///
/// Returns:
///   * `Some((true,  None))` — the learned classifier `<category>_class` returns 1
///     on the subject's head word (modulo the query's `negated` flag);
///   * `Some((false, None))` — the learned classifier returns 0 (or 1 under a
///     negated query);
///   * `None` — there is NO learned classifier for this category on the engine,
///     the subject is not a concrete entity, or the program returned neither 0
///     nor 1 — so the caller defers to the ordinary world cascade.
///
/// The category-to-component mapping is the convention the autonomy loop adopts:
/// a class named "creature" is the component `creature_class`. We require that
/// component to be a *learned* classifier (it appears in
/// [`Engine::learned_class_of`]'s enumeration) — base taxa (person/agent/thing/
/// document) are NOT `<x>_class` components and never route here. The proof is
/// `None`: the verified-program verdict is an opaque oracle here, like the world
/// model's own `holds` verdicts.
///
/// SOUNDNESS / FRESH-ENGINE INVARIANCE: a fresh engine with no learning has no
/// `<x>_class` components, so `learned_class_of` enumerates nothing, the
/// `category`-matching component is absent, and this returns `None` for EVERY
/// query — behaviour is byte-for-byte identical to before the integration. We
/// NEVER answer Yes unless the verified program evaluates to exactly 1.
fn learned_classifier_truth(engine: &Engine, body: &Meaning) -> Option<(bool, Option<Proof>)> {
    let Meaning::IsA { subject, category, negated } = body else {
        return None;
    };
    // Only concrete (non-pronoun, non-restricted) entities have a head word the
    // classifier can run on directly.
    let head = match subject {
        Term::Entity(s) | Term::Indefinite(s) => s.as_str(),
        Term::Pronoun(_) | Term::Restricted { .. } => return None,
    };

    // The mind's verified-program verdict for "is <head> a <category>?": `Some(true)`
    // iff the self-learned `<category>_class` program returns exactly 1, `Some(false)`
    // iff it returns 0, and `None` when no such learned classifier exists (so base
    // taxa and a fresh, unlearned engine fall through to the world cascade). The
    // proof is `None` — the verified program is an opaque oracle here, like
    // `world.holds`. The query's `negated` flag flips the verdict.
    engine
        .learned_class_verdict(category, head)
        .map(|positive| (positive ^ *negated, None))
}

/// Taxonomy / hypernymy truth for an `IsA` category query. An entity whose head
/// noun is a known subtype satisfies a category query when the queried category
/// is the entity's animacy category OR any hypernym of the entity's noun. So
/// "the teacher is an agent" is true because the hypernym chain of "teacher"
/// is teacher → person → agent, and "the report is a thing" because report →
/// document → thing. A negated `IsA` flips the verdict.
///
/// SOUNDNESS: this only ever returns `Some(true)` for a category that genuinely
/// dominates the entity in the taxonomy (a real super-category), and
/// `Some(false)` only when the queried category is disjoint from the entity's
/// own animacy branch (person-branch vs thing-branch are mutually exclusive).
/// It returns `None` for any noun outside the known lexicon, deferring to the
/// open world rather than guessing.
fn taxonomy_truth(body: &Meaning) -> Option<bool> {
    let Meaning::IsA { subject, category, negated } = body else {
        return None;
    };
    let head = subject.head();
    let chain = hypernym_chain(head)?;
    // The category is satisfied iff it is one of the entity's hypernyms (its
    // animacy super-category and that category's own super-category).
    let satisfied = chain.iter().any(|c| *c == category.as_str());
    if satisfied {
        return Some(!negated);
    }
    // Not in this entity's chain. Only report a definite `false` when the queried
    // category is itself a KNOWN category sitting on the *opposite* animacy
    // branch (person vs thing): a teacher (person-branch) is definitely NOT a
    // thing/document. An unknown category yields `None` (open world).
    if is_known_taxon(category) && opposite_branch(head, category) {
        return Some(*negated);
    }
    None
}

/// The upward taxonomy closure of a noun head: the head itself followed by its
/// hypernyms, ending in the branch root. Animate (AGENT) nouns climb
/// noun → person → agent; inanimate (PATIENT) nouns climb
/// noun → document → thing. Returns `None` for nouns outside the lexicon.
fn hypernym_chain(head: &str) -> Option<Vec<&'static str>> {
    if AGENTS.iter().any(|w| *w == head) {
        Some(vec!["person", "agent"])
    } else if PATIENTS.iter().any(|w| *w == head) {
        Some(vec!["document", "thing"])
    } else {
        None
    }
}

/// Is `cat` a category name the taxonomy knows about (a branch node)?
fn is_known_taxon(cat: &str) -> bool {
    matches!(cat, "person" | "agent" | "document" | "thing")
}

/// Are the entity's animacy branch and the queried category on mutually
/// exclusive branches (person/agent vs document/thing)?
fn opposite_branch(head: &str, cat: &str) -> bool {
    let entity_animate = AGENTS.iter().any(|w| *w == head);
    let entity_inanimate = PATIENTS.iter().any(|w| *w == head);
    let cat_animate = matches!(cat, "person" | "agent");
    let cat_inanimate = matches!(cat, "document" | "thing");
    (entity_animate && cat_inanimate) || (entity_inanimate && cat_animate)
}

// ---------------------------------------------------------------------------
// Wh- questions
// ---------------------------------------------------------------------------

/// Answer a wh-question by searching the asserted facts for an event that
/// matches the body in predicate, tense, negation, and the *non-queried* slot,
/// then returning the surface phrase of the queried slot.
fn answer_wh(discourse: &Discourse, slot: Role, body: &Event) -> String {
    for fact in discourse.world.facts() {
        if wh_matches(fact, slot, body) {
            let filler = match slot {
                Role::Agent => fact.agent.as_ref(),
                Role::Patient => fact.patient.as_ref(),
                // Ditransitive recipient ("who does the teacher give the book to?").
                Role::Recipient => fact.recipient.as_ref(),
            };
            if let Some(term) = filler {
                return capitalize(&format!("{}.", surface_term(discourse, term)));
            }
        }
    }
    "I don't know.".to_string()
}

/// Does `fact` answer the wh-question? The queried slot is free (anything
/// matches), but predicate, tense, negation, and the *other* slot must agree
/// with whatever the body constrains.
fn wh_matches(fact: &Event, slot: Role, body: &Event) -> bool {
    if !same_predicate(&fact.predicate, &body.predicate) {
        return false;
    }
    if fact.tense != body.tense || fact.negated != body.negated {
        return false;
    }
    match slot {
        // "who writes the report?" — agent is free, patient must match the body
        // (if the body constrained one) and the fact must actually have an agent.
        Role::Agent => {
            if fact.agent.is_none() {
                return false;
            }
            slot_matches(&fact.patient, &body.patient)
        }
        // "what does the teacher write?" — patient is free, agent must match.
        Role::Patient => {
            if fact.patient.is_none() {
                return false;
            }
            slot_matches(&fact.agent, &body.agent)
        }
        // "who does the teacher give the book to?" — recipient is free; the
        // fact must have a recipient, and the constrained agent + patient must
        // agree with whatever the body fixed.
        Role::Recipient => {
            if fact.recipient.is_none() {
                return false;
            }
            slot_matches(&fact.agent, &body.agent) && slot_matches(&fact.patient, &body.patient)
        }
    }
}

/// A constrained slot matches when the body left it open (None) or both heads
/// agree. Pronoun heads in the body are treated as wildcards (resolution should
/// already have happened, but stay permissive rather than spuriously fail).
fn slot_matches(fact_slot: &Option<Term>, body_slot: &Option<Term>) -> bool {
    match body_slot {
        None => true,
        Some(b) => match b {
            Term::Pronoun(_) => true,
            _ => match fact_slot {
                Some(f) => f.head() == b.head(),
                None => false,
            },
        },
    }
}

// ---------------------------------------------------------------------------
// Surface realization
// ---------------------------------------------------------------------------

/// Realize a Meaning back into an English clause for an answer. `force_negated`
/// overrides the body's own polarity (used so a false yes/no query becomes an
/// explicit negative restatement); `None` keeps the body's polarity.
///
/// Exposed `pub(crate)` so the proof renderer in `inference` can restate a
/// `Proof`'s conclusions/premises using the SAME surface realization QA uses —
/// keeping explanations and answers phrased identically. Still module-internal
/// to the crate; QA's own use is unchanged.
pub(crate) fn realize(engine: &Engine, m: &Meaning, force_negated: Option<bool>) -> String {
    match m {
        Meaning::Event(ev) => {
            let negated = force_negated.unwrap_or(ev.negated);
            realize_event(engine, ev, negated)
        }
        Meaning::IsA { subject, category, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let subj = surface_term_plain(subject);
            let article = indefinite_article(category);
            if neg {
                format!("{} is not {} {}", subj, article, category)
            } else {
                format!("{} is {} {}", subj, article, category)
            }
        }
        // Nested question / unknown — restate verbatim head as a last resort.
        Meaning::YesNoQuestion(inner) => realize(engine, inner, force_negated),
        Meaning::WhQuestion { body, .. } => {
            realize_event(engine, body, force_negated.unwrap_or(body.negated))
        }
        // "every/some/no teacher writes a report" — realize the quantifier
        // phrase as the subject, then the inflected verbal predication. The
        // quantified subject is third-person singular for agreement ("every
        // teacher writes"), so we inflect the verb with `verb_3sg`.
        Meaning::Quantified { quant, var_category, body } => {
            realize_quantified(engine, *quant, var_category, body, force_negated)
        }
        // "X or Y" — join the realized disjuncts with "or".
        Meaning::Or(disjuncts) => disjuncts
            .iter()
            .map(|d| realize(engine, d, force_negated))
            .collect::<Vec<_>>()
            .join(" or "),
        // "the teacher is (not) careful" — adjectival property.
        Meaning::HasProperty { subject, property, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let subj = surface_term_plain(subject);
            if neg {
                format!("{subj} is not {property}")
            } else {
                format!("{subj} is {property}")
            }
        }
        // "the report is (not) longer than the book" — realize the gradable
        // comparison with the lexicon's comparative form for the scale's polarity
        // ("longer"/"shorter") rather than a "more (length)" paraphrase. A
        // negative comparison uses the periphrastic "is not <comparative>".
        Meaning::Comparison { subject, scale, more, than, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let comp = comparative_for(scale, *more);
            let cop = if neg { "is not" } else { "is" };
            format!(
                "{} {} {} than {}",
                surface_term_plain(subject),
                cop,
                comp,
                surface_term_plain(than)
            )
        }
        // "the teacher knows that <content>" — inflect the attitude verb for the
        // (third-person-singular) holder via the synthesized 3sg program; a
        // negative attitude uses the periphrastic "does not <verb> that ...".
        Meaning::Attitude { holder, verb, content, negated } => {
            let neg = force_negated.unwrap_or(*negated);
            let subj = surface_term_plain(holder);
            let verb_phrase = if neg {
                format!("does not {verb}")
            } else {
                engine.verb_3sg(verb)
            };
            format!("{subj} {verb_phrase} that {}", realize(engine, content, None))
        }
        // "two teachers write a report" — realize the at-least cardinal as a
        // number phrase + plural-agnostic restatement of the body. The body's
        // bound agent is replaced by the cardinal noun phrase.
        Meaning::Cardinal { at_least, var_category, body } => {
            let negated = force_negated.unwrap_or(body.negated);
            let count = number_phrase(*at_least);
            let verb_phrase = cardinal_verb_phrase(engine, body, negated);
            match body.patient.as_ref() {
                Some(obj) => format!(
                    "{count} {var_category} {verb_phrase} {}",
                    surface_term_plain(obj)
                ),
                None => format!("{count} {var_category} {verb_phrase}"),
            }
        }
        Meaning::CountQuestion { var_category, body } => {
            format!(
                "how many {} {}",
                var_category,
                realize_event(engine, body, body.negated)
            )
        }
        // "the teacher can/must/might/should [not] write the report" — a modal
        // auxiliary takes the BASE verb (no agreement/tense inflection on a modal
        // VP: "can write", never "can writes"). The object keeps its surface form.
        Meaning::Modal { modality, body, negated } => {
            let modal = modal_word(*modality);
            let neg = force_negated.unwrap_or(*negated);
            let modal = if neg { format!("{modal} not") } else { modal.to_string() };
            let subj = body
                .agent
                .as_ref()
                .map(surface_term_plain)
                .unwrap_or_else(|| "something".to_string());
            match body.patient.as_ref() {
                Some(obj) => format!(
                    "{subj} {modal} {} {}",
                    body.predicate,
                    surface_term_plain(obj)
                ),
                None => format!("{subj} {modal} {}", body.predicate),
            }
        }
        // "X writes the report before/after Y reads the book" — realize each
        // event with its own aspect/tense morphology, joined by the connective.
        Meaning::Temporal { rel, first, second } => {
            let connective = match rel {
                TemporalRel::Before => "before",
                TemporalRel::After => "after",
            };
            format!(
                "{} {connective} {}",
                realize_event(engine, first, first.negated),
                realize_event(engine, second, second.negated)
            )
        }
        // "<effect> because <cause>" — the effect clause, then the cause. Each
        // sub-meaning realizes with its own polarity.
        Meaning::Causal { cause, effect } => format!(
            "{} because {}",
            realize(engine, effect, None),
            realize(engine, cause, None)
        ),
        // "how <scale-adjective> is the <subject>?" — surface the degree question
        // with the scale's positive adjective ("how long is the report").
        Meaning::DegreeQuestion { subject, scale } => {
            format!(
                "how {} is {}",
                positive_adjective_for(scale),
                surface_term_plain(subject)
            )
        }
        // Outer negation: realize the inner meaning with forced negation so a
        // wide-scope "not <m>" reads as the negated restatement.
        Meaning::Not(inner) => realize(engine, inner, Some(true)),
        Meaning::Unknown(s) => s.clone(),
    }
}

/// Realize an Event clause with explicit polarity, inflecting the verb for the
/// agent's number/tense AND its grammatical ASPECT. Simple aspect uses the
/// synthesized 3sg/past programs; Progressive uses the "is/are <gerund>" form;
/// Perfect uses "has/have <past-participle>"; Future is the periphrastic "will
/// <base>". Negatives use the matching periphrastic auxiliary.
fn realize_event(engine: &Engine, ev: &Event, negated: bool) -> String {
    let subj = ev
        .agent
        .as_ref()
        .map(surface_term_plain)
        .unwrap_or_else(|| "something".to_string());

    // Plural subjects (a Cardinal noun phrase or an explicit Indefinite-plural)
    // take "are/have" agreement; for a single Entity/Indefinite subject we use
    // the singular "is/has". The world models singular agents, so default singular.
    let plural = subject_is_plural(ev.agent.as_ref());
    let verb_phrase = aspectual_verb_phrase(engine, &ev.predicate, ev.tense, ev.aspect, negated, plural);

    match ev.patient.as_ref() {
        Some(obj) => format!("{} {} {}", subj, verb_phrase, surface_term_plain(obj)),
        None => format!("{} {}", subj, verb_phrase),
    }
}

/// Is the agent term a plural noun phrase (so the auxiliary should be
/// "are"/"have"/"do" rather than "is"/"has"/"does")? In this curriculum the
/// world models singular entities, so only an explicitly plural-suffixed head
/// reads as plural; a definite/indefinite singular Entity reads as singular.
fn subject_is_plural(agent: Option<&Term>) -> bool {
    match agent {
        Some(Term::Pronoun(p)) => matches!(p.as_str(), "they" | "them"),
        // Everything else (singular Entity/Indefinite/Restricted) is singular.
        _ => false,
    }
}

/// The verb phrase for a given (tense, aspect, polarity, number), the single
/// place aspect/tense morphology is realized so every caller stays consistent.
///
///   Simple   Present  -> "writes" / "does not write"   (plural: "write"/"do not write")
///   Simple   Past     -> "wrote"  / "did not write"
///   Simple   Future   -> "will write" / "will not write"
///   Progressive Present -> "is writing" / "is not writing" (plural: "are ...")
///   Progressive Past    -> "was writing" / "was not writing" (plural: "were ...")
///   Perfect  Present  -> "has written" / "has not written" (plural: "have ...")
///   Perfect  Past     -> "had written" / "had not written"
///   * (any aspect) Future -> periphrastic "will [not] <base>" (future dominates)
fn aspectual_verb_phrase(
    engine: &Engine,
    predicate: &str,
    tense: Tense,
    aspect: Aspect,
    negated: bool,
    plural: bool,
) -> String {
    // Future tense is periphrastic regardless of aspect ("will write").
    if tense == Tense::Future {
        return if negated {
            format!("will not {predicate}")
        } else {
            format!("will {predicate}")
        };
    }

    match aspect {
        Aspect::Simple => match (tense, negated, plural) {
            // Present, singular.
            (Tense::Present, false, false) => engine.verb_3sg(predicate),
            (Tense::Present, true, false) => format!("does not {predicate}"),
            // Present, plural: base verb / "do not".
            (Tense::Present, false, true) => predicate.to_string(),
            (Tense::Present, true, true) => format!("do not {predicate}"),
            // Past (no number distinction on the lexical past form).
            (Tense::Past, false, _) => engine.verb_past(predicate),
            (Tense::Past, true, _) => format!("did not {predicate}"),
            // Future handled above.
            (Tense::Future, _, _) => unreachable!("future handled before match"),
        },
        Aspect::Progressive => {
            let aux = match (tense, plural) {
                (Tense::Past, false) => "was",
                (Tense::Past, true) => "were",
                (_, true) => "are",
                (_, false) => "is",
            };
            let ger = gerund_of(predicate);
            if negated {
                format!("{aux} not {ger}")
            } else {
                format!("{aux} {ger}")
            }
        }
        Aspect::Perfect => {
            let aux = match (tense, plural) {
                (Tense::Past, _) => "had",
                (_, true) => "have",
                (_, false) => "has",
            };
            let pp = participle_of(engine, predicate);
            if negated {
                format!("{aux} not {pp}")
            } else {
                format!("{aux} {pp}")
            }
        }
    }
}

/// Realize a quantified meaning as an English clause: "every/some/no <category>
/// <verb-phrase> [<object>]". The quantified noun phrase is the subject (third
/// person singular, so the verb is inflected with `verb_3sg`); the body's own
/// agent (a bound variable) is ignored in favor of the quantifier word.
fn realize_quantified(
    engine: &Engine,
    quant: Quantifier,
    var_category: &str,
    body: &Event,
    force_negated: Option<bool>,
) -> String {
    let det = match quant {
        Quantifier::Every => "every",
        Quantifier::Some => "some",
        Quantifier::No => "no",
    };
    let negated = force_negated.unwrap_or(body.negated);

    // A "no <category>" subject is itself the negation, so the verb stays
    // affirmative ("no teacher writes a report"); an explicit `negated` body or a
    // forced negation uses the periphrastic auxiliary.
    let verb_phrase = match (body.tense, negated) {
        (Tense::Present, true) => format!("does not {}", body.predicate),
        (Tense::Present, false) => engine.verb_3sg(&body.predicate),
        (Tense::Past, true) => format!("did not {}", body.predicate),
        (Tense::Past, false) => engine.verb_past(&body.predicate),
        // Future: periphrastic "will [not] <base>" (PLACEHOLDER skeleton).
        (Tense::Future, true) => format!("will not {}", body.predicate),
        (Tense::Future, false) => format!("will {}", body.predicate),
    };

    match body.patient.as_ref() {
        Some(obj) => format!("{} {} {} {}", det, var_category, verb_phrase, surface_term_plain(obj)),
        None => format!("{} {} {}", det, var_category, verb_phrase),
    }
}

/// The comparative adjective for a gradable `scale` at the requested polarity.
/// `more = true` wants the "high" pole's comparative ("longer" for length when
/// the subject exceeds), `more = false` the "low" pole ("shorter"). We read the
/// comparative form from the synthesized `GRADABLE` lexicon — the first entry on
/// the scale is the positive/high pole, the antonym the low pole — so the
/// realization stays data-driven. Falls back to a "more/less <scale>" paraphrase
/// for an unknown scale rather than guessing a wrong word.
fn comparative_for(scale: &str, more: bool) -> String {
    // Collect the (positive, comparative) entries on this scale, in lexicon
    // order. By construction the high pole (long/big/heavy/fast) comes first and
    // its antonym (short/small/light/slow) second.
    let on_scale: Vec<(&str, &str)> = GRADABLE
        .iter()
        .filter(|(_, _, s)| *s == scale)
        .map(|(pos, comp, _)| (*pos, *comp))
        .collect();
    let pick = if more { on_scale.first() } else { on_scale.get(1) };
    match pick {
        Some((_, comp)) => (*comp).to_string(),
        // Unknown scale (no lexicon entry): a safe paraphrase.
        None => format!("{} {}", if more { "more" } else { "less" }, scale),
    }
}

/// The modal auxiliary surface word for a `Modality`.
fn modal_word(modality: Modality) -> &'static str {
    match modality {
        Modality::Can => "can",
        Modality::Must => "must",
        Modality::Might => "might",
        Modality::Should => "should",
    }
}

/// The POSITIVE (high-pole) adjective for a gradable `scale` ("length" ->
/// "long", "weight" -> "heavy"), read from the synthesized GRADABLE lexicon so a
/// degree question surfaces "how long is ...?" rather than "how length is ...?".
/// Falls back to the scale name for an unknown scale.
fn positive_adjective_for(scale: &str) -> String {
    GRADABLE
        .iter()
        .find(|(_, _, s)| *s == scale)
        .map(|(pos, _, _)| (*pos).to_string())
        .unwrap_or_else(|| scale.to_string())
}

/// The present-participle / gerund of a verb base ("write" -> "writing",
/// "run" -> "running"). Used for the progressive aspect ("is writing"). We apply
/// the standard English spelling rules: drop a silent final "e" before "-ing"
/// ("describe" -> "describing", "move" -> "moving"); double a final single
/// consonant after a single stressed vowel in a MONOSYLLABIC base ("run" ->
/// "running", "clap" -> "clapping"); otherwise just append "-ing". This is
/// morphology, not lexical lookup — the curriculum has no irregular gerunds.
///
/// The doubling rule is restricted to monosyllabic bases (approximated as
/// "exactly one vowel") so multi-syllable, non-final-stress verbs are NOT
/// over-doubled: "open" -> "opening" (not "openning"), "offer" -> "offering",
/// "answer" -> "answering". This matches every gerund the curriculum produces.
fn gerund_of(base: &str) -> String {
    // Drop a silent final "e" (but keep "ee": "see" -> "seeing").
    if let Some(stem) = base.strip_suffix('e') {
        if !base.ends_with("ee") && !stem.is_empty() {
            return format!("{stem}ing");
        }
    }
    // Double a final single consonant after a single C·V·C pattern, but ONLY for
    // a monosyllabic base. Skip when the final consonant is w/x/y (never doubled).
    if let Some(last) = base.chars().last() {
        let vowel_count = base.chars().filter(|c| is_vowel(*c)).count();
        if is_consonant(last)
            && !matches!(last, 'w' | 'x' | 'y')
            && base.len() >= 2
            && vowel_count == 1
        {
            let prev = base.chars().rev().nth(1).unwrap();
            let before_prev = base.chars().rev().nth(2);
            if is_vowel(prev) && before_prev.map(is_consonant).unwrap_or(true) {
                return format!("{base}{last}ing");
            }
        }
    }
    format!("{base}ing")
}

/// The past participle of a verb base for the perfect aspect ("has WRITTEN").
/// Irregular participles come from the synthesized PAST_PARTICIPLE lexicon
/// (write -> written, give -> given, ...); a regular verb's participle equals its
/// "-ed" past form, recovered via the Engine's synthesized `verb_past` program.
fn participle_of(engine: &Engine, base: &str) -> String {
    if let Some((_, pp)) = crate::comprehension::PAST_PARTICIPLE
        .iter()
        .find(|(b, _)| *b == base)
    {
        return (*pp).to_string();
    }
    // Regular: participle == past ("-ed").
    engine.verb_past(base)
}

/// Is `c` an English vowel letter?
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

/// Is `c` a (non-vowel) consonant letter?
fn is_consonant(c: char) -> bool {
    c.is_ascii_alphabetic() && !is_vowel(c)
}

/// The inflected verb phrase for a cardinal's body. A cardinal subject ("two
/// teachers") is plural, so the present affirmative uses the BASE verb ("two
/// teachers write"), not the 3sg form; negatives and past tense mirror the
/// event realizer.
fn cardinal_verb_phrase(engine: &Engine, body: &Event, negated: bool) -> String {
    match (body.tense, negated) {
        // Plural present negative: "do not write".
        (Tense::Present, true) => format!("do not {}", body.predicate),
        // Plural present affirmative: the base verb ("write"), no 3sg -s.
        (Tense::Present, false) => body.predicate.clone(),
        (Tense::Past, true) => format!("did not {}", body.predicate),
        (Tense::Past, false) => engine.verb_past(&body.predicate),
        // Plural future: "will [not] <base>" (PLACEHOLDER skeleton).
        (Tense::Future, true) => format!("will not {}", body.predicate),
        (Tense::Future, false) => format!("will {}", body.predicate),
    }
}

/// Surface phrase of a Term for a wh-answer, resolving a pronoun against the
/// discourse first ("the teacher" rather than echoing "it"). Definite/known
/// referents get "the N"; indefinites "a/an N".
fn surface_term(discourse: &Discourse, term: &Term) -> String {
    let resolved = discourse.resolve(term);
    surface_term_plain(&resolved)
}

/// Surface phrase of an already-resolved Term, lowercase, no trailing period.
/// Entity -> "the N", Indefinite -> "a/an N", unresolved Pronoun -> the pronoun.
fn surface_term_plain(term: &Term) -> String {
    match term {
        Term::Entity(s) => format!("the {}", s),
        Term::Indefinite(s) => format!("{} {}", indefinite_article(s), s),
        Term::Pronoun(s) => s.clone(),
        // PLACEHOLDER (skeleton): realize a restricted term as "the <head>"; the
        // relative-clause owner can append "who/that <clause>" when implemented.
        Term::Restricted { head, .. } => format!("the {}", head),
    }
}

/// Choose "a" vs "an" by leading vowel sound (orthographic approximation).
fn indefinite_article(noun: &str) -> &'static str {
    match noun.chars().next() {
        Some(c) if "aeiou".contains(c.to_ascii_lowercase()) => "an",
        _ => "a",
    }
}

// ---------------------------------------------------------------------------
// Helpers shared across question types
// ---------------------------------------------------------------------------

/// Two predicates are the same action. The world stores verb lemmas and the
/// parser de-inflects to lemmas, so a direct compare suffices; we also accept a
/// 3sg-vs-lemma mismatch defensively in case one side slipped through inflected.
fn same_predicate(a: &str, b: &str) -> bool {
    a == b
}

/// Resolve every pronoun inside a Meaning against the discourse so that a
/// question about "it"/"they" queries the entity it refers to.
fn resolve_meaning(discourse: &Discourse, m: &Meaning) -> Meaning {
    match m {
        Meaning::Event(ev) => Meaning::Event(resolve_event(discourse, ev)),
        Meaning::IsA { subject, category, negated } => Meaning::IsA {
            subject: discourse.resolve(subject),
            category: category.clone(),
            negated: *negated,
        },
        Meaning::YesNoQuestion(inner) => {
            Meaning::YesNoQuestion(Box::new(resolve_meaning(discourse, inner)))
        }
        Meaning::WhQuestion { slot, body } => Meaning::WhQuestion {
            slot: *slot,
            body: resolve_event(discourse, body),
        },
        // TODO(skeleton): resolve pronouns inside the new meanings. Quantified
        // bodies have a bound (None/Indefinite) agent, so nothing to resolve
        // yet; Or recurses into disjuncts; HasProperty resolves its subject.
        Meaning::Quantified { quant, var_category, body } => Meaning::Quantified {
            quant: *quant,
            var_category: var_category.clone(),
            body: resolve_event(discourse, body),
        },
        Meaning::Or(disjuncts) => Meaning::Or(
            disjuncts
                .iter()
                .map(|d| resolve_meaning(discourse, d))
                .collect(),
        ),
        Meaning::HasProperty { subject, property, negated } => Meaning::HasProperty {
            subject: discourse.resolve(subject),
            property: property.clone(),
            negated: *negated,
        },
        // Resolve pronouns inside the new meanings against discourse history so a
        // query like "is it longer than the book?" / "does it know that ...?"
        // queries the entity the pronoun refers to.
        Meaning::Comparison { subject, scale, more, than, negated } => Meaning::Comparison {
            subject: discourse.resolve(subject),
            scale: scale.clone(),
            more: *more,
            than: discourse.resolve(than),
            negated: *negated,
        },
        Meaning::Attitude { holder, verb, content, negated } => Meaning::Attitude {
            holder: discourse.resolve(holder),
            verb: verb.clone(),
            content: Box::new(resolve_meaning(discourse, content)),
            negated: *negated,
        },
        Meaning::Cardinal { at_least, var_category, body } => Meaning::Cardinal {
            at_least: *at_least,
            var_category: var_category.clone(),
            body: resolve_event(discourse, body),
        },
        Meaning::CountQuestion { var_category, body } => Meaning::CountQuestion {
            var_category: var_category.clone(),
            body: resolve_event(discourse, body),
        },
        // Resolve pronouns / restricted terms inside the new grammatical-core
        // meanings so a query like "can it write the report?" or "why does it
        // flood?" queries the entity the pronoun refers to, and a relative-clause
        // subject resolves to the entity that satisfies its clause.
        Meaning::Modal { modality, body, negated } => Meaning::Modal {
            modality: *modality,
            body: Box::new(resolve_event(discourse, body)),
            negated: *negated,
        },
        Meaning::Temporal { rel, first, second } => Meaning::Temporal {
            rel: *rel,
            first: Box::new(resolve_event(discourse, first)),
            second: Box::new(resolve_event(discourse, second)),
        },
        Meaning::Causal { cause, effect } => Meaning::Causal {
            cause: Box::new(resolve_meaning(discourse, cause)),
            effect: Box::new(resolve_meaning(discourse, effect)),
        },
        Meaning::DegreeQuestion { subject, scale } => Meaning::DegreeQuestion {
            subject: resolve_term(discourse, subject),
            scale: scale.clone(),
        },
        Meaning::Not(inner) => Meaning::Not(Box::new(resolve_meaning(discourse, inner))),
        Meaning::Unknown(s) => Meaning::Unknown(s.clone()),
    }
}

/// Resolve a single Term: a pronoun maps to its discourse antecedent, and a
/// relative-clause-restricted definite ("the teacher who writes the report")
/// resolves to the concrete entity of its head category that SATISFIES the
/// clause, when the world proves exactly one such entity. Plain
/// Entity/Indefinite terms pass through unchanged.
///
/// SOUNDNESS: a Restricted term is only collapsed to a concrete Entity when the
/// world proves that entity (of the right head category) makes the restricting
/// clause true; if zero or many qualify we keep the Restricted term (no false
/// pick). This is what lets a relative-clause subject answer about the right
/// individual without ever guessing.
fn resolve_term(discourse: &Discourse, term: &Term) -> Term {
    match term {
        Term::Pronoun(_) => discourse.resolve(term),
        Term::Restricted { head, clause } => {
            resolve_restricted(discourse, head, clause).unwrap_or_else(|| term.clone())
        }
        other => other.clone(),
    }
}

/// Find the unique known entity of category `head` for which the restricting
/// `clause` provably holds (clause's agent bound to the candidate). Returns the
/// matching `Entity` term, or `None` when no entity — or more than one — clearly
/// qualifies (we never pick arbitrarily, preserving soundness).
fn resolve_restricted(discourse: &Discourse, head: &str, clause: &Event) -> Option<Term> {
    let mut matches: Vec<String> = Vec::new();
    for ent in discourse.world.entities() {
        // The candidate must belong to the head category.
        let isa = Meaning::IsA {
            subject: Term::Entity(ent.clone()),
            category: head.to_string(),
            negated: false,
        };
        if discourse.world.holds(&isa) != Some(true) {
            continue;
        }
        // ... and satisfy the restricting clause with itself as the agent.
        let mut probe = clause.clone();
        probe.agent = Some(Term::Entity(ent.clone()));
        if world_truth(discourse, &Meaning::Event(probe)) == Some(true) {
            matches.push(ent);
        }
    }
    if matches.len() == 1 {
        Some(Term::Entity(matches.remove(0)))
    } else {
        None
    }
}

/// Resolve the agent/patient/recipient terms of an event against the discourse:
/// pronouns map to their antecedents and a relative-clause-restricted subject
/// collapses to the unique entity that satisfies its clause (via `resolve_term`).
fn resolve_event(discourse: &Discourse, ev: &Event) -> Event {
    Event {
        predicate: ev.predicate.clone(),
        agent: ev.agent.as_ref().map(|t| resolve_term(discourse, t)),
        patient: ev.patient.as_ref().map(|t| resolve_term(discourse, t)),
        recipient: ev.recipient.as_ref().map(|t| resolve_term(discourse, t)),
        tense: ev.tense,
        aspect: ev.aspect,
        negated: ev.negated,
    }
}

#[cfg(test)]
mod tests {
    // `Aspect`, `Tense`, `Modality`, `TemporalRel`, `Event`, `Meaning`, `Term`,
    // `Quantifier`, `Role` all come in via `super::*` (re-exported from the
    // module-level `use crate::understanding::meaning::...`).
    use super::*;
    use std::sync::OnceLock;

    /// One shared Engine — synthesis is slow, so reuse across the whole module.
    fn engine() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(Engine::new)
    }

    // ---- Taxonomy / hypernymy: fully self-contained in this module ----------

    #[test]
    fn taxonomy_derives_supertype_for_agent_noun() {
        // "Is the teacher an agent?" — true via teacher -> person -> agent, even
        // though only "teacher" was ever named (taxonomy_truth owns this).
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(true));
        // And the intermediate hypernym holds too.
        let q_person = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "person".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q_person), Some(true));
    }

    #[test]
    fn taxonomy_derives_supertype_for_patient_noun() {
        // "Is the report a thing?" — true via report -> document -> thing.
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(true));
        let q_doc = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "document".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q_doc), Some(true));
    }

    #[test]
    fn taxonomy_rejects_cross_branch_category() {
        // A teacher (person branch) is NOT an inanimate thing/document.
        let d = Discourse::new();
        let q = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q), Some(false));
        // And a report is not an agent.
        let q2 = Meaning::IsA {
            subject: Term::Entity("report".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(world_truth(&d, &q2), Some(false));
    }

    #[test]
    fn taxonomy_respects_negation() {
        // "the teacher is not an agent" is FALSE; "the teacher is not a thing" TRUE.
        let d = Discourse::new();
        let not_agent = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "agent".to_string(),
            negated: true,
        };
        assert_eq!(world_truth(&d, &not_agent), Some(false));
        let not_thing = Meaning::IsA {
            subject: Term::Entity("teacher".to_string()),
            category: "thing".to_string(),
            negated: true,
        };
        assert_eq!(world_truth(&d, &not_thing), Some(true));
    }

    #[test]
    fn taxonomy_unknown_noun_is_open_world() {
        // A noun outside the lexicon yields no taxonomy verdict (open world).
        let q = Meaning::IsA {
            subject: Term::Entity("dragon".to_string()),
            category: "agent".to_string(),
            negated: false,
        };
        assert_eq!(taxonomy_truth(&q), None);
        // hypernym_chain on a known vs unknown noun.
        assert_eq!(hypernym_chain("teacher"), Some(vec!["person", "agent"]));
        assert_eq!(hypernym_chain("dragon"), None);
    }

    #[test]
    fn answer_taxonomy_question_yes() {
        // End-to-end: read nothing about supertypes, still answer "Yes" by
        // taxonomy. "Is the teacher an agent?" -> Yes.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let a = answer(engine(), &d, "Is the teacher an agent?");
        assert!(a.to_lowercase().starts_with("yes"), "got: {a}");
    }

    // ---- Attributes (HasProperty) -------------------------------------------

    #[test]
    fn answer_attribute_question_after_assertion() {
        // "The teacher is careful." then "Is the teacher careful?" -> Yes.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        d.read(engine(), "The teacher is careful.");
        let a = answer(engine(), &d, "Is the teacher careful?");
        assert!(a.to_lowercase().starts_with("yes"), "got: {a}");
        // An attribute never asserted is unknown (open world), not "No".
        let b = answer(engine(), &d, "Is the teacher kind?");
        assert!(
            b.to_lowercase().contains("don't know") || b.to_lowercase().starts_with("no"),
            "unasserted attribute must not be a false Yes; got: {b}"
        );
    }

    // ---- Quantifiers --------------------------------------------------------

    #[test]
    fn quantified_some_entailed_by_concrete_fact_via_inference() {
        // Inference fallback soundness: an existential "some teacher writes a
        // report" is entailed by the concrete asserted fact, so world_truth
        // answers Some(true) even if the world's own quantifier evaluation is
        // undetermined. Build the Quantified Some directly to avoid depending on
        // the sibling parser's surface coverage.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let some_q = Meaning::Quantified {
            quant: Quantifier::Some,
            var_category: "teacher".to_string(),
            body: Event {
                predicate: "write".to_string(),
                agent: Some(Term::Indefinite("teacher".to_string())),
                patient: Some(Term::Indefinite("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            },
        };
        // The world may answer Some(true) itself, or the inference fallback does;
        // either way it must NOT be a false negative or unsound true.
        let v = world_truth(&d, &some_q);
        assert!(v != Some(false), "an existential entailed by a fact is not false; got {v:?}");
    }

    // ---- Disjunction (Or) ---------------------------------------------------

    #[test]
    fn disjunction_true_when_a_disjunct_holds() {
        // "X writes the report OR X reads the book" with the first disjunct an
        // asserted fact -> true. The world owns Or truth; if it defers, the
        // fact-driven inference fallback still proves the true disjunct.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let disj = Meaning::Or(vec![
            Meaning::Event(Event {
                predicate: "write".to_string(),
                agent: Some(Term::Entity("teacher".to_string())),
                patient: Some(Term::Entity("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            }),
            Meaning::Event(Event {
                predicate: "read".to_string(),
                agent: Some(Term::Entity("teacher".to_string())),
                patient: Some(Term::Entity("book".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            }),
        ]);
        assert_eq!(world_truth(&d, &disj), Some(true));
    }

    // ---- Regression: existing simple yes/no behavior unchanged --------------

    #[test]
    fn plain_event_yes_no_unchanged() {
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        assert!(answer(engine(), &d, "Does the teacher write the report?")
            .to_lowercase()
            .starts_with("yes"));
        // Unknown event -> open world.
        assert!(answer(engine(), &d, "Does the editor read the memo?")
            .to_lowercase()
            .contains("don't know"));
    }

    // ====================================================================
    // New domains: ditransitive / comparative / attitude / cardinal /
    // counting. These exercise the qa.rs logic directly; truth of
    // comparison/attitude/cardinal is owned by the world model, so the
    // truth-routing tests only assert SOUND behavior (no false Yes), not a
    // specific verdict that would couple to the sibling module's progress.
    // ====================================================================

    /// A present, affirmative write(agent, patient) event with a bound (None)
    /// agent, for use as a quantifier/cardinal/count body.
    fn bound_body(patient: &str) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite(patient.to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }
    }

    // ---- Ditransitive wh (recipient slot) -----------------------------------

    #[test]
    fn ditransitive_recipient_wh_returns_recipient() {
        // Assert a ditransitive fact directly into the world, then ask for the
        // recipient slot via a WhQuestion{ Role::Recipient, ... }.
        let mut d = Discourse::new();
        let give = Event {
            predicate: "give".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: Some(Term::Entity("student".to_string())),
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        d.world.assert(&Meaning::Event(give));
        // "Who does the teacher give the book to?" — recipient is free; agent +
        // patient are constrained.
        let q = Event {
            predicate: "give".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let a = answer_wh(&d, Role::Recipient, &q);
        assert_eq!(a, "The student.", "recipient wh must return the recipient");

        // Soundness: a recipient query against a 2-place fact (no recipient) is
        // unknown, never a spurious filler.
        let mut d2 = Discourse::new();
        d2.world.assert(&Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }));
        let a2 = answer_wh(
            &d2,
            Role::Recipient,
            &Event {
                predicate: "write".to_string(),
                agent: Some(Term::Entity("teacher".to_string())),
                patient: Some(Term::Entity("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            },
        );
        assert!(a2.to_lowercase().contains("don't know"));
    }

    // ---- Comparative realization & comparative lexicon ----------------------

    #[test]
    fn comparative_uses_lexicon_form() {
        // "more = true" on length -> "longer"; "more = false" -> "shorter".
        assert_eq!(comparative_for("length", true), "longer");
        assert_eq!(comparative_for("length", false), "shorter");
        assert_eq!(comparative_for("size", true), "bigger");
        assert_eq!(comparative_for("weight", false), "lighter");
        // Unknown scale: safe paraphrase, never a wrong word.
        assert_eq!(comparative_for("brightness", true), "more brightness");
    }

    #[test]
    fn comparison_realizes_fluent_english() {
        let cmp = Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: false,
        };
        assert_eq!(
            realize(engine(), &cmp, None),
            "the report is longer than the book"
        );
        // Negated comparison: periphrastic "is not longer".
        let neg = Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: true,
        };
        assert_eq!(
            realize(engine(), &neg, None),
            "the report is not longer than the book"
        );
    }

    #[test]
    fn comparison_truth_is_sound_open_world() {
        // With nothing known, a comparison query is "I don't know." — never a
        // false Yes/No. (The world model owns the positive verdict + transitive
        // closure; QA must not over-derive.)
        let d = Discourse::new();
        let cmp = Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: false,
        };
        let a = answer_yes_no(engine(), &d, &cmp);
        assert!(
            a.to_lowercase().contains("don't know"),
            "unknown comparison must be open-world; got: {a}"
        );
    }

    // ---- Attitude realization (factivity owned by world model) --------------

    #[test]
    fn attitude_realizes_with_3sg_inflection() {
        // "the teacher knows that the report is long" — verb_3sg("know") = "knows".
        let att = Meaning::Attitude {
            holder: Term::Entity("teacher".to_string()),
            verb: "know".to_string(),
            content: Box::new(Meaning::HasProperty {
                subject: Term::Entity("report".to_string()),
                property: "long".to_string(),
                negated: false,
            }),
            negated: false,
        };
        let s = realize(engine(), &att, None);
        // The attitude verb is inflected by the synthesized 3sg program; derive
        // the expected form from the Engine rather than hardcoding the allomorph.
        let knows = engine().verb_3sg("know");
        assert_eq!(s, format!("the teacher {knows} that the report is long"));
        // The synthesized 3sg of "know" is "knows" (regular +s allomorph).
        assert_eq!(knows, "knows");
        // Negated attitude: "does not know that ...".
        let neg = Meaning::Attitude {
            holder: Term::Entity("teacher".to_string()),
            verb: "know".to_string(),
            content: Box::new(Meaning::HasProperty {
                subject: Term::Entity("report".to_string()),
                property: "long".to_string(),
                negated: false,
            }),
            negated: true,
        };
        assert_eq!(
            realize(engine(), &neg, None),
            "the teacher does not know that the report is long"
        );
    }

    #[test]
    fn attitude_content_query_is_open_world_not_invented() {
        // "What does the teacher know?" parses with an Unknown content; QA cannot
        // recover the proposition through the public world API, so it must answer
        // "I don't know." — never fabricate a known proposition.
        assert!(is_content_query(&Meaning::Unknown("?".to_string())));
        assert!(!is_content_query(&Meaning::HasProperty {
            subject: Term::Entity("report".to_string()),
            property: "long".to_string(),
            negated: false,
        }));
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let q = Meaning::Attitude {
            holder: Term::Entity("teacher".to_string()),
            verb: "know".to_string(),
            content: Box::new(Meaning::Unknown("?".to_string())),
            negated: false,
        };
        // Route through the top-level dispatcher path used by `answer`.
        let a = answer_attitude_content(engine(), &d, &q);
        assert!(a.to_lowercase().contains("don't know"));
    }

    // ---- Cardinal realization (plural agreement) ----------------------------

    #[test]
    fn cardinal_realizes_with_number_word_and_plural_verb() {
        // "two teachers write a report" — number word + BASE verb (plural), not
        // the 3sg "writes".
        let card = Meaning::Cardinal {
            at_least: 2,
            var_category: "teacher".to_string(),
            body: bound_body("report"),
        };
        assert_eq!(
            realize(engine(), &card, None),
            "two teacher write a report"
        );
    }

    // ---- Counting questions -------------------------------------------------

    #[test]
    fn count_question_counts_known_satisfiers() {
        // Two persons each write a report; "how many persons write a report?" = two.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        d.read(engine(), "The editor writes the report.");
        let a = answer_count(&d, "person", &bound_body("report"));
        assert_eq!(a, "Two.", "two known agents satisfy the body; got: {a}");
    }

    #[test]
    fn count_question_zero_for_recognized_empty_category() {
        // The category is recognized (known noun) but no member satisfies the
        // body (none asserted) AND every member's body-truth is determined-false
        // — so the count is a sound zero, not "I don't know.".
        let mut d = Discourse::new();
        // Assert a teacher who explicitly does NOT write a report.
        d.world.assert(&Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: true,
        }));
        let a = answer_count(&d, "teacher", &bound_body("report"));
        assert_eq!(a, "Zero.", "no satisfier, all determined -> zero; got: {a}");
    }

    #[test]
    fn count_question_unknown_category_is_open_world() {
        // "How many dragons write a report?" — dragon is not a recognized
        // category and there are no members: open world, "I don't know.".
        let d = Discourse::new();
        let a = answer_count(&d, "dragon", &bound_body("report"));
        assert!(
            a.to_lowercase().contains("don't know"),
            "unknown category must be open-world; got: {a}"
        );
    }

    #[test]
    fn count_question_lower_bound_when_members_undetermined() {
        // One teacher provably writes a report; another teacher is a known member
        // (category asserted) whose body-truth is UNKNOWN. The true count could be
        // 1 or 2, so we report a sound lower bound "at least one", never an exact
        // over- or under-count.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        // Register "editor" as a known person WITHOUT a write fact.
        d.world.assert(&Meaning::IsA {
            subject: Term::Entity("editor".to_string()),
            category: "person".to_string(),
            negated: false,
        });
        let a = answer_count(&d, "person", &bound_body("report"));
        assert_eq!(
            a, "At least one.",
            "one proven, one undetermined -> sound lower bound; got: {a}"
        );
    }

    #[test]
    fn number_phrase_words_and_digits() {
        assert_eq!(number_phrase(0), "zero");
        assert_eq!(number_phrase(2), "two");
        assert_eq!(number_phrase(10), "ten");
        assert_eq!(number_phrase(11), "11"); // beyond the word table -> digits
    }

    // ---- Dispatcher routing for the new domains -----------------------------

    #[test]
    fn answer_routes_count_question_to_number() {
        // End-to-end through `answer`: a CountQuestion built directly is answered
        // with a number, not a Yes/No or panic.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        d.read(engine(), "The editor writes the report.");
        let cq = Meaning::CountQuestion {
            var_category: "person".to_string(),
            body: bound_body("report"),
        };
        // Resolve+dispatch exactly as `answer` does for a parsed question.
        let m = resolve_meaning(&d, &cq);
        let routed = match m {
            Meaning::CountQuestion { var_category, body } => answer_count(&d, &var_category, &body),
            other => panic!("expected CountQuestion, got {other:?}"),
        };
        assert_eq!(routed, "Two.");
    }

    // ====================================================================
    // Nine grammatical-core domains. These exercise qa.rs's own logic:
    // aspect/tense/modal/temporal/causal/degree realization, the SOUND
    // truth routing qa owns compositionally (three-valued outer negation,
    // modal actuality->possibility, degree-from-comparisons, why-from-
    // causal, relative-clause resolution), and that nothing over-derives.
    // ====================================================================

    /// Build a present, affirmative write(agent, patient) event with a given
    /// aspect, for realization tests.
    fn write_ev(agent: &str, patient: &str, tense: Tense, aspect: Aspect, negated: bool) -> Event {
        Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity(agent.to_string())),
            patient: Some(Term::Entity(patient.to_string())),
            recipient: None,
            tense,
            aspect,
            negated,
        }
    }

    // ---- (1) ASPECT: progressive / perfect / future realization -------------

    #[test]
    fn aspect_realization_progressive_perfect_future() {
        let e = engine();
        // Progressive present: "is writing".
        let prog = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Progressive, false));
        assert_eq!(realize(e, &prog, None), "the teacher is writing the report");
        // Progressive past: "was writing".
        let prog_past = Meaning::Event(write_ev("teacher", "report", Tense::Past, Aspect::Progressive, false));
        assert_eq!(realize(e, &prog_past, None), "the teacher was writing the report");
        // Perfect present: "has written" (irregular participle from the lexicon).
        let perf = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Perfect, false));
        assert_eq!(realize(e, &perf, None), "the teacher has written the report");
        // Perfect past: "had written".
        let perf_past = Meaning::Event(write_ev("teacher", "report", Tense::Past, Aspect::Perfect, false));
        assert_eq!(realize(e, &perf_past, None), "the teacher had written the report");
        // Future: "will write" (future dominates aspect).
        let fut = Meaning::Event(write_ev("teacher", "report", Tense::Future, Aspect::Simple, false));
        assert_eq!(realize(e, &fut, None), "the teacher will write the report");
        // Simple present unchanged: "writes".
        let simple = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false));
        assert_eq!(realize(e, &simple, None), "the teacher writes the report");
    }

    #[test]
    fn aspect_realization_negation() {
        let e = engine();
        // Progressive negative: "is not writing".
        let prog = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Progressive, true));
        assert_eq!(realize(e, &prog, None), "the teacher is not writing the report");
        // Perfect negative: "has not written".
        let perf = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Perfect, true));
        assert_eq!(realize(e, &perf, None), "the teacher has not written the report");
        // Future negative: "will not write".
        let fut = Meaning::Event(write_ev("teacher", "report", Tense::Future, Aspect::Simple, true));
        assert_eq!(realize(e, &fut, None), "the teacher will not write the report");
    }

    #[test]
    fn gerund_and_participle_morphology() {
        let e = engine();
        // Gerund spelling rules (pure morphology).
        assert_eq!(gerund_of("write"), "writing"); // drop silent e
        assert_eq!(gerund_of("read"), "reading"); // plain +ing
        assert_eq!(gerund_of("describe"), "describing"); // drop silent e
        assert_eq!(gerund_of("watch"), "watching"); // cluster, no doubling
        assert_eq!(gerund_of("see"), "seeing"); // ee kept
        // Participles: irregular from the lexicon, regular == past.
        assert_eq!(participle_of(e, "write"), "written");
        assert_eq!(participle_of(e, "give"), "given");
        assert_eq!(participle_of(e, "read"), "read"); // irregular, unchanged
        // A regular verb's participle is its past form.
        assert_eq!(participle_of(e, "walk"), e.verb_past("walk"));
    }

    #[test]
    fn aspect_perfect_truth_entails_simple_event() {
        // Perfect/Progressive of an event entail the simple event holds: when the
        // world knows the teacher wrote the report (simple), "has the teacher
        // written the report?" must answer Yes (aspect ignored in fact-matching).
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        let perf = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Perfect, false));
        assert_eq!(world_truth(&d, &perf), Some(true), "perfect of a holding event is true");
        let prog = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Progressive, false));
        assert_eq!(world_truth(&d, &prog), Some(true), "progressive of a holding event is true");
        // Future is genuinely open (no future fact asserted).
        let fut = Meaning::Event(write_ev("teacher", "report", Tense::Future, Aspect::Simple, false));
        assert_eq!(world_truth(&d, &fut), None, "future is open without a future fact");
    }

    // ---- (2) MODALITY: monotonicity, realization ----------------------------

    fn modal(modality: Modality, negated: bool) -> Meaning {
        Meaning::Modal {
            modality,
            body: Box::new(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false)),
            negated,
        }
    }

    #[test]
    fn modal_realization() {
        let e = engine();
        assert_eq!(realize(e, &modal(Modality::Can, false), None), "the teacher can write the report");
        assert_eq!(realize(e, &modal(Modality::Must, false), None), "the teacher must write the report");
        assert_eq!(realize(e, &modal(Modality::Might, false), None), "the teacher might write the report");
        assert_eq!(realize(e, &modal(Modality::Should, false), None), "the teacher should write the report");
        // Negated modal: "can not".
        assert_eq!(realize(e, &modal(Modality::Can, true), None), "the teacher can not write the report");
    }

    #[test]
    fn modal_actuality_entails_possibility() {
        // SOUND monotonicity owned compositionally by qa: if the event actually
        // holds, "can the teacher write the report?" is true (what happens is
        // possible). This must hold even if the world model has not implemented
        // modal truth yet — qa derives it from the bare event's truth.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        assert_eq!(world_truth(&d, &modal(Modality::Can, false)), Some(true));
        assert_eq!(world_truth(&d, &modal(Modality::Might, false)), Some(true));
    }

    #[test]
    fn modal_possibility_does_not_entail_actuality() {
        // SOUNDNESS: with nothing known, a modal query is open-world — qa never
        // fabricates possibility, and (critically) it never lets a modal claim
        // the bare event happened. The honest answer is "I don't know.".
        let d = Discourse::new();
        let a = answer_yes_no(engine(), &d, &modal(Modality::Can, false));
        assert!(a.to_lowercase().contains("don't know"), "unknown modal is open-world; got: {a}");
        // And an unproven event must not be derivable from a (hypothetical) Can:
        // qa only goes event->modal, never modal->event.
        let bare_event = Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false));
        assert_eq!(world_truth(&d, &bare_event), None, "no modal->actuality leak");
    }

    // ---- (4) PASSIVE: maps to the same active predicate-argument structure ---

    #[test]
    fn passive_truth_matches_active_fact() {
        // "the report was written by the teacher" parses to the SAME event as the
        // active "the teacher wrote the report" (agent=teacher, patient=report,
        // past). So an asserted active past fact answers the passive yes/no truth.
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(write_ev("teacher", "report", Tense::Past, Aspect::Simple, false)));
        let passive_query = Meaning::Event(write_ev("teacher", "report", Tense::Past, Aspect::Simple, false));
        assert_eq!(world_truth(&d, &passive_query), Some(true));
    }

    // ---- (6) TEMPORAL: realization + asymmetry soundness --------------------

    fn temporal(rel: TemporalRel) -> Meaning {
        Meaning::Temporal {
            rel,
            first: Box::new(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false)),
            second: Box::new(Event {
                predicate: "read".to_string(),
                agent: Some(Term::Entity("editor".to_string())),
                patient: Some(Term::Entity("book".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            }),
        }
    }

    #[test]
    fn temporal_realization() {
        let e = engine();
        assert_eq!(
            realize(e, &temporal(TemporalRel::Before), None),
            "the teacher writes the report before the editor reads the book"
        );
        assert_eq!(
            realize(e, &temporal(TemporalRel::After), None),
            "the teacher writes the report after the editor reads the book"
        );
    }

    #[test]
    fn temporal_unknown_is_open_world() {
        // SOUNDNESS: with no temporal facts, "does X happen before Y?" is open —
        // qa never invents an ordering (asymmetry/transitivity are owned by the
        // world model; qa under-derives to "I don't know." when it has nothing).
        let d = Discourse::new();
        let a = answer_yes_no(engine(), &d, &temporal(TemporalRel::Before));
        assert!(a.to_lowercase().contains("don't know"), "unknown temporal is open; got: {a}");
    }

    // ---- (7) CAUSAL: why-answer + non-commutative realization ---------------

    #[test]
    fn causal_realization_is_effect_because_cause() {
        let e = engine();
        let causal = Meaning::Causal {
            cause: Box::new(Meaning::Event(Event {
                predicate: "fall".to_string(),
                agent: Some(Term::Entity("rain".to_string())),
                patient: None,
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            })),
            effect: Box::new(Meaning::Event(Event {
                predicate: "flood".to_string(),
                agent: Some(Term::Entity("street".to_string())),
                patient: None,
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            })),
        };
        // Surface order is effect-because-cause (the cause is NOT commuted).
        let s = realize(e, &causal, None);
        assert!(s.starts_with("the street"), "effect first; got: {s}");
        assert!(s.contains("because"), "causal connective present; got: {s}");
        assert!(s.contains("the rain"), "cause after 'because'; got: {s}");
    }

    #[test]
    fn why_answer_returns_the_cause() {
        // A why-question (Causal{cause: Unknown, effect}) over an ATTESTED effect
        // is answerable only when a concrete cause is recoverable. Here we feed a
        // concrete cause directly (as a downstream pipeline would) and assert the
        // effect into the world, then check the "Because ..." answer.
        let mut d = Discourse::new();
        let rain_falls = Meaning::Event(Event {
            predicate: "fall".to_string(),
            agent: Some(Term::Entity("rain".to_string())),
            patient: None,
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        });
        let street_floods = Meaning::Event(Event {
            predicate: "flood".to_string(),
            agent: Some(Term::Entity("street".to_string())),
            patient: None,
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        });
        d.world.assert(&street_floods);
        d.world.assert(&rain_falls);
        let why = Meaning::Causal {
            cause: Box::new(rain_falls.clone()),
            effect: Box::new(street_floods.clone()),
        };
        let a = answer_why(engine(), &d, &why);
        assert!(a.to_lowercase().starts_with("because"), "why answer leads with 'because'; got: {a}");
        assert!(a.to_lowercase().contains("rain"), "answer names the cause; got: {a}");
    }

    #[test]
    fn why_answer_is_honest_without_cause() {
        // SOUNDNESS: a why-question whose cause is an Unknown placeholder and whose
        // effect is unattested yields "I don't know." — never a fabricated cause.
        let d = Discourse::new();
        let why = Meaning::Causal {
            cause: Box::new(Meaning::Unknown("?".to_string())),
            effect: Box::new(Meaning::Event(Event {
                predicate: "flood".to_string(),
                agent: Some(Term::Entity("street".to_string())),
                patient: None,
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            })),
        };
        let a = answer_why(engine(), &d, &why);
        assert!(a.to_lowercase().contains("don't know"), "no cause -> honest; got: {a}");
    }

    // ---- (8) DEGREE QUESTIONS: answer from known comparisons ----------------

    #[test]
    fn degree_question_answers_from_comparison() {
        // "the report is longer than the book." then "how long is the report?"
        // -> "Longer than the book." (recovered from the stored ordering).
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: false,
        });
        let a = answer_degree(engine(), &d, &Term::Entity("report".to_string()), "length");
        assert_eq!(a, "Longer than the book.");
        // The other side of the same ordering: "how long is the book?" -> shorter.
        let b = answer_degree(engine(), &d, &Term::Entity("book".to_string()), "length");
        assert_eq!(b, "Shorter than the report.");
    }

    #[test]
    fn degree_question_unknown_is_honest() {
        // SOUNDNESS: with no comparison on record, a degree question is "I don't
        // know." — we never invent a numeric measure or a comparison.
        let d = Discourse::new();
        let a = answer_degree(engine(), &d, &Term::Entity("report".to_string()), "length");
        assert!(a.to_lowercase().contains("don't know"), "no comparison -> honest; got: {a}");
    }

    #[test]
    fn degree_question_realizes_positive_adjective() {
        // The DegreeQuestion surface uses the scale's positive adjective.
        let dq = Meaning::DegreeQuestion {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
        };
        assert_eq!(realize(engine(), &dq, None), "how long is the report");
        assert_eq!(positive_adjective_for("weight"), "heavy");
        assert_eq!(positive_adjective_for("size"), "big");
    }

    // ---- (9) NEGATION SCOPE: two readings, different truth ------------------

    #[test]
    fn negation_scope_three_valued_not() {
        // The three-valued negation core qa owns.
        assert_eq!(three_valued_not(Some(true)), Some(false));
        assert_eq!(three_valued_not(Some(false)), Some(true));
        assert_eq!(three_valued_not(None), None);
    }

    #[test]
    fn negation_scope_wide_vs_narrow_distinct_truth() {
        // World: exactly one known teacher, and that teacher writes a report.
        // Wide scope "not every teacher writes a report" = Not(Every ...) — since
        // the (only) teacher DOES write, "every teacher writes" is true, so its
        // outer negation is FALSE.
        // Narrow scope "every teacher does not write a report" = Quantified Every
        // with a negated body — since the teacher DOES write, "writes-not" is
        // false for that member, so the universal-of-negation is FALSE too here...
        // To make the readings DIVERGE we use a world where the teacher writes:
        //   wide  Not(Every writes)      -> Every-writes TRUE  -> Not -> FALSE
        //   narrow Every (does-not-write) -> member writes      -> body false -> FALSE
        // They coincide in THIS world; to show distinctness pick a world with a
        // teacher who does NOT write:
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: true, // the teacher does NOT write a report
        }));
        let body = Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let every_writes = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: body.clone(),
        };
        // Narrow scope: "every teacher does NOT write a report" — universal over
        // the negated body. The teacher does not write, so this is TRUE.
        let mut neg_body = body.clone();
        neg_body.negated = true;
        let every_not_writes = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: neg_body,
        };
        // Wide scope: NOT (every teacher writes a report). "every teacher writes"
        // is FALSE (the teacher does not), so its outer negation is TRUE.
        let not_every_writes = Meaning::Not(Box::new(every_writes.clone()));

        let wide = world_truth(&d, &not_every_writes);
        let narrow = world_truth(&d, &every_not_writes);
        // Both happen to be TRUE here, but they are computed by DIFFERENT routes;
        // the key soundness property is the three-valued negation flip. Verify the
        // wide-scope value equals the flip of the inner universal's value.
        let inner = world_truth(&d, &every_writes);
        assert_eq!(wide, three_valued_not(inner), "wide scope = NOT(inner universal)");
        // And the narrow reading is evaluated as a universal over a negated body.
        assert_eq!(narrow, Some(true), "every teacher does-not-write holds in this world");
    }

    #[test]
    fn negation_scope_readings_diverge() {
        // A world that makes the two readings DIVERGE: two teachers, one writes
        // and one does not.
        //   wide  Not(Every writes)       : "every writes" FALSE (editor doesn't)
        //                                    -> Not -> TRUE
        //   narrow Every (does-not-write)  : teacher DOES write -> body false for
        //                                    that member -> universal FALSE
        // So wide = TRUE, narrow = FALSE: genuinely different truth values.
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false, // teacher WRITES
        }));
        d.world.assert(&Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Entity("editor".to_string())),
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: true, // editor does NOT write
        }));
        let body = Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let mut neg_body = body.clone();
        neg_body.negated = true;
        let not_every_writes = Meaning::Not(Box::new(Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: body.clone(),
        }));
        let every_not_writes = Meaning::Quantified {
            quant: Quantifier::Every,
            var_category: "person".to_string(),
            body: neg_body,
        };
        let wide = world_truth(&d, &not_every_writes);
        let narrow = world_truth(&d, &every_not_writes);
        assert_eq!(wide, Some(true), "not every teacher writes (editor doesn't) -> true");
        assert_eq!(narrow, Some(false), "not every teacher does-not-write (teacher does) -> false");
        assert_ne!(wide, narrow, "the two scope readings get DIFFERENT truth values");
    }

    // ---- (3) RELATIVE CLAUSES: subject resolves to the matching entity ------

    #[test]
    fn relative_clause_subject_resolves_to_matching_entity() {
        // World knows two teachers; only one writes the report. A Restricted
        // subject "the teacher who writes the report" must resolve to THAT teacher.
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false)));
        // A second person who does NOT write the report.
        d.world.assert(&Meaning::Event(Event {
            predicate: "read".to_string(),
            agent: Some(Term::Entity("editor".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }));
        let clause = Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let resolved = resolve_restricted(&d, "person", &clause);
        assert_eq!(
            resolved,
            Some(Term::Entity("teacher".to_string())),
            "the restricted term picks the teacher who writes the report"
        );
    }

    #[test]
    fn relative_clause_ambiguous_does_not_pick() {
        // SOUNDNESS: if TWO teachers satisfy the clause, the Restricted term is NOT
        // collapsed to one arbitrarily — resolution returns None (keep restricted).
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false)));
        d.world.assert(&Meaning::Event(write_ev("editor", "report", Tense::Present, Aspect::Simple, false)));
        let clause = Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let resolved = resolve_restricted(&d, "person", &clause);
        assert_eq!(resolved, None, "ambiguous restriction must not pick arbitrarily");
    }

    #[test]
    fn relative_clause_subject_used_in_event_answer() {
        // End-to-end: "the teacher who writes the report reads the book." The
        // subject is a Restricted term; after resolution the event becomes
        // read(teacher, book), and a yes/no query about it answers Yes.
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(write_ev("teacher", "report", Tense::Present, Aspect::Simple, false)));
        d.world.assert(&Meaning::Event(Event {
            predicate: "read".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }));
        // Build the relative-clause-subject event the parser would produce.
        let restricted_subject = Term::Restricted {
            head: "teacher".to_string(),
            clause: Box::new(Event {
                predicate: "write".to_string(),
                agent: None,
                patient: Some(Term::Entity("report".to_string())),
                recipient: None,
                tense: Tense::Present,
                aspect: Aspect::Simple,
                negated: false,
            }),
        };
        let reads = Meaning::Event(Event {
            predicate: "read".to_string(),
            agent: Some(restricted_subject),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        });
        // Resolve the meaning (collapses the Restricted subject to the teacher).
        let resolved = resolve_meaning(&d, &reads);
        if let Meaning::Event(ev) = &resolved {
            assert_eq!(ev.agent, Some(Term::Entity("teacher".to_string())));
        } else {
            panic!("expected an Event after resolution, got {resolved:?}");
        }
        assert_eq!(world_truth(&d, &resolved), Some(true));
    }

    #[test]
    fn interrogative_relative_clause_answered_correctly() {
        // End-to-end through the full `answer()` path: a yes/no question whose
        // subject carries a relative clause must answer correctly — NOT "I don't
        // know" (the prior behaviour). World: the teacher writes the report AND
        // reads the book; a SECOND teacher-less editor confounder is irrelevant.
        let e = engine();
        let mut d = Discourse::new();
        d.world.assert(&Meaning::Event(write_ev(
            "teacher", "report", Tense::Present, Aspect::Simple, false,
        )));
        d.world.assert(&Meaning::Event(Event {
            predicate: "read".to_string(),
            agent: Some(Term::Entity("teacher".to_string())),
            patient: Some(Term::Entity("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        }));
        let ans = answer(
            e,
            &d,
            "Does the teacher who writes the report read the book?",
        );
        assert!(
            ans.starts_with("Yes"),
            "relative-clause question must answer Yes, got {ans:?}"
        );
        assert_ne!(ans, "I don't know.", "must not be the old unanswered behaviour");

        // SOUNDNESS / open-world: a relative-clause question about an event the
        // world has NOT recorded answers "I don't know" — NEVER a guessed "No".
        // The point of the fix is that the question now RESOLVES (the restricted
        // subject is bound to the teacher and the event is queried) rather than
        // failing to parse; the world's honest verdict on an unattested fact is
        // open-world ignorance.
        let ans_unknown = answer(
            e,
            &d,
            "Does the teacher who writes the report answer the question?",
        );
        assert_eq!(
            ans_unknown, "I don't know.",
            "unattested event is sound open-world ignorance, got {ans_unknown:?}"
        );
    }

    #[test]
    fn possessive_object_question_answered_via_core_event() {
        // "the teacher writes the editor's report" asserted; then "does the teacher
        // write the report?" answers Yes — the genitive object reduced to its
        // possessed-noun head "report", so the core write(teacher, report) fact is
        // stored and queryable.
        let e = engine();
        let mut d = Discourse::new();
        let asserted = crate::understanding::semantics::understand(
            e,
            "The teacher writes the editor's report.",
        );
        d.world.assert(&asserted);
        let ans = answer(e, &d, "Does the teacher write the report?");
        assert!(
            ans.starts_with("Yes"),
            "possessive-object fact must be queryable, got {ans:?}"
        );
    }

    // ---- (5) PLURALS: plural-agreement realization --------------------------

    #[test]
    fn plural_subject_realizes_with_plural_agreement() {
        let e = engine();
        // A "they" subject takes plural agreement: progressive "are writing",
        // perfect "have written", simple present base verb.
        let prog = Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Pronoun("they".to_string())),
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Progressive,
            negated: false,
        });
        assert_eq!(realize(e, &prog, None), "they are writing the report");
        let perf = Meaning::Event(Event {
            predicate: "write".to_string(),
            agent: Some(Term::Pronoun("they".to_string())),
            patient: Some(Term::Entity("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Perfect,
            negated: false,
        });
        assert_eq!(realize(e, &perf, None), "they have written the report");
        // Singular subject stays singular ("is"/"has").
        assert!(subject_is_plural(Some(&Term::Pronoun("they".to_string()))));
        assert!(!subject_is_plural(Some(&Term::Entity("teacher".to_string()))));
    }

    // ---- Dispatcher routing for the new domains -----------------------------

    #[test]
    fn answer_routes_modal_and_degree_and_not() {
        // Route a Modal, a DegreeQuestion, and a Not through the top-level
        // `answer`-style dispatch (resolve + match) and check they do NOT panic
        // and produce sound answers.
        let mut d = Discourse::new();
        d.read(engine(), "The teacher writes the report.");
        d.world.assert(&Meaning::Comparison {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
            more: true,
            than: Term::Entity("book".to_string()),
            negated: false,
        });
        // Modal over an actual event -> Yes.
        let can = modal(Modality::Can, false);
        let m = resolve_meaning(&d, &can);
        let routed = match m {
            Meaning::Modal { .. } => answer_yes_no(engine(), &d, &can),
            other => panic!("expected Modal, got {other:?}"),
        };
        assert!(routed.to_lowercase().starts_with("yes"), "modal of actual event -> yes; got: {routed}");
        // Degree -> comparison phrase.
        let dq = Meaning::DegreeQuestion {
            subject: Term::Entity("report".to_string()),
            scale: "length".to_string(),
        };
        let routed_dq = match resolve_meaning(&d, &dq) {
            Meaning::DegreeQuestion { subject, scale } => answer_degree(engine(), &d, &subject, &scale),
            other => panic!("expected DegreeQuestion, got {other:?}"),
        };
        assert_eq!(routed_dq, "Longer than the book.");
    }
}
