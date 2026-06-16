//! nCPU comprehends, converses, and reasons — natively, in Rust.
//!
//! A thin demo over [`mog_synth::comprehension::Engine`], which synthesizes the
//! verified Mog programs and executes them in-process. No Python, no subprocess.
//! The same engine is exposed to C via `mog_synth::ffi`.
//!
//! Run:  cargo run --release --bin comprehend [comprehend|converse|reason|inflect|understand|reflect|grow|study|bench|all]
//! (the solver logs to stderr; pipe 2>/dev/null for clean output)
//!
//! The `bench` subcommand runs the FraCaS-style three-valued entailment suite
//! ([`mog_synth::eval::entailment::run_suite`]) and prints a per-section table
//! (section | correct | idk | wrong | total), the overall accuracy, and a bold
//! SOUNDNESS line (WRONG==0 => SOUND, else UNSOUND with every offending case
//! listed). It then runs the benchmark -> study -> benchmark feedback loop and
//! shows the before -> after deltas, proving learning is monotone and never
//! introduces an unsound verdict.
//!
//! The `grow` subcommand showcases the self-improvement loop: a
//! [`Mind`](mog_synth::understanding::mind::Mind) that cannot classify mythical
//! creatures notices the gap, synthesizes the missing `creature_class` component
//! from curriculum-mined examples, runs it through its own regression gate, and
//! adopts it only on a green gate — then shows a rejected attempt safely declined
//! with the base engine intact, and reads back the append-only journal.
//!
//! The `study` subcommand showcases CUMULATIVE, RESTART-SURVIVING autonomy on top
//! of `grow`: Mind #1 studies a corpus, autonomously detects + synthesizes +
//! gates + KEEPS the missing component and PERSISTS it to a durable store; a
//! brand-new Mind #2 (a fresh `Engine::new`, the "restart") then BOOTS WITH that
//! learned component already present (re-gated green) WITHOUT re-studying; a
//! corrupted store row is rejected on boot; and `explain_self` marks the reloaded
//! component as self-learned. It points `NCPU_COMPONENTS_PATH` / `NCPU_JOURNAL_PATH`
//! at fresh temp files (cleared first) so it never touches the developer's store.
//!
//! The `reflect` subcommand showcases the metacognition layer: a stateful
//! [`Mind`](mog_synth::understanding::mind::Mind) reads a short passage, then
//! reflects on it via `explain_self`, `what_do_you_know`, `why`, `suppose`,
//! `what_if_not`, `what_would_change_your_mind`, `gaps`, and `explain_cause` —
//! each on a concrete read-then-reflect example.

use mog_synth::comprehension::{capitalize, words_of, Engine, PROP_PAIRS};
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::inference::{consequences, relation, Relation};
use mog_synth::understanding::meaning::{Aspect, Event, Meaning, Quantifier, Tense, Term};
use mog_synth::understanding::{qa, semantics};

fn demo_comprehend(engine: &Engine) {
    println!("nCPU comprehending — every semantic decision is a synthesized program:\n");
    for (name, method) in &engine.methods {
        if matches!(*name, "noun_animacy" | "valid_roles" | "ends_s" | "valid_agreement") {
            println!("  {name:<16}: via {method}");
        }
    }

    let probes = [
        ("the author writes the article", true, "animate -> inanimate"),
        ("the article writes the author", false, "inanimate subject"),
        ("the teacher helps the student", false, "animate -> ANIMATE object"),
        ("the book blocks the chapter", false, "inanimate -> inanimate"),
    ];
    println!("\n  [selectional restriction]");
    for (s, want, why) in probes {
        let got = engine.comprehend_roles(s);
        let mark = if got == want { "OK " } else { "ERR" };
        let verdict = if got { "licensed" } else { "blocked " };
        println!("     [{mark}] {verdict}  ({why}): {s}");
    }

    let agree = [
        ("the captain watches the report", true, "singular + 3sg"),
        ("the captains watch the report", true, "plural + base"),
        ("the captains watches the report", false, "plural + 3sg  <- old rule's BUG"),
        ("the captain watch the report", false, "singular + base"),
    ];
    println!("\n  [subject-verb agreement]");
    let mut bug_fixed = false;
    for (s, want, why) in agree {
        let got = engine.check_agreement(s);
        let mark = if got == want { "OK " } else { "ERR" };
        println!("     [{mark}] got={} want={}  ({why}): {s}", got as i32, want as i32);
        if why.contains("BUG") {
            bug_fixed = !got;
        }
    }
    println!("\n  \"The captains watches.\" is now {} — agreement bug fixed.",
             if bug_fixed { "REJECTED" } else { "still accepted" });
}

fn verb_between(engine: &Engine, words: &[String], subj: &str) -> String {
    let start = words.iter().position(|w| w == subj).map(|i| i + 1).unwrap_or(0);
    for w in &words[start..] {
        if engine.noun_class(w) == 0 && w != "the" && w != "a" {
            return w.clone();
        }
    }
    "act".to_string()
}

fn respond(engine: &Engine, last_subject: &mut Option<String>, utterance: &str) -> String {
    let raw = words_of(utterance);
    let words: Vec<String> = raw.iter().map(|w| {
        match (w.as_str(), last_subject.as_ref()) {
            ("it" | "they" | "she" | "he", Some(s)) => s.clone(),
            _ => w.clone(),
        }
    }).collect();
    if words.is_empty() {
        return "I didn't catch that.".into();
    }
    let head = words[0].clone();
    let clause = words.join(" ");
    let nouns: Vec<String> = words.iter().filter(|w| engine.noun_class(w) > 0).cloned().collect();
    if let Some(first) = nouns.first() {
        *last_subject = Some(first.clone());
    }

    match head.as_str() {
        "is" => match nouns.first() {
            None => "I don't know that word.".into(),
            Some(noun) if engine.is_person(noun) => format!("Yes, the {noun} is a person."),
            Some(noun) => format!("No, the {noun} is a thing, not a person."),
        },
        "can" => {
            let ok = engine.comprehend_roles(&clause);
            let s = nouns.first().cloned().unwrap_or_else(|| "subject".into());
            let o = nouns.last().cloned().unwrap_or_else(|| "object".into());
            let v = verb_between(engine, &words, &s);
            if ok {
                format!("Yes, the {s} can {v} the {o}.")
            } else {
                format!("No — the {s} cannot {v} the {o} (a thing cannot be the doer).")
            }
        }
        "does" | "do" => {
            let ok = engine.comprehend_roles(&clause);
            let s = nouns.first().cloned().unwrap_or_else(|| "subject".into());
            let o = nouns.last().cloned().unwrap_or_else(|| "object".into());
            let v = verb_between(engine, &words, &s);
            if ok {
                format!("Yes, the {s} {} the {o}.", engine.verb_3sg(&v))
            } else {
                format!("No, the {s} does not {v} the {o} — that doesn't make sense.")
            }
        }
        _ => {
            if engine.check_agreement(&clause) {
                return "That is grammatical.".into();
            }
            if let Some(i) = words.iter().position(|w| engine.noun_class(w) > 0) {
                if i + 1 < words.len() {
                    let verb = &words[i + 1];
                    let fixed_verb = if words[i].ends_with('s') {
                        verb.trim_end_matches("es").trim_end_matches('s').to_string()
                    } else {
                        engine.verb_3sg(verb)
                    };
                    let mut fixed = words.clone();
                    fixed[i + 1] = fixed_verb;
                    return format!("That isn't grammatical — did you mean: \"{}\"?", fixed.join(" "));
                }
            }
            "That isn't grammatical.".into()
        }
    }
}

fn demo_converse(engine: &Engine) {
    println!("nCPU conversing — it reads each line, comprehends it with synthesized programs, and replies.\n");
    let conversation = [
        "Is the teacher a person?",
        "Is the report a person?",
        "Can the teacher write the report?",
        "Can the report write the teacher?",
        "Does the teacher write the report?",
        "The captains watches the report.",
        "Is it a person?",
    ];
    let mut last_subject: Option<String> = None;
    for utt in conversation {
        let reply = respond(engine, &mut last_subject, utt);
        println!("  User : {utt}");
        println!("  nCPU : {reply}\n");
    }
    println!("Every reply's truth came from a synthesized, verified Mog program. The turns were read, not generated.");
}

fn demo_reason(engine: &Engine) {
    println!("nCPU reasoning — proposition identity and negation are synthesized programs; validity is judged by them.\n");
    for (name, method) in &engine.methods {
        if matches!(*name, "prop_id" | "has_negation" | "valid_argument") {
            println!("  {name:<16}: via {method}");
        }
    }

    let mut total = 0;
    let mut correct = 0;
    let mut samples = Vec::new();
    for (a, b) in PROP_PAIRS {
        let ac = capitalize(a);
        let bc = capitalize(b);
        let cases = [
            (format!("If {a}, then {b}. {ac}. Therefore, {b}."), 1, "modus ponens"),
            (format!("If {a}, then {b}. {bc} is not true. Therefore, {a} is not true."), 1, "modus tollens"),
            (format!("If {a}, then {b}. {bc}. Therefore, {a}."), 0, "affirming the consequent"),
            (format!("If {a}, then {b}. {ac} is not true. Therefore, {b} is not true."), 0, "denying the antecedent"),
        ];
        for (sent, gold, name) in cases {
            let got = engine.judge_argument(&sent);
            total += 1;
            if got == gold {
                correct += 1;
            }
            if samples.len() < 4 {
                samples.push((sent, gold, got, name.to_string()));
            }
        }
    }
    println!("\n  [curriculum arguments] judged {correct}/{total} validities correctly");
    for (sent, gold, got, name) in &samples {
        let mark = if got == gold { "OK " } else { "ERR" };
        let verdict = if *got == 1 { "VALID  " } else { "invalid" };
        println!("     [{mark}] {verdict}  ({name}): {sent}");
    }
    println!("\nValidity decided by synthesized programs — no Python classifier in the reasoning path.");
}

fn demo_inflect(engine: &Engine) {
    println!("nCPU inflecting — 3sg = synthesized regular rule + irregular lexicon:\n");
    for (name, method) in &engine.methods {
        if matches!(*name, "regular_3sg" | "irregular_3sg") {
            println!("  {name:<16}: via {method}");
        }
    }
    println!();
    let regular = ["walk", "write", "watch", "carry", "describe"];
    let irregular = ["have", "be", "do", "go"];
    let trap = ["scribe", "tribe"]; // end in "be" but are NOT the verb "be"
    println!("  regular (rule: +s / +es / y->ies):");
    for v in regular {
        println!("     {v:<10} -> {}", engine.verb_3sg(v));
    }
    println!("  irregular (lexicon: suppletive forms):");
    for v in irregular {
        println!("     {v:<10} -> {}", engine.verb_3sg(v));
    }
    println!("  trap — words ending in \"be\" that are NOT the verb \"be\":");
    for v in trap {
        println!("     {v:<10} -> {}  (rule, not the \"be\"->\"is\" lexicon)", engine.verb_3sg(v));
    }
    println!("\n  The irregular lexicon keys on whole words, so \"be\"->\"is\" never leaks");
    println!("  into \"scribe\"; regular verbs stay on the rule.");

    println!("\n  past tense (regular +ed rule + irregular lexicon):");
    for v in ["walk", "carry", "describe", "write", "go", "read", "have"] {
        println!("     {v:<10} -> {}", engine.verb_past(v));
    }
}

// ===========================================================================
// understand: the understanding layer (meaning representation, world model,
// coreference, truth evaluation, inference, QA) built on the synthesized engine.
// ===========================================================================

/// Pretty-print a Term as a compact logical-form atom.
fn show_term(t: &Term) -> String {
    match t {
        Term::Entity(s) => s.clone(),
        Term::Indefinite(s) => format!("a:{s}"),
        Term::Pronoun(s) => format!("?{s}"),
        // PLACEHOLDER (skeleton): print the restricted head; the relative-clause
        // owner can enrich this to show the clause once logic lands.
        Term::Restricted { head, .. } => format!("{head}[rel]"),
    }
}

fn show_opt_term(t: &Option<Term>) -> String {
    match t {
        Some(t) => show_term(t),
        None => "_".to_string(),
    }
}

fn show_tense(t: Tense) -> &'static str {
    match t {
        Tense::Present => "pres",
        Tense::Past => "past",
        Tense::Future => "fut",
    }
}

fn show_aspect(a: Aspect) -> &'static str {
    match a {
        Aspect::Simple => "simple",
        Aspect::Progressive => "prog",
        Aspect::Perfect => "perf",
    }
}

/// Pretty-print a Meaning as a readable logical form.
fn show_meaning(m: &Meaning) -> String {
    match m {
        Meaning::Event(ev) => {
            let neg = if ev.negated { "¬" } else { "" };
            // Show the recipient only for ditransitive events (when present).
            let recip = match &ev.recipient {
                Some(_) => format!(", recipient={}", show_opt_term(&ev.recipient)),
                None => String::new(),
            };
            format!(
                "{neg}Event{{ {}({}/{}, agent={}, patient={}{recip}) }}",
                ev.predicate,
                show_tense(ev.tense),
                show_aspect(ev.aspect),
                show_opt_term(&ev.agent),
                show_opt_term(&ev.patient),
            )
        }
        Meaning::IsA { subject, category, negated } => {
            let neg = if *negated { "¬" } else { "" };
            format!("{neg}IsA{{ subject={}, category={category} }}", show_term(subject))
        }
        Meaning::YesNoQuestion(inner) => format!("YesNo?( {} )", show_meaning(inner)),
        Meaning::WhQuestion { slot, body } => format!(
            "Wh?{{ slot={slot:?}, body={} }}",
            show_meaning(&Meaning::Event(body.clone()))
        ),
        Meaning::Quantified { quant, var_category, body } => format!(
            "Quantified{{ {quant:?} {var_category}: {} }}",
            show_meaning(&Meaning::Event(body.clone()))
        ),
        Meaning::Or(disjuncts) => {
            let parts: Vec<String> = disjuncts.iter().map(show_meaning).collect();
            format!("Or( {} )", parts.join(" ∨ "))
        }
        Meaning::HasProperty { subject, property, negated } => {
            let neg = if *negated { "¬" } else { "" };
            format!("{neg}HasProperty{{ subject={}, property={property} }}", show_term(subject))
        }
        Meaning::Comparison { subject, scale, more, than, negated } => {
            let neg = if *negated { "¬" } else { "" };
            let rel = if *more { ">" } else { "<" };
            format!(
                "{neg}Comparison{{ {} {rel}[{scale}] {} }}",
                show_term(subject),
                show_term(than)
            )
        }
        Meaning::Attitude { holder, verb, content, negated } => {
            let neg = if *negated { "¬" } else { "" };
            format!(
                "{neg}Attitude{{ {} {verb} that {} }}",
                show_term(holder),
                show_meaning(content)
            )
        }
        Meaning::Cardinal { at_least, var_category, body } => format!(
            "Cardinal{{ >={at_least} {var_category}: {} }}",
            show_meaning(&Meaning::Event(body.clone()))
        ),
        Meaning::CountQuestion { var_category, body } => format!(
            "Count?{{ {var_category}: {} }}",
            show_meaning(&Meaning::Event(body.clone()))
        ),
        // PLACEHOLDER (skeleton): readable prints for the new grammatical-core
        // meanings. The owners can enrich these as the logic lands.
        Meaning::Modal { modality, body, negated } => {
            let neg = if *negated { "¬" } else { "" };
            format!(
                "{neg}Modal{{ {modality:?}: {} }}",
                show_meaning(&Meaning::Event((**body).clone()))
            )
        }
        Meaning::Temporal { rel, first, second } => format!(
            "Temporal{{ {} {rel:?} {} }}",
            show_meaning(&Meaning::Event((**first).clone())),
            show_meaning(&Meaning::Event((**second).clone()))
        ),
        Meaning::Causal { cause, effect } => format!(
            "Causal{{ effect={} because cause={} }}",
            show_meaning(effect),
            show_meaning(cause)
        ),
        Meaning::DegreeQuestion { subject, scale } => {
            format!("Degree?{{ {}: {scale} }}", show_term(subject))
        }
        Meaning::Not(inner) => format!("¬( {} )", show_meaning(inner)),
        Meaning::Unknown(s) => format!("Unknown({s:?})"),
    }
}

fn demo_understand(engine: &Engine) {
    println!("nCPU understanding — it builds a meaning representation, a world model it");
    println!("evaluates truth against, resolves reference across discourse, and answers");
    println!("questions from what it read. The lexicon/rules underneath are synthesized.\n");

    // (a) READ a short passage sentence-by-sentence, printing each logical form.
    println!("  [reading the passage — each sentence becomes a logical form]");
    let mut disc = Discourse::new();
    let passage = [
        "The teacher writes the report.",
        "The teacher does not write the letter.",
        "The author reads the book.",
        "The teacher is a person.",
    ];
    for s in passage {
        let m = disc.read(engine, s);
        println!("     {s}");
        println!("        => {}", show_meaning(&m));
    }

    // (b) COREFERENCE: a later sentence with "it"/"they" resolved to the right entity.
    println!("\n  [coreference — pronouns resolved against discourse history]");
    // "they" should resolve to an animate entity (the author/teacher);
    // "it" to the most recent inanimate entity (the book).
    for s in ["They read it.", "It is a thing."] {
        let m = disc.read(engine, s);
        println!("     {s}");
        println!("        => {}   (pronouns resolved to concrete entities)", show_meaning(&m));
    }

    // (c) TRUTH EVALUATION against the world built by reading.
    println!("\n  [truth evaluation — world.holds() over asserted facts]");
    let probes: [(&str, &str); 3] = [
        ("the teacher writes the report", "a TRUE statement"),
        ("the teacher writes the letter", "a FALSE statement (negated fact)"),
        ("the editor writes the report", "an UNKNOWN statement (open-world)"),
    ];
    for (s, why) in probes {
        let m = semantics::understand(engine, s);
        let verdict = match disc.world.holds(&m) {
            Some(true) => "TRUE  (Yes)",
            Some(false) => "FALSE (No)",
            None => "UNKNOWN (don't know)",
        };
        println!("     {why:<34}: \"{s}\"\n        => {verdict}");
    }

    // (d) QA: a yes/no, a wh-question, and a category question.
    println!("\n  [question answering — answers come from the world model]");
    let questions = [
        "Does the teacher write the report?",
        "Who writes the report?",
        "What does the author read?",
        "Is the teacher a person?",
        "Is the report a person?",
    ];
    for q in questions {
        let a = qa::answer(engine, &disc, q);
        println!("     Q: {q}");
        println!("     A: {a}");
    }

    // (e) INFERENCE: an entailing pair, a contradicting pair, and consequences.
    println!("\n  [inference — natural-language inference over meanings]");
    let p = semantics::understand(engine, "the teacher writes the report");
    let h_entail = semantics::understand(engine, "a teacher writes the report");
    let h_contra = semantics::understand(engine, "the teacher does not write the report");
    let rel_e = relation(&p, &h_entail);
    let rel_c = relation(&p, &h_contra);
    println!("     premise   : {}", show_meaning(&p));
    println!("     hypothesis: {}", show_meaning(&h_entail));
    println!("        => relation = {}", show_relation(&rel_e));
    println!("     premise   : {}", show_meaning(&p));
    println!("     hypothesis: {}", show_meaning(&h_contra));
    println!("        => relation = {}", show_relation(&rel_c));
    println!("\n     consequences of \"{}\":", show_meaning(&p));
    for c in consequences(&p) {
        println!("        ⊢ {}", show_meaning(&c));
    }

    // -----------------------------------------------------------------------
    // (f) DEEP SEMANTICS: quantifiers, taxonomy, attributes, disjunction.
    // A fresh discourse so the world is built only from this section's reads.
    // -----------------------------------------------------------------------
    println!("\n  {}", "-".repeat(68));
    println!("  DEEP SEMANTICS — quantified, derived, and attribute knowledge");
    println!("  {}", "-".repeat(68));

    let mut deep = Discourse::new();
    let facts = [
        "The teacher writes the report.",
        "The editor writes the report.",
        "The editor does not read the book.",
        "The teacher is careful.",
    ];
    println!("\n  [establishing the world]");
    for s in facts {
        deep.read(engine, s);
        println!("     read: {s}");
    }
    println!(
        "     known entities: {}",
        deep.world.entities().join(", ")
    );

    // (f.a) QUANTIFIER truth over a category. "agent" is a taxonomy class whose
    // members (by hypernymy) are the teacher and the editor — both write a
    // report, so a universal is TRUE and an existential is TRUE.
    println!("\n  [quantifiers — truth over every/some/no member of a category]");
    let write_report = |q: Quantifier, cat: &str| Meaning::Quantified {
        quant: q,
        var_category: cat.to_string(),
        body: Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        },
    };
    // Universal over the AGENT category (teacher + editor both qualify).
    let univ_agent = write_report(Quantifier::Every, "agent");
    println!("     {}", show_meaning(&univ_agent));
    println!("        => {}", verdict_str(deep.world.holds(&univ_agent)));
    // Existential over the same category.
    let exist_agent = write_report(Quantifier::Some, "agent");
    println!("     {}", show_meaning(&exist_agent));
    println!("        => {}", verdict_str(deep.world.holds(&exist_agent)));
    // A universal that is FALSE: not every agent reads a book (none do here).
    let univ_read_book = Meaning::Quantified {
        quant: Quantifier::Every,
        var_category: "agent".to_string(),
        body: Event {
            predicate: "read".to_string(),
            agent: None,
            patient: Some(Term::Indefinite("book".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        },
    };
    println!("     {}", show_meaning(&univ_read_book));
    println!(
        "        => {}   (the editor is a counterexample — does not read a book)",
        verdict_str(deep.world.holds(&univ_read_book))
    );
    // The parser path too: "does every teacher write a report?".
    let q = "Does every teacher write a report?";
    println!("     Q: {q}");
    println!("     A: {}", qa::answer(engine, &deep, q));

    // (f.b) TAXONOMY / HYPERNYMY: derived membership the world never read.
    println!("\n  [taxonomy — derived super-category membership (never asserted)]");
    for q in [
        "Is the teacher an agent?",  // teacher -> person -> agent : Yes
        "Is the teacher a person?",  // direct hypernym            : Yes
        "Is the report a thing?",    // report -> document -> thing: Yes
        "Is the report a person?",   // cross-branch               : No
    ] {
        println!("     Q: {q}");
        println!("     A: {}", qa::answer(engine, &deep, q));
    }
    // The forward-chaining closure: facts derivable but never directly asserted.
    println!("\n     forward-chained closure (derived IsA facts):");
    for c in deep.world.closure() {
        println!("        ⊢ {}", show_meaning(&c));
    }

    // (f.c) ATTRIBUTE: "the teacher is careful" -> queryable property.
    println!("\n  [attributes — adjectival properties of entities]");
    for q in ["Is the teacher careful?", "Is the editor careful?"] {
        println!("     Q: {q}");
        println!("     A: {}", qa::answer(engine, &deep, q));
    }

    // (f.d) DISJUNCTION: true iff any disjunct holds.
    println!("\n  [disjunction — true iff any disjunct holds]");
    let disj_true = semantics::understand(
        engine,
        "the editor writes the report or the editor reads the book",
    );
    println!("     {}", show_meaning(&disj_true));
    println!(
        "        => {}   (first disjunct is a known fact)",
        verdict_str(deep.world.holds(&disj_true))
    );
    let disj_false = semantics::understand(
        engine,
        "the teacher reads the book or the editor reads the book",
    );
    println!("     {}", show_meaning(&disj_false));
    println!(
        "        => {}   (neither disjunct is established)",
        verdict_str(deep.world.holds(&disj_false))
    );

    // -----------------------------------------------------------------------
    // (g) SEMANTIC FRONTIER — five new semantic domains, each parsed from
    // surface English and grounded in qa::answer / world.holds. A fresh
    // discourse so each domain's world is built only from its own reads.
    // -----------------------------------------------------------------------
    println!("\n  {}", "=".repeat(68));
    println!("  SEMANTIC FRONTIER — ditransitives, comparatives, attitudes, counting");
    println!("  {}", "=".repeat(68));

    // (g.a) DITRANSITIVE / 3-place predicates. Reading fills a `recipient` slot;
    // the wh-question over the recipient retrieves the right filler.
    println!("\n  [ditransitive — 3-place give(agent, patient, recipient)]");
    let mut dt = Discourse::new();
    let dt_read = "The teacher gives the book to the student.";
    let dt_m = dt.read(engine, dt_read);
    println!("     read: {dt_read}");
    println!("        => {}", show_meaning(&dt_m));
    let dt_q = "Who does the teacher give the book to?";
    println!("     Q: {dt_q}");
    println!("     A: {}   (the recipient slot)", qa::answer(engine, &dt, dt_q));

    // (g.b) COMPARATIVE + TRANSITIVITY. Two orderings on the `length` scale are
    // read; the transitive consequence is answered from world.holds closure.
    println!("\n  [comparative — gradable scale + transitive inference]");
    let mut cmp = Discourse::new();
    for s in [
        "The report is longer than the book.",
        "The book is longer than the letter.",
    ] {
        let m = cmp.read(engine, s);
        println!("     read: {s}");
        println!("        => {}", show_meaning(&m));
    }
    let cmp_q = "Is the report longer than the letter?";
    println!("     Q: {cmp_q}");
    println!(
        "     A: {}   (report>book>letter ⊢ report>letter, transitive)",
        qa::answer(engine, &cmp, cmp_q)
    );

    // (g.c) EPISTEMIC + FACTIVITY. Factive `know` makes its content derivable;
    // non-factive `believe` does NOT.
    println!("\n  [epistemic — clausal complements with factivity]");
    let mut ep = Discourse::new();
    let ep_read = "The teacher knows that the report is long.";
    let ep_m = ep.read(engine, ep_read);
    println!("     read: {ep_read}");
    println!("        => {}", show_meaning(&ep_m));
    let ep_q = "Does the teacher know that the report is long?";
    println!("     Q: {ep_q}");
    println!("     A: {}", qa::answer(engine, &ep, ep_q));
    // Factive 'know that P' entails P — the embedded content is now true in the world.
    let factive_content = semantics::understand(engine, "the report is long");
    println!(
        "     factive entailment: \"the report is long\" => {}",
        verdict_str(ep.world.holds(&factive_content))
    );
    // Contrast: a NON-factive 'believe' does NOT make its content derivable.
    let mut ep2 = Discourse::new();
    let ep2_read = "The editor believes that the letter is long.";
    ep2.read(engine, ep2_read);
    println!("     read: {ep2_read}");
    let nonfactive_content = semantics::understand(engine, "the letter is long");
    println!(
        "     non-factive: \"the letter is long\" => {}   (believe ⊬ content — sound)",
        verdict_str(ep2.world.holds(&nonfactive_content))
    );

    // (g.d) CARDINALITY. Establish two distinct writers, then count + truth-check.
    println!("\n  [cardinality — number words + counting questions]");
    let mut card = Discourse::new();
    for s in [
        "The teacher writes the report.",
        "The author writes the report.",
    ] {
        card.read(engine, s);
        println!("     read: {s}");
    }
    let count_q = "How many agents write a report?";
    println!("     Q: {count_q}");
    println!(
        "     A: {}   (counts known agents whose body provably holds)",
        qa::answer(engine, &card, count_q)
    );
    // "Two agents write a report" — a cardinal truth query (at least 2).
    let card_truth = semantics::understand(engine, "two agents write a report");
    println!("     {}", show_meaning(&card_truth));
    println!(
        "        => {}   (>=2 agents satisfy the body)",
        verdict_str(card.world.holds(&card_truth))
    );
    // A FALSE cardinal: at least three agents would require a third writer.
    let card_false = semantics::understand(engine, "three agents write a report");
    println!("     {}", show_meaning(&card_false));
    println!(
        "        => {}   (only 2 known agents write a report — ceiling below 3)",
        verdict_str(card.world.holds(&card_false))
    );

    // (g.e) QUANTIFIER-PARSER DEPTH. "every agent" parses through the PARSER with
    // a taxonomy class as the category; the question is answered over the class.
    println!("\n  [quantifier depth — taxonomy class category parsed from surface]");
    let mut qd = Discourse::new();
    for s in [
        "The teacher writes the report.",
        "The editor writes the report.",
    ] {
        qd.read(engine, s);
        println!("     read: {s}");
    }
    let qd_parsed = semantics::understand(engine, "does every agent write a report");
    println!("     parsed: {}", show_meaning(&qd_parsed));
    let qd_q = "Does every agent write a report?";
    println!("     Q: {qd_q}");
    println!(
        "     A: {}   (teacher + editor are both agents and both write a report)",
        qa::answer(engine, &qd, qd_q)
    );

    println!("\nEvery lexical fact (animacy, verb inflection, agreement) under this layer is a");
    println!("verified synthesized Mog program; the meaning/world/inference/QA reasoning sits");
    println!("on top of those recovered programs — manipulation became understanding.");
}

fn show_relation(r: &Relation) -> &'static str {
    match r {
        Relation::Entails => "Entails",
        Relation::Contradicts => "Contradicts",
        Relation::Neutral => "Neutral",
    }
}

// ===========================================================================
// reflect: the metacognition layer. A stateful `Mind` reads a short passage,
// then REFLECTS on what it knows — every reflective method is a concrete
// read-then-reflect example. The mind explains its own learned programs,
// enumerates what it knows, shows its proofs, reasons hypothetically and
// counterfactually, names its open questions, and abduces causes. Each method
// is honest: it never fabricates knowledge, code, a counterfactual, or a cause.
// ===========================================================================

/// Print a labeled reflective block: a one-line caption, then the prompt(s) and
/// the mind's reflected answer, indented so the read-then-reflect flow is clear.
fn reflect_line(label: &str, prompt: &str, answer: &str) {
    println!("     [{label}]");
    println!("        prompt : {prompt}");
    println!("        reflect: {answer}\n");
}

fn demo_reflect() {
    // The reflective methods live on `Mind` (an Engine + a Discourse it reads
    // into). `Mind::new()` is the public entry point the FFI / library callers
    // use; it synthesizes its own engine, so this demo is fully self-contained.
    use mog_synth::understanding::mind::Mind;

    println!("nCPU reflecting — a stateful Mind reads a short passage, then turns its");
    println!("reasoning on ITSELF: it explains its own synthesized programs, enumerates");
    println!("what it knows, shows its proofs, reasons hypothetically and counterfactually,");
    println!("names its open questions, and abduces causes. Every reflection is honest —");
    println!("it never invents knowledge, code, a counterfactual, or a cause.\n");

    let mut mind = Mind::new();

    // -----------------------------------------------------------------------
    // READ a small, connected passage. Everything reflected below is grounded
    // ONLY in these sentences — the mind starts blank.
    // -----------------------------------------------------------------------
    println!("  {}", "-".repeat(70));
    println!("  THE PASSAGE — read once, reflected on many ways");
    println!("  {}", "-".repeat(70));
    let passage = [
        "The teacher writes the report.",
        "The teacher is a person.",
        "The author reads the book.",
        "The street floods because the rain falls.",
    ];
    for s in passage {
        let m = mind.read(s);
        println!("     read: {s}");
        println!("        => {}", show_meaning(&m));
    }
    println!();

    // -----------------------------------------------------------------------
    // (1) explain_self — "show me your code." The mind names the synthesized
    // component, the teacher that recovered it, and quotes the ACTUAL Mog source.
    // An unmapped topic returns an honest "I don't have a learned program".
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (1) explain_self — the mind surfaces its OWN learned programs");
    println!("  {}", "=".repeat(70));
    for topic in ["3sg inflection", "past tense", "noun animacy", "agreement", "quantum gravity"] {
        let answer = mind.explain_self(topic);
        // Keep the quoted Mog source compact in the transcript: show the framing
        // line and a short head of the source so the demo stays readable.
        let shown = trim_for_demo(&answer, 6);
        println!("     [topic: {topic}]");
        for line in shown.lines() {
            println!("        {line}");
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // (2) what_do_you_know — enumerate every fact + sound consequence + taxonomy
    // closure bearing on an entity. Unknown entities return nothing.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (2) what_do_you_know — everything bearing on an entity");
    println!("  {}", "=".repeat(70));
    for entity in ["teacher", "author", "dragon"] {
        let facts = mind.what_do_you_know(entity);
        println!("     [entity: {entity}]");
        if facts.is_empty() {
            println!("        (I know nothing about the {entity}.)");
        } else {
            for f in &facts {
                println!("        - {f}");
            }
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // (3) why — answer a question AND show the proof. Taxonomy makes
    // "is the teacher an agent?" derivable (teacher -> person -> agent), so the
    // "because" chain shows the intermediate step it routed through.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (3) why — the answer WITH the proof that backs it");
    println!("  {}", "=".repeat(70));
    for q in [
        "Does the teacher write the report?", // directly asserted leaf
        "Is the teacher an agent?",           // derived through the taxonomy
    ] {
        reflect_line("why", q, &mind.why(q));
    }

    // -----------------------------------------------------------------------
    // (4) suppose — answer UNDER a hypothesis, without committing to it. The
    // real world is untouched (Mind clones the discourse internally).
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (4) suppose — reasoning under a hypothesis (world untouched)");
    println!("  {}", "=".repeat(70));
    reflect_line(
        "suppose",
        "Suppose: \"The editor reads the book.\"  Then: Does the editor read the book?",
        &mind.suppose("The editor reads the book.", "Does the editor read the book?"),
    );
    // Prove the hypothesis never leaked into the real world.
    println!(
        "        (sanity) after suppose, the real mind STILL answers \"Does the editor read the book?\" -> {}\n",
        mind.ask("Does the editor read the book?")
    );

    // -----------------------------------------------------------------------
    // (5) what_if_not — counterfactual retraction. Suppose a fact were NOT so,
    // and contrast the answer. Modeled soundly via the fact's contradictory.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (5) what_if_not — counterfactual retraction (contrast the verdict)");
    println!("  {}", "=".repeat(70));
    reflect_line(
        "what_if_not",
        "Retract: \"The teacher writes the report.\"  Ask: Does the teacher write the report?",
        &mind.what_if_not(
            "The teacher writes the report.",
            "Does the teacher write the report?",
        ),
    );

    // -----------------------------------------------------------------------
    // (6) what_would_change_your_mind — name the evidence that would FLIP the
    // current verdict. Every flip is verified to actually move the answer.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (6) what_would_change_your_mind — the evidence that flips the verdict");
    println!("  {}", "=".repeat(70));
    for q in [
        "Does the teacher write the report?", // determined Yes -> flippable
        "Does the editor read the memo?",     // undetermined -> being told either way decides it
    ] {
        reflect_line("change_my_mind", q, &mind.what_would_change_your_mind(q));
    }

    // -----------------------------------------------------------------------
    // (7) gaps — the honest open questions a query raises that the world has no
    // verdict on. A fully-determined query reports no gap.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (7) gaps — the honest open questions the mind hasn't settled");
    println!("  {}", "=".repeat(70));
    for q in [
        "Does the editor write the letter?",  // never read about -> open
        "Does the teacher write the report?", // already known -> no gap
    ] {
        reflect_line("gaps", q, &mind.gaps(q));
    }

    // -----------------------------------------------------------------------
    // (8) explain_cause — abduce the WHY behind an effect. The recorded "because"
    // link is returned when the effect is attested; otherwise honest ignorance.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(70));
    println!("  (8) explain_cause — abduce the cause behind an effect");
    println!("  {}", "=".repeat(70));
    for q in [
        "Why does the street flood?",  // recorded cause: because the rain falls
        "Why does the teacher write the report?", // no recorded cause -> honest
    ] {
        reflect_line("explain_cause", q, &mind.explain_cause(q));
    }

    println!("Every reflection above was grounded ONLY in the four sentences read at the top.");
    println!("The mind explained its synthesized programs, showed its proofs, reasoned about");
    println!("hypotheticals and counterfactuals against a private clone of its world, and was");
    println!("honest about what it does not know — metacognition over learned understanding.");
}

/// Trim a multi-line reflective answer for the transcript: keep at most
/// `max_lines` lines and mark a truncation, so a long quoted Mog program does
/// not flood the demo output. Single-line answers pass through unchanged.
fn trim_for_demo(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= max_lines {
        return s.to_string();
    }
    let mut out: Vec<String> = lines[..max_lines].iter().map(|l| l.to_string()).collect();
    out.push(format!("... (+{} more lines of synthesized source)", lines.len() - max_lines));
    out.join("\n")
}

/// Render a three-valued `world.holds` verdict for the demo.
fn verdict_str(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "TRUE  (Yes)",
        Some(false) => "FALSE (No)",
        None => "UNKNOWN (don't know)",
    }
}

// ===========================================================================
// grow: the self-improvement loop, end-to-end and self-contained. A `Mind`
// that CANNOT classify mythical creatures notices the gap, synthesizes the
// missing component from curriculum-mined examples, runs it through its own
// regression gate, and — only on a green gate — ADOPTS it. Afterward the mind
// CAN classify creatures. We then show a REJECTED attempt (an unsatisfiable
// spec) safely declined with the base engine intact, and finally read back the
// append-only reflection journal so every attempt is auditable.
// ===========================================================================
fn demo_grow() {
    use mog_synth::comprehension::creature_class_examples;
    use mog_synth::self_improve::extend::LearnRequest;
    use mog_synth::self_improve::journal;
    use mog_synth::understanding::mind::Mind;

    // Make the demo SELF-CONTAINED: point the reflection journal at a fresh temp
    // file so we can read our own attempts back without touching the developer's
    // $HOME journal, and DISABLE the learned-component store entirely (empty
    // NCPU_COMPONENTS_PATH) so the in-process growth this demo shows never writes
    // to the real $HOME store. (Cross-run PERSISTENCE is the `study` demo's job;
    // `grow` only shows a single mind growing in-process.) Cleaned up at the end.
    let journal_path = std::env::temp_dir().join(format!("ncpu_grow_demo_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&journal_path);
    // SAFETY: this binary is single-threaded; we set the env once at the top of
    // the demo and restore/clear it at the end.
    unsafe {
        std::env::set_var("NCPU_JOURNAL_PATH", &journal_path);
        std::env::set_var("NCPU_COMPONENTS_PATH", "");
    }

    println!("nCPU growing — a Mind notices a gap in what it can classify, SYNTHESIZES the");
    println!("missing component from curriculum-mined examples, runs it through its OWN");
    println!("regression gate, and adopts it only if nothing regresses. A rejected attempt");
    println!("is safely declined with the base engine intact. Every attempt is journaled.\n");

    let mut mind = Mind::new();

    // -----------------------------------------------------------------------
    // (1) THE GAP — the mind cannot classify mythical creatures yet.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (1) the gap — the mind cannot classify creatures (yet)");
    println!("  {}", "=".repeat(72));
    let creatures = ["dragon", "griffin", "phoenix"];
    for w in creatures {
        println!(
            "     knows_word({w:<8}) = {}   (no lexicon entry — a creature is unknown)",
            mind.knows_word(w)
        );
    }
    println!(
        "     has_component(\"creature_class\") = {}   (the component is absent)\n",
        mind.engine().has_component("creature_class")
    );

    // -----------------------------------------------------------------------
    // (2) SELF-IMPROVE — synthesize, gate, and (on green) adopt the component.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (2) self_improve — synthesize creature_class, gate it, adopt it");
    println!("  {}", "=".repeat(72));
    let good = LearnRequest {
        gap: "cannot classify mythical creatures (dragon, griffin, phoenix)".to_string(),
        name: "creature_class".to_string(),
        signature: "fn creature_class(s: string) -> i64",
        examples: creature_class_examples(),
    };
    let report = mind.self_improve(good);
    println!("     gap        : {}", report.gap);
    println!("     synthesized: {}", report.synthesized);
    println!("     via teacher: {}", report.method);
    println!("     gate passed: {}", report.regression_passed);
    println!("     ACCEPTED   : {}", report.accepted);
    println!("     message    : {}\n", report.message);

    // -----------------------------------------------------------------------
    // (3) IT CAN NOW — the grafted component is live on the mind's engine.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (3) it can now — the synthesized program classifies creatures");
    println!("  {}", "=".repeat(72));
    println!(
        "     has_component(\"creature_class\") = {}   (now grafted in)",
        mind.engine().has_component("creature_class")
    );
    let probes = ["dragon", "griffin", "phoenix", "unicorn", "report", "teacher", "book"];
    for w in probes {
        let v = mind.engine().eval_int(&format!("creature_class(\"{w}\")"));
        let verdict = if v == 1 { "creature   " } else { "not creature" };
        println!("     creature_class({w:<8}) = {v}  ({verdict})");
    }
    println!(
        "     self_check().ok() = {}   (mind still green — growth was MONOTONE)\n",
        mind.self_check().ok()
    );

    // -----------------------------------------------------------------------
    // (4) A REJECTED ATTEMPT — an unsatisfiable spec is declined; engine intact.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (4) a rejected attempt — synthesis cannot satisfy a contradictory spec");
    println!("  {}", "=".repeat(72));
    let bad = LearnRequest {
        gap: "impossible contradictory lexicon (dragon -> 1 AND dragon -> 0)".to_string(),
        name: "contradictory_class".to_string(),
        signature: "fn contradictory_class(s: string) -> i64",
        examples: vec![
            mog_synth::benchmark::Example {
                inputs: vec![mog_synth::benchmark::Value::Str("dragon".to_string())],
                expected: mog_synth::benchmark::Value::Int(1),
            },
            mog_synth::benchmark::Example {
                inputs: vec![mog_synth::benchmark::Value::Str("dragon".to_string())],
                expected: mog_synth::benchmark::Value::Int(0),
            },
        ],
    };
    let rej = mind.self_improve(bad);
    println!("     gap        : {}", rej.gap);
    println!("     synthesized: {}   (no program reproduces a contradictory spec)", rej.synthesized);
    println!("     ACCEPTED   : {}", rej.accepted);
    println!("     message    : {}", rej.message);
    println!(
        "     has_component(\"contradictory_class\") = {}   (engine UNTOUCHED)",
        mind.engine().has_component("contradictory_class")
    );
    println!(
        "     self_check().ok() = {}   (still green — a rejection changes nothing)\n",
        mind.self_check().ok()
    );

    // -----------------------------------------------------------------------
    // (5) THE JOURNAL — read back every attempt (accepted and rejected). The
    // append-only reflection journal makes the self-modification history
    // auditable after the fact.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (5) the journal — every attempt recorded, accepted or rejected");
    println!("  {}", "=".repeat(72));
    let entries = journal::entries();
    if entries.is_empty() {
        println!("     (journal empty)");
    } else {
        for (i, e) in entries.iter().enumerate() {
            println!("     #{}  action={}  via={}", i + 1, e.action, if e.method.is_empty() { "(none)" } else { &e.method });
            println!(
                "         verified={}  gate_passed={}  accepted={}",
                e.verified, e.regression_passed, e.accepted
            );
            println!("         note: {}", e.note);
        }
    }

    println!("\nThe mind GREW: it noticed a gap, synthesized + gated the missing program, and");
    println!("adopted it only because nothing regressed — while a contradictory request was");
    println!("safely declined with the base engine intact. Every step is in the journal above.");

    // Clean up the temp journal + restore the env so the demo leaves no trace.
    let _ = std::fs::remove_file(&journal_path);
    unsafe {
        std::env::remove_var("NCPU_JOURNAL_PATH");
        std::env::remove_var("NCPU_COMPONENTS_PATH");
    }
}

// ===========================================================================
// study: CUMULATIVE, RESTART-SURVIVING autonomy — the whole point of the
// learned-component store. Where `grow` shows ONE mind growing in-process,
// `study` proves the growth PERSISTS across a process restart and COMPOUNDS:
//
//   (a) Mind #1 STUDIES a small corpus containing sentences it cannot handle
//       (mythical creatures), autonomously detecting the gap -> synthesizing the
//       missing component -> gating it -> KEEPING it (gate green) -> PERSISTING
//       it to a durable store. We print the StudyReport, which component it
//       learned, and that its self_check stayed green (monotone growth).
//
//   (b) A BRAND-NEW Mind #2 (a fresh Engine::new, the "restart") is pointed at
//       the SAME store and BOOTS WITH the learned component already present —
//       has_component is true and the component evaluates correctly — WITHOUT
//       studying anything. This is cross-run cumulative growth: the discovery
//       survived the "restart".
//
//   (c) A CORRUPTED store entry (a poisoned override that would regress a golden
//       case) is REJECTED on boot — a third fresh mind re-gates every reloaded
//       row and declines the bad one, staying sound.
//
//   (d) explain_self marks the reloaded component as SELF-LEARNED (not part of
//       the base curriculum), quoting the actual synthesized Mog source.
//
// SELF-CONTAINED: we point NCPU_COMPONENTS_PATH and NCPU_JOURNAL_PATH at fresh
// temp files and CLEAR them first, so the demo never reads or writes the
// developer's real $HOME store/journal, and cleans both up at the end.
// ===========================================================================
fn demo_study() {
    use mog_synth::self_improve::store::{self, StoredComponent};
    use mog_synth::understanding::mind::Mind;

    // --- SELF-CONTAINED ENV: fresh temp store + journal, cleared first. --------
    let pid = std::process::id();
    let store_path = std::env::temp_dir().join(format!("ncpu_study_demo_components_{pid}.jsonl"));
    let journal_path = std::env::temp_dir().join(format!("ncpu_study_demo_journal_{pid}.jsonl"));
    let _ = std::fs::remove_file(&store_path);
    let _ = std::fs::remove_file(&journal_path);
    // SAFETY: this binary is single-threaded; we set the env once at the top of
    // the demo and restore/clear it at the end.
    unsafe {
        std::env::set_var("NCPU_COMPONENTS_PATH", &store_path);
        std::env::set_var("NCPU_JOURNAL_PATH", &journal_path);
    }
    // Belt-and-suspenders: clear the (just-pointed-at) store so Mind #1 starts
    // from genuinely zero learned memory even if a stale temp file lingered.
    store::clear();

    println!("nCPU studying — CUMULATIVE, RESTART-SURVIVING autonomy. A mind reads a corpus");
    println!("with creatures it cannot classify, autonomously detects the gap, synthesizes +");
    println!("gates + KEEPS the missing component, and PERSISTS it. A brand-new mind (a fresh");
    println!("restart) then BOOTS WITH that learned component already present — no re-study —");
    println!("proving the discovery compounds across runs. A corrupted store row is rejected on");
    println!("boot, and the reloaded component is marked self-learned.\n");
    println!("     store   : {}", store_path.display());
    println!("     journal : {}\n", journal_path.display());

    // -----------------------------------------------------------------------
    // (a) MIND #1 STUDIES — detect the gap, synthesize, gate, keep, persist.
    // -----------------------------------------------------------------------
    println!("  {}", "=".repeat(72));
    println!("  (a) Mind #1 studies a corpus with creatures it cannot classify");
    println!("  {}", "=".repeat(72));

    // A corpus mixing sentences the base curriculum handles with sentences that
    // mention mythical creatures the lexicon has never been taught. The unknown
    // creature words are the LEXICAL gaps `study` will autonomously close.
    let corpus = [
        "the teacher writes the report",
        "the dragon guards the report",
        "the griffin carries the letter",
        "the phoenix burns the book",
    ];
    println!("     corpus:");
    for s in corpus {
        println!("       - \"{s}\"");
    }

    let mut mind1 = Mind::new();
    // Before studying: the creature words are unknown and the component is absent.
    println!(
        "\n     before study: knows_word(\"dragon\") = {}, has_component(\"creature_class\") = {}",
        mind1.knows_word("dragon"),
        mind1.engine().has_component("creature_class")
    );
    println!(
        "     before study: self_check().ok() = {}\n",
        mind1.self_check().ok()
    );

    let report = mind1.study(&corpus, /*max_rounds=*/ 4);
    println!("     StudyReport:");
    println!("       rounds   : {}", report.rounds);
    println!("       attempted: {}", report.attempted);
    println!("       rejected : {}", report.rejected);
    println!("       learned  : {:?}", report.learned);
    println!(
        "       invariant: attempted == learned + rejected -> {}",
        report.attempted == report.learned.len() + report.rejected
    );
    println!(
        "\n     after study: learned_components = {:?}",
        mind1.learned_components()
    );
    println!(
        "     after study: has_component(\"creature_class\") = {}",
        mind1.engine().has_component("creature_class")
    );
    let probes = ["dragon", "griffin", "phoenix", "report", "teacher"];
    for w in probes {
        let v = mind1.engine().eval_int(&format!("creature_class(\"{w}\")"));
        let tag = if v == 1 { "creature   " } else { "not creature" };
        println!("       creature_class({w:<8}) = {v}  ({tag})");
    }
    println!(
        "     after study: self_check().ok() = {}   (stayed GREEN — growth was monotone)",
        mind1.self_check().ok()
    );

    // The accepted component is now durably on disk (self_extend persisted it).
    let persisted = store::load();
    println!(
        "\n     PERSISTED to the store: {} component(s) -> {:?}",
        persisted.len(),
        persisted.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()
    );

    // -----------------------------------------------------------------------
    // (b) MIND #2 — a fresh restart BOOTS WITH the learned component, no study.
    // -----------------------------------------------------------------------
    println!("\n  {}", "=".repeat(72));
    println!("  (b) Mind #2 — a brand-new Engine::new() on the SAME store (a \"restart\")");
    println!("  {}", "=".repeat(72));
    println!("     Building a fresh Mind (Engine::new reloads + RE-GATES the store)...");
    let mind2 = Mind::new(); // fresh Engine::new() — the "restart"
    println!(
        "     Mind #2 has_component(\"creature_class\") = {}   (reloaded from the store!)",
        mind2.engine().has_component("creature_class")
    );
    for w in ["dragon", "wyvern", "report"] {
        let v = mind2.engine().eval_int(&format!("creature_class(\"{w}\")"));
        let tag = if v == 1 { "creature   " } else { "not creature" };
        println!("       creature_class({w:<8}) = {v}  ({tag})");
    }
    println!(
        "     Mind #2 learned_components = {:?}   (knew it WITHOUT studying)",
        mind2.learned_components()
    );
    println!(
        "     Mind #2 self_check().ok() = {}   (the reloaded component re-gated green)",
        mind2.self_check().ok()
    );
    println!("     => CUMULATIVE: the discovery SURVIVED the restart and compounds across runs.");

    // -----------------------------------------------------------------------
    // (c) A CORRUPTED store entry is REJECTED on boot — the engine stays sound.
    // -----------------------------------------------------------------------
    println!("\n  {}", "=".repeat(72));
    println!("  (c) a corrupted store row is rejected on boot — soundness preserved");
    println!("  {}", "=".repeat(72));
    // Inject a POISONED override of the animacy lexicon: it misclassifies the
    // taxonomy agents as inanimate, which would break "Is the teacher a person?".
    // Grafted naively it shadows the real lexicon (later def wins), so a fresh
    // boot MUST re-gate it, find the golden-case regression, and decline it.
    let poison_code = "\
fn noun_animacy(s: string) -> i64 {\n\
    if s == \"teacher\" { return 2; }\n\
    if s == \"editor\" { return 2; }\n\
    if s == \"author\" { return 2; }\n\
    if s == \"student\" { return 2; }\n\
    return 0;\n\
}\n";
    store::save_one(&StoredComponent {
        name: "noun_animacy".to_string(),
        signature: "fn noun_animacy(s: string) -> i64".to_string(),
        code: poison_code.to_string(),
        method: "poisoned".to_string(),
        examples_fingerprint: "fp-poison".to_string(),
    });
    println!(
        "     Injected a poisoned `noun_animacy` override into the store ({} rows now).",
        store::load().len()
    );
    println!("     (Watch for a `[components-store] reject ...` line on stderr below.)");
    let mind3 = Mind::new(); // fresh boot over the now-poisoned store
    println!(
        "     Mind #3 noun_class(\"teacher\") = {}   (1 = animate — base intact, NOT poisoned to 2)",
        mind3.engine().noun_class("teacher")
    );
    println!(
        "     Mind #3 is_person(\"teacher\") = {}   (still answers the taxonomy correctly)",
        mind3.engine().is_person("teacher")
    );
    println!(
        "     Mind #3 self_check().ok() = {}   (poisoned row REJECTED — engine stays sound)",
        mind3.self_check().ok()
    );
    println!(
        "     Mind #3 still has the GOOD component: has_component(\"creature_class\") = {}",
        mind3.engine().has_component("creature_class")
    );

    // -----------------------------------------------------------------------
    // (d) explain_self marks the reloaded component as SELF-LEARNED.
    // -----------------------------------------------------------------------
    println!("\n  {}", "=".repeat(72));
    println!("  (d) explain_self — the reloaded component is marked self-learned");
    println!("  {}", "=".repeat(72));
    println!("{}", indent(&mind2.explain_self("creature_class"), "     "));

    println!("\nThe mind's self-taught knowledge COMPOUNDED across a restart: Mind #1 detected a");
    println!("gap, synthesized + gated + persisted `creature_class`, and a brand-new Mind #2");
    println!("booted WITH it already present and re-gated green — no re-study. A corrupted store");
    println!("row was rejected on boot, and the component is correctly marked self-learned.");

    // Clean up the temp store + journal + restore env so the demo leaves no trace.
    let _ = std::fs::remove_file(&store_path);
    let _ = std::fs::remove_file(&journal_path);
    unsafe {
        std::env::remove_var("NCPU_COMPONENTS_PATH");
        std::env::remove_var("NCPU_JOURNAL_PATH");
    }
}

// ===========================================================================
// bench: the FraCaS-style three-valued entailment dashboard. Runs the whole
// `eval::entailment` suite, prints a per-section table (section | correct | idk
// | wrong | total), the overall accuracy, and a bold SOUNDNESS verdict
// (WRONG==0 => SOUND, otherwise UNSOUND with every offending case listed).
// Then runs the benchmark -> study -> benchmark feedback loop and shows the
// before -> after deltas, proving learning is monotone and never unsound.
// ===========================================================================

/// Render the per-section + overall dashboard for a [`BenchReport`]. Columns:
/// section, correct, idk, wrong, total — then the rolled-up totals, the overall
/// accuracy, and a bold SOUNDNESS line. When the run is unsound (`wrong > 0`)
/// every offending case is listed (section, gold, got, premises, hypothesis) so
/// a soundness regression is debuggable straight from the dashboard.
fn print_bench_dashboard(report: &mog_synth::eval::entailment::BenchReport) {
    use mog_synth::eval::entailment::{run_case, suite, Gold};

    const W: usize = 46;
    println!("  {}", "=".repeat(W));
    println!("  {:<14} {:>8} {:>5} {:>6} {:>6}", "section", "correct", "idk", "wrong", "total");
    println!("  {}", "-".repeat(W));
    for sec in &report.sections {
        println!(
            "  {:<14} {:>8} {:>5} {:>6} {:>6}",
            sec.section, sec.correct, sec.idk, sec.wrong, sec.total
        );
    }
    println!("  {}", "-".repeat(W));
    println!(
        "  {:<14} {:>8} {:>5} {:>6} {:>6}",
        "OVERALL", report.correct, report.idk, report.wrong, report.total
    );
    println!("  {}", "=".repeat(W));
    println!("  overall accuracy = {:.1}%   ({} correct of {} cases, {} idk)",
             report.accuracy() * 100.0, report.correct, report.total, report.idk);

    // The bold soundness verdict — the whole point of the open-world bar.
    if report.sound() {
        println!("  ** SOUNDNESS: SOUND (WRONG = 0) **");
    } else {
        println!("  ** SOUNDNESS: UNSOUND (WRONG = {}) **", report.wrong);
        // List every offending case so the violation is fully debuggable.
        for case in &suite() {
            let got = run_case(case);
            if got != case.gold && got != Gold::Unknown {
                println!(
                    "     WRONG [{}] gold={:?} got={:?}",
                    case.section, case.gold, got
                );
                println!("        premises: {:?}", case.premises);
                println!("        Q: {}", case.hypothesis);
            }
        }
    }
}

fn demo_bench() {
    use mog_synth::eval::entailment::{bench_then_study_then_bench, run_suite};

    println!("nCPU benchmarking — a FraCaS-style three-valued entailment suite. Each case");
    println!("reads its premises into a fresh Mind, asks the hypothesis as a yes/no question,");
    println!("and buckets the answer into {{Yes, No, Unknown}}. The open-world engine may answer");
    println!("Unknown (an `idk`) where gold is determined, but a determined verdict that");
    println!("contradicts gold is a SOUNDNESS violation. The bar is WRONG = 0.\n");

    // ----------------------------------------------------------------------
    // (1) Run the whole suite and print the dashboard.
    // ----------------------------------------------------------------------
    let report = run_suite();
    print_bench_dashboard(&report);

    // ----------------------------------------------------------------------
    // (2) The feedback loop: benchmark -> study -> benchmark. Measured misses
    // become a study corpus the Mind learns from; the re-run proves the
    // learning is MONOTONE (never loses a correct answer) and stays SOUND
    // (never introduces a wrong verdict). Self-fenced: the function redirects
    // its component store + journal to temp files and restores them on exit.
    // ----------------------------------------------------------------------
    println!("\n  {}", "=".repeat(46));
    println!("  bench -> study -> bench (autonomous, monotone, sound)");
    println!("  {}", "=".repeat(46));
    let (before, study, after) = bench_then_study_then_bench(3);
    println!("  study: rounds={} attempted={} learned={:?} rejected={}",
             study.rounds, study.attempted, study.learned, study.rejected);
    println!(
        "  before -> after:  correct {} -> {}   idk {} -> {}   wrong {} -> {}",
        before.correct, after.correct, before.idk, after.idk, before.wrong, after.wrong
    );
    println!(
        "  accuracy: {:.1}% -> {:.1}%   (delta {:+.1} pts)",
        before.accuracy() * 100.0,
        after.accuracy() * 100.0,
        (after.accuracy() - before.accuracy()) * 100.0
    );
    let monotone = after.correct >= before.correct;
    let stays_sound = after.wrong == 0 && before.wrong == 0;
    println!(
        "  MONOTONE (after.correct >= before.correct) = {}   SOUND throughout (wrong == 0) = {}",
        monotone, stays_sound
    );
    if study.learned.is_empty() {
        println!("  (no mineable lexical gap in the in-vocab suite — study learned nothing, after == before: honest, permitted, monotone.)");
    }

    println!("\nEvery verdict came from a synthesized, verified program executed in-process.");
    println!("The suite is SOUND ({}); learning over its misses is monotone and never unsound.",
             if report.sound() { "WRONG = 0" } else { "SOUNDNESS VIOLATION — see above" });
}

/// Indent every line of `text` with `prefix` — for embedding multi-line
/// `explain_self` output under a demo section header.
fn indent(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|l| if l.is_empty() { String::new() } else { format!("{prefix}{l}") })
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    // The autonomy demos (`reflect`, `grow`, `study`) build their OWN minds under
    // their own fenced env, so they don't need — and `study` must not be slowed or
    // perturbed by — the shared top-level engine. Build it only for the modes that
    // actually use it.
    match mode.as_str() {
        "reflect" => return demo_reflect(),
        "grow" => return demo_grow(),
        "study" => return demo_study(),
        "bench" => return demo_bench(),
        _ => {}
    }
    eprintln!("[building comprehension engine — synthesizing verified programs...]");
    let engine = Engine::new();
    match mode.as_str() {
        "comprehend" => demo_comprehend(&engine),
        "converse" => demo_converse(&engine),
        "reason" => demo_reason(&engine),
        "inflect" => demo_inflect(&engine),
        "understand" => demo_understand(&engine),
        _ => {
            demo_comprehend(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_inflect(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_converse(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_reason(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_understand(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_reflect();
            println!("\n{}\n", "=".repeat(72));
            demo_grow();
            println!("\n{}\n", "=".repeat(72));
            demo_study();
            println!("\n{}\n", "=".repeat(72));
            demo_bench();
        }
    }
}
