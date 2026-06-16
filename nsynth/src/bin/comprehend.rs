//! nCPU comprehends, converses, and reasons — natively, in Rust.
//!
//! A thin demo over [`mog_synth::comprehension::Engine`], which synthesizes the
//! verified Mog programs and executes them in-process. No Python, no subprocess.
//! The same engine is exposed to C via `mog_synth::ffi`.
//!
//! Run:  cargo run --release --bin comprehend [comprehend|converse|reason|all]
//! (the solver logs to stderr; pipe 2>/dev/null for clean output)

use mog_synth::comprehension::{capitalize, words_of, Engine, PROP_PAIRS};
use mog_synth::understanding::discourse::Discourse;
use mog_synth::understanding::inference::{consequences, relation, Relation};
use mog_synth::understanding::meaning::{Event, Meaning, Quantifier, Tense, Term};
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
    }
}

/// Pretty-print a Meaning as a readable logical form.
fn show_meaning(m: &Meaning) -> String {
    match m {
        Meaning::Event(ev) => {
            let neg = if ev.negated { "¬" } else { "" };
            format!(
                "{neg}Event{{ {}({}, agent={}, patient={}) }}",
                ev.predicate,
                show_tense(ev.tense),
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
            tense: Tense::Present,
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
            tense: Tense::Present,
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

/// Render a three-valued `world.holds` verdict for the demo.
fn verdict_str(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "TRUE  (Yes)",
        Some(false) => "FALSE (No)",
        None => "UNKNOWN (don't know)",
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
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
        }
    }
}
