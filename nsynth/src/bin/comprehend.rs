//! nCPU comprehends, converses, and reasons — natively, in Rust.
//!
//! A thin demo over [`mog_synth::comprehension::Engine`], which synthesizes the
//! verified Mog programs and executes them in-process. No Python, no subprocess.
//! The same engine is exposed to C via `mog_synth::ffi`.
//!
//! Run:  cargo run --release --bin comprehend [comprehend|converse|reason|all]
//! (the solver logs to stderr; pipe 2>/dev/null for clean output)

use mog_synth::comprehension::{capitalize, words_of, Engine, PROP_PAIRS};

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
        _ => {
            demo_comprehend(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_inflect(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_converse(&engine);
            println!("\n{}\n", "=".repeat(72));
            demo_reason(&engine);
        }
    }
}
