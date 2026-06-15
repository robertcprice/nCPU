//! nCPU comprehends, converses, and reasons — natively, in Rust.
//!
//! Every semantic decision is a verified Mog program the synthesizer recovers
//! from inline I/O examples; programs are composed and executed entirely
//! in-process through the runtime. There is no Python and no subprocess — the
//! whole comprehension engine lives in the crate.
//!
//!   comprehend  — selectional-restriction + subject-verb agreement over sentences
//!   converse    — a real read -> comprehend -> respond dialogue loop
//!   reason      — modus ponens / tollens validity, judged by synthesized programs
//!
//! Run:  cargo run --release --bin comprehend [comprehend|converse|reason|all]

use std::collections::BTreeSet;

use mog_synth::benchmark::{Example, Problem, Value};
use mog_synth::runtime::execute_program;
use mog_synth::solver::solve_problem;

// ---------------------------------------------------------------------------
// Inline curriculum data (baked in once; no runtime curriculum dependency).
// ---------------------------------------------------------------------------

const AGENTS: &[&str] = &[
    "author", "captain", "child", "doctor", "editor", "engineer", "farmer",
    "friend", "guide", "neighbor", "nurse", "officer", "painter", "pilot",
    "scientist", "singer", "student", "teacher", "tutor", "writer",
];
const PATIENTS: &[&str] = &[
    "article", "book", "chapter", "essay", "lesson", "letter", "memo", "note",
    "outline", "passage", "poem", "question", "report", "riddle", "story",
    "summary",
];
const MODIFIERS: &[&str] = &[
    "brave", "calm", "careful", "cheerful", "clever", "curious", "diligent",
    "friendly", "generous", "gentle", "honest", "humble", "kind", "patient",
    "thoughtful",
];
/// Regular verbs as (base, third-singular). Bases never end in -s, so the -s
/// detector is cleanly suffix-separable.
const REG_VERBS: &[(&str, &str)] = &[
    ("walk", "walks"), ("read", "reads"), ("write", "writes"),
    ("answer", "answers"), ("describe", "describes"), ("explain", "explains"),
    ("help", "helps"), ("open", "opens"), ("need", "needs"), ("call", "calls"),
    ("watch", "watches"), ("move", "moves"),
];
/// Function/auxiliary/question words — non-nouns; enough of them that "not a
/// noun" (0) is the majority label, so unseen words default to non-noun.
const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "not", "does", "do",
    "did", "can", "could", "will", "would", "should", "may", "might", "must",
    "always", "to", "what", "who", "why", "how", "when", "where", "yes", "no",
    "and", "or", "but", "this", "that", "it", "they", "she", "he", "of", "in",
];
/// Conditional proposition pairs (antecedent, consequent) for the reasoner.
const PROP_PAIRS: &[(&str, &str)] = &[
    ("the alarm rings", "the guard wakes"),
    ("the rain falls", "the street floods"),
    ("the teacher explains", "the student learns"),
    ("the light turns green", "the cars move"),
    ("the sun rises", "the birds sing"),
    ("the engine starts", "the car moves"),
    ("the door opens", "the dog runs"),
    ("the kettle boils", "the steam rises"),
    ("the bell rings", "the class begins"),
    ("the wind blows", "the leaves scatter"),
    ("the fire spreads", "the smoke rises"),
    ("the switch flips", "the lamp glows"),
    ("the clock chimes", "the workers gather"),
    ("the river rises", "the field floods"),
    ("the gate opens", "the crowd enters"),
    ("the seed sprouts", "the garden grows"),
    ("the phone buzzes", "the screen lights"),
    ("the storm arrives", "the harbor closes"),
    ("the whistle sounds", "the players run"),
    ("the oven heats", "the bread bakes"),
    ("the snow melts", "the stream swells"),
    ("the curtain falls", "the audience claps"),
];

// ---------------------------------------------------------------------------
// Synthesis + execution helpers (in-process).
// ---------------------------------------------------------------------------

fn ex_str_int(s: &str, n: i64) -> Example {
    Example { inputs: vec![Value::Str(s.to_string())], expected: Value::Int(n) }
}
fn ex_str_str(a: &str, b: &str) -> Example {
    Example { inputs: vec![Value::Str(a.to_string())], expected: Value::Str(b.to_string()) }
}
fn ex_arr_int(a: &[i64], n: i64) -> Example {
    Example { inputs: vec![Value::Array(a.to_vec())], expected: Value::Int(n) }
}

fn make_problem(name: &str, signature: &'static str, examples: Vec<Example>) -> Problem {
    Problem {
        name: name.to_string(),
        category: "comprehension",
        description: "",
        signature,
        examples,
        holdouts: Vec::new(),
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
    }
}

/// Synthesize a verified Mog program for `signature` from inline examples.
fn synth(name: &str, signature: &'static str, examples: Vec<Example>) -> (String, String) {
    let problem = make_problem(name, signature, examples);
    let result = solve_problem(&problem);
    assert!(result.success, "failed to synthesize {name}: {:?}", result.error);
    (result.code, result.method)
}

/// Run a composed program, calling `println_i64(<call>)` for each call; return ints.
fn run_ints(program: &str, calls: &[String]) -> Vec<i64> {
    let body: String = calls.iter().map(|c| format!("  println_i64({c});\n")).collect();
    let full = format!("{program}\nfn main() -> i64 {{\n{body}  return 0;\n}}\n");
    let out = execute_program(&full).expect("execute").output;
    out.lines().filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap_or(0)).collect()
}

/// Run a composed program, calling `println(<call>)` for each call; return strings.
fn run_strs(program: &str, calls: &[String]) -> Vec<String> {
    let body: String = calls.iter().map(|c| format!("  println({c});\n")).collect();
    let full = format!("{program}\nfn main() -> i64 {{\n{body}  return 0;\n}}\n");
    let out = execute_program(&full).expect("execute").output;
    out.lines().map(|l| l.trim().to_string()).collect()
}

fn esc(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// ---------------------------------------------------------------------------
// The synthesized "brain": lexicon + rules, recovered from inline examples.
// ---------------------------------------------------------------------------

fn noun_animacy() -> (String, String) {
    let mut seen = BTreeSet::new();
    let mut ex = Vec::new();
    let mut push = |w: String, label: i64, ex: &mut Vec<Example>, seen: &mut BTreeSet<String>| {
        if seen.insert(w.clone()) {
            ex.push(ex_str_int(&w, label));
        }
    };
    for w in AGENTS {
        push(w.to_string(), 1, &mut ex, &mut seen);
        push(format!("{w}s"), 1, &mut ex, &mut seen);
    }
    for w in PATIENTS {
        push(w.to_string(), 2, &mut ex, &mut seen);
        push(format!("{w}s"), 2, &mut ex, &mut seen);
    }
    for (b, t) in REG_VERBS {
        push(b.to_string(), 0, &mut ex, &mut seen);
        push(t.to_string(), 0, &mut ex, &mut seen);
    }
    for w in MODIFIERS {
        push(w.to_string(), 0, &mut ex, &mut seen);
    }
    for w in FUNCTION_WORDS {
        push(w.to_string(), 0, &mut ex, &mut seen);
    }
    synth("noun_animacy", "fn noun_animacy(s: string) -> i64", ex)
}

fn valid_roles() -> (String, String) {
    // subject animacy {1,2}, object animacy offset {11,12}; licensed iff animate
    // subject (1) AND inanimate object (12).
    let combos = [([1, 12], 1), ([1, 11], 0), ([2, 12], 0), ([2, 11], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_roles", "fn valid_roles(arr: [i64]) -> i64", ex)
}

fn ends_s() -> (String, String) {
    let mut ex = Vec::new();
    for (b, t) in REG_VERBS {
        ex.push(ex_str_int(b, 0));
        ex.push(ex_str_int(t, 1));
    }
    for (sing, plur) in [("captain", "captains"), ("editor", "editors"),
                         ("report", "reports"), ("book", "books")] {
        ex.push(ex_str_int(sing, 0));
        ex.push(ex_str_int(plur, 1));
    }
    synth("ends_s", "fn ends_s(s: string) -> i64", ex)
}

fn valid_agreement() -> (String, String) {
    // subject -s {1 none, 2 has}, verb -s offset {11 none, 12 has}; agreement is a
    // parity: exactly one carries -s. valid iff (1,12) sing+3sg or (2,11) plur+base.
    let combos = [([1, 12], 1), ([2, 11], 1), ([1, 11], 0), ([2, 12], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_agreement", "fn valid_agreement(arr: [i64]) -> i64", ex)
}

fn verb_3sg() -> (String, String) {
    let ex = REG_VERBS.iter().map(|(b, t)| ex_str_str(b, t)).collect();
    synth("verb_3sg", "fn verb_3sg(s: string) -> string", ex)
}

fn prop_id() -> (String, String) {
    let mut clauses = BTreeSet::new();
    for (a, b) in PROP_PAIRS {
        clauses.insert(*a);
        clauses.insert(*b);
    }
    let ex = clauses.iter().enumerate()
        .map(|(i, c)| ex_str_int(c, i as i64 + 1)).collect();
    synth("prop_id", "fn prop_id(s: string) -> i64", ex)
}

fn has_negation() -> (String, String) {
    let mut ex = Vec::new();
    for (a, b) in PROP_PAIRS {
        for c in [a, b] {
            ex.push(ex_str_int(c, 0));
            ex.push(ex_str_int(&format!("it is true that {c}"), 0));
            ex.push(ex_str_int(&format!("it is not true that {c}"), 1));
            ex.push(ex_str_int(&format!("{c} is not true"), 1));
        }
    }
    synth("has_negation", "fn has_negation(s: string) -> i64", ex)
}

fn valid_argument() -> (String, String) {
    // premise tokens: assertA=1 / assertB=2, assertNeg=3. valid iff
    // (assert A, no neg) modus ponens, or (assert B, neg => assert ~B) modus tollens.
    let combos: [(&[i64], i64); 4] = [(&[1], 1), (&[2, 3], 1), (&[2], 0), (&[1, 3], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_argument", "fn valid_argument(arr: [i64]) -> i64", ex)
}

// Composition wrappers (structural syntax only; every truth is a synthesized callee).
const COMPREHEND_WRAPPER: &str = r#"
fn comprehend_roles(s: string) -> i64 {
    words := s.split(" ");
    subj_c: i64 = 0;
    obj_c: i64 = 0;
    have: i64 = 0;
    for w in words {
        c := noun_animacy(w);
        if c > 0 {
            if have == 0 { subj_c = c; have = 1; }
            obj_c = c;
        }
    }
    feats := [subj_c, obj_c + 10];
    return valid_roles(feats);
}
fn check_agreement(s: string) -> i64 {
    words := s.split(" ");
    n: i64 = words.len;
    subj_idx: i64 = -1;
    i: i64 = 0;
    while i < n {
        if subj_idx == -1 {
            if noun_animacy(words[i]) > 0 { subj_idx = i; }
        }
        i = i + 1;
    }
    if subj_idx == -1 { return 1; }
    if subj_idx + 1 >= n { return 1; }
    subj_s := ends_s(words[subj_idx]);
    verb_s := ends_s(words[subj_idx + 1]);
    feats := [1 + subj_s, 11 + verb_s];
    return valid_agreement(feats);
}
fn is_person(w: string) -> i64 {
    if noun_animacy(w) == 1 { return 1; }
    return 0;
}
fn same_prop(a: string, b: string) -> i64 {
    if prop_id(a) == prop_id(b) { return 1; }
    return 0;
}
"#;

fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Demo 1: comprehension (selectional restriction + agreement).
// ---------------------------------------------------------------------------

fn demo_comprehend() {
    println!("nCPU comprehending — every semantic decision is a synthesized program:\n");
    let (na, na_m) = noun_animacy();
    let (vr, _) = valid_roles();
    let (es, es_m) = ends_s();
    let (ag, _) = valid_agreement();
    let program = format!("{na}\n{vr}\n{es}\n{ag}\n{COMPREHEND_WRAPPER}");
    println!("  noun_animacy : lexicon via {na_m}");
    println!("  valid_roles  : selectional rule (animate subject AND inanimate object)");
    println!("  ends_s       : -s detector via {es_m}\n");

    // Selectional restriction over the four animacy combinations.
    let probes = [
        ("the author writes the article", 1, "animate -> inanimate"),
        ("the article writes the author", 0, "inanimate subject"),
        ("the teacher helps the student", 0, "animate -> ANIMATE object"),
        ("the book blocks the chapter", 0, "inanimate -> inanimate"),
    ];
    let calls: Vec<String> = probes.iter()
        .map(|(s, _, _)| format!("comprehend_roles({})", esc(&normalize(s).join(" ")))).collect();
    let got = run_ints(&program, &calls);
    println!("  [selectional restriction]");
    for ((s, want, why), g) in probes.iter().zip(&got) {
        let mark = if g == want { "OK " } else { "ERR" };
        let verdict = if *g == 1 { "licensed" } else { "blocked " };
        println!("     [{mark}] {verdict}  ({why}): {s}");
    }

    // Agreement — the bug fix.
    let agree = [
        ("the captain watches the report", 1, "singular + 3sg"),
        ("the captains watch the report", 1, "plural + base"),
        ("the captains watches the report", 0, "plural + 3sg  <- old rule's BUG"),
        ("the captain watch the report", 0, "singular + base"),
    ];
    let calls: Vec<String> = agree.iter()
        .map(|(s, _, _)| format!("check_agreement({})", esc(s))).collect();
    let got = run_ints(&program, &calls);
    println!("\n  [subject-verb agreement]");
    for ((s, want, why), g) in agree.iter().zip(&got) {
        let mark = if g == want { "OK " } else { "ERR" };
        println!("     [{mark}] got={g} want={want}  ({why}): {s}");
    }
    println!("\n  \"The captains watches.\" is now {} — agreement bug fixed.",
             if got[2] == 0 { "REJECTED" } else { "still accepted" });
}

// ---------------------------------------------------------------------------
// Demo 2: dialogue (read -> comprehend -> respond, with pronoun state).
// ---------------------------------------------------------------------------

fn nouns_in(program: &str, words: &[String]) -> Vec<String> {
    let calls: Vec<String> = words.iter().map(|w| format!("noun_animacy({})", esc(w))).collect();
    let classes = run_ints(program, &calls);
    words.iter().zip(classes).filter(|(_, c)| *c > 0).map(|(w, _)| w.clone()).collect()
}

fn verb_between(program: &str, words: &[String], subj: &str) -> String {
    let start = words.iter().position(|w| w == subj).map(|i| i + 1).unwrap_or(0);
    for w in &words[start..] {
        let c = run_ints(program, &[format!("noun_animacy({})", esc(w))])[0];
        if c == 0 && w != "the" && w != "a" {
            return w.clone();
        }
    }
    "act".to_string()
}

fn respond(program: &str, last_subject: &mut Option<String>, utterance: &str) -> String {
    let raw = normalize(utterance);
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
    let nouns = nouns_in(program, &words);
    if let Some(first) = nouns.first() {
        *last_subject = Some(first.clone());
    }

    match head.as_str() {
        "is" => {
            let Some(noun) = nouns.first() else { return "I don't know that word.".into() };
            let person = run_ints(program, &[format!("is_person({})", esc(noun))])[0];
            if person == 1 {
                format!("Yes, the {noun} is a person.")
            } else {
                format!("No, the {noun} is a thing, not a person.")
            }
        }
        "can" => {
            let ok = run_ints(program, &[format!("comprehend_roles({})", esc(&clause))])[0];
            let s = nouns.first().cloned().unwrap_or_else(|| "subject".into());
            let o = nouns.last().cloned().unwrap_or_else(|| "object".into());
            let v = verb_between(program, &words, &s);
            if ok == 1 {
                format!("Yes, the {s} can {v} the {o}.")
            } else {
                format!("No — the {s} cannot {v} the {o} (a thing cannot be the doer).")
            }
        }
        "does" | "do" => {
            let ok = run_ints(program, &[format!("comprehend_roles({})", esc(&clause))])[0];
            let s = nouns.first().cloned().unwrap_or_else(|| "subject".into());
            let o = nouns.last().cloned().unwrap_or_else(|| "object".into());
            let v = verb_between(program, &words, &s);
            if ok == 1 {
                let v3 = run_strs(program, &[format!("verb_3sg({})", esc(&v))])
                    .into_iter().next().filter(|x| !x.is_empty()).unwrap_or_else(|| v.clone());
                format!("Yes, the {s} {v3} the {o}.")
            } else {
                format!("No, the {s} does not {v} the {o} — that doesn't make sense.")
            }
        }
        _ => {
            let ok = run_ints(program, &[format!("check_agreement({})", esc(&clause))])[0];
            if ok == 1 {
                return "That is grammatical.".into();
            }
            if let Some(i) = words.iter().position(|w| {
                run_ints(program, &[format!("noun_animacy({})", esc(w))])[0] > 0
            }) {
                if i + 1 < words.len() {
                    let subj = &words[i];
                    let verb = &words[i + 1];
                    let fixed_verb = if subj.ends_with('s') {
                        verb.trim_end_matches("es").trim_end_matches('s').to_string()
                    } else {
                        run_strs(program, &[format!("verb_3sg({})", esc(verb))])
                            .into_iter().next().filter(|x| !x.is_empty()).unwrap_or_else(|| verb.clone())
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

fn demo_converse() {
    println!("nCPU conversing — it reads each line, comprehends it with synthesized programs, and replies.\n");
    let (na, _) = noun_animacy();
    let (vr, _) = valid_roles();
    let (es, _) = ends_s();
    let (ag, _) = valid_agreement();
    let (v3, _) = verb_3sg();
    let (pid, _) = prop_id();
    let program = format!("{na}\n{vr}\n{es}\n{ag}\n{v3}\n{pid}\n{COMPREHEND_WRAPPER}");

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
        let reply = respond(&program, &mut last_subject, utt);
        println!("  User : {utt}");
        println!("  nCPU : {reply}\n");
    }
    println!("Every reply's truth came from a synthesized, verified Mog program. The turns were read, not generated.");
}

// ---------------------------------------------------------------------------
// Demo 3: reasoning (logical validity, synthesized parser).
// ---------------------------------------------------------------------------

const CONNECTIVES: &[&str] = &["thus,", "therefore,", "so,", "hence,", "then,"];

fn bare(clause: &str) -> String {
    let mut c = clause.trim().trim_end_matches('.').trim().to_lowercase();
    for conn in CONNECTIVES {
        if let Some(rest) = c.strip_prefix(conn) {
            c = rest.trim().to_string();
        }
    }
    for wrapper in ["it is not the case that ", "it is not true that ", "it is true that "] {
        c = c.replace(wrapper, "");
    }
    c = c.replace(" is not true", "").replace(" does not happen", "");
    c.trim().to_string()
}

fn segment(sentence: &str) -> Option<(String, String, String, String)> {
    let parts: Vec<&str> = sentence.split('.').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if parts.len() != 3 {
        return None;
    }
    let p0 = parts[0].to_lowercase();
    let rest = p0.strip_prefix("if ")?;
    let idx = rest.find(", then ")?;
    let a = rest[..idx].trim().to_string();
    let b = rest[idx + 7..].trim().to_string();
    Some((a, b, parts[1].to_lowercase(), parts[2].to_lowercase()))
}

fn judge(program: &str, arg_fn: &str, sentence: &str) -> i64 {
    let Some((a, _b, premise, _concl)) = segment(sentence) else { return -1 };
    let a_bare = bare(&a);
    let p_is_a = run_ints(program, &[format!("same_prop({}, {})", esc(&bare(&premise)), esc(&a_bare))])[0];
    let mut toks = vec![if p_is_a == 1 { 1 } else { 2 }];
    if run_ints(program, &[format!("has_negation({})", esc(&premise))])[0] == 1 {
        toks.push(3);
    }
    let lit = format!("[{}]", toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));
    run_ints(program, &[format!("{arg_fn}({lit})")])[0]
}

fn demo_reason() {
    println!("nCPU reasoning — proposition identity and negation are synthesized programs; validity is judged by them.\n");
    let (pid, pid_m) = prop_id();
    let (neg, neg_m) = has_negation();
    let (arg, arg_m) = valid_argument();
    let program = format!("{pid}\n{neg}\n{arg}\n{COMPREHEND_WRAPPER}");
    println!("  prop_id        : proposition lexicon via {pid_m}");
    println!("  has_negation   : negation cue via {neg_m}");
    println!("  valid_argument : validity rule via {arg_m}\n");

    // Build the curriculum's own argument forms and judge them.
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
            let got = judge(&program, "valid_argument", &sent);
            total += 1;
            if got == gold {
                correct += 1;
            }
            if samples.len() < 4 {
                samples.push((sent, gold, got, name.to_string()));
            }
        }
    }
    println!("  [curriculum arguments] judged {correct}/{total} validities correctly");
    for (sent, gold, got, name) in &samples {
        let mark = if got == gold { "OK " } else { "ERR" };
        let verdict = if *got == 1 { "VALID  " } else { "invalid" };
        println!("     [{mark}] {verdict}  ({name}): {sent}");
    }
    println!("\nValidity decided by synthesized programs — no Python classifier in the reasoning path.");
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match mode.as_str() {
        "comprehend" => demo_comprehend(),
        "converse" => demo_converse(),
        "reason" => demo_reason(),
        _ => {
            demo_comprehend();
            println!("\n{}\n", "=".repeat(72));
            demo_converse();
            println!("\n{}\n", "=".repeat(72));
            demo_reason();
        }
    }
}
