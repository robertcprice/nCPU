//! Native comprehension engine: a reusable library that synthesizes the verified
//! Mog programs for English comprehension, dialogue, and logical reasoning, then
//! composes and executes them in-process. No Python, no subprocess.
//!
//! `Engine::new()` synthesizes the lexicon and rules once; the query methods run
//! the composed programs through the runtime. The `bin/comprehend` demo and the
//! C FFI in [`crate::ffi`] are both thin wrappers over this.

use std::collections::BTreeSet;

use crate::benchmark::{Example, Problem, Value};
use crate::runtime::execute_program;
use crate::solver::solve_problem;

// ---------------------------------------------------------------------------
// Inline curriculum data (baked in once; no runtime curriculum dependency).
// ---------------------------------------------------------------------------

pub const AGENTS: &[&str] = &[
    "author", "captain", "child", "doctor", "editor", "engineer", "farmer",
    "friend", "guide", "neighbor", "nurse", "officer", "painter", "pilot",
    "scientist", "singer", "student", "teacher", "tutor", "writer",
];
pub const PATIENTS: &[&str] = &[
    "article", "book", "chapter", "essay", "lesson", "letter", "memo", "note",
    "outline", "passage", "poem", "question", "report", "riddle", "story",
    "summary",
];
pub const MODIFIERS: &[&str] = &[
    "brave", "calm", "careful", "cheerful", "clever", "curious", "diligent",
    "friendly", "generous", "gentle", "honest", "humble", "kind", "patient",
    "thoughtful",
];

/// Gradable adjectives as (positive, comparative, scale). The comparative
/// surface form ("longer") and the scale name ("length") let the semantic parser
/// detect a comparative clause ("the report is longer than the book") and the
/// world model store the ordering on the right scale. Both polarities of each
/// scale point at the SAME scale name so transitive reasoning composes across
/// "longer"/"shorter" on one dimension.
pub const GRADABLE: &[(&str, &str, &str)] = &[
    ("long", "longer", "length"),
    ("short", "shorter", "length"),
    ("big", "bigger", "size"),
    ("small", "smaller", "size"),
    ("heavy", "heavier", "weight"),
    ("light", "lighter", "weight"),
    ("fast", "faster", "speed"),
    ("slow", "slower", "speed"),
];

/// Regular verbs as (base, third-singular) covering all three regular 3sg
/// allomorphs: `+s` (walk→walks), `+es` after a sibilant (watch→watches), and
/// `y→ies` after a consonant (carry→carries). Bases never end in -s, so the -s
/// detector stays cleanly suffix-separable.
pub const REG_VERBS: &[(&str, &str)] = &[
    // +s
    ("walk", "walks"), ("read", "reads"), ("write", "writes"),
    ("answer", "answers"), ("describe", "describes"), ("explain", "explains"),
    ("help", "helps"), ("open", "opens"), ("need", "needs"), ("call", "calls"),
    ("move", "moves"), ("turn", "turns"), ("pour", "pours"), ("kick", "kicks"),
    // ditransitive (+s) — give/send/show/offer/hand take a recipient
    ("give", "gives"), ("send", "sends"), ("show", "shows"),
    ("offer", "offers"), ("hand", "hands"),
    // +es after a sibilant
    ("watch", "watches"), ("wash", "washes"), ("fix", "fixes"),
    ("push", "pushes"), ("pass", "passes"), ("toss", "tosses"),
    // ditransitive (+es after sibilant ch) — teach takes a recipient
    ("teach", "teaches"),
    // y → ies after a consonant
    ("carry", "carries"), ("study", "studies"), ("copy", "copies"),
    ("try", "tries"), ("reply", "replies"), ("bury", "buries"),
];

/// Irregular 3sg verbs — suppletive/contracted forms no rule predicts. These are
/// a lexicon (stored), composed with the regular rule above.
pub const IRREGULAR_VERBS: &[(&str, &str)] = &[
    ("have", "has"), ("be", "is"), ("do", "does"), ("go", "goes"),
];

/// Regular past tense as (base, past): +ed / +d (after e) / y->ied. Excludes
/// verbs with an irregular past (write, read), which live in IRREGULAR_PAST.
pub const REG_VERBS_PAST: &[(&str, &str)] = &[
    ("walk", "walked"), ("answer", "answered"), ("describe", "described"),
    ("explain", "explained"), ("help", "helped"), ("open", "opened"),
    ("need", "needed"), ("call", "called"), ("move", "moved"), ("turn", "turned"),
    ("pour", "poured"), ("kick", "kicked"), ("watch", "watched"), ("wash", "washed"),
    ("fix", "fixed"), ("push", "pushed"), ("pass", "passed"), ("toss", "tossed"),
    ("carry", "carried"), ("study", "studied"), ("copy", "copied"), ("try", "tried"),
    ("reply", "replied"), ("bury", "buried"),
];

/// Irregular past — unpredictable forms no rule recovers (stored as a lexicon).
/// The ditransitive verbs carry their pasts here too: give→gave, send→sent,
/// tell→told, teach→taught are suppletive; show→showed is regular but kept in
/// the lexicon so `verb_past("show")` resolves without needing a REG_VERBS_PAST
/// entry.
pub const IRREGULAR_PAST: &[(&str, &str)] = &[
    ("write", "wrote"), ("read", "read"), ("go", "went"), ("do", "did"),
    ("have", "had"), ("be", "was"),
    // ditransitive pasts
    ("give", "gave"), ("send", "sent"), ("show", "showed"),
    ("tell", "told"), ("teach", "taught"),
];

/// Sentinel returned by the irregular lexicon for a regular verb ("not
/// irregular — apply the rule"). Picked so no real 3sg form collides with it.
const REGULAR_SENTINEL: &str = "-";

/// Function/auxiliary/question words — non-nouns; enough of them that "not a
/// noun" (0) is the majority label, so unseen words default to non-noun.
pub const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "not", "does", "do",
    "did", "can", "could", "will", "would", "should", "may", "might", "must",
    "always", "to", "what", "who", "why", "how", "when", "where", "yes", "no",
    "and", "or", "but", "this", "that", "it", "they", "she", "he", "of", "in",
];

/// Conditional proposition pairs (antecedent, consequent) for the reasoner.
pub const PROP_PAIRS: &[(&str, &str)] = &[
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

const CONNECTIVES: &[&str] = &["thus,", "therefore,", "so,", "hence,", "then,"];

// ---------------------------------------------------------------------------
// Example builders + synthesis.
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

fn synth(name: &str, signature: &'static str, examples: Vec<Example>) -> (String, String) {
    let problem = make_problem(name, signature, examples);
    let result = solve_problem(&problem);
    assert!(result.success, "failed to synthesize {name}: {:?}", result.error);
    (result.code, result.method)
}

fn esc(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn normalize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// The synthesized programs.
// ---------------------------------------------------------------------------

fn noun_animacy_program() -> (String, String) {
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

fn valid_roles_program() -> (String, String) {
    let combos = [([1, 12], 1), ([1, 11], 0), ([2, 12], 0), ([2, 11], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_roles", "fn valid_roles(arr: [i64]) -> i64", ex)
}

fn ends_s_program() -> (String, String) {
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

fn valid_agreement_program() -> (String, String) {
    let combos = [([1, 12], 1), ([2, 11], 1), ([1, 11], 0), ([2, 12], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_agreement", "fn valid_agreement(arr: [i64]) -> i64", ex)
}

/// Regular 3sg as a suffix-transduction rule (no irregulars, so the rule stays
/// clean and generalizes: describe→describes, not the spurious "strip be add is").
fn regular_3sg_program() -> (String, String) {
    let ex = REG_VERBS.iter().map(|(b, t)| ex_str_str(b, t)).collect();
    synth("regular_3sg", "fn regular_3sg(s: string) -> string", ex)
}

/// Irregular 3sg as a whole-word lexicon: the four suppletive verbs map to their
/// forms, every regular verb maps to the sentinel (handled by the rule). No
/// suffix transduction fits this, so the string-lexicon teacher recovers it.
fn irregular_3sg_program() -> (String, String) {
    let mut ex: Vec<Example> = IRREGULAR_VERBS.iter().map(|(b, t)| ex_str_str(b, t)).collect();
    for (b, _) in REG_VERBS {
        ex.push(ex_str_str(b, REGULAR_SENTINEL));
    }
    synth("irregular_3sg", "fn irregular_3sg(s: string) -> string", ex)
}

/// Regular past as a suffix-transduction rule (+ed / +d / y->ied).
fn regular_past_program() -> (String, String) {
    let ex = REG_VERBS_PAST.iter().map(|(b, t)| ex_str_str(b, t)).collect();
    synth("regular_past", "fn regular_past(s: string) -> string", ex)
}

/// Irregular past as a whole-word lexicon (write->wrote, read->read, go->went...);
/// regular verbs map to the sentinel so the rule handles them.
fn irregular_past_program() -> (String, String) {
    let mut ex: Vec<Example> = IRREGULAR_PAST.iter().map(|(b, t)| ex_str_str(b, t)).collect();
    for (b, _) in REG_VERBS_PAST {
        ex.push(ex_str_str(b, REGULAR_SENTINEL));
    }
    synth("irregular_past", "fn irregular_past(s: string) -> string", ex)
}

fn prop_id_program() -> (String, String) {
    let mut clauses = BTreeSet::new();
    for (a, b) in PROP_PAIRS {
        clauses.insert(*a);
        clauses.insert(*b);
    }
    let ex = clauses.iter().enumerate()
        .map(|(i, c)| ex_str_int(c, i as i64 + 1)).collect();
    synth("prop_id", "fn prop_id(s: string) -> i64", ex)
}

fn has_negation_program() -> (String, String) {
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

fn valid_argument_program() -> (String, String) {
    let combos: [(&[i64], i64); 4] = [(&[1], 1), (&[2, 3], 1), (&[2], 0), (&[1, 3], 0)];
    let mut ex = Vec::new();
    for _ in 0..4 {
        for (toks, label) in combos.iter() {
            ex.push(ex_arr_int(toks, *label));
        }
    }
    synth("valid_argument", "fn valid_argument(arr: [i64]) -> i64", ex)
}

const WRAPPERS: &str = r#"
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

/// A built comprehension engine: all programs synthesized + composed into one
/// runnable Mog source. Construct once (synthesis takes a few seconds); query
/// many times.
pub struct Engine {
    program: String,
    /// (component, teacher) for reporting which teacher recovered each program.
    pub methods: Vec<(&'static str, String)>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let (na, na_m) = noun_animacy_program();
        let (vr, vr_m) = valid_roles_program();
        let (es, es_m) = ends_s_program();
        let (ag, ag_m) = valid_agreement_program();
        let (reg, reg_m) = regular_3sg_program();
        let (irr, irr_m) = irregular_3sg_program();
        let (rpast, rpast_m) = regular_past_program();
        let (ipast, ipast_m) = irregular_past_program();
        let (pid, pid_m) = prop_id_program();
        let (neg, neg_m) = has_negation_program();
        let (arg, arg_m) = valid_argument_program();
        // Compose 3sg + past inflection: irregular lexicon first, regular rule otherwise.
        let verb_3sg_wrapper = format!(
            "fn verb_3sg(s: string) -> string {{\n    irr := irregular_3sg(s);\n    \
             if irr == \"{REGULAR_SENTINEL}\" {{ return regular_3sg(s); }}\n    return irr;\n}}\n"
        );
        let verb_past_wrapper = format!(
            "fn verb_past(s: string) -> string {{\n    irr := irregular_past(s);\n    \
             if irr == \"{REGULAR_SENTINEL}\" {{ return regular_past(s); }}\n    return irr;\n}}\n"
        );
        let program = format!(
            "{na}\n{vr}\n{es}\n{ag}\n{reg}\n{irr}\n{rpast}\n{ipast}\n{pid}\n{neg}\n{arg}\n{WRAPPERS}\n{verb_3sg_wrapper}\n{verb_past_wrapper}"
        );
        Engine {
            program,
            methods: vec![
                ("noun_animacy", na_m), ("valid_roles", vr_m), ("ends_s", es_m),
                ("valid_agreement", ag_m), ("regular_3sg", reg_m),
                ("irregular_3sg", irr_m), ("regular_past", rpast_m),
                ("irregular_past", ipast_m), ("prop_id", pid_m),
                ("has_negation", neg_m), ("valid_argument", arg_m),
            ],
        }
    }

    fn call_int(&self, call: &str) -> i64 {
        let full = format!("{}\nfn main() -> i64 {{\n  println_i64({call});\n  return 0;\n}}\n",
                           self.program);
        let out = execute_program(&full).map(|r| r.output).unwrap_or_default();
        out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(0)
    }

    fn call_str(&self, call: &str) -> String {
        let full = format!("{}\nfn main() -> i64 {{\n  println({call});\n  return 0;\n}}\n",
                           self.program);
        let out = execute_program(&full).map(|r| r.output).unwrap_or_default();
        out.lines().next().map(|l| l.trim().to_string()).unwrap_or_default()
    }

    /// Animacy class of a single word: 1 animate noun, 2 inanimate noun, 0 not a noun.
    pub fn noun_class(&self, word: &str) -> i64 {
        self.call_int(&format!("noun_animacy({})", esc(word)))
    }

    /// Is the word an animate noun (a "person")?
    pub fn is_person(&self, word: &str) -> bool {
        self.call_int(&format!("is_person({})", esc(word))) == 1
    }

    /// Is the action in the sentence semantically licensed (animate subject acting
    /// on an inanimate object)?
    pub fn comprehend_roles(&self, sentence: &str) -> bool {
        let norm = normalize(sentence).join(" ");
        self.call_int(&format!("comprehend_roles({})", esc(&norm))) == 1
    }

    /// Is the sentence grammatical in subject-verb agreement?
    pub fn check_agreement(&self, sentence: &str) -> bool {
        let norm = normalize(sentence).join(" ");
        self.call_int(&format!("check_agreement({})", esc(&norm))) == 1
    }

    /// Third-person-singular form of a verb base.
    pub fn verb_3sg(&self, base: &str) -> String {
        let v = self.call_str(&format!("verb_3sg({})", esc(base)));
        if v.is_empty() { base.to_string() } else { v }
    }

    /// Past-tense form of a verb base (regular rule + irregular lexicon).
    pub fn verb_past(&self, base: &str) -> String {
        let v = self.call_str(&format!("verb_past({})", esc(base)));
        if v.is_empty() { base.to_string() } else { v }
    }

    /// Judge a conditional argument's validity: 1 valid, 0 invalid, -1 unparseable.
    pub fn judge_argument(&self, sentence: &str) -> i64 {
        let Some((a, _b, premise, _concl)) = segment(sentence) else { return -1 };
        let a_bare = bare(&a);
        let p_is_a = self.call_int(&format!("same_prop({}, {})", esc(&bare(&premise)), esc(&a_bare)));
        let mut toks = vec![if p_is_a == 1 { 1 } else { 2 }];
        if self.call_int(&format!("has_negation({})", esc(&premise))) == 1 {
            toks.push(3);
        }
        let lit = format!("[{}]", toks.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));
        self.call_int(&format!("valid_argument({lit})"))
    }
}

// ---------------------------------------------------------------------------
// Lexing helpers for the reasoner (pure syntax — no semantic decisions).
// ---------------------------------------------------------------------------

pub fn bare(clause: &str) -> String {
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

pub fn segment(sentence: &str) -> Option<(String, String, String, String)> {
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

/// Tokenize an utterance into lowercase alphabetic words (mechanical lexing).
pub fn words_of(text: &str) -> Vec<String> {
    normalize(text)
}

pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

#[cfg(test)]
mod vocab_tests {
    use super::*;

    /// The ditransitive verbs are present in REG_VERBS with their 3sg forms, and
    /// each 3sg form is exactly the base plus a regular suffix (so the synthesized
    /// regular_3sg rule covers them without a new allomorph).
    #[test]
    fn ditransitive_verbs_in_reg_verbs() {
        for (base, f3) in [
            ("give", "gives"),
            ("send", "sends"),
            ("show", "shows"),
            ("offer", "offers"),
            ("hand", "hands"),
            ("teach", "teaches"),
        ] {
            assert!(
                REG_VERBS.contains(&(base, f3)),
                "{base} -> {f3} missing from REG_VERBS"
            );
            // The 3sg is base+"s" (give->gives) or base+"es" after a sibilant
            // (teach->teaches) — both covered by the existing regular rule.
            assert!(
                f3 == format!("{base}s") || f3 == format!("{base}es"),
                "{base} -> {f3} is not a regular 3sg suffix"
            );
        }
    }

    /// Ditransitive irregular pasts are stored in the lexicon.
    #[test]
    fn ditransitive_pasts_in_irregular_past() {
        for (base, past) in [
            ("give", "gave"),
            ("send", "sent"),
            ("show", "showed"),
            ("tell", "told"),
            ("teach", "taught"),
        ] {
            assert!(
                IRREGULAR_PAST.contains(&(base, past)),
                "{base} -> {past} missing from IRREGULAR_PAST"
            );
        }
    }

    /// "student" is an animate AGENT (a valid ditransitive recipient).
    #[test]
    fn student_is_an_agent() {
        assert!(AGENTS.contains(&"student"), "student must be an AGENT");
    }

    /// GRADABLE pairs the positive adjective with its comparative and scale, and
    /// both polarities of a scale share the same scale name so transitive
    /// reasoning composes ("longer"/"shorter" on length).
    #[test]
    fn gradable_table_well_formed() {
        let find = |pos: &str| GRADABLE.iter().find(|(p, _, _)| *p == pos).copied();
        assert_eq!(find("long"), Some(("long", "longer", "length")));
        assert_eq!(find("short"), Some(("short", "shorter", "length")));
        assert_eq!(find("big"), Some(("big", "bigger", "size")));
        assert_eq!(find("heavy"), Some(("heavy", "heavier", "weight")));
        assert_eq!(find("fast"), Some(("fast", "faster", "speed")));
        // The comparative form is distinct from the positive, and every scale
        // name is reused by at least the antonym so orderings compose.
        for (pos, comp, scale) in GRADABLE {
            assert_ne!(pos, comp, "{pos} comparative must differ from positive");
            assert!(!scale.is_empty());
        }
    }
}
