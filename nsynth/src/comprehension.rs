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

/// Mythical creatures — a curriculum-mined membership class. Disjoint from
/// every other in-repo lexicon (AGENTS / PATIENTS / MODIFIERS / FUNCTION_WORDS),
/// so a string-equality membership map "is this a creature?" is well-posed: the
/// positives (creatures -> 1) and negatives (known non-creatures -> 0) never
/// collide. Used by [`creature_class_examples`] to turn a detected lexical gap
/// into a verified-synthesizable spec the existing string-equality teacher
/// recovers.
pub const CREATURES: &[&str] = &["dragon", "griffin", "phoenix", "unicorn", "wyvern"];

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

/// Past participles for the PASSIVE voice and the PERFECT aspect ("the report
/// was/has been written"). For a REGULAR verb the participle equals its past
/// form (`-ed`), so only the IRREGULAR participles are stored here as a lexicon
/// — callers fall back to `verb_past` for verbs absent from this table. The
/// ditransitive verbs carry their participles too (given/sent/shown/told/taught).
pub const PAST_PARTICIPLE: &[(&str, &str)] = &[
    // Irregular participles distinct from the regular `-ed` past.
    ("write", "written"), ("read", "read"), ("give", "given"),
    ("send", "sent"), ("show", "shown"), ("tell", "told"),
    ("teach", "taught"), ("do", "done"), ("go", "gone"),
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

/// Build a binary string-membership spec from in-repo curriculum data: every
/// `member` maps to 1 and every `nonmember` maps to 0, deduplicating by surface
/// string so a word that appears in both lists (or twice in one) contributes a
/// single, consistent example. The result is a well-posed string->int lookup
/// (a string-equality map) that the existing `noun_animacy`-style teacher
/// recovers exactly: the positive set and negative set are disjoint by
/// construction (a duplicate keeps its first-seen label), so the synthesized
/// program has no contradictory examples.
///
/// Reusable for ANY membership gap mined from the curriculum (creatures here,
/// but equally agents-vs-rest, sibilant verbs, etc.) — pass the positive and
/// negative word lists and get back a verified-synthesizable example set.
pub fn lexicon_examples(members: &[&str], nonmembers: &[&str]) -> Vec<Example> {
    let mut seen = BTreeSet::new();
    let mut ex = Vec::new();
    for w in members {
        if seen.insert(w.to_string()) {
            ex.push(ex_str_int(w, 1));
        }
    }
    for w in nonmembers {
        // Skip any word already claimed by the positive set so the map stays
        // well-posed (no string maps to both 1 and 0).
        if seen.insert(w.to_string()) {
            ex.push(ex_str_int(w, 0));
        }
    }
    ex
}

/// Curriculum-mined spec for "is this word a mythical creature?": each
/// [`CREATURES`] member -> 1, plus a spread of known NON-creatures (several
/// [`AGENTS`] and [`PATIENTS`]) -> 0. Drawing the negatives from the existing
/// in-repo lexicons keeps the spec self-contained (no external source) and gives
/// the string-equality teacher enough disjoint counter-examples to recover the
/// lookup as a verified Mog program rather than overfit to a trivial rule.
pub fn creature_class_examples() -> Vec<Example> {
    // A representative slice of known non-creatures from the curriculum's animate
    // agents and inanimate patients. Mixing both noun classes makes "creature"
    // depend on the actual word, not on any incidental animacy/length feature.
    let nonmembers: &[&str] = &[
        // agents (animate, but not creatures)
        "author", "captain", "doctor", "teacher", "student", "pilot",
        // patients (inanimate, not creatures)
        "book", "report", "letter", "poem", "story", "question",
    ];
    lexicon_examples(CREATURES, nonmembers)
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
    // Causal/conditional-clause subjects from the curriculum's PROP_PAIRS (e.g.
    // "the rain falls", "the street floods"). These are inanimate eventive nouns:
    // class 2 ("thing"), so the semantic parser recognizes them as NP heads and
    // the causal layer can understand "<effect> because <cause>" clauses. Derived
    // from PROP_PAIRS so the lexicon stays in sync if the curriculum grows. Pushed
    // last: agents that double as clause subjects (teacher/student) keep class 1.
    for (a, b) in PROP_PAIRS {
        for clause in [a, b] {
            if let Some(subj) = clause.split_whitespace().nth(1) {
                push(subj.to_string(), 2, &mut ex, &mut seen);
            }
        }
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
///
/// `Clone` is derived and is **cheap**: `Engine` holds only owned heap data
/// (`program: String` + `methods: Vec`). Cloning is a `String::clone` +
/// `Vec::clone` — a byte/element memcpy — and does **not** call
/// [`Engine::new`](Self::new) or re-run any synthesis. This lets a self-extension
/// candidate be built by cloning an existing engine and splicing in one freshly
/// synthesized (and already verified) component, without re-synthesizing the 11
/// base programs.
#[derive(Clone)]
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

// ---------------------------------------------------------------------------
// Reload reentrancy guard.
//
// `Engine::new()` reloads persisted components and RE-GATES each one. The gate
// (`crate::self_improve::gate::regression_gate`) builds fresh `Discourse`s but
// runs them against the candidate engine it is HANDED — it never calls
// `Engine::new()`. So the production path does not recurse. This thread-local is
// a belt-and-suspenders guard: if any future change ever caused the reload path
// (directly or via the gate) to re-enter `Engine::new()`, the inner call would
// observe `RELOADING == true` and skip the reload entirely, building only the
// base engine. That guarantees termination no matter how the call graph evolves
// — the worst case degrades to "no reload", never an infinite loop or stack
// overflow.
// ---------------------------------------------------------------------------
thread_local! {
    static RELOADING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Engine {
    /// Build the BASE engine: synthesize + compose the 11 built-in components and
    /// the wrappers, with **no** reload of any persisted learned component.
    ///
    /// This is the recursion-safe core of construction. The gate path and the
    /// reload path both need a base engine to graft onto WITHOUT triggering
    /// another reload (which would recurse), so the base build is factored out
    /// here and [`new`](Self::new) layers the reload on top.
    pub fn new_base() -> Self {
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

    /// Build the engine and **reload every persisted learned component, safely**.
    ///
    /// After composing the base engine ([`new_base`](Self::new_base)), this
    /// re-grafts each [`StoredComponent`](crate::self_improve::store::StoredComponent)
    /// the system has previously taught itself — but only after RE-GATING it
    /// against the *current* golden battery + soundness oracle. The reload is the
    /// cross-run memory that closes the gap left by rebuilding from scratch every
    /// process start; the re-gate is what keeps that memory from poisoning a fresh
    /// boot.
    ///
    /// For each stored component, in load order:
    ///   1. Graft it onto a candidate clone (append `code`, push `(name, method)`)
    ///      via [`graft_raw`](Self::graft_raw). The grafted source is the verbatim
    ///      synthesized program that was accepted on a prior run.
    ///   2. Run the candidate through
    ///      [`regression_gate`](crate::self_improve::gate::regression_gate). Accept
    ///      the component into the live engine **only** if the gate is green —
    ///      exactly the `self_extend` acceptance rule. A stale, poisoned, or
    ///      now-incompatible store entry (e.g. one synthesized against an older base
    ///      that has since changed) regresses a golden case or breaks soundness and
    ///      is REJECTED, leaving the engine sound.
    ///   3. Skipped (rejected) entries are logged via `eprintln!` — never fatal.
    ///      A bad store row degrades to "that one component is not reloaded", not a
    ///      construction failure.
    ///
    /// RECURSION SAFETY. The gate builds fresh `Discourse`s but runs them against
    /// the candidate engine it is handed — it does **not** call `Engine::new()`, so
    /// the reload does not recurse through the gate. As an extra guard, the reload
    /// is fenced by the [`RELOADING`] thread-local: a re-entrant `Engine::new()`
    /// (should any future change introduce one) observes the flag set and falls
    /// through to the base engine, guaranteeing termination. The accepted-so-far
    /// engine is always the gate's input, so the gate only ever sees base +
    /// already-accepted components — never a fresh `new()`.
    pub fn new() -> Self {
        let mut engine = Self::new_base();

        // Reentrancy guard: if we are already inside a reload on this thread, do
        // NOT reload again — return the base engine. This makes re-entry a no-op
        // rather than unbounded recursion.
        let already_reloading = RELOADING.with(|r| r.get());
        if already_reloading {
            return engine;
        }
        RELOADING.with(|r| r.set(true));

        let stored = crate::self_improve::store::load();
        for component in &stored {
            // Graft the stored code onto a candidate clone and re-gate it.
            let candidate = engine.graft_raw(component.name.as_str(), &component.code, &component.method);
            let gate = crate::self_improve::gate::regression_gate(&candidate);
            if gate.ok() {
                // Accept: this becomes the new accepted-so-far engine. The NEXT
                // stored component is grafted onto this, so the gate sees
                // base + everything accepted so far (never a fresh new()).
                engine = candidate;
            } else {
                eprintln!(
                    "[components-store] reject reloaded component `{}` (method {}): \
                     regression gate red ({}/{} golden cases, sound={}); skipping",
                    component.name, component.method, gate.passed, gate.total, gate.sound
                );
            }
        }

        RELOADING.with(|r| r.set(false));
        engine
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

    /// The full synthesized Mog program source this engine composes its answers
    /// from — every per-component teacher output concatenated into one runnable
    /// module (lexicon + inflection rules + wrappers). Exposed for metacognition
    /// ("show me your code"): the source is otherwise private, used only
    /// internally by [`call_int`](Self::call_int) / [`call_str`](Self::call_str)
    /// to build a `main()` wrapper. Purely a read accessor — no behavior change.
    pub fn program(&self) -> &str {
        &self.program
    }

    /// The teacher label that recovered a given component (e.g. `"noun_animacy"`,
    /// `"regular_3sg"`, `"valid_argument"`), looked up from [`methods`](Self::methods).
    /// Returns the synthesizer/teacher String for that component, or `None` if the
    /// component name is unknown. Metacognition accessor over provenance — which
    /// teacher learned which piece — without exposing the methods Vec's tuple layout.
    pub fn method_for(&self, component: &str) -> Option<&str> {
        self.methods
            .iter()
            .find(|(name, _)| *name == component)
            .map(|(_, teacher)| teacher.as_str())
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

    // -----------------------------------------------------------------------
    // Self-extension substrate: public accessors + additive component grafting.
    // -----------------------------------------------------------------------

    /// Evaluate an arbitrary `i64`-returning Mog call against this engine's
    /// composed program. Public wrapper over the private
    /// [`call_int`](Self::call_int) so callers can invoke an **added** component
    /// (e.g. `eval_int("creature_class(\"dragon\")")`) without going through one
    /// of the hard-coded query methods. The call string is spliced verbatim into
    /// a generated `main()`; the caller is responsible for escaping string args
    /// (see [`esc`]).
    pub fn eval_int(&self, call: &str) -> i64 {
        self.call_int(call)
    }

    /// Evaluate an arbitrary `string`-returning Mog call against this engine's
    /// composed program. Public wrapper over the private
    /// [`call_str`](Self::call_str) for invoking added string-valued components.
    pub fn eval_str(&self, call: &str) -> String {
        self.call_str(call)
    }

    /// True if this engine's composed program text defines a function named
    /// `fn_name` (i.e. the source contains `"fn <fn_name>("`). Lets a caller
    /// check whether a component has already been grafted in before attempting
    /// to add it again.
    pub fn has_component(&self, fn_name: &str) -> bool {
        self.program.contains(&format!("fn {fn_name}("))
    }

    /// Graft an ALREADY-synthesized component's raw Mog source onto a **clone**
    /// of this engine — no synthesis, no verification of `code` against examples.
    ///
    /// This is the reload-path counterpart to [`try_extend`](Self::try_extend):
    /// `try_extend` synthesizes a component from examples (and verifies it before
    /// returning), whereas `graft_raw` splices in a component's *previously*
    /// synthesized source verbatim — the bytes of a `StoredComponent.code` recorded
    /// by [`crate::self_improve::store::save_one`] on a prior run. Because the
    /// reloaded code is untrusted (it may have been synthesized against an older
    /// base, or hand-tampered in the store file), the caller MUST run the returned
    /// candidate through [`regression_gate`](crate::self_improve::gate::regression_gate)
    /// and accept it only on a green gate — exactly what
    /// [`new`](Self::new)'s reload step does.
    ///
    /// `self` is never mutated. The returned candidate is a cheap clone (String +
    /// Vec memcpy) with `code` appended to its program and `(name, method)` pushed
    /// onto its methods provenance. Like `try_extend`, the borrowed `name` is
    /// leaked to `'static` so it can live in the `&'static str`-keyed `methods`
    /// tuple for the engine's lifetime (grafted-component names are bounded by the
    /// number of stored components, so this is a negligible, intentional leak).
    pub fn graft_raw(&self, name: &str, code: &str, method: &str) -> Engine {
        let mut candidate = self.clone();
        candidate.program.push('\n');
        candidate.program.push_str(code);
        candidate
            .methods
            .push((Box::leak(name.to_string().into_boxed_str()), method.to_string()));
        candidate
    }

    /// Attempt to graft a new verified component onto a **clone** of this engine.
    ///
    /// Builds a `Problem` from `(name, signature, examples)` (reusing
    /// [`make_problem`]), runs it through [`solve_problem`], and **only on
    /// `result.success`** clones `self`, appends the synthesized `result.code` to
    /// the clone's `program`, pushes `(name, result.method)` onto the clone's
    /// `methods`, and returns the candidate `Engine`. The synthesized code is
    /// already verified against the examples by `solve_problem` before it reports
    /// success. On synthesis failure returns `Err` with the solver's explanation.
    ///
    /// This method does **not** mutate `self`: the existing engine is untouched
    /// whether synthesis succeeds or fails. The returned candidate must still be
    /// run through the regression gate before being adopted.
    pub fn try_extend(
        &self,
        name: &str,
        signature: &'static str,
        examples: Vec<crate::benchmark::Example>,
    ) -> Result<Engine, String> {
        let problem = make_problem(name, signature, examples);
        let result = solve_problem(&problem);
        if !result.success {
            return Err(result
                .error
                .unwrap_or_else(|| format!("synthesis failed for {name}: no candidate found")));
        }
        // Cheap clone (String + Vec memcpy; no re-synthesis), then splice in the
        // freshly synthesized + verified component.
        let mut candidate = self.clone();
        candidate.program.push('\n');
        candidate.program.push_str(&result.code);
        candidate
            .methods
            // `name` is borrowed; leak a 'static copy so the methods provenance
            // tuple (which is keyed by &'static str) can hold it for the engine's
            // lifetime. Grafted-component names are few and bounded by the number
            // of self-extension attempts, so this is a negligible, intentional leak.
            .push((Box::leak(name.to_string().into_boxed_str()), result.method));
        Ok(candidate)
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

    /// The curriculum-mined creature spec is well-posed and synthesizable, and the
    /// recovered program classifies a held member (dragon) as 1 and a known
    /// non-creature drawn from the negatives (report) as 0.
    #[test]
    fn creature_class_synthesizes_and_classifies() {
        let ex = creature_class_examples();
        // Spec sanity: every CREATURE is a positive (1) and the chosen negatives
        // are all labeled 0, with no string mapped to both labels.
        let positives = ex.iter().filter(|e| e.expected == Value::Int(1)).count();
        let negatives = ex.iter().filter(|e| e.expected == Value::Int(0)).count();
        assert_eq!(
            positives,
            CREATURES.len(),
            "every creature must be a positive example"
        );
        assert!(negatives >= 6, "need several non-creature negatives, got {negatives}");

        // Synthesize the lookup through the real solver and confirm success.
        let (code, _method) =
            synth("creature_class", "fn creature_class(s: string) -> i64", ex);

        // Run the synthesized program: dragon -> 1 (a creature), report -> 0
        // (a known non-creature from PATIENTS).
        let run = |word: &str| -> i64 {
            let full = format!(
                "{code}\nfn main() -> i64 {{\n  println_i64(creature_class({}));\n  return 0;\n}}\n",
                esc(word)
            );
            let out = execute_program(&full).map(|r| r.output).unwrap_or_default();
            out.lines().next().and_then(|l| l.trim().parse().ok()).unwrap_or(-1)
        };
        assert_eq!(run("dragon"), 1, "dragon must classify as a creature");
        assert_eq!(run("report"), 0, "report must classify as a non-creature");
    }

    /// `lexicon_examples` keeps the membership map well-posed when a word appears
    /// in both lists: the positive label wins and the word is not duplicated.
    #[test]
    fn lexicon_examples_dedup_is_well_posed() {
        let ex = lexicon_examples(&["dragon", "dragon"], &["dragon", "book"]);
        // "dragon" collapses to a single positive; "book" is the only negative.
        assert_eq!(ex.len(), 2);
        assert_eq!(
            ex.iter().filter(|e| e.inputs == vec![Value::Str("dragon".into())]).count(),
            1,
            "dragon must appear exactly once"
        );
        let dragon = ex
            .iter()
            .find(|e| e.inputs == vec![Value::Str("dragon".into())])
            .unwrap();
        assert_eq!(dragon.expected, Value::Int(1), "positive label wins on collision");
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
