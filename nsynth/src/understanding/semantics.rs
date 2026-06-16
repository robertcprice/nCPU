//! Semantic parser: maps a raw sentence to its logical form (`Meaning`).
//!
//! THE CORE of the understanding layer. Handles declarative events, negated
//! events, copular/category statements, yes/no questions, and wh-questions.
//! Subject = first noun (noun_class > 0), verb = word after the subject,
//! object = last noun. Determiner "the" -> Entity, "a"/"an" -> Indefinite,
//! pronoun it/they/he/she -> Pronoun. Verb lemma = de-inflect (strip
//! -s/-es/-ies or reverse the irregular form). Lexical facts come from the
//! synthesized Engine (noun_class, is_person, verb_3sg), never hardcoded.

use crate::comprehension::{words_of, Engine, IRREGULAR_VERBS, REG_VERBS};
use crate::understanding::meaning::{Event, Meaning, Role, Tense, Term};

/// Map a raw sentence to its logical form.
///
/// Strategy: tokenize mechanically, classify the leading word to decide
/// question vs. declarative, then locate subject (first noun), verb (the word
/// after the subject), and object (last noun after the verb). Determiner and
/// pronoun handling produce the right `Term` variant; the verb is de-inflected
/// to its lemma by reusing the Engine's synthesized inflection program.
pub fn understand(engine: &Engine, sentence: &str) -> Meaning {
    let toks = words_of(sentence);
    if toks.is_empty() {
        return Meaning::Unknown(sentence.to_string());
    }

    let head = toks[0].as_str();

    // --- Question detection (purely structural: leading word) -----------------
    match head {
        // wh-questions ------------------------------------------------------
        "who" => return parse_who(engine, &toks, sentence),
        "what" => return parse_what(engine, &toks, sentence),
        // yes/no question: "does/do [not] X verb Y?" -----------------------
        "does" | "do" | "did" | "is" | "are" | "was" | "were" => {
            return parse_yes_no(engine, &toks, sentence);
        }
        _ => {}
    }

    // --- Declarative ----------------------------------------------------------
    parse_declarative(engine, &toks, sentence)
}

// ===========================================================================
// Declarative parsing
// ===========================================================================

/// Parse a declarative sentence: a copular "X is a/an C" / "X is C" statement
/// (-> IsA) or an action "X verb Y" (-> Event).
fn parse_declarative(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    // Locate the subject: first token that is a noun (noun_class > 0).
    let Some(subj_idx) = first_noun_idx(engine, toks, 0) else {
        return Meaning::Unknown(original.to_string());
    };
    let subject = term_from(toks, subj_idx);

    // Find the main verb / copula: the first verb-ish token after the subject.
    // We allow auxiliaries ("does"/"do"/"did") and negation ("not") between the
    // subject and the lexical verb.
    let after = subj_idx + 1;

    // Copular statement: "<subject> is/are/was/were [not] [a/an] <category>".
    if let Some(cop_idx) = find_copula(toks, after) {
        return build_isa(engine, toks, subject, cop_idx);
    }

    // Action event.
    build_event_declarative(engine, toks, subject, after, original)
}

/// Build an `IsA` meaning from a copular statement.
/// `cop_idx` is the index of the copula ("is"/"are"/"was"/"were").
fn build_isa(engine: &Engine, toks: &[String], subject: Term, cop_idx: usize) -> Meaning {
    let mut i = cop_idx + 1;
    let mut negated = false;
    if i < toks.len() && toks[i] == "not" {
        negated = true;
        i += 1;
    }
    // Skip an indefinite article in "is a person" / "is an editor".
    if i < toks.len() && (toks[i] == "a" || toks[i] == "an") {
        i += 1;
    }
    // The category is the head word (a noun like "person"/"thing", or an
    // adjective/property). Prefer the last meaningful word.
    let category = if i < toks.len() {
        // Use the last remaining content token as the category head.
        let last = toks.len() - 1;
        toks[last].clone()
    } else {
        // Degenerate "X is" — fall back to animacy-derived category.
        animacy_category(engine, subject.head())
    };

    Meaning::IsA {
        subject,
        category,
        negated,
    }
}

/// Build an `Event` meaning from a declarative action sentence.
/// `after` is the index just past the subject.
fn build_event_declarative(
    engine: &Engine,
    toks: &[String],
    subject: Term,
    after: usize,
    original: &str,
) -> Meaning {
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, after);

    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };

    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    // An auxiliary "did" forces past; "does"/"do" forces present. Otherwise
    // the surface inflection of the lexical verb decides.
    let tense = tense_hint.unwrap_or(surface_tense);

    // Object = last noun after the verb (if any).
    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Meaning::Event(Event {
        predicate,
        agent: Some(subject),
        patient,
        tense,
        negated,
    })
}

// ===========================================================================
// Yes/no question parsing: "does/do/did [not] X verb Y?"  or "is X a C?"
// ===========================================================================

fn parse_yes_no(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    let head = toks[0].as_str();

    // Copular yes/no: "is the teacher a person?" / "is X C?"
    if matches!(head, "is" | "are" | "was" | "were") {
        // Subject is the first noun after the copula.
        if let Some(subj_idx) = first_noun_idx(engine, toks, 1) {
            let subject = term_from(toks, subj_idx);
            // Look for a category after the subject.
            let mut i = subj_idx + 1;
            let mut negated = false;
            if i < toks.len() && toks[i] == "not" {
                negated = true;
                i += 1;
            }
            if i < toks.len() && (toks[i] == "a" || toks[i] == "an") {
                i += 1;
            }
            if i < toks.len() {
                let last = toks.len() - 1;
                let category = toks[last].clone();
                return Meaning::YesNoQuestion(Box::new(Meaning::IsA {
                    subject,
                    category,
                    negated,
                }));
            }
        }
        return Meaning::Unknown(original.to_string());
    }

    // Auxiliary yes/no: "does the teacher write the report?"
    // Subject = first noun after the auxiliary.
    let Some(subj_idx) = first_noun_idx(engine, toks, 1) else {
        return Meaning::Unknown(original.to_string());
    };
    let subject = term_from(toks, subj_idx);

    // Tense from the auxiliary.
    let aux_tense = match head {
        "did" => Tense::Past,
        _ => Tense::Present,
    };

    let after = subj_idx + 1;
    let (negated, _t, verb_idx) = scan_aux_negation(toks, after);
    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };

    // In a "does X verb Y?" question the lexical verb is bare; de-inflect it
    // anyway (handles "does the teacher writes" robustly) and use aux tense.
    let (predicate, _) = lemma_and_tense(engine, &toks[vidx]);

    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Meaning::YesNoQuestion(Box::new(Meaning::Event(Event {
        predicate,
        agent: Some(subject),
        patient,
        tense: aux_tense,
        negated,
    })))
}

// ===========================================================================
// wh-question parsing
// ===========================================================================

/// "who writes the report?" -> WhQuestion{ slot: Agent, body: Event(write, patient=report) }
fn parse_who(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    // After "who" comes the verb (possibly with negation), then the object.
    let after = 1;
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, after);
    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = tense_hint.unwrap_or(surface_tense);

    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Meaning::WhQuestion {
        slot: Role::Agent,
        body: Event {
            predicate,
            agent: None,
            patient,
            tense,
            negated,
        },
    }
}

/// "what does the teacher write?" -> WhQuestion{ slot: Patient, body: Event(write, agent=teacher) }
/// "what writes the report?" (rare) also handled -> slot Patient, agent None.
fn parse_what(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    // Two shapes:
    //   (a) "what does/do/did <subject> <verb>?"  (object is the wh-slot)
    //   (b) "what <verb> <subject>?"              (subject inverted; rare)
    let mut i = 1;

    // Optional auxiliary right after "what".
    let aux_tense = match toks.get(i).map(|s| s.as_str()) {
        Some("does") | Some("do") => {
            i += 1;
            Some(Tense::Present)
        }
        Some("did") => {
            i += 1;
            Some(Tense::Past)
        }
        _ => None,
    };

    // Subject = first noun from i.
    let agent = first_noun_idx(engine, toks, i).map(|si| term_from(toks, si));

    // Verb = first verb after the subject (or after the auxiliary if no subject
    // noun was found before it).
    let scan_from = match first_noun_idx(engine, toks, i) {
        Some(si) => si + 1,
        None => i,
    };
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, scan_from);
    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = aux_tense.or(tense_hint).unwrap_or(surface_tense);

    Meaning::WhQuestion {
        slot: Role::Patient,
        body: Event {
            predicate,
            agent,
            patient: None,
            tense,
            negated,
        },
    }
}

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Is this token a noun-phrase head — either a lexicon noun (noun_class > 0) or
/// a pronoun (it/they/he/she/...)? Pronouns are valid argument heads whose
/// referent discourse coreference resolves later.
fn is_np_head(engine: &Engine, word: &str) -> bool {
    engine.noun_class(word) > 0 || is_pronoun(word)
}

/// Index of the first noun-phrase head (lexicon noun or pronoun) at or after
/// `from`.
fn first_noun_idx(engine: &Engine, toks: &[String], from: usize) -> Option<usize> {
    (from..toks.len()).find(|&i| is_np_head(engine, &toks[i]))
}

/// Index of the last noun-phrase head (lexicon noun or pronoun) at or after
/// `from`.
fn last_noun_idx(engine: &Engine, toks: &[String], from: usize) -> Option<usize> {
    (from..toks.len()).rev().find(|&i| is_np_head(engine, &toks[i]))
}

/// Build a `Term` from a noun token, consulting the determiner immediately
/// before it. "the" -> Entity; "a"/"an" -> Indefinite; a pronoun token itself
/// -> Pronoun. A bare noun with no article defaults to Entity.
fn term_from(toks: &[String], idx: usize) -> Term {
    let word = &toks[idx];

    // Pronouns are their own terms (unresolved at parse time).
    if is_pronoun(word) {
        return Term::Pronoun(word.clone());
    }

    // Determiner immediately before the noun.
    if idx > 0 {
        match toks[idx - 1].as_str() {
            "a" | "an" => return Term::Indefinite(word.clone()),
            "the" => return Term::Entity(word.clone()),
            _ => {}
        }
    }
    // Default: a known/definite referent.
    Term::Entity(word.clone())
}

/// Is the token a (subject/object) pronoun we model as `Pronoun`?
fn is_pronoun(word: &str) -> bool {
    matches!(word, "it" | "they" | "he" | "she" | "them" | "him" | "her")
}

/// Locate a copula ("is"/"are"/"was"/"were") at or after `from`. Returns its
/// index. Only treated as a copula when it is NOT immediately functioning as a
/// passive/progressive auxiliary — for this curriculum bare copular statements
/// are "X is [a] C", so any copula in a declarative is the main predicate.
fn find_copula(toks: &[String], from: usize) -> Option<usize> {
    (from..toks.len()).find(|&i| matches!(toks[i].as_str(), "is" | "are" | "was" | "were"))
}

/// Scan auxiliaries and negation between a subject and the lexical verb.
///
/// Returns `(negated, tense_hint, verb_idx)`:
///   - `negated`   true if "not" appears in the auxiliary cluster,
///   - `tense_hint` Present for does/do, Past for did, None if no auxiliary,
///   - `verb_idx`  index of the first lexical (non-aux, non-"not") token, which
///     is taken to be the verb.
fn scan_aux_negation(toks: &[String], from: usize) -> (bool, Option<Tense>, Option<usize>) {
    let mut negated = false;
    let mut tense_hint = None;
    let mut i = from;
    while i < toks.len() {
        match toks[i].as_str() {
            "not" => {
                negated = true;
                i += 1;
            }
            "does" | "do" => {
                tense_hint = Some(Tense::Present);
                i += 1;
            }
            "did" => {
                tense_hint = Some(Tense::Past);
                i += 1;
            }
            // Skip stray determiners/articles that are not the verb.
            "the" | "a" | "an" => {
                i += 1;
            }
            _ => return (negated, tense_hint, Some(i)),
        }
    }
    (negated, tense_hint, None)
}

/// De-inflect a surface verb token to `(lemma, tense)` by reusing the Engine's
/// synthesized 3sg inflection program plus the curriculum verb lexicons.
///
/// Soundness: we only return a known lemma when the surface form provably
/// matches that lemma's base or its synthesized 3sg/past/gerund form. If no
/// known verb matches, we fall back to a conservative morphological strip so
/// the predicate is still a reasonable lemma.
fn lemma_and_tense(engine: &Engine, surface: &str) -> (String, Tense) {
    // 1) Exact base match (present, bare form). Includes irregular bases.
    for (base, _f3) in REG_VERBS.iter().chain(IRREGULAR_VERBS.iter()) {
        if surface == *base {
            return ((*base).to_string(), Tense::Present);
        }
    }

    // 2) Irregular 3sg form (e.g. "has" -> "have", "does" -> "do").
    for (base, f3) in IRREGULAR_VERBS.iter() {
        if surface == *f3 {
            return ((*base).to_string(), Tense::Present);
        }
    }

    // 3) Regular/irregular 3sg via the synthesized inflection program.
    //    For each known base, ask the Engine for its 3sg and compare.
    for (base, _f3) in REG_VERBS.iter().chain(IRREGULAR_VERBS.iter()) {
        if surface == engine.verb_3sg(base) {
            return ((*base).to_string(), Tense::Present);
        }
    }

    // 4) Past tense of a known regular base: base+"ed", base+"d", or
    //    consonant-doubled "<base><last>ed" (clap -> clapped). y->ied for
    //    bases ending in a consonant + y (carry -> carried).
    for (base, _f3) in REG_VERBS.iter() {
        if past_forms_match(base, surface) {
            return ((*base).to_string(), Tense::Past);
        }
    }

    // 5) Surface-only morphology fallback (unknown verb).
    deinflect_unknown(surface)
}

/// Does `surface` equal any regular past form of `base`?
fn past_forms_match(base: &str, surface: &str) -> bool {
    // base + "ed"   (walk -> walked)
    if surface == format!("{base}ed") {
        return true;
    }
    // base + "d"    (move -> moved, describe -> described)
    if base.ends_with('e') && surface == format!("{base}d") {
        return true;
    }
    // consonant + y -> ied  (carry -> carried)
    if let Some(stem) = base.strip_suffix('y') {
        if !stem.is_empty() {
            let last = stem.chars().last().unwrap();
            if !is_vowel(last) && surface == format!("{stem}ied") {
                return true;
            }
        }
    }
    // consonant-doubling  (clap -> clapped)
    if let Some(last) = base.chars().last() {
        if !is_vowel(last) && base.len() >= 2 {
            let prev = base.chars().rev().nth(1).unwrap();
            if is_vowel(prev) && surface == format!("{base}{last}ed") {
                return true;
            }
        }
    }
    false
}

/// Conservative de-inflection of an unknown verb form into `(lemma, tense)`.
fn deinflect_unknown(surface: &str) -> (String, Tense) {
    // Past: "-ied" -> "y", "-ed" -> strip; treat as Past.
    if let Some(stem) = surface.strip_suffix("ied") {
        return (format!("{stem}y"), Tense::Past);
    }
    if let Some(stem) = surface.strip_suffix("ed") {
        // Undo a doubled final consonant: "clapped" -> "clap".
        if let Some(undoubled) = undouble(stem) {
            return (undoubled, Tense::Past);
        }
        // "moved" stem is "mov" -> restore the silent e heuristically only if
        // the bare stem is implausible; default to the stem as lemma.
        return (stem.to_string(), Tense::Past);
    }
    // Present 3sg: "-ies" -> "y", "-es" after a sibilant -> strip "es",
    // plain "-s" -> strip "s".
    if let Some(stem) = surface.strip_suffix("ies") {
        return (format!("{stem}y"), Tense::Present);
    }
    if let Some(stem) = surface.strip_suffix("es") {
        if ends_sibilant_stem(stem) {
            return (stem.to_string(), Tense::Present);
        }
        // "writes" -> stem "writ"+? No: "writes" should drop only "s".
        // Fall through to the plain "-s" rule below by not returning here
        // when the stem isn't a sibilant.
    }
    if let Some(stem) = surface.strip_suffix('s') {
        if !stem.is_empty() {
            return (stem.to_string(), Tense::Present);
        }
    }
    // Bare form — assume present.
    (surface.to_string(), Tense::Present)
}

/// Undo a doubled trailing consonant ("clapp" -> "clap"); None if not doubled.
fn undouble(stem: &str) -> Option<String> {
    let mut chars = stem.chars().rev();
    let last = chars.next()?;
    let prev = chars.next()?;
    if last == prev && !is_vowel(last) {
        let mut s = stem.to_string();
        s.pop();
        Some(s)
    } else {
        None
    }
}

/// Does the stem end in a sibilant cluster that takes "-es" (e.g. watch, wash,
/// fix, pass, push, toss)?
fn ends_sibilant_stem(stem: &str) -> bool {
    stem.ends_with("ch")
        || stem.ends_with("sh")
        || stem.ends_with('x')
        || stem.ends_with("ss")
        || stem.ends_with('z')
        || stem.ends_with('s')
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

/// Derive a category ("person"/"thing") from a noun's animacy via the Engine.
fn animacy_category(engine: &Engine, head: &str) -> String {
    match engine.noun_class(head) {
        1 => "person".to_string(),
        2 => "thing".to_string(),
        _ => "thing".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// A single shared Engine: building it synthesizes the verified programs and
    /// is slow, so all tests in this module reuse one instance.
    fn engine() -> &'static Engine {
        static E: OnceLock<Engine> = OnceLock::new();
        E.get_or_init(Engine::new)
    }

    #[test]
    fn understand_declarative_event() {
        let m = understand(engine(), "The teacher writes the report.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.predicate, "write");
        assert_eq!(ev.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(ev.patient, Some(Term::Entity("report".to_string())));
        assert_eq!(ev.tense, Tense::Present);
        assert!(!ev.negated);
    }

    #[test]
    fn understand_negated_event() {
        let m = understand(engine(), "The teacher does not write the report.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.predicate, "write");
        assert!(ev.negated, "negation must be detected");
    }

    #[test]
    fn understand_wh_question_agent() {
        // "who writes the report?" -> WhQuestion{ slot: Agent, body has patient=report }
        let m = understand(engine(), "Who writes the report?");
        let Meaning::WhQuestion { slot, body } = m else {
            panic!("expected a WhQuestion, got {m:?}");
        };
        assert_eq!(slot, Role::Agent);
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, None, "queried slot is open");
        assert_eq!(body.patient, Some(Term::Entity("report".to_string())));
    }

    #[test]
    fn understand_wh_question_patient() {
        // "what does the teacher write?" -> WhQuestion{ slot: Patient, body has agent=teacher }
        let m = understand(engine(), "What does the teacher write?");
        let Meaning::WhQuestion { slot, body } = m else {
            panic!("expected a WhQuestion, got {m:?}");
        };
        assert_eq!(slot, Role::Patient);
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(body.patient, None, "queried slot is open");
    }

    #[test]
    fn understand_copular_isa() {
        let m = understand(engine(), "The teacher is a person.");
        let Meaning::IsA { subject, category, negated } = m else {
            panic!("expected an IsA, got {m:?}");
        };
        assert_eq!(subject, Term::Entity("teacher".to_string()));
        assert_eq!(category, "person");
        assert!(!negated);
    }

    #[test]
    fn understand_pronoun_subject_and_object() {
        // A pronoun must be a valid argument head (coreference resolves it later).
        let m = understand(engine(), "They read it.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.predicate, "read");
        assert_eq!(ev.agent, Some(Term::Pronoun("they".to_string())));
        assert_eq!(ev.patient, Some(Term::Pronoun("it".to_string())));
    }
}
