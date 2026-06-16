//! Semantic parser: maps a raw sentence to its logical form (`Meaning`).
//!
//! THE CORE of the understanding layer. Handles declarative events, negated
//! events, copular/category statements, yes/no questions, and wh-questions.
//! Subject = first noun (noun_class > 0), verb = word after the subject,
//! object = last noun. Determiner "the" -> Entity, "a"/"an" -> Indefinite,
//! pronoun it/they/he/she -> Pronoun. Verb lemma = de-inflect (strip
//! -s/-es/-ies or reverse the irregular form). Lexical facts come from the
//! synthesized Engine (noun_class, is_person, verb_3sg), never hardcoded.

use crate::comprehension::{words_of, Engine, GRADABLE, IRREGULAR_VERBS, MODIFIERS, REG_VERBS};
use crate::understanding::meaning::{Event, Meaning, Quantifier, Role, Tense, Term};

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

    // --- Disjunction "<clause> or <clause> [or <clause>]" ---------------------
    // Split on a top-level "or" connective (not a noun/argument) FIRST, so that a
    // coordinated sentence of any clause type ("the teacher writes or the editor
    // reads", "is the teacher careful or the editor calm?") becomes an `Or` over
    // the recursively-understood disjuncts. We only split when at least one "or"
    // separates two non-empty token groups, so a stray "or" never fragments a
    // single clause into Unknowns.
    if let Some(disjuncts) = split_disjunction(&toks) {
        // For a question disjunction ("is X careful or Y calm?") the leading
        // interrogative scopes over the whole coordination; re-distribute it so
        // each disjunct parses as its own question and the result is a yes/no
        // over the Or of the queried bodies.
        return build_disjunction(engine, &toks, &disjuncts, sentence);
    }

    let head = toks[0].as_str();

    // --- Counting question "how many <category> <verb> ...?" ------------------
    // Detect before the generic wh- / quantifier paths: the answer is a NUMBER,
    // not a slot filler or a truth value.
    if head == "how" && toks.get(1).map(|w| w == "many").unwrap_or(false) {
        return parse_count_question(engine, &toks, sentence);
    }

    // --- Cardinal declaratives "<number> <noun> <verb> ..." -------------------
    // A leading number word ("two teachers write a report") is an at-least
    // cardinal claim, distinct from the every/some/no quantifiers.
    if let Some(n) = number_word(head) {
        if let Some(m) = parse_cardinal(engine, n, &toks) {
            return m;
        }
    }

    // --- Quantified declaratives "every/some/no <noun> <verb> ..." ------------
    if let Some(quant) = quantifier_word(head) {
        return parse_quantified(engine, quant, &toks, sentence);
    }

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
// Quantifiers
// ===========================================================================

/// Recognize a leading quantifier word. Returns `None` for non-quantifier
/// words so the caller falls through to the ordinary parse.
///
/// Deliberately limited to the three contract quantifiers plus their closest
/// unambiguous synonyms ("each"/"all" = universal). We do NOT treat "a"/"an" as
/// an existential quantifier here: in this curriculum "a report" is an
/// indefinite argument, and an "a"-leading declarative ("a teacher writes the
/// report") must keep parsing as an ordinary `Event`, not a `Quantified`. This
/// keeps every existing declarative/indefinite behavior intact.
fn quantifier_word(word: &str) -> Option<Quantifier> {
    match word {
        "every" | "each" | "all" => Some(Quantifier::Every),
        "some" => Some(Quantifier::Some),
        "no" => Some(Quantifier::No),
        _ => None,
    }
}

/// The four taxonomy/animacy class names ("agent", "person", "thing",
/// "document") in their SINGULAR form. A quantifier or cardinal may range over
/// one of these classes, not just a leaf noun: "every agent writes a report"
/// binds the variable to the `agent` class. The world model treats these as
/// recognized categories and resolves their members through the hypernym chain,
/// so the parser only needs to admit them as valid `var_category` heads.
fn is_taxonomy_class_word(word: &str) -> bool {
    matches!(word, "agent" | "person" | "thing" | "document")
}

/// Is `word` a taxonomy class word in EITHER number — singular ("agent") or the
/// plural a cardinal/count-question surfaces ("agents", "people" excepted as it
/// is irregular and out of curriculum scope; we accept only regular "+s"
/// plurals: agents/persons/things/documents). Used to detect a class head; the
/// downstream `singular_category` normalizes it back to the singular.
fn is_taxonomy_class_head(word: &str) -> bool {
    if is_taxonomy_class_word(word) {
        return true;
    }
    word.strip_suffix('s')
        .map(is_taxonomy_class_word)
        .unwrap_or(false)
}

/// Index of the first token at or after `from` that can head a quantified /
/// cardinal category: a lexicon noun (noun_class > 0) OR a taxonomy class word
/// in singular or plural ("agent"/"agents"/...). The class words are NOT in the
/// synthesized noun lexicon, so we admit them explicitly here.
fn first_category_idx(engine: &Engine, toks: &[String], from: usize) -> Option<usize> {
    (from..toks.len())
        .find(|&i| engine.noun_class(&toks[i]) > 0 || is_taxonomy_class_head(&toks[i]))
}

/// Normalize a category head to its SINGULAR lexicon form so the world model can
/// match it. Cardinals/count-questions surface the plural ("two teachers", "how
/// many teachers"), but the taxonomy stores singular AGENT/PATIENT nouns and
/// singular class names. If the word as-is is already a known singular noun or a
/// taxonomy class, keep it; otherwise, when it ends in "s" and the de-pluralized
/// stem IS a known noun, return the singular. Falls back to the word unchanged.
///
/// Soundness: we only de-pluralize when the singular stem is a recognized noun,
/// so we never invent a category the lexicon does not know.
fn singular_category(engine: &Engine, word: &str) -> String {
    // Already a known singular noun (class 1/2) or a singular taxonomy class -> keep it.
    if (engine.noun_class(word) > 0 && !word.ends_with('s'))
        || is_taxonomy_class_word(word)
    {
        return word.to_string();
    }
    // "teachers" -> "teacher" / "agents" -> "agent": strip a trailing "s" when
    // the stem is a known noun OR a taxonomy class word.
    if let Some(stem) = word.strip_suffix('s') {
        if engine.noun_class(stem) > 0 || is_taxonomy_class_word(stem) {
            return stem.to_string();
        }
    }
    // Otherwise leave it as-is (e.g. an irregular/unknown form).
    word.to_string()
}

/// Parse "every/some/no <noun> <verb> [a/the <noun>]" into a `Quantified`.
///
/// For a declarative the quantifier word is at index 0, so the quantified noun
/// is searched from index 1. The quantified noun's lemma becomes `var_category`.
/// The body Event has its agent left `None` — the quantifier binds that slot,
/// ranging over entities of `var_category`; the verb and any object fill the
/// rest of the body.
fn parse_quantified(
    engine: &Engine,
    quant: Quantifier,
    toks: &[String],
    original: &str,
) -> Meaning {
    // Quantifier at index 0 -> noun search starts at index 1.
    match parse_quantified_body(engine, quant, toks, 0) {
        Some(m) => m,
        None => Meaning::Unknown(original.to_string()),
    }
}

/// Shared core: build a `Quantified` meaning from tokens, where `quant_idx` is
/// the index of the quantifier word (so the quantified noun is searched from
/// `quant_idx + 1`). Returns `None` if no quantified noun or verb is found.
///
/// Used both by the declarative path (quantifier at index 0) and the
/// interrogative path "does every <noun> <verb>?" (quantifier after the aux).
fn parse_quantified_body(
    engine: &Engine,
    quant: Quantifier,
    toks: &[String],
    quant_idx: usize,
) -> Option<Meaning> {
    // The quantified noun: first noun OR taxonomy class word after the
    // quantifier. Use the head directly (not term_from) — the quantifier, not a
    // determiner, governs it, so its surface "category" is the bare noun/class.
    // Admitting class words ("every agent ...") gives the parser quantifier
    // depth over the taxonomy, not just leaf nouns.
    let noun_idx = first_category_idx(engine, toks, quant_idx + 1)?;
    let var_category = toks[noun_idx].clone();

    // The verb: first lexical (non-aux, non-"not") token after the noun.
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, noun_idx + 1);
    let vidx = verb_idx?;
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = tense_hint.unwrap_or(surface_tense);

    // Object = last noun after the verb (if any). It keeps its own determiner
    // ("a report" -> Indefinite, "the report" -> Entity).
    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Some(Meaning::Quantified {
        quant,
        var_category,
        // The agent slot is bound by the quantifier and left open.
        body: Event {
            predicate,
            agent: None,
            patient,
            recipient: None,
            tense,
            negated,
        },
    })
}

// ===========================================================================
// Cardinality: "<number> <noun> <verb> ..." and "how many <category> <verb>?"
// ===========================================================================

/// Map a leading English number word to its integer value. Deliberately small
/// (the curriculum range two..six); `one` is intentionally NOT a cardinal here —
/// "one teacher" reads as an ordinary singular subject, not an at-least claim,
/// so we leave it to the declarative path. `None` for non-numbers.
fn number_word(word: &str) -> Option<usize> {
    match word {
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        _ => None,
    }
}

/// Parse "<number> <noun> <verb> [a/the <noun>]" into a `Cardinal`.
///
/// The number word is at index 0, so the quantified category noun is searched
/// from index 1. As with quantifiers the category head becomes `var_category`
/// and the agent slot is left open (the cardinal binds it, ranging over entities
/// of that category). Returns `None` if no category noun or verb is found, so a
/// stray number word never swallows an otherwise-unparseable sentence.
fn parse_cardinal(engine: &Engine, n: usize, toks: &[String]) -> Option<Meaning> {
    // Category head: first noun OR taxonomy class word after the number.
    // Normalize to the singular so the world model can match it ("teachers" ->
    // "teacher").
    let noun_idx = first_category_idx(engine, toks, 1)?;
    let var_category = singular_category(engine, &toks[noun_idx]);

    // The verb: first lexical (non-aux, non-"not") token after the category.
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, noun_idx + 1);
    let vidx = verb_idx?;
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = tense_hint.unwrap_or(surface_tense);

    // Object = last noun after the verb (keeps its own determiner).
    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Some(Meaning::Cardinal {
        at_least: n,
        var_category,
        body: Event {
            predicate,
            agent: None,
            patient,
            recipient: None,
            tense,
            negated,
        },
    })
}

/// Parse "how many <category> <verb> [a/the <noun>]?" into a `CountQuestion`.
///
/// "how" is at index 0, "many" at index 1, so the category head is searched from
/// index 2 (admitting both lexicon nouns and taxonomy class words). The body's
/// agent slot is left open — the count ranges over entities of the category.
fn parse_count_question(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    let Some(noun_idx) = first_category_idx(engine, toks, 2) else {
        return Meaning::Unknown(original.to_string());
    };
    // Normalize to the singular lexicon form ("teachers" -> "teacher").
    let var_category = singular_category(engine, &toks[noun_idx]);

    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, noun_idx + 1);
    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = tense_hint.unwrap_or(surface_tense);

    let patient = last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Meaning::CountQuestion {
        var_category,
        body: Event {
            predicate,
            agent: None,
            patient,
            recipient: None,
            tense,
            negated,
        },
    }
}

// ===========================================================================
// Disjunction
// ===========================================================================

/// Split a token sequence on top-level "or" connectives into disjunct token
/// groups. Returns `None` when there is no usable "or" (no "or", or an "or"
/// that would leave an empty group), so a single clause is never fragmented.
///
/// Soundness/robustness: we only treat "or" as a clause-coordinator when BOTH
/// sides are non-empty. A trailing/leading "or" or a doubled "or or" collapses
/// to a non-split, falling back to the ordinary single-clause parse.
fn split_disjunction(toks: &[String]) -> Option<Vec<Vec<String>>> {
    if !toks.iter().any(|t| t == "or") {
        return None;
    }
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for t in toks {
        if t == "or" {
            if cur.is_empty() {
                // Leading or doubled "or" — not a clean clausal split.
                return None;
            }
            groups.push(std::mem::take(&mut cur));
        } else {
            cur.push(t.clone());
        }
    }
    if cur.is_empty() {
        // Trailing "or" with nothing after it.
        return None;
    }
    groups.push(cur);
    if groups.len() < 2 {
        return None;
    }
    Some(groups)
}

/// Build an `Or` meaning (or a yes/no question wrapping one) from disjunct
/// groups. Handles the gapping case where a leading interrogative or subject
/// scopes over later disjuncts: the leading clause's prefix (the interrogative
/// word, or "the <noun> is") is prepended to bare later disjuncts so each parses
/// as a full clause.
fn build_disjunction(
    engine: &Engine,
    toks: &[String],
    groups: &[Vec<String>],
    original: &str,
) -> Meaning {
    let head = toks[0].as_str();

    // A question disjunction: the leading interrogative scopes over all
    // disjuncts. Re-distribute it so each disjunct is understood as a question,
    // then lift the inner queried bodies into a single yes/no over their Or.
    let is_question_lead =
        matches!(head, "is" | "are" | "was" | "were" | "does" | "do" | "did" | "who" | "what");

    let mut disjunct_meanings: Vec<Meaning> = Vec::new();
    let mut all_yes_no = true;

    for (idx, g) in groups.iter().enumerate() {
        // For disjuncts after the first, if the original led with an
        // interrogative/copular gap, prepend that interrogative to the bare
        // disjunct ("is X careful or Y calm?" -> 2nd disjunct "is Y calm").
        let clause: String = if idx > 0 && is_question_lead {
            format!("{} {}", head, g.join(" "))
        } else {
            g.join(" ")
        };
        let m = understand(engine, &clause);
        match &m {
            Meaning::YesNoQuestion(inner) => disjunct_meanings.push((**inner).clone()),
            _ => {
                all_yes_no = false;
                disjunct_meanings.push(m);
            }
        }
    }

    // If every disjunct came back `Unknown`, the whole coordination is
    // unparseable — report a single `Unknown` rather than a vacuous `Or`.
    if !disjunct_meanings.is_empty()
        && disjunct_meanings
            .iter()
            .all(|d| matches!(d, Meaning::Unknown(_)))
    {
        return Meaning::Unknown(original.to_string());
    }

    let or = Meaning::Or(disjunct_meanings);
    if is_question_lead && all_yes_no {
        // "is X careful or is Y calm?" -> yes/no over the disjunction of bodies.
        Meaning::YesNoQuestion(Box::new(or))
    } else {
        or
    }
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

    // Propositional attitude: "<subject> <attitude>s [not] that <embedded S>"
    // (-> Attitude). Detected before the copula/event paths because the verb is
    // a clause-taking attitude verb and a "that"-complement follows.
    if let Some(att) = parse_attitude(engine, toks, &subject, after) {
        return att;
    }

    // Comparative: "<subject> is [not] <comparative> than <standard>"
    // (-> Comparison). Detected before the generic copula path so the
    // comparative complement is not misread as a bare property/category.
    if let Some(cmp) = parse_comparative(engine, toks, &subject, after) {
        return cmp;
    }

    // Copular statement: "<subject> is/are/was/were [not] [a/an] <category>"
    // (-> IsA) OR "<subject> is [not] <adjective>" (-> HasProperty). The
    // complement's lexical class (noun vs. adjective) decides which.
    if let Some(cop_idx) = find_copula(toks, after) {
        return build_copular(engine, toks, subject, cop_idx);
    }

    // Action event (possibly ditransitive: "... gives the book to the student").
    build_event_declarative(engine, toks, subject, after, original)
}

/// Build the meaning of a copular statement, choosing `HasProperty` when the
/// complement is an adjective (a MODIFIER) and `IsA` when it is a noun/category.
///
/// "the teacher is a person" / "the teacher is a thing" -> IsA.
/// "the teacher is careful" -> HasProperty.
fn build_copular(engine: &Engine, toks: &[String], subject: Term, cop_idx: usize) -> Meaning {
    let mut i = cop_idx + 1;
    let mut negated = false;
    if i < toks.len() && toks[i] == "not" {
        negated = true;
        i += 1;
    }

    // An indefinite article ("is a/an ...") signals a nominal category — an
    // adjectival predicate never takes an article ("*is a careful").
    let has_article = i < toks.len() && (toks[i] == "a" || toks[i] == "an");

    // The complement head is the last content token.
    if i >= toks.len() {
        // Degenerate "X is" — fall back to the original IsA behavior.
        let cat = animacy_category(engine, subject.head());
        return build_isa_from(subject, cat, negated);
    }
    let complement = toks[toks.len() - 1].clone();

    // Adjective complement (and no nominal article) -> property.
    if !has_article && is_adjective(&complement) {
        return Meaning::HasProperty {
            subject,
            property: complement,
            negated,
        };
    }

    // Otherwise a category statement.
    build_isa_from(subject, complement, negated)
}

/// Is the word a lexicon adjective — either a plain MODIFIER ("careful") or a
/// gradable POSITIVE adjective ("long", "heavy")? Both are adjectival predicates
/// that, as a bare (article-less) copular complement, denote a property of the
/// subject rather than a nominal category. Reuses the curriculum's synthesized
/// MODIFIERS list and the GRADABLE table rather than a hardcoded set.
///
/// Including the gradable positives makes "the report is long" a `HasProperty`
/// (the semantically correct property predication), which in turn lets an
/// embedded attitude clause ("knows that the report is long") carry a property
/// content. The comparative forms ("longer") are handled separately by
/// `parse_comparative` and never reach here as bare complements.
fn is_adjective(word: &str) -> bool {
    MODIFIERS.iter().any(|m| *m == word) || positive_scale(word).is_some()
}

/// Construct an `IsA` from explicit parts.
fn build_isa_from(subject: Term, category: String, negated: bool) -> Meaning {
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

    // Ditransitive: a 3-place verb ("give"/"send"/...) fills patient + recipient.
    let (patient, recipient) = extract_objects(engine, toks, &predicate, vidx + 1);

    Meaning::Event(Event {
        predicate,
        agent: Some(subject),
        patient,
        recipient,
        tense,
        negated,
    })
}

// ===========================================================================
// Yes/no question parsing: "does/do/did [not] X verb Y?"  or "is X a C?"
// ===========================================================================

fn parse_yes_no(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    let head = toks[0].as_str();

    // Copular yes/no: "is the teacher a person?" (-> IsA) /
    // "is the teacher careful?" (-> HasProperty). The complement's lexical class
    // (noun vs. adjective) decides which, exactly as in the declarative copula.
    if matches!(head, "is" | "are" | "was" | "were") {
        // Subject is the first noun after the copula.
        if let Some(subj_idx) = first_noun_idx(engine, toks, 1) {
            let subject = term_from(toks, subj_idx);

            // Comparative yes/no: "is the report longer than the book?" — the
            // inverted copula sits at index 0, so the comparative search starts
            // right after the subject. Wrap the Comparison as a yes/no.
            if let Some(cmp) = parse_comparative(engine, toks, &subject, subj_idx + 1) {
                return Meaning::YesNoQuestion(Box::new(cmp));
            }

            // Look for a complement after the subject.
            let mut i = subj_idx + 1;
            let mut negated = false;
            if i < toks.len() && toks[i] == "not" {
                negated = true;
                i += 1;
            }
            let has_article = i < toks.len() && (toks[i] == "a" || toks[i] == "an");
            if i < toks.len() {
                let complement = toks[toks.len() - 1].clone();
                // Adjective complement (no nominal article) -> property query.
                let inner = if !has_article && is_adjective(&complement) {
                    Meaning::HasProperty {
                        subject,
                        property: complement,
                        negated,
                    }
                } else {
                    Meaning::IsA {
                        subject,
                        category: complement,
                        negated,
                    }
                };
                return Meaning::YesNoQuestion(Box::new(inner));
            }
        }
        return Meaning::Unknown(original.to_string());
    }

    // Quantified yes/no: "does every teacher write a report?" — a quantifier
    // immediately after the auxiliary scopes the whole question. Build the
    // Quantified body and wrap it as a yes/no over the universal/existential.
    if let Some(q) = toks.get(1).and_then(|w| quantifier_word(w)) {
        if let Some(qm) = parse_quantified_body(engine, q, toks, 1) {
            return Meaning::YesNoQuestion(Box::new(qm));
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

    // Attitude yes/no: "does the teacher know that <S>?" — the verb is a
    // clause-taking attitude verb and a "that" complement follows. Build the
    // Attitude over the embedded meaning and wrap it as a yes/no.
    if is_attitude_verb(&predicate) {
        if let Some(that_idx) = find_that(toks, vidx + 1) {
            let content = understand(engine, &toks[that_idx + 1..].join(" "));
            return Meaning::YesNoQuestion(Box::new(Meaning::Attitude {
                holder: subject,
                verb: predicate,
                content: Box::new(content),
                negated,
            }));
        }
    }

    // Ditransitive yes/no: "does the teacher give the book to the student?".
    let (patient, recipient) = extract_objects(engine, toks, &predicate, vidx + 1);

    Meaning::YesNoQuestion(Box::new(Meaning::Event(Event {
        predicate,
        agent: Some(subject),
        patient,
        recipient,
        tense: aux_tense,
        negated,
    })))
}

// ===========================================================================
// wh-question parsing
// ===========================================================================

/// "who writes the report?" -> WhQuestion{ slot: Agent, body: Event(write, patient=report) }
///
/// Recipient case: "who does the teacher give the book to?" — an auxiliary plus
/// an explicit subject after "who" means the queried slot is NOT the agent
/// (the agent is named); a ditransitive verb makes it the RECIPIENT, so
/// -> WhQuestion{ slot: Recipient, body: Event(give, agent=teacher, patient=book) }.
fn parse_who(engine: &Engine, toks: &[String], original: &str) -> Meaning {
    // An auxiliary right after "who" signals subject-aux inversion: the queried
    // "who" is an OBJECT/RECIPIENT, and a concrete subject follows ("who does
    // the teacher give the book to?"). Without an auxiliary, "who" is the
    // agent ("who writes the report?").
    let aux_after_who = matches!(
        toks.get(1).map(|s| s.as_str()),
        Some("does") | Some("do") | Some("did")
    );

    if aux_after_who {
        if let Some(m) = parse_who_recipient(engine, toks) {
            return m;
        }
    }

    // After "who" comes the verb (possibly with negation), then the object.
    let after = 1;
    let (negated, tense_hint, verb_idx) = scan_aux_negation(toks, after);
    let Some(vidx) = verb_idx else {
        return Meaning::Unknown(original.to_string());
    };
    let (predicate, surface_tense) = lemma_and_tense(engine, &toks[vidx]);
    let tense = tense_hint.unwrap_or(surface_tense);

    // If a "to" follows (ditransitive: "who gives the book to the student?"), the
    // PATIENT is the noun BEFORE "to" — the noun after "to" is the recipient. Take
    // the recipient too so the query is fully constrained. Otherwise the patient
    // is simply the last noun after the verb.
    let to_pos = toks
        .iter()
        .enumerate()
        .skip(vidx + 1)
        .find(|(_, w)| w.as_str() == "to")
        .map(|(i, _)| i);
    let (patient, recipient) = match to_pos {
        Some(tp) => (
            last_noun_idx_within(engine, toks, vidx + 1, tp).map(|oi| term_from(toks, oi)),
            last_noun_idx(engine, toks, tp + 1).map(|ri| term_from(toks, ri)),
        ),
        None => (last_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi)), None),
    };

    Meaning::WhQuestion {
        slot: Role::Agent,
        body: Event {
            predicate,
            agent: None,
            patient,
            recipient,
            tense,
            negated,
        },
    }
}

/// "who does the teacher give the book to?" -> WhQuestion over the RECIPIENT.
///
/// The aux is at index 1; the subject (agent) is the first noun after it; the
/// verb follows; the patient is the noun between the verb and the trailing "to".
/// Returns `None` (so the caller falls back to the agent reading) unless the
/// verb is a 3-place ditransitive — only those have a recipient slot to query.
fn parse_who_recipient(engine: &Engine, toks: &[String]) -> Option<Meaning> {
    let aux_tense = match toks.get(1).map(|s| s.as_str()) {
        Some("did") => Tense::Past,
        _ => Tense::Present,
    };

    // Subject (agent) = first noun after the auxiliary.
    let subj_idx = first_noun_idx(engine, toks, 2)?;
    let agent = term_from(toks, subj_idx);

    // Verb after the subject.
    let (negated, _t, verb_idx) = scan_aux_negation(toks, subj_idx + 1);
    let vidx = verb_idx?;
    let (predicate, _) = lemma_and_tense(engine, &toks[vidx]);

    // Only ditransitive verbs have a recipient to ask about.
    if !is_ditransitive_verb(&predicate) {
        return None;
    }

    // Patient = first noun after the verb (the direct object "the book").
    let patient = first_noun_idx(engine, toks, vidx + 1).map(|oi| term_from(toks, oi));

    Some(Meaning::WhQuestion {
        slot: Role::Recipient,
        body: Event {
            predicate,
            agent: Some(agent),
            patient,
            // The queried recipient is left open.
            recipient: None,
            tense: aux_tense,
            negated,
        },
    })
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
            recipient: None,
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

// ===========================================================================
// Ditransitive / 3-place predicates
// ===========================================================================

/// The verb lemmas that take a recipient (a third "to"-marked / indirect-object
/// argument): give/send/show/offer/hand/teach/tell. Reuses the curriculum verb
/// set — these are exactly the ditransitives added to REG_VERBS/IRREGULAR_PAST.
fn is_ditransitive_verb(lemma: &str) -> bool {
    matches!(
        lemma,
        "give" | "send" | "show" | "offer" | "hand" | "teach" | "tell"
    )
}

/// Extract the (patient, recipient) object slots for a verb, starting the scan
/// at `from` (just past the verb).
///
/// For a 2-place verb the patient is the last noun (unchanged legacy behavior)
/// and the recipient is `None`. For a ditransitive verb there are two shapes:
///   * prepositional: "gives the book TO the student" — the patient is the noun
///     before "to" (the direct object), the recipient the noun after "to".
///   * double-object: "gives the student the book" — the FIRST noun is the
///     recipient (indirect object) and the SECOND is the patient (direct object).
/// If only one object noun is present, it is taken as the patient with no
/// recipient (e.g. "the teacher gives the book").
fn extract_objects(
    engine: &Engine,
    toks: &[String],
    predicate: &str,
    from: usize,
) -> (Option<Term>, Option<Term>) {
    if !is_ditransitive_verb(predicate) {
        // 2-place verb: patient = last noun after the verb.
        let patient = last_noun_idx(engine, toks, from).map(|oi| term_from(toks, oi));
        return (patient, None);
    }

    // Prepositional dative: a "to" between the verb and the end signals
    // "... <patient> to <recipient>".
    if let Some(to_idx) = find_word(toks, from, "to") {
        // Patient: last object noun BEFORE "to" (the direct object).
        let patient = last_noun_idx_within(engine, toks, from, to_idx).map(|oi| term_from(toks, oi));
        // Recipient: first noun AFTER "to".
        let recipient = first_noun_idx(engine, toks, to_idx + 1).map(|oi| term_from(toks, oi));
        return (patient, recipient);
    }

    // Double-object dative: "<recipient> <patient>" — two object nouns, no "to".
    let nouns = object_noun_indices(engine, toks, from);
    match nouns.len() {
        // "gives the student the book": first noun = recipient, second = patient.
        n if n >= 2 => {
            let recipient = Some(term_from(toks, nouns[0]));
            let patient = Some(term_from(toks, nouns[1]));
            (patient, recipient)
        }
        // A single object noun: take it as the patient, no recipient.
        1 => (Some(term_from(toks, nouns[0])), None),
        // No object noun at all.
        _ => (None, None),
    }
}

/// Indices of every noun-phrase head in `toks[from..]`, in surface order.
fn object_noun_indices(engine: &Engine, toks: &[String], from: usize) -> Vec<usize> {
    (from..toks.len())
        .filter(|&i| is_np_head(engine, &toks[i]))
        .collect()
}

/// Index of the last noun-phrase head strictly within `toks[from..to]`.
fn last_noun_idx_within(engine: &Engine, toks: &[String], from: usize, to: usize) -> Option<usize> {
    (from..to).rev().find(|&i| is_np_head(engine, &toks[i]))
}

/// Index of the first occurrence of `word` at or after `from`.
fn find_word(toks: &[String], from: usize, word: &str) -> Option<usize> {
    (from..toks.len()).find(|&i| toks[i] == word)
}

// ===========================================================================
// Comparatives
// ===========================================================================

/// The "lesser-pole" positive gradable adjectives — the antonyms whose
/// comparative asserts the subject FALLS BELOW the standard on the shared scale
/// ("shorter"/"smaller"/"lighter"/"slower"). Their counterparts (long/big/
/// heavy/fast) are the "greater pole". Both poles share a scale name so the
/// world model composes orderings on one dimension; the pole determines the
/// DIRECTION of the ordering (`more`).
///
/// SOUNDNESS: this is the single source of comparative polarity. "A is shorter
/// than B" must NOT be stored as "A exceeds B" — it is "A falls below B"
/// (equivalently B exceeds A). Mis-signing `more` here would silently invert
/// every ordering, so the lesser-pole set is enumerated explicitly.
const LESSER_POLE: &[&str] = &["short", "small", "light", "slow"];

/// Is the positive adjective the lesser pole of its scale (short/small/...)?
fn is_lesser_pole(positive: &str) -> bool {
    LESSER_POLE.iter().any(|w| *w == positive)
}

/// If `word` is a comparative form in the GRADABLE table ("longer", "shorter"),
/// return `(scale, more)` where `scale` is the gradable dimension and `more` is
/// the ordering direction: `true` for the greater pole ("longer" -> subject
/// exceeds), `false` for the lesser pole ("shorter" -> subject falls below).
/// `None` if `word` is not a known comparative. Reuses the synthesized
/// gradable-adjective table rather than a hardcoded set.
fn comparative_scale(word: &str) -> Option<(&'static str, bool)> {
    GRADABLE
        .iter()
        .find(|(_, comp, _)| *comp == word)
        .map(|(pos, _, scale)| (*scale, !is_lesser_pole(pos)))
}

/// If `word` is a POSITIVE gradable adjective ("long", "small"), return
/// `(scale, more)` with the same pole convention as `comparative_scale`. Used
/// for the periphrastic "more <adj> than" comparative ("more heavy", and also
/// the well-formed but unusual "more light" reads as the lesser pole).
fn positive_scale(word: &str) -> Option<(&'static str, bool)> {
    GRADABLE
        .iter()
        .find(|(pos, _, _)| *pos == word)
        .map(|(pos, _, scale)| (*scale, !is_lesser_pole(pos)))
}

/// Parse a comparative clause "<subject> is [not] <comparative> than <standard>"
/// (or the periphrastic "<subject> is [not] more <adj> than <standard>") into a
/// `Comparison`. `after` is the index just past the subject; the copula is
/// located (declarative "the report is longer ...") or — when absent because the
/// copula was fronted in a yes/no question ("is the report longer ...?") — the
/// complement scan starts right at `after`. Returns `None` if the clause is not
/// a comparative, so callers fall through to the ordinary copular/event parse.
///
/// Soundness: `more` reflects the comparative's POLE on the shared scale, not a
/// constant. "A is longer than B" -> `more: true` (A exceeds B on length);
/// "A is shorter than B" -> `more: false` (A falls below B on the SAME length
/// scale). Both poles point at one scale so the world model composes orderings
/// on a single dimension and never infers symmetry; the pole keeps the stored
/// ordering directionally correct.
fn parse_comparative(
    engine: &Engine,
    toks: &[String],
    subject: &Term,
    after: usize,
) -> Option<Meaning> {
    // The complement begins right after the (possibly fronted) copula. In a
    // declarative the copula sits at/after `after`; in an inverted yes/no it was
    // fronted to index 0, so the complement starts at `after` itself.
    let mut i = match find_copula(toks, after) {
        Some(cop_idx) => cop_idx + 1,
        None => after,
    };

    let mut negated = false;
    if toks.get(i).map(|w| w == "not").unwrap_or(false) {
        negated = true;
        i += 1;
    }

    // A "than" must be present for this to be a comparative.
    let than_idx = find_word(toks, i, "than")?;

    // The comparative degree word sits between the copula (post-negation) and
    // "than". Accept either a synthetic comparative ("longer") or the
    // periphrastic "more <adj>". Each yields (scale, more-pole).
    let (scale, more) = if toks.get(i).map(|w| w == "more").unwrap_or(false) {
        // "is more <adj> than": the adjective is the token right after "more".
        let adj = toks.get(i + 1)?;
        positive_scale(adj)?
    } else {
        // A single comparative word ("longer"); it is the token just before
        // "than" (handles a stray determiner gap defensively by scanning back).
        let mut found = None;
        for j in (i..than_idx).rev() {
            if let Some(s) = comparative_scale(&toks[j]) {
                found = Some(s);
                break;
            }
        }
        found?
    };

    // The standard of comparison: the first noun after "than".
    let std_idx = first_noun_idx(engine, toks, than_idx + 1)?;
    let than = term_from(toks, std_idx);

    Some(Meaning::Comparison {
        subject: subject.clone(),
        scale: scale.to_string(),
        // Pole-determined direction: greater pole exceeds, lesser pole falls below.
        more,
        than,
        negated,
    })
}

// ===========================================================================
// Propositional attitudes (clausal complements)
// ===========================================================================

/// The attitude verb lemmas with a clausal "that"-complement: know/believe/
/// think/say. FACTIVITY (know entails its content; believe/think/say do not) is
/// decided downstream by `verb`; the parser only recognizes the surface verbs.
fn is_attitude_verb(lemma: &str) -> bool {
    matches!(lemma, "know" | "believe" | "think" | "say")
}

/// Index of the first "that" at or after `from`.
fn find_that(toks: &[String], from: usize) -> Option<usize> {
    find_word(toks, from, "that")
}

/// Parse a propositional attitude "<subject> <attitude>s [not] that <embedded S>"
/// into an `Attitude` over the recursively-understood embedded clause. `after`
/// is the index just past the subject. Returns `None` when the verb is not an
/// attitude verb or no "that" complement follows, so callers fall through to the
/// ordinary parse.
///
/// 3sg forms ("knows"/"believes"/...) and the bare base ("does the teacher know
/// that ...?" sends the lemma here) are both handled because the verb token is
/// de-inflected to its lemma before the attitude test.
fn parse_attitude(
    engine: &Engine,
    toks: &[String],
    subject: &Term,
    after: usize,
) -> Option<Meaning> {
    let (negated, _tense_hint, verb_idx) = scan_aux_negation(toks, after);
    let vidx = verb_idx?;
    let (predicate, _surface_tense) = lemma_and_tense(engine, &toks[vidx]);

    if !is_attitude_verb(&predicate) {
        return None;
    }

    // A clausal "that" complement must follow the attitude verb.
    let that_idx = find_that(toks, vidx + 1)?;
    let embedded = toks[that_idx + 1..].join(" ");
    if embedded.trim().is_empty() {
        return None;
    }

    // Recursively understand the embedded clause as its own Meaning.
    let content = understand(engine, &embedded);

    Some(Meaning::Attitude {
        holder: subject.clone(),
        verb: predicate,
        content: Box::new(content),
        negated,
    })
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

    // -------------------------------------------------------------------
    // New capabilities: quantifiers, disjunction, attributes.
    // -------------------------------------------------------------------

    #[test]
    fn understand_universal_quantifier() {
        // "every teacher writes a report" -> Quantified{ Every, teacher, body }.
        let m = understand(engine(), "Every teacher writes a report.");
        let Meaning::Quantified { quant, var_category, body } = m else {
            panic!("expected a Quantified, got {m:?}");
        };
        assert_eq!(quant, Quantifier::Every);
        assert_eq!(var_category, "teacher");
        assert_eq!(body.predicate, "write");
        // The quantifier binds the agent slot — it must be left open.
        assert_eq!(body.agent, None, "quantified agent slot is bound/open");
        assert_eq!(body.patient, Some(Term::Indefinite("report".to_string())));
        assert!(!body.negated);
        assert_eq!(body.tense, Tense::Present);
    }

    #[test]
    fn understand_existential_and_negative_quantifiers() {
        let some = understand(engine(), "Some editor reads the book.");
        let Meaning::Quantified { quant, var_category, .. } = some else {
            panic!("expected a Quantified for 'some', got {some:?}");
        };
        assert_eq!(quant, Quantifier::Some);
        assert_eq!(var_category, "editor");

        let none = understand(engine(), "No student answers the question.");
        let Meaning::Quantified { quant, var_category, body } = none else {
            panic!("expected a Quantified for 'no', got {none:?}");
        };
        assert_eq!(quant, Quantifier::No);
        assert_eq!(var_category, "student");
        assert_eq!(body.predicate, "answer");
    }

    #[test]
    fn understand_quantified_yes_no_question() {
        // "does every teacher write a report?" -> YesNoQuestion(Quantified{Every,...}).
        let m = understand(engine(), "Does every teacher write a report?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::Quantified { quant, var_category, body } = *inner else {
            panic!("expected a Quantified inside the question");
        };
        assert_eq!(quant, Quantifier::Every);
        assert_eq!(var_category, "teacher");
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, None);
    }

    #[test]
    fn understand_disjunction_declarative() {
        // "the teacher writes the report or the editor reads the book" -> Or[..].
        let m = understand(
            engine(),
            "The teacher writes the report or the editor reads the book.",
        );
        let Meaning::Or(ds) = m else {
            panic!("expected an Or, got {m:?}");
        };
        assert_eq!(ds.len(), 2, "two disjuncts");
        // First disjunct is the teacher-writes event.
        let Meaning::Event(e0) = &ds[0] else {
            panic!("expected first disjunct Event, got {:?}", ds[0]);
        };
        assert_eq!(e0.predicate, "write");
        assert_eq!(e0.agent, Some(Term::Entity("teacher".to_string())));
        let Meaning::Event(e1) = &ds[1] else {
            panic!("expected second disjunct Event, got {:?}", ds[1]);
        };
        assert_eq!(e1.predicate, "read");
        assert_eq!(e1.agent, Some(Term::Entity("editor".to_string())));
    }

    #[test]
    fn understand_attribute_declarative() {
        // "the teacher is careful" -> HasProperty (NOT IsA), because "careful"
        // is an adjective (a MODIFIER), not a noun category.
        let m = understand(engine(), "The teacher is careful.");
        let Meaning::HasProperty { subject, property, negated } = m else {
            panic!("expected a HasProperty, got {m:?}");
        };
        assert_eq!(subject, Term::Entity("teacher".to_string()));
        assert_eq!(property, "careful");
        assert!(!negated);
    }

    #[test]
    fn understand_negated_attribute() {
        let m = understand(engine(), "The teacher is not careful.");
        let Meaning::HasProperty { property, negated, .. } = m else {
            panic!("expected a HasProperty, got {m:?}");
        };
        assert_eq!(property, "careful");
        assert!(negated, "negation must be detected on the property");
    }

    #[test]
    fn understand_attribute_question_vs_category_question() {
        // "is the teacher careful?" -> YesNoQuestion(HasProperty).
        let m = understand(engine(), "Is the teacher careful?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::HasProperty { property, negated, subject } = *inner else {
            panic!("expected HasProperty inside the question");
        };
        assert_eq!(subject, Term::Entity("teacher".to_string()));
        assert_eq!(property, "careful");
        assert!(!negated);

        // "is the teacher a person?" stays a category (IsA) question — the
        // nominal article + noun complement keeps it nominal.
        let m2 = understand(engine(), "Is the teacher a person?");
        let Meaning::YesNoQuestion(inner2) = m2 else {
            panic!("expected a YesNoQuestion, got {m2:?}");
        };
        assert!(
            matches!(*inner2, Meaning::IsA { .. }),
            "noun complement must stay IsA, got {inner2:?}"
        );
    }

    #[test]
    fn isa_with_noun_complement_unaffected() {
        // Regression guard: a noun complement without an article ("the teacher
        // is person") must NOT be misread as an adjective property — "person"
        // is not in MODIFIERS, so it stays an IsA.
        let m = understand(engine(), "The teacher is person.");
        assert!(
            matches!(m, Meaning::IsA { .. }),
            "noun complement must be IsA, got {m:?}"
        );
    }

    // ===================================================================
    // A. Ditransitive / 3-place predicates
    // ===================================================================

    #[test]
    fn understand_ditransitive_prepositional() {
        // "the teacher gives the book to the student" -> give(teacher, book, student).
        let m = understand(engine(), "The teacher gives the book to the student.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.predicate, "give");
        assert_eq!(ev.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(ev.patient, Some(Term::Entity("book".to_string())));
        assert_eq!(ev.recipient, Some(Term::Entity("student".to_string())));
    }

    #[test]
    fn understand_ditransitive_double_object() {
        // "the teacher gives the student the book" — double-object order:
        // first noun = recipient (student), second = patient (book).
        let m = understand(engine(), "The teacher gives the student the book.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.predicate, "give");
        assert_eq!(ev.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(ev.patient, Some(Term::Entity("book".to_string())));
        assert_eq!(ev.recipient, Some(Term::Entity("student".to_string())));
    }

    #[test]
    fn two_place_verb_keeps_recipient_none() {
        // Regression: a plain 2-place verb must NOT get a recipient.
        let m = understand(engine(), "The teacher writes the report.");
        let Meaning::Event(ev) = m else {
            panic!("expected an Event, got {m:?}");
        };
        assert_eq!(ev.recipient, None);
    }

    #[test]
    fn ditransitive_wh_recipient_question() {
        // "who does the teacher give the book to?" -> Wh over the RECIPIENT slot,
        // with the agent + patient pinned and the recipient left open.
        let m = understand(engine(), "Who does the teacher give the book to?");
        let Meaning::WhQuestion { slot, body } = m else {
            panic!("expected a WhQuestion, got {m:?}");
        };
        assert_eq!(slot, Role::Recipient);
        assert_eq!(body.predicate, "give");
        assert_eq!(body.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(body.patient, Some(Term::Entity("book".to_string())));
        assert_eq!(body.recipient, None, "queried recipient slot is open");
    }

    #[test]
    fn plain_who_still_agent_question() {
        // Regression: a bare "who writes the report?" stays an AGENT question,
        // not a recipient one (no auxiliary, so no inversion).
        let m = understand(engine(), "Who writes the report?");
        let Meaning::WhQuestion { slot, .. } = m else {
            panic!("expected a WhQuestion, got {m:?}");
        };
        assert_eq!(slot, Role::Agent);
    }

    #[test]
    fn ditransitive_yes_no_question() {
        // "does the teacher give the book to the student?" -> yes/no over a
        // ditransitive event with all three roles filled.
        let m = understand(engine(), "Does the teacher give the book to the student?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::Event(ev) = *inner else {
            panic!("expected an Event inside the question");
        };
        assert_eq!(ev.predicate, "give");
        assert_eq!(ev.agent, Some(Term::Entity("teacher".to_string())));
        assert_eq!(ev.patient, Some(Term::Entity("book".to_string())));
        assert_eq!(ev.recipient, Some(Term::Entity("student".to_string())));
    }

    // ===================================================================
    // B. Comparatives
    // ===================================================================

    #[test]
    fn understand_comparative_declarative() {
        // "the report is longer than the book" -> Comparison on the length scale.
        let m = understand(engine(), "The report is longer than the book.");
        let Meaning::Comparison { subject, scale, more, than, negated } = m else {
            panic!("expected a Comparison, got {m:?}");
        };
        assert_eq!(subject, Term::Entity("report".to_string()));
        assert_eq!(scale, "length");
        assert!(more, "longer asserts exceed");
        assert_eq!(than, Term::Entity("book".to_string()));
        assert!(!negated);
    }

    #[test]
    fn understand_comparative_antonym_same_scale() {
        // "the book is shorter than the report" — the antonym maps to the SAME
        // scale ("length") so the world model composes orderings on one axis,
        // but the lesser pole flips the DIRECTION: book falls BELOW report.
        let m = understand(engine(), "The book is shorter than the report.");
        let Meaning::Comparison { subject, scale, more, than, .. } = m else {
            panic!("expected a Comparison, got {m:?}");
        };
        assert_eq!(subject, Term::Entity("book".to_string()));
        assert_eq!(scale, "length");
        assert!(!more, "shorter is the lesser pole: subject falls below the standard");
        assert_eq!(than, Term::Entity("report".to_string()));
    }

    #[test]
    fn understand_comparative_yes_no_question() {
        // "is the report longer than the book?" -> yes/no over a Comparison.
        let m = understand(engine(), "Is the report longer than the book?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::Comparison { subject, scale, than, .. } = *inner else {
            panic!("expected a Comparison inside the question");
        };
        assert_eq!(subject, Term::Entity("report".to_string()));
        assert_eq!(scale, "length");
        assert_eq!(than, Term::Entity("book".to_string()));
    }

    #[test]
    fn comparative_periphrastic_more() {
        // "the report is more heavy than the book" — periphrastic comparative.
        let m = understand(engine(), "The report is more heavy than the book.");
        let Meaning::Comparison { scale, more, .. } = m else {
            panic!("expected a Comparison, got {m:?}");
        };
        assert_eq!(scale, "weight");
        assert!(more);
    }

    #[test]
    fn negated_comparative() {
        // "the report is not longer than the book" -> negated Comparison.
        let m = understand(engine(), "The report is not longer than the book.");
        let Meaning::Comparison { negated, .. } = m else {
            panic!("expected a Comparison, got {m:?}");
        };
        assert!(negated, "negation must be detected on the comparison");
    }

    // ===================================================================
    // C. Epistemic / clausal complements
    // ===================================================================

    #[test]
    fn understand_attitude_factive_declarative() {
        // "the teacher knows that the report is long" -> Attitude{know, content}.
        let m = understand(engine(), "The teacher knows that the report is long.");
        let Meaning::Attitude { holder, verb, content, negated } = m else {
            panic!("expected an Attitude, got {m:?}");
        };
        assert_eq!(holder, Term::Entity("teacher".to_string()));
        assert_eq!(verb, "know");
        assert!(!negated);
        // The embedded clause is its own Meaning (a HasProperty over "long").
        assert!(
            matches!(*content, Meaning::HasProperty { ref property, .. } if property == "long"),
            "embedded content must be the recursively-understood clause, got {content:?}"
        );
    }

    #[test]
    fn understand_attitude_nonfactive_declarative() {
        // "the teacher believes that the report is long" — non-factive verb. The
        // PARSER does not decide factivity; it only records the verb lemma.
        let m = understand(engine(), "The teacher believes that the report is long.");
        let Meaning::Attitude { verb, .. } = m else {
            panic!("expected an Attitude, got {m:?}");
        };
        assert_eq!(verb, "believe");
    }

    #[test]
    fn understand_attitude_embedded_event() {
        // "the teacher says that the student writes a report" — embedded Event.
        let m = understand(engine(), "The teacher says that the student writes a report.");
        let Meaning::Attitude { verb, content, .. } = m else {
            panic!("expected an Attitude, got {m:?}");
        };
        assert_eq!(verb, "say");
        let Meaning::Event(ev) = *content else {
            panic!("expected an embedded Event, got {content:?}");
        };
        assert_eq!(ev.predicate, "write");
        assert_eq!(ev.agent, Some(Term::Entity("student".to_string())));
    }

    #[test]
    fn understand_attitude_yes_no_question() {
        // "does the teacher know that the report is long?" -> yes/no over Attitude.
        let m = understand(engine(), "Does the teacher know that the report is long?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::Attitude { holder, verb, content, .. } = *inner else {
            panic!("expected an Attitude inside the question, got it bare");
        };
        assert_eq!(holder, Term::Entity("teacher".to_string()));
        assert_eq!(verb, "know");
        assert!(matches!(*content, Meaning::HasProperty { .. }));
    }

    #[test]
    fn negated_attitude() {
        // "the teacher does not know that the report is long" -> negated Attitude.
        let m = understand(engine(), "The teacher does not know that the report is long.");
        let Meaning::Attitude { negated, verb, .. } = m else {
            panic!("expected an Attitude, got {m:?}");
        };
        assert_eq!(verb, "know");
        assert!(negated, "negation must be detected on the attitude");
    }

    // ===================================================================
    // D. Cardinality
    // ===================================================================

    #[test]
    fn understand_cardinal_declarative() {
        // "two teachers write a report" -> Cardinal{ at_least: 2, teacher, body }.
        let m = understand(engine(), "Two teachers write a report.");
        let Meaning::Cardinal { at_least, var_category, body } = m else {
            panic!("expected a Cardinal, got {m:?}");
        };
        assert_eq!(at_least, 2);
        assert_eq!(var_category, "teacher");
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, None, "cardinal binds the agent slot");
        assert_eq!(body.patient, Some(Term::Indefinite("report".to_string())));
    }

    #[test]
    fn understand_cardinal_three() {
        let m = understand(engine(), "Three editors read the book.");
        let Meaning::Cardinal { at_least, var_category, .. } = m else {
            panic!("expected a Cardinal, got {m:?}");
        };
        assert_eq!(at_least, 3);
        assert_eq!(var_category, "editor");
    }

    #[test]
    fn understand_count_question() {
        // "how many teachers write a report?" -> CountQuestion. The plural
        // category head is normalized to the singular lexicon form "teacher".
        let m = understand(engine(), "How many teachers write a report?");
        let Meaning::CountQuestion { var_category, body } = m else {
            panic!("expected a CountQuestion, got {m:?}");
        };
        assert_eq!(var_category, "teacher");
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, None, "count ranges over the category");
        assert_eq!(body.patient, Some(Term::Indefinite("report".to_string())));
    }

    #[test]
    fn count_question_over_taxonomy_class() {
        // "how many agents write a report?" — the plural taxonomy CLASS head
        // ("agents") is recognized and normalized to the singular class "agent".
        let m = understand(engine(), "How many agents write a report?");
        let Meaning::CountQuestion { var_category, body } = m else {
            panic!("expected a CountQuestion, got {m:?}");
        };
        assert_eq!(var_category, "agent");
        assert_eq!(body.predicate, "write");
    }

    #[test]
    fn cardinal_singularizes_plural_category() {
        // "two teachers ..." surfaces the plural; var_category is the singular.
        let m = understand(engine(), "Two teachers write a report.");
        let Meaning::Cardinal { var_category, .. } = m else {
            panic!("expected a Cardinal, got {m:?}");
        };
        assert_eq!(var_category, "teacher", "category normalized to singular");
    }

    // ===================================================================
    // E. Quantifier-parser depth over taxonomy classes
    // ===================================================================

    #[test]
    fn quantifier_over_taxonomy_class_declarative() {
        // "every agent writes a report" — the category is a taxonomy CLASS word
        // ("agent"), not a leaf noun, and the parser must still bind it.
        let m = understand(engine(), "Every agent writes a report.");
        let Meaning::Quantified { quant, var_category, body } = m else {
            panic!("expected a Quantified, got {m:?}");
        };
        assert_eq!(quant, Quantifier::Every);
        assert_eq!(var_category, "agent");
        assert_eq!(body.predicate, "write");
        assert_eq!(body.agent, None);
    }

    #[test]
    fn quantifier_over_taxonomy_class_yes_no() {
        // "does every agent write a report?" through the PARSER (not direct
        // construction) sets var_category to the taxonomy class "agent".
        let m = understand(engine(), "Does every agent write a report?");
        let Meaning::YesNoQuestion(inner) = m else {
            panic!("expected a YesNoQuestion, got {m:?}");
        };
        let Meaning::Quantified { quant, var_category, .. } = *inner else {
            panic!("expected a Quantified inside the question");
        };
        assert_eq!(quant, Quantifier::Every);
        assert_eq!(var_category, "agent");
    }

    #[test]
    fn quantifier_over_taxonomy_class_thing() {
        // "some thing is a document" is degenerate, but "no thing moves the book"
        // exercises an inanimate taxonomy class as the bound variable.
        let m = understand(engine(), "No thing moves the book.");
        let Meaning::Quantified { quant, var_category, .. } = m else {
            panic!("expected a Quantified, got {m:?}");
        };
        assert_eq!(quant, Quantifier::No);
        assert_eq!(var_category, "thing");
    }

    // ===================================================================
    // Helper-level unit tests (lexical predicates stay sound)
    // ===================================================================

    #[test]
    fn lexical_predicates_are_sound() {
        // Ditransitive vs. plain verb classification.
        assert!(is_ditransitive_verb("give"));
        assert!(is_ditransitive_verb("teach"));
        assert!(!is_ditransitive_verb("write"));
        assert!(!is_ditransitive_verb("read"));
        // Attitude verbs.
        assert!(is_attitude_verb("know"));
        assert!(is_attitude_verb("believe"));
        assert!(!is_attitude_verb("write"));
        // Comparative scale + pole lookups via GRADABLE.
        assert_eq!(comparative_scale("longer"), Some(("length", true)));
        assert_eq!(comparative_scale("bigger"), Some(("size", true)));
        // Lesser-pole comparatives flip the direction (subject falls below).
        assert_eq!(comparative_scale("shorter"), Some(("length", false)));
        assert_eq!(comparative_scale("slower"), Some(("speed", false)));
        assert_eq!(comparative_scale("write"), None);
        assert_eq!(positive_scale("heavy"), Some(("weight", true)));
        assert_eq!(positive_scale("light"), Some(("weight", false)));
        // Number words; "one" is intentionally not a cardinal.
        assert_eq!(number_word("two"), Some(2));
        assert_eq!(number_word("three"), Some(3));
        assert_eq!(number_word("one"), None);
        assert_eq!(number_word("teacher"), None);
        // Taxonomy class words (singular) and class heads (singular or plural).
        assert!(is_taxonomy_class_word("agent"));
        assert!(is_taxonomy_class_word("thing"));
        assert!(!is_taxonomy_class_word("teacher"));
        assert!(!is_taxonomy_class_word("agents"), "plural is not the singular class word");
        assert!(is_taxonomy_class_head("agent"));
        assert!(is_taxonomy_class_head("agents"));
        assert!(is_taxonomy_class_head("things"));
        assert!(!is_taxonomy_class_head("teacher"));
    }
}
