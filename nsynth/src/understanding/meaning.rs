//! Semantic representation: the logical form a sentence is understood AS. This is
//! what turns "manipulating language" into "understanding it" — the semantic
//! parser maps a sentence to a Meaning; the world model evaluates a Meaning's
//! truth; inference relates Meanings; QA queries them.

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    /// definite, known referent: "the teacher" -> Entity("teacher")
    Entity(String),
    /// indefinite: "a report" -> Indefinite("report")
    Indefinite(String),
    /// unresolved pronoun before coreference: "it"/"they" -> Pronoun("it")
    Pronoun(String),
    /// definite term restricted by a relative clause:
    /// "the teacher who writes the report" -> Restricted{head:"teacher",
    /// clause: Event(write, patient=report)}. The referent is the entity of
    /// category `head` that satisfies `clause`.
    Restricted { head: String, clause: Box<Event> },
}

impl Term {
    /// surface noun/word of the term (without article)
    pub fn head(&self) -> &str {
        match self {
            Term::Entity(s) | Term::Indefinite(s) | Term::Pronoun(s) => s,
            Term::Restricted { head, .. } => head,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role { Agent, Patient, Recipient }

/// A natural-language quantifier scoping over a category variable.
/// `Every` is universal ("every teacher"), `Some` existential ("some teacher"),
/// `No` negative-existential ("no teacher").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantifier { Every, Some, No }   // universal, existential, negative

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tense { Past, Present, Future }   // wrote / writes / will write

/// Grammatical aspect on an event: how the action relates to its own internal
/// time-course. `Simple` ("writes"/"wrote"), `Progressive` ("is writing" —
/// ongoing), `Perfect` ("has written" — completed with present relevance).
/// Both Progressive and Perfect entail the simple event holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Aspect { Simple, Progressive, Perfect }   // writes / is writing / has written

/// An event/action predication: write(agent: teacher, patient: report), present, not negated.
/// A ditransitive event also fills the `recipient` slot:
/// give(agent: teacher, patient: book, recipient: student).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Event {
    pub predicate: String,        // verb lemma: "write"
    pub agent: Option<Term>,
    pub patient: Option<Term>,
    /// ditransitive recipient ("... to the student"): `None` for 2-place verbs.
    pub recipient: Option<Term>,
    pub tense: Tense,
    /// grammatical aspect: Simple / Progressive ("is writing") / Perfect ("has written").
    pub aspect: Aspect,
    pub negated: bool,
}

/// Modal force over an event. `Can` possibility, `Must` necessity (entails
/// `Can`), `Might` epistemic possibility, `Should` weak deontic necessity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality { Can, Must, Might, Should }

/// Temporal ordering relation between two events. `Before` and `After` are
/// converses; `Before` is transitive and asymmetric (sound).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalRel { Before, After }

/// The meaning of one sentence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Meaning {
    /// an action/event: "the teacher writes the report"
    Event(Event),
    /// a category/property: "the teacher is a person" -> IsA{ subject, category, negated }
    IsA { subject: Term, category: String, negated: bool },
    /// a yes/no question: "does the teacher write the report?" wraps the queried meaning
    YesNoQuestion(Box<Meaning>),
    /// a wh-question: "who writes the report?" -> { slot: Agent, body: Event(write, patient=report) }
    WhQuestion { slot: Role, body: Event },
    /// "every/some/no <category> <verb> ..." e.g. Every teacher writes a report.
    /// `var_category` is the lemma of the quantified noun ("teacher"); `body`
    /// is the verbal predication whose agent ranges over entities of that
    /// category (the body's agent slot is left None or a fresh Indefinite
    /// placeholder — the quantifier binds it).
    Quantified { quant: Quantifier, var_category: String, body: Event },
    /// "X or Y" — true iff any disjunct is true.
    Or(Vec<Meaning>),
    /// "the teacher is careful" — an adjectival property of an entity.
    HasProperty { subject: Term, property: String, negated: bool },
    /// "the report is longer than the book" — gradable comparison.
    /// `scale` is the gradable dimension ("length"/"size"/...); `more` is the
    /// polarity of the comparative (longer/bigger -> `true`); `subject` is
    /// asserted to exceed `than` on that scale when `more` (or fall below when
    /// `!more`). `negated` flips the whole assertion.
    Comparison { subject: Term, scale: String, more: bool, than: Term, negated: bool },
    /// "the teacher knows that <S>" — a propositional attitude over an embedded
    /// meaning. `verb` is the attitude lemma (know/believe/think/say); `content`
    /// is the embedded clause's Meaning. FACTIVITY is decided downstream by the
    /// verb: "know that P" entails P, "believe/think/say that P" does not.
    Attitude { holder: Term, verb: String, content: Box<Meaning>, negated: bool },
    /// "two teachers write a report" — cardinal (at-least) quantification.
    /// True iff at least `at_least` known entities of `var_category` satisfy
    /// `body`.
    Cardinal { at_least: usize, var_category: String, body: Event },
    /// "how many teachers write a report?" — a counting question whose answer is
    /// the number of entities of `var_category` that satisfy `body`.
    CountQuestion { var_category: String, body: Event },
    /// "the teacher can/must/might/should write the report" — modal force over an
    /// event. `Must` entails `Can`; possibility does NOT entail actuality.
    Modal { modality: Modality, body: Box<Event>, negated: bool },
    /// "X writes the report before Y reads the book" — a temporal ordering of two
    /// events. `Before` is transitive and asymmetric.
    Temporal { rel: TemporalRel, first: Box<Event>, second: Box<Event> },
    /// "the street floods because the rain falls" — a causal link. Asserting it
    /// presupposes both `cause` and `effect` happened. NOT symmetric.
    Causal { cause: Box<Meaning>, effect: Box<Meaning> },
    /// "if the rain falls then the street floods" — a stated material/defeasible
    /// implication. STRICTLY WEAKER than `Causal`: asserting it does NOT
    /// presuppose either `antecedent` or `consequent` happened — it only states
    /// the rule. Sound truth: vacuously true when the antecedent is false; the
    /// consequent must hold when the antecedent holds. `negated` flips the whole
    /// conditional (its sound contradictory). Modus ponens (antecedent known
    /// true → derive consequent) lives in inference/world_model, not here.
    Conditional { antecedent: Box<Meaning>, consequent: Box<Meaning>, negated: bool },
    /// "how long is the report?" — a degree question over a gradable `scale`,
    /// answered from known comparison facts (or honestly "I don't know").
    DegreeQuestion { subject: Term, scale: String },
    /// outer (wide-scope) negation for SCOPE distinctions: "not every teacher
    /// writes a report" -> Not(Quantified{Every,...}). Three-valued negation of
    /// the inner meaning's truth.
    Not(Box<Meaning>),
    /// unparseable into a meaning
    Unknown(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(s: &str) -> Term {
        Term::Entity(s.to_string())
    }

    /// A ditransitive event fills the new `recipient` slot, and `Role::Recipient`
    /// is a distinct thematic role.
    #[test]
    fn ditransitive_event_has_recipient_and_role() {
        let give = Event {
            predicate: "give".to_string(),
            agent: Some(ent("teacher")),
            patient: Some(ent("book")),
            recipient: Some(ent("student")),
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        assert_eq!(give.recipient, Some(ent("student")));
        // The three roles are distinct.
        assert_ne!(Role::Recipient, Role::Agent);
        assert_ne!(Role::Recipient, Role::Patient);
    }

    /// The four new Meaning variants construct with the contract-specified shapes
    /// and the factive/non-factive distinction lives in the `verb` field.
    #[test]
    fn new_meaning_variants_construct() {
        let comparison = Meaning::Comparison {
            subject: ent("report"),
            scale: "length".to_string(),
            more: true,
            than: ent("book"),
            negated: false,
        };
        assert!(matches!(comparison, Meaning::Comparison { more: true, .. }));

        // "the teacher knows that the report is long" — factive attitude.
        let attitude = Meaning::Attitude {
            holder: ent("teacher"),
            verb: "know".to_string(),
            content: Box::new(Meaning::HasProperty {
                subject: ent("report"),
                property: "long".to_string(),
                negated: false,
            }),
            negated: false,
        };
        let Meaning::Attitude { verb, content, .. } = &attitude else {
            panic!("expected Attitude");
        };
        assert_eq!(verb, "know"); // factivity decided downstream by the verb
        assert!(matches!(**content, Meaning::HasProperty { .. }));

        let body = Event {
            predicate: "write".to_string(),
            agent: None,
            patient: Some(Term::Indefinite("report".to_string())),
            recipient: None,
            tense: Tense::Present,
            aspect: Aspect::Simple,
            negated: false,
        };
        let cardinal = Meaning::Cardinal {
            at_least: 2,
            var_category: "teacher".to_string(),
            body: body.clone(),
        };
        assert!(matches!(cardinal, Meaning::Cardinal { at_least: 2, .. }));

        let count = Meaning::CountQuestion {
            var_category: "teacher".to_string(),
            body,
        };
        assert!(matches!(count, Meaning::CountQuestion { .. }));
    }
}
