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
}

impl Term {
    /// surface noun/word of the term (without article)
    pub fn head(&self) -> &str {
        match self { Term::Entity(s) | Term::Indefinite(s) | Term::Pronoun(s) => s }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role { Agent, Patient }

/// A natural-language quantifier scoping over a category variable.
/// `Every` is universal ("every teacher"), `Some` existential ("some teacher"),
/// `No` negative-existential ("no teacher").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quantifier { Every, Some, No }   // universal, existential, negative

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tense { Past, Present }

/// An event/action predication: write(agent: teacher, patient: report), present, not negated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Event {
    pub predicate: String,        // verb lemma: "write"
    pub agent: Option<Term>,
    pub patient: Option<Term>,
    pub tense: Tense,
    pub negated: bool,
}

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
    /// unparseable into a meaning
    Unknown(String),
}
