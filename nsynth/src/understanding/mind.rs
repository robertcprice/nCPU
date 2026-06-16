//! A stateful "mind": an Engine plus a Discourse it reads into and answers from.
//! This is the top-level understanding handle — read sentences to build a world
//! model, then ask questions answered from what was read. Exposed to C in
//! [`crate::ffi`] as `ncpu_mind_new` / `ncpu_read` / `ncpu_ask`.

use crate::comprehension::Engine;
use crate::understanding::discourse::Discourse;
use crate::understanding::inference::render_proof;
use crate::understanding::meaning::Meaning;
use crate::understanding::{qa, semantics};

pub struct Mind {
    engine: Engine,
    discourse: Discourse,
}

impl Default for Mind {
    fn default() -> Self {
        Self::new()
    }
}

impl Mind {
    /// Build a mind: synthesizes the lexicon/rules once (slow), then ready to read.
    pub fn new() -> Self {
        Mind { engine: Engine::new(), discourse: Discourse::new() }
    }

    /// Read a sentence into the world model — resolves coreference against what
    /// was read before, asserts the resulting facts, and returns the resolved
    /// Meaning.
    pub fn read(&mut self, sentence: &str) -> Meaning {
        self.discourse.read(&self.engine, sentence)
    }

    /// Answer a question from what has been read (truth from the world model,
    /// closed under sound inference).
    pub fn ask(&self, question: &str) -> String {
        qa::answer(&self.engine, &self.discourse, question)
    }

    /// Answer a question AND show the reasoning behind it — metacognition over
    /// what was read. Parses the question through the SAME path as [`ask`]
    /// (`semantics::understand`, so pronouns / relative clauses resolve
    /// identically), then routes the parsed meaning through
    /// [`qa::answer_explained`], which returns the same string `ask` would PLUS a
    /// sound [`Proof`](crate::understanding::inference::Proof) when the verdict
    /// rests on a derivation.
    ///
    /// When a proof is present, the answer (`"Yes, ..."` / `"No, ..."`) is
    /// followed by the rendered "because" chain — each derivation step named and
    /// its premises bottoming out in `you told me ...` leaves, so a transitively
    /// entailed answer surfaces the intermediate facts it routed through. When no
    /// proof backs the verdict (a world-owned/opaque truth, a wh-filler, a count,
    /// a cause, or the honest `"I don't know."` of an unprovable query), the plain
    /// answer is returned with NO fabricated justification.
    pub fn why(&mut self, question: &str) -> String {
        // Parse exactly as `ask` does, then explain the parsed meaning.
        let parsed = semantics::understand(&self.engine, question);
        let (answer, proof) = qa::answer_explained(&self.engine, &self.discourse, &parsed);
        match proof {
            Some(proof) => format!("{answer} {}.", render_proof(&proof, &self.engine)),
            None => answer,
        }
    }

    /// The logical form of a sentence, without asserting it.
    pub fn understand(&self, sentence: &str) -> Meaning {
        semantics::understand(&self.engine, sentence)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mind_reads_then_answers_from_the_world() {
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        mind.read("The author reads the book.");
        // answered from the world model built by reading
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("What does the author read?").to_lowercase().contains("book"));
        // a category question, from animacy
        assert!(mind.ask("Is the report a person?").to_lowercase().starts_with("no"));
    }

    #[test]
    fn mind_understands_vp_conjunction() {
        let mut mind = Mind::new();
        // VP-conjunction with subject ellipsis: both facts must enter the world.
        mind.read("The teacher writes the report and reads the book.");
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("Who reads the book?").to_lowercase().contains("teacher"));
        assert!(mind.ask("What does the teacher read?").to_lowercase().contains("book"));
    }

    #[test]
    fn mind_understands_clausal_conjunction() {
        let mut mind = Mind::new();
        mind.read("The teacher writes the report and the editor reads the book.");
        assert!(mind.ask("Who writes the report?").to_lowercase().contains("teacher"));
        assert!(mind.ask("Who reads the book?").to_lowercase().contains("editor"));
    }

    #[test]
    fn why_renders_multi_step_because_chain_for_transitive_entailment() {
        // Read one concrete fact, then ask a doubly-generalized query whose truth
        // is reached by CHAINING two sound inference steps: drop-patient (the
        // teacher writes the report ⊢ the teacher writes) then generalize-agent
        // (the teacher writes ⊢ a teacher writes). `why` must surface that full
        // derivation as a nested "because" chain, with the INTERMEDIATE fact
        // ("the teacher writes") appearing between the goal and the asserted leaf.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        let w = mind.why("Does a teacher write something?");
        let lw = w.to_lowercase();
        // Affirmative verdict, identical to what `ask` returns, then the chain.
        assert!(lw.starts_with("yes"), "transitive query is true: {w}");
        // A genuine derivation: at least one "because" link...
        assert!(lw.contains("because"), "must render a derivation: {w}");
        // ...bottoming out in an honest "you told me ..." leaf naming the asserted fact.
        assert!(
            lw.contains("you told me the teacher writes the report"),
            "leaf must restate the asserted fact: {w}"
        );
        // The INTERMEDIATE step ("the teacher writes") is mentioned between the
        // generalized goal and the asserted leaf — the analogue of a transitive
        // middle term — and the named rules attribute each hop.
        assert!(lw.contains("the teacher writes"), "intermediate fact shown: {w}");
        assert!(lw.contains("(by drop-patient)"), "names the drop-patient hop: {w}");
        assert!(
            lw.contains("(by generalize-agent)"),
            "names the generalize-agent hop: {w}"
        );
        // Two distinct hops ⇒ two "because" connectives (a genuine multi-step chain).
        assert!(
            lw.matches("because").count() >= 2,
            "multi-step chain needs ≥2 'because' links: {w}"
        );
    }

    #[test]
    fn why_renders_single_step_because_for_a_one_hop_derivation() {
        // A one-hop entailment (drop-patient) still gets a "because" chain whose
        // single premise is the asserted fact — the smallest honest explanation.
        let mut mind = Mind::new();
        mind.read("The author reads the book.");
        let w = mind.why("Does the author read something?");
        let lw = w.to_lowercase();
        assert!(lw.starts_with("yes"), "entailed query is true: {w}");
        assert!(lw.contains("because"), "one-hop still explains: {w}");
        assert!(
            lw.contains("you told me the author reads the book"),
            "premise is the asserted fact: {w}"
        );
        assert!(lw.contains("(by drop-patient)"), "names the hop: {w}");
    }

    #[test]
    fn why_of_unprovable_query_honestly_says_it_does_not_know_without_fabricating() {
        // Nothing about the author was ever read. An open-world query must answer
        // honestly "I don't know." — and crucially carry NO "because" justification
        // (no fabricated derivation for something we cannot prove).
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        let w = mind.why("Does the author read the book?");
        assert!(
            w.to_lowercase().contains("don't know"),
            "unprovable query is honest: {w}"
        );
        assert!(
            !w.to_lowercase().contains("because"),
            "no fabricated derivation for an unknown: {w}"
        );
        assert!(
            !w.to_lowercase().contains("you told me"),
            "no invented premise for an unknown: {w}"
        );
    }

    #[test]
    fn why_matches_ask_for_an_opaque_fact_but_chains_a_transitive_comparison() {
        // A directly-asserted EVENT fact is a verdict the world MODEL owns with no
        // public derivation, so `why` must equal `ask` with no fabricated "because".
        // A TRANSITIVE comparison, by contrast, IS reconstructible: the world owns
        // the closure verdict, but `prove` rebuilds the chain over the asserted
        // edges, so `why` shows its work and names the intermediate.
        let mut mind = Mind::new();
        mind.read("The teacher writes the report.");
        mind.read("The report is longer than the book.");
        mind.read("The book is longer than the letter.");

        // Directly asserted EVENT ⇒ world-owned opaque verdict, so `why` == `ask`.
        let direct = mind.why("Does the teacher write the report?");
        assert_eq!(
            direct,
            mind.ask("Does the teacher write the report?"),
            "world-owned event verdict: why must equal ask"
        );
        assert!(!direct.to_lowercase().contains("because"), "no chain on opaque fact: {direct}");

        // Transitive comparison ⇒ reconstructed chain naming the `book` intermediate.
        let cmp = mind.why("Is the report longer than the letter?");
        let low = cmp.to_lowercase();
        assert!(low.starts_with("yes"), "transitive comparison true: {cmp}");
        assert!(low.contains("because"), "transitive comparison must show its work: {cmp}");
        assert!(low.contains("book"), "transitive proof must name the intermediate 'book': {cmp}");
        assert!(
            low.contains("comparison-transitivity"),
            "chain must be labeled with the transitivity rule: {cmp}"
        );
    }
}
