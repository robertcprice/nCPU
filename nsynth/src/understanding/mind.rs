//! A stateful "mind": an Engine plus a Discourse it reads into and answers from.
//! This is the top-level understanding handle — read sentences to build a world
//! model, then ask questions answered from what was read. Exposed to C in
//! [`crate::ffi`] as `ncpu_mind_new` / `ncpu_read` / `ncpu_ask`.

use crate::comprehension::Engine;
use crate::understanding::discourse::Discourse;
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
}
