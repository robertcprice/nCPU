//! The understanding layer: a meaning representation, world model, inference,
//! discourse-level coreference, and question answering — built on top of the
//! existing synthesized comprehension engine. Where `comprehension` recovers
//! verified Mog programs for lexical/rule decisions, this layer turns those
//! decisions into genuine understanding: sentences become logical forms, a
//! world model evaluates their truth, inference relates them, discourse resolves
//! reference, and QA answers from the model built by reading.

pub mod meaning;
pub mod semantics;
pub mod world_model;
pub mod inference;
pub mod discourse;
pub mod qa;
