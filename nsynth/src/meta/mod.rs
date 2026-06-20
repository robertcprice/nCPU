//! Meta-level self-improvement (Phase 5).
//!
//! Currently houses [`recursive`], the bounded recursive-self-improvement loop
//! that tunes the ranker weight vector against the benchmark under hard safety
//! rails. Future meta passes (curriculum, architecture search) live here too.

pub mod recursive;
