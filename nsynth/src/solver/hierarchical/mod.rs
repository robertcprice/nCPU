//! Hierarchical synthesis for large programs
//!
//! Decomposes specifications into modules, synthesizes incrementally,
//! and composes final programs from synthesized components.

pub mod decomposition;
pub mod interface;
pub mod incremental;
pub mod refinement;
pub mod cache;

pub use decomposition::{decompose, ModuleSpec};
pub use interface::{discover_interface, Interface};
pub use incremental::{synthesize_incremental, SynthCache};
pub use refinement::refine_with_types;
