//! Hierarchical synthesis for large programs
//!
//! Decomposes specifications into modules, synthesizes incrementally,
//! and composes final programs from synthesized components.

pub mod cache;
pub mod decomposition;
pub mod incremental;
pub mod interface;
pub mod refinement;

pub use decomposition::{decompose, ModuleSpec};
pub use incremental::{synthesize_incremental, SynthCache};
pub use interface::{discover_interface, Interface};
pub use refinement::refine_with_types;
