//! Native Rust gradient-based program synthesis.
//!
//! The synthesis module is split by responsibility so array synthesis and
//! teacher-driven distillation can evolve without concentrating everything into
//! one file.

mod array;
pub mod array_transform;
pub(crate) mod common;
/// Reflective stateful-reducer capability surface (UNWALL-1-STATEFUL-NL): exposes
/// the engine's `(state, arr) -> state` per-tick reducer family as mineable NL
/// capabilities, bound to the engine op surface via an exhaustive guard.
pub mod stateful_reducer_surface;
mod native_array;
mod register_machine;
mod structured_array;
mod templates;
mod two_array;
mod universal;
mod universal_array;
mod utbus;

// Shared primitives (constants, soft-op/soft-cmp, Adam, fd_grad, pseudo_rand,
// train_program, analytical gradient structs). Re-exported so every sibling
// module and the scalar orchestrator in core_impl.rs reach them via `super::*`.
pub(crate) use common::*;

pub use array::{synthesize_array, synthesize_array_from_teacher};
pub use register_machine::synthesize_register_machine;
pub use universal_array::prior_gen;

/// Per-QUERY wall-clock budget for the whole solve, held by callers at the request
/// boundary (the coding-agent product entry, the never-wrong sweep). The gradient
/// synthesizers install their own per-ATTEMPT caps with `TrainDeadline::set_min`, so
/// they can only ever TIGHTEN this shared deadline — a query that fans out into many
/// sequential solve attempts still finishes within the budget instead of resetting
/// the clock on every attempt (the cause of the "move all zeroes" multi-attempt hang).
/// RAII: drop restores the previous deadline, so it nests cleanly. No-op when the
/// process sets no budget.
pub struct QuerySolveBudget(common::TrainDeadline);
impl QuerySolveBudget {
    /// Bound the current thread's solve to `millis` from now.
    pub fn millis(millis: u64) -> Self {
        QuerySolveBudget(common::TrainDeadline::set(std::time::Duration::from_millis(millis)))
    }
}

// Re-exported so sibling modules (register_machine, universal_array) pick it
// up via `use super::*;`. Internal to the synthesis module tree only.
pub(crate) use native_array::ArrExample;
pub(crate) use templates::try_scalar_templates;
pub use universal::{
    rand_description, record_from_synthesis, synthesize_universal_and_collect,
    synthesize_universal_warm_start, synthetic_record, MetaRecord, SlotDesc, SoftUniversalProgram,
    UniversalProgramDescription, N_INIT_SLOTS, N_LOOP_SLOTS, N_POST_SLOTS, N_UNIV_SLOTS,
};

include!("core_impl.rs");
