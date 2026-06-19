//! Native Rust gradient-based program synthesis.
//!
//! The synthesis module is split by responsibility so array synthesis and
//! teacher-driven distillation can evolve without concentrating everything into
//! one file.

mod array;
mod array_transform;
mod common;
mod native_array;
mod register_machine;
mod structured_array;
mod templates;
mod two_array;
mod universal;
mod universal_array;

// Shared primitives (constants, soft-op/soft-cmp, Adam, fd_grad, pseudo_rand,
// train_program, analytical gradient structs). Re-exported so every sibling
// module and the scalar orchestrator in core_impl.rs reach them via `super::*`.
pub(crate) use common::*;

pub use array::{synthesize_array, synthesize_array_from_teacher};
pub use register_machine::synthesize_register_machine;
pub use universal_array::prior_gen;

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
