//! Core ARM64 CPU emulation layer.
//!
//! This module implements the fundamental fetch/decode/execute engine,
//! micro-operations, and instruction semantics that run on the Metal GPU.
//!
//! Everything else (neural ALU, JEPA scheduling, OS processes) is built on top
//! of this deterministic core.

pub mod full_arm64;
pub mod micro_op;
pub mod microkernel;

// Re-exports for the most important types
pub use full_arm64::{FULL_ARM64_SHADER, FullARM64CPU};
pub use microkernel::GpuMicrokernel;
pub use micro_op::{MicroOp, OpKind};