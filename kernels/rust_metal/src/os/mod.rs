//! Operating system and runtime layer.
//!
//! This module contains the process model, scheduler, launcher, and everything
//! needed to turn the raw GPU CPU emulator into a multi-process UNIX-like
//! environment capable of running BusyBox, Alpine, and a self-hosting C compiler.
//!
//! This is a core part of the "GPU *is* the computer" thesis.

pub mod launcher;
pub mod process;

// Re-exports for convenience
pub use launcher::GpuLauncher;
pub use process::{Process, ProcessManager, ProcessState};