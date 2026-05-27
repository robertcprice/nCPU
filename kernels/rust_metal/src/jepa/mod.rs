//! JEPA Neural Kernel + Neural OS layer.
//!
//! This is the current high-signal direction: a learned bottom-up neural machine
//! that observes real execution on the deterministic Rust Metal substrate and
//! actively steers scheduling via three decision levers (bias override, Ready demotion,
//! and adaptive de-prio skips driven by recency-aware churn).

pub mod neural_jepa_kernel;
pub mod neural_os;
pub mod neural_os_models;

// Re-exports for the JEPA layer (high-signal learned scheduling)
pub use neural_jepa_kernel::NeuralJepaKernel;