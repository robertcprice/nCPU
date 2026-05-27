//! Neural execution components: ALU implemented by trained networks,
//! weight loading, dispatch logic, neural CPU variants, and hybrid modes.
//!
//! These power the "neural ALU in shader" part of the hero thesis and the
//! differentiable research surface.

pub mod neural_alu;
pub mod neural_cpu;
pub mod neural_cpu_fast;
pub mod neural_dispatch;
pub mod neural_dispatch_embedding;
pub mod neural_display;
pub mod neural_hybrid;
pub mod neural_weights;

// Key re-exports for the neural layer
pub use neural_alu::NeuralALUKernel;
pub use neural_weights::{ModelWeights, NeuralWeightCollection};