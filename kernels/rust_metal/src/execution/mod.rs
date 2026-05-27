//! Different execution engine variants and kernel implementations.
//!
//! This module groups the many "kinds of kernels" that exist in the crate:
//! - Pure GPU execution
//! - Various optimized paths (ultra, fusion, etc.)
//! - Out-of-order (OOO) execution
//! - Differentiable / JIT variants for training and research
//! - Parallel and continuous modes
//!
//! Having these under one `execution` module makes the architectural variety
//! of the GPU computer explicit and easier to navigate/extend.

pub mod async_gpu;
pub mod continuous;
pub mod diff_jit;
pub mod differentiable_ooo;
pub mod fusion;
pub mod gpu_optimizer;
pub mod jit_compiler;
pub mod multi_kernel;
pub mod neural_ooo;
pub mod ooo_exec;
pub mod optimized;
pub mod parallel;
pub mod pure_gpu;
pub mod trace_jit;
pub mod ultra;
pub mod ultra_optimized;
pub mod unified_diff_cpu;
pub mod unified_test_kernel;

// Note: This module intentionally exposes many variants under their own submodules
// so developers can clearly see the different execution strategies available.