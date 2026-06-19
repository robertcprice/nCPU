//! Sandboxed execution environment for safely running synthesized code.
//!
//! This module provides comprehensive sandboxing capabilities including:
//! - Process isolation with resource limits
//! - Timeout enforcement
//! - Multi-language support (Rust, JavaScript, Python)
//! - Detailed error capture and reporting
//! - Security isolation features

pub mod sandbox;

pub use sandbox::{
    Example, ExecutionMetrics, ExecutionResult, InputValue, Language, Sandbox,
    SandboxConfig, SandboxError, VerificationReport,
};
