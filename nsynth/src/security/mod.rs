//! Security analysis and vulnerability detection modules.
//!
//! This module provides tools for analyzing synthesized code for security issues,
//! including static analysis, taint analysis, and vulnerability detection.

pub mod vulnerability;

pub use vulnerability::{
    scan_vulnerabilities, scan_vulnerabilities_with_config, Confidence, Location, ScanConfig,
    ScanResult, TaintSink, TaintSource, VulnerabilityFinding, VulnerabilityScanner,
    VulnerabilityType, VULN_PATTERNS,
};
