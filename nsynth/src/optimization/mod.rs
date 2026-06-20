//! Optimization module for nsynth code synthesis pipeline.
//!
//! This module provides intelligent optimization capabilities including:
//! - Algorithm selection based on input characteristics
//! - Profiler-guided optimization for hotspots and bottlenecks
//! - Automatic parallelization for independent loops
//! - Multi-language code generation support

pub mod algorithm;
pub mod parallel;
pub mod profiler;

pub use algorithm::{AlgorithmChoice, AlgorithmSelector, Constraints, InputCharacteristics};
pub use parallel::{ParallelConfig, ParallelStrategy, Parallelizer, TargetLanguage};
pub use profiler::{generate_optimizations, profile_code, OptimizationType, ProfiledCode};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_selector() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 1000,
            ..Default::default()
        };
        let constraints = Constraints::default();

        let selection =
            selector.select_algorithm(algorithm::OperationType::Search, &input, &constraints);

        assert!(selection.confidence > 0.0);
    }

    #[test]
    fn test_parallelizer_detection() {
        let code = r#"
fn example() {
    for i in 0..10000 {
        let result = i * 2;
    }
}
"#;

        let parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Rust);

        assert!(!candidates.is_empty());
    }

    #[tokio::test]
    async fn test_profiler_integration() {
        let code = r#"
fn example() {
    for i in 0..10000 {
        for j in 0..10000 {
            let _ = i * j;
        }
    }
}
"#;

        let profiled = profile_code(code, "rust", &profiler::ProfilerConfig::default()).await;

        assert!(profiled.is_ok());
        let profiled = profiled.unwrap();
        assert!(!profiled.hotspots.is_empty() || !profiled.bottlenecks.is_empty());
    }
}
