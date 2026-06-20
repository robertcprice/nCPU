//! Performance regression tests for nSynth
//!
//! Benchmark suite to detect performance degradation across versions

use crate::benchmark::{Value, Example};
use super::{RobustnessTestCase, TestCategory, Complexity, PerformanceBaseline};
use std::time::Duration;

/// Baseline performance metrics
pub fn get_baseline() -> PerformanceBaseline {
    PerformanceBaseline {
        median_latency_ms: 100.0,
        p95_latency_ms: 500.0,
        p99_latency_ms: 1000.0,
        throughput_per_sec: 10.0,
        memory_mb: 100.0,
    }
}

/// Generate performance test cases
pub fn generate_performance_tests() -> Vec<RobustnessTestCase> {
    vec![
        // Micro-benchmarks
        RobustnessTestCase {
            name: "perf_add_two_ints".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Trivial,
            problem: "Add two integers".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_multiply_ints".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Trivial,
            problem: "Multiply two integers".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_array_access".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Access element from array by index".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_string_concat".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Concatenate two strings".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 2048,
        },
        // Sorting benchmarks
        RobustnessTestCase {
            name: "perf_sort_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Sort array of 100 integers".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "perf_sort_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Sort array of 1000 integers".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "perf_sort_10000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Sort array of 10000 integers".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(2),
            max_memory: 1024 * 1024,
        },
        // Search benchmarks
        RobustnessTestCase {
            name: "perf_linear_search_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Linear search in array of 100 elements".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_binary_search_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Binary search in sorted array of 1000 elements".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(5),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_binary_search_10000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Binary search in sorted array of 10000 elements".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        // Iteration benchmarks
        RobustnessTestCase {
            name: "perf_map_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Map function over array of 100 elements".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "perf_map_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Map function over array of 1000 elements".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "perf_filter_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Filter array of 100 elements by predicate".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "perf_reduce_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Reduce array of 100 elements to single value".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(30),
            max_memory: 10 * 1024,
        },
        // Recursion benchmarks
        RobustnessTestCase {
            name: "perf_recursive_fib_10".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Calculate 10th Fibonacci number recursively".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "perf_recursive_fib_20".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Calculate 20th Fibonacci number recursively".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "perf_iterative_fib_50".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Calculate 50th Fibonacci number iteratively".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        // Data structure benchmarks
        RobustnessTestCase {
            name: "perf_hashmap_insert_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Insert 100 key-value pairs into hash map".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "perf_hashmap_lookup_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Lookup 100 keys from hash map".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "perf_tree_traversal_100".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Traverse binary tree with 100 nodes".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 20 * 1024,
        },
        // String benchmarks
        RobustnessTestCase {
            name: "perf_string_split_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Split string of 1000 characters by delimiter".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "perf_string_replace_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Replace all occurrences in string of 1000 characters".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(30),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "perf_string_reverse_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Simple,
            problem: "Reverse string of 1000 characters".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 50 * 1024,
        },
        // Algorithm benchmarks
        RobustnessTestCase {
            name: "perf_gcd_euclidean".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Calculate GCD of two numbers using Euclidean algorithm".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_prime_check_10000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Check if number up to 10000 is prime".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 1024,
        },
        RobustnessTestCase {
            name: "perf_factorial_20".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Calculate factorial of 20".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(10),
            max_memory: 1024,
        },
        // Matrix benchmarks
        RobustnessTestCase {
            name: "perf_matrix_add_10x10".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Add two 10x10 matrices".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(20),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "perf_matrix_multiply_10x10".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Multiply two 10x10 matrices".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 20 * 1024,
        },
        RobustnessTestCase {
            name: "perf_matrix_transpose_50x50".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Transpose 50x50 matrix".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 50 * 1024,
        },
        // List operations
        RobustnessTestCase {
            name: "perf_list_reverse_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Reverse linked list of 1000 nodes".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 20 * 1024,
        },
        RobustnessTestCase {
            name: "perf_list_append_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Append 1000 elements to list".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 30 * 1024,
        },
        // Throughput benchmarks
        RobustnessTestCase {
            name: "perf_throughput_simple_10000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Process 10000 simple operations in sequence".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(2),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "perf_throughput_complex_1000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::VeryComplex,
            problem: "Process 1000 complex operations in sequence".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(5),
            max_memory: 200 * 1024,
        },
        // Memory efficiency
        RobustnessTestCase {
            name: "perf_memory_efficient_sort".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Medium,
            problem: "Sort 1000 elements using memory-efficient algorithm".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "perf_in_place_operations".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Perform operations in-place without allocation".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
        // Parallelism potential
        RobustnessTestCase {
            name: "perf_parallelizable_map_10000".to_string(),
            category: TestCategory::Performance,
            complexity: Complexity::Complex,
            problem: "Map operation over 10000 elements (parallelizable)".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 200 * 1024,
        },
    ]
}

/// Performance metrics for a single test run
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub name: String,
    pub duration_ms: f64,
    pub memory_bytes: usize,
    pub operations_per_sec: f64,
    pub regression_detected: bool,
}

/// Calculate regression threshold based on baseline
pub fn regression_threshold(baseline: &PerformanceBaseline, percentile: f64) -> Duration {
    let threshold_ms = baseline.p99_latency_ms * percentile;
    Duration::from_millis(threshold_ms as u64)
}

/// Detect performance regression
pub fn detect_regression(
    current: &PerformanceMetrics,
    baseline: &PerformanceBaseline,
) -> bool {
    // Regression if:
    // 1. Duration > 1.5x baseline p99
    // 2. Memory > 2x baseline memory
    let duration_threshold = baseline.p99_latency_ms * 1.5;
    let memory_threshold = baseline.memory_mb * 1024.0 * 1024.0 * 2.0;

    current.duration_ms > duration_threshold
        || current.memory_bytes as f64 > memory_threshold
}

/// Get performance test examples
pub fn performance_test_examples(name: &str) -> Vec<Example> {
    match name {
        "perf_add_two_ints" => vec![
            (vec![Value::Int(5), Value::Int(3)], Value::Int(8)),
            (vec![Value::Int(10), Value::Int(-5)], Value::Int(5)),
        ],
        "perf_sort_100" => vec![
            (vec![Value::Array((0..100).rev().map(Value::Int).collect())], Value::Array((0..100).map(Value::Int).collect())),
        ],
        "perf_binary_search_1000" => vec![
            (vec![Value::Array((0..1000).map(Value::Int).collect()), Value::Int(500)], Value::Int(500)),
        ],
        "perf_map_100" => vec![
            (vec![Value::Array((0..100).map(Value::Int).collect())], Value::Array((0..100).map(|i| Value::Int(i * 2)).collect())),
        ],
        "perf_iterative_fib_50" => vec![
            (vec![Value::Int(10)], Value::Int(55)),
        ],
        "perf_gcd_euclidean" => vec![
            (vec![Value::Int(48), Value::Int(18)], Value::Int(6)),
            (vec![Value::Int(1071), Value::Int(462)], Value::Int(21)),
        ],
        "perf_factorial_20" => vec![
            (vec![Value::Int(5)], Value::Int(120)),
            (vec![Value::Int(10)], Value::Int(3628800)),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_test_generation() {
        let tests = generate_performance_tests();
        assert!(tests.len() > 30);

        // Verify all tests are performance category
        for test in &tests {
            assert_eq!(test.category, TestCategory::Performance);
        }
    }

    #[test]
    fn test_baseline_creation() {
        let baseline = get_baseline();
        assert!(baseline.median_latency_ms > 0.0);
        assert!(baseline.p95_latency_ms > baseline.median_latency_ms);
        assert!(baseline.p99_latency_ms > baseline.p95_latency_ms);
    }

    #[test]
    fn test_regression_detection() {
        let baseline = get_baseline();
        let normal = PerformanceMetrics {
            name: "test".to_string(),
            duration_ms: 500.0,
            memory_bytes: 50 * 1024 * 1024,
            operations_per_sec: 1000.0,
            regression_detected: false,
        };
        let slow = PerformanceMetrics {
            name: "test".to_string(),
            duration_ms: 2000.0,  // 2x baseline p99
            memory_bytes: 50 * 1024 * 1024,
            operations_per_sec: 250.0,
            regression_detected: false,
        };

        assert!(!detect_regression(&normal, &baseline));
        assert!(detect_regression(&slow, &baseline));
    }

    #[test]
    fn test_regression_threshold() {
        let baseline = get_baseline();
        let threshold = regression_threshold(&baseline, 1.5);
        assert_eq!(threshold, Duration::from_millis(1500));
    }

    #[test]
    fn test_performance_examples() {
        let examples = performance_test_examples("perf_add_two_ints");
        assert_eq!(examples.len(), 2);

        let empty = performance_test_examples("nonexistent");
        assert!(empty.is_empty());
    }
}
