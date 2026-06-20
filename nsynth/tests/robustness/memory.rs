//! Memory leak detection and memory usage tests for nSynth
//!
//! Track memory allocation patterns and detect leaks

use crate::benchmark::{Value, Example};
use super::{RobustnessTestCase, TestCategory, Complexity};
use std::time::Duration;

/// Memory leak detection result
#[derive(Debug, Clone)]
pub struct MemoryLeakResult {
    pub test_name: String,
    pub initial_memory_kb: usize,
    pub peak_memory_kb: usize,
    pub final_memory_kb: usize,
    pub leaked_bytes: usize,
    pub leak_detected: bool,
    pub leak_severity: LeakSeverity,
}

/// Severity of memory leak
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeakSeverity {
    None,
    Minor,    // < 1MB leak
    Moderate, // 1-10MB leak
    Severe,   // > 10MB leak
}

/// Memory profile for a test
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    pub allocations: usize,
    pub deallocations: usize,
    pub peak_bytes: usize,
    pub current_bytes: usize,
    pub allocation_count: usize,
}

/// Generate memory test cases
pub fn generate_memory_tests() -> Vec<RobustnessTestCase> {
    vec![
        // Basic allocation tests
        RobustnessTestCase {
            name: "mem_allocate_deallocate_simple".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Simple,
            problem: "Allocate and deallocate simple array".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "mem_allocate_deallocate_nested".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Allocate and deallocate nested structures".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 100 * 1024,
        },
        // Array operations
        RobustnessTestCase {
            name: "mem_array_grow_shrink".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Grow array to 10000 elements then shrink to 10".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 1024 * 1024,
        },
        RobustnessTestCase {
            name: "mem_array_slice_operations".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Perform slice operations without copying".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 100 * 1024,
        },
        // String operations
        RobustnessTestCase {
            name: "mem_string_concat_loop".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Concatenate 1000 strings in loop efficiently".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 200 * 1024,
        },
        RobustnessTestCase {
            name: "mem_string_builder_pattern".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Build large string using builder pattern".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 500 * 1024,
        },
        RobustnessTestCase {
            name: "mem_string_release".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Simple,
            problem: "Release large string after use".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 1024 * 1024,
        },
        // Recursive structures
        RobustnessTestCase {
            name: "mem_recursive_list_build".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Build and release recursive list of 1000 nodes".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(2),
            max_memory: 200 * 1024,
        },
        RobustnessTestCase {
            name: "mem_tree_build_release".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Build binary tree and release all nodes".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(2),
            max_memory: 500 * 1024,
        },
        // Closure capture tests
        RobustnessTestCase {
            name: "mem_closure_small_capture".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Capture small values in closure without leak".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "mem_closure_large_capture".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Capture large array in closure without leak".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 1024 * 1024,
        },
        // Iterator patterns
        RobustnessTestCase {
            name: "mem_iterator_chain".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Chain iterator operations without intermediate allocations".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "mem_lazy_evaluation".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Use lazy evaluation to avoid allocations".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 50 * 1024,
        },
        // Collection reclamation
        RobustnessTestCase {
            name: "mem_hashmap_clear".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Clear hash map and verify memory released".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 100 * 1024,
        },
        RobustnessTestCase {
            name: "mem_vec_clear_shrink".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Clear vector and shrink capacity".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 1024 * 1024,
        },
        // Stress tests
        RobustnessTestCase {
            name: "mem_stress_allocate_100000".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::VeryComplex,
            problem: "Allocate and deallocate 100000 small objects".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(5),
            max_memory: 50 * 1024 * 1024,
        },
        RobustnessTestCase {
            name: "mem_stress_large_allocations".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::VeryComplex,
            problem: "Allocate 100 large 10MB buffers then release".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(10),
            max_memory: 1024 * 1024 * 1024,
        },
        // Reference counting
        RobustnessTestCase {
            name: "mem_rc_cycle_detection".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Detect and break reference cycles".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 50 * 1024,
        },
        RobustnessTestCase {
            name: "mem_arc_clone_release".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Clone and release Arc references".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(200),
            max_memory: 20 * 1024,
        },
        // Buffer reuse
        RobustnessTestCase {
            name: "mem_buffer_reuse".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Reuse buffer across operations".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "mem_pooling_pattern".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Use object pooling to reduce allocations".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(2),
            max_memory: 100 * 1024,
        },
        // Zero-copy operations
        RobustnessTestCase {
            name: "mem_zero_copy_slice".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Use slice instead of copy for view".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "mem_borrow_instead_of_clone".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Simple,
            problem: "Use borrow instead of clone".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 5 * 1024,
        },
        // Memory fragmentation
        RobustnessTestCase {
            name: "mem_fragmentation_resistance".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::VeryComplex,
            problem: "Handle allocation pattern that causes fragmentation".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(5),
            max_memory: 50 * 1024 * 1024,
        },
        // Shared state
        RobustnessTestCase {
            name: "mem_mutex_shared_state".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Share state via mutex without leak".to_string(),
            expected_success: true,
            max_duration: Duration::from_secs(1),
            max_memory: 20 * 1024,
        },
        // Stack vs heap
        RobustnessTestCase {
            name: "mem_stack_preference".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Simple,
            problem: "Prefer stack allocation for small objects".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(50),
            max_memory: 2 * 1024,
        },
        RobustnessTestCase {
            name: "mem_heap_minimization".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Complex,
            problem: "Minimize heap allocations in hot path".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(500),
            max_memory: 10 * 1024,
        },
        // Cleanup patterns
        RobustnessTestCase {
            name: "mem_drop_guard".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Use drop guards for cleanup".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
        RobustnessTestCase {
            name: "mem_scope_based_cleanup".to_string(),
            category: TestCategory::Memory,
            complexity: Complexity::Medium,
            problem: "Scope-based resource cleanup".to_string(),
            expected_success: true,
            max_duration: Duration::from_millis(100),
            max_memory: 10 * 1024,
        },
    ]
}

/// Detect memory leak from measurements
pub fn detect_leak(
    initial_memory_kb: usize,
    peak_memory_kb: usize,
    final_memory_kb: usize,
) -> MemoryLeakResult {
    let leaked_bytes = final_memory_kb.saturating_sub(initial_memory_kb) * 1024;
    let leak_detected = leaked_bytes > 100 * 1024; // > 100KB
    let leak_severity = if leaked_bytes < 1024 * 1024 {
        LeakSeverity::Minor
    } else if leaked_bytes < 10 * 1024 * 1024 {
        LeakSeverity::Moderate
    } else {
        LeakSeverity::Severe
    };

    MemoryLeakResult {
        test_name: String::new(), // Set by caller
        initial_memory_kb,
        peak_memory_kb,
        final_memory_kb,
        leaked_bytes,
        leak_detected,
        leak_severity,
    }
}

/// Check if memory usage is within acceptable bounds
pub fn memory_within_bounds(used_bytes: usize, max_bytes: usize) -> bool {
    used_bytes <= max_bytes
}

/// Calculate memory efficiency score (0.0 to 1.0)
pub fn memory_efficiency_score(
    used_bytes: usize,
    max_bytes: usize,
    leaked_bytes: usize,
) -> f64 {
    let usage_ratio = used_bytes as f64 / max_bytes.max(1) as f64;
    let leak_penalty = (leaked_bytes as f64 / (1024.0 * 1024.0)).min(1.0);

    (1.0 - usage_ratio * 0.5 - leak_penalty * 0.5).max(0.0)
}

/// Get memory test examples
pub fn memory_test_examples(name: &str) -> Vec<Example> {
    match name {
        "mem_array_grow_shrink" => vec![
            (vec![Value::Array((0..100).map(Value::Int).collect())], Value::Array((0..10).map(Value::Int).collect())),
        ],
        "mem_string_concat_loop" => vec![
            (vec![Value::Array((0..100).map(|i| Value::String(format!("item_{}", i))).collect())], Value::String("".to_string())),
        ],
        "mem_tree_build_release" => vec![
            (vec![Value::Int(10)], Value::Unit),
        ],
        _ => vec![],
    }
}

/// Memory statistics for benchmarking
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub peak_memory_bytes: usize,
    pub current_memory_bytes: usize,
    pub allocation_objects: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            allocation_objects: 0,
        }
    }
}

impl MemoryStats {
    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocations += 1;
        self.current_memory_bytes += size;
        self.peak_memory_bytes = self.peak_memory_bytes.max(self.current_memory_bytes);
        self.allocation_objects += 1;
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size);
        self.allocation_objects = self.allocation_objects.saturating_sub(1);
    }

    /// Check for potential leak
    pub fn potential_leak(&self) -> bool {
        // Leak if: significantly more allocations than deallocations
        // AND current memory is non-trivial
        let alloc_diff = self.total_allocations.saturating_sub(self.total_deallocations);
        alloc_diff > 100 && self.current_memory_bytes > 1024 * 1024
    }

    /// Get leak report
    pub fn leak_report(&self) -> String {
        let alloc_diff = self.total_allocations.saturating_sub(self.total_deallocations);
        format!(
            "Allocations: {}, Deallocations: {}, Net: {}, Peak: {}KB, Current: {}KB",
            self.total_allocations,
            self.total_deallocations,
            alloc_diff,
            self.peak_memory_bytes / 1024,
            self.current_memory_bytes / 1024
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_test_generation() {
        let tests = generate_memory_tests();
        assert!(tests.len() > 20);

        // Verify all tests are memory category
        for test in &tests {
            assert_eq!(test.category, TestCategory::Memory);
        }
    }

    #[test]
    fn test_leak_detection() {
        let result = detect_leak(1000, 5000, 1200); // 200KB leak
        assert!(result.leak_detected);
        assert_eq!(result.leak_severity, LeakSeverity::Minor);

        let no_leak = detect_leak(1000, 5000, 1000); // No leak
        assert!(!no_leak.leak_detected);
        assert_eq!(no_leak.leak_severity, LeakSeverity::None);
    }

    #[test]
    fn test_memory_within_bounds() {
        assert!(memory_within_bounds(1024, 2048));
        assert!(!memory_within_bounds(2048, 1024));
    }

    #[test]
    fn test_memory_efficiency_score() {
        let score1 = memory_efficiency_score(512 * 1024, 1024 * 1024, 0); // 50% usage, no leak
        assert!(score1 > 0.7 && score1 <= 1.0);

        let score2 = memory_efficiency_score(1024 * 1024, 1024 * 1024, 0); // 100% usage
        assert!(score2 < 0.6);

        let score3 = memory_efficiency_score(512 * 1024, 1024 * 1024, 5 * 1024 * 1024); // 5MB leak
        assert!(score3 < 0.5);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::default();
        stats.record_allocation(1024);
        stats.record_allocation(2048);
        stats.record_deallocation(1024);

        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.current_memory_bytes, 2048);
        assert_eq!(stats.peak_memory_bytes, 3072);
        assert!(!stats.potential_leak());

        stats.record_allocation(10 * 1024 * 1024); // Big leak
        assert!(stats.potential_leak());
    }

    #[test]
    fn test_memory_examples() {
        let examples = memory_test_examples("mem_array_grow_shrink");
        assert!(!examples.is_empty());

        let empty = memory_test_examples("nonexistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_leak_severity_levels() {
        let minor = detect_leak(1000, 1000, 1500); // 500KB leak
        assert_eq!(minor.leak_severity, LeakSeverity::Minor);

        let moderate = detect_leak(1000, 1000, 3000); // 2MB leak
        assert_eq!(moderate.leak_severity, LeakSeverity::Moderate);

        let severe = detect_leak(1000, 1000, 15000); // 14MB leak
        assert_eq!(severe.leak_severity, LeakSeverity::Severe);
    }
}
