//! Algorithm selection module for nsynth optimization pipeline.
//!
//! This module provides intelligent algorithm selection based on input characteristics,
//! constraints, and performance profiling feedback. It implements a decision tree with
//! confidence scoring to choose optimal algorithms for search and sort operations.

use std::fmt;

/// Algorithm choices available for different operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgorithmChoice {
    /// Linear search - O(n) time, O(1) space
    LinearSearch,
    /// Binary search - O(log n) time, O(1) space, requires sorted data
    BinarySearch,
    /// B-tree search - O(log n) time, O(n) space, good for cached scenarios
    BTreeSearch,
    /// Hash map lookup - O(1) average time, O(n) space
    HashMapLookup,
    /// Quick sort - O(n log n) average, O(n²) worst, in-place
    QuickSort,
    /// Merge sort - O(n log n) guaranteed, O(n) space, stable
    MergeSort,
    /// Heap sort - O(n log n) guaranteed, O(1) space, not stable
    HeapSort,
}

impl fmt::Display for AlgorithmChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearSearch => write!(f, "LinearSearch"),
            Self::BinarySearch => write!(f, "BinarySearch"),
            Self::BTreeSearch => write!(f, "BTreeSearch"),
            Self::HashMapLookup => write!(f, "HashMapLookup"),
            Self::QuickSort => write!(f, "QuickSort"),
            Self::MergeSort => write!(f, "MergeSort"),
            Self::HeapSort => write!(f, "HeapSort"),
        }
    }
}

/// Operation type for algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Search,
    Sort,
}

/// Constraints for algorithm selection.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Available memory in bytes
    pub memory_available: Option<usize>,
    /// Time constraint in microseconds
    pub time_limit_us: Option<u64>,
    /// Whether data is sorted
    pub is_sorted: bool,
    /// Degree of sortedness (0.0 = random, 1.0 = fully sorted)
    pub sortedness: f64,
    /// Whether stability is required
    pub requires_stability: bool,
    /// Whether in-place sorting is required
    pub in_place_required: bool,
    /// Expected access pattern (sequential vs random)
    pub access_pattern: AccessPattern,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            memory_available: None,
            time_limit_us: None,
            is_sorted: false,
            sortedness: 0.0,
            requires_stability: false,
            in_place_required: false,
            access_pattern: AccessPattern::Unknown,
        }
    }
}

/// Access pattern for data operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Unknown access pattern
    Unknown,
}

/// Input characteristics for algorithm selection.
#[derive(Debug, Clone)]
pub struct InputCharacteristics {
    /// Number of elements
    pub data_size: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Whether the data fits in cache
    pub fits_in_cache: bool,
    /// Distribution of values (uniform, skewed, normal)
    pub value_distribution: ValueDistribution,
}

impl Default for InputCharacteristics {
    fn default() -> Self {
        Self {
            data_size: 0,
            element_size: std::mem::size_of::<u8>(),
            fits_in_cache: false,
            value_distribution: ValueDistribution::Unknown,
        }
    }
}

/// Distribution of values in the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueDistribution {
    /// Uniform distribution
    Uniform,
    /// Skewed distribution (few unique values)
    Skewed,
    /// Normal distribution
    Normal,
    /// Unknown distribution
    Unknown,
}

/// Result of algorithm selection with confidence score.
#[derive(Debug, Clone)]
pub struct AlgorithmSelection {
    /// Selected algorithm
    pub algorithm: AlgorithmChoice,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Reasoning for the selection
    pub reasoning: String,
}

/// Algorithm selector with constraint analysis and decision logic.
pub struct AlgorithmSelector {
    /// Cache size in bytes for determining if data fits in cache
    cache_size: usize,
    /// Profile feedback from previous runs (algorithm -> average performance)
    profile_feedback: std::collections::HashMap<AlgorithmChoice, f64>,
}

impl AlgorithmSelector {
    /// Create a new algorithm selector with default L3 cache size.
    pub fn new() -> Self {
        Self::with_cache_size(8 * 1024 * 1024) // 8MB default L3 cache
    }

    /// Create a new algorithm selector with specified cache size.
    pub fn with_cache_size(cache_size: usize) -> Self {
        Self {
            cache_size,
            profile_feedback: std::collections::HashMap::new(),
        }
    }

    /// Analyze constraints and determine feasibility of algorithms.
    pub fn analyze_constraints(
        &self,
        constraints: &Constraints,
        input: &InputCharacteristics,
    ) -> ConstraintAnalysis {
        let mut feasible = vec![
            AlgorithmChoice::LinearSearch,
            AlgorithmChoice::BinarySearch,
            AlgorithmChoice::BTreeSearch,
            AlgorithmChoice::HashMapLookup,
            AlgorithmChoice::QuickSort,
            AlgorithmChoice::MergeSort,
            AlgorithmChoice::HeapSort,
        ];

        let mut reasons = Vec::new();

        // Memory constraints
        if let Some(mem_avail) = constraints.memory_available {
            let total_data = input.data_size * input.element_size;

            // Hash map requires ~2x memory overhead
            if total_data * 2 > mem_avail {
                feasible.retain(|&a| a != AlgorithmChoice::HashMapLookup);
                reasons.push(format!(
                    "HashMapLookup excluded: requires ~{} bytes, only {} available",
                    total_data * 2,
                    mem_avail
                ));
            }

            // Merge sort requires O(n) additional space
            if total_data > mem_avail {
                feasible.retain(|&a| a != AlgorithmChoice::MergeSort);
                reasons.push(format!(
                    "MergeSort excluded: requires additional {} bytes for merge buffer",
                    total_data
                ));
            }
        }

        // Sortedness constraints
        if !constraints.is_sorted && constraints.sortedness < 0.9 {
            feasible.retain(|&a| a != AlgorithmChoice::BinarySearch);
            feasible.retain(|&a| a != AlgorithmChoice::BTreeSearch);
            reasons.push(format!(
                "BinarySearch and BTreeSearch excluded: data not sorted (sortedness: {:.2})",
                constraints.sortedness
            ));
        }

        // Stability constraints
        if constraints.requires_stability {
            feasible.retain(|&a| matches!(a, AlgorithmChoice::MergeSort));
            reasons.push(
                "Stability required: only MergeSort is stable among O(n log n) algorithms".to_string()
            );
        }

        // In-place constraints
        if constraints.in_place_required {
            feasible.retain(|&a| {
                matches!(a, AlgorithmChoice::QuickSort | AlgorithmChoice::HeapSort | AlgorithmChoice::LinearSearch)
            });
            reasons.push(
                "In-place required: excluding algorithms requiring additional space".to_string()
            );
        }

        ConstraintAnalysis {
            feasible_algorithms: feasible,
            reasons,
        }
    }

    /// Select the best algorithm based on input characteristics and constraints.
    pub fn select_algorithm(
        &mut self,
        operation: OperationType,
        input: &InputCharacteristics,
        constraints: &Constraints,
    ) -> AlgorithmSelection {
        // Analyze constraints first
        let analysis = self.analyze_constraints(constraints, input);

        // Decision tree for selection
        let (algorithm, confidence, reasoning) = match operation {
            OperationType::Search => self.select_search_algorithm(input, constraints, &analysis),
            OperationType::Sort => self.select_sort_algorithm(input, constraints, &analysis),
        };

        // Check profile feedback for confidence adjustment
        let adjusted_confidence = self.adjust_confidence_with_feedback(
            algorithm,
            confidence,
        );

        AlgorithmSelection {
            algorithm,
            confidence: adjusted_confidence,
            reasoning,
        }
    }

    /// Select best search algorithm using decision tree.
    fn select_search_algorithm(
        &self,
        input: &InputCharacteristics,
        constraints: &Constraints,
        analysis: &ConstraintAnalysis,
    ) -> (AlgorithmChoice, f64, String) {
        let size = input.data_size;

        // Small data: linear search is fastest due to simplicity and cache effects
        if size < 64 {
            return (
                AlgorithmChoice::LinearSearch,
                0.95,
                format!(
                    "Small dataset (n={}): LinearSearch wins on simplicity and cache locality",
                    size
                ),
            );
        }

        // Check if binary search is feasible (sorted data)
        if analysis.feasible_algorithms.contains(&AlgorithmChoice::BinarySearch) {
            // For large sorted datasets, binary search is optimal
            if size > 1000 {
                return (
                    AlgorithmChoice::BinarySearch,
                    0.98,
                    format!(
                        "Large sorted dataset (n={}, sortedness={:.2}): BinarySearch provides O(log n) performance",
                        size,
                        constraints.sortedness
                    ),
                );
            }
            // Medium sized sorted data: binary search still wins
            return (
                AlgorithmChoice::BinarySearch,
                0.90,
                format!(
                    "Medium sorted dataset (n={}): BinarySearch优于 LinearSearch",
                    size
                ),
            );
        }

        // For unsorted data, decide between HashMap and LinearSearch
        if analysis.feasible_algorithms.contains(&AlgorithmChoice::HashMapLookup) {
            // Large unsorted data: hash map is worth the memory cost
            if size > 1000 {
                return (
                    AlgorithmChoice::HashMapLookup,
                    0.92,
                    format!(
                        "Large unsorted dataset (n={}): HashMapLookup justifies memory cost for O(1) access",
                        size
                    ),
                );
            }
        }

        // Default to linear search
        (
            AlgorithmChoice::LinearSearch,
            0.85,
            format!(
                "Default choice for n={}: LinearSearch is simple and cache-friendly",
                size
            ),
        )
    }

    /// Select best sort algorithm using decision tree.
    fn select_sort_algorithm(
        &self,
        input: &InputCharacteristics,
        constraints: &Constraints,
        analysis: &ConstraintAnalysis,
    ) -> (AlgorithmChoice, f64, String) {
        let size = input.data_size;

        // Tiny datasets: insertion sort would be best, but we default to quicksort
        if size < 16 {
            return (
                AlgorithmChoice::QuickSort,
                0.80,
                "Tiny dataset: QuickSort has acceptable overhead".to_string(),
            );
        }

        // Check if only one algorithm is feasible due to constraints
        if analysis.feasible_algorithms.len() == 1 {
            let algo = analysis.feasible_algorithms[0];
            return (
                algo,
                1.0,
                format!(
                    "Constraints mandate {:?}: {}",
                    algo,
                    analysis.reasons.join("; ")
                ),
            );
        }

        // Data fits in cache: quicksort excels with good cache locality
        if input.fits_in_cache {
            if analysis.feasible_algorithms.contains(&AlgorithmChoice::QuickSort) {
                return (
                    AlgorithmChoice::QuickSort,
                    0.93,
                    format!(
                        "Data fits in cache ({} bytes): QuickSort maximizes cache locality",
                        input.data_size * input.element_size
                    ),
                );
            }
        }

        // Memory constrained: heap sort for O(1) space guarantee
        if constraints.memory_available.is_some() &&
           analysis.feasible_algorithms.contains(&AlgorithmChoice::HeapSort) {
            return (
                AlgorithmChoice::HeapSort,
                0.88,
                "Memory constrained: HeapSort provides guaranteed O(n log n) with O(1) space".to_string(),
            );
        }

        // Stability required: merge sort is the choice
        if constraints.requires_stability {
            return (
                AlgorithmChoice::MergeSort,
                0.95,
                "Stability required: MergeSort is the stable O(n log n) choice".to_string(),
            );
        }

        // For data that doesn't fit in cache, consider access pattern
        match constraints.access_pattern {
            AccessPattern::Sequential => {
                // Merge sort has better sequential access patterns
                if analysis.feasible_algorithms.contains(&AlgorithmChoice::MergeSort) {
                    return (
                        AlgorithmChoice::MergeSort,
                        0.87,
                        "Sequential access pattern: MergeSort has predictable memory access".to_string(),
                    );
                }
            }
            AccessPattern::Random => {
                // Quick sort handles random access better
                if analysis.feasible_algorithms.contains(&AlgorithmChoice::QuickSort) {
                    return (
                        AlgorithmChoice::QuickSort,
                        0.89,
                        "Random access pattern: QuickSort works well with random access".to_string(),
                    );
                }
            }
            AccessPattern::Unknown => {}
        }

        // Partially sorted data: consider timsort (not implemented, default to merge sort)
        if constraints.sortedness > 0.5 &&
           analysis.feasible_algorithms.contains(&AlgorithmChoice::MergeSort) {
            return (
                AlgorithmChoice::MergeSort,
                0.86,
                format!(
                    "Partially sorted (sortedness={:.2}): MergeSort can exploit existing order",
                    constraints.sortedness
                ),
            );
        }

        // Default: quick sort for general case
        if analysis.feasible_algorithms.contains(&AlgorithmChoice::QuickSort) {
            return (
                AlgorithmChoice::QuickSort,
                0.84,
                "Default choice: QuickSort provides good average-case performance".to_string(),
            );
        }

        // Fallback to merge sort if quicksort not available
        if analysis.feasible_algorithms.contains(&AlgorithmChoice::MergeSort) {
            return (
                AlgorithmChoice::MergeSort,
                0.82,
                "Fallback to MergeSort: reliable O(n log n) guarantee".to_string(),
            );
        }

        // Last resort: heap sort
        (
            AlgorithmChoice::HeapSort,
            0.75,
            "Last resort: HeapSort guarantees O(n log n) with O(1) space".to_string(),
        )
    }

    /// Adjust confidence score based on profile feedback.
    fn adjust_confidence_with_feedback(&self, algorithm: AlgorithmChoice, base_confidence: f64) -> f64 {
        if let Some(&feedback_score) = self.profile_feedback.get(&algorithm) {
            // Blend base confidence with feedback score
            // feedback_score should be normalized to 0-1 range
            (base_confidence * 0.7 + feedback_score * 0.3).min(1.0)
        } else {
            base_confidence
        }
    }

    /// Update profile feedback with new performance data.
    pub fn update_feedback(&mut self, algorithm: AlgorithmChoice, performance_score: f64) {
        // Normalize performance score to 0-1 range
        // Assume higher is better, normalize by expected best performance
        let normalized = (performance_score / 100.0).min(1.0).max(0.0);

        // Exponential moving average update
        let current = self.profile_feedback.get(&algorithm).copied().unwrap_or(normalized);
        let updated = current * 0.8 + normalized * 0.2;

        self.profile_feedback.insert(algorithm, updated);
    }

    /// Get current profile feedback for an algorithm.
    pub fn get_feedback(&self, algorithm: AlgorithmChoice) -> Option<f64> {
        self.profile_feedback.get(&algorithm).copied()
    }

    /// Reset all profile feedback.
    pub fn reset_feedback(&mut self) {
        self.profile_feedback.clear();
    }
}

impl Default for AlgorithmSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of constraint analysis.
#[derive(Debug, Clone)]
pub struct ConstraintAnalysis {
    /// Algorithms that satisfy all constraints
    pub feasible_algorithms: Vec<AlgorithmChoice>,
    /// Reasons for infeasibility of excluded algorithms
    pub reasons: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_data_linear_search() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 32,
            ..Default::default()
        };
        let constraints = Constraints::default();

        let selection = selector.select_algorithm(OperationType::Search, &input, &constraints);

        assert_eq!(selection.algorithm, AlgorithmChoice::LinearSearch);
        assert!(selection.confidence > 0.9);
    }

    #[test]
    fn test_sorted_binary_search() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 10000,
            ..Default::default()
        };
        let constraints = Constraints {
            is_sorted: true,
            sortedness: 1.0,
            ..Default::default()
        };

        let selection = selector.select_algorithm(OperationType::Search, &input, &constraints);

        assert_eq!(selection.algorithm, AlgorithmChoice::BinarySearch);
        assert!(selection.confidence > 0.9);
    }

    #[test]
    fn test_stability_requires_merge_sort() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 1000,
            ..Default::default()
        };
        let constraints = Constraints {
            requires_stability: true,
            ..Default::default()
        };

        let selection = selector.select_algorithm(OperationType::Sort, &input, &constraints);

        assert_eq!(selection.algorithm, AlgorithmChoice::MergeSort);
    }

    #[test]
    fn test_memory_constrained_heap_sort() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 10000,
            element_size: 100,
            ..Default::default()
        };
        let constraints = Constraints {
            memory_available: Some(500_000), // Less than data size
            in_place_required: true,
            ..Default::default()
        };

        let selection = selector.select_algorithm(OperationType::Sort, &input, &constraints);

        // Heap sort should be selected for in-place memory constraint
        assert!(matches!(selection.algorithm, AlgorithmChoice::HeapSort | AlgorithmChoice::QuickSort));
    }

    #[test]
    fn test_feedback_update() {
        let mut selector = AlgorithmSelector::new();

        selector.update_feedback(AlgorithmChoice::QuickSort, 80.0);

        let feedback = selector.get_feedback(AlgorithmChoice::QuickSort);
        assert!(feedback.is_some());
        assert!(feedback.unwrap() > 0.0);
    }

    #[test]
    fn test_constraint_analysis_memory() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 1000,
            element_size: 1000,
            ..Default::default()
        };
        let constraints = Constraints {
            memory_available: Some(500_000), // Less than 2x data size
            ..Default::default()
        };

        let analysis = selector.analyze_constraints(&constraints, &input);

        // HashMap should be excluded due to memory constraints
        assert!(!analysis.feasible_algorithms.contains(&AlgorithmChoice::HashMapLookup));
    }

    #[test]
    fn test_unsorted_binary_search_excluded() {
        let mut selector = AlgorithmSelector::new();
        let input = InputCharacteristics {
            data_size: 1000,
            ..Default::default()
        };
        let constraints = Constraints {
            is_sorted: false,
            sortedness: 0.3,
            ..Default::default()
        };

        let analysis = selector.analyze_constraints(&constraints, &input);

        assert!(!analysis.feasible_algorithms.contains(&AlgorithmChoice::BinarySearch));
        assert!(!analysis.feasible_algorithms.contains(&AlgorithmChoice::BTreeSearch));
    }
}
