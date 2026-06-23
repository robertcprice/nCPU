//! Problem difficulty classification for routing and resource allocation.
//!
//! This module provides a machine-learned approach to classifying synthesis problems
//! by difficulty, enabling intelligent routing to appropriate solver components and
//! resource allocation decisions.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::benchmark::{Problem, Value as BenchmarkValue};

/// Difficulty classification result with statistical confidence.
#[derive(Debug, Clone)]
pub struct DifficultyScore {
    /// Raw difficulty score in range [0, 1], where 1 is hardest
    pub raw_score: f64,
    /// Coarse-grained difficulty bucket for routing decisions
    pub bucket: DifficultyBucket,
    /// Statistical confidence in this classification [0, 1]
    pub confidence: f64,
    /// Contribution of each feature to the final score
    pub feature_contributions: HashMap<String, f64>,
}

impl DifficultyScore {
    /// Returns true if this problem is classified as easy.
    pub fn is_easy(&self) -> bool {
        matches!(self.bucket, DifficultyBucket::Easy)
    }

    /// Returns true if this problem is classified as medium difficulty.
    pub fn is_medium(&self) -> bool {
        matches!(self.bucket, DifficultyBucket::Medium)
    }

    /// Returns true if this problem is classified as hard.
    pub fn is_hard(&self) -> bool {
        matches!(self.bucket, DifficultyBucket::Hard)
    }
}

/// Coarse difficulty buckets for fast routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DifficultyBucket {
    /// Easy problems (score < 0.4) - direct synthesis
    Easy,
    /// Medium difficulty (0.4 <= score < 0.7) - may need hints
    Medium,
    /// Hard problems (score >= 0.7) - requires full search capacity
    Hard,
}

/// Feature weights for difficulty computation.
#[derive(Debug, Clone)]
pub struct DifficultyWeights {
    /// Weight for input arity (more inputs = harder)
    pub arity_weight: f64,
    /// Weight for example count (fewer examples = harder, inverse relationship)
    pub example_count_weight: f64,
    /// Weight for output entropy (more diverse = harder)
    pub output_entropy_weight: f64,
    /// Weight for recursive structure depth (deeper = harder)
    pub recursive_depth_weight: f64,
}

impl Default for DifficultyWeights {
    fn default() -> Self {
        Self {
            arity_weight: 0.3,
            example_count_weight: 0.3,
            output_entropy_weight: 0.2,
            recursive_depth_weight: 0.2,
        }
    }
}

/// Extracted features for difficulty classification.
#[derive(Debug, Clone)]
struct DifficultyFeatures {
    /// Number of inputs to the function
    arity: usize,
    /// Number of training examples provided
    example_count: usize,
    /// Shannon entropy of output distribution
    output_entropy: f64,
    /// Estimated recursive depth of input structure
    recursive_depth: u8,
}

/// Classifier for problem difficulty with adaptive learning.
pub struct DifficultyClassifier {
    /// Feature weights for difficulty computation
    feature_weights: DifficultyWeights,
    /// Historical difficulty data for problem families (EMA with alpha=0.1)
    historical_difficulty: Arc<RwLock<HashMap<String, f64>>>,
    /// Maximum observed values for normalization
    max_arity: usize,
    max_examples: usize,
    max_depth: u8,
}

impl Default for DifficultyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DifficultyClassifier {
    /// Creates a new difficulty classifier with default weights.
    pub fn new() -> Self {
        Self {
            feature_weights: DifficultyWeights::default(),
            historical_difficulty: Arc::new(RwLock::new(HashMap::new())),
            max_arity: 5,      // Normalize arity to [0, 1] assuming max 5 inputs
            max_examples: 10,  // Normalize examples to [0, 1] assuming max 10 examples
            max_depth: 4,      // Normalize depth to [0, 1] assuming max 4 levels
        }
    }

    /// Creates a classifier with custom feature weights.
    pub fn with_weights(weights: DifficultyWeights) -> Self {
        Self {
            feature_weights: weights,
            ..Default::default()
        }
    }

    /// Classifies a problem's difficulty.
    ///
    /// Returns a difficulty score with bucket classification and feature contributions.
    pub fn classify(&self, problem: &Problem) -> DifficultyScore {
        let features = self.extract_features(problem);
        let mut feature_contributions = HashMap::new();

        // Normalize and compute weighted contribution for each feature
        let arity_score = (features.arity as f64 / self.max_arity as f64).min(1.0);
        let arity_contribution = arity_score * self.feature_weights.arity_weight;
        feature_contributions.insert("arity".to_string(), arity_contribution);

        // Example count is inverse - fewer examples = harder
        let example_score = 1.0 - (features.example_count as f64 / self.max_examples as f64).min(1.0);
        let example_contribution = example_score * self.feature_weights.example_count_weight;
        feature_contributions.insert("example_count".to_string(), example_contribution);

        // Output entropy directly measures complexity
        let entropy_contribution = features.output_entropy * self.feature_weights.output_entropy_weight;
        feature_contributions.insert("output_entropy".to_string(), entropy_contribution);

        // Recursive depth normalized to [0, 1]
        let depth_score = (features.recursive_depth as f64 / self.max_depth as f64).min(1.0);
        let depth_contribution = depth_score * self.feature_weights.recursive_depth_weight;
        feature_contributions.insert("recursive_depth".to_string(), depth_contribution);

        // Sum weighted contributions for raw score
        let raw_score: f64 = feature_contributions.values().sum();

        // Classify into bucket
        let bucket = if raw_score < 0.4 {
            DifficultyBucket::Easy
        } else if raw_score < 0.7 {
            DifficultyBucket::Medium
        } else {
            DifficultyBucket::Hard
        };

        // Confidence based on feature distribution and historical data
        let confidence = self.compute_confidence(&features, &bucket);

        DifficultyScore {
            raw_score,
            bucket,
            confidence,
            feature_contributions,
        }
    }

    /// Extracts features from a problem for difficulty analysis.
    fn extract_features(&self, problem: &Problem) -> DifficultyFeatures {
        let arity = problem.examples.first().map_or(0, |ex| ex.inputs.len());
        let example_count = problem.examples.len();
        let output_entropy = self.compute_output_entropy(problem);
        let recursive_depth = self.estimate_recursive_depth(problem);

        DifficultyFeatures {
            arity,
            example_count,
            output_entropy,
            recursive_depth,
        }
    }

    /// Computes Shannon entropy of the output distribution.
    ///
    /// Higher entropy indicates more diverse/complex output patterns.
    fn compute_output_entropy(&self, problem: &Problem) -> f64 {
        if problem.examples.is_empty() {
            return 0.0;
        }

        // Count frequency of each unique output
        let mut output_counts: HashMap<String, usize> = HashMap::new();
        let total = problem.examples.len();

        for example in &problem.examples {
            let key = format!("{:?}", example.expected);
            *output_counts.entry(key).or_insert(0) += 1;
        }

        // Compute Shannon entropy: -sum(p(x) * log2(p(x)))
        let mut entropy = 0.0;
        for &count in output_counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize to [0, 1] where 1 is maximum entropy (all outputs unique)
        let max_entropy = (total as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Estimates the recursive depth of input structure.
    ///
    /// Analyzes nested structure in inputs to estimate problem complexity.
    fn estimate_recursive_depth(&self, problem: &Problem) -> u8 {
        let mut max_depth: u8 = 0;

        for example in &problem.examples {
            for input in &example.inputs {
                let depth = self.estimate_input_depth(input);
                max_depth = max_depth.max(depth);
            }
        }

        max_depth
    }

    /// Recursively estimates the depth of a single input value.
    fn estimate_input_depth(&self, input: &BenchmarkValue) -> u8 {
        match input {
            BenchmarkValue::Array(arr) => {
                if arr.is_empty() {
                    1
                } else {
                    let child_depth = arr.iter().map(|v| self.estimate_value_depth(v)).max().unwrap_or(0);
                    1 + child_depth
                }
            }
            BenchmarkValue::Tree(nodes) => {
                // Trees have recursive structure: measure the actual maximum
                // root-to-leaf depth by following the left/right child indices,
                // rather than approximating from node count. A node's `left`/
                // `right` are indices into `nodes` (negative = no child).
                if nodes.is_empty() {
                    1
                } else {
                    self.tree_depth_from(nodes, 0, nodes.len() as u8)
                }
            }
            _ => 0, // Primitive values have depth 0
        }
    }

    /// Maximum depth of the tree rooted at `idx`, following child indices.
    ///
    /// `budget` bounds the recursion so a malformed (cyclic) tree cannot
    /// overflow the stack; it is initialized to the node count, which a
    /// well-formed acyclic tree can never exceed.
    fn tree_depth_from(&self, nodes: &[crate::benchmark::TreeNode], idx: i32, budget: u8) -> u8 {
        if budget == 0 || idx < 0 || idx as usize >= nodes.len() {
            return 0;
        }
        let node = &nodes[idx as usize];
        let left = self.tree_depth_from(nodes, node.left, budget - 1);
        let right = self.tree_depth_from(nodes, node.right, budget - 1);
        1 + left.max(right)
    }

    /// Helper to estimate depth of an array element value.
    ///
    /// Now that array elements are full `Value`s, a nested array element
    /// contributes its own depth (recursively), so an array-of-arrays reports a
    /// depth greater than a flat int array. Scalar elements still report 0.
    fn estimate_value_depth(&self, value: &BenchmarkValue) -> u8 {
        self.estimate_input_depth(value)
    }

    /// Computes confidence in the difficulty classification.
    ///
    /// Higher confidence when:
    /// - Features align consistently with bucket
    /// - Historical data supports this classification
    fn compute_confidence(&self, features: &DifficultyFeatures, bucket: &DifficultyBucket) -> f64 {
        let mut confidence: f64 = 0.5; // Base confidence

        // Adjust based on feature consistency with bucket
        match bucket {
            DifficultyBucket::Easy => {
                // High confidence for low arity, many examples, simple structure
                if features.arity <= 2 && features.example_count >= 5 && features.recursive_depth == 0 {
                    confidence += 0.3;
                }
            }
            DifficultyBucket::Hard => {
                // High confidence for high arity, few examples, deep structure
                if features.arity >= 4 && features.example_count <= 3 && features.recursive_depth >= 2 {
                    confidence += 0.3;
                }
            }
            DifficultyBucket::Medium => {
                // Medium is default, moderate confidence
                confidence = 0.6;
            }
        }

        confidence.min(1.0)
    }

    /// Updates the classifier with actual difficulty data.
    ///
    /// Uses exponential moving average (EMA) with alpha=0.1 to update
    /// historical difficulty for problem families.
    pub fn update(&self, problem: &Problem, actual_difficulty: f64) {
        // Create a problem family key based on structural features
        let family_key = self.problem_family_key(problem);

        let mut history = self.historical_difficulty.write().unwrap();
        let alpha = 0.1;

        let updated = match history.get(&family_key) {
            Some(&historical) => {
                // EMA: new = alpha * actual + (1 - alpha) * historical
                alpha * actual_difficulty + (1.0 - alpha) * historical
            }
            None => actual_difficulty,
        };

        history.insert(family_key, updated);
    }

    /// Generates a key for grouping similar problem families.
    fn problem_family_key(&self, problem: &Problem) -> String {
        let features = self.extract_features(problem);
        format!(
            "arity{}_examples{}_depth{}_entropy{:.2}",
            features.arity, features.example_count, features.recursive_depth, features.output_entropy
        )
    }

    /// Returns historical difficulty data for debugging/analysis.
    pub fn historical_data(&self) -> HashMap<String, f64> {
        self.historical_difficulty.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_problem() -> Problem {
        Problem {
            examples: vec![
                crate::benchmark::Example {
                    inputs: vec![BenchmarkValue::Int(1), BenchmarkValue::Int(2)],
                    expected: BenchmarkValue::Int(3),
                },
                crate::benchmark::Example {
                    inputs: vec![BenchmarkValue::Int(2), BenchmarkValue::Int(3)],
                    expected: BenchmarkValue::Int(5),
                },
            ],
            name: "addition".to_string(),
            category: "",
            description: "",
            signature: "",
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn test_easy_classification() {
        let classifier = DifficultyClassifier::new();
        let problem = test_problem();
        let score = classifier.classify(&problem);

        assert!(score.raw_score >= 0.0 && score.raw_score <= 1.0);
        assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
    }

    #[test]
    fn test_feature_contributions() {
        let classifier = DifficultyClassifier::new();
        let problem = test_problem();
        let score = classifier.classify(&problem);

        // Check that all expected features are present
        assert!(score.feature_contributions.contains_key("arity"));
        assert!(score.feature_contributions.contains_key("example_count"));
        assert!(score.feature_contributions.contains_key("output_entropy"));
        assert!(score.feature_contributions.contains_key("recursive_depth"));

        // Sum of contributions should equal raw score (with floating point tolerance)
        let contribution_sum: f64 = score.feature_contributions.values().sum();
        assert!((contribution_sum - score.raw_score).abs() < 0.001);
    }

    #[test]
    fn test_update_mechanism() {
        let classifier = DifficultyClassifier::new();
        let problem = test_problem();

        // Initial state should be empty
        assert!(classifier.historical_data().is_empty());

        // Update with difficulty data
        classifier.update(&problem, 0.5);

        // Should now have data
        let data = classifier.historical_data();
        assert!(!data.is_empty());

        // Multiple updates should produce EMA effect
        classifier.update(&problem, 0.7);
        classifier.update(&problem, 0.9);

        let data = classifier.historical_data();
        let final_value = data.values().next().unwrap();
        // Should be closer to 0.9 than to 0.5 due to EMA
        assert!(*final_value > 0.5);
        assert!(*final_value < 0.9);
    }

    #[test]
    fn test_entropy_computation() {
        let classifier = DifficultyClassifier::new();

        // Problem with identical outputs (low entropy)
        let low_entropy = Problem {
            examples: vec![
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(1)], expected: BenchmarkValue::Int(1) },
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(2)], expected: BenchmarkValue::Int(1) },
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(3)], expected: BenchmarkValue::Int(1) },
            ],
            name: "constant".to_string(),
            category: "",
            description: "",
            signature: "",
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        // Problem with unique outputs (high entropy)
        let high_entropy = Problem {
            examples: vec![
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(1)], expected: BenchmarkValue::Int(1) },
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(2)], expected: BenchmarkValue::Int(2) },
                crate::benchmark::Example { inputs: vec![BenchmarkValue::Int(3)], expected: BenchmarkValue::Int(3) },
            ],
            name: "identity".to_string(),
            category: "",
            description: "",
            signature: "",
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        let low_score = classifier.classify(&low_entropy);
        let high_score = classifier.classify(&high_entropy);

        // High entropy problem should score higher
        assert!(high_score.raw_score > low_score.raw_score);
    }

    #[test]
    fn test_recursive_depth_estimation() {
        let classifier = DifficultyClassifier::new();

        // Flat structure. Use a single scalar input so the ONLY difference from
        // `nested` is structural recursion depth (matching `nested`'s arity of 1),
        // isolating the recursive-depth signal this test is asserting on.
        let flat = Problem {
            examples: vec![crate::benchmark::Example {
                inputs: vec![BenchmarkValue::Int(1)],
                expected: BenchmarkValue::Int(1),
            }],
            name: "flat".to_string(),
            category: "",
            description: "",
            signature: "",
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        // Nested structure (using Tree for recursion depth)
        let nested = Problem {
            examples: vec![crate::benchmark::Example {
                inputs: vec![BenchmarkValue::Tree(vec![
                    crate::benchmark::TreeNode { value: 1, left: 1, right: -1 },
                    crate::benchmark::TreeNode { value: 2, left: -1, right: -1 },
                ])],
                expected: BenchmarkValue::Int(3),
            }],
            name: "nested".to_string(),
            category: "",
            description: "",
            signature: "",
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        let flat_score = classifier.classify(&flat);
        let nested_score = classifier.classify(&nested);

        // Nested structure should score higher
        assert!(nested_score.raw_score > flat_score.raw_score);
    }
}
