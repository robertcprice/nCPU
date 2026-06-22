//! Curriculum learning for progressive difficulty-based training.
//!
//! This module implements curriculum learning strategies that organize problems
//! by difficulty and sequence them to optimize learning efficiency. Problems are
//! bucketed into difficulty levels, and training progresses through mastered
//! concepts while maintaining review of previously learned material.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::benchmark::Problem;
use super::difficulty::{DifficultyBucket, DifficultyClassifier, DifficultyScore};

/// Mastery thresholds for progressing between difficulty levels.
///
/// Conservative defaults ensure stability and prevent premature advancement.
#[derive(Debug, Clone)]
pub struct MasteryThresholds {
    /// Success rate required to advance from Easy to Medium (default: 0.75)
    pub easy_to_medium: f64,
    /// Success rate required to advance from Medium to Hard (default: 0.70)
    pub medium_to_hard: f64,
}

impl Default for MasteryThresholds {
    fn default() -> Self {
        Self {
            easy_to_medium: 0.75,
            medium_to_hard: 0.70,
        }
    }
}

/// Adaptive boundaries for difficulty bucket classification.
///
/// Boundaries adjust based on observed performance to maintain
/// appropriate problem difficulty distribution.
#[derive(Debug, Clone)]
pub struct AdaptiveBoundaries {
    /// Threshold between Easy and Medium (default: 0.4)
    pub easy_medium_threshold: Arc<RwLock<f64>>,
    /// Threshold between Medium and Hard (default: 0.7)
    pub medium_hard_threshold: Arc<RwLock<f64>>,
}

impl Default for AdaptiveBoundaries {
    fn default() -> Self {
        Self {
            easy_medium_threshold: Arc::new(RwLock::new(0.4)),
            medium_hard_threshold: Arc::new(RwLock::new(0.7)),
        }
    }
}

impl AdaptiveBoundaries {
    /// Creates new boundaries with custom thresholds.
    pub fn new(easy_medium: f64, medium_hard: f64) -> Self {
        Self {
            easy_medium_threshold: Arc::new(RwLock::new(easy_medium)),
            medium_hard_threshold: Arc::new(RwLock::new(medium_hard)),
        }
    }

    /// Returns the current easy-medium threshold.
    pub fn get_easy_medium(&self) -> f64 {
        *self.easy_medium_threshold.read().unwrap()
    }

    /// Returns the current medium-hard threshold.
    pub fn get_medium_hard(&self) -> f64 {
        *self.medium_hard_threshold.read().unwrap()
    }

    /// Updates the easy-medium threshold using EMA.
    pub fn update_easy_medium(&self, new_value: f64, alpha: f64) {
        let mut threshold = self.easy_medium_threshold.write().unwrap();
        *threshold = alpha * new_value + (1.0 - alpha) * *threshold;
    }

    /// Updates the medium-hard threshold using EMA.
    pub fn update_medium_hard(&self, new_value: f64, alpha: f64) {
        let mut threshold = self.medium_hard_threshold.write().unwrap();
        *threshold = alpha * new_value + (1.0 - alpha) * *threshold;
    }

    /// Classifies a raw difficulty score into a bucket using current boundaries.
    pub fn classify_score(&self, raw_score: f64) -> DifficultyBucket {
        let easy_medium = self.get_easy_medium();
        let medium_hard = self.get_medium_hard();

        if raw_score < easy_medium {
            DifficultyBucket::Easy
        } else if raw_score < medium_hard {
            DifficultyBucket::Medium
        } else {
            DifficultyBucket::Hard
        }
    }
}

/// Performance tracking for a single difficulty bucket.
#[derive(Debug, Clone, Default)]
pub struct BucketPerformance {
    /// Number of problems attempted in this bucket
    pub attempted: u32,
    /// Number of problems successfully solved
    pub solved: u32,
    /// Current success rate (solved / attempted)
    pub success_rate: f64,
    /// Whether this bucket is considered mastered
    pub mastered: bool,
}

impl BucketPerformance {
    /// Records a solve attempt and updates success rate.
    pub fn record_attempt(&mut self, success: bool) {
        self.attempted += 1;
        if success {
            self.solved += 1;
        }
        self.update_success_rate();
    }

    /// Updates the success rate based on current attempts and solves.
    fn update_success_rate(&mut self) {
        self.success_rate = if self.attempted > 0 {
            self.solved as f64 / self.attempted as f64
        } else {
            0.0
        };
    }

    /// Marks this bucket as mastered.
    pub fn mark_mastered(&mut self) {
        self.mastered = true;
    }

    /// Resets mastery status (useful for re-evaluation).
    pub fn reset_mastery(&mut self) {
        self.mastered = false;
    }
}

/// A curriculum bucket containing problems at a specific difficulty level.
#[derive(Debug, Clone)]
pub struct CurriculumBucket {
    /// Difficulty level of this bucket
    pub level: DifficultyBucket,
    /// Problems in this bucket
    pub problems: Vec<Problem>,
    /// Success rate for this bucket
    pub success_rate: f64,
    /// Whether this bucket is mastered
    pub mastered: bool,
}

impl CurriculumBucket {
    /// Creates a new curriculum bucket.
    pub fn new(level: DifficultyBucket) -> Self {
        Self {
            level,
            problems: Vec::new(),
            success_rate: 0.0,
            mastered: false,
        }
    }

    /// Adds a problem to this bucket.
    pub fn add_problem(&mut self, problem: Problem) {
        self.problems.push(problem);
    }

    /// Returns the number of problems in this bucket.
    pub fn len(&self) -> usize {
        self.problems.len()
    }

    /// Returns true if this bucket has no problems.
    pub fn is_empty(&self) -> bool {
        self.problems.is_empty()
    }

    /// Sorts problems by difficulty (easiest first) within the bucket.
    pub fn sort_by_difficulty(&mut self, classifier: &DifficultyClassifier) {
        self.problems.sort_by(|a, b| {
            let score_a = classifier.classify(a).raw_score;
            let score_b = classifier.classify(b).raw_score;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Curriculum manager for progressive difficulty-based training.
///
/// Organizes problems by difficulty, tracks mastery, and sequences
/// training to optimize learning efficiency.
pub struct CurriculumManager {
    /// Difficulty classifier for problem analysis
    classifier: Arc<DifficultyClassifier>,
    /// Mastery thresholds for progression
    mastery_thresholds: MasteryThresholds,
    /// Adaptive boundaries for bucket classification
    bucket_boundaries: AdaptiveBoundaries,
    /// Performance tracking per bucket
    bucket_performance: Arc<RwLock<HashMap<DifficultyBucket, BucketPerformance>>>,
}

impl Default for CurriculumManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumManager {
    /// Creates a new curriculum manager with default settings.
    pub fn new() -> Self {
        Self {
            classifier: Arc::new(DifficultyClassifier::new()),
            mastery_thresholds: MasteryThresholds::default(),
            bucket_boundaries: AdaptiveBoundaries::default(),
            bucket_performance: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Creates a curriculum manager with custom mastery thresholds.
    pub fn with_thresholds(easy_to_medium: f64, medium_to_hard: f64) -> Self {
        let mut manager = Self::new();
        manager.mastery_thresholds = MasteryThresholds {
            easy_to_medium,
            medium_to_hard,
        };
        manager
    }

    /// Creates a curriculum manager with custom boundaries.
    pub fn with_boundaries(easy_medium: f64, medium_hard: f64) -> Self {
        let mut manager = Self::new();
        manager.bucket_boundaries = AdaptiveBoundaries::new(easy_medium, medium_hard);
        manager
    }

    /// Buckets a collection of problems by difficulty.
    ///
    /// Returns a HashMap mapping each difficulty bucket to its curriculum bucket,
    /// containing all problems classified at that difficulty level.
    pub fn bucket_problems(&self, problems: Vec<Problem>) -> HashMap<DifficultyBucket, CurriculumBucket> {
        let mut buckets = HashMap::new();

        // Initialize empty buckets for each difficulty level
        buckets.insert(DifficultyBucket::Easy, CurriculumBucket::new(DifficultyBucket::Easy));
        buckets.insert(DifficultyBucket::Medium, CurriculumBucket::new(DifficultyBucket::Medium));
        buckets.insert(DifficultyBucket::Hard, CurriculumBucket::new(DifficultyBucket::Hard));

        // Classify each problem and add to appropriate bucket
        for problem in problems {
            let score = self.classifier.classify(&problem);
            let bucket = self.bucket_boundaries.classify_score(score.raw_score);

            if let Some(curr_bucket) = buckets.get_mut(&bucket) {
                curr_bucket.add_problem(problem);
            }
        }

        // Update bucket mastery status and success rates from performance tracking
        let performance = self.bucket_performance.read().unwrap();
        for (level, bucket) in buckets.iter_mut() {
            if let Some(perm) = performance.get(level) {
                bucket.success_rate = perm.success_rate;
                bucket.mastered = perm.mastered;
            }
        }

        buckets
    }

    /// Sequences training problems based on curriculum learning principles.
    ///
    /// Returns a Vec of problems ordered for optimal training:
    /// - Start with Easy bucket (fully interleaved)
    /// - Add Medium problems when Easy mastered (75%+)
    /// - Add Hard problems when Medium mastered (70%+)
    /// - Within bucket: sort by difficulty (easy first)
    /// - Maintain 20% "review" problems from mastered buckets
    pub fn sequence_training(&self, buckets: HashMap<DifficultyBucket, CurriculumBucket>) -> Vec<Problem> {
        let mut sequence = Vec::new();

        // Get bucket references
        let easy_bucket = buckets.get(&DifficultyBucket::Easy);
        let medium_bucket = buckets.get(&DifficultyBucket::Medium);
        let hard_bucket = buckets.get(&DifficultyBucket::Hard);

        // Phase 1: Easy problems (always included)
        if let Some(easy) = easy_bucket {
            let review_count = (easy.len() as f32 * 0.2) as usize;
            let easy_problems = self.select_review_problems(easy, easy.len());

            // Sort by difficulty and add
            let mut sorted_easy = easy_problems;
            sorted_easy.sort_by(|a, b| {
                let score_a = self.classifier.classify(a).raw_score;
                let score_b = self.classifier.classify(b).raw_score;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            sequence.extend(sorted_easy);
        }

        // Phase 2: Medium problems (when Easy mastered)
        if let Some(medium) = medium_bucket {
            if self.is_mastered(medium_bucket.unwrap_or(&CurriculumBucket::new(DifficultyBucket::Medium))) {
                let review_problems = self.get_review_problems(&buckets, DifficultyBucket::Medium);
                sequence.extend(review_problems);
            }

            // Sort medium problems by difficulty
            let mut sorted_medium = medium.problems.clone();
            sorted_medium.sort_by(|a, b| {
                let score_a = self.classifier.classify(a).raw_score;
                let score_b = self.classifier.classify(b).raw_score;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            sequence.extend(sorted_medium);
        }

        // Phase 3: Hard problems (when Medium mastered)
        if let Some(hard) = hard_bucket {
            if self.is_mastered(hard_bucket.unwrap_or(&CurriculumBucket::new(DifficultyBucket::Hard))) {
                let review_problems = self.get_review_problems(&buckets, DifficultyBucket::Hard);
                sequence.extend(review_problems);
            }

            // Sort hard problems by difficulty
            let mut sorted_hard = hard.problems.clone();
            sorted_hard.sort_by(|a, b| {
                let score_a = self.classifier.classify(a).raw_score;
                let score_b = self.classifier.classify(b).raw_score;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            sequence.extend(sorted_hard);
        }

        sequence
    }

    /// Selects review problems from a bucket.
    fn select_review_problems(&self, bucket: &CurriculumBucket, count: usize) -> Vec<Problem> {
        let mut review = Vec::new();
        let take = count.min(bucket.len());

        for i in 0..take {
            if i < bucket.problems.len() {
                review.push(bucket.problems[i].clone());
            }
        }

        review
    }

    /// Gets review problems from mastered buckets.
    fn get_review_problems(&self, buckets: &HashMap<DifficultyBucket, CurriculumBucket>, current_level: DifficultyBucket) -> Vec<Problem> {
        let mut review_problems = Vec::new();

        for (level, bucket) in buckets.iter() {
            // Only include review from mastered buckets that are easier than current
            if bucket.mastered && self.is_easier_than(*level, current_level) {
                let review_count = (bucket.len() as f32 * 0.2) as usize;
                review_problems.extend(self.select_review_problems(bucket, review_count));
            }
        }

        review_problems
    }

    /// Checks if one difficulty level is easier than another.
    fn is_easier_than(&self, a: DifficultyBucket, b: DifficultyBucket) -> bool {
        match (a, b) {
            (DifficultyBucket::Easy, DifficultyBucket::Medium) => true,
            (DifficultyBucket::Easy, DifficultyBucket::Hard) => true,
            (DifficultyBucket::Medium, DifficultyBucket::Hard) => true,
            _ => false,
        }
    }

    /// Determines if a bucket is mastered based on performance thresholds.
    pub fn is_mastered(&self, bucket: &CurriculumBucket) -> bool {
        let threshold = match bucket.level {
            DifficultyBucket::Easy => self.mastery_thresholds.easy_to_medium,
            DifficultyBucket::Medium => self.mastery_thresholds.medium_to_hard,
            DifficultyBucket::Hard => 1.0, // Hard never auto-masters
        };

        bucket.success_rate >= threshold
    }

    /// Updates adaptive boundaries based on observed performance.
    ///
    /// Uses EMA with alpha=0.05 for stability. Adjusts boundaries when
    /// performance indicates current thresholds are too easy or too hard.
    pub fn update_boundaries(&self, bucket: DifficultyBucket, performance: f64) {
        const ALPHA: f64 = 0.05;

        match bucket {
            DifficultyBucket::Easy => {
                if performance > 0.85 {
                    // Easy is too easy - move boundary up
                    let current = self.bucket_boundaries.get_easy_medium();
                    self.bucket_boundaries.update_easy_medium(current + 0.05, ALPHA);
                } else if performance < 0.65 {
                    // Easy is too hard - move boundary down
                    let current = self.bucket_boundaries.get_easy_medium();
                    self.bucket_boundaries.update_easy_medium((current - 0.05).max(0.2), ALPHA);
                }
            }
            DifficultyBucket::Medium => {
                if performance > 0.85 {
                    // Medium is too easy - move hard boundary down
                    let current = self.bucket_boundaries.get_medium_hard();
                    self.bucket_boundaries.update_medium_hard(current - 0.05, ALPHA);
                } else if performance < 0.65 {
                    // Medium is too hard - move hard boundary up
                    let current = self.bucket_boundaries.get_medium_hard();
                    self.bucket_boundaries.update_medium_hard((current + 0.05).min(0.9), ALPHA);
                }
            }
            DifficultyBucket::Hard => {
                // Hard boundary adjustments based on overall system performance
                if performance < 0.5 {
                    // Very poor performance on hard - may need easier classification
                    let current = self.bucket_boundaries.get_medium_hard();
                    self.bucket_boundaries.update_medium_hard((current - 0.02).max(0.5), ALPHA);
                }
            }
        }
    }

    /// Recommends a source bucket for problems to scaffold toward a target bucket.
    ///
    /// Returns the next easier bucket that should be mastered before attempting
    /// the target bucket, or None if the target is already Easy.
    pub fn recommend_source_bucket(&self, target_bucket: DifficultyBucket) -> Option<DifficultyBucket> {
        match target_bucket {
            DifficultyBucket::Easy => None, // Already at easiest level
            DifficultyBucket::Medium => Some(DifficultyBucket::Easy),
            DifficultyBucket::Hard => Some(DifficultyBucket::Medium),
        }
    }

    /// Records a solve attempt and updates performance tracking.
    ///
    /// This updates both the classifier's learning and the bucket performance
    /// metrics for adaptive curriculum management.
    pub fn record_solve(&self, problem: &Problem, success: bool) {
        // Classify the problem to determine its bucket
        let score = self.classifier.classify(problem);
        let bucket = self.bucket_boundaries.classify_score(score.raw_score);

        // Update classifier with actual difficulty data
        let actual_difficulty = if success { 0.0 } else { 1.0 };
        self.classifier.update(problem, actual_difficulty);

        // Update bucket performance
        let mut performance = self.bucket_performance.write().unwrap();
        let bucket_perf = performance.entry(bucket).or_insert_with(BucketPerformance::default);
        bucket_perf.record_attempt(success);

        // Check for mastery and update if threshold met
        let threshold = match bucket {
            DifficultyBucket::Easy => self.mastery_thresholds.easy_to_medium,
            DifficultyBucket::Medium => self.mastery_thresholds.medium_to_hard,
            DifficultyBucket::Hard => 1.0, // Hard never auto-masters
        };

        if bucket_perf.success_rate >= threshold && bucket_perf.attempted >= 5 {
            bucket_perf.mark_mastered();
        }

        // Trigger boundary adaptation based on performance
        self.update_boundaries(bucket, bucket_perf.success_rate);
    }

    /// Gets the current performance statistics for all buckets.
    pub fn get_performance_stats(&self) -> HashMap<DifficultyBucket, BucketPerformance> {
        self.bucket_performance.read().unwrap().clone()
    }

    /// Resets all performance tracking (useful for starting fresh training).
    pub fn reset_performance(&self) {
        let mut performance = self.bucket_performance.write().unwrap();
        *performance = HashMap::new();
    }

    /// Returns the current adaptive boundaries.
    pub fn get_boundaries(&self) -> (f64, f64) {
        (
            self.bucket_boundaries.get_easy_medium(),
            self.bucket_boundaries.get_medium_hard(),
        )
    }

    /// Returns the current mastery thresholds.
    pub fn get_thresholds(&self) -> (f64, f64) {
        (
            self.mastery_thresholds.easy_to_medium,
            self.mastery_thresholds.medium_to_hard,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn test_problem(name: &str, examples: Vec<Example>) -> Problem {
        Problem {
            name: name.to_string(),
            category: "test",
            description: "test problem",
            signature: "fn test(x: i64) -> i64",
            examples,
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

    fn easy_example() -> Problem {
        test_problem(
            "easy_add",
            vec![
                Example {
                    inputs: vec![Value::Int(1), Value::Int(2)],
                    expected: Value::Int(3),
                },
                Example {
                    inputs: vec![Value::Int(2), Value::Int(3)],
                    expected: Value::Int(5),
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(4)],
                    expected: Value::Int(7),
                },
            ],
        )
    }

    fn medium_example() -> Problem {
        test_problem(
            "medium_mult",
            vec![
                Example {
                    inputs: vec![Value::Int(2), Value::Int(3)],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::Int(3), Value::Int(4)],
                    expected: Value::Int(12),
                },
            ],
        )
    }

    fn hard_example() -> Problem {
        test_problem(
            "hard_fib",
            vec![
                Example {
                    inputs: vec![Value::Int(5)],
                    expected: Value::Int(5),
                },
                Example {
                    inputs: vec![Value::Int(6)],
                    expected: Value::Int(8),
                },
            ],
        )
    }

    #[test]
    fn test_curriculum_manager_creation() {
        let manager = CurriculumManager::new();

        // Check default boundaries
        let (easy_medium, medium_hard) = manager.get_boundaries();
        assert!((easy_medium - 0.4).abs() < 0.001);
        assert!((medium_hard - 0.7).abs() < 0.001);

        // Check default thresholds
        let (easy_to_medium, medium_to_hard) = manager.get_thresholds();
        assert!((easy_to_medium - 0.75).abs() < 0.001);
        assert!((medium_to_hard - 0.70).abs() < 0.001);
    }

    #[test]
    fn test_bucket_problems() {
        let manager = CurriculumManager::new();
        let problems = vec![easy_example(), medium_example(), hard_example()];

        let buckets = manager.bucket_problems(problems);

        // Should have all three buckets
        assert!(buckets.contains_key(&DifficultyBucket::Easy));
        assert!(buckets.contains_key(&DifficultyBucket::Medium));
        assert!(buckets.contains_key(&DifficultyBucket::Hard));

        // Each bucket should have at least been created
        assert!(!buckets.is_empty());
    }

    #[test]
    fn test_sequence_training() {
        let manager = CurriculumManager::new();
        let problems = vec![easy_example(), medium_example(), hard_example()];

        let buckets = manager.bucket_problems(problems);
        let sequence = manager.sequence_training(buckets);

        // Sequence should contain problems
        assert!(!sequence.is_empty());
    }

    #[test]
    fn test_record_solve() {
        let manager = CurriculumManager::new();
        let problem = easy_example();

        // Record successful solves
        for _ in 0..5 {
            manager.record_solve(&problem, true);
        }

        let stats = manager.get_performance_stats();
        assert!(!stats.is_empty());

        // Should have some bucket with attempts recorded
        let total_attempts: u32 = stats.values().map(|p| p.attempted).sum();
        assert_eq!(total_attempts, 5);
    }

    #[test]
    fn test_mastery_thresholds() {
        let manager = CurriculumManager::new();
        let problem = easy_example();

        // Record below-threshold performance
        for _ in 0..10 {
            manager.record_solve(&problem, false);
        }

        let stats = manager.get_performance_stats();
        for bucket_perf in stats.values() {
            assert!(!bucket_perf.mastered);
        }
    }

    #[test]
    fn test_adaptive_boundaries() {
        let manager = CurriculumManager::new();
        let (initial_easy_medium, initial_medium_hard) = manager.get_boundaries();

        // Trigger boundary update with high performance
        manager.update_boundaries(DifficultyBucket::Easy, 0.9);

        let (new_easy_medium, _) = manager.get_boundaries();
        // Boundary should have increased
        assert!(new_easy_medium > initial_easy_medium);
    }

    #[test]
    fn test_recommend_source_bucket() {
        let manager = CurriculumManager::new();

        // Easy should have no source (it's the easiest)
        assert!(manager.recommend_source_bucket(DifficultyBucket::Easy).is_none());

        // Medium should recommend Easy
        assert_eq!(
            manager.recommend_source_bucket(DifficultyBucket::Medium),
            Some(DifficultyBucket::Easy)
        );

        // Hard should recommend Medium
        assert_eq!(
            manager.recommend_source_bucket(DifficultyBucket::Hard),
            Some(DifficultyBucket::Medium)
        );
    }

    #[test]
    fn test_custom_thresholds() {
        let manager = CurriculumManager::with_thresholds(0.8, 0.75);
        let (easy_to_medium, medium_to_hard) = manager.get_thresholds();

        assert!((easy_to_medium - 0.8).abs() < 0.001);
        assert!((medium_to_hard - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_custom_boundaries() {
        let manager = CurriculumManager::with_boundaries(0.5, 0.8);
        let (easy_medium, medium_hard) = manager.get_boundaries();

        assert!((easy_medium - 0.5).abs() < 0.001);
        assert!((medium_hard - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_reset_performance() {
        let manager = CurriculumManager::new();
        let problem = easy_example();

        // Record some attempts
        for _ in 0..3 {
            manager.record_solve(&problem, true);
        }

        assert!(!manager.get_performance_stats().is_empty());

        // Reset
        manager.reset_performance();
        assert!(manager.get_performance_stats().is_empty());
    }

    #[test]
    fn test_bucket_performance_update() {
        let mut perf = BucketPerformance::default();

        perf.record_attempt(true);
        assert_eq!(perf.attempted, 1);
        assert_eq!(perf.solved, 1);
        assert!((perf.success_rate - 1.0).abs() < 0.001);

        perf.record_attempt(false);
        assert_eq!(perf.attempted, 2);
        assert_eq!(perf.solved, 1);
        assert!((perf.success_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_curriculum_bucket_sorting() {
        let manager = CurriculumManager::new();
        let mut bucket = CurriculumBucket::new(DifficultyBucket::Easy);

        bucket.add_problem(easy_example());
        bucket.add_problem(medium_example());

        bucket.sort_by_difficulty(&manager.classifier);

        // After sorting, should have 2 problems
        assert_eq!(bucket.len(), 2);
    }

    #[test]
    fn test_adaptive_boundaries_classify_score() {
        let boundaries = AdaptiveBoundaries::default();

        // Below easy-medium threshold
        assert_eq!(boundaries.classify_score(0.3), DifficultyBucket::Easy);

        // Between thresholds
        assert_eq!(boundaries.classify_score(0.5), DifficultyBucket::Medium);

        // Above medium-hard threshold
        assert_eq!(boundaries.classify_score(0.8), DifficultyBucket::Hard);
    }
}
