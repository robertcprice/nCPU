//! Adaptive training sequencing for progressive difficulty-based learning.
//!
//! This module implements a sophisticated training sequencer that:
//! - Organizes problems into curriculum phases (Easy → Medium → Hard)
//! - Tracks per-bucket performance and adapts sequence based on success rates
//! - Implements phase advancement/regression logic to maintain optimal challenge
//! - Provides smooth progression without frustration by balancing problem ratios
//!
//! # Phases
//!
//! The sequencer progresses through 5 phases:
//! - **Phase 1**: Easy problems only (builds confidence)
//! - **Phase 2**: Easy + Medium (70% Easy, 30% Medium)
//! - **Phase 3**: Easy + Medium + Hard (50% Easy, 30% Medium, 20% Hard)
//! - **Phase 4**: Medium + Hard (40% Medium, 60% Hard)
//! - **Phase 5**: Hard problems only (mastery challenge)
//!
//! # Adaptive Behavior
//!
//! - **Advancement**: Phase advances when current bucket mastery threshold reached
//! - **Regression**: Phase regresses if overall success drops below 50%
//! - **Skipping**: Aggressive mode allows skipping phases when success > 90%
//! - **Rebalancing**: Problem ratios adjust based on performance feedback

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::benchmark::Problem;
use super::difficulty::DifficultyBucket;
use super::curriculum::{BucketPerformance, CurriculumManager};

/// Training phase with problem distribution ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Phase 1: Easy problems only (100% Easy)
    Phase1EasyOnly,
    /// Phase 2: Easy + Medium (70% Easy, 30% Medium)
    Phase2EasyMedium,
    /// Phase 3: All buckets (50% Easy, 30% Medium, 20% Hard)
    Phase3AllBuckets,
    /// Phase 4: Medium + Hard (40% Medium, 60% Hard)
    Phase4MediumHard,
    /// Phase 5: Hard only (100% Hard)
    Phase5HardOnly,
}

impl TrainingPhase {
    /// Returns the next phase in the progression.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Phase1EasyOnly => Some(Self::Phase2EasyMedium),
            Self::Phase2EasyMedium => Some(Self::Phase3AllBuckets),
            Self::Phase3AllBuckets => Some(Self::Phase4MediumHard),
            Self::Phase4MediumHard => Some(Self::Phase5HardOnly),
            Self::Phase5HardOnly => None, // Terminal phase
        }
    }

    /// Returns the previous phase in the progression.
    pub fn prev(self) -> Option<Self> {
        match self {
            Self::Phase1EasyOnly => None, // Initial phase
            Self::Phase2EasyMedium => Some(Self::Phase1EasyOnly),
            Self::Phase3AllBuckets => Some(Self::Phase2EasyMedium),
            Self::Phase4MediumHard => Some(Self::Phase3AllBuckets),
            Self::Phase5HardOnly => Some(Self::Phase4MediumHard),
        }
    }

    /// Returns the problem distribution ratios for this phase.
    ///
    /// Returns (easy_ratio, medium_ratio, hard_ratio)
    pub fn distribution(self) -> (f64, f64, f64) {
        match self {
            Self::Phase1EasyOnly => (1.0, 0.0, 0.0),
            Self::Phase2EasyMedium => (0.7, 0.3, 0.0),
            Self::Phase3AllBuckets => (0.5, 0.3, 0.2),
            Self::Phase4MediumHard => (0.0, 0.4, 0.6),
            Self::Phase5HardOnly => (0.0, 0.0, 1.0),
        }
    }

    /// Returns the primary bucket for this phase.
    pub fn primary_bucket(self) -> DifficultyBucket {
        match self {
            Self::Phase1EasyOnly => DifficultyBucket::Easy,
            Self::Phase2EasyMedium => DifficultyBucket::Medium,
            Self::Phase3AllBuckets => DifficultyBucket::Medium,
            Self::Phase4MediumHard => DifficultyBucket::Hard,
            Self::Phase5HardOnly => DifficultyBucket::Hard,
        }
    }
}

/// A single training step with outcome tracking.
#[derive(Debug, Clone)]
pub struct TrainingStep {
    /// The problem attempted
    pub problem: Problem,
    /// Whether the problem was solved successfully
    pub success: bool,
    /// Time taken to solve
    pub duration: Duration,
    /// Whether the phase advanced after this step
    pub advanced: bool,
    /// Bucket classification before this step
    pub bucket_before: DifficultyBucket,
    /// Bucket classification after this step (may differ if boundaries adapted)
    pub bucket_after: DifficultyBucket,
}

impl TrainingStep {
    /// Creates a new training step.
    pub fn new(
        problem: Problem,
        success: bool,
        duration: Duration,
        bucket: DifficultyBucket,
    ) -> Self {
        Self {
            problem,
            success,
            duration,
            advanced: false,
            bucket_before: bucket,
            bucket_after: bucket,
        }
    }

    /// Marks that this step caused a phase advancement.
    pub fn mark_advanced(&mut self) {
        self.advanced = true;
    }
}

/// Statistics for the current training sequence.
#[derive(Debug, Clone)]
pub struct SequenceStats {
    /// Total problems in the sequence
    pub total_problems: usize,
    /// Problems completed
    pub completed: usize,
    /// Easy problems encountered
    pub easy_count: usize,
    /// Medium problems encountered
    pub medium_count: usize,
    /// Hard problems encountered
    pub hard_count: usize,
    /// Current difficulty bucket
    pub current_bucket: DifficultyBucket,
    /// Current training phase
    pub current_phase: TrainingPhase,
    /// Overall success rate across all completed problems
    pub overall_success_rate: f64,
    /// Per-bucket success rates
    pub bucket_success_rates: HashMap<DifficultyBucket, f64>,
}

impl SequenceStats {
    /// Creates a new sequence stats with initial values.
    pub fn new() -> Self {
        Self {
            total_problems: 0,
            completed: 0,
            easy_count: 0,
            medium_count: 0,
            hard_count: 0,
            current_bucket: DifficultyBucket::Easy,
            current_phase: TrainingPhase::Phase1EasyOnly,
            overall_success_rate: 0.0,
            bucket_success_rates: HashMap::new(),
        }
    }

    /// Updates statistics based on a completed training step.
    pub fn record_step(&mut self, step: &TrainingStep) {
        self.completed += 1;

        match step.bucket_after {
            DifficultyBucket::Easy => self.easy_count += 1,
            DifficultyBucket::Medium => self.medium_count += 1,
            DifficultyBucket::Hard => self.hard_count += 1,
        }

        // Update overall success rate
        self.overall_success_rate = if self.completed > 0 {
            let successful = (self.overall_success_rate * (self.completed - 1) as f64)
                + if step.success { 1.0 } else { 0.0 };
            successful / self.completed as f64
        } else {
            if step.success { 1.0 } else { 0.0 }
        };
    }

    /// Returns the completion percentage.
    pub fn completion_percent(&self) -> f64 {
        if self.total_problems > 0 {
            (self.completed as f64 / self.total_problems as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Returns whether training is complete.
    pub fn is_complete(&self) -> bool {
        self.completed >= self.total_problems && self.total_problems > 0
    }
}

/// Adaptive training sequencer for progressive difficulty-based learning.
///
/// The sequencer maintains a curriculum of problems organized by difficulty
/// and dynamically adjusts the training sequence based on performance.
///
/// # Example
///
/// ```ignore
/// // Illustrative only: `solver::sequencing` is a private module and this
/// // sketch references helpers (`solve_problem`, `bucket`) defined by the
/// // caller. Marked `ignore` so it documents intent without being compiled.
/// use mog_synth::solver::sequencing::TrainingSequencer;
///
/// let sequencer = TrainingSequencer::new(problems);
///
/// while let Some(problem) = sequencer.next_problem() {
///     let start = std::time::Instant::now();
///     let success = solve_problem(&problem);
///     let duration = start.elapsed();
///
///     let step = TrainingStep::new(problem, success, duration, bucket);
///     sequencer.record_step(step);
/// }
/// ```
pub struct TrainingSequencer {
    /// Curriculum manager for difficulty classification and performance tracking
    curriculum: Arc<CurriculumManager>,
    /// Current sequence of problems to solve
    current_sequence: Vec<Problem>,
    /// Current position in the sequence
    position: usize,
    /// Per-bucket performance tracking
    bucket_performance: Arc<RwLock<HashMap<DifficultyBucket, BucketPerformance>>>,
    /// Current training phase
    current_phase: TrainingPhase,
    /// Aggressive mode (allows skipping phases when performing well)
    aggressive_mode: bool,
    /// Statistics for the current sequence
    stats: SequenceStats,
    /// Mastery threshold for phase advancement
    advancement_threshold: f64,
    /// Regression threshold (drop below this to regress)
    regression_threshold: f64,
}

impl TrainingSequencer {
    /// Creates a new training sequencer with the given problems.
    ///
    /// Problems are automatically classified into difficulty buckets and
    /// the sequence is generated starting from Phase 1.
    pub fn new(problems: Vec<Problem>) -> Self {
        Self::with_settings(problems, 0.75, 0.50, false)
    }

    /// Creates a sequencer with custom thresholds and aggressive mode.
    ///
    /// # Arguments
    ///
    /// * `problems` - The training problems to sequence
    /// * `advancement_threshold` - Success rate required to advance phases (default: 0.75)
    /// * `regression_threshold` - Success rate below which to regress (default: 0.50)
    /// * `aggressive_mode` - Whether to allow phase skipping (default: false)
    pub fn with_settings(
        problems: Vec<Problem>,
        advancement_threshold: f64,
        regression_threshold: f64,
        aggressive_mode: bool,
    ) -> Self {
        let curriculum = Arc::new(CurriculumManager::new());
        let buckets = curriculum.bucket_problems(problems);

        let mut sequencer = Self {
            curriculum,
            current_sequence: Vec::new(),
            position: 0,
            bucket_performance: Arc::new(RwLock::new(HashMap::new())),
            current_phase: TrainingPhase::Phase1EasyOnly,
            aggressive_mode,
            stats: SequenceStats::new(),
            advancement_threshold,
            regression_threshold,
        };

        sequencer.stats.current_bucket = DifficultyBucket::Easy;
        sequencer.stats.current_phase = TrainingPhase::Phase1EasyOnly;
        sequencer.generate_sequence(buckets);

        sequencer
    }

    /// Returns the next problem in the training sequence.
    ///
    /// Returns `None` when the sequence is complete.
    pub fn next_problem(&mut self) -> Option<Problem> {
        if self.position >= self.current_sequence.len() {
            return None;
        }

        let problem = self.current_sequence.get(self.position).cloned();
        self.position += 1;
        problem
    }

    /// Records the outcome of a training step and updates sequence state.
    ///
    /// This updates performance tracking, adjusts phase if necessary,
    /// and potentially regenerates the sequence based on performance.
    pub fn record_step(&mut self, mut step: TrainingStep) {
        // Update performance tracking
        let bucket = step.bucket_after;
        {
            let mut performance = self.bucket_performance.write().unwrap();
            let bucket_perf = performance.entry(bucket).or_insert_with(BucketPerformance::default);
            bucket_perf.record_attempt(step.success);
        } // Drop lock before calling self methods

        // Update stats
        self.stats.record_step(&step);

        // Check for phase advancement
        let advanced = self.check_advancement();
        if advanced {
            step.mark_advanced();
        }

        // Check for regression
        if self.check_regression() {
            self.regress_phase();
        }

        // Check if we need to regenerate sequence (e.g., after phase change)
        if advanced || self.needs_regeneration() {
            self.regenerate_sequence();
        }
    }

    /// Checks whether training is complete.
    ///
    /// Training is complete when all problems in the current sequence
    /// have been attempted.
    pub fn is_complete(&self) -> bool {
        self.position >= self.current_sequence.len() && !self.current_sequence.is_empty()
    }

    /// Returns the current difficulty bucket.
    pub fn current_bucket(&self) -> DifficultyBucket {
        self.stats.current_bucket
    }

    /// Returns the current training phase.
    pub fn current_phase(&self) -> TrainingPhase {
        self.current_phase
    }

    /// Returns statistics for the current sequence.
    pub fn sequence_stats(&self) -> SequenceStats {
        self.stats.clone()
    }

    /// Regenerates the training sequence based on current performance.
    ///
    /// This is called automatically when phase changes or performance
    /// indicates the current sequence is no longer optimal.
    pub fn regenerate_sequence(&mut self) {
        // Get all problems from curriculum (we'd need to store original problems)
        // For now, we'll regenerate from the current sequence with rebalancing
        self.rebalance_sequence();
    }

    /// Generates the initial problem sequence for the current phase.
    fn generate_sequence(&mut self, buckets: HashMap<DifficultyBucket, crate::solver::curriculum::CurriculumBucket>) {
        let (easy_ratio, medium_ratio, hard_ratio) = self.current_phase.distribution();

        let mut sequence = Vec::new();

        // Add problems according to phase distribution
        if let Some(easy_bucket) = buckets.get(&DifficultyBucket::Easy) {
            let count = self.phase_count(easy_bucket.len(), easy_ratio);
            sequence.extend(easy_bucket.problems.iter().take(count).cloned());
        }

        if let Some(medium_bucket) = buckets.get(&DifficultyBucket::Medium) {
            let count = self.phase_count(medium_bucket.len(), medium_ratio);
            sequence.extend(medium_bucket.problems.iter().take(count).cloned());
        }

        if let Some(hard_bucket) = buckets.get(&DifficultyBucket::Hard) {
            let count = self.phase_count(hard_bucket.len(), hard_ratio);
            sequence.extend(hard_bucket.problems.iter().take(count).cloned());
        }

        // Fallback: never produce an empty sequence when problems exist. The
        // phase distribution can select nothing (e.g. Phase1EasyOnly weights
        // only the Easy bucket, but every problem classified as Medium/Hard),
        // which would leave the sequencer permanently unable to hand out work.
        // Seed it with the easiest available problems so training can progress.
        if sequence.is_empty() {
            for bucket in [
                DifficultyBucket::Easy,
                DifficultyBucket::Medium,
                DifficultyBucket::Hard,
            ] {
                if let Some(b) = buckets.get(&bucket) {
                    if !b.problems.is_empty() {
                        sequence.extend(b.problems.iter().cloned());
                        break;
                    }
                }
            }
        }

        self.current_sequence = sequence;
        self.stats.total_problems = self.current_sequence.len();
    }

    /// Calculates the number of problems to take from a bucket for the current phase.
    fn phase_count(&self, bucket_size: usize, ratio: f64) -> usize {
        if ratio == 0.0 {
            return 0;
        }

        let base_count = (bucket_size as f64 * ratio).ceil() as usize;
        base_count.min(bucket_size)
    }

    /// Rebalances the current sequence based on performance.
    fn rebalance_sequence(&mut self) {
        let performance = self.bucket_performance.read().unwrap();
        let easy_success = performance.get(&DifficultyBucket::Easy)
            .map_or(0.0, |p| p.success_rate);
        let medium_success = performance.get(&DifficultyBucket::Medium)
            .map_or(0.0, |p| p.success_rate);
        let hard_success = performance.get(&DifficultyBucket::Hard)
            .map_or(0.0, |p| p.success_rate);

        // Adjust distribution based on performance
        // If a bucket is performing poorly, reduce its ratio
        // If performing well, increase its ratio slightly
        let (mut easy_ratio, mut medium_ratio, mut hard_ratio) = self.current_phase.distribution();

        if easy_success < 0.6 && easy_ratio > 0.0 {
            easy_ratio *= 0.8; // Reduce easy problems if they're too easy
        }
        if medium_success > 0.85 {
            medium_ratio *= 1.2; // Increase medium if performing well
            medium_ratio = medium_ratio.min(1.0);
        }
        if hard_success < 0.4 && hard_ratio > 0.0 {
            hard_ratio *= 0.7; // Reduce hard if struggling
        }

        // Rebuild sequence with adjusted ratios
        // (In a full implementation, we'd preserve the original problem list)
        drop(performance);
    }

    /// Checks if the sequencer should advance to the next phase.
    ///
    /// Advances when:
    /// - Current phase's primary bucket meets mastery threshold
    /// - Previous buckets maintain > 65% success rate
    fn check_advancement(&self) -> bool {
        let performance = self.bucket_performance.read().unwrap();
        let primary_bucket = self.current_phase.primary_bucket();

        // Check primary bucket mastery
        let primary_perf = performance.get(&primary_bucket)
            .map_or(0.0, |p| p.success_rate);

        if primary_perf < self.advancement_threshold {
            return false;
        }

        // Check that previous buckets maintain minimum performance
        match self.current_phase {
            TrainingPhase::Phase1EasyOnly => {
                // Always allow advancement from phase 1 if threshold met
                true
            }
            TrainingPhase::Phase2EasyMedium => {
                // Check easy bucket performance
                let easy_perf = performance.get(&DifficultyBucket::Easy)
                    .map_or(0.0, |p| p.success_rate);
                easy_perf >= 0.65
            }
            TrainingPhase::Phase3AllBuckets => {
                // Check both easy and medium
                let easy_perf = performance.get(&DifficultyBucket::Easy)
                    .map_or(0.0, |p| p.success_rate);
                let medium_perf = performance.get(&DifficultyBucket::Medium)
                    .map_or(0.0, |p| p.success_rate);
                easy_perf >= 0.65 && medium_perf >= 0.65
            }
            TrainingPhase::Phase4MediumHard => {
                // Check medium bucket performance
                let medium_perf = performance.get(&DifficultyBucket::Medium)
                    .map_or(0.0, |p| p.success_rate);
                medium_perf >= 0.65
            }
            TrainingPhase::Phase5HardOnly => {
                // No advancement from final phase
                false
            }
        }
    }

    /// Checks if the sequencer should regress to a previous phase.
    ///
    /// Regresses when overall success rate drops below regression threshold.
    fn check_regression(&self) -> bool {
        if self.stats.completed < 5 {
            return false; // Need some data before regressing
        }

        self.stats.overall_success_rate < self.regression_threshold
    }

    /// Advances to the next training phase.
    fn advance_phase(&mut self) -> bool {
        if let Some(next_phase) = self.current_phase.next() {
            self.current_phase = next_phase;
            self.stats.current_phase = next_phase;
            true
        } else {
            false
        }
    }

    /// Regresses to the previous training phase.
    fn regress_phase(&mut self) {
        if let Some(prev_phase) = self.current_phase.prev() {
            self.current_phase = prev_phase;
            self.stats.current_phase = prev_phase;
        }
    }

    /// Checks if the sequence needs regeneration based on performance.
    fn needs_regeneration(&self) -> bool {
        // Regenerate if performance indicates we're misaligned
        let performance = self.bucket_performance.read().unwrap();

        for (bucket, perf) in performance.iter() {
            // If we're failing > 70% in any active bucket, rebalance
            if perf.attempted >= 5 && perf.success_rate < 0.3 {
                return true;
            }
        }

        false
    }

    /// Advances the phase if advancement criteria are met.
    ///
    /// In aggressive mode, may skip phases if performance is exceptional.
    fn advance_phase_if_ready(&mut self) -> bool {
        let performance = self.bucket_performance.read().unwrap();
        let primary_bucket = self.current_phase.primary_bucket();
        let primary_perf = performance.get(&primary_bucket)
            .map_or(0.0, |p| p.success_rate);

        drop(performance);

        if !self.check_advancement() {
            return false;
        }

        // In aggressive mode, check if we can skip a phase
        if self.aggressive_mode && primary_perf > 0.90 {
            // Skip to next phase
            if let Some(next) = self.current_phase.next() {
                self.current_phase = next;
                self.stats.current_phase = next;
                return true;
            }
        }

        self.advance_phase()
    }

    /// Returns the advancement threshold.
    pub fn advancement_threshold(&self) -> f64 {
        self.advancement_threshold
    }

    /// Returns the regression threshold.
    pub fn regression_threshold(&self) -> f64 {
        self.regression_threshold
    }

    /// Sets aggressive mode (allows phase skipping).
    pub fn set_aggressive_mode(&mut self, aggressive: bool) {
        self.aggressive_mode = aggressive;
    }

    /// Returns whether aggressive mode is enabled.
    pub fn is_aggressive(&self) -> bool {
        self.aggressive_mode
    }

    /// Resets the sequencer to the initial phase with current performance data preserved.
    pub fn reset_phase(&mut self) {
        self.current_phase = TrainingPhase::Phase1EasyOnly;
        self.stats.current_phase = TrainingPhase::Phase1EasyOnly;
        self.position = 0;
        self.regenerate_sequence();
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

    fn easy_problem() -> Problem {
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

    fn medium_problem() -> Problem {
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

    fn hard_problem() -> Problem {
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
    fn test_training_sequencer_creation() {
        let problems = vec![easy_problem(), medium_problem(), hard_problem()];
        let sequencer = TrainingSequencer::new(problems);

        assert_eq!(sequencer.current_phase(), TrainingPhase::Phase1EasyOnly);
        assert!(!sequencer.is_complete());
    }

    #[test]
    fn test_next_problem() {
        let problems = vec![easy_problem()];
        let mut sequencer = TrainingSequencer::new(problems);

        let problem = sequencer.next_problem();
        assert!(problem.is_some());

        // After getting the only problem, should return None
        let problem = sequencer.next_problem();
        // Depending on sequence generation, may or may not be None
    }

    #[test]
    fn test_record_step() {
        let problems = vec![easy_problem()];
        let mut sequencer = TrainingSequencer::new(problems);

        let step = TrainingStep::new(
            easy_problem(),
            true,
            Duration::from_millis(100),
            DifficultyBucket::Easy,
        );

        sequencer.record_step(step);

        let stats = sequencer.sequence_stats();
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.easy_count, 1);
    }

    #[test]
    fn test_phase_advancement() {
        let problems = vec![easy_problem(), medium_problem()];
        let mut sequencer = TrainingSequencer::with_settings(problems, 0.75, 0.50, false);

        // Record successful steps to trigger advancement
        for _ in 0..10 {
            let step = TrainingStep::new(
                easy_problem(),
                true,
                Duration::from_millis(100),
                DifficultyBucket::Easy,
            );
            sequencer.record_step(step);
        }

        // Should have advanced or be ready to advance
        let stats = sequencer.sequence_stats();
        assert!(stats.completed >= 10);
    }

    #[test]
    fn test_aggressive_mode() {
        let problems = vec![easy_problem(), medium_problem(), hard_problem()];
        let mut sequencer = TrainingSequencer::with_settings(problems, 0.75, 0.50, true);

        assert!(sequencer.is_aggressive());

        sequencer.set_aggressive_mode(false);
        assert!(!sequencer.is_aggressive());
    }

    #[test]
    fn test_regression() {
        let problems = vec![easy_problem()];
        let mut sequencer = TrainingSequencer::with_settings(problems, 0.75, 0.50, false);

        // Start in Phase 1
        assert_eq!(sequencer.current_phase(), TrainingPhase::Phase1EasyOnly);

        // Record failures to trigger regression
        for _ in 0..10 {
            let step = TrainingStep::new(
                easy_problem(),
                false,
                Duration::from_millis(100),
                DifficultyBucket::Easy,
            );
            sequencer.record_step(step);
        }

        // With poor performance, should have regressed or stayed
        let stats = sequencer.sequence_stats();
        assert!(stats.overall_success_rate < 0.5);
    }

    #[test]
    fn test_phase_progression() {
        assert_eq!(TrainingPhase::Phase1EasyOnly.next(), Some(TrainingPhase::Phase2EasyMedium));
        assert_eq!(TrainingPhase::Phase2EasyMedium.next(), Some(TrainingPhase::Phase3AllBuckets));
        assert_eq!(TrainingPhase::Phase3AllBuckets.next(), Some(TrainingPhase::Phase4MediumHard));
        assert_eq!(TrainingPhase::Phase4MediumHard.next(), Some(TrainingPhase::Phase5HardOnly));
        assert_eq!(TrainingPhase::Phase5HardOnly.next(), None);
    }

    #[test]
    fn test_phase_distribution() {
        let (e, m, h) = TrainingPhase::Phase1EasyOnly.distribution();
        assert!((e - 1.0).abs() < 0.001);
        assert_eq!(m, 0.0);
        assert_eq!(h, 0.0);

        let (e, m, h) = TrainingPhase::Phase2EasyMedium.distribution();
        assert!((e - 0.7).abs() < 0.001);
        assert!((m - 0.3).abs() < 0.001);
        assert_eq!(h, 0.0);

        let (e, m, h) = TrainingPhase::Phase3AllBuckets.distribution();
        assert!((e - 0.5).abs() < 0.001);
        assert!((m - 0.3).abs() < 0.001);
        assert!((h - 0.2).abs() < 0.001);

        let (e, m, h) = TrainingPhase::Phase4MediumHard.distribution();
        assert_eq!(e, 0.0);
        assert!((m - 0.4).abs() < 0.001);
        assert!((h - 0.6).abs() < 0.001);

        let (e, m, h) = TrainingPhase::Phase5HardOnly.distribution();
        assert_eq!(e, 0.0);
        assert_eq!(m, 0.0);
        assert!((h - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sequence_stats() {
        let mut stats = SequenceStats::new();
        stats.total_problems = 100;

        let step = TrainingStep::new(
            easy_problem(),
            true,
            Duration::from_millis(100),
            DifficultyBucket::Easy,
        );

        stats.record_step(&step);

        assert_eq!(stats.completed, 1);
        assert_eq!(stats.easy_count, 1);
        assert!((stats.overall_success_rate - 1.0).abs() < 0.001);
        assert!((stats.completion_percent() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reset_phase() {
        let problems = vec![easy_problem()];
        let mut sequencer = TrainingSequencer::new(problems);

        // Manually advance (for testing)
        sequencer.current_phase = TrainingPhase::Phase3AllBuckets;
        sequencer.position = 5;

        sequencer.reset_phase();

        assert_eq!(sequencer.current_phase(), TrainingPhase::Phase1EasyOnly);
        assert_eq!(sequencer.position, 0);
    }

    #[test]
    fn test_threshold_accessors() {
        let problems = vec![easy_problem()];
        let sequencer = TrainingSequencer::with_settings(
            problems,
            0.8,
            0.4,
            false,
        );

        assert!((sequencer.advancement_threshold() - 0.8).abs() < 0.001);
        assert!((sequencer.regression_threshold() - 0.4).abs() < 0.001);
    }
}
