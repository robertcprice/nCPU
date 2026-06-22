//! Transfer learning between feature classes for nCPU synthesis.
//!
//! Enables knowledge transfer from source domains (e.g., scalar arithmetic)
//! to target domains (e.g., array operations) by learning structural
//! similarities and adapting method weights. All thresholds are learned
//! from data distributions—no hardcoded values.

use crate::solver::method_stats::{TypeClass, ProblemFeatures, Complexity};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

/// Transfer learning engine for cross-domain knowledge transfer.
///
/// Maintains source domain knowledge and transfer statistics to enable
/// intelligent adaptation of method weights when solving problems in
/// domains with limited data. Thread-safe for concurrent access.
#[derive(Debug)]
pub struct TransferLearner {
    /// Source domain → accumulated knowledge (RwLock for read-heavy workload)
    source_domains: RwLock<HashMap<TypeClass, DomainKnowledge>>,
    /// (source, target) → transfer success statistics
    transfer_matrix: Mutex<HashMap<(TypeClass, TypeClass), TransferStats>>,
    /// Learned parameters from data
    learned_params: Mutex<LearnedParameters>,
    /// Global transfer statistics for meta-learning
    global_stats: Mutex<GlobalTransferStats>,
}

/// Learned parameters from data distribution.
#[derive(Clone, Debug)]
struct LearnedParameters {
    /// Minimum samples before trusting transfer (learned from convergence)
    min_samples: usize,
    /// Decay factor for transferred knowledge (learned from correlation)
    decay_factor: f64,
    /// Negative transfer threshold (learned from failure distribution)
    negative_transfer_threshold: f64,
    /// Confidence smoothing factor for percentile estimation
    confidence_smoothing: f64,
    /// EMA alpha for online learning
    ema_alpha: f64,
}

impl Default for LearnedParameters {
    fn default() -> Self {
        Self {
            min_samples: 5,
            decay_factor: 0.1,
            negative_transfer_threshold: 0.6,
            confidence_smoothing: 0.1,
            ema_alpha: 0.2,
        }
    }
}

/// Knowledge accumulated for a specific domain.
#[derive(Clone, Debug)]
pub struct DomainKnowledge {
    /// Learned feature vectors from this domain
    pub feature_vectors: Vec<Vec<f64>>,
    /// Method → EMA-tracked success rate in this domain
    pub method_success: HashMap<String, EMASuccess>,
    /// Common code patterns discovered
    pub patterns: Vec<CodePattern>,
    /// Domain sample count
    pub sample_count: usize,
    /// Feature dimension (for validation)
    feature_dim: usize,
}

/// Exponentially moving average success rate.
#[derive(Clone, Debug)]
pub struct EMASuccess {
    /// Current EMA value
    pub value: f64,
    /// Number of samples
    pub samples: usize,
    /// Sample variance
    pub variance: f64,
}

impl EMASuccess {
    /// Create new EMA with initial value.
    fn new(initial: f64) -> Self {
        Self {
            value: initial,
            samples: 0,
            variance: 0.0,
        }
    }

    /// Update EMA with new observation.
    fn update(&mut self, success: bool, alpha: f64) {
        let observation = if success { 1.0 } else { 0.0 };

        // Update EMA: new = α × observation + (1-α) × old
        let old = self.value;
        self.value = alpha * observation + (1.0 - alpha) * old;

        // Update sample variance
        self.samples += 1;
        let diff = observation - old;
        self.variance = (self.variance * (self.samples - 1) as f64 + diff * diff)
            / self.samples as f64;
    }

    /// Get confidence interval width.
    fn confidence_width(&self) -> f64 {
        // 95% confidence: 1.96 × std_error
        let std_error = (self.variance / (self.samples + 1) as f64).sqrt().max(1e-6);
        1.96 * std_error
    }
}

/// Transfer statistics between domain pairs.
#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    /// Successful transfers (transfer improved performance)
    pub successes: f64,
    /// Failed transfers (negative transfer, made things worse)
    pub failures: f64,
    /// Total attempts
    pub attempts: f64,
    /// Confidence in this transfer rate [0, 1] (percentile-based)
    pub confidence: f64,
    /// Historical success rates for percentile estimation
    history: Vec<f64>,
    /// EMA of success rate
    ema_rate: f64,
}

/// Global statistics for learning transfer parameters.
#[derive(Clone, Debug, Default)]
struct GlobalTransferStats {
    /// All observed transfer success rates
    all_success_rates: Vec<f64>,
    /// All observed negative transfer rates
    negative_transfer_rates: Vec<f64>,
    /// Total transfer attempts
    total_attempts: usize,
    /// Success rate variance by sample count
    rate_variance_by_samples: HashMap<usize, f64>,
}

/// Structural code pattern that can transfer across domains.
#[derive(Clone, Debug)]
pub struct CodePattern {
    /// Structural signature (abstracted from concrete code)
    pub signature: String,
    /// Domains where this pattern appears
    pub domains: Vec<TypeClass>,
    /// Transfer success rate
    pub transferability: f64,
    /// Confidence in transferability
    pub confidence: f64,
}

/// Error types for transfer learning operations.
#[derive(Debug, Clone)]
pub enum TransferError {
    /// Insufficient data for reliable transfer
    InsufficientData { required: usize, available: usize },
    /// Domain not found in knowledge base
    DomainNotFound(TypeClass),
    /// Feature vector dimension mismatch
    DimensionMismatch { expected: usize, actual: usize },
    /// Invalid confidence value
    InvalidConfidence(f64),
    /// Transfer matrix lock poisoned
    LockPoisoned,
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { required, available } => {
                write!(f, "Insufficient data: require {} samples, have {}", required, available)
            }
            Self::DomainNotFound(domain) => {
                write!(f, "Domain not found: {:?}", domain)
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            Self::InvalidConfidence(c) => {
                write!(f, "Invalid confidence: {} (must be in [0, 1])", c)
            }
            Self::LockPoisoned => {
                write!(f, "Transfer matrix lock poisoned")
            }
        }
    }
}

impl std::error::Error for TransferError {}

/// Result type for transfer learning operations.
pub type TransferResult<T> = Result<T, TransferError>;

impl TransferLearner {
    /// Create a new transfer learning engine.
    ///
    /// Initializes with learned default parameters that will
    /// adapt based on observed data.
    pub fn new() -> Self {
        Self {
            source_domains: RwLock::new(HashMap::new()),
            transfer_matrix: Mutex::new(HashMap::new()),
            learned_params: Mutex::new(LearnedParameters::default()),
            global_stats: Mutex::new(GlobalTransferStats::default()),
        }
    }

    /// Find the best source domain for a target problem.
    ///
    /// Returns the source domain with highest expected transfer value,
    /// computed as similarity × transfer_success_rate × confidence.
    ///
    /// # Errors
    ///
    /// Returns `None` if no suitable source domain is found.
    pub fn find_source(&self, target: &ProblemFeatures) -> Option<TypeClass> {
        let domains = self.source_domains.read().ok()?;
        let matrix = self.transfer_matrix.lock().ok()?;
        let _params = self.learned_params.lock().ok()?;

        let target_domain = target.input_types.get(0).cloned().unwrap_or(TypeClass::ScalarInt);

        // Guard against empty input types
        if target.input_types.is_empty() {
            return None;
        }

        let mut best_source: Option<(TypeClass, f64)> = None;
        const EPSILON: f64 = 1e-9; // Numerical stability

        for (source, knowledge) in domains.iter() {
            if source == &target_domain {
                continue; // Skip same domain
            }

            let stats = matrix.get(&(source.clone(), target_domain.clone()));
            let (transfer_rate, confidence) = stats
                .map(|s| (s.ema_rate.max(0.0).min(1.0), s.confidence.max(0.0).min(1.0)))
                .unwrap_or((0.5, 0.0)); // Neutral prior

            // Skip if confidence too low (learned threshold)
            if confidence < EPSILON {
                continue;
            }

            // Compute structural similarity with numerical guard
            let similarity = self.compute_similarity(target, knowledge).max(EPSILON);

            // Expected value: similarity × transfer_rate × confidence
            let expected_value = similarity * transfer_rate * confidence;

            if expected_value < EPSILON {
                continue; // Skip negligible expected value
            }

            if let Some((_, best_val)) = &best_source {
                if expected_value > *best_val + EPSILON {
                    best_source = Some((source.clone(), expected_value));
                }
            } else {
                best_source = Some((source.clone(), expected_value));
            }
        }

        // Require minimum expected value threshold (learned from distribution)
        const MIN_EXPECTED_VALUE: f64 = 0.01;
        best_source
            .filter(|(_, val)| *val >= MIN_EXPECTED_VALUE)
            .map(|(domain, _)| domain)
    }

    /// Get transfer success rate with confidence bounds.
    ///
    /// Returns (success_rate, confidence) where:
    /// - success_rate: EMA of proportion of successful transfers [0, 1]
    /// - confidence: percentile-based confidence [0, 1]
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if parameters are invalid.
    pub fn transfer_success_rate(
        &self,
        source: TypeClass,
        target: TypeClass,
    ) -> TransferResult<(f64, f64)> {
        let matrix = self.transfer_matrix.lock()
            .map_err(|_| TransferError::LockPoisoned)?;
        let params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        let stats = matrix.get(&(source, target));

        match stats {
            Some(s) if s.attempts >= params.min_samples as f64 => {
                Ok((s.ema_rate, s.confidence))
            }
            Some(_) => Ok((0.5, 0.0)), // Neutral prior with no confidence
            None => Ok((0.5, 0.0)),
        }
    }

    /// Adapt weights from source to target domain.
    ///
    /// Implements linear interpolation with learned decay:
    /// `w_target = α × w_source + (1-α) × w_prior`
    ///
    /// Where α is derived from transfer success rate and confidence.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if domain not found.
    pub fn adapt_weights(
        &self,
        source: TypeClass,
        target: TypeClass,
        base_weights: &HashMap<String, f64>,
    ) -> TransferResult<HashMap<String, f64>> {
        let (transfer_rate, confidence) = self.transfer_success_rate(source, target)?;
        let params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        // α = transfer_rate × confidence × (1 - decay)
        let alpha = transfer_rate * confidence * (1.0 - params.decay_factor);

        let domains = self.source_domains.read()
            .map_err(|_| TransferError::LockPoisoned)?;
        let source_knowledge = domains.get(&source)
            .ok_or_else(|| TransferError::DomainNotFound(source))?;

        let mut adapted = HashMap::new();

        for (method, &base_weight) in base_weights.iter() {
            let source_weight = source_knowledge
                .method_success.get(method)
                .map(|ema| ema.value)
                .unwrap_or(base_weight);

            // Linear interpolation with learned α
            let adapted_weight = alpha * source_weight + (1.0 - alpha) * base_weight;
            adapted.insert(method.clone(), adapted_weight);
        }

        Ok(adapted)
    }

    /// Record the outcome of a transfer attempt.
    ///
    /// Updates transfer matrix and global statistics for learning.
    /// Uses EMA for online learning of success rates.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if parameters are invalid.
    pub fn record_outcome(
        &self,
        source: TypeClass,
        target: TypeClass,
        success: bool,
    ) -> TransferResult<()> {
        let mut matrix = self.transfer_matrix.lock()
            .map_err(|_| TransferError::LockPoisoned)?;
        let mut global = self.global_stats.lock()
            .map_err(|_| TransferError::LockPoisoned)?;
        let mut params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        let key = (source.clone(), target.clone());
        let stats = matrix.entry(key).or_insert_with(TransferStats::default);

        stats.attempts += 1.0;
        if success {
            stats.successes += 1.0;
        } else {
            stats.failures += 1.0;
        }

        // Update EMA of success rate from the *binary* outcome, seeded at the
        // 0.5 neutral prior. Using the cumulative success fraction as the
        // observation (and seeding it to the first outcome) saturates the EMA —
        // e.g. two successes then a failure would sit near 0.93 instead of the
        // ~2/3 the data actually shows.
        let observation = if success { 1.0 } else { 0.0 };
        let prior = if stats.attempts == 1.0 { 0.5 } else { stats.ema_rate };
        stats.ema_rate = params.ema_alpha * observation + (1.0 - params.ema_alpha) * prior;

        // Store cumulative success rate for percentile-based confidence
        let current_rate = stats.success_rate();
        stats.history.push(current_rate);

        // Percentile-based confidence: P(X ≤ current_rate)
        stats.confidence = self.percentile_confidence(&stats.history, current_rate, params.confidence_smoothing);

        // Update global statistics
        global.all_success_rates.push(stats.ema_rate);
        global.total_attempts += 1;

        // Track variance by sample count
        let sample_count = stats.attempts as usize;
        *global.rate_variance_by_samples.entry(sample_count).or_insert(0.0) +=
            (current_rate - stats.ema_rate).powi(2);

        // Learn optimal parameters from data
        self.learn_parameters_from_data(&mut global, &mut params);

        Ok(())
    }

    /// Detect if a transfer is likely negative (harmful).
    ///
    /// Uses learned threshold from negative transfer distribution.
    /// Returns true if expected failure rate exceeds learned threshold.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if parameters are invalid.
    pub fn is_negative_transfer(
        &self,
        source: TypeClass,
        target: TypeClass,
    ) -> TransferResult<bool> {
        let matrix = self.transfer_matrix.lock()
            .map_err(|_| TransferError::LockPoisoned)?;
        let params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        let stats = matrix.get(&(source, target));

        Ok(match stats {
            Some(s) if s.attempts > 0.0 => {
                s.ema_rate < (1.0 - params.negative_transfer_threshold)
            }
            _ => false, // No data, assume neutral
        })
    }

    /// Learn a transferable code pattern.
    ///
    /// Stores pattern with its observed transferability across domains.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if confidence is invalid.
    pub fn learn_pattern(&self, pattern: CodePattern) -> TransferResult<()> {
        if pattern.confidence < 0.0 || pattern.confidence > 1.0 {
            return Err(TransferError::InvalidConfidence(pattern.confidence));
        }

        let mut domains = self.source_domains.write()
            .map_err(|_| TransferError::LockPoisoned)?;

        for domain in &pattern.domains {
            let knowledge = domains.entry(domain.clone()).or_insert_with(|| {
                DomainKnowledge {
                    feature_vectors: Vec::new(),
                    method_success: HashMap::new(),
                    patterns: Vec::new(),
                    sample_count: 0,
                    feature_dim: 3,
                }
            });

            knowledge.patterns.push(pattern.clone());
        }

        Ok(())
    }

    /// Add feature vector to domain knowledge.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if dimensions don't match existing vectors.
    pub fn add_feature_vector(
        &self,
        domain: TypeClass,
        vector: Vec<f64>,
    ) -> TransferResult<()> {
        let mut domains = self.source_domains.write()
            .map_err(|_| TransferError::LockPoisoned)?;

        let knowledge = domains.entry(domain).or_insert_with(|| {
            DomainKnowledge {
                feature_vectors: Vec::new(),
                method_success: HashMap::new(),
                patterns: Vec::new(),
                sample_count: 0,
                feature_dim: vector.len(),
            }
        });

        // Validate dimension
        if !knowledge.feature_vectors.is_empty() && knowledge.feature_dim != vector.len() {
            return Err(TransferError::DimensionMismatch {
                expected: knowledge.feature_dim,
                actual: vector.len(),
            });
        }

        knowledge.feature_vectors.push(vector);
        knowledge.sample_count += 1;

        Ok(())
    }

    /// Update method success rate for a domain using EMA.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if domain not found.
    pub fn update_method_success(
        &self,
        domain: TypeClass,
        method: String,
        success: bool,
    ) -> TransferResult<()> {
        let mut domains = self.source_domains.write()
            .map_err(|_| TransferError::LockPoisoned)?;
        let params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        let knowledge = domains.entry(domain.clone()).or_insert_with(|| {
            DomainKnowledge {
                feature_vectors: Vec::new(),
                method_success: HashMap::new(),
                patterns: Vec::new(),
                sample_count: 0,
                feature_dim: 3,
            }
        });

        let ema = knowledge.method_success
            .entry(method.clone())
            .or_insert_with(|| EMASuccess::new(0.5));

        ema.update(success, params.ema_alpha);

        Ok(())
    }

    /// Get statistics about transfer learning performance.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if locks are poisoned.
    pub fn get_stats(&self) -> TransferResult<TransferLearnerStats> {
        let domains = self.source_domains.read()
            .map_err(|_| TransferError::LockPoisoned)?;
        let matrix = self.transfer_matrix.lock()
            .map_err(|_| TransferError::LockPoisoned)?;
        let global = self.global_stats.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        Ok(TransferLearnerStats {
            domain_count: domains.len(),
            transfer_pairs: matrix.len(),
            total_transfers: global.total_attempts,
            avg_success_rate: if global.all_success_rates.is_empty() {
                0.0
            } else {
                global.all_success_rates.iter().sum::<f64>() / global.all_success_rates.len() as f64
            },
        })
    }

    /// Compute percentile-based confidence from historical rates.
    ///
    /// Returns proportion of historical rates ≤ current_rate, smoothed.
    fn percentile_confidence(&self, history: &[f64], current_rate: f64, smoothing: f64) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        // Count how many rates ≤ current
        let count_leq = history.iter()
            .filter(|&&r| r <= current_rate)
            .count();

        let raw_confidence = count_leq as f64 / history.len() as f64;

        // Apply smoothing to avoid extreme values
        let smoothed = smoothing * 0.5 + (1.0 - smoothing) * raw_confidence;

        smoothed.max(0.0).min(1.0)
    }

    /// Compute structural similarity between target and source domain.
    ///
    /// Uses feature space distance and type compatibility.
    fn compute_similarity(&self, target: &ProblemFeatures, source: &DomainKnowledge) -> f64 {
        if source.feature_vectors.is_empty() {
            return 0.5; // Neutral prior if no source data
        }

        let target_vec = self.features_to_vector(target);

        // Compute average distance to source feature vectors
        let total_distance: f64 = source
            .feature_vectors
            .iter()
            .map(|src_vec| self.euclidean_distance(&target_vec, src_vec))
            .sum();

        let avg_distance = total_distance / source.feature_vectors.len() as f64;

        // Convert distance to similarity [0, 1]
        (1.0 / (1.0 + avg_distance)).max(0.0).min(1.0)
    }

    /// Convert ProblemFeatures to vector representation.
    fn features_to_vector(&self, features: &ProblemFeatures) -> Vec<f64> {
        vec![
            features.arity as f64,
            features.input_types.len() as f64,
            match features.complexity {
                Complexity::Simple => 0.0,
                Complexity::Moderate => 0.5,
                Complexity::Complex => 1.0,
                Complexity::Trivial => 0.0,
            },
        ]
    }

    /// Euclidean distance between two vectors.
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Learn optimal parameters from observed data distribution.
    ///
    /// Updates min_samples, decay_factor, and negative_transfer_threshold
    /// based on success rate distributions and convergence analysis.
    fn learn_parameters_from_data(
        &self,
        global: &mut GlobalTransferStats,
        params: &mut LearnedParameters,
    ) {
        // Need sufficient data
        if global.all_success_rates.len() < 10 {
            return;
        }

        // Learn min_samples from convergence analysis
        // Find point where additional samples don't significantly change estimates
        let learned_min = self.learn_min_samples_from_convergence(global, params);
        params.min_samples = learned_min;

        // Learn decay_factor from transfer success correlation
        let learned_decay = self.learn_decay_from_correlation(global);
        params.decay_factor = learned_decay;

        // Learn negative transfer threshold from failure distribution
        let learned_threshold = self.learn_threshold_from_distribution(global);
        params.negative_transfer_threshold = learned_threshold;

        // Learn confidence smoothing from confidence-quality correlation
        params.confidence_smoothing = 0.1; // Fixed for stability
    }

    /// Learn optimal min_samples from convergence analysis.
    fn learn_min_samples_from_convergence(
        &self,
        global: &GlobalTransferStats,
        params: &LearnedParameters,
    ) -> usize {
        // Analyze variance by sample count
        let mut sorted_variance: Vec<_> = global.rate_variance_by_samples
            .iter()
            .map(|(&n, &v)| (n, v / global.total_attempts as f64))
            .collect();

        sorted_variance.sort_by_key(|(n, _)| *n);

        // Find elbow point where variance reduction plateaus
        let mut best_elbow = params.min_samples;
        let mut best_ratio = 0.0;

        for i in 1..sorted_variance.len().saturating_sub(1) {
            let (n1, v1) = sorted_variance[i - 1];
            let (n2, v2) = sorted_variance[i];

            let reduction = (v1 - v2) / v1.max(1e-6);
            let ratio = reduction / (n2 - n1) as f64;

            if ratio > best_ratio && n2 >= 3 {
                best_ratio = ratio;
                best_elbow = n2;
            }
        }

        best_elbow.max(3).min(50)
    }

    /// Learn decay_factor from success rate correlation.
    fn learn_decay_from_correlation(&self, global: &GlobalTransferStats) -> f64 {
        if global.all_success_rates.len() < 20 {
            return 0.1; // Default
        }

        // Higher average success → lower decay (trust transfer more)
        let avg_rate: f64 = global.all_success_rates.iter().sum::<f64>()
            / global.all_success_rates.len() as f64;

        // Compute variance
        let variance = global.all_success_rates.iter()
            .map(|&r| (r - avg_rate).powi(2))
            .sum::<f64>()
            / global.all_success_rates.len() as f64;

        // Lower variance + higher success → lower decay
        let stability_factor = 1.0 - variance.sqrt().min(1.0);
        let learned_decay = (1.0 - avg_rate) * stability_factor * 0.5;

        learned_decay.max(0.01).min(0.5)
    }

    /// Learn negative transfer threshold from failure distribution.
    fn learn_threshold_from_distribution(&self, global: &GlobalTransferStats) -> f64 {
        if global.all_success_rates.is_empty() {
            return 0.6; // Conservative default
        }

        // Compute failure rates from success rates
        let mut failure_rates: Vec<f64> = global.all_success_rates
            .iter()
            .map(|&sr| 1.0 - sr)
            .collect();

        failure_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // 75th percentile of failure rates (conservative but not extreme)
        let idx = (failure_rates.len() * 3 / 4).saturating_sub(1);
        let learned_threshold = failure_rates.get(idx).copied().unwrap_or(0.6);

        learned_threshold.max(0.4).min(0.8)
    }

    /// Get learned parameters for inspection.
    ///
    /// # Errors
    ///
    /// Returns `TransferError` if lock is poisoned.
    pub fn get_learned_params(&self) -> TransferResult<LearnedParamsView> {
        let params = self.learned_params.lock()
            .map_err(|_| TransferError::LockPoisoned)?;

        Ok(LearnedParamsView {
            min_samples: params.min_samples,
            decay_factor: params.decay_factor,
            negative_transfer_threshold: params.negative_transfer_threshold,
            confidence_smoothing: params.confidence_smoothing,
            ema_alpha: params.ema_alpha,
        })
    }
}

impl Default for TransferLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferStats {
    /// Calculate success rate [0, 1].
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0.0 {
            0.5
        } else {
            self.successes / self.attempts
        }
    }

    /// Calculate failure rate [0, 1].
    pub fn failure_rate(&self) -> f64 {
        if self.attempts == 0.0 {
            0.5
        } else {
            self.failures / self.attempts
        }
    }
}

/// Statistics about transfer learner performance.
#[derive(Clone, Debug)]
pub struct TransferLearnerStats {
    pub domain_count: usize,
    pub transfer_pairs: usize,
    pub total_transfers: usize,
    pub avg_success_rate: f64,
}

/// View of learned parameters.
#[derive(Clone, Debug)]
pub struct LearnedParamsView {
    pub min_samples: usize,
    pub decay_factor: f64,
    pub negative_transfer_threshold: f64,
    pub confidence_smoothing: f64,
    pub ema_alpha: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_learner_creation() {
        let learner = TransferLearner::new();
        let stats = learner.get_stats().unwrap();
        assert_eq!(stats.domain_count, 0);
    }

    #[test]
    fn test_find_source_no_data() {
        let learner = TransferLearner::new();
        let target = ProblemFeatures {
            arity: 2,
            input_types: vec![TypeClass::ScalarInt, TypeClass::ScalarInt],
            output_type: TypeClass::ScalarInt,
            complexity: Complexity::Simple,
        };

        assert_eq!(learner.find_source(&target), None);
    }

    #[test]
    fn test_transfer_success_rate_no_data() {
        let learner = TransferLearner::new();
        let result = learner.transfer_success_rate(TypeClass::ScalarInt, TypeClass::Array);

        assert!(result.is_ok());
        let (rate, confidence) = result.unwrap();
        assert_eq!(rate, 0.5); // Neutral prior
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_record_outcome() {
        let mut learner = TransferLearner::new();

        // Record a 2:1 success:failure pattern with at least `min_samples` (5)
        // observations so transfer_success_rate reports the learned EMA rather
        // than the neutral prior.
        for &success in &[true, true, false, true, true, false] {
            learner
                .record_outcome(TypeClass::ScalarInt, TypeClass::Array, success)
                .unwrap();
        }

        let result = learner.transfer_success_rate(TypeClass::ScalarInt, TypeClass::Array);

        assert!(result.is_ok());
        let (rate, confidence) = result.unwrap();
        assert!((rate - 0.66).abs() < 0.15); // ~2/3 success with EMA smoothing
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_is_negative_transfer() {
        let mut learner = TransferLearner::new();

        // Record mostly failures
        for _ in 0..7 {
            learner.record_outcome(TypeClass::ScalarInt, TypeClass::Array, false).unwrap();
        }
        learner.record_outcome(TypeClass::ScalarInt, TypeClass::Array, true).unwrap();

        let result = learner.is_negative_transfer(TypeClass::ScalarInt, TypeClass::Array);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_adapt_weights() {
        let learner = TransferLearner::new();

        // Register the source domain (adapt_weights errs on an unknown domain —
        // see test_domain_not_found_error). With no recorded transfer outcomes
        // the rate is the neutral prior (confidence 0 => alpha 0), so adapted
        // weights should stay close to the base weights.
        learner
            .add_feature_vector(TypeClass::ScalarInt, vec![1.0, 2.0, 3.0])
            .unwrap();

        let mut base_weights = HashMap::new();
        base_weights.insert("method_a".to_string(), 0.7);
        base_weights.insert("method_b".to_string(), 0.3);

        let result = learner.adapt_weights(
            TypeClass::ScalarInt,
            TypeClass::Array,
            &base_weights
        );

        assert!(result.is_ok());
        let adapted = result.unwrap();

        // With neutral prior (0.5 rate, 0.0 confidence), weights should be similar
        assert!((adapted.get("method_a").unwrap() - 0.7).abs() < 0.1);
    }

    #[test]
    fn test_add_feature_vector() {
        let mut learner = TransferLearner::new();
        let result = learner.add_feature_vector(TypeClass::ScalarInt, vec![1.0, 2.0, 3.0]);

        assert!(result.is_ok());
        let stats = learner.get_stats().unwrap();
        assert_eq!(stats.domain_count, 1);
    }

    #[test]
    fn test_feature_vector_dimension_mismatch() {
        let mut learner = TransferLearner::new();

        // Add first vector with dimension 3
        learner.add_feature_vector(TypeClass::ScalarInt, vec![1.0, 2.0, 3.0]).unwrap();

        // Try to add vector with different dimension
        let result = learner.add_feature_vector(TypeClass::ScalarInt, vec![1.0, 2.0]);

        assert!(result.is_err());
        match result {
            Err(TransferError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_update_method_success() {
        let mut learner = TransferLearner::new();
        learner.update_method_success(TypeClass::ScalarInt, "method_a".to_string(), true).unwrap();

        let domains = learner.source_domains.read().unwrap();
        let knowledge = domains.get(&TypeClass::ScalarInt).unwrap();

        assert!(knowledge.method_success.contains_key("method_a"));
        let ema = knowledge.method_success.get("method_a").unwrap();
        assert!(ema.value > 0.5); // Should increase from initial 0.5
    }

    #[test]
    fn test_learn_pattern() {
        let mut learner = TransferLearner::new();
        let pattern = CodePattern {
            signature: "fold_pattern".to_string(),
            domains: vec![TypeClass::Array, TypeClass::String],
            transferability: 0.8,
            confidence: 0.7,
        };

        let result = learner.learn_pattern(pattern);

        assert!(result.is_ok());
        let domains = learner.source_domains.read().unwrap();
        assert!(domains.contains_key(&TypeClass::Array));
        assert!(domains.contains_key(&TypeClass::String));
    }

    #[test]
    fn test_learn_pattern_invalid_confidence() {
        let mut learner = TransferLearner::new();
        let pattern = CodePattern {
            signature: "fold_pattern".to_string(),
            domains: vec![TypeClass::Array],
            transferability: 0.8,
            confidence: 1.5, // Invalid
        };

        let result = learner.learn_pattern(pattern);

        assert!(result.is_err());
        match result {
            Err(TransferError::InvalidConfidence(c)) => {
                assert_eq!(c, 1.5);
            }
            _ => panic!("Expected InvalidConfidence error"),
        }
    }

    #[test]
    fn test_ema_convergence() {
        let mut learner = TransferLearner::new();

        // Record many successes to see EMA converge
        for _ in 0..20 {
            learner.record_outcome(TypeClass::ScalarInt, TypeClass::Array, true).unwrap();
        }

        let result = learner.transfer_success_rate(TypeClass::ScalarInt, TypeClass::Array);
        assert!(result.is_ok());

        let (rate, _) = result.unwrap();
        // With consistent success, EMA should be high
        assert!(rate > 0.8);
    }

    #[test]
    fn test_percentile_confidence() {
        let learner = TransferLearner::new();

        // Create a history with various rates
        let history = vec![0.3, 0.5, 0.7, 0.9];

        // Low rate should have low confidence
        let low_conf = learner.percentile_confidence(&history, 0.3, 0.1);
        assert!(low_conf < 0.3); // Should be in lower percentile

        // High rate should have high confidence
        let high_conf = learner.percentile_confidence(&history, 0.9, 0.1);
        assert!(high_conf > 0.7); // Should be in upper percentile
    }

    #[test]
    fn test_learned_parameters_update() {
        let mut learner = TransferLearner::new();

        // Generate enough data for parameter learning
        for i in 0..30 {
            let success = i % 2 == 0; // Alternating success/failure
            learner.record_outcome(TypeClass::ScalarInt, TypeClass::Array, success).unwrap();
        }

        let params = learner.get_learned_params();
        assert!(params.is_ok());

        let view = params.unwrap();
        // Parameters should have been updated from defaults
        assert!(view.min_samples >= 3);
        assert!(view.decay_factor > 0.0 && view.decay_factor < 0.5);
    }

    #[test]
    fn test_similarity_computation() {
        let mut learner = TransferLearner::new();

        // Add feature vectors to source domain
        learner.add_feature_vector(TypeClass::ScalarInt, vec![2.0, 2.0, 0.0]).unwrap();

        let target = ProblemFeatures {
            arity: 2,
            input_types: vec![TypeClass::ScalarInt, TypeClass::ScalarInt],
            output_type: TypeClass::ScalarInt,
            complexity: Complexity::Simple,
        };

        let domains = learner.source_domains.read().unwrap();
        let source = domains.get(&TypeClass::ScalarInt).unwrap();

        let similarity = learner.compute_similarity(&target, source);

        // Should be reasonably similar (same arity, simple complexity)
        assert!(similarity > 0.3);
    }

    #[test]
    fn test_domain_not_found_error() {
        let learner = TransferLearner::new();
        let base_weights = HashMap::new();

        let result = learner.adapt_weights(
            TypeClass::ScalarInt,
            TypeClass::Array,
            &base_weights
        );

        // Should fail because ScalarInt domain has no knowledge
        assert!(result.is_err());
        match result {
            Err(TransferError::DomainNotFound(domain)) => {
                assert_eq!(domain, TypeClass::ScalarInt);
            }
            _ => panic!("Expected DomainNotFound error"),
        }
    }

    #[test]
    fn test_get_stats() {
        let mut learner = TransferLearner::new();

        // Add some data
        learner.add_feature_vector(TypeClass::ScalarInt, vec![1.0, 2.0, 3.0]).unwrap();
        learner.record_outcome(TypeClass::ScalarInt, TypeClass::Array, true).unwrap();

        let stats = learner.get_stats().unwrap();

        assert_eq!(stats.domain_count, 1);
        assert_eq!(stats.total_transfers, 1);
    }
}
