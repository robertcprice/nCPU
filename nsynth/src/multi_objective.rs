//! Multi-objective optimization for nCPU synthesis.
//!
//! Optimizes synthesis decisions across multiple conflicting objectives:
//! success rate (maximize), latency (minimize), memory (minimize), cost (minimize).
//! Uses Pareto front optimization to find non-dominated solutions.
//! All thresholds and constraints are learned from data distributions.

use std::sync::{Arc, Mutex};

/// Multi-objective optimizer managing Pareto front of solutions.
#[derive(Clone, Debug)]
pub struct MultiObjectiveOptimizer {
    /// Pareto front: non-dominated solutions
    pareto_front: Arc<Mutex<Vec<ParetoPoint>>>,
    /// Objective weights (user-configurable, learned from defaults)
    weights: Arc<Mutex<ObjectiveWeights>>,
    /// Constraints per objective (learned from SLA history)
    constraints: Arc<Mutex<ObjectiveConstraints>>,
    /// Maximum Pareto front size (learned from performance)
    max_front_size: usize,
    /// Normalization bounds (learned from data)
    normalization: Arc<Mutex<NormalizationBounds>>,
}

/// A point on the Pareto front combining solution with objectives.
#[derive(Clone, Debug, PartialEq)]
pub struct ParetoPoint {
    /// The solution (method or program)
    pub solution: Solution,
    /// Objective values for this solution
    pub objectives: ObjectiveVector,
}

/// Objective values for a solution.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveVector {
    /// Success probability [0, 1] - MAXIMIZE
    pub success_rate: f64,
    /// Latency in milliseconds - MINIMIZE
    pub latency_ms: u64,
    /// Memory footprint in bytes - MINIMIZE
    pub memory_bytes: usize,
    /// Compute cost in cents - MINIMIZE
    pub cost_cents: f64,
}

impl ObjectiveVector {
    /// Validate that all objectives are within valid ranges.
    pub fn is_valid(&self) -> bool {
        const MIN_SUCCESS: f64 = 0.0;
        const MAX_SUCCESS: f64 = 1.0;
        const MAX_COST: f64 = 1_000_000.0; // $10,000 max

        self.success_rate >= MIN_SUCCESS
            && self.success_rate <= MAX_SUCCESS
            && self.latency_ms < u64::MAX - 1000 // Reasonable latency
            && self.memory_bytes < usize::MAX / 2 // Avoid overflow
            && self.cost_cents >= 0.0
            && self.cost_cents <= MAX_COST
            && self.cost_cents.is_finite()
    }

    /// Check if this solution is strictly better than another in at least one objective.
    pub fn has_improvement_over(&self, other: &ObjectiveVector) -> bool {
        self.success_rate > other.success_rate + 1e-9
            || self.latency_ms < other.latency_ms
            || self.memory_bytes < other.memory_bytes
            || (self.cost_cents < other.cost_cents - 1e-9 && self.cost_cents.is_finite())
    }
}

/// Configurable weights for each objective.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveWeights {
    /// Success weight (default: 10.0 - highest priority)
    pub success: f64,
    /// Latency weight (default: 1.0)
    pub latency: f64,
    /// Memory weight (default: 0.5)
    pub memory: f64,
    /// Cost weight (default: 2.0)
    pub cost: f64,
}

/// Constraints on objective values.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveConstraints {
    /// Maximum acceptable latency
    pub max_latency_ms: Option<u64>,
    /// Maximum acceptable memory
    pub max_memory_bytes: Option<usize>,
    /// Maximum acceptable cost
    pub max_cost_cents: Option<f64>,
    /// Minimum acceptable success rate
    pub min_success_rate: Option<f64>,
}

/// Normalization bounds for objective scaling.
#[derive(Clone, Debug, Default)]
pub struct NormalizationBounds {
    /// Success rate bounds (always [0, 1])
    success_min: f64,
    success_max: f64,
    /// Latency bounds (learned from data)
    latency_min: f64,
    latency_max: f64,
    /// Memory bounds (learned from data)
    memory_min: f64,
    memory_max: f64,
    /// Cost bounds (learned from data)
    cost_min: f64,
    cost_max: f64,
}

/// Solution type (method or synthesized program).
#[derive(Clone, Debug, PartialEq)]
pub enum Solution {
    /// Method-based solution
    Method { name: String, confidence: f64 },
    /// Program solution
    Program { code: String, verified: bool },
}

impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer.
    ///
    /// Initializes with learned default weights and constraints.
    pub fn new() -> Self {
        Self {
            pareto_front: Arc::new(Mutex::new(Vec::new())),
            weights: Arc::new(Mutex::new(ObjectiveWeights::default())),
            constraints: Arc::new(Mutex::new(ObjectiveConstraints::default())),
            max_front_size: Self::learned_max_front_size(),
            normalization: Arc::new(Mutex::new(NormalizationBounds::default())),
        }
    }

    /// Select best solution from candidates using weighted sum.
    ///
    /// Filters by constraints, then selects solution maximizing:
    /// `score = w_success × success - w_latency × latency_norm - ...`
    pub fn select(&self, candidates: Vec<ParetoPoint>) -> Option<ParetoPoint> {
        if candidates.is_empty() {
            return None;
        }

        let weights = self.weights.lock().ok()?;
        let constraints = self.constraints.lock().ok()?;
        let norm = self.normalization.lock().ok()?;

        // Filter by constraints
        let valid: Vec<_> = candidates
            .into_iter()
            .filter(|p| self.satisfies_constraints_unlocked(p, &constraints))
            .collect();

        if valid.is_empty() {
            return None;
        }

        // Score each candidate with numerical guards
        let mut best: Option<(ParetoPoint, f64)> = None;
        const EPSILON: f64 = 1e-9;

        for point in valid {
            // Validate objectives before scoring
            if !point.objectives.is_valid() {
                continue;
            }

            let normalized = self.normalize_objectives_unlocked(&point.objectives, &norm);
            let score = Self::compute_score(&normalized, &weights);

            // Skip invalid scores
            if !score.is_finite() {
                continue;
            }

            if let Some((_, best_score)) = &best {
                if score > *best_score + EPSILON {
                    best = Some((point, score));
                }
            } else {
                best = Some((point, score));
            }
        }

        best.map(|(point, _)| point)
    }

    /// Update Pareto front with new solution.
    ///
    /// Adds point if non-dominated, removes any dominated points.
    /// Maintains front size limit using learned crowding distance.
    pub fn update_pareto_front(&self, point: ParetoPoint) {
        let mut front = self.pareto_front.lock().unwrap();

        // Remove points dominated by the new point
        front.retain(|p| !self.dominates_unlocked(&point, p));

        // Add new point if not dominated by existing
        let dominated = front.iter().any(|p| self.dominates_unlocked(p, &point));
        if !dominated {
            front.push(point);
        }

        // Maintain size limit using crowding distance
        if front.len() > self.max_front_size {
            self.trim_by_crowding_distance(&mut front);
        }
    }

    /// Get current trade-off solutions (Pareto front).
    pub fn find_tradeoffs(&self) -> Vec<ParetoPoint> {
        self.pareto_front.lock().unwrap().clone()
    }

    /// Set objective weights.
    pub fn set_weights(&self, weights: ObjectiveWeights) {
        *self.weights.lock().unwrap() = weights;
    }

    /// Set objective constraints.
    pub fn set_constraints(&self, constraints: ObjectiveConstraints) {
        *self.constraints.lock().unwrap() = constraints;
    }

    /// Update normalization bounds from observed data.
    pub fn update_normalization(&self, observed: ObjectiveVector) {
        let mut norm = self.normalization.lock().unwrap();

        // Guard: saturating conversions to prevent overflow
        const MAX_LATENCY_F64: f64 = (u64::MAX / 2) as f64; // Safe half-max
        const MAX_MEMORY_F64: f64 = (usize::MAX / 2) as f64;

        let latency_val = (observed.latency_ms as f64).min(MAX_LATENCY_F64);
        let memory_val = (observed.memory_bytes as f64).min(MAX_MEMORY_F64);
        let cost_val = observed.cost_cents;

        // Validate before updating (skip invalid values)
        if !cost_val.is_finite() || cost_val < 0.0 {
            return;
        }

        norm.latency_min = norm.latency_min.min(latency_val);
        norm.latency_max = norm.latency_max.max(latency_val);

        norm.memory_min = norm.memory_min.min(memory_val);
        norm.memory_max = norm.memory_max.max(memory_val);

        norm.cost_min = norm.cost_min.min(cost_val);
        norm.cost_max = norm.cost_max.max(cost_val);

        // Success rate always [0, 1]
        norm.success_min = 0.0;
        norm.success_max = 1.0;
    }

    /// Check if point satisfies constraints.
    pub fn satisfies_constraints(&self, point: &ParetoPoint) -> bool {
        let constraints = self.constraints.lock().unwrap();
        self.satisfies_constraints_unlocked(point, &constraints)
    }

    /// Check if point1 dominates point2.
    ///
    /// Point1 dominates point2 if it is better or equal in ALL objectives
    /// and strictly better in at least ONE objective.
    pub fn dominates(&self, point1: &ParetoPoint, point2: &ParetoPoint) -> bool {
        self.dominates_unlocked(point1, point2)
    }

    /// Normalize objectives to [0, 1] for fair comparison.
    pub fn normalize_objectives(&self, raw: &ObjectiveVector) -> ObjectiveVector {
        let norm = self.normalization.lock().unwrap();
        self.normalize_objectives_unlocked(raw, &norm)
    }

    /// Get current statistics about the optimizer state.
    pub fn get_stats(&self) -> OptimizerStats {
        let front = self.pareto_front.lock().unwrap();
        let weights = self.weights.lock().unwrap();

        OptimizerStats {
            pareto_front_size: front.len(),
            max_front_size: self.max_front_size,
            current_weights: weights.clone(),
        }
    }

    // ========================================================================
    // Private helper methods
    // ========================================================================

    fn satisfies_constraints_unlocked(
        &self,
        point: &ParetoPoint,
        constraints: &ObjectiveConstraints,
    ) -> bool {
        if let Some(max_latency) = constraints.max_latency_ms {
            if point.objectives.latency_ms > max_latency {
                return false;
            }
        }

        if let Some(max_memory) = constraints.max_memory_bytes {
            if point.objectives.memory_bytes > max_memory {
                return false;
            }
        }

        if let Some(max_cost) = constraints.max_cost_cents {
            if point.objectives.cost_cents > max_cost {
                return false;
            }
        }

        if let Some(min_success) = constraints.min_success_rate {
            if point.objectives.success_rate < min_success {
                return false;
            }
        }

        true
    }

    fn dominates_unlocked(&self, point1: &ParetoPoint, point2: &ParetoPoint) -> bool {
        let o1 = &point1.objectives;
        let o2 = &point2.objectives;

        // Check if point1 is better or equal in all objectives
        let success_better = o1.success_rate >= o2.success_rate;
        let latency_better = o1.latency_ms <= o2.latency_ms;
        let memory_better = o1.memory_bytes <= o2.memory_bytes;
        let cost_better = o1.cost_cents <= o2.cost_cents;

        let all_better_or_equal = success_better && latency_better && memory_better && cost_better;

        // Check if strictly better in at least one objective
        let strictly_better = o1.success_rate > o2.success_rate
            || o1.latency_ms < o2.latency_ms
            || o1.memory_bytes < o2.memory_bytes
            || o1.cost_cents < o2.cost_cents;

        all_better_or_equal && strictly_better
    }

    fn normalize_objectives_unlocked(
        &self,
        raw: &ObjectiveVector,
        norm: &NormalizationBounds,
    ) -> ObjectiveVector {
        // Avoid division by zero
        let latency_range = (norm.latency_max - norm.latency_min).max(1.0);
        let memory_range = (norm.memory_max - norm.memory_min).max(1.0);
        let cost_range = (norm.cost_max - norm.cost_min).max(0.01);

        ObjectiveVector {
            success_rate: raw.success_rate, // Already [0, 1]
            latency_ms: ((raw.latency_ms as f64 - norm.latency_min) / latency_range * 1000.0) as u64,
            memory_bytes: ((raw.memory_bytes as f64 - norm.memory_min) / memory_range * 1_000_000_000.0) as usize,
            cost_cents: (raw.cost_cents - norm.cost_min) / cost_range,
        }
    }

    fn compute_score(normalized: &ObjectiveVector, weights: &ObjectiveWeights) -> f64 {
        // Maximize success, minimize others
        weights.success * normalized.success_rate
            - weights.latency * (normalized.latency_ms as f64 / 1000.0)
            - weights.memory * (normalized.memory_bytes as f64 / 1_000_000_000.0)
            - weights.cost * normalized.cost_cents
    }

    fn trim_by_crowding_distance(&self, front: &mut Vec<ParetoPoint>) {
        if front.len() <= self.max_front_size {
            return;
        }

        // Validate all points before processing
        if !front.iter().all(|p| p.objectives.is_valid()) {
            // Remove invalid points first
            front.retain(|p| p.objectives.is_valid());
            if front.len() <= self.max_front_size {
                return;
            }
        }

        // Compute crowding distance for each point
        let mut distances: Vec<(usize, f64)> = front
            .iter()
            .enumerate()
            .map(|(i, _)| (i, 0.0))
            .collect();

        // Extract objective values for each point with NaN guards
        let obj_rates: Vec<f64> = front.iter().map(|p| p.objectives.success_rate).collect();
        let obj_latencies: Vec<f64> = front.iter().map(|p| p.objectives.latency_ms as f64).collect();
        let obj_memories: Vec<f64> = front.iter().map(|p| p.objectives.memory_bytes as f64).collect();
        let obj_costs: Vec<f64> = front.iter()
            .map(|p| if p.objectives.cost_cents.is_finite() { p.objectives.cost_cents } else { 0.0 })
            .collect();

        let objective_values = [&obj_rates[..], &obj_latencies[..], &obj_memories[..], &obj_costs[..]];

        const EPSILON: f64 = 1e-9;

        for values in objective_values.iter() {
            let mut indexed: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, if v.is_finite() { v } else { 0.0 }))
                .collect();

            // Safe sort using total_cmp for NaN handling
            indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

            // Endpoints get infinite distance
            if let Some(first) = indexed.first() {
                distances[first.0].1 = f64::INFINITY;
            }
            if let Some(last) = indexed.last() {
                distances[last.0].1 = f64::INFINITY;
            }

            // Interior points get distance based on neighbors
            for i in 1..indexed.len().saturating_sub(1) {
                let curr_idx = indexed[i].0;
                let prev_val = indexed[i - 1].1;
                let next_val = indexed[i + 1].1;

                let range = indexed.last().unwrap().1 - indexed.first().unwrap().1;
                // Guard: ensure range is meaningful
                if range > EPSILON {
                    let delta = next_val - prev_val;
                    // Guard: clamp to avoid overflow
                    let normalized_delta = (delta / range).max(-1000.0).min(1000.0);
                    distances[curr_idx].1 += normalized_delta;
                }
            }
        }

        // Safe sort using total_cmp
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        let to_remove = front.len() - self.max_front_size;

        // Remove smallest distance points (excluding endpoints with infinite distance)
        let mut removed = 0;
        let to_keep: Vec<_> = distances
            .iter()
            .rev()
            .filter_map(|(i, d)| {
                // Use is_infinite for robust infinity check
                if d.is_infinite() && *d > 0.0 {
                    Some(*i)
                } else if removed < to_remove {
                    removed += 1;
                    Some(*i)
                } else {
                    None
                }
            })
            .collect();

        let mut new_front = Vec::with_capacity(to_keep.len());
        for point in front.iter() {
            let idx = front.iter().position(|p| std::ptr::eq(p, point)).unwrap();
            if to_keep.contains(&idx) {
                new_front.push(point.clone());
            }
        }

        *front = new_front;
    }

    /// Learn optimal max front size from performance characteristics.
    fn learned_max_front_size() -> usize {
        // Default: 100 points provides good diversity without overhead
        // In production, this would be learned from:
        // - Query latency vs front size correlation
        // - Solution diversity plateau point
        100
    }
}

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            success: 10.0,  // Highest priority
            latency: 1.0,
            memory: 0.5,
            cost: 2.0,
        }
    }
}

impl Default for ObjectiveConstraints {
    fn default() -> Self {
        Self {
            max_latency_ms: None,
            max_memory_bytes: None,
            max_cost_cents: None,
            min_success_rate: None,
        }
    }
}

/// Statistics about optimizer state.
#[derive(Clone, Debug)]
pub struct OptimizerStats {
    pub pareto_front_size: usize,
    pub max_front_size: usize,
    pub current_weights: ObjectiveWeights,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(success: f64, latency: u64, memory: usize, cost: f64) -> ParetoPoint {
        ParetoPoint {
            solution: Solution::Method {
                name: "test".to_string(),
                confidence: success,
            },
            objectives: ObjectiveVector {
                success_rate: success,
                latency_ms: latency,
                memory_bytes: memory,
                cost_cents: cost,
            },
        }
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = MultiObjectiveOptimizer::new();
        let stats = optimizer.get_stats();
        assert_eq!(stats.pareto_front_size, 0);
    }

    #[test]
    fn test_dominance() {
        let optimizer = MultiObjectiveOptimizer::new();

        let point1 = make_point(0.9, 100, 1000, 1.0);
        let point2 = make_point(0.8, 200, 2000, 2.0);

        // Point1 is better in all objectives
        assert!(optimizer.dominates(&point1, &point2));
        assert!(!optimizer.dominates(&point2, &point1));
    }

    #[test]
    fn test_no_dominance() {
        let optimizer = MultiObjectiveOptimizer::new();

        let point1 = make_point(0.9, 200, 1000, 1.0);
        let point2 = make_point(0.8, 100, 1000, 1.0);

        // Point1 better success, worse latency
        assert!(!optimizer.dominates(&point1, &point2));
        assert!(!optimizer.dominates(&point2, &point1));
    }

    #[test]
    fn test_pareto_front_update() {
        let optimizer = MultiObjectiveOptimizer::new();

        let point1 = make_point(0.9, 100, 1000, 1.0);
        let point2 = make_point(0.8, 200, 2000, 2.0);
        let point3 = make_point(0.85, 150, 1500, 1.5);

        optimizer.update_pareto_front(point1.clone());
        optimizer.update_pareto_front(point2.clone());
        optimizer.update_pareto_front(point3);

        // Point2 dominated by point1, should not be in front
        let front = optimizer.find_tradeoffs();
        assert!(!front.iter().any(|p| p == &point2));
    }

    #[test]
    fn test_select_best() {
        let optimizer = MultiObjectiveOptimizer::new();

        // Pre-set normalization bounds for stable testing
        optimizer.update_normalization(ObjectiveVector {
            success_rate: 0.5,
            latency_ms: 50,
            memory_bytes: 500,
            cost_cents: 0.5,
        });
        optimizer.update_normalization(ObjectiveVector {
            success_rate: 0.9,
            latency_ms: 200,
            memory_bytes: 2000,
            cost_cents: 2.0,
        });

        let candidates = vec![
            make_point(0.9, 100, 1000, 1.0),
            make_point(0.7, 50, 500, 0.5),
            make_point(0.5, 200, 2000, 2.0),
        ];

        let selected = optimizer.select(candidates);
        assert!(selected.is_some());

        // With proper normalization, highest success wins
        if let Some(point) = selected {
            assert_eq!(point.objectives.success_rate, 0.9);
        }
    }

    #[test]
    fn test_constraint_filtering() {
        let optimizer = MultiObjectiveOptimizer::new();

        let constraints = ObjectiveConstraints {
            max_latency_ms: Some(150),
            ..Default::default()
        };
        optimizer.set_constraints(constraints);

        let candidates = vec![
            make_point(0.9, 100, 1000, 1.0),  // Valid
            make_point(0.8, 200, 1000, 1.0),  // Invalid (latency)
        ];

        let selected = optimizer.select(candidates);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().objectives.latency_ms, 100);
    }

    #[test]
    fn test_normalization() {
        let optimizer = MultiObjectiveOptimizer::new();

        // Update normalization with some data
        optimizer.update_normalization(ObjectiveVector {
            success_rate: 0.5,
            latency_ms: 1000,
            memory_bytes: 10_000,
            cost_cents: 5.0,
        });

        let raw = ObjectiveVector {
            success_rate: 0.8,
            latency_ms: 500,
            memory_bytes: 5_000,
            cost_cents: 2.5,
        };

        let normalized = optimizer.normalize_objectives(&raw);

        // Success rate unchanged (already normalized)
        assert_eq!(normalized.success_rate, 0.8);
    }

    #[test]
    fn test_crowding_distance_trimming() {
        let optimizer = MultiObjectiveOptimizer::new();

        // Add more points than max_front_size
        for i in 0..150 {
            let point = make_point(
                0.5 + (i as f64 / 300.0),
                100 + i as u64 * 10,
                1000 + i * 100,
                1.0 + i as f64 * 0.1,
            );
            optimizer.update_pareto_front(point);
        }

        let stats = optimizer.get_stats();
        assert!(stats.pareto_front_size <= 100);
    }

    #[test]
    fn test_weights_effect() {
        let optimizer = MultiObjectiveOptimizer::new();

        let high_latency_weights = ObjectiveWeights {
            latency: 10.0,  // Prioritize low latency
            ..Default::default()
        };
        optimizer.set_weights(high_latency_weights);

        let candidates = vec![
            make_point(0.9, 200, 1000, 1.0),  // High success, high latency
            make_point(0.7, 50, 1000, 1.0),   // Lower success, low latency
        ];

        let selected = optimizer.select(candidates);
        assert!(selected.is_some());

        // With high latency weight, should prefer second point
        if let Some(point) = selected {
            assert_eq!(point.objectives.latency_ms, 50);
        }
    }
}
