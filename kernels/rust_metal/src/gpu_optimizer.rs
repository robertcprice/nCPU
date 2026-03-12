/*
 * GPU-Accelerated Optimization Module
 *
 * Genetic algorithms and neural scheduling running on CPU/Rust side
 * for integration with the self-optimizing engine.
 */

use std::collections::HashMap;
use pyo3::prelude::*;

/// Configuration for genetic algorithm
#[pyclass]
#[derive(Debug, Clone)]
pub struct GaConfig {
    #[pyo3(get, set)]
    pub population_size: u32,
    #[pyo3(get, set)]
    pub elite_size: u32,
    #[pyo3(get, set)]
    pub mutation_rate: f32,
    #[pyo3(get, set)]
    pub crossover_rate: f32,
    #[pyo3(get, set)]
    pub generations: u32,
    #[pyo3(get, set)]
    pub tournament_size: u32,
}

impl Default for GaConfig {
    fn default() -> Self {
        GaConfig {
            population_size: 20,
            elite_size: 4,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            generations: 50,
            tournament_size: 3,
        }
    }
}

/// An individual in the genetic algorithm population
#[derive(Debug, Clone)]
struct GaIndividual {
    weights: Vec<f32>,
    fitness: f32,
}

/// GPU-accelerated Genetic Algorithm Optimizer
#[pyclass]
pub struct GpuGeneticOptimizer {
    config: GaConfig,
    population: Vec<GaIndividual>,
    best_weights: Vec<f32>,
    best_fitness: f32,
    history: Vec<f32>,
}

#[pymethods]
impl GpuGeneticOptimizer {
    #[new]
    fn new(config: Option<GaConfig>) -> Self {
        let config = config.unwrap_or_default();
        GpuGeneticOptimizer {
            config,
            population: Vec::new(),
            best_weights: Vec::new(),
            best_fitness: f32::NEG_INFINITY,
            history: Vec::new(),
        }
    }

    fn initialize(&mut self, weight_dim: usize) {
        self.population = Vec::new();

        for _ in 0..self.config.population_size {
            let weights: Vec<f32> = (0..weight_dim)
                .map(|_| Self::rand_f32() * 0.2 - 0.1)
                .collect();

            self.population.push(GaIndividual {
                weights,
                fitness: 0.0,
            });
        }
    }

    fn run(&mut self, max_generations: Option<u32>) -> GaResult {
        let generations = max_generations.unwrap_or(self.config.generations);

        if self.population.is_empty() {
            return GaResult {
                best_weights: vec![],
                best_fitness: 0.0,
                history: vec![],
                generations: 0,
            };
        }

        // Initialize best with first individual
        self.best_weights = self.population[0].weights.clone();
        self.best_fitness = f32::NEG_INFINITY;
        self.history = Vec::new();

        for _ in 0..generations {
            // Sort by fitness descending
            self.population.sort_by(|a, b| {
                b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Track best
            if self.population[0].fitness > self.best_fitness {
                self.best_fitness = self.population[0].fitness;
                self.best_weights = self.population[0].weights.clone();
            }
            self.history.push(self.best_fitness);

            // Elitism: keep best
            let elite_count = (self.config.elite_size as usize).min(self.population.len());
            let mut new_pop: Vec<GaIndividual> = self.population[..elite_count]
                .iter()
                .cloned()
                .collect();

            // Generate rest
            while new_pop.len() < self.population.len() {
                let parent1 = self.tournament_select();
                let parent2 = self.tournament_select();
                let child_weights = self.crossover(&parent1.weights, &parent2.weights);
                let mut mutated = child_weights;
                self.mutate(&mut mutated);
                new_pop.push(GaIndividual {
                    weights: mutated,
                    fitness: 0.0,
                });
            }

            self.population = new_pop;
        }

        GaResult {
            best_weights: self.best_weights.clone(),
            best_fitness: self.best_fitness,
            history: self.history.clone(),
            generations,
        }
    }

    fn set_fitness(&mut self, individual_idx: usize, fitness: f32) {
        if individual_idx < self.population.len() {
            self.population[individual_idx].fitness = fitness;
        }
    }

    fn get_best_weights(&self) -> Vec<f32> {
        self.best_weights.clone()
    }

    fn get_best_fitness(&self) -> f32 {
        self.best_fitness
    }
}

impl GpuGeneticOptimizer {
    fn tournament_select(&self) -> &GaIndividual {
        let tournament_size = (self.config.tournament_size as usize).min(self.population.len());
        let start = Self::rand_usize() % self.population.len();

        let mut best: Option<&GaIndividual> = None;

        for i in 0..tournament_size {
            let idx = (start + i) % self.population.len();
            let candidate = &self.population[idx];

            if best.is_none() || candidate.fitness > best.as_ref().unwrap().fitness {
                best = Some(candidate);
            }
        }

        best.unwrap()
    }

    fn crossover(&self, parent1: &[f32], parent2: &[f32]) -> Vec<f32> {
        if Self::rand_f32() > self.config.crossover_rate {
            return parent1.to_vec();
        }

        let alpha = 0.5;
        let mut child = vec![0.0; parent1.len()];

        for i in 0..child.len() {
            let min_w = parent1[i].min(parent2[i]);
            let max_w = parent1[i].max(parent2[i]);
            let range = max_w - min_w;

            child[i] = min_w - alpha * range + Self::rand_f32() * (max_w - min_w + 2.0 * alpha * range);
        }

        child
    }

    fn mutate(&self, weights: &mut Vec<f32>) {
        for w in weights.iter_mut() {
            if Self::rand_f32() < self.config.mutation_rate {
                *w += Self::rand_gaussian() * 0.1;
            }
        }
    }

    fn rand_f32() -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        (nanos as f32) / (u32::MAX as f32)
    }

    fn rand_usize() -> usize {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        nanos as usize
    }

    fn rand_gaussian() -> f32 {
        let u1 = Self::rand_f32().max(1e-10);
        let u2 = Self::rand_f32();
        (2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Result from genetic algorithm
#[pyclass]
#[derive(Debug, Clone)]
pub struct GaResult {
    #[pyo3(get)]
    pub best_weights: Vec<f32>,
    #[pyo3(get)]
    pub best_fitness: f32,
    #[pyo3(get)]
    pub history: Vec<f32>,
    #[pyo3(get)]
    pub generations: u32,
}

/// GPU-accelerated Neural Scheduler with Learning
#[pyclass]
pub struct GpuNeuralScheduler {
    weights: HashMap<String, f32>,
}

#[pymethods]
impl GpuNeuralScheduler {
    #[new]
    fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("priority".to_string(), 1.0);
        weights.insert("cpu_burst".to_string(), 0.8);
        weights.insert("io_wait".to_string(), 0.5);
        weights.insert("age".to_string(), 0.3);
        weights.insert("cache_locality".to_string(), 0.2);

        GpuNeuralScheduler { weights }
    }

    fn compute_score(&self, priority: f32, cpu_burst: f32, io_wait: f32, age: f32, cache_locality: f32) -> f32 {
        let mut score = 0.0;

        score += self.weights.get("priority").unwrap_or(&1.0) * priority;
        score += self.weights.get("cpu_burst").unwrap_or(&0.8) * cpu_burst;
        score += self.weights.get("io_wait").unwrap_or(&0.5) * io_wait;
        score += self.weights.get("age").unwrap_or(&0.3) * age;
        score += self.weights.get("cache_locality").unwrap_or(&0.2) * cache_locality;

        score
    }

    fn get_weights(&self) -> HashMap<String, f32> {
        self.weights.clone()
    }

    fn set_weight(&mut self, key: String, value: f32) {
        self.weights.insert(key, value);
    }
}

impl Default for GpuNeuralScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark function exposed to Python
#[pyfunction]
pub fn benchmark_genetic(weight_dim: usize, generations: u32) -> GaResult {
    let mut ga = GpuGeneticOptimizer::new(Some(GaConfig {
        population_size: 20,
        elite_size: 4,
        mutation_rate: 0.1,
        crossover_rate: 0.7,
        generations,
        tournament_size: 3,
    }));

    ga.initialize(weight_dim);

    // Initialize with random fitness for benchmark
    for i in 0..ga.population.len() {
        let fitness = GpuGeneticOptimizer::rand_gaussian();
        ga.population[i].fitness = fitness;
    }

    ga.run(Some(generations))
}

/// Register module with Python
pub fn register_gpu_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GaConfig>()?;
    m.add_class::<GpuGeneticOptimizer>()?;
    m.add_class::<GaResult>()?;
    m.add_class::<GpuNeuralScheduler>()?;
    m.add_function(wrap_pyfunction!(benchmark_genetic, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genetic_initialization() {
        let mut ga = GpuGeneticOptimizer::new(None);
        ga.initialize(10);
        assert!(ga.population.len() > 0);
    }

    #[test]
    fn test_neural_scheduler() {
        let scheduler = GpuNeuralScheduler::new();
        let mut state = HashMap::new();
        state.insert("priority".to_string(), 1.0);

        let score = scheduler.compute_score(&state);
        assert!(score > 0.0);
    }
}
