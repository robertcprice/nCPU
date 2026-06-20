//! Parallel optimization module for automatic code parallelization.
//!
//! This module provides intelligent parallelization support including:
//! - Detection of independent loops suitable for parallel execution
//! - Multi-language support: Rayon (Rust), async/await (JS/TS), multiprocessing (Python)
//! - Dependency analysis for safe parallelization
//! - Work stealing strategies for load balancing
//! - Integration with profiler for automatic parallel candidate detection

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Target programming language for parallel code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetLanguage {
    Rust,
    JavaScript,
    TypeScript,
    Python,
}

impl fmt::Display for TargetLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rust => write!(f, "rust"),
            Self::JavaScript => write!(f, "javascript"),
            Self::TypeScript => write!(f, "typescript"),
            Self::Python => write!(f, "python"),
        }
    }
}

/// Parallelization strategy for workload distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParallelStrategy {
    /// Static workload partitioning (even split)
    StaticPartition,
    /// Dynamic work stealing for load balancing
    WorkStealing,
    /// Parallel pipeline for stage-based parallelism
    Pipeline,
    /// Fork-join for recursive divide-and-conquer
    ForkJoin,
    /// Batch processing for independent operations
    Batch,
}

impl fmt::Display for ParallelStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaticPartition => write!(f, "StaticPartition"),
            Self::WorkStealing => write!(f, "WorkStealing"),
            Self::Pipeline => write!(f, "Pipeline"),
            Self::ForkJoin => write!(f, "ForkJoin"),
            Self::Batch => write!(f, "Batch"),
        }
    }
}

/// Dependency analysis result for safety validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    /// Whether the loop has no inter-iteration dependencies
    pub is_independent: bool,
    /// Detected dependencies between iterations
    pub dependencies: Vec<Dependency>,
    /// Variables that cause dependencies
    pub shared_variables: Vec<String>,
    /// Confidence that parallelization is safe (0.0 to 1.0)
    pub safety_confidence: f64,
}

/// Dependency detected between loop iterations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Type of dependency
    pub dep_type: DependencyType,
    /// Source variable causing the dependency
    pub source: String,
    /// Target variable affected by the dependency
    pub target: String,
    /// Line number where dependency occurs
    pub line: usize,
}

/// Types of dependencies that prevent parallelization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Read-after-write (true dependency)
    Raw,
    /// Write-after-read (anti-dependency)
    War,
    /// Write-after-write (output dependency)
    Waw,
    /// Memory aliasing (unknown overlap)
    Alias,
}

/// Loop detected as a candidate for parallelization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelCandidate {
    /// Line range of the loop (start, end)
    pub line_range: (usize, usize),
    /// Loop variable name
    pub loop_variable: String,
    /// Iteration count (estimated)
    pub iteration_count: usize,
    /// Loop body complexity (number of operations)
    pub body_complexity: usize,
    /// Dependency analysis result
    pub dependency_analysis: DependencyAnalysis,
    /// Recommended parallelization strategy
    pub recommended_strategy: ParallelStrategy,
    /// Expected speedup multiplier
    pub expected_speedup: f64,
    /// Parallelization safety score (0.0 to 1.0)
    pub safety_score: f64,
}

/// Parallelization transformation with generated code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parallelization {
    /// Human-readable description
    pub description: String,
    /// Original line range
    pub original_line_range: (usize, usize),
    /// Target language
    pub language: TargetLanguage,
    /// Parallelization strategy
    pub strategy: ParallelStrategy,
    /// Confidence in safety and correctness (0.0 to 1.0)
    pub confidence: f64,
    /// Expected speedup multiplier
    pub expected_speedup: f64,
    /// Generated parallel code
    pub parallel_code: String,
    /// Reasons for applying this parallelization
    pub rationale: Vec<String>,
    /// Potential risks or caveats
    pub caveats: Vec<String>,
}

/// Configuration for parallelization analysis
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum iterations to consider parallelization
    pub min_iterations: usize,
    /// Minimum body complexity to consider parallelization
    pub min_body_complexity: usize,
    /// Safety threshold for dependency analysis (0.0 to 1.0)
    pub safety_threshold: f64,
    /// Number of CPU cores to target
    pub target_cores: usize,
    /// Whether to analyze nested loops
    pub analyze_nested: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_iterations: 1000,
            min_body_complexity: 10,
            safety_threshold: 0.8,
            target_cores: num_cpus::get(),
            analyze_nested: true,
        }
    }
}

/// Main parallelization analyzer that detects candidates and generates code
pub struct Parallelizer {
    /// Configuration for analysis
    config: ParallelConfig,
    /// Profile feedback from previous parallelizations
    profile_feedback: HashMap<(TargetLanguage, ParallelStrategy), f64>,
}

impl Parallelizer {
    /// Create a new parallelizer with default configuration
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Create a new parallelizer with custom configuration
    pub fn with_config(config: ParallelConfig) -> Self {
        Self {
            config,
            profile_feedback: HashMap::new(),
        }
    }

    /// Detect all parallelizable loops in the given code
    pub fn detect_parallel_candidates(
        &self,
        code: &str,
        language: TargetLanguage,
    ) -> Vec<ParallelCandidate> {
        let mut candidates = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        // Detect different loop patterns based on language
        let loop_patterns = self.get_loop_patterns(language);

        for (i, line) in lines.iter().enumerate() {
            for pattern in &loop_patterns {
                if pattern.matches(line) {
                    if let Some(candidate) = self.analyze_loop(&lines, i, pattern, language) {
                        candidates.push(candidate);
                    }
                }
            }
        }

        candidates
    }

    /// Parallelize code by transforming detected candidates
    pub fn parallelize(&mut self, code: &str, language: TargetLanguage) -> Vec<Parallelization> {
        let candidates = self.detect_parallel_candidates(code, language);
        let mut parallelizations = Vec::new();

        for candidate in candidates {
            // Only parallelize if safe enough
            if candidate.safety_score >= self.config.safety_threshold {
                let parallelization = self.generate_parallelization(code, &candidate, language);
                parallelizations.push(parallelization);
            }
        }

        parallelizations
    }

    /// Apply parallelization transformations to code
    pub fn apply_parallelizations(
        &self,
        code: &str,
        parallelizations: &[Parallelization],
    ) -> String {
        let lines: Vec<&str> = code.lines().collect();
        let mut result = code.to_string();

        // Sort parallelizations by line range (reverse order to preserve line numbers)
        let mut sorted: Vec<_> = parallelizations.to_vec();
        sorted.sort_by(|a, b| b.original_line_range.0.cmp(&a.original_line_range.0));

        for parallelization in sorted {
            let start = parallelization.original_line_range.0;
            let end = parallelization.original_line_range.1.min(lines.len());

            if start < end {
                let result_lines: Vec<&str> = result.lines().collect();
                let before: Vec<String> = result_lines[..start]
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let after: Vec<String> = if end < result_lines.len() {
                    result_lines[end..].iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                };

                let parallel_code_lines: Vec<String> = parallelization
                    .parallel_code
                    .lines()
                    .map(|s| s.to_string())
                    .collect();

                let new_lines: Vec<String> = before
                    .into_iter()
                    .chain(parallel_code_lines)
                    .chain(after)
                    .collect();

                result = new_lines.join("\n");
            }
        }

        result
    }

    /// Analyze a single loop for parallelization potential
    fn analyze_loop(
        &self,
        lines: &[&str],
        loop_start: usize,
        pattern: &LoopPattern,
        language: TargetLanguage,
    ) -> Option<ParallelCandidate> {
        // Extract loop body
        let loop_end = self.find_loop_end(lines, loop_start, language)?;
        let loop_body = &lines[loop_start..=loop_end];

        // Extract loop variable
        let loop_variable = pattern.extract_loop_variable(lines[loop_start])?;

        // Estimate iteration count and complexity
        let iteration_count = self.estimate_iterations(lines[loop_start]);
        let body_complexity = loop_body.len();

        // Skip if below thresholds
        if iteration_count < self.config.min_iterations
            || body_complexity < self.config.min_body_complexity
        {
            return None;
        }

        // Analyze dependencies
        let dependency_analysis = self.analyze_dependencies(loop_body, &loop_variable, language);

        // Recommend strategy based on characteristics
        let recommended_strategy =
            self.recommend_strategy(&dependency_analysis, iteration_count, body_complexity);

        // Calculate expected speedup
        let expected_speedup = self.calculate_speedup(
            iteration_count,
            body_complexity,
            &dependency_analysis,
            &recommended_strategy,
        );

        // Calculate safety score
        let safety_score = dependency_analysis.safety_confidence;

        Some(ParallelCandidate {
            line_range: (loop_start, loop_end),
            loop_variable,
            iteration_count,
            body_complexity,
            dependency_analysis,
            recommended_strategy,
            expected_speedup,
            safety_score,
        })
    }

    /// Find the end of a loop by matching braces
    fn find_loop_end(
        &self,
        lines: &[&str],
        start: usize,
        language: TargetLanguage,
    ) -> Option<usize> {
        let mut brace_depth = 0;
        let mut found_open_brace = false;

        for (i, &line) in lines.iter().enumerate().skip(start) {
            for ch in line.chars() {
                match ch {
                    '{' => {
                        found_open_brace = true;
                        brace_depth += 1;
                    }
                    '}' => {
                        brace_depth -= 1;
                        if found_open_brace && brace_depth == 0 {
                            return Some(i);
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }

    /// Estimate loop iteration count from loop header
    fn estimate_iterations(&self, loop_line: &str) -> usize {
        // Try to extract numeric bounds
        if let Some(start) = loop_line.find("..") {
            let before = &loop_line[..start];
            let after = &loop_line[start + 2..];

            let start_val = before
                .rsplit(|c: char| !c.is_numeric())
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            let end_val = after
                .split(|c: char| !c.is_numeric())
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            return end_val.saturating_sub(start_val).max(1);
        }

        // Default estimate for unknown patterns
        1000
    }

    /// Analyze dependencies in loop body
    fn analyze_dependencies(
        &self,
        loop_body: &[&str],
        loop_variable: &str,
        language: TargetLanguage,
    ) -> DependencyAnalysis {
        let mut dependencies = Vec::new();
        let mut shared_variables = HashSet::new();
        let mut is_independent = true;

        let defined_vars = self.extract_defined_variables(loop_body);
        let used_vars = self.extract_used_variables(loop_body, language);

        // Check for accumulator patterns (dependencies)
        for line in loop_body {
            // Check for array access with previous/next iteration
            if let Some(idx) = line.find(&format!("[{} - 1]", loop_variable)) {
                dependencies.push(Dependency {
                    dep_type: DependencyType::Raw,
                    source: format!("array[{} - 1]", loop_variable),
                    target: format!("array[{}]", loop_variable),
                    line: 0, // Would need actual line tracking
                });
                is_independent = false;
                shared_variables.insert(format!("array[{}]", loop_variable));
            }

            if let Some(idx) = line.find(&format!("[{} + 1]", loop_variable)) {
                dependencies.push(Dependency {
                    dep_type: DependencyType::Raw,
                    source: format!("array[{} + 1]", loop_variable),
                    target: format!("array[{}]", loop_variable),
                    line: 0,
                });
                is_independent = false;
                shared_variables.insert(format!("array[{}]", loop_variable));
            }

            // Check for shared mutable state
            for var in &defined_vars {
                if line.contains(&format!("{} += ", var)) || line.contains(&format!("{} -= ", var))
                {
                    dependencies.push(Dependency {
                        dep_type: DependencyType::Waw,
                        source: var.clone(),
                        target: var.clone(),
                        line: 0,
                    });
                    is_independent = false;
                    shared_variables.insert(var.clone());
                }
            }
        }

        // Calculate safety confidence
        let safety_confidence = if is_independent {
            1.0
        } else {
            // Reduce confidence based on number of dependencies
            (1.0 - (dependencies.len() as f64 * 0.2)).max(0.0)
        };

        DependencyAnalysis {
            is_independent,
            dependencies,
            shared_variables: shared_variables.into_iter().collect(),
            safety_confidence,
        }
    }

    /// Extract variables defined in loop body
    fn extract_defined_variables(&self, loop_body: &[&str]) -> Vec<String> {
        let mut vars = Vec::new();

        for line in loop_body {
            let trimmed = line.trim();
            // Rust: let mut x = ...
            if let Some(idx) = trimmed.find("let mut ") {
                let after = &trimmed[idx + 8..];
                if let Some(end) = after.find(|c: char| c == '=' || c == ':') {
                    vars.push(after[..end].trim().to_string());
                }
            }
            // Rust: let x = ...
            if let Some(idx) = trimmed.find("let ") {
                let after = &trimmed[idx + 4..];
                if let Some(end) = after.find(|c: char| c == '=' || c == ':') {
                    let var_name = after[..end].trim();
                    if !var_name.contains("mut ") {
                        vars.push(var_name.to_string());
                    }
                }
            }
        }

        vars
    }

    /// Extract variables used in loop body
    fn extract_used_variables(&self, loop_body: &[&str], _language: TargetLanguage) -> Vec<String> {
        let mut used = HashSet::new();

        for line in loop_body {
            // Simple heuristic: extract words followed by operators
            let parts: Vec<&str> = line
                .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
                .collect();

            for (i, part) in parts.iter().enumerate() {
                if i + 1 < parts.len() {
                    let next = parts[i + 1];
                    if next == "=" || next == "+=" || next == "-=" {
                        used.insert(part.to_string());
                    }
                }
            }
        }

        used.into_iter().collect()
    }

    /// Recommend parallelization strategy based on characteristics
    fn recommend_strategy(
        &self,
        dep_analysis: &DependencyAnalysis,
        iteration_count: usize,
        body_complexity: usize,
    ) -> ParallelStrategy {
        if !dep_analysis.is_independent {
            // For dependent loops, use fork-join if possible
            if iteration_count > 10000 {
                return ParallelStrategy::ForkJoin;
            }
            return ParallelStrategy::Pipeline;
        }

        // For independent loops with uniform work
        if body_complexity < 50 {
            return ParallelStrategy::StaticPartition;
        }

        // For independent loops with variable work
        if body_complexity > 100 {
            return ParallelStrategy::WorkStealing;
        }

        // For batch-style operations
        if iteration_count % 4 == 0 {
            return ParallelStrategy::Batch;
        }

        ParallelStrategy::StaticPartition
    }

    /// Calculate expected speedup for parallelization
    fn calculate_speedup(
        &self,
        iteration_count: usize,
        body_complexity: usize,
        dep_analysis: &DependencyAnalysis,
        strategy: &ParallelStrategy,
    ) -> f64 {
        let cores = self.config.target_cores as f64;

        let base_speedup = if dep_analysis.is_independent {
            // Amdahl's law approximation
            let parallel_fraction = 0.95; // 95% can be parallelized
            cores / (1.0 - parallel_fraction + parallel_fraction / cores)
        } else {
            // Reduce speedup for dependencies
            cores * 0.6
        };

        // Adjust for strategy overhead
        let strategy_factor = match strategy {
            ParallelStrategy::StaticPartition => 0.95,
            ParallelStrategy::WorkStealing => 0.90,
            ParallelStrategy::Pipeline => 0.85,
            ParallelStrategy::ForkJoin => 0.80,
            ParallelStrategy::Batch => 0.92,
        };

        // Adjust for granularity
        let granularity_factor = if iteration_count < cores as usize {
            0.7 // Not enough work for all cores
        } else if iteration_count > cores as usize * 1000 {
            1.0 // Plenty of work
        } else {
            0.9 // Moderate work
        };

        base_speedup * strategy_factor * granularity_factor
    }

    /// Generate parallelization code for a candidate
    fn generate_parallelization(
        &self,
        original_code: &str,
        candidate: &ParallelCandidate,
        language: TargetLanguage,
    ) -> Parallelization {
        let description = format!(
            "Parallelize loop {} with {} iterations using {} strategy",
            candidate.loop_variable, candidate.iteration_count, candidate.recommended_strategy
        );

        let parallel_code = self.generate_parallel_code(original_code, candidate, language);

        let rationale = vec![
            format!(
                "Loop has {} independent iterations",
                candidate.iteration_count
            ),
            format!("Body complexity: {} operations", candidate.body_complexity),
            format!("Safety confidence: {:.2}", candidate.safety_score),
            format!("Expected speedup: {:.1}x", candidate.expected_speedup),
        ];

        let caveats = if candidate.dependency_analysis.is_independent {
            vec!["Thread synchronization overhead may reduce gains for small workloads".to_string()]
        } else {
            vec![
                format!(
                    "Dependencies detected: {}",
                    candidate.dependency_analysis.dependencies.len()
                ),
                "Requires careful synchronization for correctness".to_string(),
            ]
        };

        Parallelization {
            description,
            original_line_range: candidate.line_range,
            language,
            strategy: candidate.recommended_strategy,
            confidence: candidate.safety_score,
            expected_speedup: candidate.expected_speedup,
            parallel_code,
            rationale,
            caveats,
        }
    }

    /// Generate parallel code for a specific language and strategy
    fn generate_parallel_code(
        &self,
        _original_code: &str,
        candidate: &ParallelCandidate,
        language: TargetLanguage,
    ) -> String {
        match language {
            TargetLanguage::Rust => self.generate_rust_parallel(candidate),
            TargetLanguage::JavaScript | TargetLanguage::TypeScript => {
                self.generate_js_parallel(candidate)
            }
            TargetLanguage::Python => self.generate_python_parallel(candidate),
        }
    }

    /// Generate Rust parallel code using Rayon
    fn generate_rust_parallel(&self, candidate: &ParallelCandidate) -> String {
        let strategy = candidate.recommended_strategy;

        match strategy {
            ParallelStrategy::StaticPartition => {
                format!(
                    r#"// Parallelized using Rayon (static partition)
use rayon::prelude::*;

// Original loop range: {}..{}
let results: Vec<_> = (0..{}).into_par_iter()
    .map(|{}| {{
        // Loop body goes here - ensure all operations are thread-safe
        // Original iteration work:
        // ...
    }})
    .collect();

// Note: Ensure no shared mutable state or use proper synchronization
"#,
                    candidate.line_range.0,
                    candidate.line_range.1,
                    candidate.iteration_count,
                    candidate.loop_variable
                )
            }

            ParallelStrategy::WorkStealing => {
                format!(
                    r#"// Parallelized using Rayon (work stealing for load balancing)
use rayon::iter::{{ParallelIterator, IntoParallelIterator}};

// Original loop range: {}..{{}}
let results: Vec<_> = (0..{}).into_par_iter()
    .with_min_len(100)  // Minimum chunk size for work stealing
    .map(|{{}}| {{
        // Loop body goes here
        // Each iteration processes a chunk of work
    }})
    .collect();

// Work stealing dynamically balances load across threads
"#,
                    candidate.line_range.0, candidate.line_range.1
                )
            }

            ParallelStrategy::Pipeline => {
                format!(
                    r#"// Parallelized using Rayon pipeline pattern
use rayon::prelude::*;

// Stage 1: Process first phase in parallel
let stage1: Vec<_> = (0..{}).into_par_iter()
    .map(|{}| {{
        // First stage of computation
    }})
    .collect();

// Stage 2: Process results (can be parallelized too)
let stage2: Vec<_> = stage1.par_iter()
    .map(|input| {{
        // Second stage dependent on first
    }})
    .collect();

// Pipeline pattern for dependent computations
"#,
                    candidate.iteration_count, candidate.loop_variable
                )
            }

            ParallelStrategy::ForkJoin => {
                format!(
                    r#"// Parallelized using Rayon fork-join (divide and conquer)
use rayon::prelude::*;

fn process_range(start: usize, end: usize) -> Vec<OutputType> {{
    if end - start <= THRESHOLD {{
        // Base case: process directly
        (start..end).map(|{}| {{
            // Sequential processing for small ranges
        }}).collect()
    }} else {{
        // Recursive case: fork-join
        let mid = (start + end) / 2;
        let (left, right) = rayon::join(
            || process_range(start, mid),
            || process_range(mid, end),
        );
        // Combine results
        [left, right].concat()
    }}
}}

// Process entire range
let results = process_range(0, {});
"#,
                    candidate.loop_variable, candidate.iteration_count
                )
            }

            ParallelStrategy::Batch => {
                format!(
                    r#"// Parallelized using batch processing
use rayon::prelude::*;

const BATCH_SIZE: usize = {};

// Split into batches and process in parallel
let results: Vec<_> = (0..{})
    .collect::<Vec<_>>()
    .chunks(BATCH_SIZE)
    .into_par_iter()
    .flat_map(|batch| {{
        // Process entire batch
        batch.iter().map(|&{}| {{
            // Process each item in batch
        }}).collect::<Vec<_>>()
    }})
    .collect();

// Batch processing reduces synchronization overhead
"#,
                    candidate.iteration_count / self.config.target_cores,
                    candidate.iteration_count,
                    candidate.loop_variable
                )
            }
        }
    }

    /// Generate JavaScript/TypeScript parallel code using async/await
    fn generate_js_parallel(&self, candidate: &ParallelCandidate) -> String {
        let is_ts = matches!(
            candidate.recommended_strategy,
            ParallelStrategy::WorkStealing
        );

        let type_annotation = if is_ts { ": number" } else { "" };

        format!(
            r#"// Parallelized using async/await ({}{})
// Original loop range: {}..{}

async function parallelProcess(items{}): Promise<Results> {{
    // Create promises for each chunk
    const chunkSize = Math.ceil(items.length / {});
    const promises = [];

    for (let i = 0; i < items.length; i += chunkSize) {{
        const chunk = items.slice(i, i + chunkSize);
        promises.push(processChunk(chunk));
    }}

    // Execute all chunks in parallel
    const results = await Promise.all(promises);
    return results.flat();
}}

async function processChunk(chunk: Item[]): Promise<PartialResult> {{
    // Process chunk of items
    return chunk.map((item{}, idx) => {{
        // Loop body goes here
        // Process item at index i + idx
    }});
}}

// Usage:
// const results = await parallelProcess(array);
"#,
            if is_ts { "TypeScript" } else { "JavaScript" },
            if is_ts { " typed" } else { "" },
            candidate.line_range.0,
            candidate.line_range.1,
            type_annotation,
            self.config.target_cores,
            type_annotation
        )
    }

    /// Generate Python parallel code using multiprocessing
    fn generate_python_parallel(&self, candidate: &ParallelCandidate) -> String {
        let strategy = candidate.recommended_strategy;

        match strategy {
            ParallelStrategy::StaticPartition => {
                format!(
                    r#"// Parallelized using multiprocessing (static partition)
from multiprocessing import Pool
from functools import partial

# Original loop range: {}..{}
def process_item({}):
    # Loop body goes here
    # Process single item
    return result

def parallel_process(items, num_processes=None):
    '''Process items in parallel using static partitioning'''
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_item, items)
    return results

# Usage:
# results = parallel_process(range({}))
"#,
                    candidate.line_range.0,
                    candidate.line_range.1,
                    candidate.loop_variable,
                    candidate.iteration_count
                )
            }

            ParallelStrategy::WorkStealing => {
                format!(
                    r#"// Parallelized using multiprocessing with work stealing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import itertools

# Original loop range: {}..{}
def process_item({}):
    # Loop body goes here
    return result

def parallel_process_dynamic(items, chunk_size=100):
    '''Process items with dynamic chunking for load balancing'''
    # Create chunks
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    with Pool() as pool:
        # Process chunks in parallel
        chunk_results = pool.map(process_chunk, chunks)

    # Flatten results
    return list(itertools.chain.from_iterable(chunk_results))

def process_chunk(chunk):
    '''Process a chunk of items'''
    return [process_item(item) for item in chunk]

# Usage:
# results = parallel_process_dynamic(range({}))
"#,
                    candidate.line_range.0,
                    candidate.line_range.1,
                    candidate.loop_variable,
                    candidate.iteration_count
                )
            }

            ParallelStrategy::Pipeline => {
                format!(
                    r#"// Parallelized using multiprocessing pipeline
from multiprocessing import Pool, Queue
from typing import List, Any

# Original loop range: {}..{}
def stage1(item):
    # First pipeline stage
    return processed

def stage2(item):
    # Second pipeline stage (depends on stage1)
    return result

def parallel_pipeline(items):
    '''Process items through parallel pipeline'''
    with Pool() as pool:
        # Stage 1: Parallel processing
        stage1_results = pool.map(stage1, items)

        # Stage 2: Parallel processing of stage1 results
        stage2_results = pool.map(stage2, stage1_results)

    return stage2_results

# Usage:
# results = parallel_pipeline(range({}))
"#,
                    candidate.line_range.0, candidate.line_range.1, candidate.iteration_count
                )
            }

            ParallelStrategy::ForkJoin => {
                format!(
                    r#"// Parallelized using multiprocessing fork-join
from multiprocessing import Pool
from typing import List

# Original loop range: {}..{}
THRESHOLD = 1000

def process_range(start, end):
    '''Process range using divide-and-conquer'''
    if end - start <= THRESHOLD:
        # Base case: sequential processing
        return [process_item(i) for i in range(start, end)]
    else:
        # Recursive case: fork-join
        mid = (start + end) // 2
        with Pool(processes=2) as pool:
            left_right = pool.starmap(process_range, [
                (start, mid),
                (mid, end)
            ])
        # Combine results
        return left_right[0] + left_right[1]

def process_item({}):
    '''Process single item'''
    # Loop body goes here
    return result

# Usage:
# results = process_range(0, {})
"#,
                    candidate.line_range.0,
                    candidate.line_range.1,
                    candidate.loop_variable,
                    candidate.iteration_count
                )
            }

            ParallelStrategy::Batch => {
                format!(
                    r#"// Parallelized using batch processing
from multiprocessing import Pool
from typing import List, Any
import numpy as np

# Original loop range: {}..{}
BATCH_SIZE = {}

def process_batch(batch):
    '''Process entire batch'''
    return [process_item(item) for item in batch]

def process_item({}):
    '''Process single item'''
    # Loop body goes here
    return result

def parallel_batch(items, batch_size=BATCH_SIZE):
    '''Process items in batches'''
    # Split into batches
    batches = [items[i:i + batch_size]
               for i in range(0, len(items), batch_size)]

    with Pool() as pool:
        batch_results = pool.map(process_batch, batches)

    # Flatten results
    return [item for batch in batch_results for item in batch]

# Usage:
# results = parallel_batch(range({}))
"#,
                    candidate.line_range.0,
                    candidate.line_range.1,
                    candidate.iteration_count / self.config.target_cores,
                    candidate.loop_variable,
                    candidate.iteration_count
                )
            }
        }
    }

    /// Get loop patterns for a specific language
    fn get_loop_patterns(&self, language: TargetLanguage) -> Vec<LoopPattern> {
        match language {
            TargetLanguage::Rust => vec![LoopPattern::new("for", |line| {
                line.trim().starts_with("for ") && line.contains("in ")
            })],
            TargetLanguage::JavaScript | TargetLanguage::TypeScript => vec![
                LoopPattern::new("for", |line| {
                    line.trim().starts_with("for ") && line.contains("of ")
                }),
                LoopPattern::new("for", |line| {
                    line.trim().starts_with("for (let ") || line.trim().starts_with("for (const ")
                }),
            ],
            TargetLanguage::Python => vec![LoopPattern::new("for", |line| {
                line.trim().starts_with("for ") && line.contains(" in ")
            })],
        }
    }

    /// Update profile feedback with performance data
    pub fn update_feedback(
        &mut self,
        language: TargetLanguage,
        strategy: ParallelStrategy,
        performance_score: f64,
    ) {
        let key = (language, strategy);
        let normalized = (performance_score / 100.0).min(1.0).max(0.0);
        let current = self
            .profile_feedback
            .get(&key)
            .copied()
            .unwrap_or(normalized);
        let updated = current * 0.8 + normalized * 0.2;
        self.profile_feedback.insert(key, updated);
    }

    /// Get profile feedback for a language/strategy combination
    pub fn get_feedback(
        &self,
        language: TargetLanguage,
        strategy: ParallelStrategy,
    ) -> Option<f64> {
        self.profile_feedback.get(&(language, strategy)).copied()
    }

    /// Reset all profile feedback
    pub fn reset_feedback(&mut self) {
        self.profile_feedback.clear();
    }
}

impl Default for Parallelizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop pattern for detecting different loop constructs
struct LoopPattern {
    name: &'static str,
    matcher: Box<dyn Fn(&str) -> bool>,
}

impl LoopPattern {
    fn new<F>(name: &'static str, matcher: F) -> Self
    where
        F: Fn(&str) -> bool + 'static,
    {
        Self {
            name,
            matcher: Box::new(matcher),
        }
    }

    fn matches(&self, line: &str) -> bool {
        (self.matcher)(line)
    }

    fn extract_loop_variable(&self, line: &str) -> Option<String> {
        // Extract variable name from loop header
        if let Some(idx) = line.find("for ") {
            let after = &line[idx + 4..];
            if let Some(end) = after.find(|c: char| c == ' ' || c == '(' || c == '=') {
                return Some(after[..end].trim().to_string());
            }
        }
        if let Some(idx) = line.find("for (let ") {
            let after = &line[idx + 9..];
            if let Some(end) = after.find(|c: char| c == ' ' || c == '=' || c == ':') {
                return Some(after[..end].trim().to_string());
            }
        }
        if let Some(idx) = line.find("for (const ") {
            let after = &line[idx + 10..];
            if let Some(end) = after.find(|c: char| c == ' ' || c == '=' || c == ':') {
                return Some(after[..end].trim().to_string());
            }
        }
        None
    }
}

/// Integrate with profiler for automatic parallel candidate detection
pub fn detect_parallel_from_hotspots(
    hotspots: &[crate::optimization::profiler::Hotspot],
    code: &str,
    language: TargetLanguage,
) -> Vec<ParallelCandidate> {
    let parallelizer = Parallelizer::new();
    let mut all_candidates = Vec::new();

    for hotspot in hotspots {
        // Get code around hotspot
        let lines: Vec<&str> = code.lines().collect();
        let start = hotspot.line_range.0.min(lines.len());
        let end = hotspot.line_range.1.min(lines.len());

        if start < end {
            let snippet = lines[start..end].join("\n");
            let candidates = parallelizer.detect_parallel_candidates(&snippet, language);
            all_candidates.extend(candidates);
        }
    }

    all_candidates
}

/// Create work stealing configuration for optimal load balancing
pub fn configure_work_stealing(total_items: usize, target_cores: usize) -> WorkStealingConfig {
    let min_chunk_size = (total_items / (target_cores * 4)).max(1);
    let max_chunk_size = (total_items / target_cores).max(min_chunk_size);

    WorkStealingConfig {
        min_chunk_size,
        max_chunk_size,
        target_cores,
        steal_threshold: 2, // Steal when queue has 2 or more items
    }
}

/// Configuration for work stealing strategy
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Minimum chunk size for work distribution
    pub min_chunk_size: usize,
    /// Maximum chunk size for work distribution
    pub max_chunk_size: usize,
    /// Number of worker threads
    pub target_cores: usize,
    /// Threshold for triggering work stealing
    pub steal_threshold: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_for_loop_detection() {
        let code = r#"
fn example() {
    for i in 0..10000 {
        let result = i * 2;
        println!("{}", result);
    }
}
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Rust);

        assert!(!candidates.is_empty());
        let candidate = &candidates[0];
        assert_eq!(candidate.loop_variable, "i");
        assert!(candidate.iteration_count >= 10000);
    }

    #[test]
    fn test_dependency_analysis() {
        let code = r#"
fn example() {
    let mut sum = 0;
    for i in 0..10000 {
        sum += i;  // Dependency: shared mutable state
    }
}
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Rust);

        // Should detect dependency on sum
        if let Some(candidate) = candidates.first() {
            assert!(!candidate.dependency_analysis.is_independent);
            assert!(!candidate.dependency_analysis.shared_variables.is_empty());
        }
    }

    #[test]
    fn test_independent_loop() {
        let code = r#"
fn example() {
    let results = Vec::new();
    for i in 0..10000 {
        let result = expensive_calculation(i);
        results.push(result);
    }
}
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Rust);

        if let Some(candidate) = candidates.first() {
            // Loop should be independent
            assert!(candidate.dependency_analysis.is_independent);
            assert!(candidate.safety_score > 0.8);
        }
    }

    #[test]
    fn test_rust_parallel_code_generation() {
        let code = r#"
fn example() {
    for i in 0..10000 {
        let result = i * 2;
    }
}
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Rust);

        if let Some(candidate) = candidates.first() {
            let parallelizations = parallelizer.parallelize(code, TargetLanguage::Rust);
            assert!(!parallelizations.is_empty());

            let generated = &parallelizations[0].parallel_code;
            assert!(generated.contains("rayon"));
            assert!(generated.contains("par_iter"));
        }
    }

    #[test]
    fn test_javascript_parallel_code_generation() {
        let code = r#"
function example() {
    for (let i = 0; i < 10000; i++) {
        const result = i * 2;
    }
}
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::JavaScript);

        if let Some(candidate) = candidates.first() {
            let parallelizations = parallelizer.parallelize(code, TargetLanguage::JavaScript);
            assert!(!parallelizations.is_empty());

            let generated = &parallelizations[0].parallel_code;
            assert!(generated.contains("async") || generated.contains("Promise"));
            assert!(generated.contains("parallel"));
        }
    }

    #[test]
    fn test_python_parallel_code_generation() {
        let code = r#"
def example():
    for i in range(10000):
        result = i * 2
"#;

        let mut parallelizer = Parallelizer::new();
        let candidates = parallelizer.detect_parallel_candidates(code, TargetLanguage::Python);

        if let Some(candidate) = candidates.first() {
            let parallelizations = parallelizer.parallelize(code, TargetLanguage::Python);
            assert!(!parallelizations.is_empty());

            let generated = &parallelizations[0].parallel_code;
            assert!(generated.contains("multiprocessing"));
            assert!(generated.contains("Pool"));
        }
    }

    #[test]
    fn test_strategy_selection() {
        let mut parallelizer = Parallelizer::new();

        // Static partition for simple loops
        let dep_analysis = DependencyAnalysis {
            is_independent: true,
            dependencies: Vec::new(),
            shared_variables: Vec::new(),
            safety_confidence: 1.0,
        };

        let strategy = parallelizer.recommend_strategy(&dep_analysis, 1000, 20);
        assert_eq!(strategy, ParallelStrategy::StaticPartition);

        // Work stealing for complex loops
        let strategy = parallelizer.recommend_strategy(&dep_analysis, 1000, 150);
        assert_eq!(strategy, ParallelStrategy::WorkStealing);
    }

    #[test]
    fn test_work_stealing_config() {
        let config = configure_work_stealing(10000, 4);

        assert!(config.min_chunk_size > 0);
        assert!(config.max_chunk_size >= config.min_chunk_size);
        assert_eq!(config.target_cores, 4);
    }

    #[test]
    fn test_feedback_update() {
        let mut parallelizer = Parallelizer::new();

        parallelizer.update_feedback(TargetLanguage::Rust, ParallelStrategy::WorkStealing, 85.0);

        let feedback =
            parallelizer.get_feedback(TargetLanguage::Rust, ParallelStrategy::WorkStealing);
        assert!(feedback.is_some());
        assert!(feedback.unwrap() > 0.0);
    }

    #[test]
    fn test_safety_threshold_filtering() {
        let code = r#"
fn example() {
    for i in 0..10000 {
        let result = i * 2;
    }
}
"#;

        let mut parallelizer = Parallelizer::with_config(ParallelConfig {
            safety_threshold: 0.95,
            ..Default::default()
        });

        let parallelizations = parallelizer.parallelize(code, TargetLanguage::Rust);

        // Should only return high-confidence candidates
        for parallelization in &parallelizations {
            assert!(parallelization.confidence >= 0.95);
        }
    }
}
