//! Incremental synthesis for modular programs
//!
//! Synthesizes modules incrementally, using results from
//! dependencies to inform subsequent synthesis.

use crate::benchmark::Problem;
use crate::solver::SolveResult;
pub use super::decomposition::ModuleSpec;
use std::collections::HashMap;

/// Synthesis cache for reuse
#[derive(Debug, Clone)]
pub struct SynthCache {
    /// Cache of synthesized modules
    modules: HashMap<String, SynthesizedModule>,
    /// Cache of solved problems
    problems: HashMap<String, SolveResult>,
}

/// Synthesized module
#[derive(Debug, Clone)]
pub struct SynthesizedModule {
    pub name: String,
    pub code: String,
    pub exports: Vec<String>,
}

impl SynthCache {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            problems: HashMap::new(),
        }
    }

    /// Get cached module
    pub fn get_module(&self, name: &str) -> Option<&SynthesizedModule> {
        self.modules.get(name)
    }

    /// Cache a module
    pub fn cache_module(&mut self, module: SynthesizedModule) {
        self.modules.insert(module.name.clone(), module);
    }

    /// Get cached problem result
    pub fn get_problem(&self, key: &str) -> Option<&SolveResult> {
        self.problems.get(key)
    }

    /// Cache a problem result
    pub fn cache_problem(&mut self, key: String, result: SolveResult) {
        self.problems.insert(key, result);
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.modules.clear();
        self.problems.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            module_count: self.modules.len(),
            problem_count: self.problems.len(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub module_count: usize,
    pub problem_count: usize,
}

impl Default for SynthCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Synthesize modules incrementally
pub fn synthesize_incremental(
    modules: Vec<ModuleSpec>,
    cache: &mut SynthCache,
) -> Result<IncrementalResult, String> {
    let mut synthesized = Vec::new();
    let mut failed = Vec::new();

    // Sort modules by dependency (leaf modules first)
    let sorted = sort_by_dependencies(&modules);

    for module in sorted {
        // Check if already cached
        if let Some(cached) = cache.get_module(&module.name) {
            synthesized.push(cached.clone());
            continue;
        }

        // Synthesize this module
        match synthesize_module(&module, cache) {
            Ok(result) => {
                let synth_module = SynthesizedModule {
                    name: module.name.clone(),
                    code: result.code.clone(),
                    exports: extract_exports(&result),
                };
                cache.cache_module(synth_module.clone());
                synthesized.push(synth_module);
            }
            Err(e) => {
                failed.push((module.name.clone(), e));
            }
        }
    }

    Ok(IncrementalResult {
        modules: synthesized,
        failed,
    })
}

/// Result of incremental synthesis
#[derive(Debug, Clone)]
pub struct IncrementalResult {
    pub modules: Vec<SynthesizedModule>,
    pub failed: Vec<(String, String)>,
}

/// Sort modules by dependencies (leaf modules first)
fn sort_by_dependencies(modules: &[ModuleSpec]) -> Vec<&ModuleSpec> {
    let mut sorted = Vec::new();
    let mut remaining: Vec<&ModuleSpec> = modules.iter().collect();

    while !remaining.is_empty() {
        // Find modules with no unresolved dependencies
        let mut ready = Vec::new();
        let mut still_waiting = Vec::new();

        for module in remaining {
            let deps_resolved = module.dependencies.iter().all(|dep| {
                sorted.iter().any(|s: &&ModuleSpec| s.name == *dep)
            });

            if deps_resolved {
                ready.push(module);
            } else {
                still_waiting.push(module);
            }
        }

        if ready.is_empty() {
            // Circular dependency - break by picking first available
            ready.push(still_waiting.remove(0));
        }

        sorted.extend(ready.clone());
        remaining = still_waiting;
    }

    sorted
}

/// Synthesize a single module
fn synthesize_module(
    module: &ModuleSpec,
    _cache: &SynthCache,
) -> Result<SolveResult, String> {
    // Convert module to problem
    let problem = module_to_problem(module)?;

    // Solve the problem
    let result = crate::solver::solve_problem(&problem);

    Ok(result)
}

/// Convert module spec to problem
fn module_to_problem(module: &ModuleSpec) -> Result<Problem, String> {
    if module.examples.is_empty() {
        return Err("Module has no examples".to_string());
    }

    let first = &module.examples[0];

    Ok(Problem {
        name: module.name.clone(),
        category: "hierarchical",
        description: "",
        signature: "",
        examples: module.examples.clone(),
        holdouts: Vec::new(),
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: first.inputs.iter().any(|v| matches!(v, crate::benchmark::Value::Tree(_))),
        explicit_stack: false,
        functions: Vec::new(),
    })
}

/// Extract exports from solve result
fn extract_exports(result: &SolveResult) -> Vec<String> {
    // Simple extraction - in production would parse the code
    vec![result.method.clone()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_by_dependencies() {
        let modules = vec![
            ModuleSpec {
                name: "a".to_string(),
                interface: super::super::Interface {
                    imports: Vec::new(),
                    exports: Vec::new(),
                    types: Vec::new(),
                },
                examples: Vec::new(),
                dependencies: vec!["b".to_string()],
                types: HashMap::new(),
            },
            ModuleSpec {
                name: "b".to_string(),
                interface: super::super::Interface {
                    imports: Vec::new(),
                    exports: Vec::new(),
                    types: Vec::new(),
                },
                examples: Vec::new(),
                dependencies: Vec::new(),
                types: HashMap::new(),
            },
        ];

        let sorted = sort_by_dependencies(&modules);
        assert_eq!(sorted[0].name, "b"); // b first (no deps)
        assert_eq!(sorted[1].name, "a"); // a second (depends on b)
    }

    #[test]
    fn test_cache() {
        let mut cache = SynthCache::new();
        assert_eq!(cache.stats().module_count, 0);

        cache.cache_module(SynthesizedModule {
            name: "test".to_string(),
            code: "fn test() {}".to_string(),
            exports: Vec::new(),
        });

        assert_eq!(cache.stats().module_count, 1);
        assert!(cache.get_module("test").is_some());
    }
}
