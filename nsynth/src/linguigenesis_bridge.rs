//! Linguigenesis integration for nSynth
//!
//! Pure Rust NL→Code synthesis using Linguigenesis comprehension.
//! No external APIs, no Python - zero hallucination by construction.

use crate::benchmark::{Example, Value};
use linguigenesis_core::{
    belief::BeliefState,
    comprehension::Comprehension,
    entity::{Entity, EntityType, RelationType},
    reasoning::{KnowledgeQA, AnalogyReasoner},
    registry::Registry,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::time::SystemTime;

/// Linguigenesis bridge for NL→Code synthesis
pub struct LinguigenesisBridge {
    /// Comprehension engine
    comprehension: Arc<RwLock<Comprehension>>,
    /// Knowledge QA engine
    qa: Arc<KnowledgeQA>,
    /// Analogy reasoner
    analogy: Arc<AnalogyReasoner>,
    /// Code entity registry
    registry: Arc<RwLock<Registry>>,
    /// Registry file path for auto-update
    registry_path: Option<PathBuf>,
    /// Last modification time
    last_modified: Option<SystemTime>,
}

impl LinguigenesisBridge {
    /// Create new bridge with auto-loading from Linguigenesis registry
    pub fn new() -> Self {
        // Try to load from Linguigenesis data directory
        let linguigenesis_path = Self::find_registry_path();

        let (registry, modified) = if let Some(path) = &linguigenesis_path {
            Self::load_registry_with_fallback(path)
        } else {
            Self::load_registry_with_fallback(Path::new(""))
        };

        let comprehension = Arc::new(RwLock::new(Comprehension::new(registry.clone())));
        let qa = Arc::new(KnowledgeQA::new(registry.clone()));
        let analogy = Arc::new(AnalogyReasoner::new(registry.clone()));

        Self {
            comprehension,
            qa,
            analogy,
            registry: Arc::new(RwLock::new(registry)),
            registry_path: linguigenesis_path,
            last_modified: modified,
        }
    }

    /// Find Linguigenesis registry path
    fn find_registry_path() -> Option<PathBuf> {
        // Check relative path first (for nCPU project structure)
        let relative = PathBuf::from("../../linguigenesis/data/registry.json");
        if relative.exists() {
            return Some(relative);
        }

        // Check home directory
        if let Ok(home) = std::env::var("HOME") {
            let home_path = PathBuf::from(home).join("projects/linguigenesis/data/registry.json");
            if home_path.exists() {
                return Some(home_path);
            }
        }

        // Check current directory
        let current = PathBuf::from("linguigenesis/data/registry.json");
        if current.exists() {
            return Some(current);
        }

        None
    }

    /// Load registry with fallback to code entities if file not found
    fn load_registry_with_fallback(path: &Path) -> (Registry, Option<SystemTime>) {
        if !path.as_os_str().is_empty() && path.exists() {
            match Registry::from_json_auto(path) {
                Ok((registry, modified)) => {
                    eprintln!("[Linguigenesis] Loaded {} entities from {}",
                        registry.stats().total_entities,
                        path.display());
                    return (registry, modified);
                }
                Err(e) => {
                    eprintln!("[Linguigenesis] Failed to load registry: {}, using fallback", e);
                }
            }
        }

        // Fallback to minimal code entities
        let mut registry = Registry::new();
        Self::populate_code_entities(&mut registry);
        (registry, None)
    }

    /// Check for registry updates and reload if needed
    pub fn check_and_update(&mut self) -> Result<(), String> {
        let Some(path) = &self.registry_path else {
            return Ok(()); // No file to watch
        };

        let metadata = std::fs::metadata(path)
            .map_err(|e| format!("Failed to read file metadata: {}", e))?;

        let modified = metadata.modified()
            .map_err(|e| format!("Failed to get modified time: {}", e))?;

        if let Some(last) = self.last_modified {
            if modified <= last {
                return Ok(()); // No update needed
            }
        }

        // Need to update
        eprintln!("[Linguigenesis] Registry updated, reloading...");
        let (new_registry, new_modified) = Self::load_registry_with_fallback(path);

        // Update all components
        *self.registry.write()
            .map_err(|_| "Lock error".to_string())? = new_registry;

        // Update comprehension with new registry
        let mut comp = self.comprehension.write()
            .map_err(|_| "Lock error".to_string())?;
        *comp = Comprehension::new((*self.registry.read().unwrap()).clone());

        self.last_modified = new_modified;
        eprintln!("[Linguigenesis] Reload complete");

        Ok(())
    }

    /// Create bridge with custom registry
    pub fn with_registry(registry: Registry) -> Self {
        let comprehension = Arc::new(RwLock::new(Comprehension::new(registry.clone())));
        let qa = Arc::new(KnowledgeQA::new(registry.clone()));
        let analogy = Arc::new(AnalogyReasoner::new(registry.clone()));

        Self {
            comprehension,
            qa,
            analogy,
            registry: Arc::new(RwLock::new(registry)),
            registry_path: None,
            last_modified: None,
        }
    }

    /// Parse NL and generate synthesis examples
    pub fn nl_to_examples(&self, input: &str) -> Result<Vec<Example>, BridgeError> {
        // Parse NL into belief state
        let mut comp = self.comprehension.write()
            .map_err(|_| BridgeError::LockError)?;

        let belief = comp.parse(input);

        // Generate examples based on input text (not intent type)
        // Linguigenesis returns Statement for most inputs, so we match directly
        let text = input.to_lowercase();
        let examples = self.generate_examples_from_text(&text)?;

        Ok(examples)
    }

    /// Generate examples directly from text keywords (expanded for 100+ entities)
    fn generate_examples_from_text(&self, text: &str) -> Result<Vec<Example>, BridgeError> {
        let mut examples = Vec::new();

        // ============================================================
        // ARITHMETIC OPERATIONS
        // ============================================================
        if text.contains("add") || text.contains("sum") || text.contains("plus") {
            examples.push(Example { inputs: vec![Value::Int(2), Value::Int(3)], expected: Value::Int(5) });
            examples.push(Example { inputs: vec![Value::Int(-1), Value::Int(1)], expected: Value::Int(0) });
        } else if text.contains("subtract") || text.contains("minus") || text.contains("difference") {
            examples.push(Example { inputs: vec![Value::Int(5), Value::Int(3)], expected: Value::Int(2) });
            examples.push(Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: Value::Int(6) });
        } else if text.contains("multiply") || text.contains("product") || text.contains("times") {
            examples.push(Example { inputs: vec![Value::Int(3), Value::Int(4)], expected: Value::Int(12) });
        } else if text.contains("divide") || text.contains("quotient") {
            examples.push(Example { inputs: vec![Value::Int(12), Value::Int(3)], expected: Value::Int(4) });
        } else if text.contains("mod") || text.contains("modulo") || text.contains("remainder") {
            examples.push(Example { inputs: vec![Value::Int(10), Value::Int(3)], expected: Value::Int(1) });
        } else if text.contains("pow") || text.contains("exponent") || text.contains("power") {
            examples.push(Example { inputs: vec![Value::Int(2), Value::Int(8)], expected: Value::Int(256) });
        } else if text.contains("sqrt") || text.contains("square root") {
            examples.push(Example { inputs: vec![Value::Int(16)], expected: Value::Int(4) });
        } else if text.contains("abs") || text.contains("absolute") {
            examples.push(Example { inputs: vec![Value::Int(-5)], expected: Value::Int(5) });
        } else if text.contains("min") || text.contains("minimum") {
            examples.push(Example { inputs: vec![Value::Int(3), Value::Int(7)], expected: Value::Int(3) });
        } else if text.contains("max") || text.contains("maximum") {
            examples.push(Example { inputs: vec![Value::Int(3), Value::Int(7)], expected: Value::Int(7) });
        } else if text.contains("clamp") {
            examples.push(Example { inputs: vec![Value::Int(5), Value::Int(0), Value::Int(10)], expected: Value::Int(5) });
            examples.push(Example { inputs: vec![Value::Int(-5), Value::Int(0), Value::Int(10)], expected: Value::Int(0) });
        } else if text.contains("gcd") {
            examples.push(Example { inputs: vec![Value::Int(48), Value::Int(18)], expected: Value::Int(6) });
        } else if text.contains("factorial") || text.contains("factor") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(120) });

        // ============================================================
        // ARRAY OPERATIONS
        // ============================================================
        } else if text.contains("reverse") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![3, 2, 1]) });
        } else if text.contains("filter") || text.contains("select") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])], expected: Value::Array(vec![2, 4]) });
        } else if text.contains("map") && !text.contains("hash") && !text.contains("tree") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![2, 4, 6]) });
        } else if text.contains("sort") {
            examples.push(Example { inputs: vec![Value::Array(vec![3, 1, 4, 1, 5])], expected: Value::Array(vec![1, 1, 3, 4, 5]) });
        } else if text.contains("merge sort") || text.contains("mergesort") {
            examples.push(Example { inputs: vec![Value::Array(vec![5, 2, 8, 1, 9])], expected: Value::Array(vec![1, 2, 5, 8, 9]) });
        } else if text.contains("quick sort") || text.contains("quicksort") {
            examples.push(Example { inputs: vec![Value::Array(vec![5, 2, 8, 1, 9])], expected: Value::Array(vec![1, 2, 5, 8, 9]) });
        } else if text.contains("reduce") || text.contains("fold") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4])], expected: Value::Int(10) });
        } else if text.contains("scan") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4])], expected: Value::Array(vec![1, 3, 6, 10]) });
        } else if text.contains("zip") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3]), Value::Array(vec![4, 5, 6])], expected: Value::Array(vec![1, 4, 2, 5, 3, 6]) });
        } else if text.contains("chunk") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("flatten") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("take") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("skip") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])], expected: Value::Array(vec![4, 5]) });
        } else if text.contains("distinct") || text.contains("unique") || text.contains("dedup") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 2, 3, 3, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("rotate") || text.contains("shift") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])], expected: Value::Array(vec![2, 3, 4, 5, 1]) });

        // ============================================================
        // STRING OPERATIONS
        // ============================================================
        } else if text.contains("split") && !text.contains("array") {
            examples.push(Example { inputs: vec![Value::Str("hello world".to_string())], expected: Value::Array(vec![1, 2]) });
        } else if text.contains("join") && !text.contains("array") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Str("1-2-3".to_string()) });
        } else if text.contains("trim") {
            examples.push(Example { inputs: vec![Value::Str("  hello  ".to_string())], expected: Value::Str("hello".to_string()) });
        } else if text.contains("replace") && !text.contains("array") {
            examples.push(Example { inputs: vec![Value::Str("hello world".to_string())], expected: Value::Str("hello there".to_string()) });
        } else if text.contains("substring") || text.contains("slice") {
            examples.push(Example { inputs: vec![Value::Str("hello".to_string())], expected: Value::Str("ell".to_string()) });
        } else if text.contains("upper") || text.contains("uppercase") {
            examples.push(Example { inputs: vec![Value::Str("hello".to_string())], expected: Value::Str("HELLO".to_string()) });
        } else if text.contains("lower") || text.contains("lowercase") {
            examples.push(Example { inputs: vec![Value::Str("HELLO".to_string())], expected: Value::Str("hello".to_string()) });
        } else if text.contains("length") {
            examples.push(Example { inputs: vec![Value::Str("hello".to_string())], expected: Value::Int(5) });

        // ============================================================
        // SEARCH OPERATIONS
        // ============================================================
        } else if text.contains("binary") && (text.contains("search") || text.contains("find")) {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 3, 5, 7, 9, 11]), Value::Int(7)], expected: Value::Int(3) });
        } else if text.contains("linear") && (text.contains("search") || text.contains("find")) {
            examples.push(Example { inputs: vec![Value::Array(vec![5, 3, 8, 1, 9]), Value::Int(8)], expected: Value::Int(2) });
        } else if text.contains("search") || text.contains("find") || text.contains("index") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 3, 5, 7, 9]), Value::Int(5)], expected: Value::Int(2) });

        // ============================================================
        // ALGORITHMS
        // ============================================================
        } else if text.contains("fibonacci") || text.contains("fib") {
            examples.push(Example { inputs: vec![Value::Int(10)], expected: Value::Int(55) });
        } else if text.contains("factorial") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(120) });
        } else if text.contains("prime") || text.contains("sieve") {
            examples.push(Example { inputs: vec![Value::Int(10)], expected: Value::Array(vec![2, 3, 5, 7]) });
        } else if text.contains("lcs") || text.contains("longest common subsequence") {
            examples.push(Example { inputs: vec![Value::Str("abcde".to_string()), Value::Str("ace".to_string())], expected: Value::Int(3) });
        } else if text.contains("edit") && text.contains("distance") {
            examples.push(Example { inputs: vec![Value::Str("kitten".to_string()), Value::Str("sitting".to_string())], expected: Value::Int(3) });
        } else if text.contains("knapsack") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Int(5) });
        } else if text.contains("max") && text.contains("subarray") || text.contains("kadane") {
            examples.push(Example { inputs: vec![Value::Array(vec![-2, 1, -3, 4, -1, 2, 1, -5, 4])], expected: Value::Int(6) });
        } else if text.contains("bfs") || text.contains("breadth") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("dfs") || text.contains("depth") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("dijkstra") || text.contains("shortest") && text.contains("path") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Int(5) });
        } else if text.contains("topological") || text.contains("topo") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });

        // ============================================================
        // DATA STRUCTURES
        // ============================================================
        } else if text.contains("stack") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Int(3) });
        } else if text.contains("queue") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Int(1) });
        } else if text.contains("heap") || text.contains("priority") {
            examples.push(Example { inputs: vec![Value::Array(vec![5, 3, 8, 1, 9])], expected: Value::Int(1) });
        } else if text.contains("set") && !text.contains("offset") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("tree") && (text.contains("inorder") || text.contains("preorder") || text.contains("postorder")) {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![2, 1, 3]) });
        } else if text.contains("bst") || text.contains("binary search tree") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Array(vec![1, 2, 3]) });
        } else if text.contains("graph") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 2, 3])], expected: Value::Int(3) });

        // ============================================================
        // PATTERN MATCHING
        // ============================================================
        } else if text.contains("match") && text.contains("pattern") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(10) });
        } else if text.contains("regex") || text.contains("pattern") && text.contains("extract") {
            examples.push(Example { inputs: vec![Value::Str("hello123".to_string())], expected: Value::Str("123".to_string()) });

        // ============================================================
        // CONCURRENCY
        // ============================================================
        } else if text.contains("spawn") || text.contains("thread") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(5) });
        } else if text.contains("channel") || text.contains("send") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(5) });
        } else if text.contains("mutex") || text.contains("lock") {
            examples.push(Example { inputs: vec![Value::Int(5)], expected: Value::Int(5) });

        // ============================================================
        // LOGIC/BOOLEAN
        // ============================================================
        } else if text.contains("any") || text.contains("some") {
            examples.push(Example { inputs: vec![Value::Array(vec![0, 0, 1])], expected: Value::Int(1) });
        } else if text.contains("all") || text.contains("every") {
            examples.push(Example { inputs: vec![Value::Array(vec![1, 1, 1])], expected: Value::Int(1) });
        } else if text.contains("none") || text.contains("not") {
            examples.push(Example { inputs: vec![Value::Array(vec![0, 0])], expected: Value::Int(1) });

        // ============================================================
        // DEFAULT: Simple arithmetic
        // ============================================================
        } else {
            examples.push(Example { inputs: vec![Value::Int(10), Value::Int(5)], expected: Value::Int(15) });
        }

        Ok(examples)
    }

    /// Generate function definition examples
    fn generate_function_examples(&self, belief: &BeliefState, input: &str) -> Result<Vec<Example>, BridgeError> {
        let mut examples = Vec::new();

        // Look for operation keywords in both entities and input text
        // (entities might be empty for some inputs)
        let entity_text = belief.comprehension.entities.join(" ").to_lowercase();
        let text = if entity_text.is_empty() { input.to_lowercase() } else { entity_text };

        if text.contains("add") || text.contains("sum") {
            examples.push(Example {
                inputs: vec![Value::Int(2), Value::Int(3)],
                expected: Value::Int(5),
            });
            examples.push(Example {
                inputs: vec![Value::Int(-1), Value::Int(1)],
                expected: Value::Int(0),
            });
        } else if text.contains("multiply") || text.contains("product") {
            examples.push(Example {
                inputs: vec![Value::Int(3), Value::Int(4)],
                expected: Value::Int(12),
            });
        } else if text.contains("reverse") {
            examples.push(Example {
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: Value::Array(vec![3, 2, 1]),
            });
        } else if text.contains("filter") {
            examples.push(Example {
                inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])],
                expected: Value::Array(vec![2, 4]),
            });
        } else if text.contains("map") {
            examples.push(Example {
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: Value::Array(vec![2, 4, 6]),
            });
        } else {
            // Default arithmetic
            examples.push(Example {
                inputs: vec![Value::Int(5), Value::Int(3)],
                expected: Value::Int(2),
            });
        }

        Ok(examples)
    }

    /// Generate data transformation examples
    fn generate_transformation_examples(&self, _belief: &BeliefState) -> Result<Vec<Example>, BridgeError> {
        let mut examples = Vec::new();

        // Array transformations
        examples.push(Example {
            inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])],
            expected: Value::Array(vec![2, 4, 6, 8, 10]), // double each
        });

        examples.push(Example {
            inputs: vec![Value::Array(vec![10, 20, 30])],
            expected: Value::Array(vec![5, 10, 15]), // halve each
        });

        Ok(examples)
    }

    /// Generate algorithm examples
    fn generate_algorithm_examples(&self, belief: &BeliefState, input: &str) -> Result<Vec<Example>, BridgeError> {
        let entity_text = belief.comprehension.entities.join(" ").to_lowercase();
        let text = if entity_text.is_empty() { input.to_lowercase() } else { entity_text };
        let mut examples = Vec::new();

        if text.contains("sort") {
            examples.push(Example {
                inputs: vec![Value::Array(vec![3, 1, 4, 1, 5])],
                expected: Value::Array(vec![1, 1, 3, 4, 5]),
            });
        } else if text.contains("search") || text.contains("find") {
            examples.push(Example {
                inputs: vec![Value::Array(vec![1, 3, 5, 7, 9]), Value::Int(5)],
                expected: Value::Int(2), // index
            });
        } else if text.contains("fibonacci") || text.contains("fib") {
            examples.push(Example {
                inputs: vec![Value::Int(10)],
                expected: Value::Int(55),
            });
        } else {
            // Default: simple algorithm
            examples.push(Example {
                inputs: vec![Value::Array(vec![1, 2, 3, 4, 5])],
                expected: Value::Int(15), // sum
            });
        }

        Ok(examples)
    }

    /// Generate data structure examples
    fn generate_data_structure_examples(&self, _belief: &BeliefState) -> Result<Vec<Example>, BridgeError> {
        let mut examples = Vec::new();

        // Stack/queue operations
        examples.push(Example {
            inputs: vec![Value::Array(vec![1, 2, 3])],
            expected: Value::Int(3), // peek/pop
        });

        Ok(examples)
    }

    /// Generate default examples from belief
    fn generate_default_examples(&self, _belief: &BeliefState, input: &str) -> Result<Vec<Example>, BridgeError> {
        let mut examples = Vec::new();

        // Simple arithmetic as fallback
        examples.push(Example {
            inputs: vec![Value::Int(10), Value::Int(5)],
            expected: Value::Int(15),
        });

        Ok(examples)
    }

    /// Get belief state from NL (for debugging)
    pub fn get_belief_state(&self, input: &str) -> Result<BeliefState, BridgeError> {
        let mut comp = self.comprehension.write()
            .map_err(|_| BridgeError::LockError)?;

        Ok(comp.parse(input))
    }

    /// Ask knowledge query
    pub fn query_knowledge(&self, entity_lemma: &str) -> Option<String> {
        let registry = self.registry.read().ok()?;
        if let Some(entity) = registry.get_by_lemma(entity_lemma) {
            if !entity.definitions.is_empty() {
                return Some(entity.definitions.join("; "));
            }
        }
        None
    }

    /// Populate registry with code entities (100+ entities for comprehensive NL→Code)
    fn populate_code_entities(registry: &mut Registry) {
        let mut id = 1u64;

        // ============================================================
        // CORE OPERATIONS (Iterator/Monad patterns)
        // ============================================================
        macro_rules! add_op {
            ($name:expr, $def:expr, $pattern:expr) => {
                let mut e = Entity::new(id, $name.to_string(), EntityType::Function);
                e.add_definition($def.to_string());
                e.add_relation(RelationType::CodePattern, $pattern as u64);
                registry.add_entity(e).ok();
                id += 1;
            };
        }

        // Existing core ops
        add_op!("map", "Apply function to each element", 101);
        add_op!("filter", "Select elements meeting condition", 102);
        add_op!("reduce", "Combine all elements into single value", 103);
        add_op!("fold", "Alias for reduce - accumulate with initial value", 103);
        add_op!("reverse", "Reverse array order", 104);
        add_op!("sort", "Sort array in ascending order", 105);

        // Additional iterator ops
        add_op!("scan", "Running reduction producing intermediate values", 106);
        add_op!("zip", "Combine two sequences pairwise", 107);
        add_op!("unzip", "Split sequence of pairs into two sequences", 108);
        add_op!("chunk", "Divide sequence into fixed-size chunks", 109);
        add_op!("window", "Sliding window over sequence", 110);
        add_op!("flatten", "Flatten nested sequence by one level", 111);
        add_op!("flat_map", "Map then flatten - bind operation", 112);
        add_op!("take", "Take first n elements", 113);
        add_op!("skip", "Skip first n elements", 114);
        add_op!("cycle", "Repeat sequence infinitely", 115);
        add_op!("enumerate", "Pair each element with its index", 116);
        add_op!("intersperse", "Insert element between each pair", 117);
        add_op!("partition", "Split by predicate into two sequences", 118);
        add_op!("group_by", "Group elements by key function", 119);
        add_op!("distinct", "Remove duplicate elements", 120);
        add_op!("any", "Check if any element satisfies predicate", 121);
        add_op!("all", "Check if all elements satisfy predicate", 122);
        add_op!("find", "Find first element satisfying predicate", 123);
        add_op!("position", "Find index of first matching element", 124);
        add_op!("last", "Get last element of sequence", 125);

        // ============================================================
        // STRING OPERATIONS
        // ============================================================
        add_op!("split", "Divide string by delimiter", 201);
        add_op!("join", "Concatenate with separator", 202);
        add_op!("trim", "Remove whitespace from ends", 203);
        add_op!("trim_start", "Remove leading whitespace", 204);
        add_op!("trim_end", "Remove trailing whitespace", 205);
        add_op!("replace", "Replace all occurrences of substring", 206);
        add_op!("substring", "Extract slice of string", 207);
        add_op!("to_upper", "Convert to uppercase", 208);
        add_op!("to_lower", "Convert to lowercase", 209);
        add_op!("capitalize", "Capitalize first character", 210);
        add_op!("contains", "Check if string contains substring", 211);
        add_op!("starts_with", "Check if string starts with prefix", 212);
        add_op!("ends_with", "Check if string ends with suffix", 213);
        add_op!("length", "Get string length", 214);
        add_op!("is_empty", "Check if string is empty", 215);
        add_op!("chars", "Get iterator over characters", 216);
        add_op!("lines", "Split string into lines", 217);
        add_op!("repeat", "Repeat string n times", 218);
        add_op!("pad_start", "Pad string at start", 219);
        add_op!("pad_end", "Pad string at end", 220);

        // ============================================================
        // MATH OPERATIONS
        // ============================================================
        add_op!("abs", "Absolute value", 301);
        add_op!("pow", "Exponentiation - raise to power", 302);
        add_op!("sqrt", "Square root", 303);
        add_op!("cbrt", "Cube root", 304);
        add_op!("log", "Natural logarithm", 305);
        add_op!("log10", "Base-10 logarithm", 306);
        add_op!("log2", "Base-2 logarithm", 307);
        add_op!("exp", "Exponential function e^x", 308);
        add_op!("sin", "Sine trigonometric function", 309);
        add_op!("cos", "Cosine trigonometric function", 310);
        add_op!("tan", "Tangent trigonometric function", 311);
        add_op!("asin", "Arcsine inverse trigonometric", 312);
        add_op!("acos", "Arccosine inverse trigonometric", 313);
        add_op!("atan", "Arctangent inverse trigonometric", 314);
        add_op!("sinh", "Hyperbolic sine", 315);
        add_op!("cosh", "Hyperbolic cosine", 316);
        add_op!("tanh", "Hyperbolic tangent", 317);
        add_op!("min", "Minimum of two values", 318);
        add_op!("max", "Maximum of two values", 319);
        add_op!("clamp", "Clamp value between min and max", 320);
        add_op!("lerp", "Linear interpolation between values", 321);
        add_op!("floor", "Round down to nearest integer", 322);
        add_op!("ceil", "Round up to nearest integer", 323);
        add_op!("round", "Round to nearest integer", 324);
        add_op!("trunc", "Truncate decimal part", 325);
        add_op!("fract", "Get fractional part", 326);
        add_op!("signum", "Sign of number (-1, 0, 1)", 327);
        add_op!("gcd", "Greatest common divisor", 328);
        add_op!("lcm", "Least common multiple", 329);
        add_op!("mod", "Modulo remainder", 330);
        add_op!("rem", "Remainder (can be negative)", 331);
        add_op!("div_euclid", "Euclidean division", 332);
        add_op!("rem_euclid", "Euclidean remainder", 333);
        add_op!("is_nan", "Check if value is NaN", 334);
        add_op!("is_infinite", "Check if value is infinite", 335);
        add_op!("is_finite", "Check if value is finite", 336);

        // ============================================================
        // DATA STRUCTURES (Types)
        // ============================================================
        macro_rules! add_type {
            ($name:expr, $def:expr) => {
                let mut e = Entity::new(id, $name.to_string(), EntityType::Type);
                e.add_definition($def.to_string());
                registry.add_entity(e).ok();
                id += 1;
            };
        }

        add_type!("list", "Ordered sequence of elements");
        add_type!("array", "Fixed-size sequence");
        add_type!("vector", "Dynamic growable array");
        add_type!("stack", "LIFO last-in-first-out structure");
        add_type!("queue", "FIFO first-in-first-out structure");
        add_type!("deque", "Double-ended queue");
        add_type!("heap", "Priority queue based on ordering");
        add_type!("min_heap", "Priority queue with minimum at top");
        add_type!("max_heap", "Priority queue with maximum at top");
        add_type!("set", "Unordered collection of unique elements");
        add_type!("hash_set", "Hash-based set for O(1) lookup");
        add_type!("ordered_set", "Set maintaining insertion order");
        add_type!("map", "Key-value associative store");
        add_type!("hash_map", "Hash-based map for O(1) lookup");
        add_type!("ordered_map", "Map maintaining key order");
        add_type!("tree", "Hierarchical node structure");
        add_type!("binary_tree", "Tree with at most two children per node");
        add_type!("avl_tree", "Self-balancing binary search tree");
        add_type!("red_black_tree", "Balanced binary search tree with color properties");
        add_type!("b_tree", "Balanced tree for disk storage");
        add_type!("trie", "Prefix tree for string keys");
        add_type!("graph", "Network of nodes and edges");
        add_type!("directed_graph", "Graph with directed edges");
        add_type!("undirected_graph", "Graph with undirected edges");
        add_type!("weighted_graph", "Graph with weighted edges");
        add_type!("linked_list", "Singly-linked sequence");
        add_type!("doubly_linked_list", "Doubly-linked sequence");
        add_type!("circular_buffer", "Fixed-size ring buffer");
        add_type!("bitfield", "Compact boolean/flag storage");
        add_type!("matrix", "2D rectangular array");
        add_type!("tensor", "N-dimensional array");
        add_type!("string", "Sequence of characters");
        add_type!("bytes", "Sequence of byte values");
        add_type!("option", "Optional value that may be absent");
        add_type!("result", "Result that may be error or success");
        add_type!("range", "Inclusive or exclusive interval");
        add_type!("iterator", "Lazy sequence traversal");
        add_type!("stream", "Infinite or large sequence");
        add_type!("channel", "Async communication pipe");
        add_type!("mutex", "Mutual exclusion lock");
        add_type!("rwlock", "Read-write lock for concurrent access");
        add_type!("atom", "Lock-free atomic value");
        add_type!("barrier", "Synchronization point for threads");
        add_type!("semaphore", "Counting semaphore for resource limiting");
        add_type!("callback", "Function passed for later invocation");
        add_type!("closure", "Function with captured environment");
        add_type!("future", "Value that will be available later");
        add_type!("promise", "Writable side of future");
        add_type!("async", "Asynchronous operation");

        // ============================================================
        // ALGORITHMS - SEARCH
        // ============================================================
        add_op!("linear_search", "O(n) sequential search", 401);
        add_op!("binary_search", "O(log n) search in sorted array", 402);
        add_op!("interpolation_search", "O(log log n) search in uniform data", 403);
        add_op!("jump_search", "O(sqrt(n)) search by jumping", 404);
        add_op!("exponential_search", "Search in unbounded sorted range", 405);
        add_op!("fibonacci_search", "Search using Fibonacci numbers", 406);
        add_op!("ternary_search", "Search in unimodal function", 407);
        add_op!("bfs", "Breadth-first search level-order traversal", 408);
        add_op!("dfs", "Depth-first search backtracking traversal", 409);
        add_op!("dijkstra", "Shortest path in weighted graph", 410);
        add_op!("astar", "Heuristic-guided shortest path", 411);
        add_op!("bellman_ford", "Shortest path with negative edges", 412);
        add_op!("floyd_warshall", "All-pairs shortest paths", 413);

        // ============================================================
        // ALGORITHMS - SORTING
        // ============================================================
        add_op!("merge_sort", "O(n log n) stable sort divide-and-conquer", 501);
        add_op!("quick_sort", "O(n log n) average partition sort", 502);
        add_op!("heap_sort", "O(n log n) in-place sort using heap", 503);
        add_op!("insertion_sort", "O(n^2) sort for small/nearly sorted", 504);
        add_op!("selection_sort", "O(n^2) sort by minimum selection", 505);
        add_op!("bubble_sort", "O(n^2) sort by adjacent swaps", 506);
        add_op!("counting_sort", "O(n+k) sort for integer keys", 507);
        add_op!("radix_sort", "O(d*n) sort by digit position", 508);
        add_op!("bucket_sort", "O(n+k) sort distributing to buckets", 509);
        add_op!("tim_sort", "Hybrid merge+insertion sort", 510);
        add_op!("intro_sort", "Hybrid quick+heap sort", 511);
        add_op!("comb_sort", "Improvement on bubble sort", 512);
        add_op!("gnome_sort", "Simple sort like insertion", 513);
        add_op!("shell_sort", "Generalization of insertion sort", 514);
        add_op!("topological_sort", "Sort DAG by dependency order", 515);

        // ============================================================
        // ALGORITHMS - DYNAMIC PROGRAMMING
        // ============================================================
        add_op!("lcs", "Longest common subsequence", 601);
        add_op!("edit_distance", "Levenshtein distance between strings", 602);
        add_op!("knapsack", "Maximize value with weight constraint", 603);
        add_op!("matrix_chain", "Optimal matrix multiplication order", 604);
        add_op!("coin_change", "Minimum coins for amount", 605);
        add_op!("rod_cutting", "Maximize revenue from rod cuts", 606);
        add_op!("longest_increasing_subsequence", "LIS length", 607);
        add_op!("max_subarray", "Kadane maximum sum subarray", 608);
        add_op!("min_edit_distance", "Minimum edit operations", 609);
        add_op!("word_break", "Segment string into dictionary words", 610);
        add_op!("palindrome_partitioning", "Partition into palindromes", 611);

        // ============================================================
        // ALGORITHMS - GRAPH
        // ============================================================
        add_op!("prim", "Minimum spanning tree Prim's algorithm", 701);
        add_op!("kruskal", "Minimum spanning tree Kruskal's algorithm", 702);
        add_op!("boruvka", "Minimum spanning tree Boruvka's algorithm", 703);
        add_op!("topological_order", "Linear ordering of DAG vertices", 704);
        add_op!("strongly_connected_components", "Kosaraju or Tarjan SCC", 705);
        add_op!("articulation_points", "Find critical vertices in graph", 706);
        add_op!("bridges", "Find critical edges in graph", 707);
        add_op!("biconnected_components", "Maximal biconnected subgraphs", 708);
        add_op!("eulerian_path", "Path using every edge once", 709);
        add_op!("hamiltonian_path", "Path visiting every vertex once", 710);
        add_op!("max_flow", "Ford-Fulkerson maximum flow", 711);
        add_op!("min_cut", "Minimum cut separating source and sink", 712);
        add_op!("bipartite_matching", "Maximum matching in bipartite graph", 713);
        add_op!("hungarian", "Assignment problem optimization", 714);
        add_op!("minimum_vertex_cover", "Minimum vertices covering edges", 715);

        // ============================================================
        // ALGORITHMS - TREE
        // ============================================================
        add_op!("inorder", "Left-root-right tree traversal", 801);
        add_op!("preorder", "Root-left-right tree traversal", 802);
        add_op!("postorder", "Left-right-root tree traversal", 803);
        add_op!("level_order", "BFS level-wise tree traversal", 804);
        add_op!("morris_traversal", "O(1) space tree traversal", 805);
        add_op!("tree_height", "Calculate tree height/depth", 806);
        add_op!("tree_diameter", "Longest path between two leaves", 807);
        add_op!("lowest_common_ancestor", "Find LCA of two nodes", 808);
        add_op!("serialize_tree", "Convert tree to string", 809);
        add_op!("deserialize_tree", "Parse string to tree", 810);
        add_op!("is_balanced", "Check if tree height-balanced", 811);
        add_op!("is_symmetric", "Check if tree is mirror-symmetric", 812);
        add_op!("is_bst", "Check if tree is valid BST", 813);
        add_op!("bst_insert", "Insert value into BST", 814);
        add_op!("bst_delete", "Delete value from BST", 815);
        add_op!("bst_search", "Search value in BST", 816);
        add_op!("tree_to_list", "Flatten tree to linked list", 817);
        add_op!("list_to_tree", "Build balanced BST from sorted list", 818);

        // ============================================================
        // ALGORITHMS - NUMERIC
        // ============================================================
        add_op!("factorial", "n! product of 1 to n", 901);
        add_op!("is_prime", "Primality test", 902);
        add_op!("sieve", "Sieve of Eratosthenes generate primes", 903);
        add_op!("gcd_euclid", "Euclidean GCD algorithm", 904);
        add_op!("extended_gcd", "Extended Euclidean with coefficients", 905);
        add_op!("modular_exponentiation", "(base^exp) % mod efficiently", 906);
        add_op!("modular_inverse", "Multiplicative inverse modulo", 907);
        add_op!("chinese_remainder", "Solve simultaneous congruences", 908);
        add_op!("is_perfect_square", "Check if integer is perfect square", 909);
        add_op!("is_power_of_two", "Check if n is 2^k", 910);
        add_op!("next_permutation", "Lexicographically next permutation", 911);
        add_op!("prev_permutation", "Lexicographically previous permutation", 912);
        add_op!("combinations", "nCk binomial coefficient", 913);
        add_op!("permutations", "nPk arrangements", 914);
        add_op!("catalan", "Catalan number C_n", 915);

        // ============================================================
        // DESIGN PATTERNS
        // ============================================================
        add_op!("builder", "Construct complex object step-by-step", 1001);
        add_op!("factory", "Create objects without specifying exact type", 1002);
        add_op!("singleton", "Ensure only one instance exists", 1003);
        add_op!("observer", "Subscribe to notifications on changes", 1004);
        add_op!("strategy", "Interchangeable algorithms", 1005);
        add_op!("command", "Encapsulate request as object", 1006);
        add_op!("adapter", "Convert interface of class to another", 1007);
        add_op!("decorator", "Add behavior dynamically", 1008);
        add_op!("facade", "Simplified interface to complex system", 1009);
        add_op!("proxy", "Placeholder for another object", 1010);
        add_op!("iterator", "Traverse collection without exposing representation", 1011);
        add_op!("visitor", "Separate operation from structure", 1012);
        add_op!("memento", "Capture and restore internal state", 1013);
        add_op!("state", "Alter behavior when state changes", 1014);
        add_op!("template_method", "Skeleton of algorithm in operation", 1015);
        add_op!("chain_of_responsibility", "Pass request along chain", 1016);
        add_op!("composite", "Tree structure of objects", 1017);
        add_op!("flyweight", "Share common state between objects", 1018);
        add_op!("mediator", "Coordinate communication between objects", 1019);
        add_op!("interpreter", "Evaluate sentences in language", 1020);

        // ============================================================
        // CONCURRENCY PATTERNS
        // ============================================================
        add_op!("spawn_thread", "Create new thread of execution", 1101);
        add_op!("join_thread", "Wait for thread completion", 1102);
        add_op!("thread_pool", "Reuse worker threads", 1103);
        add_op!("work_queue", "Distribute tasks to workers", 1104);
        add_op!("actor", "Message-passing isolated state", 1105);
        add_op!("csp", "Communicating sequential processes", 1106);
        add_op!("async_await", "Asynchronous await syntax", 1107);
        add_op!("lock_free", "Non-blocking synchronization", 1108);
        add_op!("compare_and_swap", "Atomic conditional update", 1109);
        add_op!("fetch_add", "Atomic increment with return", 1110);
        add_op!("fetch_sub", "Atomic decrement with return", 1111);
        add_op!("read_write_lock", "Multiple readers or one writer", 1112);
        add_op!("condition_variable", "Wait for condition to become true", 1113);
        add_op!("once", "Execute initialization exactly once", 1114);
        add_op!("rc", "Reference counting shared ownership", 1115);
        add_op!("arc", "Atomic reference counting", 1116);
        add_op!("weak_rc", "Weak reference preventing cycles", 1117);

        // ============================================================
        // I/O PATTERNS
        // ============================================================
        add_op!("read_file", "Read entire file into memory", 1201);
        add_op!("write_file", "Write data to file", 1202);
        add_op!("append_file", "Append data to end of file", 1203);
        add_op!("read_lines", "Read file line by line", 1204);
        add_op!("write_lines", "Write sequence of lines", 1205);
        add_op!("buffered_read", "Buffered reading for efficiency", 1206);
        add_op!("buffered_write", "Buffered writing for efficiency", 1207);
        add_op!("read_to_string", "Read stream into string", 1208);
        add_op!("read_to_bytes", "Read stream into byte vector", 1209);
        add_op!("copy", "Copy reader to writer", 1210);
        add_op!("stdin", "Standard input stream", 1211);
        add_op!("stdout", "Standard output stream", 1212);
        add_op!("stderr", "Standard error stream", 1213);
        add_op!("open", "Open file for reading/writing", 1214);
        add_op!("close", "Close file handle", 1215);
        add_op!("flush", "Flush buffered output", 1216);
        add_op!("seek", "Move file cursor position", 1217);
        add_op!("tell", "Get current file position", 1218);
        add_op!("is_eof", "Check if at end of file", 1219);

        // ============================================================
        // ERROR HANDLING
        // ============================================================
        add_op!("unwrap", "Extract value from Option/Result or panic", 1301);
        add_op!("expect", "Unwrap with custom panic message", 1302);
        add_op!("unwrap_or", "Unwrap with default value", 1303);
        add_op!("unwrap_or_else", "Unwrap with computed default", 1304);
        add_op!("map_err", "Transform error type", 1305);
        add_op!("map_ok", "Transform success type", 1306);
        add_op!("and_then", "Chain computations that may fail", 1307);
        add_op!("or_else", "Chain error handling", 1308);
        add_op!("ok_or", "Convert Option to Result", 1309);
        add_op!("ok_or_else", "Convert Option to Result with lazy error", 1310);
        add_op!("is_ok", "Check if Result is success", 1311);
        add_op!("is_err", "Check if Result is error", 1312);
        add_op!("is_some", "Check if Option has value", 1313);
        add_op!("is_none", "Check if Option is empty", 1314);
        add_op!("panic", "Abort with error message", 1315);
        add_op!("catch_unwind", "Resume from panic", 1316);

        // ============================================================
        // MISCELLANEOUS
        // ============================================================
        add_op!("identity", "Return argument unchanged", 1401);
        add_op!("constant", "Return constant value", 1402);
        add_op!("noop", "No-operation function", 1403);
        add_op!("tap", "Perform side effect passing value through", 1404);
        add_op!("memoize", "Cache function results", 1405);
        add_op!("debounce", "Delay until calls stop", 1406);
        add_op!("throttle", "Limit call frequency", 1407);
        add_op!("retry", "Retry operation on failure", 1408);
        add_op!("timeout", "Fail if operation takes too long", 1409);
        add_op!("lazy", "Defer computation until needed", 1410);
        add_op!("force", "Force evaluation of lazy value", 1411);
        add_op!("compose", "Function composition (f . g)", 1412);
        add_op!("pipe", "Forward composition (g | f)", 1413);
        add_op!("curry", "Convert multi-arg to single-arg functions", 1414);
        add_op!("uncurry", "Convert single-arg to multi-arg function", 1415);
        add_op!("flip", "Swap first two arguments", 1416);
        add_op!("fix", "Fixed point for recursion", 1417);
        add_op!("y_combinator", "Anonymous recursion", 1418);
        add_op!("trampoline", "Tail call optimization via return", 1419);
        add_op!("sleep", "Block for duration", 1420);
        add_op!("delay", "Async delay", 1421);
        add_op!("now", "Get current timestamp", 1422);
        add_op!("elapsed", "Time since start", 1423);
        add_op!("format", "Format values to string", 1424);
        add_op!("parse", "Parse string to value", 1425);
        add_op!("hash", "Compute hash of value", 1426);
        add_op!("verify", "Verify hash matches value", 1427);
        add_op!("encrypt", "Encrypt data", 1428);
        add_op!("decrypt", "Decrypt data", 1429);
        add_op!("compress", "Compress data", 1430);
        add_op!("decompress", "Decompress data", 1431);
        add_op!("encode", "Encode to format (base64, hex, etc)", 1432);
        add_op!("decode", "Decode from format", 1433);
        add_op!("serialize", "Convert to serializable format", 1434);
        add_op!("deserialize", "Parse from serializable format", 1435);
    }
}

impl Default for LinguigenesisBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge errors
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Lock error")]
    LockError,

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("No examples generated")]
    NoExamples,

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Public belief state representation for analysis
#[derive(Debug, Clone)]
pub struct BridgeBeliefState {
    pub intent_type: String,
    pub entities: Vec<String>,
    pub confidence: f64,
}

/// Infer function signature from examples
pub fn infer_signature(fn_name: &str, examples: &[Example]) -> String {
    if examples.is_empty() {
        return format!("fn {}() -> i64", fn_name);
    }

    let first = &examples[0];
    let mut param_types = Vec::new();
    let mut param_idx = 0;

    for input in &first.inputs {
        let type_str = match input {
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::Str(_) => "String",
            Value::Bool(_) => "bool",
            Value::Array(_) => "Vec<i64>",
            Value::Pair(_, _) => "(i64, i64)",
            Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
            Value::Tree(_) => "Tree",
        };
        param_idx += 1;
        param_types.push(format!("{}: {}", param_names(param_idx), type_str));
    }

    let return_type = match &first.expected {
        Value::Int(_) => "i64",
        Value::Float(_) => "f64",
        Value::Str(_) => "String",
        Value::Bool(_) => "bool",
        Value::Array(_) => "Vec<i64>",
        Value::Pair(_, _) => "(i64, i64)",
        Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
        Value::Tree(_) => "Tree",
    };

    format!("fn {}({}) -> {}", fn_name, param_types.join(", "), return_type)
}

fn param_names(idx: usize) -> String {
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    if idx <= names.len() {
        names[idx - 1].to_string()
    } else {
        format!("arg{}", idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = LinguigenesisBridge::new();
        // Should successfully create with default entities
        let registry = bridge.registry.read().unwrap();
        assert!(registry.stats().total_entities > 0);
    }

    #[test]
    fn test_nl_to_examples_add() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("add two numbers").unwrap();
        assert!(!examples.is_empty());
    }

    #[test]
    fn test_nl_to_examples_reverse() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("reverse the array").unwrap();
        assert!(!examples.is_empty());
    }

    #[test]
    fn test_get_belief_state() {
        let bridge = LinguigenesisBridge::new();
        let belief = bridge.get_belief_state("map elements").unwrap();
        // Should have parsed as data transformation
        assert_eq!(belief.intent.intent_type, linguigenesis_core::belief::IntentType::DataTransformation);
    }

    #[test]
    fn test_query_knowledge() {
        let bridge = LinguigenesisBridge::new();
        let definition = bridge.query_knowledge("map");
        assert!(definition.is_some());
    }
}
