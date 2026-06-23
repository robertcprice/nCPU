//! Linguigenesis integration for nSynth
//!
//! Pure Rust NL→Code synthesis using Linguigenesis comprehension.
//! No external APIs, no Python - zero hallucination by construction.

use crate::benchmark::{Example, Problem, Value};
use linguigenesis_core::{
    belief::BeliefState,
    coding_comprehension::{CodingComprehension, ComprehensionOutcome},
    coding_dialogue::{
        apply_clarification, build_clarifications, format_clarification_prompt,
        needs_clarification, ClarificationField, ClarificationQuestion,
    },
    coding_requirements::{LiteralValue, SynthesisRequirement},
    comprehension::Comprehension,
    entity_resolution::EntityResolver,
    computing_knowledge_import::merge_computing_knowledge,
    reasoning::{AnalogyReasoner, KnowledgeQA},
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
                Ok((mut registry, modified)) => {
                    eprintln!(
                        "[Linguigenesis] Loaded {} entities from {}",
                        registry.stats().total_entities,
                        path.display()
                    );
                    Self::merge_coding_registry(&mut registry);
                    return (registry, modified);
                }
                Err(e) => {
                    eprintln!(
                        "[Linguigenesis] Failed to load registry: {}, using fallback",
                        e
                    );
                }
            }
        }

        let mut registry = Registry::new();
        Self::merge_coding_registry(&mut registry);
        if registry.stats().total_entities == 0 {
            eprintln!(
                "[Linguigenesis] No coding registry loaded — set ../../linguigenesis/data/coding_registry.json"
            );
        }
        (registry, None)
    }

    fn find_coding_registry_path() -> Option<PathBuf> {
        let relative = PathBuf::from("../../linguigenesis/data/coding_registry.json");
        if relative.exists() {
            return Some(relative);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_path =
                PathBuf::from(home).join("projects/linguigenesis/data/coding_registry.json");
            if home_path.exists() {
                return Some(home_path);
            }
        }
        let current = PathBuf::from("linguigenesis/data/coding_registry.json");
        if current.exists() {
            return Some(current);
        }
        None
    }

    fn find_computing_knowledge_path() -> Option<PathBuf> {
        let relative = PathBuf::from("../../linguigenesis/data/computing_knowledge.json");
        if relative.exists() {
            return Some(relative);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_path =
                PathBuf::from(home).join("projects/linguigenesis/data/computing_knowledge.json");
            if home_path.exists() {
                return Some(home_path);
            }
        }
        let current = PathBuf::from("linguigenesis/data/computing_knowledge.json");
        if current.exists() {
            return Some(current);
        }
        None
    }

    fn merge_coding_registry(registry: &mut Registry) {
        if let Some(path) = Self::find_coding_registry_path() {
            if let Ok(coding) = Registry::from_json(&path) {
                if let Err(e) = registry.merge_registry(&coding) {
                    eprintln!("[Linguigenesis] coding_registry merge warning: {}", e);
                }
            }
        }
        if let Some(path) = Self::find_computing_knowledge_path() {
            if let Err(e) = merge_computing_knowledge(registry, &path) {
                eprintln!("[Linguigenesis] computing_knowledge merge warning: {}", e);
            }
        }
        if let Err(errors) =
            linguigenesis_core::coding_registry_validate::validate_coding_registry(registry)
        {
            eprintln!(
                "[Linguigenesis] coding registry validation warnings: {}",
                errors.join("; ")
            );
        }
    }

    /// Check for registry updates and reload if needed
    pub fn check_and_update(&mut self) -> Result<(), String> {
        let Some(path) = &self.registry_path else {
            return Ok(()); // No file to watch
        };

        let metadata =
            std::fs::metadata(path).map_err(|e| format!("Failed to read file metadata: {}", e))?;

        let modified = metadata
            .modified()
            .map_err(|e| format!("Failed to get modified time: {}", e))?;

        if let Some(last) = self.last_modified {
            if modified <= last {
                return Ok(()); // No update needed
            }
        }

        // Need to update
        eprintln!("[Linguigenesis] Registry updated, reloading...");
        let (new_registry, new_modified) = Self::load_registry_with_fallback(path);

        *self
            .registry
            .write()
            .map_err(|_| "Lock error".to_string())? = new_registry.clone();
        *self
            .comprehension
            .write()
            .map_err(|_| "Lock error".to_string())? = Comprehension::new(new_registry.clone());
        self.qa = Arc::new(KnowledgeQA::new(new_registry.clone()));
        self.analogy = Arc::new(AnalogyReasoner::new(new_registry));

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

    /// Clone the loaded KVRM registry (for clarification / comprehension helpers).
    pub fn registry_clone(&self) -> Result<Registry, BridgeError> {
        Ok(self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone())
    }

    /// Parse NL into registry-derived synthesis requirements (KVRM only).
    pub fn nl_to_requirement(&self, input: &str) -> Result<SynthesisRequirement, BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        let mut coding = CodingComprehension::new(registry);
        let req = coding.comprehend(input);
        if req.examples.is_empty() {
            let questions = build_clarifications(&req, coding.registry());
            if !questions.is_empty() {
                return Err(BridgeError::ClarificationNeeded {
                    partial: req,
                    questions,
                });
            }
            return Err(BridgeError::ParseError(if req.unresolved.is_empty() {
                "no examples derived from registry".to_string()
            } else {
                req.unresolved.join("; ")
            }));
        }
        if needs_clarification(&req) {
            let questions = build_clarifications(&req, coding.registry());
            if !questions.is_empty() {
                return Err(BridgeError::ClarificationNeeded {
                    partial: req,
                    questions,
                });
            }
        }
        // SOUNDNESS GATE (P0-NL fail-closed): a registry NL match whose only
        // evidence is the op's canned example(s) must NOT be reported as a
        // confident solve when STRUCTURAL signals show the request was not
        // actually understood. All signals are computed from `req` + the raw
        // request — no phrase blocklist, no request->refuse map.
        if let Some(reason) = unsound_confident_solve(input, &req, coding.registry()) {
            let mut partial = req;
            // Reuse the existing "no operation" clarification path so the
            // downgrade flows through build_clarifications unchanged.
            partial.unresolved.push(reason);
            let questions = build_clarifications(&partial, coding.registry());
            return Err(BridgeError::ClarificationNeeded { partial, questions });
        }
        Ok(req)
    }

    /// Comprehend NL with clarification outcome (ready or typed questions).
    pub fn comprehend_outcome(&self, input: &str) -> Result<ComprehensionOutcome, BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        let mut coding = CodingComprehension::new(registry);
        Ok(coding.comprehend_outcome(input))
    }

    /// Apply a clarification answer and return an updated requirement.
    pub fn apply_clarification(
        &self,
        partial: &mut SynthesisRequirement,
        field: ClarificationField,
        answer: &str,
    ) -> Result<(), BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        if apply_clarification(partial, &field, answer, &registry) {
            Ok(())
        } else {
            Err(BridgeError::InvalidInput(format!(
                "could not apply clarification '{}' for {:?}",
                answer, field
            )))
        }
    }

    /// Parse NL and generate synthesis examples from registry `example_cases`.
    pub fn nl_to_examples(&self, input: &str) -> Result<Vec<Example>, BridgeError> {
        let req = self.nl_to_requirement(input)?;
        synthesis_requirement_to_examples(&req)
    }

    /// Build a synthesis `Problem` from a registry-derived requirement (universal entry).
    pub fn problem_from_requirement(
        &self,
        req: &SynthesisRequirement,
        fn_name: Option<&str>,
    ) -> Result<Problem, BridgeError> {
        let examples = synthesis_requirement_to_examples(req)?;
        let name = fn_name.unwrap_or(&req.function_name).to_string();
        let signature = infer_signature(&name, &examples);
        let signature = Box::leak(signature.into_boxed_str());
        let category = Box::leak(req.category.clone().into_boxed_str());
        let description = Box::leak(req.description.clone().into_boxed_str());
        Ok(Problem {
            name,
            category,
            description,
            signature,
            examples,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        })
    }

    /// NL description → verified synthesis via registry requirements (no keyword routing).
    pub fn synthesize_from_description(
        &self,
        description: &str,
        fn_name: Option<&str>,
    ) -> Result<crate::solver::SolveResult, String> {
        let req = match self.nl_to_requirement(description) {
            Ok(req) => req,
            Err(BridgeError::ClarificationNeeded { questions, .. }) => {
                return Err(format_clarification_prompt(&questions));
            }
            Err(e) => return Err(e.to_string()),
        };
        let problem = self
            .problem_from_requirement(&req, fn_name)
            .map_err(|e| e.to_string())?;
        Ok(crate::solver::solve_problem(&problem))
    }

    /// Synthesize from an already-derived requirement (e.g. after clarification).
    pub fn synthesize_from_requirement(
        &self,
        req: &SynthesisRequirement,
        fn_name: Option<&str>,
    ) -> Result<crate::solver::SolveResult, String> {
        let problem = self
            .problem_from_requirement(req, fn_name)
            .map_err(|e| e.to_string())?;
        Ok(crate::solver::solve_problem(&problem))
    }

    /// Get belief state from NL (for debugging)
    pub fn get_belief_state(&self, input: &str) -> Result<BeliefState, BridgeError> {
        let mut comp = self
            .comprehension
            .write()
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

    #[error("Clarification needed")]
    ClarificationNeeded {
        partial: SynthesisRequirement,
        questions: Vec<ClarificationQuestion>,
    },

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

/// Structural soundness gate for a *registry-derived* confident match.
///
/// Returns `Some(reason)` (a marker pushed into `unresolved`, NOT a user-facing
/// blocklist string) when the requirement should be DOWNGRADED to
/// ClarificationNeeded instead of declared solved. Every signal is computed from
/// the parse the bridge already holds — never a hardcoded bad-phrase list:
///
///   1. **Discrimination floor** — registry NL matches carry no reference impl,
///      so the only generalization evidence is the canned `example_cases`. Fewer
///      than two examples cannot discriminate the intended function from
///      alternatives (e.g. one (`[1,2,3]->[3,2,1]`) pair is consistent with
///      reverse, rotate, sort-desc, ...). Require >=2.
///   2. **Request/signature TYPE mismatch** — the request names a value type
///      (string / array / list) absent from the resolved op's signature
///      (e.g. "reverse a STRING" resolving to `fn reverse(a: Vec<i64>)`).
///   3. **Operation not named by the request** — none of the request's content
///      words corresponds to the resolved op's identity (function name /
///      description / category). The op was guessed from leftover tokens while
///      the actual content words ("parse", "csv", "file") were dropped.
///
/// Inline-example requests (the user supplied their own I/O) are exempt: their
/// evidence is the demonstrated behaviour, not a registry guess. We detect that
/// the same way the comprehension layer does — a non-generic function name with
/// the user's own examples is treated as user-specified.
fn unsound_confident_solve(
    input: &str,
    req: &SynthesisRequirement,
    registry: &Registry,
) -> Option<String> {
    use linguigenesis_core::nl_tokens::tokenize_lower;

    // Inline-example requests carry the user's OWN demonstrated I/O pairs as the
    // spec (registry comprehension is bypassed by `apply_inline_examples`), so
    // there is no registry-misresolution to guard against — exempt them. Detected
    // structurally with the SAME parser the comprehension uses, not a phrase
    // heuristic. (The separate "is a single-canned-example registry op actually
    // proven against FRESH holdouts?" concern is P2's job, deliberately NOT folded
    // in here — this gate only catches confident-WRONG resolution.)
    if !linguigenesis_core::inline_examples::parse_inline_examples(input).is_empty() {
        return None;
    }

    let req_tokens: Vec<String> = tokenize_lower(input);
    let sig_lower = req.signature.to_lowercase();

    // (2) Request value-type vs resolved-signature value-type mismatch.
    // Map each request type-noun to the signature fragment that must be present.
    let type_mentions: &[(&[&str], &[&str])] = &[
        // request says "string"/"text"/"char" -> sig must carry a string type
        (&["string", "char", "text"], &["string", "str", "&str"]),
        // request says "array"/"list"/"vector" -> sig must carry an array/vec type
        (&["array", "list", "vector", "arrays", "lists"], &["[", "vec<"]),
    ];
    for (request_words, sig_needles) in type_mentions {
        let mentioned = req_tokens.iter().any(|t| request_words.contains(&t.as_str()));
        if mentioned {
            let satisfied = sig_needles.iter().any(|n| sig_lower.contains(n));
            if !satisfied {
                return Some(format!(
                    "no operation confidently resolved: request mentions a '{}' value but \
                     resolved op '{}' has signature '{}' (type mismatch)",
                    request_words[0], req.function_name, req.signature
                ));
            }
        }
    }

    // (3) Operation identity, via the RESOLVER (handles synonyms + morphology the
    // same way comprehension did — e.g. "absolute"->abs, "maximum"->array_max,
    // "combine"->add). For each request content word that resolves to an
    // operation: if it resolves to the SAME op the request was assigned, the op
    // was genuinely named; if it resolves to a DIFFERENT op, that op was silently
    // dropped (a compositional request like "sum of squares" -> array_sum dropping
    // "squares"->square) and we fail closed. If NO content word names the resolved
    // op, it was not actually understood. All registry-driven — no phrase list,
    // and `req.description` (the request echoed back) is deliberately NOT used as
    // op identity. Generic operand/value words are filtered first; they resolve to
    // no operation. Inline requests already returned above.
    const VALUE_NOUNS: &[&str] = &[
        "number", "numbers", "integer", "integers", "value", "values", "int",
        "ints", "array", "arrays", "list", "lists", "string", "strings", "char",
        "chars", "text", "element", "elements", "item", "items", "two", "three",
        "a", "an", "the", "of", "to", "from", "in", "on", "and", "or",
    ];
    let content_words: Vec<&String> = req_tokens
        .iter()
        .filter(|t| !VALUE_NOUNS.contains(&t.as_str()))
        .collect();
    if !content_words.is_empty() {
        let resolver = EntityResolver::new(registry.clone());
        let mut names_resolved_op = false;
        for word in &content_words {
            if let Some(resolved) = resolver.resolve_operation_surface(word) {
                // Only HIGH-CONFIDENCE resolution methods count as genuinely naming
                // an operation. Coincidental fuzzy-edit-distance ("file" ~ "filter")
                // and definition-overlap matches are exactly HOW the request got
                // MIS-resolved in the first place, so they must NOT be treated as
                // evidence the op was understood. direct/morphology/synonym/relation
                // are genuine; fuzzy_lemma/definition_overlap are not.
                let m = resolved.evidence.method;
                if m == "fuzzy_lemma" || m == "definition_overlap" {
                    continue;
                }
                let fname = resolved
                    .entity
                    .get_property("default_fn_name")
                    .cloned()
                    .unwrap_or_else(|| resolved.entity.lemma.clone());
                if fname == req.function_name {
                    names_resolved_op = true;
                } else {
                    return Some(format!(
                        "request also names operation '{}' (resolves to '{}'), dropped in favor \
                         of '{}' — compositional request not yet supported",
                        word, fname, req.function_name
                    ));
                }
            }
        }
        if !names_resolved_op {
            return Some(format!(
                "no operation confidently resolved: request content words {:?} do not name the \
                 resolved op '{}'",
                content_words, req.function_name
            ));
        }
    }

    None
}

fn synthesis_requirement_to_examples(
    req: &SynthesisRequirement,
) -> Result<Vec<Example>, BridgeError> {
    req.examples
        .iter()
        .map(|spec| {
            Ok(Example {
                inputs: spec
                    .inputs
                    .iter()
                    .map(literal_to_value)
                    .collect::<Result<Vec<_>, _>>()?,
                expected: literal_to_value(&spec.expected)?,
            })
        })
        .collect()
}

fn literal_to_value(lit: &LiteralValue) -> Result<Value, BridgeError> {
    Ok(match lit {
        LiteralValue::Int(v) => Value::Int(*v),
        LiteralValue::Float(v) => Value::Float(v.to_bits()),
        LiteralValue::Str(s) => Value::Str(s.clone()),
        LiteralValue::Bool(b) => Value::Bool(*b),
        LiteralValue::Array(a) => Value::int_array(a),
        LiteralValue::Pair(a, b) => Value::Pair(*a, *b),
    })
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
            Value::Array(_) => "[i64]",
            Value::Pair(_, _) => "(i64, i64)",
            Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
            Value::Tree(_) => "Tree",
            Value::Tuple(_) => "Tuple",
            Value::Struct(_) => "Struct",
        };
        param_idx += 1;
        param_types.push(format!("{}: {}", param_names(param_idx), type_str));
    }

    let return_type = match &first.expected {
        Value::Int(_) => "i64",
        Value::Float(_) => "f64",
        Value::Str(_) => "String",
        Value::Bool(_) => "bool",
        Value::Array(_) => "[i64]",
        Value::Pair(_, _) => "(i64, i64)",
        Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
        Value::Tree(_) => "Tree",
        Value::Tuple(_) => "Tuple",
        Value::Struct(_) => "Struct",
    };

    format!(
        "fn {}({}) -> {}",
        fn_name,
        param_types.join(", "),
        return_type
    )
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

    /// P0-NL ACCEPT (fail-closed): requests that the registry SILENTLY
    /// MIS-RESOLVES (returned a confident-wrong, strict-"verified" WRONG program
    /// before this gate) must now be REFUSED via ClarificationNeeded — caught by
    /// the STRUCTURAL signals (type-mismatch / operation-not-named), never a
    /// phrase blocklist. A confidently-wrong coding agent is worse than one that
    /// asks. Empirically these resolved to: sum-of-squares->add (array vs scalar
    /// sig + content words don't name 'add'), reverse-a-string->reverse(Vec)
    /// (string vs Vec sig), parse-a-CSV->filter (content words don't name
    /// 'filter').
    #[test]
    fn nl_failclosed_refuses_confident_wrong_resolution() {
        let bridge = LinguigenesisBridge::new();
        for phrase in [
            "return the sum of squares of an array",
            "reverse a string",
            "parse a CSV file",
        ] {
            match bridge.nl_to_requirement(phrase) {
                Err(BridgeError::ClarificationNeeded { .. }) => {}
                other => panic!(
                    "request {phrase:?} must fail closed (ClarificationNeeded), \
                     got {other:?} — confident-wrong resolution leaked"
                ),
            }
        }
    }

    /// P0-NL must NOT over-refuse: genuine in-vocab single-op requests whose
    /// content word actually names the resolved op, with no type mismatch, still
    /// resolve to a Requirement (and synthesize). Proves the gate keys on
    /// structural signals, not on "any registry match is suspect".
    #[test]
    fn nl_failclosed_keeps_genuine_in_vocab_ops() {
        let bridge = LinguigenesisBridge::new();
        // Resolve cleanly (guard does not fire).
        for phrase in ["add two numbers", "square a number"] {
            bridge.nl_to_requirement(phrase).unwrap_or_else(|e| {
                panic!("genuine request {phrase:?} must still resolve, got {e:?}")
            });
        }
        // And still synthesize end-to-end.
        let result = bridge
            .synthesize_from_description("add two numbers", Some("add"))
            .expect("genuine op must still synthesize");
        assert!(result.success, "add must still solve: {:?}", result.error);
    }

    /// P0-NL must EXEMPT inline-example requests: the user supplied the spec
    /// directly (comprehension bypassed), so the operation-not-named signal must
    /// not gate them even though the demonstrated op has no registry identity.
    #[test]
    fn nl_failclosed_exempts_inline_example_requests() {
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("a function mapping [1,2,3] -> [2,3,4] and [5,6] -> [6,7]")
            .expect("inline-example request must be exempt from the fail-closed gate");
        assert_eq!(req.examples.len(), 2);
    }

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
    fn test_nl_to_examples_map_double() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("map the array").unwrap();
        assert!(!examples.is_empty());
        assert_eq!(examples[0].expected, Value::int_array(&[2, 4, 6]));
    }

    #[test]
    fn test_synthesize_from_description_add() {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description("add two numbers", Some("add"))
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_nl_to_examples_reverse_array() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("reverse array").unwrap();
        assert!(!examples.is_empty());
        assert!(
            matches!(examples[0].expected, Value::Array(_)),
            "expected={:?}",
            examples[0].expected
        );
    }

    #[test]
    fn test_synthesize_from_description_reverse_array() {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description("reverse array", Some("reverse"))
            .unwrap();
        assert!(result.success, "failed: {:?}", result.error);
        assert!(
            result.method.contains("array_transform")
                || result.method.contains("search_")
                || result.code.contains("push"),
            "method={} code={}",
            result.method,
            result.code
        );
    }

    #[test]
    fn test_get_belief_state() {
        let bridge = LinguigenesisBridge::new();
        let belief = bridge.get_belief_state("reverse the array").unwrap();
        assert_eq!(
            belief.intent.intent_type,
            linguigenesis_core::belief::IntentType::DataTransformation
        );
    }

    #[test]
    fn test_nl_to_examples_combine() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("combine two numbers").unwrap();
        assert!(!examples.is_empty());
        assert_eq!(examples[0].expected, Value::Int(5));
    }

    #[test]
    fn test_clarification_on_gibberish() {
        let bridge = LinguigenesisBridge::new();
        let err = bridge
            .nl_to_requirement("xyzqwerty qwerty qwerty")
            .unwrap_err();
        assert!(matches!(err, BridgeError::ClarificationNeeded { .. }));
    }

    #[test]
    fn inline_examples_synthesize_unseen_operation() {
        // "quux" is not a registry operation. The agent should still synthesize
        // it purely from the demonstrated I/O examples (here: multiply by 10),
        // routing through the same typed solver as the verified benchmark.
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("implement quux(1)=10, quux(2)=20, quux(3)=30")
            .expect("inline examples should yield a ready requirement");
        assert_eq!(req.examples.len(), 3, "examples: {:?}", req.examples);
        assert_eq!(req.function_name, "quux");
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("synthesis call");
        assert!(result.success, "failed to synthesize unseen op: {:?}", result.error);
    }

    #[test]
    fn inline_examples_array_unseen_operation() {
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("a function mapping [1,2,3] -> [2,3,4] and [5,6] -> [6,7]")
            .expect("inline array examples ready");
        assert_eq!(req.examples.len(), 2);
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("synthesis call");
        assert!(result.success, "failed array synth: {:?}", result.error);
    }

    #[test]
    fn inline_examples_contradiction_asks_for_clarification() {
        // Same input, two different outputs describes no deterministic function.
        // The agent must flag the conflict and ask the user — never silently
        // emit a probabilistic sampler and call it a successful synthesis.
        let bridge = LinguigenesisBridge::new();
        match bridge.nl_to_requirement("define bad(1)=1, bad(1)=2") {
            Err(BridgeError::ClarificationNeeded { partial, questions }) => {
                assert!(
                    partial.unresolved.iter().any(|u| u.starts_with("conflicting examples")),
                    "unresolved={:?}",
                    partial.unresolved
                );
                assert!(!questions.is_empty());
            }
            other => panic!("expected ClarificationNeeded for conflicting examples, got {other:?}"),
        }
    }

    // ---- Newly-registered ops: NL phrase (NO inline examples) must resolve,
    // synthesize, AND generalize. Generalization is proven by executing the
    // synthesized program on HOLDOUT inputs absent from the registry
    // `example_cases`, so a green run rejects example overfit. ----

    fn ex(inputs: Vec<Value>, expected: Value) -> Example {
        Example { inputs, expected }
    }

    /// Resolve `phrase` from NL alone (no inline examples), synthesize, and
    /// verify the synthesized program against holdout inputs.
    fn assert_nl_synthesizes_and_generalizes(
        phrase: &str,
        fn_name: &str,
        signature: &'static str,
        holdouts: Vec<Example>,
    ) {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description(phrase, None)
            .unwrap_or_else(|e| panic!("{phrase:?}: NL did not resolve/synthesize: {e}"));
        assert!(
            result.success,
            "{phrase:?}: solver returned failure (method={}, err={:?})",
            result.method, result.error
        );
        let holdout_problem = Problem {
            name: fn_name.to_string(),
            category: "test",
            description: "holdout generalization",
            signature,
            examples: holdouts,
            ..Default::default()
        };
        crate::runtime::verify_problem_code(&holdout_problem, &result.code).unwrap_or_else(|e| {
            panic!(
                "{phrase:?}: OVERFIT — holdout generalization failed: {e}\nmethod={}\nCODE:\n{}",
                result.method, result.code
            )
        });
    }

    #[test]
    fn nl_abs_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the absolute value of a number",
            "abs",
            "fn abs(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(-50)], Value::Int(50)),
                ex(vec![Value::Int(42)], Value::Int(42)),
                ex(vec![Value::Int(-100)], Value::Int(100)),
                ex(vec![Value::Int(7)], Value::Int(7)),
                ex(vec![Value::Int(-6)], Value::Int(6)),
            ],
        );
    }

    #[test]
    fn nl_triple_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "triple a number",
            "triple",
            "fn triple(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(7)], Value::Int(21)),
                ex(vec![Value::Int(100)], Value::Int(300)),
                ex(vec![Value::Int(-50)], Value::Int(-150)),
                ex(vec![Value::Int(8)], Value::Int(24)),
            ],
        );
    }

    #[test]
    fn nl_square_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "square a number",
            "square",
            "fn square(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(7)], Value::Int(49)),
                ex(vec![Value::Int(10)], Value::Int(100)),
                ex(vec![Value::Int(1)], Value::Int(1)),
                ex(vec![Value::Int(9)], Value::Int(81)),
            ],
        );
    }

    #[test]
    fn nl_negate_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "negate a number",
            "negate",
            "fn negate(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(100)], Value::Int(-100)),
                ex(vec![Value::Int(-50)], Value::Int(50)),
                ex(vec![Value::Int(13)], Value::Int(-13)),
            ],
        );
    }

    #[test]
    fn nl_array_sum_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the total of an array",
            "array_sum",
            "fn array_sum(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 1])], Value::Int(101)),
                ex(vec![Value::int_array(&[4, 4, 4])], Value::Int(12)),
                ex(vec![Value::int_array(&[-1, -2, -3])], Value::Int(-6)),
                ex(vec![Value::int_array(&[5])], Value::Int(5)),
            ],
        );
    }

    #[test]
    fn nl_array_max_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the largest of an array",
            "array_max",
            "fn array_max(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 2, 50])], Value::Int(100)),
                ex(vec![Value::int_array(&[5, 5, 5])], Value::Int(5)),
                ex(vec![Value::int_array(&[1, 2, 3, 4])], Value::Int(4)),
            ],
        );
    }

    #[test]
    fn nl_array_min_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the smallest of an array",
            "array_min",
            "fn array_min(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 2, 50])], Value::Int(2)),
                ex(vec![Value::int_array(&[5, 5, 5])], Value::Int(5)),
                ex(vec![Value::int_array(&[4, 3, 2, 1])], Value::Int(1)),
            ],
        );
    }

    /// `sum3` is reachable as a proven registry op even though plain English has
    /// no single-token trigger distinct from 2-arg `add` (see roadmap deferral
    /// note). Drive it through the registry-seed requirement path and prove it
    /// generalizes on holdouts.
    #[test]
    fn registry_sum3_synthesizes_and_generalizes() {
        use linguigenesis_core::entity::EntityType;
        let bridge = LinguigenesisBridge::new();
        let registry = bridge.registry_clone().expect("registry clone");
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| e.lemma == "sum3")
            .expect("sum3 registered");
        let req = SynthesisRequirement::from_operation_entity(&entity).expect("sum3 requirement");
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("sum3 synthesis");
        assert!(result.success, "sum3 failed: {:?}", result.error);
        let holdout_problem = Problem {
            name: "sum3".to_string(),
            category: "test",
            description: "holdout",
            signature: "fn sum3(a: i64, b: i64, c: i64) -> i64",
            examples: vec![
                ex(vec![Value::Int(5), Value::Int(5), Value::Int(5)], Value::Int(15)),
                ex(vec![Value::Int(10), Value::Int(-3), Value::Int(1)], Value::Int(8)),
                ex(vec![Value::Int(0), Value::Int(0), Value::Int(7)], Value::Int(7)),
                ex(vec![Value::Int(100), Value::Int(1), Value::Int(1)], Value::Int(102)),
            ],
            ..Default::default()
        };
        crate::runtime::verify_problem_code(&holdout_problem, &result.code)
            .unwrap_or_else(|e| panic!("sum3 OVERFIT: {e}\nCODE:\n{}", result.code));
    }

    /// Integrity gate: every operation declared in the coding registry (i.e.
    /// every function entity carrying `example_cases`) must actually synthesize
    /// through the real solver. This prevents the registry from advertising
    /// vocabulary the engine cannot deliver — declared capability == proven
    /// capability, per the no-cheating contract.
    #[test]
    fn every_registry_operation_is_synthesizable() {
        use linguigenesis_core::entity::EntityType;

        let bridge = LinguigenesisBridge::new();
        let registry = bridge.registry_clone().expect("registry clone");

        let mut checked = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for entity in registry.get_by_type(&EntityType::Function) {
            let req = match SynthesisRequirement::from_operation_entity(&entity) {
                Some(r) => r,
                None => continue,
            };
            checked += 1;
            match bridge.synthesize_from_requirement(&req, Some(&req.function_name)) {
                Ok(result) if result.success => {}
                Ok(result) => failures.push(format!(
                    "{}: solver returned failure (method={}, err={:?})",
                    entity.lemma, result.method, result.error
                )),
                Err(e) => failures.push(format!("{}: {}", entity.lemma, e)),
            }
        }

        assert!(checked >= 20, "expected >=20 registry ops, checked {checked}");
        assert!(
            failures.is_empty(),
            "{}/{} registry ops failed to synthesize:\n{}",
            failures.len(),
            checked,
            failures.join("\n")
        );
    }
}
