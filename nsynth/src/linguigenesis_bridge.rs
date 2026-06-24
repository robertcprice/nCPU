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
        // COMPOSITIONAL path first: a request that emergently names a two-op
        // array pipeline is comprehensible even though the fail-closed gate would
        // downgrade it (the second op would look "dropped"). Build+verify it here.
        if let Some(outcome) = self.try_compose_pipeline(description) {
            return outcome.map(|o| o.into_solve_result());
        }
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

    /// COMPOSITIONAL front door: if the request emergently names a two-op
    /// array pipeline `reduce(map(arr))` (or its reduce-only/map-anchored
    /// degenerate), build that pipeline from the resolved primitives, synthesize
    /// it through the EXISTING solver, and strict-verify it on FRESH holdouts
    /// labelled by an independent reference composition. Returns:
    ///   * `Some(Ok(outcome))` — a pipeline was recognised AND accepted,
    ///   * `Some(Err(reason))` — a pipeline was recognised but could not be built
    ///     or did not strict-verify (fail honestly, do NOT silently fall back),
    ///   * `None` — the request is not a two-op pipeline (caller uses single-op).
    ///
    /// No phrase→plan table: the plan is `classify_pipeline` over
    /// `resolved_content_ops`, i.e. purely what each word EMERGENTLY resolves to.
    pub fn try_compose_pipeline(&self, description: &str) -> Option<Result<PipelineOutcome, String>> {
        // Inline-example requests are user-specified; never treat as a pipeline.
        if !linguigenesis_core::inline_examples::parse_inline_examples(description).is_empty() {
            return None;
        }
        let registry = match self.registry_clone() {
            Ok(r) => r,
            Err(e) => return Some(Err(e.to_string())),
        };
        let req = match self.requirement_for_pipeline(description, &registry) {
            Some(r) => r,
            None => return None,
        };
        let ops = resolved_content_ops(description, &registry);
        let plan = classify_pipeline(&req, &ops)?;
        Some(self.build_and_verify_pipeline(description, &plan))
    }

    /// Build a requirement we can hand to `classify_pipeline` even when the
    /// fail-closed gate would downgrade the request. We reuse the comprehension
    /// layer directly (bypassing the gate) so `req.function_name` reflects the
    /// op the registry actually assigned — that is the very thing the pipeline
    /// detector compares the second op against.
    fn requirement_for_pipeline(
        &self,
        description: &str,
        registry: &Registry,
    ) -> Option<SynthesisRequirement> {
        let mut coding = CodingComprehension::new(registry.clone());
        let req = coding.comprehend(description);
        if req.function_name.is_empty() {
            return None;
        }
        Some(req)
    }

    fn build_and_verify_pipeline(
        &self,
        description: &str,
        plan: &CompositionPlan,
    ) -> Result<PipelineOutcome, String> {
        // 1. Synthesize each primitive through the EXISTING solver from its
        //    registry example_cases. The reduce primitive defines the fold; the
        //    optional map primitive defines the element transform.
        let reduce_code = self.synthesize_primitive(&plan.reduce)?;
        let map_code = match &plan.map {
            Some(m) => Some((m.fn_name.clone(), self.synthesize_primitive(m)?)),
            None => None,
        };

        // 2. Classify the reduce fold by EXECUTING the synthesized reduce code on
        //    probe inputs (behaviour-driven; never keyed on the op's name). An
        //    ArrayReduce is probed with arrays; a BinaryFoldSeed (e.g. add) with
        //    scalar pairs — the shape comes from its emergent `op_role`.
        let fold = match op_role(&plan.reduce) {
            OpRole::ArrayReduce => classify_array_fold(&reduce_code, &plan.reduce.fn_name),
            OpRole::BinaryFoldSeed => classify_binary_fold(&reduce_code, &plan.reduce.fn_name),
            _ => None,
        }
        .ok_or_else(|| format!("could not classify fold for reduce op '{}'", plan.reduce.fn_name))?;

        // 3. Emit the composed REFERENCE: the map fn body (if any) plus a fused
        //    driver loop applying `fold` over `map(arr[i])`. This is an INDEPENDENT
        //    implementation of the pipeline, used only to LABEL fresh holdouts.
        let composed_name = pipeline_fn_name(plan);
        let reference = emit_pipeline_reference(&composed_name, fold, map_code.as_ref());
        let reference: &'static str = Box::leak(reference.into_boxed_str());
        let signature: &'static str =
            Box::leak(format!("fn {}(a: [i64]) -> i64", composed_name).into_boxed_str());

        // 4. Build a Problem whose reference IS the composition, so the existing
        //    strict verifier samples FRESH arrays and labels them by RUNNING the
        //    reference. Seed examples are likewise produced by the reference.
        let mut problem = crate::benchmark::problem_from_reference(
            &composed_name,
            signature,
            reference,
        )
        .map_err(|e| format!("pipeline reference unrunnable: {e}"))?;
        let category: &'static str = Box::leak("nl-compose".to_string().into_boxed_str());
        let descr: &'static str =
            Box::leak(format!("two-op pipeline for: {description}").into_boxed_str());
        problem.category = category;
        problem.description = descr;

        // 5. Synthesize the WHOLE pipeline through the existing solver from the
        //    seed examples (the array engine handles map/reduce per U5a/U5b).
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "two-op pipeline recognised ({}) but solver could not synthesize it (method={}, err={:?})",
                describe_plan(plan),
                solved.method,
                solved.error
            ));
        }

        // 6. STRICT verification on FRESH holdouts (reference-labelled): the solved
        //    program must match the independent composition on unseen arrays.
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!(
                "two-op pipeline OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}",
                solved.code
            )
        })?;

        Ok(PipelineOutcome {
            description: description.to_string(),
            fn_name: composed_name.clone(),
            map_fn: plan.map.as_ref().map(|m| m.fn_name.clone()),
            reduce_fn: plan.reduce.fn_name.clone(),
            fold,
            code: solved.code,
            method: format!("nl-compose-2op:{}", solved.method),
        })
    }

    /// Synthesize a single registry primitive (map or reduce op) through the
    /// existing solver, returning its verified code. The primitive is described
    /// by its registry entity's `example_cases`, so this is the same proven path
    /// `every_registry_operation_is_synthesizable` exercises.
    fn synthesize_primitive(&self, op: &ResolvedContentOp) -> Result<String, String> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| {
                e.get_property("default_fn_name")
                    .map(|f| f == &op.fn_name)
                    .unwrap_or(false)
                    || e.lemma == op.fn_name
            })
            .ok_or_else(|| format!("primitive '{}' not found in registry", op.fn_name))?;
        let req = SynthesisRequirement::from_operation_entity(&entity)
            .ok_or_else(|| format!("primitive '{}' is not synthesizable", op.fn_name))?;
        let result = self
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .map_err(|e| format!("primitive '{}' synthesis: {e}", op.fn_name))?;
        if !result.success {
            return Err(format!(
                "primitive '{}' did not synthesize (method={}, err={:?})",
                op.fn_name, result.method, result.error
            ));
        }
        Ok(result.code)
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

    // EMERGENT operation inventory: ask the RESOLVER what each surface token names,
    // and let the resolved ENTITY itself decide operand-vs-operation. There is no
    // VALUE_NOUNS wordlist, no type-noun→signature table, and no `evidence.method`
    // string switch. A token is an *operand* (number/value/array/the/of/...) exactly
    // when it does not resolve to a high-confidence programming operation; that is
    // observed, never enumerated. (`resolved_content_ops` filters by
    // `entity_type ∈ {Function, Operator}` and `evidence.score >= OP_RESOLVE_FLOOR`.)
    let ops = resolved_content_ops(input, registry);

    // If the request EMERGENTLY names a second operation that forms a
    // transform+aggregate (or reduce-only / map-only degenerate) array pipeline,
    // it is *comprehensible* — not unsound. Composition is built downstream
    // (`try_compose_pipeline`); here we simply decline to fail-closed so the
    // pipeline path can run. Single-op requests (one op, naming the resolved op)
    // fall through unchanged.
    if classify_pipeline(req, &ops).is_some() {
        return None;
    }

    let sig_lower = req.signature.to_lowercase();

    // (1b) ARRAY-DOMAIN vs SCALAR-OP mismatch, derived emergently from SIGNATURES.
    // If the resolved requirement op is SCALAR (its signature carries no array
    // type) yet some request token resolves to an ARRAY-domain operation (an op
    // whose declared `input_types` contains a vector type), the request is about
    // arrays but was collapsed onto a scalar op — a confidently-wrong resolution
    // (e.g. "sum of squares OF AN ARRAY" → `fn add(i64,i64)`, where the array
    // operation word was dropped). This compares the operand domain the request
    // names against the resolved op's domain; it is NOT a value-noun wordlist.
    // (A genuinely array-typed req op — array_sum, reverse — carries an array type
    // in its own signature, so this never fires for them.)
    let req_sig_is_array = sig_lower.contains('[') || sig_lower.contains("vec<");
    if !req_sig_is_array {
        if let Some(arr_word) = array_domain_word(input, registry, &req.function_name) {
            return Some(format!(
                "no operation confidently resolved: request names an array operand ('{}') but \
                 resolved op '{}' has scalar signature '{}' (domain mismatch)",
                arr_word, req.function_name, req.signature
            ));
        }
    }

    // (2) Request value-type vs resolved-signature value-type MISMATCH, derived
    // emergently: a token that resolves (high-confidence) to a registry *Type*
    // entity (e.g. "string", "array") asserts the operand's value type. If the
    // resolved op's signature carries none of that type's surface forms, the op
    // was applied to the wrong domain ("reverse a STRING" → `fn reverse(Vec<i64>)`).
    // The type's surface forms come from the registry entity, not a literal table.
    if let Some((type_word, needles)) = mentioned_value_type(input, registry) {
        let satisfied = needles.iter().any(|n| sig_lower.contains(n.as_str()));
        if !satisfied {
            return Some(format!(
                "no operation confidently resolved: request mentions a '{}' value but \
                 resolved op '{}' has signature '{}' (type mismatch)",
                type_word, req.function_name, req.signature
            ));
        }
    }

    // (3) Operation identity. For each emergently-resolved operation in the
    // request: if it IS the resolved op, the op was genuinely named. If a content
    // word resolves to a DIFFERENT op and the pair is NOT a buildable pipeline
    // (already returned above), the named op was silently dropped — fail closed.
    // If NO content word names the resolved op at all, it was not understood.
    if !ops.is_empty() {
        let mut names_resolved_op = false;
        for op in &ops {
            if op.fn_name == req.function_name {
                names_resolved_op = true;
            } else {
                return Some(format!(
                    "request also names operation '{}' (resolves to '{}'), dropped in favor \
                     of '{}' — compositional request not yet supported",
                    op.surface, op.fn_name, req.function_name
                ));
            }
        }
        if !names_resolved_op {
            let surfaces: Vec<&str> = ops.iter().map(|o| o.surface.as_str()).collect();
            return Some(format!(
                "no operation confidently resolved: request content words {:?} do not name the \
                 resolved op '{}'",
                surfaces, req.function_name
            ));
        }
    } else if !tokenize_lower(input).is_empty() {
        // Tokens present but NONE resolve to any operation (pure operands / gibberish):
        // there is no evidence the resolved op was named. Fail closed.
        return Some(format!(
            "no operation confidently resolved: no request token names the resolved op '{}'",
            req.function_name
        ));
    }

    None
}

/// Minimum `ResolutionEvidence.score` for a surface token to count as genuinely
/// naming a programming operation. Coincidental low-confidence links
/// (fuzzy edit-distance ~0.64, definition-overlap ~0.51) fall below this and are
/// treated as operands, exactly as the prior `evidence.method` blocklist intended
/// — but now via the resolver's own numeric confidence, not a method-name switch.
const OP_RESOLVE_FLOOR: f32 = 0.80;

/// A content word that EMERGENTLY resolved to a programming operation.
#[derive(Clone, Debug)]
struct ResolvedContentOp {
    /// The surface token as it appeared in the request.
    surface: String,
    /// The op's canonical function name (`default_fn_name` or lemma).
    fn_name: String,
    /// Declared arity from the registry entity, if any.
    arity: Option<u32>,
    /// `input_types` property (e.g. "i64", "i64,i64", "Vec<i64>"), lowercased.
    input_types: String,
    /// `output_type` property, lowercased.
    output_type: String,
}

/// Resolve every request token to a high-confidence programming operation, in
/// request order, de-duplicated by function name. A token is an *operand* (and
/// silently dropped here) precisely when it does NOT resolve to a
/// Function/Operator entity at or above [`OP_RESOLVE_FLOOR`] — operand-vs-operation
/// is decided by the resolver + entity type, never a wordlist.
fn resolved_content_ops(input: &str, registry: &Registry) -> Vec<ResolvedContentOp> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    let mut ops: Vec<ResolvedContentOp> = Vec::new();
    for tok in tokenize_lower(input) {
        let Some(resolved) = resolver.resolve_operation_surface(&tok) else {
            continue;
        };
        if resolved.evidence.score < OP_RESOLVE_FLOOR {
            continue;
        }
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            continue;
        }
        let fn_name = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        let arity = resolved
            .entity
            .get_property("arity")
            .and_then(|s| s.parse::<u32>().ok());
        let input_types = resolved
            .entity
            .get_property("input_types")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        let output_type = resolved
            .entity
            .get_property("output_type")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        if ops.iter().any(|o| o.fn_name == fn_name) {
            continue;
        }
        ops.push(ResolvedContentOp {
            surface: tok,
            fn_name,
            arity,
            input_types,
            output_type,
        });
    }
    ops
}

/// Structural role of an emergently-resolved op, inferred from its arity +
/// declared `input_types`/`output_type` — NOT its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpRole {
    /// Arity-1 scalar transform `i64 -> i64` (square, abs, negate, increment, ...).
    ScalarMap,
    /// Arity-1 array aggregate `Vec<i64> -> i64` (array_sum, array_max, array_min).
    ArrayReduce,
    /// Arity-2 scalar binary `(i64, i64) -> i64` that seeds a fold (add → sum-fold,
    /// multiply → product-fold).
    BinaryFoldSeed,
    /// Anything else (not part of the supported transform+aggregate shape).
    Other,
}

fn op_role(op: &ResolvedContentOp) -> OpRole {
    let out_scalar = op.output_type == "i64";
    match op.arity {
        Some(1) if op.input_types == "i64" && out_scalar => OpRole::ScalarMap,
        Some(1) if op.input_types.contains("vec") && out_scalar => OpRole::ArrayReduce,
        Some(2) if op.input_types == "i64,i64" && out_scalar => OpRole::BinaryFoldSeed,
        _ => OpRole::Other,
    }
}

/// A buildable array pipeline derived purely from the emergently-resolved ops.
#[derive(Clone, Debug)]
struct CompositionPlan {
    /// Optional element-wise map op (`fn map(a:i64)->i64`). `None` ⇒ reduce-only.
    map: Option<ResolvedContentOp>,
    /// The aggregate over the (possibly-mapped) array.
    reduce: ResolvedContentOp,
}

/// Decide whether the resolved-op set forms a supported array pipeline:
///   * exactly one reduce (ArrayReduce or BinaryFoldSeed) + at most one ScalarMap,
///   * and the request is NOT a plain single-op request (which the normal path
///     already handles — that case returns `None`).
///
/// Returns the plan when the shape is `reduce(map(arr))`, `reduce(arr)` (reduce-only
/// where the request *also* names a non-array reduce on an implicit array, e.g.
/// "sum of the values"), or rejects (`None`). All from op ROLES, no phrase table.
fn classify_pipeline(req: &SynthesisRequirement, ops: &[ResolvedContentOp]) -> Option<CompositionPlan> {
    let mut maps: Vec<&ResolvedContentOp> = Vec::new();
    let mut reduces: Vec<&ResolvedContentOp> = Vec::new();
    let mut other = false;
    for op in ops {
        match op_role(op) {
            OpRole::ScalarMap => maps.push(op),
            OpRole::ArrayReduce | OpRole::BinaryFoldSeed => reduces.push(op),
            OpRole::Other => other = true,
        }
    }
    // Supported shape: exactly one reduce + at most one map. Anything richer
    // (3-op, two maps, an unclassifiable op) is left to a documented follow-on.
    if other || reduces.len() != 1 || maps.len() > 1 {
        return None;
    }
    let reduce = reduces[0].clone();
    let map = maps.first().map(|m| (*m).clone());

    // A *single* op that is itself the resolved requirement op is NOT a pipeline —
    // it is the ordinary single-op request, handled unchanged by the normal path.
    if map.is_none() && reduce.fn_name == req.function_name {
        // Reduce-only AND the reduce IS the requirement op: this is just the plain
        // op (e.g. "compute the total of an array" → array_sum). Not compositional.
        return None;
    }
    // A bare scalar map with no reduce is not an array pipeline either; require a
    // reduce to anchor the aggregate. (Pure "square a number" never reaches here:
    // it has a ScalarMap but no reduce, so `reduces.len() != 1` already returned.)
    Some(CompositionPlan { map, reduce })
}

/// EMERGENT value-type mention: find the first request token that resolves
/// (high-confidence) to a registry **Type** entity, and return that type's
/// signature surface forms (its lemma + synonyms) so the gate can check the
/// resolved op's signature actually carries the domain. No literal type-noun table.
fn mentioned_value_type(input: &str, registry: &Registry) -> Option<(String, Vec<String>)> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    for tok in tokenize_lower(input) {
        let Some(resolved) = resolver.resolve_surface(&tok) else {
            continue;
        };
        if resolved.entity.entity_type != EntityType::Type {
            continue;
        }
        if resolved.evidence.score < OP_RESOLVE_FLOOR {
            continue;
        }
        // The type's accepted signature surface forms: its lemma plus any declared
        // `signature_aliases` property (comma-separated) — emergent from the entity.
        let mut needles: Vec<String> = vec![resolved.entity.lemma.to_lowercase()];
        if let Some(aliases) = resolved.entity.get_property("signature_aliases") {
            for a in aliases.split(',') {
                let a = a.trim().to_lowercase();
                if !a.is_empty() {
                    needles.push(a);
                }
            }
        }
        return Some((tok, needles));
    }
    None
}

/// Minimal resolution score for a token to count as *evidence of the array
/// domain*. This is intentionally BELOW [`OP_RESOLVE_FLOOR`]: an array-context
/// word like "array" links to an array op only weakly (definition-overlap), yet
/// its mere presence — against a SCALAR resolved op — is a real domain signal.
/// Pure non-resolving noise (score 0 / no match) never reaches here.
const ARRAY_DOMAIN_FLOOR: f32 = 0.50;

/// Find the first request token that resolves to an ARRAY-domain operation (an op
/// whose declared `input_types` contains a vector type) other than `req_fn`.
/// Returns the surface word so the gate can report the domain mismatch. Emergent:
/// the operand domain is read from the resolved entity's signature, not a list.
fn array_domain_word(input: &str, registry: &Registry, req_fn: &str) -> Option<String> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    for tok in tokenize_lower(input) {
        let Some(resolved) = resolver.resolve_operation_surface(&tok) else {
            continue;
        };
        if resolved.evidence.score < ARRAY_DOMAIN_FLOOR {
            continue;
        }
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            continue;
        }
        let in_types = resolved
            .entity
            .get_property("input_types")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        if !in_types.contains("vec") {
            continue;
        }
        let fname = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        if fname != req_fn {
            return Some(tok);
        }
    }
    None
}

/// The associative fold a reduce primitive computes, determined by EXECUTING the
/// synthesized reduce code on probe arrays — never inferred from the op's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldKind {
    /// `acc = acc + x`, identity 0.
    Sum,
    /// `acc = acc * x`, identity 1.
    Product,
    /// running maximum (seeded with the first element).
    Max,
    /// running minimum (seeded with the first element).
    Min,
}

/// Accepted, strict-verified two-op pipeline result.
#[derive(Clone, Debug)]
pub struct PipelineOutcome {
    /// The original NL request.
    pub description: String,
    /// The synthesized pipeline's own function name (call this to run `code`).
    pub fn_name: String,
    /// Element map op function name, if the pipeline has a map stage.
    pub map_fn: Option<String>,
    /// Aggregate (reduce) op function name.
    pub reduce_fn: String,
    /// The fold kind the reduce computes (behaviour-classified).
    pub fold: FoldKind,
    /// The solver-synthesized program for the whole pipeline.
    pub code: String,
    /// `nl-compose-2op:<inner-solver-method>`.
    pub method: String,
}

impl PipelineOutcome {
    /// True iff this is a genuine TWO-stage pipeline (map + reduce), not a
    /// degenerate reduce-only match. Used by the accept-test to reject a
    /// coincidental single-op solve masquerading as a composition.
    pub fn is_two_stage(&self) -> bool {
        self.map_fn.is_some()
    }

    fn into_solve_result(self) -> crate::solver::SolveResult {
        crate::solver::SolveResult {
            success: true,
            code: self.code,
            method: self.method,
            error: None,
            metadata: Default::default(),
        }
    }
}

/// Deterministic function name for a pipeline, so the strict verifier's holdout
/// seed (which hashes `problem.name`) is stable.
fn pipeline_fn_name(plan: &CompositionPlan) -> String {
    match &plan.map {
        Some(m) => format!("compose_{}_{}", plan.reduce.fn_name, m.fn_name),
        None => format!("compose_{}", plan.reduce.fn_name),
    }
}

fn describe_plan(plan: &CompositionPlan) -> String {
    match &plan.map {
        Some(m) => format!("reduce={} ∘ map={}", plan.reduce.fn_name, m.fn_name),
        None => format!("reduce={} (reduce-only)", plan.reduce.fn_name),
    }
}

/// Classify an ARRAY-reduce primitive's fold by running it on probe arrays and
/// matching the output against each candidate fold's known result. Behaviour-
/// driven: the op's NAME is only a label in error messages.
fn classify_array_fold(reduce_code: &str, reduce_fn: &str) -> Option<FoldKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "reduce_probe".to_string(),
        category: "probe",
        description: "fold classification",
        signature: "fn f(a: [i64]) -> i64",
        examples: vec![],
        ..Default::default()
    };
    // Distinct probes so the four folds give DIFFERENT answers (no collision).
    let probes: &[&[i64]] = &[&[3, 1, 2], &[2, 5, 4], &[2, 3, 4]];
    let run = |arr: &[i64]| -> Option<i64> {
        match crate::runtime::execute_function_for_problem(
            reduce_code,
            reduce_fn,
            &[Value::int_array(arr)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Int(v)) => Some(v),
            _ => None,
        }
    };
    let candidates = [FoldKind::Sum, FoldKind::Product, FoldKind::Max, FoldKind::Min];
    for cand in candidates {
        let mut all_match = true;
        for p in probes {
            let expected = match cand {
                FoldKind::Sum => p.iter().sum::<i64>(),
                FoldKind::Product => p.iter().product::<i64>(),
                FoldKind::Max => *p.iter().max().unwrap(),
                FoldKind::Min => *p.iter().min().unwrap(),
            };
            if run(p) != Some(expected) {
                all_match = false;
                break;
            }
        }
        if all_match {
            return Some(cand);
        }
    }
    None
}

/// Classify a BINARY scalar reduce-seed (e.g. `add`/`multiply`) by running it on
/// scalar PROBE PAIRS to learn its combiner — `a+b` ⇒ Sum-fold, `a*b` ⇒
/// Product-fold. Behaviour-driven (the op is executed), never name-keyed. A
/// binary op whose behaviour is neither addition nor multiplication is not a
/// supported fold seed (returns `None`).
fn classify_binary_fold(reduce_code: &str, reduce_fn: &str) -> Option<FoldKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "binop_probe".to_string(),
        category: "probe",
        description: "binary fold classification",
        signature: "fn f(a: i64, b: i64) -> i64",
        examples: vec![],
        ..Default::default()
    };
    let run = |a: i64, b: i64| -> Option<i64> {
        match crate::runtime::execute_function_for_problem(
            reduce_code,
            reduce_fn,
            &[Value::Int(a), Value::Int(b)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Int(v)) => Some(v),
            _ => None,
        }
    };
    // Probe pairs chosen so + and * disagree on every pair.
    let pairs = [(2, 3), (4, 5), (1, 7), (3, 6)];
    let is_sum = pairs.iter().all(|&(a, b)| run(a, b) == Some(a + b));
    if is_sum {
        return Some(FoldKind::Sum);
    }
    let is_prod = pairs.iter().all(|&(a, b)| run(a, b) == Some(a * b));
    if is_prod {
        return Some(FoldKind::Product);
    }
    None
}

/// Emit an INDEPENDENT reference implementation of the pipeline in the runtime
/// DSL: the optional map fn body verbatim, plus a driver that folds `map(arr[i])`
/// (or `arr[i]` when there is no map) with the classified combiner. This is used
/// only to LABEL fresh holdouts; the accepted program is what the solver finds.
fn emit_pipeline_reference(
    composed_name: &str,
    fold: FoldKind,
    map: Option<&(String, String)>,
) -> String {
    let mut out = String::new();
    // Map fn body (verified single-op synthesis) goes first so the driver can call it.
    let elem = if let Some((map_fn, map_code)) = map {
        out.push_str(map_code);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        format!("{}(arr[i])", map_fn)
    } else {
        "arr[i]".to_string()
    };

    match fold {
        FoldKind::Sum | FoldKind::Product => {
            let (init, op) = match fold {
                FoldKind::Sum => (0, "+"),
                FoldKind::Product => (1, "*"),
                _ => unreachable!(),
            };
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> i64 {{\n    \
                 acc: i64 = {init};\n    \
                 i: i64 = 0;\n    \
                 while i < arr.len {{\n        \
                 acc = acc {op} {elem};\n        \
                 i = i + 1;\n    }}\n    \
                 return acc;\n}}\n",
                name = composed_name,
                init = init,
                op = op,
                elem = elem,
            ));
        }
        FoldKind::Max | FoldKind::Min => {
            let cmp = if fold == FoldKind::Max { ">" } else { "<" };
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> i64 {{\n    \
                 i: i64 = 0;\n    \
                 acc: i64 = {elem};\n    \
                 i = 1;\n    \
                 while i < arr.len {{\n        \
                 v: i64 = {elem};\n        \
                 if v {cmp} acc {{ acc = v; }}\n        \
                 i = i + 1;\n    }}\n    \
                 return acc;\n}}\n",
                name = composed_name,
                elem = elem,
                cmp = cmp,
            ));
        }
    }
    out
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
    /// before this gate) must be REFUSED via ClarificationNeeded — caught by the
    /// EMERGENT STRUCTURAL signals (array-domain-vs-scalar-op mismatch derived from
    /// signatures, operation-not-named), never a phrase blocklist and never a
    /// value-noun / type-noun wordlist. A confidently-wrong coding agent is worse
    /// than one that asks. Empirically:
    ///   * "...sum of squares OF AN ARRAY" → `add` (scalar sig): the array operand
    ///     the request names ("array" resolves to an array-input op) does not match
    ///     the scalar resolved op → DOMAIN MISMATCH → refuse.
    ///   * "parse a CSV file" → no request token names the resolved op at all (the
    ///     content words resolve to nothing above the confidence floor) → refuse.
    #[test]
    fn nl_failclosed_refuses_confident_wrong_resolution() {
        let bridge = LinguigenesisBridge::new();
        for phrase in [
            "return the sum of squares of an array",
            "parse a CSV file",
        ] {
            match bridge.nl_to_requirement(phrase) {
                Err(BridgeError::ClarificationNeeded { .. }) => {}
                Err(BridgeError::ParseError(_)) => {}
                other => panic!(
                    "request {phrase:?} must fail closed (ClarificationNeeded/ParseError), \
                     got {other:?} — confident-wrong resolution leaked"
                ),
            }
        }
    }

    /// DOCUMENTED type-case (Bucket B of the unseen-phrasing benchmark): "reverse a
    /// string" may PASS or REFUSE. The coding registry currently has no `Type`
    /// entity for "string", so — by design, per the EMERGENT-only rule — the bridge
    /// cannot detect a string/Vec type mismatch without a value-type wordlist (which
    /// the global guardrail forbids). It therefore resolves "reverse" to the array
    /// `reverse`. This is acceptable (a benign over-acceptance, not a confidently-
    /// WRONG numeric answer); tightening it requires adding a `Type` entity to
    /// linguigenesis-core (a documented follow-on), NOT a phrase/type blocklist here.
    #[test]
    fn nl_reverse_a_string_is_a_documented_type_case() {
        let bridge = LinguigenesisBridge::new();
        // Whatever the outcome, it must NOT be a non-refusal that ALSO produces a
        // composed-pipeline mis-solve; the type case stays a plain single-op resolve.
        assert!(
            bridge.try_compose_pipeline("reverse a string").is_none(),
            "'reverse a string' must not be mis-built as a numeric pipeline"
        );
        // It is allowed to resolve (single-op reverse) OR refuse; both are fine.
        let _ = bridge.nl_to_requirement("reverse a string");
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
