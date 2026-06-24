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
    coding_requirements::{
        CompositionPlan, LiteralValue, OpRef, OpRole, SynthesisRequirement,
    },
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

    /// TEST-ONLY constructor: identical to [`new`] but WITHOUT the WordNet
    /// coding-edges file merged. Used by the recall-lift benchmark to compute the
    /// registry-only baseline so the lift is attributable solely to the edges.
    /// (Not env-driven, to stay race-free under parallel `cargo test`.)
    pub fn new_without_wordnet_edges() -> Self {
        let linguigenesis_path = Self::find_registry_path();
        let (registry, modified) = if let Some(path) = &linguigenesis_path {
            Self::load_registry_with_fallback_opt(path, false)
        } else {
            Self::load_registry_with_fallback_opt(Path::new(""), false)
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

    /// Load registry with fallback to code entities if file not found
    fn load_registry_with_fallback(path: &Path) -> (Registry, Option<SystemTime>) {
        Self::load_registry_with_fallback_opt(path, true)
    }

    fn load_registry_with_fallback_opt(
        path: &Path,
        include_wordnet: bool,
    ) -> (Registry, Option<SystemTime>) {
        if !path.as_os_str().is_empty() && path.exists() {
            match Registry::from_json_auto(path) {
                Ok((mut registry, modified)) => {
                    eprintln!(
                        "[Linguigenesis] Loaded {} entities from {}",
                        registry.stats().total_entities,
                        path.display()
                    );
                    Self::merge_coding_registry_opt(&mut registry, include_wordnet);
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
        Self::merge_coding_registry_opt(&mut registry, include_wordnet);
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

    /// Find the WordNet coding-edges file (additive synonym/similar closure for
    /// existing ops). Mirrors [`find_coding_registry_path`] path probing.
    fn find_wordnet_edges_path() -> Option<PathBuf> {
        let relative = PathBuf::from("../../linguigenesis/data/wordnet_coding_edges.json");
        if relative.exists() {
            return Some(relative);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_path =
                PathBuf::from(home).join("projects/linguigenesis/data/wordnet_coding_edges.json");
            if home_path.exists() {
                return Some(home_path);
            }
        }
        let current = PathBuf::from("linguigenesis/data/wordnet_coding_edges.json");
        if current.exists() {
            return Some(current);
        }
        None
    }

    fn merge_coding_registry(registry: &mut Registry) {
        Self::merge_coding_registry_opt(registry, true);
    }

    /// Collision-safe merge of the WordNet edge file into an ALREADY-populated
    /// registry. Each NEW closure word is re-added with a FRESH high id (base +
    /// 1000+, mirroring `merge_computing_knowledge`) so it never overwrites a
    /// like-id coding entity; its `synonym`/`similar` relations are then re-linked
    /// BY LEMMA against the real (example-bearing) seed entity in `registry`.
    /// Entities whose lemma already exists (the co-declared seed stubs, or any word
    /// the hand-table already owns) are left untouched — purely additive.
    fn merge_wordnet_edges_collision_safe(registry: &mut Registry, edges: &Registry) {
        use linguigenesis_core::entity::{Entity, RelationType};
        let mut next_id = registry.stats().total_entities as u64 + 1000;
        // (source_lemma, rel_type, target_lemma) links to establish after adding.
        let mut pending: Vec<(String, RelationType, String)> = Vec::new();

        for src in edges.all_entities() {
            // Skip anything already present (seed stubs, hand-table words): additive.
            if registry.get_by_lemma(&src.lemma).is_some() {
                continue;
            }
            // Re-create the entity with a fresh, non-colliding id. Carry over the
            // type, definitions, and properties (incl. wn_synset provenance) but
            // DROP the edge-file-local relation ids — they are re-linked by lemma
            // below so they bind to the real seed, never a stale numeric target.
            let mut entity = Entity::new(next_id, src.lemma.clone(), src.entity_type.clone());
            next_id += 1;
            for def in &src.definitions {
                entity.add_definition(def.clone());
            }
            for (k, v) in &src.properties {
                entity.add_property(k.clone(), v.clone());
            }
            if let Err(e) = registry.add_entity(entity) {
                eprintln!("[Linguigenesis] wordnet edge add warning ({}): {}", src.lemma, e);
                continue;
            }
            // Queue this word's relations to its seed target(s), resolved by lemma
            // within the EDGE registry (so the underscore-free seed lemma is known).
            for (rel_type, target_ids) in &src.relations {
                for tid in target_ids {
                    if let Some(target) = edges.get_entity(*tid) {
                        pending.push((src.lemma.clone(), rel_type.clone(), target.lemma.clone()));
                    }
                }
            }
        }

        for (source, rel_type, target) in pending {
            // Links only succeed when BOTH lemmas exist in `registry`; the target
            // is the real seed op (present from coding_registry), so the closure
            // word now resolves through a genuine synonym/similar edge to the seed.
            let _ = registry.link_lemma_relation(&source, rel_type, &target);
        }
    }

    fn merge_coding_registry_opt(registry: &mut Registry, include_wordnet: bool) {
        if let Some(path) = Self::find_coding_registry_path() {
            if let Ok(coding) = Registry::from_json(&path) {
                if let Err(e) = registry.merge_registry(&coding) {
                    eprintln!("[Linguigenesis] coding_registry merge warning: {}", e);
                }
            }
        }
        // 3rd merge: WordNet synonym/similar closure edges for EXISTING ops. Must
        // run AFTER coding_registry so each seed op already exists in the merged
        // registry — the closure word's synonym->seed edge then links to the real,
        // example-bearing seed entity.
        //
        // COLLISION-SAFE merge (NOT `merge_registry`): `Registry::from_json`
        // numbers the edge file's entities 1..N, and `merge_registry` PRESERVES
        // those low ids when re-adding, which OVERWRITES the like-id coding_registry
        // entities in the id-keyed map (corrupting e.g. `add`). We instead allocate
        // FRESH high ids — exactly as `merge_computing_knowledge` does (base +
        // 1000) — so the closure words never collide. Seed stubs in the edge file
        // already exist (skipped); their only purpose is to be the `synonym`
        // targets, which we re-link by lemma against the real seed.
        if include_wordnet {
            if let Some(path) = Self::find_wordnet_edges_path() {
                match Registry::from_json(&path) {
                    Ok(edges) => {
                        Self::merge_wordnet_edges_collision_safe(registry, &edges);
                    }
                    Err(e) => {
                        eprintln!("[Linguigenesis] wordnet_coding_edges load warning: {}", e);
                    }
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
    /// No phrase→plan table: linguigenesis-core's `comprehend` EMITS the plan in
    /// `req.pipeline` (see `coding_requirements::classify_pipeline`). The bridge no
    /// longer derives it — it only EXECUTES the comprehended plan.
    pub fn try_compose_pipeline(&self, description: &str) -> Option<Result<PipelineOutcome, String>> {
        let registry = match self.registry_clone() {
            Ok(r) => r,
            Err(e) => return Some(Err(e.to_string())),
        };
        // Comprehend the request; linguigenesis-core decides — purely from registry
        // signatures — whether it is a composable pipeline and, if so, populates
        // `req.pipeline`. Single-op (and inline-example) requests come back with
        // `pipeline = None`, so the caller falls through to the single-op door.
        let mut coding = CodingComprehension::new(registry);
        let req = coding.comprehend(description);
        let plan = req.pipeline.clone()?;
        Some(self.build_and_verify_pipeline(description, &plan))
    }

    fn build_and_verify_pipeline(
        &self,
        description: &str,
        plan: &CompositionPlan,
    ) -> Result<PipelineOutcome, String> {
        // 1. Synthesize each primitive through the EXISTING solver from its
        //    registry example_cases. Each map op in the CHAIN defines one element
        //    transform; the optional reduce primitive defines the fold.
        //    `map_chain` is in REQUEST ORDER (outer→inner).
        let mut map_chain: Vec<(String, String)> = Vec::with_capacity(plan.maps.len());
        for m in &plan.maps {
            map_chain.push((m.fn_name.clone(), self.synthesize_primitive(m)?));
        }

        // 2. Classify the reduce fold (if any) by EXECUTING the synthesized reduce
        //    code on probe inputs (behaviour-driven; never keyed on the op's name).
        //    An ArrayReduce is probed with arrays; a BinaryFoldSeed (e.g. add) with
        //    scalar pairs — the shape comes from its emergent `op_role`.
        let fold = match &plan.reduce {
            Some(r) => {
                let reduce_code = self.synthesize_primitive(r)?;
                let fk = match r.role {
                    OpRole::ArrayReduce => classify_array_fold(&reduce_code, &r.fn_name),
                    OpRole::BinaryFoldSeed => classify_binary_fold(&reduce_code, &r.fn_name),
                    _ => None,
                }
                .ok_or_else(|| {
                    format!("could not classify fold for reduce op '{}'", r.fn_name)
                })?;
                Some(fk)
            }
            None => None,
        };

        // 2b. Classify the array transform (if any) behaviourally — never by name.
        //     First try EXECUTING the synthesized transform code on probe arrays;
        //     if that is inconclusive (a single-example registry op can be
        //     affine-overfit by the solver — e.g. reverse's lone `[1,2,3]->[3,2,1]`
        //     fits `y=-x+4`), fall back to matching the op's REGISTRY example_cases
        //     (its verified output spec) against the candidate transforms. Both
        //     paths are output-grounded, not name-keyed.
        let array_xfm = match &plan.array_transform {
            Some(t) => {
                let by_exec = self
                    .synthesize_primitive(t)
                    .ok()
                    .and_then(|code| classify_array_transform_by_exec(&code, &t.fn_name));
                let kind = match by_exec {
                    Some(k) => k,
                    None => self.classify_array_transform_by_spec(t).ok_or_else(|| {
                        format!(
                            "could not classify array transform '{}' as sort/reverse \
                             (neither execution nor registry example_cases matched)",
                            t.fn_name
                        )
                    })?,
                };
                Some(kind)
            }
            None => None,
        };

        // 3. Emit the composed REFERENCE: the map fn bodies (chained), then the
        //    optional array transform on the built array, then either a fused fold
        //    driver (shape a, scalar out) or the array itself (shape b, array out).
        //    This is an INDEPENDENT implementation of the pipeline, used only to
        //    LABEL fresh holdouts.
        let composed_name = pipeline_fn_name(plan);
        let reference = emit_pipeline_reference(&composed_name, fold, array_xfm, &map_chain);
        let reference: &'static str = Box::leak(reference.into_boxed_str());
        // Scalar output when a reduce is present; array output for a pure map chain.
        let ret = if fold.is_some() { "i64" } else { "[i64]" };
        let signature: &'static str =
            Box::leak(format!("fn {}(a: [i64]) -> {}", composed_name, ret).into_boxed_str());

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
            Box::leak(format!("transform-chain pipeline for: {description}").into_boxed_str());
        problem.category = category;
        problem.description = descr;

        // 5. Synthesize the WHOLE pipeline through the existing solver from the
        //    seed examples (the array engine handles map/reduce per U5a/U5b and
        //    array→array transforms per array_transform).
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "transform-chain pipeline recognised ({}) but solver could not synthesize it (method={}, err={:?})",
                describe_plan(plan),
                solved.method,
                solved.error
            ));
        }

        // 6. STRICT verification on FRESH holdouts (reference-labelled): the solved
        //    program must match the independent composition on unseen arrays.
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!(
                "transform-chain pipeline OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}",
                solved.code
            )
        })?;

        Ok(PipelineOutcome {
            description: description.to_string(),
            fn_name: composed_name.clone(),
            map_fns: plan.maps.iter().map(|m| m.fn_name.clone()).collect(),
            array_xfm_fn: plan.array_transform.as_ref().map(|t| t.fn_name.clone()),
            array_xfm,
            reduce_fn: plan.reduce.as_ref().map(|r| r.fn_name.clone()),
            fold,
            code: solved.code,
            method: format!("nl-compose-chain:{}", solved.method),
        })
    }

    /// Synthesize a single registry primitive (map or reduce op) through the
    /// existing solver, returning its verified code. The primitive is described
    /// by its registry entity's `example_cases`, so this is the same proven path
    /// `every_registry_operation_is_synthesizable` exercises.
    fn synthesize_primitive(&self, op: &OpRef) -> Result<String, String> {
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

    /// Classify an ArrayTransform op by matching its REGISTRY `example_cases` (the
    /// op's verified output spec) against each candidate transform. This is the
    /// output-grounded fallback for when the synthesized primitive is an affine
    /// overfit of a single example (so executing it is inconclusive). It reads the
    /// op's labelled (input-array → output-array) pairs from the registry and asks
    /// which of {sort, reverse} reproduces EVERY pair — never the op's name.
    fn classify_array_transform_by_spec(&self, op: &OpRef) -> Option<ArrayTransformKind> {
        use linguigenesis_core::coding_requirements::LiteralValue;
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().ok()?;
        let entity = registry.get_by_type(&EntityType::Function).into_iter().find(|e| {
            e.get_property("default_fn_name")
                .map(|f| f == &op.fn_name)
                .unwrap_or(false)
                || e.lemma == op.fn_name
        })?;
        let req = SynthesisRequirement::from_operation_entity(&entity)?;
        // Collect (input array, output array) pairs from the op's example_cases.
        let mut pairs: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
        for spec in &req.examples {
            let (Some(LiteralValue::Array(inp)), LiteralValue::Array(out)) =
                (spec.inputs.first(), &spec.expected)
            else {
                return None; // not an array→array op spec
            };
            pairs.push((inp.clone(), out.clone()));
        }
        if pairs.is_empty() {
            return None;
        }
        for cand in [ArrayTransformKind::Sort, ArrayTransformKind::Reverse] {
            let all = pairs.iter().all(|(inp, out)| {
                let mut expected = inp.clone();
                match cand {
                    ArrayTransformKind::Sort => expected.sort(),
                    ArrayTransformKind::Reverse => expected.reverse(),
                }
                &expected == out
            });
            if all {
                return Some(cand);
            }
        }
        None
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

    /// TEST-SUPPORT: resolve a single surface word to its highest-confidence
    /// programming op via the SAME emergent resolver the gate uses
    /// ([`resolved_content_ops`]). Returns `(default_fn_name, score)` for the
    /// top op, or `None` if the word resolves to no Function/Operator op.
    /// Used by the WordNet-recall benchmark's resolution/lift/refuse probes.
    pub fn resolve_op_probe(&self, word: &str) -> Option<(String, f32)> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().ok()?;
        let resolver = EntityResolver::new(registry);
        let resolved = resolver.resolve_operation_surface(word)?;
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            return None;
        }
        let fn_name = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        Some((fn_name, resolved.evidence.score))
    }

    /// TEST-SUPPORT: every DISTINCT op (`default_fn_name`) a word resolves to at
    /// or above `floor`, across ALL ranked candidates — used to assert
    /// cluster-separation (a paraphrase must not resolve to two different ops).
    pub fn resolve_ops_above_floor(&self, word: &str, floor: f32) -> Vec<String> {
        use linguigenesis_core::entity::EntityType;
        let Ok(registry) = self.registry_clone() else {
            return Vec::new();
        };
        let resolver = EntityResolver::new(registry.clone());
        let mut ops: Vec<String> = Vec::new();
        for cand in resolver.rank_candidates(word) {
            if cand.evidence.score < floor {
                continue;
            }
            // Map the candidate to its canonical op (follow synonym to the
            // example-bearing seed), exactly as the resolver's op path does.
            let Some(op) = resolver.resolve_operation_surface(&cand.entity.lemma) else {
                continue;
            };
            if !matches!(
                op.entity.entity_type,
                EntityType::Function | EntityType::Operator
            ) {
                continue;
            }
            let fn_name = op
                .entity
                .get_property("default_fn_name")
                .cloned()
                .unwrap_or_else(|| op.entity.lemma.clone());
            if !ops.contains(&fn_name) {
                ops.push(fn_name);
            }
        }
        ops
    }

    /// TEST-SUPPORT: prove a paraphrase RESOLVES to `expected_op` via the WordNet
    /// edge AND that op synthesizes a program passing strict differential
    /// verification on FRESH holdouts the verifier samples and labels by RUNNING
    /// an INDEPENDENT reference (`problem_from_reference` path — never
    /// example_cases). Errors carry the failure reason.
    pub fn resolve_and_strict_verify(
        &self,
        paraphrase: &str,
        expected_op: &str,
        ref_name: &'static str,
        ref_signature: &'static str,
        reference_code: &'static str,
    ) -> Result<(), String> {
        // 1. RESOLUTION: the paraphrase must reach `expected_op` at the floor.
        match self.resolve_op_probe(paraphrase) {
            Some((fn_name, score)) if fn_name == expected_op && score >= OP_RESOLVE_FLOOR => {}
            Some((fn_name, score)) => {
                return Err(format!(
                    "{paraphrase:?} resolved to {fn_name:?}@{score:.3}, expected {expected_op:?}@>={OP_RESOLVE_FLOOR}"
                ));
            }
            None => return Err(format!("{paraphrase:?} did not resolve to any op")),
        }
        // 2. SYNTHESIS + STRICT FRESH-HOLDOUT VERIFY via the reference path.
        let mut problem = crate::benchmark::problem_from_reference(
            ref_name,
            ref_signature,
            reference_code,
        )
        .map_err(|e| format!("reference unrunnable: {e}"))?;
        problem.category = "nl-wordnet-recall";
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "resolved to {expected_op} but solver could not synthesize (method={}, err={:?})",
                solved.method, solved.error
            ));
        }
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!("OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}", solved.code)
        })
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

    // If linguigenesis-core COMPREHENDED the request as a transform+aggregate (or
    // reduce-only / map-only degenerate) array pipeline, it populated
    // `req.pipeline` — the request is *comprehensible*, not unsound. Composition is
    // built downstream (`try_compose_pipeline`); here we simply decline to
    // fail-closed so the pipeline path can run. Single-op requests (`pipeline =
    // None`) fall through unchanged.
    if req.pipeline.is_some() {
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

/// The array→array reordering an ArrayTransform primitive computes, determined by
/// EXECUTING the synthesized transform code on probe arrays — never inferred from
/// the op's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayTransformKind {
    /// Sort ascending.
    Sort,
    /// Reverse element order.
    Reverse,
}

/// Classify an ARRAY-transform primitive (`Vec<i64> -> Vec<i64>`) by running its
/// SYNTHESIZED code on probe arrays and matching the output against each candidate
/// transform's known result. Behaviour-driven: the op's NAME is only a label, so
/// "is it sort vs reverse" is decided by execution, not assumption. Returns `None`
/// when neither candidate matches (e.g. the synthesized code is an affine misfit
/// of a single-example op) so the caller can fall back to the registry spec.
fn classify_array_transform_by_exec(transform_code: &str, transform_fn: &str) -> Option<ArrayTransformKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "arrxfm_probe".to_string(),
        category: "probe",
        description: "array transform classification",
        signature: "fn f(a: [i64]) -> [i64]",
        examples: vec![],
        ..Default::default()
    };
    // Probes where sort and reverse disagree (already-sorted/reversed inputs would
    // not discriminate), and where identity differs from both.
    let probes: &[&[i64]] = &[&[3, 1, 2], &[5, 2, 8, 1], &[4, 7, 3, 9, 1]];
    let run = |arr: &[i64]| -> Option<Vec<i64>> {
        match crate::runtime::execute_function_for_problem(
            transform_code,
            transform_fn,
            &[Value::int_array(arr)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Array(vs)) => vs
                .iter()
                .map(|v| match v {
                    crate::runtime::Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let candidates = [ArrayTransformKind::Sort, ArrayTransformKind::Reverse];
    for cand in candidates {
        let mut all_match = true;
        for p in probes {
            let mut expected: Vec<i64> = p.to_vec();
            match cand {
                ArrayTransformKind::Sort => expected.sort(),
                ArrayTransformKind::Reverse => expected.reverse(),
            }
            if run(p).as_deref() != Some(expected.as_slice()) {
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

/// Accepted, strict-verified transform-chain pipeline result.
#[derive(Clone, Debug)]
pub struct PipelineOutcome {
    /// The original NL request.
    pub description: String,
    /// The synthesized pipeline's own function name (call this to run `code`).
    pub fn_name: String,
    /// Element map op function names in REQUEST ORDER (outer→inner). Empty for a
    /// reduce-only pipeline.
    pub map_fns: Vec<String>,
    /// Array-transform op function name, if the pipeline has a sort/reverse stage
    /// between the map chain and any reduce. `None` when no array transform.
    pub array_xfm_fn: Option<String>,
    /// The array transform kind (behaviour-classified), if present.
    pub array_xfm: Option<ArrayTransformKind>,
    /// Aggregate (reduce) op function name, if the pipeline has a reduce stage
    /// (shape a). `None` for an array-output map chain (shape b).
    pub reduce_fn: Option<String>,
    /// The fold kind the reduce computes (behaviour-classified). `None` for an
    /// array-output map chain (no reduce).
    pub fold: Option<FoldKind>,
    /// The solver-synthesized program for the whole pipeline.
    pub code: String,
    /// `nl-compose-chain:<inner-solver-method>`.
    pub method: String,
}

impl PipelineOutcome {
    /// True iff this is a genuine multi-stage pipeline (>=1 map and a reduce, or
    /// a chain of >=2 maps) rather than a coincidental single-op solve. Used by
    /// the accept-test to reject a single-op match masquerading as a composition.
    pub fn is_two_stage(&self) -> bool {
        // Any reduce-bearing pipeline (shape a — including reduce-on-implicit-
        // array), a chain of >=2 maps, or a map+array-transform / standalone
        // non-req array transform is genuinely multi-stage. A lone single map
        // with no reduce never reaches acceptance (classify_pipeline returns None
        // for the plain single op).
        self.reduce_fn.is_some() || self.map_fns.len() >= 2 || self.array_xfm_fn.is_some()
    }

    /// True iff this pipeline contains a genuine array-transform stage (sort /
    /// reverse) composed over a map chain — the >=2-stage array→array shape the
    /// NL-COMPOSE-ARRTRANSFORM accept-criterion requires.
    pub fn has_array_transform(&self) -> bool {
        self.array_xfm_fn.is_some()
    }

    /// Length of the element-transform chain (number of composed ScalarMaps).
    pub fn map_chain_len(&self) -> usize {
        self.map_fns.len()
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
    let mut name = String::from("compose");
    if let Some(r) = &plan.reduce {
        name.push('_');
        name.push_str(&r.fn_name);
    } else {
        name.push_str("_maps");
    }
    if let Some(t) = &plan.array_transform {
        name.push('_');
        name.push_str(&t.fn_name);
    }
    for m in &plan.maps {
        name.push('_');
        name.push_str(&m.fn_name);
    }
    name
}

fn describe_plan(plan: &CompositionPlan) -> String {
    let maps: Vec<&str> = plan.maps.iter().map(|m| m.fn_name.as_str()).collect();
    let chain = if maps.is_empty() {
        "(no map)".to_string()
    } else {
        maps.join(" ∘ ")
    };
    let xfm = match &plan.array_transform {
        Some(t) => format!(" arrayxfm={}", t.fn_name),
        None => String::new(),
    };
    match &plan.reduce {
        Some(r) => format!("reduce={} ∘{} mapchain=[{}]", r.fn_name, xfm, chain),
        None => format!("mapchain=[{}]{} -> array (no reduce)", chain, xfm),
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
/// DSL. The map chain's fn bodies (verified single-op synthesis) go first so the
/// driver can call them; the element expression nests the chain in REQUEST ORDER
/// (`maps[0](maps[1](...maps[n-1](arr[i])))`, outer→inner). Then:
///   * if an `array_xfm` is present, the driver MATERIALIZES the mapped array,
///     applies the sort/reverse to it, and finally folds it (shape a) or returns
///     it (shape b) — the array transform sits between the map chain and reduce;
///   * otherwise the driver folds the element expression inline (shape a) or
///     builds the mapped array (shape b), exactly as before.
/// Used only to LABEL fresh holdouts; the accepted program is what the solver finds.
fn emit_pipeline_reference(
    composed_name: &str,
    fold: Option<FoldKind>,
    array_xfm: Option<ArrayTransformKind>,
    map_chain: &[(String, String)],
) -> String {
    let mut out = String::new();
    // Emit each distinct map fn body once (the chain may legitimately repeat an
    // op, but the body only needs to appear a single time).
    let mut emitted: Vec<&str> = Vec::new();
    for (map_fn, map_code) in map_chain {
        if emitted.contains(&map_fn.as_str()) {
            continue;
        }
        emitted.push(map_fn.as_str());
        out.push_str(map_code);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }

    // Nest the chain outer→inner around the element: maps[0]( ... maps[n-1](inner) ).
    let elem = |inner: &str| -> String {
        let mut expr = inner.to_string();
        for (map_fn, _) in map_chain.iter().rev() {
            expr = format!("{}({})", map_fn, expr);
        }
        expr
    };

    // ── ARRAY-TRANSFORM present: materialize mapped array, reorder, then fold or
    //    return. The reorder uses the SAME DSL the dedicated sort/reverse
    //    array_transform candidates emit (`mapped.sort()` / index-loop reverse). ──
    if let Some(kind) = array_xfm {
        let elem_item = elem("item");
        // Build the mapped array, then apply the array transform into `xfm`.
        let build = format!(
            "    mapped: [i64] = [];\n    for item in arr {{\n        mapped.push({elem_item});\n    }}\n"
        );
        let reorder = match kind {
            ArrayTransformKind::Sort => {
                // sort in place; the reordered array is `mapped`.
                "    mapped.sort();\n".to_string()
            }
            ArrayTransformKind::Reverse => String::new(), // handled per-shape below
        };
        match fold {
            // Shape (a): reduce over the reordered array.
            Some(fk) => {
                // Reverse does not change sum/max/min/product, but we still emit a
                // faithful reorder so the reference is an honest independent impl.
                let reordered_array = match kind {
                    ArrayTransformKind::Sort => {
                        format!("{build}{reorder}")
                    }
                    ArrayTransformKind::Reverse => format!(
                        "{build}    xfm: [i64] = [];\n    i: i64 = mapped.len - 1;\n    while i >= 0 {{\n        xfm.push(mapped[i]);\n        i = i - 1;\n    }}\n"
                    ),
                };
                let arr_name = match kind {
                    ArrayTransformKind::Sort => "mapped",
                    ArrayTransformKind::Reverse => "xfm",
                };
                let body = emit_fold_over_named_array(fk, arr_name);
                out.push_str(&format!(
                    "fn {composed_name}(arr: [i64]) -> i64 {{\n{reordered_array}{body}}}\n"
                ));
            }
            // Shape (b): return the reordered mapped array.
            None => match kind {
                ArrayTransformKind::Sort => {
                    out.push_str(&format!(
                        "fn {composed_name}(arr: [i64]) -> [i64] {{\n{build}{reorder}    return mapped;\n}}\n"
                    ));
                }
                ArrayTransformKind::Reverse => {
                    out.push_str(&format!(
                        "fn {composed_name}(arr: [i64]) -> [i64] {{\n{build}    result: [i64] = [];\n    i: i64 = mapped.len - 1;\n    while i >= 0 {{\n        result.push(mapped[i]);\n        i = i - 1;\n    }}\n    return result;\n}}\n"
                    ));
                }
            },
        }
        return out;
    }

    match fold {
        // ── Shape (a): reduce present → fold the mapped element to a scalar. ──
        Some(FoldKind::Sum) | Some(FoldKind::Product) => {
            let elem = elem("arr[i]");
            let (init, op) = match fold {
                Some(FoldKind::Sum) => (0, "+"),
                Some(FoldKind::Product) => (1, "*"),
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
        Some(FoldKind::Max) | Some(FoldKind::Min) => {
            let elem = elem("arr[i]");
            let cmp = if fold == Some(FoldKind::Max) { ">" } else { "<" };
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
        // ── Shape (b): no reduce → build the mapped array (array output). ─────
        None => {
            let elem = elem("item");
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> [i64] {{\n    \
                 result: [i64] = [];\n    \
                 for item in arr {{\n        \
                 result.push({elem});\n    }}\n    \
                 return result;\n}}\n",
                name = composed_name,
                elem = elem,
            ));
        }
    }
    out
}

/// Emit a fold body (Sum/Product/Max/Min) over an already-built array variable
/// `arr_name`, returning the scalar. Used by the array-transform reference path
/// where the array has already been mapped + reordered into `arr_name`. The
/// emitted block ENDS with `return acc;` and is wrapped by the caller's `fn`.
fn emit_fold_over_named_array(fold: FoldKind, arr_name: &str) -> String {
    match fold {
        FoldKind::Sum | FoldKind::Product => {
            let (init, op) = match fold {
                FoldKind::Sum => (0, "+"),
                FoldKind::Product => (1, "*"),
                _ => unreachable!(),
            };
            format!(
                "    acc: i64 = {init};\n    j: i64 = 0;\n    while j < {arr}.len {{\n        acc = acc {op} {arr}[j];\n        j = j + 1;\n    }}\n    return acc;\n",
                arr = arr_name
            )
        }
        FoldKind::Max | FoldKind::Min => {
            let cmp = if fold == FoldKind::Max { ">" } else { "<" };
            format!(
                "    acc: i64 = {arr}[0];\n    j: i64 = 1;\n    while j < {arr}.len {{\n        v: i64 = {arr}[j];\n        if v {cmp} acc {{ acc = v; }}\n        j = j + 1;\n    }}\n    return acc;\n",
                arr = arr_name
            )
        }
    }
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

    /// NL-COMPREHEND-IN-LGCORE ANTI-CHEAT: linguigenesis-core — NOT this bridge —
    /// now COMPREHENDS the composition. Calling `CodingComprehension::comprehend`
    /// directly (the lg-core entry point) on a compositional request must return a
    /// `SynthesisRequirement` whose `pipeline` field is `Some(reduce=sum, maps=
    /// [negate])`. The bridge only EXECUTES this emitted plan; it derives nothing.
    /// (This mirrors the lg-core-side unit test `comprehend_emits_pipeline_for_sum_
    /// of_negated`, but runs under the nsynth build which links lg-core as a path
    /// dependency, so the relocated comprehension is proven end-to-end here too.)
    #[test]
    fn lgcore_comprehend_emits_composition_plan() {
        let registry = LinguigenesisBridge::new()
            .registry_clone()
            .expect("registry");
        let mut coding = CodingComprehension::new(registry);
        let req = coding.comprehend("sum of the negated values");
        let plan = req
            .pipeline
            .as_ref()
            .unwrap_or_else(|| panic!("lg-core must emit pipeline; unresolved={:?}", req.unresolved));
        assert_eq!(plan.maps.len(), 1, "maps={:?}", plan.maps);
        assert_eq!(plan.maps[0].role, OpRole::ScalarMap);
        assert_eq!(plan.maps[0].fn_name, "negate");
        let reduce = plan.reduce.as_ref().expect("reduce stage");
        assert!(
            matches!(reduce.role, OpRole::ArrayReduce | OpRole::BinaryFoldSeed),
            "reduce role={:?}",
            reduce.role
        );
        assert!(
            reduce.fn_name.contains("sum") || reduce.fn_name == "add",
            "reduce fn_name={}",
            reduce.fn_name
        );
        // Single-op requests must comprehend with NO pipeline (unchanged path).
        for single in ["square a number", "compute the total of an array"] {
            let r = coding.comprehend(single);
            assert!(
                r.pipeline.is_none(),
                "{single:?} must have no pipeline, got {:?}",
                r.pipeline
            );
        }
    }

    /// NL-COMPREHEND-IN-LGCORE STRUCTURAL PROOF: the comprehended plan is what
    /// drives acceptance — a composed program built from the emitted plan must
    /// strict-verify on FRESH holdouts. (Distinct from the integration suite: this
    /// asserts the bridge consumed `req.pipeline`, not a bridge-local derivation.)
    #[test]
    fn lgcore_plan_drives_strict_verified_composition() {
        let bridge = LinguigenesisBridge::new();
        let outcome = bridge
            .try_compose_pipeline("sum of the negated values")
            .expect("recognised as pipeline (plan came from lg-core comprehend)")
            .expect("built and strict-verified");
        assert!(outcome.is_two_stage(), "must be a genuine multi-stage pipeline");
        assert_eq!(outcome.map_fns, vec!["negate".to_string()]);
        assert!(outcome.reduce_fn.is_some(), "reduce stage present");
        assert!(
            outcome.method.starts_with("nl-compose-chain:"),
            "method tag={}",
            outcome.method
        );
    }

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
