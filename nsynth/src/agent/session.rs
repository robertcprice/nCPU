//! Universal coding-agent session: any NL query → registry workflow route + full tools.

use crate::agent::capability_registry::CapabilityRegistry;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{GuardrailPolicy, RepoAgent, RepoAgentRunResult};
use crate::agent::repository::{retrieve_paths, RepoIndex};
use crate::agent::runtime::{AgentRunId, CodeTaskSpec};
use crate::agent::session_persistence::{
    load_session_snapshot, save_session_snapshot, session_path, truncate_preview, PendingQuery,
    SessionSnapshot,
};
use crate::agent::tools::{SecureToolRuntime, ToolCall, ToolOutput};
use crate::linguigenesis_bridge::{BridgeError, LinguigenesisBridge};
use linguigenesis_core::coding_comprehension::ComprehensionOutcome;
use linguigenesis_core::coding_dialogue::{
    build_clarifications, format_clarification_prompt, needs_clarification,
};
use linguigenesis_core::coding_requirements::{CodingWorkflow, SynthesisRequirement};
use linguigenesis_core::entity::RelationType;
use linguigenesis_core::reasoning::KnowledgeQA;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Registry-derived workflow route (no keyword tables in Rust).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryRoute {
    SynthesizeFunction,
    RepoRepair,
    ExplainCode,
    CodeReview,
    GreenfieldProject,
    ToolExplore,
    Clarification,
}

/// Result of handling one user query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentQueryResult {
    pub route: QueryRoute,
    pub success: bool,
    pub response: String,
    pub workflow: String,
    pub clarification_questions: Vec<String>,
    pub synthesis_method: Option<String>,
    #[serde(skip)]
    pub repo_result: Option<RepoAgentRunResult>,
    pub tool_trace: Vec<(String, String)>,
}

/// Session with secure tool access and multi-workflow routing.
pub struct CodingAgentSession {
    root: PathBuf,
    policy: GuardrailPolicy,
    tools: SecureToolRuntime,
    session_id: String,
    pending: Option<PendingQuery>,
    history_len: usize,
}

impl CodingAgentSession {
    pub fn new(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        Self::with_session_id(root, policy, AgentRunId::new().0)
    }

    pub fn with_session_id(
        root: impl Into<PathBuf>,
        policy: GuardrailPolicy,
        session_id: String,
    ) -> Self {
        let root = root.into();
        let tools = SecureToolRuntime::for_general_agent(&root, policy.clone());
        Self {
            root,
            policy,
            tools,
            session_id,
            pending: None,
            history_len: 0,
        }
    }

    /// Load or create a named session from `.nsynth/sessions/`.
    pub fn load(
        root: impl Into<PathBuf>,
        policy: GuardrailPolicy,
        session_id: &str,
    ) -> Result<Self, String> {
        let root = root.into();
        let path = session_path(&root, session_id);
        if !path.is_file() {
            return Ok(Self::with_session_id(root, policy, session_id.to_string()));
        }
        let snapshot = load_session_snapshot(&path)?;
        let mut session = Self::with_session_id(root, policy, snapshot.session_id);
        session.history_len = snapshot.history_len;
        session.pending = snapshot.pending;
        Ok(session)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn has_pending_clarification(&self) -> bool {
        self.pending.is_some()
    }

    pub fn tools(&self) -> &SecureToolRuntime {
        &self.tools
    }

    pub fn allowed_tool_capabilities(&self) -> Vec<String> {
        self.tools.allowed_capabilities()
    }

    /// Persist session state for resume / clarification.
    pub fn persist(&self, last_query: Option<&str>, last_result: Option<&AgentQueryResult>) -> Result<PathBuf, String> {
        let snapshot = SessionSnapshot {
            session_id: self.session_id.clone(),
            root: self.root.to_string_lossy().to_string(),
            last_query: last_query.map(str::to_string),
            last_route: last_result.map(|r| r.route.clone()),
            last_success: last_result.map(|r| r.success),
            last_response_preview: last_result
                .map(|r| truncate_preview(&r.response, 400)),
            pending: self.pending.clone(),
            history_len: self.history_len,
            ..Default::default()
        };
        save_session_snapshot(&self.root, &snapshot)
    }

    /// Apply a clarification answer; may return another clarification turn if still ambiguous.
    pub fn clarify_and_continue(&mut self, answer: &str) -> Result<AgentQueryResult, String> {
        let pending = self
            .pending
            .clone()
            .ok_or_else(|| "no pending clarification".to_string())?;
        let bridge = LinguigenesisBridge::new();
        let mut partial = pending.partial.clone();
        let field = pending
            .questions
            .first()
            .map(|q| q.field.clone())
            .ok_or_else(|| "no clarification field in pending session".to_string())?;
        bridge
            .apply_clarification(&mut partial, field.clone(), answer)
            .map_err(|e| e.to_string())?;
        let mut answers = pending.answers;
        answers.push((field, answer.to_string()));

        let registry = bridge.registry_clone().map_err(|e| e.to_string())?;
        let questions = build_clarifications(&partial, &registry);
        if needs_clarification(&partial) && !questions.is_empty() {
            let result = self.clarification_result(&pending.query, &partial, &questions);
            self.pending = Some(PendingQuery {
                query: pending.query.clone(),
                partial,
                questions: questions.clone(),
                answers,
            });
            self.record_result(&pending.query, &result);
            return Ok(result);
        }

        self.pending = None;
        // gate=false: the user has explicitly disambiguated via clarification, so
        // the fail-closed gate must NOT re-judge the (often gibberish) original
        // query — that would refuse a user-confirmed op.
        let result = self.dispatch(&pending.query, &partial, false);
        self.record_result(&pending.query, &result);
        Ok(result)
    }

    fn clarification_result(
        &self,
        query: &str,
        req: &SynthesisRequirement,
        questions: &[linguigenesis_core::coding_dialogue::ClarificationQuestion],
    ) -> AgentQueryResult {
        AgentQueryResult {
            route: QueryRoute::Clarification,
            success: false,
            response: format_clarification_prompt(questions),
            workflow: workflow_label(&req.workflow),
            clarification_questions: questions.iter().map(|q| q.prompt.clone()).collect(),
            synthesis_method: None,
            repo_result: None,
            tool_trace: Vec::new(),
        }
    }

    /// Handle any NL coding query via KVRM workflow routing.
    pub fn handle_query(&mut self, query: &str) -> AgentQueryResult {
        // LEARN-ON-THE-FLY (UNWALL-4-LEARN-ON-THE-FLY-NL): if the request DEFINES a
        // new named op (by examples or by composition) or REUSES a previously-learned
        // op, route it through the EXISTING regression-gated self-extension +
        // durable-persistence path (`self_improve::extend::self_extend` →
        // `regression_gate` → `store::save_one`; reuse resolves from the gated
        // reload performed by `Engine::new`). Intercept BEFORE comprehension, which
        // would otherwise mis-route a teach/reuse request. Non-teach/reuse requests
        // fall through unchanged (`LearnIntake::NotLearn`).
        match crate::learn_nl::classify(query) {
            crate::learn_nl::LearnIntake::NotLearn => {}
            intake => {
                let result = self.run_learn_intake(intake);
                self.record_result(query, &result);
                return result;
            }
        }

        // REFERENCE INTAKE (UNWALL-3-REFERENCE-INTAKE-NL): if the request CARRIES
        // a runnable reference implementation ("behaves like THIS: <fn>"), the
        // reference's behavior IS the spec. Intercept BEFORE comprehension (which
        // would mis-route a request containing code) and route it through the
        // existing reference path (Spec::Reference → problem_from_reference →
        // solve_problem → strict-verify against fresh inputs run through the
        // reference). Structural signal (an embedded `fn` block), not a phrase
        // table; an unparseable reference is refused honestly. Requests with no
        // embedded reference fall through unchanged.
        match crate::reference_nl::classify(query) {
            crate::reference_nl::ReferenceIntake::Reference {
                name,
                signature,
                code,
            } => {
                let result = self.run_reference_synthesis(&name, &signature, &code);
                self.record_result(query, &result);
                return result;
            }
            crate::reference_nl::ReferenceIntake::Unparseable(reason) => {
                let result = self.refuse_unparseable_reference(&reason);
                self.record_result(query, &result);
                return result;
            }
            crate::reference_nl::ReferenceIntake::NotReference => {}
        }

        // TENSOR REACH (NL-BRIDGE-3B-TENSOR-FORWARD): intercept tensor requests
        // BEFORE generic comprehension, which would otherwise divert a
        // 'train a model' to a clarification loop or a forward op to ToolExplore.
        // A forward-inference request → codegen; a training request → honest
        // refusal (training is a no-op here). Non-tensor requests are unaffected.
        let tensor_route = crate::tensor_nl::classify(query);
        if !matches!(tensor_route, crate::tensor_nl::TensorRouteOutcome::NotTensor) {
            let result = match tensor_route {
                crate::tensor_nl::TensorRouteOutcome::RefuseTraining => {
                    self.refuse_tensor_training()
                }
                crate::tensor_nl::TensorRouteOutcome::Forward { lemma, code } => {
                    self.run_tensor_forward(&lemma, &code)
                }
                crate::tensor_nl::TensorRouteOutcome::NotTensor => unreachable!(),
            };
            self.record_result(query, &result);
            return result;
        }

        // COMPOSITIONAL-DESCRIPTION INTAKE (P2C-PROMPT-TO-CONTRACT): a bare prose
        // description of a SEQUENCE of known ops ("the larger of two numbers, then
        // triple it") — with NO user-supplied examples — auto-generates its own
        // contract. Each clause resolves (via the same emergent resolver the
        // single-op gate uses) to a registry primitive with an emittable body; the
        // ordered chain is emitted as a runnable reference whose BEHAVIOUR is the
        // spec, then handed to the EXISTING reference path (problem_from_reference
        // manufactures the example pairs + holdout oracle → solve → strict-verify).
        // Intercept BEFORE comprehension's single-op parroting door so a
        // multi-step description is not mis-routed to one of its sub-ops. A
        // description whose HEAD is not a scalar op falls through unchanged
        // (NotCompositional → array-pipeline / single-op); a confirmed scalar
        // composition with an UNRESOLVABLE atom is refused, never fabricated.
        //
        // MULTI-COMPONENT GUARD (BUILD-B2-FIX-CLI-MULTICOMP-ROUTING): a request
        // that DESCRIBES >=2 component functions ("a module with a function that
        // ... then ..., and a function that ... then ...") is a PROJECT, not a
        // single scalar composition. The single-fn intercept below splits the
        // WHOLE string on every `then` and mashes the two components into one
        // nonsense chain; it must NOT fire here. We consult the SAME structural
        // signal `synthesize_project` uses to decompose (comprehend_project's
        // function-head split) — so a multi-component request falls through to
        // `comprehend_outcome` → `dispatch` → `synthesize_project` (the multi-file
        // GreenfieldProject door). A bare composition (0 heads) and a SINGLE
        // described function (1 head) still hit the single-fn P2C intercept
        // unchanged.
        {
            let bridge = LinguigenesisBridge::new();
            if !bridge.is_multi_component(query) {
              if let Ok(registry) = bridge.registry_clone() {
                match crate::reference_nl::classify_compositional(query, &registry) {
                    crate::reference_nl::CompositionalIntake::Compositional {
                        name,
                        signature,
                        chain,
                    } => {
                        let result =
                            self.run_compositional_synthesis(&bridge, &name, &signature, &chain);
                        self.record_result(query, &result);
                        return result;
                    }
                    crate::reference_nl::CompositionalIntake::Unresolvable(reason) => {
                        let result = self.refuse_compositional(&reason);
                        self.record_result(query, &result);
                        return result;
                    }
                    crate::reference_nl::CompositionalIntake::NotCompositional => {}
                }
                // P2C WIDEN (BUILD-A): the scalar door declined — try an ARRAY or
                // STRING `then`-composition (map-then-reduce / nested string
                // transforms). A NotDomain result falls through unchanged to the
                // pipeline / single-op doors; an Unresolvable in-domain atom
                // refuses honestly rather than fabricating.
                match crate::reference_nl::classify_domain_compositional(query, &registry) {
                    crate::reference_nl::DomainCompositionalIntake::Array {
                        name,
                        signature,
                        maps,
                        reduce,
                    } => {
                        let reference = bridge.emit_array_reference(&name, &maps, reduce.as_ref());
                        let result =
                            self.run_emitted_compositional(&name, &signature, reference);
                        self.record_result(query, &result);
                        return result;
                    }
                    crate::reference_nl::DomainCompositionalIntake::StringT {
                        name,
                        signature,
                        steps,
                    } => {
                        let reference = bridge.emit_string_reference(&name, &steps);
                        let result =
                            self.run_emitted_compositional(&name, &signature, reference);
                        self.record_result(query, &result);
                        return result;
                    }
                    crate::reference_nl::DomainCompositionalIntake::Unresolvable(reason) => {
                        let result = self.refuse_compositional(&reason);
                        self.record_result(query, &result);
                        return result;
                    }
                    crate::reference_nl::DomainCompositionalIntake::NotDomain => {}
                }
              }
            }
        }

        // TEACH INTAKE (registry growth): "teach web: a testimonials section
        // means customer quotes in a row, also called reviews" — persists the
        // concept into the domain's data registry so it resolves in every
        // future request AND process. The universal-substrate growth seam.
        if let Some((domain, concept)) = crate::registry_hub::parse_teach(query) {
            let lemma = concept.lemma.clone();
            let kind = concept.kind.clone();
            let result = match crate::registry_hub::teach_concept(domain, concept) {
                Ok(()) => AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: true,
                    response: format!(
                        "Learned {domain:?} concept '{lemma}' ({kind}) — resolvable in all future requests, persisted."
                    ),
                    workflow: "registry.teach".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("registry-teach".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                },
                Err(e) => AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: false,
                    response: format!("teach failed: {e}"),
                    workflow: "registry.teach".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("registry-teach".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                },
            };
            self.record_result(query, &result);
            return result;
        }

        // STRUCTURE-SCAFFOLD INTAKE: "make a new project organized like
        // structure.md" — a construction request naming ORGANIZATION plus a spec
        // FILE that exists in the root. The spec is the oracle: the generated
        // tree is walk-asserted against it; .html nodes become real verified
        // pages. Declines without both the cue and a resolvable file.
        {
            use linguigenesis_core::entity_resolution::morphological_variants;
            let lower = query.to_lowercase();
            let toks: Vec<&str> = lower
                .split(|c: char| !c.is_alphanumeric() && c != '.' && c != '/')
                .filter(|t| !t.is_empty())
                .collect();
            // Token-level + morphology (never substring): "organized"->organize
            // via the shared morphological stemmer; "disorganized" does NOT match.
            let morph_eq = |tok: &str, name: &str| -> bool {
                if tok == name {
                    return true;
                }
                let mut tv = morphological_variants(tok);
                tv.push(tok.to_string());
                let mut nv = morphological_variants(name);
                nv.push(name.to_string());
                tv.iter().any(|v| nv.contains(v))
            };
            let wants_structure = toks.iter().any(|t| {
                ["organize", "structure", "layout"].iter().any(|w| morph_eq(t, w))
            });
            let has_cue = toks.iter().any(|t| {
                ["make", "create", "build", "new", "generate", "scaffold"]
                    .iter()
                    .any(|w| morph_eq(t, w))
            });
            if wants_structure && has_cue {
                if let Some(spec_path) = crate::site::structure_file_from_prose(&self.root, query) {
                    let result = self.run_scaffold_structure(&spec_path);
                    self.record_result(query, &result);
                    return result;
                }
            }
        }

        // SITE INTAKE (web-artifact front door): a construction request naming a
        // page/site — "add a new page called portfolio, modern theme, hero and
        // gallery, teal and charcoal" — builds the page with request-derived
        // structural verification (every requested section/color proven present).
        // comprehend_site_request self-gates on construction cue + web noun, so
        // op requests ("paginate the array") fall through untouched.
        if let Some(req) = crate::site::comprehend_site_request(query) {
            let result = self.run_build_site(&req);
            self.record_result(query, &result);
            return result;
        }

        // CLI INTAKE: "build a CLI tool for a function double where double(2)=4"
        // — a construction cue + a CLI noun + a function name carrying inline
        // examples. Synthesizes the verified function, wraps it in a runnable
        // command-line tool, and VERIFIES the CLI compiles + computes an example
        // (fail-closed). Requires inline examples, so it never hijacks a plain
        // "build a tool" or an api/site ask (those have no NAME(x)=y clause).
        if let Some(ask) = crate::cli_emit::comprehend_cli_request(query) {
            let result = match crate::cli_emit::build_cli_ask(&self.root, query, &ask) {
                Ok(written) => AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: true,
                    response: format!("Built CLI tool '{}' — compile+run verified: {}", ask.name, written.join(", ")),
                    workflow: "cli.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("cli-intake".to_string()),
                    repo_result: None,
                    tool_trace: written.iter().map(|p| (format!("fs.write:{p}"), "ok".to_string())).collect(),
                },
                Err(e) => AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: false,
                    response: format!("CLI build failed: {e}"),
                    workflow: "cli.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("cli-intake".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                },
            };
            self.record_result(query, &result);
            return result;
        }

        // BACKEND INTAKE (hub backend domain): "make me an api with a health
        // check and a users store" — routes through the registry hub's backend
        // resolution (synonym edges: api->endpoint, service->server), builds via
        // the unified door (rule clauses w/ examples) or the structural server
        // (health + store), COMPILE-repair-gated. Site asks were consumed above.
        if let Some(ask) = crate::backend_intake::comprehend_backend_prose(query) {
            let result = match crate::backend_intake::build_backend_ask(&self.root, query, &ask) {
                Ok(written) => AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: true,
                    response: format!(
                        "Built backend (store {:?}, rules [{}]) — compile-gated: {}",
                        ask.store,
                        ask.rule_names.join(", "),
                        written.join(", "),
                    ),
                    workflow: "backend.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("backend-intake".to_string()),
                    repo_result: None,
                    tool_trace: written
                        .iter()
                        .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                        .collect(),
                },
                Err(e) => AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: false,
                    response: format!("backend build failed: {e}"),
                    workflow: "backend.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("backend-intake".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                },
            };
            self.record_result(query, &result);
            return result;
        }

        // COMPONENT INTAKE (component layer front door): a CONSTRUCTION request
        // naming known component(s) — "build a counter", "an accumulator", "give me
        // array statistics" — builds a verified crate directly (resolve ->
        // synthesize verified leaves -> compose + struct glue -> compile-gate ->
        // behavior-gate). Double-gated to never hijack an OPERATION request:
        // `route_component_build` requires a construction cue AND that the matching
        // surface token resolves to NO coding op (so "count the array" — count ->
        // array_sum — and "sort a list of counters" both fall through untouched).
        // These phrases currently dead-end at Clarification, so this strictly
        // rescues them; anything it declines routes exactly as before.
        {
            let bridge = LinguigenesisBridge::new();
            let specs = crate::component::route_component_build(&bridge, query);
            if !specs.is_empty() {
                let result = self.run_build_components(&specs);
                self.record_result(query, &result);
                return result;
            }
        }

        let bridge = LinguigenesisBridge::new();
        let result = match bridge.comprehend_outcome(query) {
            Ok(ComprehensionOutcome::Ready(req)) => self.dispatch(query, &req, true),
            Ok(ComprehensionOutcome::NeedsClarification(req, questions)) => {
                self.pending = Some(PendingQuery {
                    query: query.to_string(),
                    partial: req.clone(),
                    questions: questions.clone(),
                    answers: Vec::new(),
                });
                self.clarification_result(query, &req, &questions)
            }
            Err(BridgeError::ClarificationNeeded { partial, questions }) => {
                self.pending = Some(PendingQuery {
                    query: query.to_string(),
                    partial: partial.clone(),
                    questions: questions.clone(),
                    answers: Vec::new(),
                });
                self.clarification_result(query, &partial, &questions)
            }
            Err(_) => self.handle_explore(query),
        };
        self.record_result(query, &result);
        result
    }

    fn record_result(&mut self, query: &str, result: &AgentQueryResult) {
        self.history_len += 1;
        let _ = self.persist(Some(query), Some(result));
    }

    /// COMPONENT LAYER product entry point (opt-in; not yet in the auto-router,
    /// which would need tight NL gating to avoid hijacking op requests like
    /// "count the array"). Build the KNOWN component(s) an NL phrase names into the
    /// session root as ONE verified crate: resolve -> synthesize verified leaf ops
    /// -> compose (+ raw-Rust struct glue for structural components) -> compile-gate
    /// -> behavioral-gate. Returns `None` when the phrase names no known component,
    /// so a caller can fall back to normal routing.
    pub fn try_build_components(&mut self, query: &str) -> Option<AgentQueryResult> {
        let specs = crate::component::resolve_components(query);
        if specs.is_empty() {
            return None;
        }
        let result = self.run_build_components(&specs);
        self.record_result(query, &result);
        Some(result)
    }

    /// Build a comprehended site/page request into the session root, reporting
    /// the request-fidelity verification in the tool trace. Fail-closed: an
    /// emission that doesn't verify against the request is a failed result.
    /// Scaffold the session root from a structure-spec file; the SPEC IS THE
    /// ORACLE (walk-asserted inside scaffold_from_structure — fail-closed).
    fn run_scaffold_structure(&mut self, spec_path: &std::path::Path) -> AgentQueryResult {
        let spec = match std::fs::read_to_string(spec_path) {
            Ok(s) => s,
            Err(e) => {
                return AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: false,
                    response: format!("structure spec unreadable: {e}"),
                    workflow: "site.scaffold".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("structure-scaffold".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                }
            }
        };
        match crate::site::scaffold_from_structure(&self.root, &spec) {
            Ok(written) => AgentQueryResult {
                route: QueryRoute::GreenfieldProject,
                success: true,
                response: format!(
                    "Scaffolded {} node(s) from {} — structure oracle verified (every spec node exists; .html nodes are generated verified pages)",
                    written.len(),
                    spec_path.file_name().unwrap_or_default().to_string_lossy(),
                ),
                workflow: "site.scaffold".to_string(),
                clarification_questions: Vec::new(),
                synthesis_method: Some("structure-scaffold".to_string()),
                repo_result: None,
                tool_trace: written
                    .iter()
                    .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                    .collect(),
            },
            Err(e) => AgentQueryResult {
                route: QueryRoute::GreenfieldProject,
                success: false,
                response: format!("scaffold failed: {e}"),
                workflow: "site.scaffold".to_string(),
                clarification_questions: Vec::new(),
                synthesis_method: Some("structure-scaffold".to_string()),
                repo_result: None,
                tool_trace: Vec::new(),
            },
        }
    }

    fn run_build_site(&mut self, req: &crate::site::SiteRequest) -> AgentQueryResult {
        // EXTEND when a site already exists (follow its conventions, rewire the
        // nav in every page, whole-site link integrity); CREATE otherwise.
        let site_exists = std::fs::read_dir(self.root.join("site"))
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_string_lossy().ends_with(".html"))
            })
            .unwrap_or(false);
        let build = if site_exists {
            crate::site::extend_site(&self.root, req)
        } else {
            crate::site::build_site_page(&self.root, req)
        };
        match build {
            Ok(mut written) => {
                // SITE+BACKEND CLOSED LOOP: an api-wired form promises a live
                // target (POST /events). If no backend exists yet, PROVISION a
                // structural one through the same compile+serve gate — the
                // promise is verified, never aspirational. Fail-closed: a
                // wired form with no provisionable target fails the ask.
                if req.api_form && !self.root.join("backend/main.rs").exists() {
                    let ask = crate::backend_intake::BackendAsk {
                        store: crate::backend_ir::StoreKind::Memory,
                        rule_names: Vec::new(),
                    };
                    let english = format!(
                        "backend accepting form submissions from site page '{}'",
                        req.page
                    );
                    match crate::backend_intake::build_backend_ask(&self.root, &english, &ask) {
                        Ok(mut backend_written) => written.append(&mut backend_written),
                        Err(e) => {
                            return AgentQueryResult {
                                route: QueryRoute::GreenfieldProject,
                                success: false,
                                response: format!(
                                    "site pages written but api target provision failed: {e}"
                                ),
                                workflow: "site.build".to_string(),
                                clarification_questions: Vec::new(),
                                synthesis_method: Some("site-domain".to_string()),
                                repo_result: None,
                                tool_trace: Vec::new(),
                            }
                        }
                    }
                }
                let mut tool_trace: Vec<(String, String)> = written
                    .iter()
                    .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                    .collect();
                tool_trace.push(("site.verify".to_string(), "ok".to_string()));
                AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success: true,
                    response: format!(
                        "Built page '{}' (theme {}, sections [{}], colors [{}]) — request-fidelity verified: {}",
                        req.page,
                        req.theme,
                        req.sections.join(", "),
                        req.colors.join(", "),
                        written.join(", "),
                    ),
                    workflow: "site.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("site-domain".to_string()),
                    repo_result: None,
                    tool_trace,
                }
            }
            Err(e) => AgentQueryResult {
                route: QueryRoute::GreenfieldProject,
                success: false,
                response: format!("site build failed: {e}"),
                workflow: "site.build".to_string(),
                clarification_questions: Vec::new(),
                synthesis_method: Some("site-domain".to_string()),
                repo_result: None,
                tool_trace: Vec::new(),
            },
        }
    }

    fn run_build_components(
        &mut self,
        specs: &[&'static crate::component::ComponentSpec],
    ) -> AgentQueryResult {
        use crate::component::BehaviorStatus;
        let bridge = LinguigenesisBridge::new();
        let names: Vec<String> = specs.iter().map(|s| s.name.to_string()).collect();
        match crate::component::build_project(&bridge, specs, &self.root) {
            Ok(build) => {
                let mut tool_trace: Vec<(String, String)> = build
                    .outcome
                    .written
                    .iter()
                    .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                    .collect();
                let compile_ok = build.outcome.compile.is_ok();
                tool_trace.push((
                    "cargo.check".to_string(),
                    if compile_ok { "ok" } else { "failed" }.to_string(),
                ));
                let behavior_note = match &build.behavior {
                    BehaviorStatus::Passed => {
                        tool_trace.push(("cargo.test".to_string(), "ok".to_string()));
                        " behavior: PASSED".to_string()
                    }
                    BehaviorStatus::Failed(e) => {
                        tool_trace.push(("cargo.test".to_string(), "failed".to_string()));
                        format!(" behavior: FAILED\n{e}")
                    }
                    BehaviorStatus::Unverified(e) => {
                        tool_trace.push(("cargo.test".to_string(), "unverified".to_string()));
                        format!(" behavior: UNVERIFIED ({e})")
                    }
                    BehaviorStatus::NotRun => String::new(),
                };
                // Success requires a clean compile AND that no declared behavioral
                // contract FAILED (NotRun/Unverified don't sink a bundle).
                let success = compile_ok && build.behavior.not_failed();
                let response = format!(
                    "Built {} component(s) [{}] into a verified crate: {} verified leaf op(s), \
                     {} struct(s), compile {}.{}",
                    names.len(),
                    names.join(", "),
                    build.leaves_verified.len(),
                    build.structs.len(),
                    if compile_ok { "OK" } else { "FAILED" },
                    behavior_note,
                );
                AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success,
                    response,
                    workflow: "component.build".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("component-layer".to_string()),
                    repo_result: None,
                    tool_trace,
                }
            }
            Err(e) => AgentQueryResult {
                route: QueryRoute::GreenfieldProject,
                success: false,
                response: format!("component build failed: {e}"),
                workflow: "component.build".to_string(),
                clarification_questions: Vec::new(),
                synthesis_method: Some("component-layer".to_string()),
                repo_result: None,
                tool_trace: Vec::new(),
            },
        }
    }

    /// Invoke a tool through the session's secure runtime.
    pub fn invoke_tool(&self, tool: &str, call: &ToolCall) -> Result<ToolOutput, String> {
        self.tools
            .invoke(tool, call)
            .map_err(|e| e.to_string())
    }

    /// `gate` selects whether the emergent fail-closed gate (FIX C) is consulted
    /// on the single-op synthesis path. It is TRUE for an INITIAL `handle_query`
    /// (the raw request may have mis-resolved, so domain/type/identity must be
    /// checked against it) and FALSE for a clarify-continuation, where the user
    /// has EXPLICITLY disambiguated the operation through dialogue — re-running
    /// the gate against the original (often gibberish) query would wrongly refuse
    /// the user-confirmed op.
    /// True iff `lemma` is a KNOWN registry Function op carrying its own canonical
    /// `example_cases` — the trust signal that a request genuinely resolved to a
    /// real operation (whose examples are the op's ground truth), as opposed to a
    /// prose-fabricated function name with self-derived (possibly mis-comprehended)
    /// examples. Used to gate the Unknown-workflow synthesis promotion.
    fn registry_op_has_examples(&self, lemma: &str) -> bool {
        let bridge = LinguigenesisBridge::new();
        match bridge.registry_clone() {
            Ok(reg) => reg
                .get_by_lemma(lemma)
                .map(|e| e.get_property("example_cases").is_some())
                .unwrap_or(false),
            Err(_) => false,
        }
    }

    fn dispatch(&mut self, query: &str, req: &SynthesisRequirement, gate: bool) -> AgentQueryResult {
        // TENSOR REACH (NL-BRIDGE-3B-TENSOR-FORWARD): consult the tensor route
        // FIRST, before workflow routing, so a forward-inference request
        // (relu/sigmoid/softmax/transpose/matmul — vocab reflected from
        // `crate::tensor::ops`) is solved by CODEGEN and a TRAINING request is
        // REFUSED (training is a no-op here), regardless of which generic
        // workflow the comprehension layer assigned. Non-tensor requests fall
        // through unchanged, so every prior type/route is untouched.
        match crate::tensor_nl::classify(query) {
            crate::tensor_nl::TensorRouteOutcome::RefuseTraining => {
                return self.refuse_tensor_training();
            }
            crate::tensor_nl::TensorRouteOutcome::Forward { lemma, code } => {
                return self.run_tensor_forward(&lemma, &code);
            }
            crate::tensor_nl::TensorRouteOutcome::NotTensor => {}
        }

        // A concretely-RESOLVED registry op that the intent classifier left as
        // `Unknown` (so route_from_workflow sends it to ToolExplore/Clarification)
        // is a synthesis request in all but the workflow label — e.g. a phrase-
        // resolved op ("absolute value" -> abs) the workflow cue-scanner missed.
        // Promote ONLY when the function_name is a KNOWN registry op carrying its
        // own canonical example_cases (a genuinely-resolved op, NOT a prose-
        // fabricated name whose self-derived examples mis-comprehend the task):
        // that trust signal is what separates abs (resolves, solves) from a
        // mis-comprehended complex task (which must stay REFUSED, not ship wrong).
        // run_synthesis still STRICT-VERIFIES (gate=true).
        let resolved_registry_op = !req.function_name.is_empty()
            && !req.examples.is_empty()
            && self
                .registry_op_has_examples(&req.function_name);
        let route = match route_from_workflow(&req.workflow) {
            QueryRoute::ToolExplore | QueryRoute::Clarification if resolved_registry_op => {
                QueryRoute::SynthesizeFunction
            }
            r => r,
        };
        match route {
            QueryRoute::SynthesizeFunction => self.run_synthesis(query, req, gate),
            QueryRoute::RepoRepair => self.run_repo_repair(query, req),
            QueryRoute::ExplainCode | QueryRoute::CodeReview => self.run_explain(query, req),
            QueryRoute::GreenfieldProject => self.run_greenfield(query, req, gate),
            QueryRoute::ToolExplore => self.handle_explore(query),
            QueryRoute::Clarification => self.handle_explore(query),
        }
    }

    fn run_synthesis(
        &mut self,
        query: &str,
        req: &SynthesisRequirement,
        gate: bool,
    ) -> AgentQueryResult {
        let bridge = LinguigenesisBridge::new();
        let intent = CodingIntent::from_requirement(req);

        // FIX A (portability): if the coding registry failed to load (location-
        // independent probing exhausted), the agent has zero operations and would
        // otherwise silently report every request as "unknown / clarification".
        // Surface the load failure EXPLICITLY instead of pretending the op is
        // simply not understood.
        if let Some(load_err) = bridge.registry_load_error() {
            return AgentQueryResult {
                route: QueryRoute::SynthesizeFunction,
                success: false,
                response: format!("registry load error: {load_err}"),
                workflow: workflow_label(&req.workflow),
                clarification_questions: Vec::new(),
                synthesis_method: None,
                repo_result: None,
                tool_trace: Vec::new(),
            };
        }

        // MULTI-FILE front door (NL-MULTIFILE-PROGRAM): if linguigenesis-core
        // splits the request into >=2 independent component functions, synthesize
        // each and write a real multi-file crate (src/<module>.rs per component +
        // src/lib.rs + Cargo.toml) to the sandbox root. Single-function (and
        // single-pipeline) requests yield a 1-component plan here, so this branch
        // is skipped and behaviour is unchanged. The split is structural
        // (comprehend_project), not a phrase→file table.
        if let Ok((solved, skipped)) = bridge.synthesize_project(query) {
            if solved.len() >= 2 {
                return self.write_multifile_program(req, solved, skipped);
            }
        }

        // COMPOSITIONAL front door FIRST: if linguigenesis-core comprehends this
        // request as a multi-op array pipeline (req.pipeline populated), build and
        // strict-verify it through the existing pipeline machinery — mirroring
        // synthesize_from_description's pipeline-first precedence
        // (linguigenesis_bridge.rs:511). Returns None for single-op requests, so
        // single-op behaviour is unchanged.
        // A compose-pipeline forces an ARRAY-input signature (`fn f(a: [i64])`).
        // Take it ONLY when the task's own examples are not DEFINITIVELY scalar: a
        // scalar-input task ("product of the odd DIGITS of n" — every doctest input
        // `in:[5]` is an Int) must NOT be re-typed as an array reduce, which
        // mis-models the input and ships a phantom array pipeline, starving the
        // scalar digit-decompose search. Empty/array/mixed examples preserve prior
        // behavior, so genuine array pipelines are untouched.
        let scalar_typed_task = !req.examples.is_empty()
            && req.examples.iter().all(|ex| {
                ex.inputs.iter().all(|v| {
                    matches!(
                        v,
                        linguigenesis_core::coding_requirements::LiteralValue::Int(_)
                    )
                })
            });
        let composed = if scalar_typed_task {
            None
        } else {
            bridge.try_compose_pipeline(query)
        };
        let synthesis = match composed {
            Some(Ok(outcome)) => outcome.into_solve_result(),
            Some(Err(error)) => {
                return AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: false,
                    response: error,
                    workflow: workflow_label(&req.workflow),
                    clarification_questions: Vec::new(),
                    synthesis_method: None,
                    repo_result: None,
                    tool_trace: Vec::new(),
                };
            }
            None => {
                // FIX C (must-refuse): SINGLE-OP path. `handle_query` reached here
                // via `comprehend_outcome`, which BYPASSES the emergent fail-closed
                // gate baked into `nl_to_requirement`. Consult that same gate now
                // (domain / type / operation-identity + the >=2-example floor —
                // resolver score & registry signatures, NOT a phrase blocklist). If
                // it fires, REFUSE with a clarification instead of synthesizing
                // out-of-domain / signature-mismatched / thinly-specified code.
                // Pipeline & inline-example requests were exempted upstream (handled
                // by `try_compose_pipeline` above / inside the gate), so genuine
                // compositions and demonstrated I/O are untouched.
                if gate {
                if let Some(reason) = bridge.fail_closed_reason(query, req) {
                    return AgentQueryResult {
                        route: QueryRoute::Clarification,
                        success: false,
                        response: format!(
                            "cannot synthesize confidently (fail-closed): {reason}"
                        ),
                        workflow: workflow_label(&req.workflow),
                        clarification_questions: vec![reason],
                        synthesis_method: None,
                        repo_result: None,
                        tool_trace: Vec::new(),
                    };
                }
                }
                match bridge.synthesize_from_requirement(req, Some(&intent.function_name)) {
                    Ok(result) => result,
                    Err(error) => {
                        return AgentQueryResult {
                            route: QueryRoute::SynthesizeFunction,
                            success: false,
                            response: error,
                            workflow: workflow_label(&req.workflow),
                            clarification_questions: Vec::new(),
                            synthesis_method: None,
                            repo_result: None,
                            tool_trace: Vec::new(),
                        };
                    }
                }
            }
        };
        let mut tool_trace = Vec::new();
        if synthesis.success {
            let filename = format!("synth_{}.mog", intent.function_name);
            if let Ok(out) = self.tools.invoke(
                "fs",
                &ToolCall::new("write")
                    .arg("path", filename.clone())
                    .arg("content", synthesis.code.clone()),
            ) {
                tool_trace.push((format!("fs.write:{filename}"), out.content));
            }
        }
        // INTERACTIVE-APP front door: when the request wants a RUNNABLE page and the
        // logic synthesized+verified, wrap the PROVEN function in an interactive widget
        // (transpile → strip → emit → re-verify wiring) and write it as an .html file.
        // The response becomes the working app; its behavior IS the verified function.
        // A wrap failure never downgrades the solve — the function result still stands.
        let mut app_html: Option<String> = None;
        if synthesis.success && crate::site::wants_interactive_app(query) {
            // Wire the name + params the SYNTHESIZED CODE actually defines — the solver
            // may rename (e.g. "double" resolves to the library op `times_two`), and the
            // widget's runner must call the real fn or verify_widget fails.
            let code_fn = crate::site::fn_name_from_mog(&synthesis.code)
                .unwrap_or_else(|| intent.function_name.clone());
            let params = crate::site::params_from_signature(&synthesis.code);
            let param_refs: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
            if let Ok(html) = crate::site::build_widget_from_mog(
                &intent.function_name,
                &code_fn,
                &param_refs,
                &synthesis.code,
            ) {
                let fname = format!("app_{}.html", intent.function_name);
                if let Ok(out) = self.tools.invoke(
                    "fs",
                    &ToolCall::new("write")
                        .arg("path", fname.clone())
                        .arg("content", html.clone()),
                ) {
                    tool_trace.push((format!("fs.write:{fname}"), out.content));
                }
                app_html = Some(html);
            }
        }
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: synthesis.success,
            response: if synthesis.success {
                app_html.clone().unwrap_or_else(|| synthesis.code.clone())
            } else {
                synthesis
                    .error
                    .clone()
                    .unwrap_or_else(|| "synthesis failed".to_string())
            },
            workflow: workflow_label(&req.workflow),
            clarification_questions: Vec::new(),
            synthesis_method: Some(synthesis.method.clone()),
            repo_result: None,
            tool_trace,
        }
    }

    /// LEARN-ON-THE-FLY (UNWALL-4): dispatch a parsed teach/reuse intake through
    /// the EXISTING self-extension + persistence machinery.
    ///
    /// * `TeachByExamples` / `TeachByComposition` → `learn_nl::teach_*`, which
    ///   build a [`LearnRequest`](crate::self_improve::extend::LearnRequest) and
    ///   call `self_extend` (synthesize → regression-gate → persist on green). The
    ///   op is durably stored, so a SEPARATE later process reloads it (gated) via
    ///   `Engine::new`.
    /// * `Reuse` → resolve the learned op from a FRESH `Engine::new` (which
    ///   re-gates the reloaded store) and evaluate it. A name not in the persisted
    ///   store is reported honestly as not-learned.
    /// * `Unparseable` → honest refusal; never fabricate a spec.
    ///
    /// Persistence is fenced by the same env contract the substrate uses: when
    /// `NCPU_COMPONENTS_PATH` is empty the store is a no-op (tests), and when it
    /// points at a real file a learned op survives across CLI invocations.
    fn run_learn_intake(&mut self, intake: crate::learn_nl::LearnIntake) -> AgentQueryResult {
        use crate::learn_nl::{LearnIntake, LearnOutcome};
        let outcome: LearnOutcome = match intake {
            LearnIntake::TeachByExamples { name, examples } => {
                // Teach against a FRESH engine so any previously-learned ops are
                // reloaded (gated) and visible — and so the gate runs against the
                // current durable state.
                let engine = crate::comprehension::Engine::new();
                crate::learn_nl::teach_by_examples(&engine, &name, &examples)
            }
            LearnIntake::TeachByComposition { name, steps } => {
                let engine = crate::comprehension::Engine::new();
                crate::learn_nl::teach_by_composition(&engine, &name, &steps)
            }
            LearnIntake::Reuse { name, arg } => {
                // A FRESH engine performs the gated reload of every persisted
                // component — this is the cross-invocation resolution path.
                let engine = crate::comprehension::Engine::new();
                crate::learn_nl::reuse(&engine, &name, arg)
            }
            LearnIntake::Unparseable(reason) => LearnOutcome {
                success: false,
                message: format!("cannot learn (unparseable teach/reuse request): {reason}"),
                method: None,
            },
            // `NotLearn` is filtered out by the caller (`handle_query`) before this
            // method is reached; handled here only to keep the match exhaustive.
            LearnIntake::NotLearn => LearnOutcome {
                success: false,
                message: "not a learn/reuse request".to_string(),
                method: None,
            },
        };
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: outcome.success,
            response: outcome.message,
            workflow: "learn-on-the-fly".to_string(),
            clarification_questions: Vec::new(),
            synthesis_method: outcome.method,
            repo_result: None,
            tool_trace: Vec::new(),
        }
    }

    /// REFERENCE SYNTHESIS (UNWALL-3): synthesize a program equivalent to the
    /// supplied reference implementation. The reference's behavior IS the spec:
    /// [`crate::agent::coding_intent::Spec::Reference`] reduces to a `Problem`
    /// whose seed I/O examples are MANUFACTURED by running the reference and whose
    /// `reference_code` stays set, so [`crate::solver::solve_problem`] SEARCHES for
    /// a program that strict-verifies against fresh inputs run through the
    /// reference ([`crate::benchmark::generated_holdouts`]). The synthesized code
    /// is NOT a copy of the reference text — it is a solver result that matches the
    /// reference's behavior. If the reference cannot be parsed/run into a Problem,
    /// refuse honestly (never fabricate).
    fn run_reference_synthesis(
        &mut self,
        name: &str,
        signature: &str,
        reference_code: &str,
    ) -> AgentQueryResult {
        use crate::agent::coding_intent::Spec;
        let spec = Spec::Reference {
            name: name.to_string(),
            signature: signature.to_string(),
            code: reference_code.to_string(),
        };
        let problem = match spec.to_problem() {
            Ok(p) => p,
            Err(error) => {
                // The reference could not be parsed/run into a Problem (e.g. it
                // errored on every sampled input, or its signature is
                // unsampleable). Honest refusal — no fabricated success.
                return AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: false,
                    response: format!(
                        "cannot synthesize from reference (intake refused): {error}"
                    ),
                    workflow: "reference.synthesize".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("reference-intake-refused".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                };
            }
        };

        let solved = crate::solver::solve_problem(&problem);

        // Defense in depth: even on a reported solve, strict-verify the result
        // against the reference-derived holdouts so a "success" is always backed by
        // differential agreement with the reference, never just the seed examples.
        let verified = solved.success
            && crate::runtime::verify_problem_code_strict(&problem, &solved.code).is_ok();

        let mut tool_trace = Vec::new();
        if verified {
            let filename = format!("synth_{name}.mog");
            if let Ok(out) = self.tools.invoke(
                "fs",
                &ToolCall::new("write")
                    .arg("path", filename.clone())
                    .arg("content", solved.code.clone()),
            ) {
                tool_trace.push((format!("fs.write:{filename}"), out.content));
            }
        }

        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: verified,
            response: if verified {
                solved.code.clone()
            } else {
                solved
                    .error
                    .clone()
                    .unwrap_or_else(|| {
                        "reference-equivalent synthesis failed (no candidate matched the \
                         reference on the held-out inputs)"
                            .to_string()
                    })
            },
            workflow: "reference.synthesize".to_string(),
            clarification_questions: Vec::new(),
            synthesis_method: Some(format!("reference-intake:{}", solved.method)),
            repo_result: None,
            tool_trace,
        }
    }

    /// P2C-PROMPT-TO-CONTRACT: synthesize from a comprehended scalar composition.
    /// Emits a runnable reference for the resolved chain, then REUSES the existing
    /// reference path (`run_reference_synthesis` → problem_from_reference →
    /// solve_problem → strict-verify). The emitted reference's behaviour is the
    /// spec; the example pairs + holdouts are manufactured by running it (zero
    /// human examples). If the chain cannot be emitted (a primitive fails to
    /// synthesize) this refuses honestly rather than fabricating success.
    fn run_compositional_synthesis(
        &mut self,
        bridge: &LinguigenesisBridge,
        name: &str,
        signature: &str,
        chain: &[crate::reference_nl::CompositionalStep],
    ) -> AgentQueryResult {
        let reference_code = match bridge.emit_scalar_reference(name, chain) {
            Ok(code) => code,
            Err(error) => {
                return AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: false,
                    response: format!(
                        "cannot emit a reference for the comprehended composition: {error}"
                    ),
                    workflow: "compositional.synthesize".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("compositional-emit-refused".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                };
            }
        };
        // REUSE the reference door unchanged: it manufactures examples by running
        // the emitted reference, solves, and strict-verifies against fresh
        // reference-labelled holdouts.
        let mut result = self.run_reference_synthesis(name, signature, &reference_code);
        result.workflow = "compositional.synthesize".to_string();
        if let Some(method) = result.synthesis_method.take() {
            result.synthesis_method = Some(method.replacen("reference-intake", "compositional", 1));
        }
        result
    }

    /// P2C WIDEN (BUILD-A): finish an ARRAY/STRING composition once its reference
    /// has been emitted. REUSES the reference door unchanged
    /// (`run_reference_synthesis` → problem_from_reference → solve_problem →
    /// strict-verify); the emitted reference's behaviour is the spec and the
    /// examples + holdouts are manufactured by running it (zero human examples).
    /// An emit failure (a primitive that would not synthesize, or an
    /// unclassifiable fold) refuses honestly rather than fabricating success.
    fn run_emitted_compositional(
        &mut self,
        name: &str,
        signature: &str,
        reference: Result<String, String>,
    ) -> AgentQueryResult {
        let reference_code = match reference {
            Ok(code) => code,
            Err(error) => {
                return AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success: false,
                    response: format!(
                        "cannot emit a reference for the comprehended composition: {error}"
                    ),
                    workflow: "compositional.synthesize".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some("compositional-emit-refused".to_string()),
                    repo_result: None,
                    tool_trace: Vec::new(),
                };
            }
        };
        let mut result = self.run_reference_synthesis(name, signature, &reference_code);
        result.workflow = "compositional.synthesize".to_string();
        if let Some(method) = result.synthesis_method.take() {
            result.synthesis_method = Some(method.replacen("reference-intake", "compositional", 1));
        }
        result
    }

    /// HONEST REFUSAL: a confirmed scalar `then`-composition contained an atomic
    /// step that does not resolve to a primitive with an emittable body. Refuse
    /// (clarify) rather than fabricate a contract from a half-understood
    /// description (the soundness guard).
    fn refuse_compositional(&self, reason: &str) -> AgentQueryResult {
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: false,
            response: format!(
                "cannot synthesize from description (compositional intake refused): {reason}"
            ),
            workflow: "compositional.synthesize".to_string(),
            clarification_questions: Vec::new(),
            synthesis_method: Some("compositional-unresolvable".to_string()),
            repo_result: None,
            tool_trace: Vec::new(),
        }
    }

    /// HONEST REFUSAL: the request pointed at a reference (fenced code / a
    /// `behaves like`-style marker) but no runnable `fn NAME(params) -> RET { ... }`
    /// could be extracted. Refuse rather than fabricate a spec.
    fn refuse_unparseable_reference(&self, reason: &str) -> AgentQueryResult {
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: false,
            response: format!("cannot synthesize from reference (unparseable): {reason}"),
            workflow: "reference.synthesize".to_string(),
            clarification_questions: Vec::new(),
            synthesis_method: Some("reference-intake-unparseable".to_string()),
            repo_result: None,
            tool_trace: Vec::new(),
        }
    }

    /// HONEST REFUSAL: tensor TRAINING is a no-op in this engine
    /// (`Trainer::train` backprop is a TODO, autodiff is disconnected), so a
    /// 'train a model' / 'fit' / 'backprop' request is refused rather than
    /// emitting code that pretends to learn. success = false.
    fn refuse_tensor_training(&self) -> AgentQueryResult {
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: false,
            response: "cannot train (no-op): tensor TRAINING is unimplemented in this engine \
                       (Trainer::train backprop is a TODO and autodiff is disconnected). \
                       Forward-inference ops (relu/sigmoid/softmax/transpose/matmul) ARE supported."
                .to_string(),
            workflow: "tensor.forward".to_string(),
            clarification_questions: Vec::new(),
            synthesis_method: Some("tensor-train-refused".to_string()),
            repo_result: None,
            tool_trace: Vec::new(),
        }
    }

    /// TENSOR FORWARD CODEGEN: write the emitted `crate::tensor`-calling program
    /// (a self-contained crate with a path dep on the canonical `mog_synth`
    /// crate) and verify it through the cargo-check compile gate. Reports
    /// success ONLY when the gate is clean (the emitted code genuinely
    /// type-checks + links against the real engine op).
    fn run_tensor_forward(&mut self, lemma: &str, code: &str) -> AgentQueryResult {
        let files = crate::tensor_nl::tensor_crate_files(lemma, code);
        match crate::agent::repo::write_tensor_program(&self.root, &files) {
            Ok(outcome) => {
                let mut tool_trace: Vec<(String, String)> = outcome
                    .written
                    .iter()
                    .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                    .collect();
                let (success, note) = match &outcome.compile {
                    crate::agent::repo::CompileStatus::Ok => {
                        tool_trace.push(("cargo.check".to_string(), "ok".to_string()));
                        (true, String::new())
                    }
                    crate::agent::repo::CompileStatus::Failed(err) => {
                        tool_trace.push(("cargo.check".to_string(), "failed".to_string()));
                        (false, format!("\ncompile gate FAILED:\n{err}"))
                    }
                    crate::agent::repo::CompileStatus::Unverified(why) => {
                        tool_trace.push(("cargo.check".to_string(), "unverified".to_string()));
                        (false, format!("\ncompile gate UNVERIFIED (cargo unavailable): {why}"))
                    }
                };
                let mut response = format!(
                    "tensor forward-inference program (op `{lemma}`, codegen → crate::tensor):\n{code}"
                );
                response.push_str(&note);
                AgentQueryResult {
                    route: QueryRoute::SynthesizeFunction,
                    success,
                    response,
                    workflow: "tensor.forward".to_string(),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some(format!("tensor-forward-codegen:{lemma}")),
                    repo_result: None,
                    tool_trace,
                }
            }
            Err(error) => AgentQueryResult {
                route: QueryRoute::SynthesizeFunction,
                success: false,
                response: format!("tensor program write failed: {error}"),
                workflow: "tensor.forward".to_string(),
                clarification_questions: Vec::new(),
                synthesis_method: None,
                repo_result: None,
                tool_trace: Vec::new(),
            },
        }
    }

    /// Write a synthesized multi-component program to disk and report the paths.
    fn write_multifile_program(
        &mut self,
        req: &SynthesisRequirement,
        solved: Vec<(String, crate::solver::SolveResult)>,
        skipped: Vec<String>,
    ) -> AgentQueryResult {
        let methods: Vec<String> = solved.iter().map(|(_, r)| r.method.clone()).collect();
        let components: Vec<(String, String)> = solved
            .into_iter()
            .map(|(name, r)| (name, r.code))
            .collect();
        let pkg = req
            .function_name
            .is_empty()
            .then(|| "generated".to_string())
            .unwrap_or_else(|| req.function_name.clone());
        match crate::agent::repo::write_synthesized_project(&self.root, &pkg, &components) {
            Ok(outcome) => {
                let mut tool_trace: Vec<(String, String)> = outcome
                    .written
                    .iter()
                    .map(|p| (format!("fs.write:{p}"), "ok".to_string()))
                    .collect();
                // (D) Report success:true ONLY when the compile gate is clean.
                let (success, gate_note) = match &outcome.compile {
                    crate::agent::repo::CompileStatus::Ok => {
                        tool_trace.push(("cargo.check".to_string(), "ok".to_string()));
                        (true, String::new())
                    }
                    crate::agent::repo::CompileStatus::Failed(err) => {
                        tool_trace.push(("cargo.check".to_string(), "failed".to_string()));
                        (false, format!("\ncompile gate FAILED:\n{err}"))
                    }
                    crate::agent::repo::CompileStatus::Unverified(why) => {
                        tool_trace.push(("cargo.check".to_string(), "unverified".to_string()));
                        (false, format!("\ncompile gate UNVERIFIED (cargo unavailable): {why}"))
                    }
                };
                let mut response = format!(
                    "wrote {}-component multi-file program:\n{}",
                    components.len(),
                    outcome
                        .written
                        .iter()
                        .map(|p| format!("  {p}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                if !skipped.is_empty() {
                    response.push_str("\nskipped:\n");
                    for s in &skipped {
                        response.push_str(&format!("  {s}\n"));
                    }
                }
                response.push_str(&gate_note);
                AgentQueryResult {
                    route: QueryRoute::GreenfieldProject,
                    success,
                    response,
                    workflow: workflow_label(&req.workflow),
                    clarification_questions: Vec::new(),
                    synthesis_method: Some(methods.join("+")),
                    repo_result: None,
                    tool_trace,
                }
            }
            Err(error) => AgentQueryResult {
                route: QueryRoute::GreenfieldProject,
                success: false,
                response: format!("multi-file write failed: {error}"),
                workflow: workflow_label(&req.workflow),
                clarification_questions: Vec::new(),
                synthesis_method: None,
                repo_result: None,
                tool_trace: Vec::new(),
            },
        }
    }

    fn run_repo_repair(&mut self, query: &str, req: &SynthesisRequirement) -> AgentQueryResult {
        let intent = CodingIntent::from_requirement(req);
        let test_command = default_repo_test_command(&self.root);
        let spec = CodeTaskSpec::from_nl(
            self.root.to_string_lossy(),
            query,
            intent,
            test_command,
            vec!["src/**".into(), "tests/**".into()],
            4,
        );
        let mut agent = RepoAgent::new(&self.root, self.policy.clone());
        let repo_result = agent.run(&spec);
        let success = repo_result.success;
        let response = if success {
            format!(
                "repair succeeded after {} iterations (phases: {})",
                repo_result.repair_iterations,
                repo_result.phases_completed.join(" → ")
            )
        } else {
            format!(
                "repair failed after {} iterations (phases: {})\n{}\n\n{}",
                repo_result.repair_iterations,
                if repo_result.phases_completed.is_empty() {
                    "none".to_string()
                } else {
                    repo_result.phases_completed.join(" → ")
                },
                repo_result
                    .error
                    .clone()
                    .unwrap_or_else(|| "repair did not satisfy oracle".to_string()),
                repair_capability_notes()
            )
        };
        AgentQueryResult {
            route: QueryRoute::RepoRepair,
            success,
            response,
            workflow: workflow_label(&req.workflow),
            clarification_questions: Vec::new(),
            synthesis_method: None,
            repo_result: Some(repo_result),
            tool_trace: Vec::new(),
        }
    }

    fn run_explain(&mut self, query: &str, req: &SynthesisRequirement) -> AgentQueryResult {
        let mut tool_trace = Vec::new();
        let index = match RepoIndex::build(&self.root, &self.policy) {
            Ok(index) => index,
            Err(error) => {
                return AgentQueryResult {
                    route: QueryRoute::ExplainCode,
                    success: false,
                    response: error,
                    workflow: workflow_label(&req.workflow),
                    clarification_questions: Vec::new(),
                    synthesis_method: None,
                    repo_result: None,
                    tool_trace,
                };
            }
        };
        let paths = retrieve_paths(&index, query, 8);
        let mut sections = Vec::new();
        let knowledge = knowledge_notes(req);
        if !knowledge.is_empty() {
            sections.push(knowledge);
        }
        for path in &paths {
            if let Ok(out) = self.tools.invoke(
                "fs",
                &ToolCall::new("read").arg("path", path.clone()),
            ) {
                tool_trace.push((format!("fs.read:{path}"), truncate(&out.content, 120)));
                sections.push(format!("### {path}\n{}", truncate(&out.content, 2000)));
            }
        }
        if let Ok(out) = self.tools.invoke("git", &ToolCall::new("status")) {
            tool_trace.push(("git.status".into(), truncate(&out.content, 120)));
            sections.push(format!("### git status\n{}", truncate(&out.content, 800)));
        }
        let response = if sections.is_empty() {
            format!(
                "indexed {} files; no matching paths for query",
                index.files.len()
            )
        } else {
            sections.join("\n\n")
        };
        AgentQueryResult {
            route: if req.workflow == CodingWorkflow::CodeReview {
                QueryRoute::CodeReview
            } else {
                QueryRoute::ExplainCode
            },
            success: true,
            response,
            workflow: workflow_label(&req.workflow),
            clarification_questions: Vec::new(),
            synthesis_method: None,
            repo_result: None,
            tool_trace,
        }
    }

    fn run_greenfield(
        &mut self,
        query: &str,
        req: &SynthesisRequirement,
        gate: bool,
    ) -> AgentQueryResult {
        let synthesis = self.run_synthesis(query, req, gate);
        if !synthesis.success {
            return synthesis;
        }
        let mut tool_trace = synthesis.tool_trace;
        if let Ok(out) = self.tools.invoke(
            "fs",
            &ToolCall::new("write")
                .arg("path", "src/generated.rs")
                .arg("content", synthesis.response.clone()),
        ) {
            tool_trace.push(("fs.write:src/generated.rs".into(), out.content));
        }
        AgentQueryResult {
            route: QueryRoute::GreenfieldProject,
            success: true,
            response: format!(
                "greenfield scaffold written to src/generated.rs\n\n{}",
                truncate(&synthesis.response, 500)
            ),
            workflow: workflow_label(&req.workflow),
            clarification_questions: Vec::new(),
            synthesis_method: synthesis.synthesis_method,
            repo_result: None,
            tool_trace,
        }
    }

    fn handle_explore(&mut self, query: &str) -> AgentQueryResult {
        let bridge = LinguigenesisBridge::new();
        if let Ok(outcome) = bridge.comprehend_outcome(query) {
            match outcome {
                ComprehensionOutcome::Ready(req) if req.workflow != CodingWorkflow::Unknown => {
                    return self.dispatch(query, &req, true);
                }
                ComprehensionOutcome::Ready(req) => {
                    if let Ok(registry) = bridge.registry_clone() {
                        let questions = build_clarifications(&req, &registry);
                        if !questions.is_empty() {
                            self.pending = Some(PendingQuery {
                                query: query.to_string(),
                                partial: req.clone(),
                                questions: questions.clone(),
                                answers: Vec::new(),
                            });
                            return self.clarification_result(query, &req, &questions);
                        }
                    }
                }
                ComprehensionOutcome::NeedsClarification(req, questions) => {
                    self.pending = Some(PendingQuery {
                        query: query.to_string(),
                        partial: req.clone(),
                        questions: questions.clone(),
                        answers: Vec::new(),
                    });
                    return self.clarification_result(query, &req, &questions);
                }
            }
        }

        // TERMINAL FALLTHROUGH: comprehension could not classify this as any
        // buildable workflow and produced no clarifying questions. If the prompt
        // READS as a construction request, refuse HONESTLY — never dress a repo
        // file listing up as a successful build (the confident-wrong failure the
        // audit flagged). Genuine informational/exploration prompts fall through
        // to the repo listing below, which for them is a real answer.
        if has_build_intent(query) {
            return AgentQueryResult {
                route: QueryRoute::Clarification,
                success: false,
                response: format!(
                    "I couldn't confidently understand \"{}\" as something I can build \
                     or compute — nothing resolved to a known operation, component, or \
                     artifact. Try naming a concrete function with an example (e.g. \"a \
                     function f where f(2)=4\"), or a known artifact: a website/page, an \
                     api, or a component like a counter or a stack.",
                    truncate(query, 100)
                ),
                workflow: "unknown".into(),
                clarification_questions: Vec::new(),
                synthesis_method: None,
                repo_result: None,
                tool_trace: Vec::new(),
            };
        }

        let mut tool_trace = Vec::new();
        if let Ok(out) = self.tools.invoke("git", &ToolCall::new("status")) {
            tool_trace.push(("git.status".into(), truncate(&out.content, 120)));
        }
        if let Ok(out) = self.tools.invoke("fs", &ToolCall::new("list").arg("path", "src")) {
            tool_trace.push(("fs.list:src".into(), truncate(&out.content, 120)));
        }
        let index = match RepoIndex::build(&self.root, &self.policy) {
            Ok(index) => index,
            Err(error) => {
                return AgentQueryResult {
                    route: QueryRoute::ToolExplore,
                    success: false,
                    response: error,
                    workflow: "explore".into(),
                    clarification_questions: Vec::new(),
                    synthesis_method: None,
                    repo_result: None,
                    tool_trace,
                };
            }
        };
        let paths = retrieve_paths(&index, query, 4);
        for path in paths {
            if let Ok(out) = self
                .tools
                .invoke("fs", &ToolCall::new("read").arg("path", path.clone()))
            {
                tool_trace.push((format!("fs.read:{path}"), truncate(&out.content, 80)));
            }
        }
        AgentQueryResult {
            route: QueryRoute::ToolExplore,
            success: true,
            response: format!(
                "explored repository ({} indexed files); tool trace: {}",
                index.files.len(),
                tool_trace
                    .iter()
                    .map(|(k, _)| k.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            workflow: "explore".into(),
            clarification_questions: Vec::new(),
            synthesis_method: None,
            repo_result: None,
            tool_trace,
        }
    }
}

/// True if the prompt reads as a request to BUILD/PRODUCE something (an
/// imperative construction verb), vs an informational/exploration question. Used
/// only to decide, at the terminal explore fallthrough, whether an unclassified
/// prompt should refuse honestly (a failed build) or get the repo listing (a
/// genuine inspection). Discourse-type detection, not domain vocabulary.
fn has_build_intent(query: &str) -> bool {
    const BUILD_VERBS: [&str; 13] = [
        "build", "make", "create", "implement", "write", "generate", "develop", "design", "add",
        "code", "program", "produce", "scaffold",
    ];
    let lower = query.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|tok| BUILD_VERBS.contains(&tok))
}

fn route_from_workflow(workflow: &CodingWorkflow) -> QueryRoute {
    match workflow {
        CodingWorkflow::SynthesizeFunction => QueryRoute::SynthesizeFunction,
        CodingWorkflow::BugFix | CodingWorkflow::AddTests | CodingWorkflow::Refactor
        | CodingWorkflow::UpdateDependencies | CodingWorkflow::MigrateApi | CodingWorkflow::ResumeTask => {
            QueryRoute::RepoRepair
        }
        CodingWorkflow::ExplainCode => QueryRoute::ExplainCode,
        CodingWorkflow::CodeReview => QueryRoute::CodeReview,
        CodingWorkflow::GreenfieldProject => QueryRoute::GreenfieldProject,
        CodingWorkflow::Unknown => QueryRoute::ToolExplore,
    }
}

fn workflow_label(workflow: &CodingWorkflow) -> String {
    match workflow {
        CodingWorkflow::SynthesizeFunction => "synthesize_function".into(),
        CodingWorkflow::BugFix => "bug_fix".into(),
        CodingWorkflow::AddTests => "add_tests".into(),
        CodingWorkflow::Refactor => "refactor".into(),
        CodingWorkflow::UpdateDependencies => "update_dependencies".into(),
        CodingWorkflow::MigrateApi => "migrate_api".into(),
        CodingWorkflow::CodeReview => "code_review".into(),
        CodingWorkflow::ExplainCode => "explain_code".into(),
        CodingWorkflow::GreenfieldProject => "greenfield_project".into(),
        CodingWorkflow::ResumeTask => "resume_task".into(),
        CodingWorkflow::Unknown => "unknown".into(),
    }
}

fn default_repo_test_command(root: &Path) -> String {
    if root.join("Cargo.toml").is_file() {
        "cargo test".to_string()
    } else {
        "cargo test".to_string()
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max).collect::<String>())
    }
}

fn repair_capability_notes() -> String {
    let reg = CapabilityRegistry::package_b_native_runtime();
    let mut lines = vec!["### Repair capability status (runtime registry)".to_string()];
    for name in [
        "repo_agent_closed_loop",
        "nl_synthesis_repair_proposer",
        "repo_workflow_runner",
    ] {
        if let Some(cap) = reg.get(name) {
            lines.push(format!(
                "- **{}**: {:?} — {} (conformance: {})",
                cap.name,
                cap.status,
                cap.evidence,
                cap.conformance_test.as_deref().unwrap_or("none")
            ));
        }
    }
    lines.push(
        "Probe a specific NL synthesis request via `agent_query`; route + synthesis_method report what fired."
            .to_string(),
    );
    lines.join("\n")
}

fn knowledge_notes(req: &SynthesisRequirement) -> String {
    let bridge = LinguigenesisBridge::new();
    let registry = match bridge.registry_clone() {
        Ok(registry) => registry,
        Err(_) => return String::new(),
    };
    let qa = KnowledgeQA::new(registry.clone());
    let mut lines = Vec::new();
    for entity_id in &req.evidence_entity_ids {
        if let Some(entity) = registry.get_entity(*entity_id) {
            if let Some(def) = qa.what_is(&entity.lemma) {
                lines.push(format!("- **{}**: {}", entity.lemma, def));
            }
            for (key, value) in &entity.properties {
                lines.push(format!("- **{}** ({}): {}", entity.lemma, key, value));
            }
            let capabilities = qa.what_can(*entity_id);
            if !capabilities.is_empty() {
                lines.push(format!(
                    "- **{}** patterns: {}",
                    entity.lemma,
                    capabilities.join(", ")
                ));
            }
            for rel in [
                RelationType::Synonym,
                RelationType::Similar,
                RelationType::Hypernym,
            ] {
                let related = qa.query_relations(*entity_id, rel.clone());
                if !related.is_empty() {
                    let names: Vec<_> = related.iter().map(|e| e.lemma.clone()).collect();
                    lines.push(format!(
                        "- **{}** {:?}: {}",
                        entity.lemma,
                        rel,
                        names.join(", ")
                    ));
                }
            }
        }
    }
    if lines.is_empty() {
        String::new()
    } else {
        format!("### Registry knowledge\n{}", lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::write_nl_fixture_crate;
    use std::fs;
    use std::sync::Mutex;

    static SESSION_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn temp_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("nsynth_session_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn session_routes_synthesis_query() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("synth");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result = session.handle_query("add two numbers");
        assert_eq!(result.route, QueryRoute::SynthesizeFunction);
        assert!(result.success);
        assert!(result.response.contains("a") || result.response.contains("+"));
        let _ = fs::remove_dir_all(root);
    }

    /// Agree-on-held-out helper: run two programs over a fresh, deterministic input
    /// set DISJOINT from any seeds/holdouts the solver used, and assert they agree.
    /// This is the un-gameable differential check — the synthesized program must
    /// match the reference on inputs neither was tuned on.
    fn agree_on_holdout(reference: &str, synthesized: &str, fn_name: &str, sig: &'static str) {
        let sig_owned: &'static str = sig;
        let problem = crate::benchmark::Problem {
            name: format!("{fn_name}_holdout_check"),
            category: "reference",
            description: "held-out agreement check",
            signature: sig_owned,
            ..Default::default()
        };
        // Inputs the solver never saw (range distinct from solver's [-64,64]).
        let mut agreed = 0usize;
        for x in [-500i64, -333, -77, 0, 13, 256, 999, 4321] {
            let args = vec![crate::benchmark::Value::Int(x)];
            let r = crate::runtime::benchmark_value_from_runtime(
                &crate::runtime::execute_function_for_problem(reference, fn_name, &args, &problem)
                    .expect("reference runs on held-out input"),
            )
            .expect("reference output representable");
            let s = crate::runtime::benchmark_value_from_runtime(
                &crate::runtime::execute_function_for_problem(
                    synthesized,
                    fn_name,
                    &args,
                    &problem,
                )
                .expect("synthesized runs on held-out input"),
            )
            .expect("synthesized output representable");
            assert_eq!(
                r, s,
                "reference and synthesized disagree on held-out input {x}"
            );
            agreed += 1;
        }
        assert!(agreed >= 8, "held-out agreement set must be non-empty");
    }

    /// UNWALL-3 ACCEPT (polynomial): a 'behaves like THIS: <polynomial fn>' CLI
    /// request synthesizes a verified-equivalent program — searched + strict-
    /// verified against fresh inputs RUN through the reference, NOT a verbatim copy
    /// of the reference text, agreeing on a held-out input set.
    #[test]
    fn session_reference_intake_polynomial_synthesizes_verified_equivalent() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("ref_poly");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());

        let reference = "fn f(x: i64) -> i64 { return x * x - x; }";
        let result = session
            .handle_query(&format!("make a function that behaves like THIS: {reference}"));

        assert_eq!(result.route, QueryRoute::SynthesizeFunction);
        assert!(
            result.success,
            "reference-equivalent synthesis must succeed: {:?}",
            result.response
        );
        // (a) The method is a REAL solver method, tagged as reference-intake.
        let method = result.synthesis_method.clone().unwrap_or_default();
        assert!(
            method.starts_with("reference-intake:"),
            "method must be reference-intake-tagged, got {method}"
        );
        let inner = method.trim_start_matches("reference-intake:");
        assert!(
            !inner.is_empty()
                && inner != "reference-intake-refused"
                && inner != "reference-intake-unparseable",
            "inner method must be a real solver method, got {inner}"
        );
        // (b) NOT a verbatim copy of the reference text.
        assert_ne!(
            result.response.trim(),
            reference.trim(),
            "synthesized code must not be a verbatim copy of the reference"
        );
        // (c) Agrees with the reference on a held-out input set.
        agree_on_holdout(reference, &result.response, "f", "fn f(x: i64) -> i64");
        let _ = fs::remove_dir_all(root);
    }

    /// UNWALL-3 ACCEPT (piecewise/abs): a second, structurally-distinct reference
    /// (branching abs) also synthesizes a verified-equivalent, non-copy program.
    #[test]
    fn session_reference_intake_piecewise_synthesizes_verified_equivalent() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("ref_abs");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());

        let reference = "fn g(a: i64) -> i64 { if a < 0 { return -a; } return a; }";
        let result = session
            .handle_query(&format!("synthesize a function equivalent to: {reference}"));

        assert_eq!(result.route, QueryRoute::SynthesizeFunction);
        assert!(
            result.success,
            "abs reference synthesis must succeed: {:?}",
            result.response
        );
        let method = result.synthesis_method.clone().unwrap_or_default();
        assert!(method.starts_with("reference-intake:"), "got {method}");
        assert_ne!(result.response.trim(), reference.trim());
        agree_on_holdout(reference, &result.response, "g", "fn g(a: i64) -> i64");
        let _ = fs::remove_dir_all(root);
    }

    /// UNWALL-3 ANTI-GAME (prior path): a plain NL request (NO embedded reference)
    /// must NOT be hijacked by the reference route — it routes through normal
    /// comprehension. Proves the reference intake fires ONLY on a structural
    /// reference signal, not on every request.
    #[test]
    fn session_plain_request_does_not_trigger_reference_route() {
        // classify() returns NotReference for a plain request, so handle_query
        // never enters run_reference_synthesis (workflow != "reference.synthesize").
        assert_eq!(
            crate::reference_nl::classify("add two numbers"),
            crate::reference_nl::ReferenceIntake::NotReference
        );
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("ref_plain");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result = session.handle_query("add two numbers");
        assert_ne!(
            result.workflow, "reference.synthesize",
            "a plain request must not route through reference intake"
        );
        assert!(result.success);
        let _ = fs::remove_dir_all(root);
    }

    /// UNWALL-3 ANTI-GAME (honest refusal): a request that signals a reference but
    /// supplies no runnable `fn` is REFUSED (success=false), not fabricated.
    #[test]
    fn session_unparseable_reference_is_refused_honestly() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("ref_refuse");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result =
            session.handle_query("make a function that behaves like this reference implementation");
        assert!(!result.success, "an unparseable reference must be refused");
        assert_eq!(result.workflow, "reference.synthesize");
        assert_eq!(
            result.synthesis_method.as_deref(),
            Some("reference-intake-unparseable")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_synthesizes_unseen_op_from_inline_examples() {
        // Full agent path: a function never named in the registry, demonstrated
        // only by inline examples, must route to synthesis and succeed.
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("inline_synth");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result = session.handle_query("build wibble(1)=3, wibble(2)=6, wibble(4)=12");
        assert_eq!(result.route, QueryRoute::SynthesizeFunction);
        assert!(result.success, "response={}", result.response);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_exposes_all_tool_capabilities() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("caps");
        fs::create_dir_all(&root).unwrap();
        let session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let caps = session.allowed_tool_capabilities();
        assert!(caps.iter().any(|c| c.starts_with("fs.")));
        assert!(caps.iter().any(|c| c == "shell.run"));
        assert!(caps.iter().any(|c| c == "git.status"));
        assert!(caps.iter().any(|c| c == "http.get"));
        assert!(caps.iter().any(|c| c == "database.select"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_tools_read_fixture_repo() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("explore");
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("fixture");
        let session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let list = session
            .invoke_tool("fs", &ToolCall::new("list").arg("path", "src"))
            .expect("list");
        assert!(list.content.contains("lib.rs"));
        let _ = session.invoke_tool("database", &ToolCall::new("list_tables"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_invoke_tool_roundtrip() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("invoke");
        fs::create_dir_all(&root).unwrap();
        let session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        session
            .invoke_tool(
                "fs",
                &ToolCall::new("write")
                    .arg("path", "note.txt")
                    .arg("content", "hello"),
            )
            .unwrap();
        let out = session
            .invoke_tool("fs", &ToolCall::new("read").arg("path", "note.txt"))
            .unwrap();
        assert_eq!(out.content, "hello");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_routes_explain_with_registry_knowledge() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("explain");
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("fixture");
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result = session.handle_query("explain the codebase");
        assert_eq!(result.route, QueryRoute::ExplainCode);
        assert!(result.success);
        assert!(result.response.contains("lib.rs"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_multi_turn_workflow_then_synthesis() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("multi");
        fs::create_dir_all(&root).unwrap();
        let mut session =
            CodingAgentSession::with_session_id(&root, GuardrailPolicy::default(), "multi".into());
        let first = session.handle_query("help with something");
        assert_eq!(first.route, QueryRoute::Clarification);
        let second = session
            .clarify_and_continue("implement a new function")
            .expect("workflow clarify");
        assert!(
            second.route == QueryRoute::Clarification || second.route == QueryRoute::SynthesizeFunction
        );
        if second.route == QueryRoute::Clarification {
            let third = session.clarify_and_continue("add").expect("op clarify");
            assert_eq!(third.route, QueryRoute::SynthesizeFunction);
            assert!(third.success);
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_open_ended_prompt_gets_clarification_not_blind_explore() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("open");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        let result = session.handle_query("something about code maybe");
        assert!(
            result.route == QueryRoute::Clarification || result.route == QueryRoute::ToolExplore
        );
        if result.route == QueryRoute::Clarification {
            assert!(!result.clarification_questions.is_empty());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unclassified_build_request_refuses_honestly() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("honest_refuse");
        fs::create_dir_all(&root).unwrap();
        let mut session = CodingAgentSession::new(&root, GuardrailPolicy::default());
        // A construction request nothing can classify (no web archetype, backend
        // concept, component, or resolvable op) must REFUSE honestly — never dress
        // a repo file listing up as a successful build.
        let r = session.handle_query("build a snake game with keyboard controls");
        assert!(!r.success, "unclassified build must not report success: {}", r.response);
        assert!(
            !r.response.contains("explored repository"),
            "must not present a file listing as a successful build: {}",
            r.response
        );
        assert!(has_build_intent("build a snake game"), "sanity: build intent detected");
        assert!(!has_build_intent("what does this project do"), "sanity: question is not build intent");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_clarify_and_continue_synthesis() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("clarify");
        fs::create_dir_all(&root).unwrap();
        let mut session =
            CodingAgentSession::with_session_id(&root, GuardrailPolicy::default(), "clarify".into());
        let first = session.handle_query("xyzqwerty qwerty qwerty");
        assert_eq!(first.route, QueryRoute::Clarification);
        assert!(session.has_pending_clarification());
        let second = session
            .clarify_and_continue("implement a function")
            .expect("workflow clarify");
        let third = if second.route == QueryRoute::Clarification {
            session.clarify_and_continue("add").expect("op clarify")
        } else {
            second
        };
        assert_eq!(third.route, QueryRoute::SynthesizeFunction);
        assert!(third.success);
        assert!(!session.has_pending_clarification());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_clarification_persist_and_resume_across_load() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("clarify_persist");
        fs::create_dir_all(&root).unwrap();
        let session_id = "clarify-resume";
        let policy = GuardrailPolicy::default();

        {
            let mut session =
                CodingAgentSession::with_session_id(&root, policy.clone(), session_id.into());
            let first = session.handle_query("xyzqwerty qwerty qwerty");
            assert_eq!(first.route, QueryRoute::Clarification);
            assert!(session.has_pending_clarification());
            let path = session_path(&root, session_id);
            assert!(path.is_file(), "clarification turn must persist snapshot");
            let snap = load_session_snapshot(&path).expect("read snapshot");
            assert!(snap.pending.is_some(), "snapshot must retain pending query");
        }

        let mut resumed =
            CodingAgentSession::load(&root, policy, session_id).expect("resume session");
        assert!(resumed.has_pending_clarification());
        let second = resumed
            .clarify_and_continue("implement a function")
            .expect("first clarify after resume");
        let third = if second.route == QueryRoute::Clarification {
            resumed.clarify_and_continue("add").expect("op clarify")
        } else {
            second
        };
        assert_eq!(third.route, QueryRoute::SynthesizeFunction);
        assert!(third.success);
        assert!(!resumed.has_pending_clarification());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn session_persist_and_resume() {
        let _guard = SESSION_TEST_LOCK.lock().unwrap();
        let root = temp_root("persist");
        fs::create_dir_all(&root).unwrap();
        let mut session =
            CodingAgentSession::with_session_id(&root, GuardrailPolicy::default(), "main".into());
        let result = session.handle_query("add two numbers");
        assert!(result.success);
        let path = session.persist(Some("add two numbers"), Some(&result)).expect("persist");
        assert!(path.is_file());
        let resumed = CodingAgentSession::load(&root, GuardrailPolicy::default(), "main")
            .expect("load");
        assert_eq!(resumed.session_id(), "main");
        assert_eq!(resumed.history_len, 1);
        let _ = fs::remove_dir_all(root);
    }
}
