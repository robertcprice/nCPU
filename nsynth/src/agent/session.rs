//! Universal coding-agent session: any NL query → registry workflow route + full tools.

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

        let route = route_from_workflow(&req.workflow);
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
        let synthesis = match bridge.try_compose_pipeline(query) {
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
        AgentQueryResult {
            route: QueryRoute::SynthesizeFunction,
            success: synthesis.success,
            response: if synthesis.success {
                synthesis.code.clone()
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
                "repair succeeded after {} iterations",
                repo_result.repair_iterations
            )
        } else {
            repo_result
                .error
                .clone()
                .unwrap_or_else(|| "repair did not satisfy oracle".to_string())
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
