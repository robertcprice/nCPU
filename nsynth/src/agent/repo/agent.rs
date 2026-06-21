use crate::agent::repo::trace::AgentTrace;
use crate::agent::repo::{
    CreditLedger, FailureAnalysis, FailureParser, GuardrailPolicy, HardnessMiner, HardnessProfile,
    HardnessTier, PatchGate, PatchGateResult, RepoSignal, RepoTaskKind, RepoTaskSpec,
};
use std::path::{Path, PathBuf};

pub struct RepairAgent {
    miner: HardnessMiner,
    gate: PatchGate,
    parser: FailureParser,
    credit: CreditLedger,
    trace: AgentTrace,
}

impl RepairAgent {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            miner: HardnessMiner::new(root),
            gate: PatchGate::default(),
            parser: FailureParser,
            credit: CreditLedger::default(),
            trace: AgentTrace::default(),
        }
    }

    pub fn with_policy(root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        Self {
            miner: HardnessMiner::new(root),
            gate: PatchGate::new(policy),
            parser: FailureParser,
            credit: CreditLedger::default(),
            trace: AgentTrace::default(),
        }
    }

    pub fn root(&self) -> &Path {
        self.miner.root()
    }

    pub fn scan_signals(&mut self) -> Result<Vec<RepoSignal>, std::io::Error> {
        let signals = self.miner.scan_repo()?;
        self.trace.push(
            "scan_repo",
            self.miner.root().to_string_lossy(),
            format!("signals={}", signals.len()),
            "ok",
        );
        Ok(signals)
    }

    pub fn propose_task(
        &mut self,
        id: impl Into<String>,
        kind: RepoTaskKind,
        issue: impl Into<String>,
        test_command: impl Into<String>,
        allowed_files: Vec<String>,
        max_iterations: usize,
    ) -> Result<RepoTaskSpec, std::io::Error> {
        let signals = self.scan_signals()?;
        let task = self.miner.propose_task(
            id,
            kind,
            issue,
            test_command,
            allowed_files,
            max_iterations,
            signals,
        );
        self.trace.push(
            "propose_task",
            task.issue.clone(),
            format!(
                "tier={:?} score={:.2}",
                task.hardness.tier(),
                task.hardness.score()
            ),
            "ok",
        );
        Ok(task)
    }

    pub fn validate_patch(&mut self, diff: &str, allowed_files: &[String]) -> PatchGateResult {
        let result = self.gate.validate_diff(diff, allowed_files);
        self.trace.push(
            "validate_patch",
            format!("allowed_files={allowed_files:?}"),
            format!("rejected={:?}", result.rejected),
            if result.allowed { "ok" } else { "rejected" }.to_string(),
        );
        result
    }

    pub fn parse_failure(&mut self, output: &str) -> FailureAnalysis {
        let analysis = self.parser.parse(output);
        self.trace.push(
            "parse_failure",
            output.lines().next().unwrap_or("").to_string(),
            format!("kind={:?}", analysis.kind),
            "ok",
        );
        analysis
    }

    pub fn assign_credit(
        &mut self,
        category: crate::agent::repo::CreditCategory,
        score: f64,
        evidence: impl Into<String>,
    ) {
        let evidence = evidence.into();
        self.credit.assign(category, score, evidence.clone());
        self.trace
            .push("assign_credit", format!("{category:?}"), evidence, "ok");
    }

    pub fn credit(&self) -> &CreditLedger {
        &self.credit
    }

    pub fn trace(&self) -> &AgentTrace {
        &self.trace
    }

    pub(crate) fn trace_mut(&mut self) -> &mut AgentTrace {
        &mut self.trace
    }

    pub fn hardness_for(&self, kind: RepoTaskKind, signals: &[RepoSignal]) -> HardnessProfile {
        HardnessProfile::from_signals(kind, signals)
    }

    pub fn tier_for(&self, kind: RepoTaskKind, signals: &[RepoSignal]) -> HardnessTier {
        self.hardness_for(kind, signals).tier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_agent_mines_and_validates() {
        let tmp = std::env::temp_dir().join(format!("nsynth_repair_agent_{}", std::process::id()));
        let _ = std::fs::create_dir_all(tmp.join("src"));
        let _ = std::fs::write(tmp.join("src/lib.rs"), "pub fn a() {}\n");
        let mut agent = RepairAgent::new(&tmp);
        let task = agent
            .propose_task(
                "task-1",
                RepoTaskKind::BugFix,
                "fix a",
                "cargo test",
                vec!["src/**".to_string()],
                3,
            )
            .unwrap();
        assert_eq!(task.id, "task-1");
        let diff =
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n-pub fn a() {}\n+pub fn a() -> i32 { 1 }\n";
        assert!(agent.validate_patch(diff, &["src/**".to_string()]).allowed);
        let analysis = agent.parse_failure("error[E0308]: mismatched types\n --> src/lib.rs:2:1");
        assert_eq!(analysis.file.as_deref(), Some("src/lib.rs"));
        let _ = std::fs::remove_dir_all(tmp);
    }
}
