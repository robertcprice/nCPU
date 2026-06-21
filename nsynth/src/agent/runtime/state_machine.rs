use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

/// Stable run identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentRunId(pub String);

impl AgentRunId {
    pub fn new() -> Self {
        Self(format!(
            "run-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ))
    }
}

/// Canonical agent run state machine (Phase 1 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRunStatus {
    Created,
    Understanding,
    Planning,
    Executing,
    Verifying,
    Revising,
    Succeeded,
    Failed,
    Cancelled,
}

impl AgentRunStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    pub fn needs_clarification(self, questions_len: usize) -> bool {
        self == Self::Understanding && questions_len > 0
    }
}

/// Recorded state transition for durable runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRunEvent {
    pub from: AgentRunStatus,
    pub to: AgentRunStatus,
    pub detail: String,
}

/// Legal transitions for the canonical state machine.
pub fn legal_transition(from: AgentRunStatus, to: AgentRunStatus) -> bool {
    use AgentRunStatus::*;
    if from.is_terminal() {
        return false;
    }
    matches!(
        (from, to),
        (Created, Understanding)
            | (Understanding, Planning)
            | (Understanding, Executing)
            | (Understanding, Failed)
            | (Understanding, Cancelled)
            | (Planning, Executing)
            | (Planning, Cancelled)
            | (Planning, Failed)
            | (Executing, Verifying)
            | (Executing, Succeeded)
            | (Executing, Failed)
            | (Executing, Cancelled)
            | (Verifying, Revising)
            | (Verifying, Succeeded)
            | (Verifying, Failed)
            | (Verifying, Cancelled)
            | (Revising, Executing)
            | (Revising, Verifying)
            | (Revising, Failed)
            | (Revising, Cancelled)
    )
}

pub fn transition(
    current: AgentRunStatus,
    next: AgentRunStatus,
    detail: impl Into<String>,
) -> Result<(AgentRunStatus, AgentRunEvent), String> {
    if !legal_transition(current, next) {
        return Err(format!("illegal transition {:?} -> {:?}", current, next));
    }
    let event = AgentRunEvent {
        from: current,
        to: next,
        detail: detail.into(),
    };
    Ok((next, event))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_nl_synthesis_path() {
        assert!(legal_transition(
            AgentRunStatus::Created,
            AgentRunStatus::Understanding
        ));
        assert!(legal_transition(
            AgentRunStatus::Understanding,
            AgentRunStatus::Executing
        ));
        assert!(legal_transition(
            AgentRunStatus::Executing,
            AgentRunStatus::Succeeded
        ));
    }

    #[test]
    fn legal_repair_loop_path() {
        assert!(legal_transition(
            AgentRunStatus::Executing,
            AgentRunStatus::Verifying
        ));
        assert!(legal_transition(
            AgentRunStatus::Verifying,
            AgentRunStatus::Revising
        ));
        assert!(legal_transition(
            AgentRunStatus::Revising,
            AgentRunStatus::Executing
        ));
    }

    #[test]
    fn terminal_states_block_further_transitions() {
        assert!(!legal_transition(
            AgentRunStatus::Succeeded,
            AgentRunStatus::Executing
        ));
        assert!(!legal_transition(
            AgentRunStatus::Cancelled,
            AgentRunStatus::Understanding
        ));
    }

    #[test]
    fn transition_rejects_illegal_edges() {
        let err =
            transition(AgentRunStatus::Created, AgentRunStatus::Succeeded, "skip").unwrap_err();
        assert!(err.contains("illegal"));
    }
}
