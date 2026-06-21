//! Durable session snapshots for resume and clarification (Package I/J).

use crate::agent::runtime::{AgentRunId, SCHEMA_VERSION};
use crate::agent::session::{AgentQueryResult, QueryRoute};
use linguigenesis_core::coding_dialogue::ClarificationQuestion;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Pending clarification state between turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingQuery {
    pub query: String,
    pub partial: linguigenesis_core::coding_requirements::SynthesisRequirement,
    pub questions: Vec<ClarificationQuestion>,
    pub answers: Vec<(linguigenesis_core::coding_dialogue::ClarificationField, String)>,
}

/// Persisted coding-agent session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub schema_version: u32,
    pub session_id: String,
    pub root: String,
    pub last_query: Option<String>,
    pub last_route: Option<QueryRoute>,
    pub last_success: Option<bool>,
    pub last_response_preview: Option<String>,
    pub pending: Option<PendingQuery>,
    pub history_len: usize,
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            session_id: AgentRunId::new().0,
            root: String::new(),
            last_query: None,
            last_route: None,
            last_success: None,
            last_response_preview: None,
            pending: None,
            history_len: 0,
        }
    }
}

pub fn session_dir(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join(".nsynth").join("sessions")
}

pub fn session_path(root: impl AsRef<Path>, session_id: &str) -> PathBuf {
    session_dir(root).join(format!("{session_id}.json"))
}

pub fn save_session_snapshot(
    root: impl AsRef<Path>,
    snapshot: &SessionSnapshot,
) -> Result<PathBuf, String> {
    let dir = session_dir(&root);
    fs::create_dir_all(&dir).map_err(|e| format!("create session dir: {e}"))?;
    let path = session_path(&root, &snapshot.session_id);
    let json = serde_json::to_string_pretty(snapshot)
        .map_err(|e| format!("serialize session: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("write session: {e}"))?;
    Ok(path)
}

pub fn load_session_snapshot(path: impl AsRef<Path>) -> Result<SessionSnapshot, String> {
    let content =
        fs::read_to_string(path.as_ref()).map_err(|e| format!("read session: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("parse session: {e}"))
}

pub fn truncate_preview(text: &str, max: usize) -> String {
    if text.len() <= max {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn session_snapshot_roundtrip() {
        let root = env::temp_dir().join(format!("nsynth_sess_persist_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let snapshot = SessionSnapshot {
            session_id: "test-session".into(),
            root: root.to_string_lossy().to_string(),
            last_query: Some("add two numbers".into()),
            last_route: Some(QueryRoute::SynthesizeFunction),
            last_success: Some(true),
            history_len: 1,
            ..Default::default()
        };
        let path = save_session_snapshot(&root, &snapshot).expect("save");
        let loaded = load_session_snapshot(&path).expect("load");
        assert_eq!(loaded.session_id, "test-session");
        assert_eq!(loaded.last_query.as_deref(), Some("add two numbers"));
        let _ = fs::remove_dir_all(root);
    }
}
