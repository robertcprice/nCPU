//! Tool-calling framework for Phase 2 agents.
//!
//! A uniform [`Tool`] trait plus a [`ToolRegistry`] over five real,
//! side-effecting tools:
//!
//! - [`FsTool`]: sandboxed filesystem access
//! - [`ShellTool`]: allowlisted process execution
//! - [`GitTool`]: version-control operations via the `git` binary
//! - [`HttpTool`]: HTTP requests via the `curl` binary
//! - [`DbTool`]: in-memory relational-style table store
//!
//! Every tool performs genuine work; safety is enforced per tool (filesystem
//! sandboxing, shell allowlist).

pub mod database;
pub mod fs;
pub mod git;
pub mod http;
pub mod registry;
pub mod secure_runtime;
pub mod shell;

pub use database::DbTool;
pub use fs::FsTool;
pub use git::GitTool;
pub use http::HttpTool;
pub use registry::{Tool, ToolCall, ToolError, ToolOutput, ToolRegistry};
pub use secure_runtime::SecureToolRuntime;
pub use shell::ShellTool;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_with_defaults_has_all_tools() {
        let root = std::env::temp_dir().join(format!("nsynth_tools_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        let registry = ToolRegistry::with_defaults(&root);
        let names = registry.names();
        assert_eq!(
            names,
            vec![
                "database".to_string(),
                "fs".to_string(),
                "git".to_string(),
                "http".to_string(),
                "shell".to_string(),
            ]
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn test_end_to_end_fs_through_registry() {
        let root = std::env::temp_dir().join(format!("nsynth_tools_e2e_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        let registry = ToolRegistry::with_defaults(&root);
        registry
            .invoke(
                "fs",
                &ToolCall::new("write")
                    .arg("path", "note.txt")
                    .arg("content", "agentic"),
            )
            .unwrap();
        let read = registry
            .invoke("fs", &ToolCall::new("read").arg("path", "note.txt"))
            .unwrap();
        assert_eq!(read.content, "agentic");
        let _ = std::fs::remove_dir_all(&root);
    }
}
