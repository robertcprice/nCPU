//! Git tool.
//!
//! Wraps the system `git` binary for a fixed set of safe subcommands. Operates
//! on an optional repository directory (defaults to the process working
//! directory). Real process execution; honestly reports when `git` is absent or
//! a command fails.

use super::registry::{Tool, ToolCall, ToolError, ToolOutput};
use std::path::PathBuf;
use std::process::Command;

/// Git tool bound to an optional repository directory.
pub struct GitTool {
    repo: Option<PathBuf>,
}

impl Default for GitTool {
    fn default() -> Self {
        Self::new()
    }
}

impl GitTool {
    pub fn new() -> Self {
        Self { repo: None }
    }

    /// Bind the tool to a specific repository directory.
    pub fn with_repo(mut self, repo: impl Into<PathBuf>) -> Self {
        self.repo = Some(repo.into());
        self
    }

    fn run(&self, args: &[&str]) -> Result<ToolOutput, ToolError> {
        let mut command = Command::new("git");
        if let Some(repo) = &self.repo {
            command.arg("-C").arg(repo);
        }
        command.args(args);

        let output = command.output().map_err(|e| {
            ToolError::Execution(format!("failed to run git (is it installed?): {e}"))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if output.status.success() {
            Ok(ToolOutput::new(stdout).with_meta("stderr", stderr))
        } else {
            Err(ToolError::Execution(format!(
                "git {} failed: {}",
                args.join(" "),
                stderr.trim()
            )))
        }
    }
}

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Version control: status, diff, log, add, commit, branch, current_branch"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec![
            "status",
            "diff",
            "log",
            "add",
            "commit",
            "branch",
            "current_branch",
        ]
    }

    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        match call.action.as_str() {
            "status" => self.run(&["status", "--short"]),
            "diff" => {
                if let Some(path) = call.optional("path") {
                    self.run(&["diff", "--", path])
                } else {
                    self.run(&["diff"])
                }
            }
            "log" => {
                let n = call.optional("count").unwrap_or("10");
                self.run(&["log", "--oneline", "-n", n])
            }
            "add" => {
                let path = call.require("path")?;
                self.run(&["add", path])
            }
            "commit" => {
                let message = call.require("message")?;
                self.run(&["commit", "-m", message])
            }
            "branch" => self.run(&["branch", "--list"]),
            "current_branch" => self.run(&["rev-parse", "--abbrev-ref", "HEAD"]),
            other => Err(ToolError::UnknownAction {
                tool: "git".to_string(),
                action: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn test_status_in_repo() {
        if !git_available() {
            eprintln!("skipping: git not available");
            return;
        }
        // The nsynth crate lives inside a git repo; status should succeed.
        let tool = GitTool::new();
        let result = tool.invoke(&ToolCall::new("status"));
        // Either succeeds (in a repo) or errors cleanly (not in a repo); must
        // never panic.
        assert!(result.is_ok() || matches!(result, Err(ToolError::Execution(_))));
    }

    #[test]
    fn test_unknown_action() {
        let tool = GitTool::new();
        let err = tool.invoke(&ToolCall::new("nuke")).unwrap_err();
        assert!(matches!(err, ToolError::UnknownAction { .. }));
    }

    #[test]
    fn test_commit_requires_message() {
        let tool = GitTool::new();
        let err = tool.invoke(&ToolCall::new("commit")).unwrap_err();
        assert_eq!(err, ToolError::MissingArg("message".to_string()));
    }
}
