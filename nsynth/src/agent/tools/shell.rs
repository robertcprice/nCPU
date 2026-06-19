//! Allowlisted shell tool.
//!
//! Runs only programs on an explicit allowlist, so an agent cannot execute
//! arbitrary commands. Captures stdout/stderr and the exit code. Real process
//! execution via [`std::process::Command`].

use super::registry::{Tool, ToolCall, ToolError, ToolOutput};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

/// Shell tool that executes allowlisted programs.
pub struct ShellTool {
    allowlist: HashSet<String>,
    cwd: Option<PathBuf>,
}

impl Default for ShellTool {
    fn default() -> Self {
        let allowed = [
            "echo", "ls", "cat", "pwd", "grep", "wc", "head", "tail", "find", "date", "uname",
            "true", "false",
        ];
        Self {
            allowlist: allowed.iter().map(|s| s.to_string()).collect(),
            cwd: None,
        }
    }
}

impl ShellTool {
    /// Construct with a custom allowlist of permitted program names.
    pub fn with_allowed<I, S>(programs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowlist: programs.into_iter().map(|s| s.into()).collect(),
            cwd: None,
        }
    }

    /// Set the working directory for executed commands.
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn allows(&self, program: &str) -> bool {
        self.allowlist.contains(program)
    }
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Run an allowlisted program with arguments and capture its output"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec!["run", "allowed"]
    }

    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        match call.action.as_str() {
            "allowed" => {
                let mut names: Vec<&String> = self.allowlist.iter().collect();
                names.sort();
                Ok(ToolOutput::new(
                    names
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                ))
            }
            "run" => {
                let program = call.require("cmd")?;
                if !self.allows(program) {
                    return Err(ToolError::PermissionDenied(format!(
                        "program '{program}' is not on the allowlist"
                    )));
                }
                let arg_str = call.optional("args").unwrap_or("");
                let args: Vec<&str> = arg_str.split_whitespace().collect();

                let mut command = Command::new(program);
                command.args(&args);
                if let Some(cwd) = &self.cwd {
                    command.current_dir(cwd);
                }

                let output = command
                    .output()
                    .map_err(|e| ToolError::Execution(format!("{program}: {e}")))?;

                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let code = output.status.code().unwrap_or(-1);

                Ok(ToolOutput::new(stdout)
                    .with_meta("exit_code", code.to_string())
                    .with_meta("stderr", stderr))
            }
            other => Err(ToolError::UnknownAction {
                tool: "shell".to_string(),
                action: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_runs() {
        let tool = ShellTool::default();
        let out = tool
            .invoke(&ToolCall::new("run").arg("cmd", "echo").arg("args", "hello world"))
            .unwrap();
        assert_eq!(out.content.trim(), "hello world");
        assert_eq!(out.metadata.get("exit_code").map(|s| s.as_str()), Some("0"));
    }

    #[test]
    fn test_non_allowlisted_rejected() {
        let tool = ShellTool::default();
        let err = tool
            .invoke(&ToolCall::new("run").arg("cmd", "rm").arg("args", "-rf /"))
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[test]
    fn test_custom_allowlist() {
        let tool = ShellTool::with_allowed(["echo"]);
        assert!(tool.allows("echo"));
        assert!(!tool.allows("ls"));
    }

    #[test]
    fn test_allowed_action_lists_programs() {
        let tool = ShellTool::default();
        let out = tool.invoke(&ToolCall::new("allowed")).unwrap();
        assert!(out.content.contains("echo"));
    }
}
