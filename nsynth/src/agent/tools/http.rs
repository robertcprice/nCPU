//! HTTP tool.
//!
//! Performs real HTTP requests by shelling out to `curl` (no extra crate
//! dependency). Honestly reports when `curl` is unavailable. Supports GET and
//! POST with optional headers and body.

use super::registry::{Tool, ToolCall, ToolError, ToolOutput};
use std::process::Command;

/// HTTP client tool backed by the system `curl` binary.
pub struct HttpTool {
    timeout_secs: u32,
}

impl Default for HttpTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTool {
    pub fn new() -> Self {
        Self { timeout_secs: 30 }
    }

    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs;
        self
    }

    fn curl(&self, args: &[String]) -> Result<ToolOutput, ToolError> {
        let mut command = Command::new("curl");
        command
            .arg("-s")
            .arg("-S")
            .arg("--max-time")
            .arg(self.timeout_secs.to_string())
            // Emit the HTTP status code on its own trailing line so callers can
            // inspect it without an extra request.
            .arg("-w")
            .arg("\n%{http_code}");
        for a in args {
            command.arg(a);
        }

        let output = command.output().map_err(|e| {
            ToolError::Execution(format!("failed to run curl (is it installed?): {e}"))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(ToolError::Execution(format!(
                "curl failed: {}",
                stderr.trim()
            )));
        }

        let combined = String::from_utf8_lossy(&output.stdout).to_string();
        // Split the trailing status code line written by `-w`.
        let (body, status) = match combined.rsplit_once('\n') {
            Some((body, status)) => (body.to_string(), status.trim().to_string()),
            None => (combined.clone(), String::new()),
        };

        Ok(ToolOutput::new(body).with_meta("status", status))
    }
}

impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }

    fn description(&self) -> &str {
        "HTTP client (via curl): get, post"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec!["get", "post"]
    }

    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        match call.action.as_str() {
            "get" => {
                let url = call.require("url")?;
                let mut args = vec!["-X".to_string(), "GET".to_string()];
                if let Some(header) = call.optional("header") {
                    args.push("-H".to_string());
                    args.push(header.to_string());
                }
                args.push(url.to_string());
                self.curl(&args)
            }
            "post" => {
                let url = call.require("url")?;
                let mut args = vec!["-X".to_string(), "POST".to_string()];
                let content_type = call.optional("content_type").unwrap_or("application/json");
                args.push("-H".to_string());
                args.push(format!("Content-Type: {content_type}"));
                if let Some(header) = call.optional("header") {
                    args.push("-H".to_string());
                    args.push(header.to_string());
                }
                if let Some(body) = call.optional("body") {
                    args.push("--data".to_string());
                    args.push(body.to_string());
                }
                args.push(url.to_string());
                self.curl(&args)
            }
            other => Err(ToolError::UnknownAction {
                tool: "http".to_string(),
                action: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_requires_url() {
        let tool = HttpTool::new();
        let err = tool.invoke(&ToolCall::new("get")).unwrap_err();
        assert_eq!(err, ToolError::MissingArg("url".to_string()));
    }

    #[test]
    fn test_unknown_action() {
        let tool = HttpTool::new();
        let err = tool.invoke(&ToolCall::new("delete")).unwrap_err();
        assert!(matches!(err, ToolError::UnknownAction { .. }));
    }

    // Network-dependent GET is intentionally not asserted in unit tests to keep
    // the suite hermetic; the curl integration is exercised manually.
}
