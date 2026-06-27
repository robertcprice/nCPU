//! Deny-by-default tool runtime with guardrail preflight (Package F).

use super::registry::{Tool, ToolCall, ToolError, ToolOutput, ToolRegistry};
use crate::agent::repo::{GuardrailDecision, GuardrailPolicy, RepairVerification};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Policy-gated tool execution: tools and actions must be explicitly allowed.
pub struct SecureToolRuntime {
    sandbox_root: PathBuf,
    guardrails: GuardrailPolicy,
    registry: ToolRegistry,
    allowed_actions: HashSet<(String, String)>,
}

impl SecureToolRuntime {
    /// Empty allowlist — every `invoke` is denied until `allow` is called.
    pub fn deny_by_default(sandbox_root: impl Into<PathBuf>) -> Self {
        let sandbox_root = sandbox_root.into();
        let registry = build_sandbox_registry(&sandbox_root);
        Self {
            sandbox_root,
            guardrails: GuardrailPolicy::default(),
            registry,
            allowed_actions: HashSet::new(),
        }
    }

    pub fn with_guardrails(mut self, policy: GuardrailPolicy) -> Self {
        self.guardrails = policy;
        self
    }

    /// Allow a single tool action pair.
    pub fn allow(&mut self, tool: &str, action: &str) -> &mut Self {
        self.allowed_actions
            .insert((tool.to_string(), action.to_string()));
        self
    }

    /// Full agent session: all tools allowed under guardrails (Package F / session API).
    pub fn for_general_agent(sandbox_root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        let sandbox_root = sandbox_root.into();
        let registry = build_general_agent_registry(&sandbox_root);
        let mut runtime = Self {
            sandbox_root,
            guardrails: policy,
            registry,
            allowed_actions: HashSet::new(),
        };
        for action in [
            "read", "write", "append", "list", "exists", "mkdir", "remove",
        ] {
            runtime.allow("fs", action);
        }
        for action in ["run", "allowed"] {
            runtime.allow("shell", action);
        }
        for action in [
            "status",
            "diff",
            "log",
            "add",
            "commit",
            "branch",
            "current_branch",
        ] {
            runtime.allow("git", action);
        }
        for action in ["get", "post"] {
            runtime.allow("http", action);
        }
        for action in [
            "create_table",
            "insert",
            "select",
            "delete",
            "count",
            "list_tables",
        ] {
            runtime.allow("database", action);
        }
        runtime
    }

    /// Sorted list of explicitly allowed (tool, action) pairs for introspection.
    pub fn allowed_capabilities(&self) -> Vec<String> {
        let mut caps: Vec<String> = self
            .allowed_actions
            .iter()
            .map(|(tool, action)| format!("{tool}.{action}"))
            .collect();
        caps.sort();
        caps
    }

    /// Typical repo-agent sandbox: read/write fs under guardrails, read-only git, no shell by default.
    pub fn for_repo_repair(sandbox_root: impl Into<PathBuf>, policy: GuardrailPolicy) -> Self {
        let sandbox_root = sandbox_root.into();
        let registry = build_sandbox_registry(&sandbox_root);
        let mut runtime = Self {
            sandbox_root,
            guardrails: policy,
            registry,
            allowed_actions: HashSet::new(),
        };
        for action in [
            "read", "write", "append", "list", "exists", "mkdir", "remove",
        ] {
            runtime.allow("fs", action);
        }
        for action in ["status", "diff", "log", "current_branch", "branch"] {
            runtime.allow("git", action);
        }
        runtime
    }

    pub fn sandbox_root(&self) -> &Path {
        &self.sandbox_root
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    pub fn is_allowed(&self, tool: &str, action: &str) -> bool {
        self.allowed_actions
            .contains(&(tool.to_string(), action.to_string()))
    }

    /// Invoke a tool after deny-by-default and guardrail checks.
    pub fn invoke(&self, tool: &str, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        if !self.is_allowed(tool, &call.action) {
            return Err(ToolError::PermissionDenied(format!(
                "deny-by-default: tool '{tool}' action '{action}' is not allowed",
                action = call.action
            )));
        }
        self.preflight(tool, call)?;
        self.registry.invoke(tool, call)
    }

    /// Run an acceptance oracle command (`cargo test` / `cargo check` only).
    pub fn run_verification_command(&self, command: &str) -> Result<RepairVerification, String> {
        if command.trim().is_empty() {
            return Err("verification command is empty".to_string());
        }
        if !is_allowed_verification_command(command) {
            return Err(format!(
                "verification command not on allowlist (cargo test|check only): {command}"
            ));
        }

        match self.guardrails.check_command(command) {
            GuardrailDecision::Deny(reason) => return Err(reason),
            GuardrailDecision::Ask(reason) => {
                return Ok(RepairVerification {
                    success: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: reason,
                    command: command.to_string(),
                });
            }
            GuardrailDecision::Allow => {}
        }

        let mut command_builder = Command::new("sh");
        command_builder
            .arg("-c")
            .arg(command)
            .current_dir(&self.sandbox_root);
        if command.contains("cargo") {
            command_builder.env("CARGO_TARGET_DIR", self.sandbox_root.join("target"));
            command_builder.env("CARGO_INCREMENTAL", "0");
        }

        let output = command_builder
            .output()
            .map_err(|e| format!("failed to run verification command: {e}"))?;

        Ok(RepairVerification {
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            command: command.to_string(),
        })
    }

    fn preflight(&self, tool: &str, call: &ToolCall) -> Result<(), ToolError> {
        match tool {
            "fs" => {
                let path = call.optional("path").unwrap_or(".");
                let writable = matches!(
                    call.action.as_str(),
                    "write" | "append" | "remove" | "mkdir"
                );
                self.guardrail_path(path, writable)
            }
            "shell" => {
                let program = call.require("cmd")?;
                let args = call.optional("args").unwrap_or("");
                let full = if args.is_empty() {
                    program.to_string()
                } else {
                    format!("{program} {args}")
                };
                self.guardrail_command(&full)
            }
            "git" => self.guardrail_command(&git_command_line(call)?),
            "http" => {
                let url = call.require("url").unwrap_or("");
                match self.guardrails.check_http_url(url) {
                    GuardrailDecision::Allow => Ok(()),
                    GuardrailDecision::Deny(reason) | GuardrailDecision::Ask(reason) => {
                        Err(ToolError::PermissionDenied(reason))
                    }
                }
            }
            _ => Ok(()),
        }
    }

    fn guardrail_path(&self, path: &str, writable: bool) -> Result<(), ToolError> {
        match self.guardrails.check_path(path, writable) {
            GuardrailDecision::Allow => Ok(()),
            GuardrailDecision::Deny(reason) | GuardrailDecision::Ask(reason) => {
                Err(ToolError::PermissionDenied(reason))
            }
        }
    }

    fn guardrail_command(&self, command: &str) -> Result<(), ToolError> {
        match self.guardrails.check_command(command) {
            GuardrailDecision::Allow => Ok(()),
            GuardrailDecision::Deny(reason) | GuardrailDecision::Ask(reason) => {
                Err(ToolError::PermissionDenied(reason))
            }
        }
    }
}

fn build_sandbox_registry(sandbox: &Path) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(super::fs::FsTool::new(sandbox)));
    registry.register(Box::new(
        super::shell::ShellTool::default().with_cwd(sandbox),
    ));
    registry.register(Box::new(super::git::GitTool::new().with_repo(sandbox)));
    registry.register(Box::new(super::http::HttpTool::new()));
    registry.register(Box::new(super::database::DbTool::new()));
    registry
}

fn build_general_agent_registry(sandbox: &Path) -> ToolRegistry {
    let shell_allow = [
        "echo", "ls", "cat", "pwd", "grep", "wc", "head", "tail", "find", "date", "uname", "true",
        "false", "cargo", "rustc", "git", "curl", "python3", "which", "env",
    ];
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(super::fs::FsTool::new(sandbox)));
    registry.register(Box::new(
        super::shell::ShellTool::with_allowed(shell_allow).with_cwd(sandbox),
    ));
    registry.register(Box::new(super::git::GitTool::new().with_repo(sandbox)));
    registry.register(Box::new(super::http::HttpTool::new()));
    registry.register(Box::new(super::database::DbTool::new()));
    registry
}

fn is_allowed_verification_command(command: &str) -> bool {
    let trimmed = command.trim();
    trimmed.starts_with("cargo test") || trimmed.starts_with("cargo check")
}

fn git_command_line(call: &ToolCall) -> Result<String, ToolError> {
    let mut parts = vec!["git".to_string(), call.action.clone()];
    match call.action.as_str() {
        "diff" => {
            if let Some(path) = call.optional("path") {
                parts.push("--".to_string());
                parts.push(path.to_string());
            }
        }
        "log" => {
            let n = call.optional("count").unwrap_or("10");
            parts.push("--oneline".to_string());
            parts.push("-n".to_string());
            parts.push(n.to_string());
        }
        "add" => parts.push(call.require("path")?.to_string()),
        "commit" => parts.push(call.require("message")?.to_string()),
        _ => {}
    }
    Ok(parts.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::Command;
    use std::thread;

    fn temp_sandbox(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nsynth_secure_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn local_http_once(body: &'static str) -> (String, thread::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind local http server");
        let addr = listener.local_addr().expect("local addr");
        let url = format!("http://{addr}/api/rules");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept one request");
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).expect("read request");
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .expect("write response");
            request
        });
        (url, handle)
    }

    fn curl_available() -> bool {
        Command::new("curl")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn deny_by_default_rejects_unlisted_tool() {
        let root = temp_sandbox("deny");
        let runtime = SecureToolRuntime::deny_by_default(&root);
        let err = runtime
            .invoke("fs", &ToolCall::new("read").arg("path", "x.txt"))
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repo_repair_runtime_allows_guarded_fs_write() {
        let root = temp_sandbox("fs_write");
        let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
        runtime
            .invoke(
                "fs",
                &ToolCall::new("write")
                    .arg("path", "ok.txt")
                    .arg("content", "hi"),
            )
            .unwrap();
        let read = runtime
            .invoke("fs", &ToolCall::new("read").arg("path", "ok.txt"))
            .unwrap();
        assert_eq!(read.content, "hi");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn guardrail_denies_dotenv_read_via_fs() {
        let root = temp_sandbox("dotenv");
        let runtime = SecureToolRuntime::for_repo_repair(&root, GuardrailPolicy::default());
        let err = runtime
            .invoke("fs", &ToolCall::new("read").arg("path", ".env"))
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verification_allowlist_rejects_shell_injection() {
        let root = temp_sandbox("verify");
        let runtime = SecureToolRuntime::deny_by_default(&root);
        let err = runtime.run_verification_command("rm -rf /").unwrap_err();
        assert!(err.contains("allowlist"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn verification_runs_cargo_check() {
        let root = temp_sandbox("cargo");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"tiny\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        let runtime = SecureToolRuntime::deny_by_default(&root);
        let report = runtime
            .run_verification_command("cargo check")
            .expect("cargo check");
        assert!(report.success, "stderr: {}", report.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn general_agent_runtime_allows_all_tool_families() {
        let root = temp_sandbox("general");
        let runtime = SecureToolRuntime::for_general_agent(&root, GuardrailPolicy::default());
        let caps = runtime.allowed_capabilities();
        assert!(caps.iter().any(|c| c.starts_with("fs.")));
        assert!(caps.iter().any(|c| c == "shell.run"));
        assert!(caps.iter().any(|c| c == "http.get"));
        assert!(caps.iter().any(|c| c == "database.list_tables"));
        runtime
            .invoke("database", &ToolCall::new("list_tables"))
            .unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn http_tool_denies_disallowed_host() {
        let root = temp_sandbox("http_deny");
        let runtime = SecureToolRuntime::for_general_agent(&root, GuardrailPolicy::default());
        let err = runtime
            .invoke(
                "http",
                &ToolCall::new("get").arg("url", "https://evil.example/data"),
            )
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn general_agent_runtime_executes_local_http_then_records_database_row() {
        let root = temp_sandbox("http_db_flow");
        if !curl_available() {
            eprintln!("skipping secure HTTP+DB flow test: curl unavailable");
            let _ = fs::remove_dir_all(root);
            return;
        }
        let runtime = SecureToolRuntime::for_general_agent(&root, GuardrailPolicy::default());
        let (url, handle) = local_http_once("rule=fall_speed_f;status=learned");

        let http = runtime
            .invoke("http", &ToolCall::new("get").arg("url", &url))
            .expect("localhost HTTP GET should be allowed and executed");
        let request = handle.join().expect("local server thread");
        assert!(
            request.starts_with("GET /api/rules HTTP/"),
            "unexpected HTTP request: {request}"
        );
        assert_eq!(http.content, "rule=fall_speed_f;status=learned");
        assert_eq!(http.metadata.get("status").map(|s| s.as_str()), Some("200"));

        runtime
            .invoke(
                "database",
                &ToolCall::new("create_table")
                    .arg("table", "rule_events")
                    .arg("columns", "kind,value"),
            )
            .expect("create table");
        runtime
            .invoke(
                "database",
                &ToolCall::new("insert")
                    .arg("table", "rule_events")
                    .arg("values", "http_status,200"),
            )
            .expect("insert http status row");
        runtime
            .invoke(
                "database",
                &ToolCall::new("insert")
                    .arg("table", "rule_events")
                    .arg("values", "payload,rule=fall_speed_f;status=learned"),
            )
            .expect("insert payload row");

        let selected = runtime
            .invoke(
                "database",
                &ToolCall::new("select")
                    .arg("table", "rule_events")
                    .arg("where", "kind=payload"),
            )
            .expect("select payload row");
        assert_eq!(
            selected.metadata.get("matched").map(|s| s.as_str()),
            Some("1")
        );
        assert!(selected
            .content
            .contains("rule=fall_speed_f;status=learned"));
        let _ = fs::remove_dir_all(root);
    }
}
