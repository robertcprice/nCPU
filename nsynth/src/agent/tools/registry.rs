//! Tool-calling framework for Phase 2 agents.
//!
//! Provides a uniform [`Tool`] trait, a [`ToolRegistry`], and the call/result
//! types agents use to invoke real, side-effecting capabilities (filesystem,
//! shell, git, http, database). Every tool performs real work — there are no
//! stubs. Safety is enforced per tool: the filesystem tool is sandboxed to a
//! root directory, and the shell tool runs only allowlisted programs.

use std::collections::HashMap;

/// A request to invoke a tool: an action verb plus string-keyed arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCall {
    pub action: String,
    pub args: HashMap<String, String>,
}

impl ToolCall {
    /// Create a call for the given action with no arguments.
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            args: HashMap::new(),
        }
    }

    /// Builder-style argument insertion.
    pub fn arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.insert(key.into(), value.into());
        self
    }

    /// Fetch a required argument, erroring if absent.
    pub fn require(&self, key: &str) -> Result<&str, ToolError> {
        self.args
            .get(key)
            .map(|s| s.as_str())
            .ok_or_else(|| ToolError::MissingArg(key.to_string()))
    }

    /// Fetch an optional argument.
    pub fn optional(&self, key: &str) -> Option<&str> {
        self.args.get(key).map(|s| s.as_str())
    }
}

/// The result of a successful tool invocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolOutput {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl ToolOutput {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Errors a tool can return. Side-effecting failures are reported, never panicked.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("unknown action '{action}' for tool '{tool}'")]
    UnknownAction { tool: String, action: String },
    #[error("missing required argument: {0}")]
    MissingArg(String),
    #[error("invalid argument '{0}': {1}")]
    InvalidArg(String, String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

/// A real, side-effecting capability an agent can invoke.
pub trait Tool: Send + Sync {
    /// Unique tool name used for registry lookup.
    fn name(&self) -> &str;
    /// Human-readable description of the tool's purpose.
    fn description(&self) -> &str;
    /// Action verbs this tool supports (for discovery / introspection).
    fn actions(&self) -> Vec<&'static str>;
    /// Execute an action. Implementations must validate arguments and enforce
    /// their own safety constraints.
    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError>;
}

/// Registry of available tools, keyed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register (or replace) a tool.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Borrow a registered tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Sorted list of registered tool names.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tools.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Invoke a tool by name, dispatching the call to it.
    pub fn invoke(&self, tool: &str, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        self.get(tool)
            .ok_or_else(|| ToolError::UnknownTool(tool.to_string()))?
            .invoke(call)
    }

    /// Build a registry with all default tools registered. The filesystem tool
    /// is sandboxed to `sandbox_root`; shell/git/http/database use their
    /// default configurations.
    pub fn with_defaults(sandbox_root: impl Into<std::path::PathBuf>) -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(super::fs::FsTool::new(sandbox_root)));
        registry.register(Box::new(super::shell::ShellTool::default()));
        registry.register(Box::new(super::git::GitTool::new()));
        registry.register(Box::new(super::http::HttpTool::new()));
        registry.register(Box::new(super::database::DbTool::new()));
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "returns the 'text' argument"
        }
        fn actions(&self) -> Vec<&'static str> {
            vec!["say"]
        }
        fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
            match call.action.as_str() {
                "say" => Ok(ToolOutput::new(call.require("text")?.to_string())),
                other => Err(ToolError::UnknownAction {
                    tool: "echo".to_string(),
                    action: other.to_string(),
                }),
            }
        }
    }

    #[test]
    fn test_register_and_invoke() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.names(), vec!["echo".to_string()]);

        let call = ToolCall::new("say").arg("text", "hello");
        let out = reg.invoke("echo", &call).unwrap();
        assert_eq!(out.content, "hello");
    }

    #[test]
    fn test_unknown_tool() {
        let reg = ToolRegistry::new();
        let err = reg.invoke("missing", &ToolCall::new("x")).unwrap_err();
        assert_eq!(err, ToolError::UnknownTool("missing".to_string()));
    }

    #[test]
    fn test_missing_arg() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let err = reg.invoke("echo", &ToolCall::new("say")).unwrap_err();
        assert_eq!(err, ToolError::MissingArg("text".to_string()));
    }

    #[test]
    fn test_unknown_action() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(EchoTool));
        let err = reg.invoke("echo", &ToolCall::new("bogus")).unwrap_err();
        assert!(matches!(err, ToolError::UnknownAction { .. }));
    }
}
