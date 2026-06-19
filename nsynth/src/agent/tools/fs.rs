//! Sandboxed filesystem tool.
//!
//! All paths are resolved relative to a sandbox root. Absolute paths and any
//! path containing a `..` component are rejected, so a tool call can never
//! escape the sandbox. Every operation performs real filesystem I/O.

use super::registry::{Tool, ToolCall, ToolError, ToolOutput};
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Filesystem tool rooted at a sandbox directory.
pub struct FsTool {
    root: PathBuf,
}

impl FsTool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a caller-supplied relative path against the sandbox root,
    /// rejecting absolute paths and `..` traversal.
    fn safe_path(&self, rel: &str) -> Result<PathBuf, ToolError> {
        let p = Path::new(rel);
        if p.is_absolute() {
            return Err(ToolError::PermissionDenied(format!(
                "absolute paths are not allowed: {rel}"
            )));
        }
        for comp in p.components() {
            match comp {
                Component::ParentDir => {
                    return Err(ToolError::PermissionDenied(format!(
                        "'..' traversal is not allowed: {rel}"
                    )));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(ToolError::PermissionDenied(format!(
                        "rooted paths are not allowed: {rel}"
                    )));
                }
                _ => {}
            }
        }
        Ok(self.root.join(p))
    }
}

impl Tool for FsTool {
    fn name(&self) -> &str {
        "fs"
    }

    fn description(&self) -> &str {
        "Sandboxed filesystem access: read, write, append, list, exists, mkdir, remove"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec![
            "read", "write", "append", "list", "exists", "mkdir", "remove",
        ]
    }

    fn invoke(&self, call: &ToolCall) -> Result<ToolOutput, ToolError> {
        match call.action.as_str() {
            "read" => {
                let path = self.safe_path(call.require("path")?)?;
                let content =
                    fs::read_to_string(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                Ok(ToolOutput::new(content))
            }
            "write" => {
                let path = self.safe_path(call.require("path")?)?;
                let content = call.require("content")?;
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
                }
                fs::write(&path, content).map_err(|e| ToolError::Io(e.to_string()))?;
                Ok(ToolOutput::new("ok").with_meta("bytes", content.len().to_string()))
            }
            "append" => {
                use std::io::Write;
                let path = self.safe_path(call.require("path")?)?;
                let content = call.require("content")?;
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
                }
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| ToolError::Io(e.to_string()))?;
                file.write_all(content.as_bytes())
                    .map_err(|e| ToolError::Io(e.to_string()))?;
                Ok(ToolOutput::new("ok").with_meta("bytes", content.len().to_string()))
            }
            "list" => {
                let rel = call.optional("path").unwrap_or(".");
                let path = self.safe_path(rel)?;
                let mut names: Vec<String> = Vec::new();
                let entries =
                    fs::read_dir(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                for entry in entries {
                    let entry = entry.map_err(|e| ToolError::Io(e.to_string()))?;
                    names.push(entry.file_name().to_string_lossy().to_string());
                }
                names.sort();
                Ok(ToolOutput::new(names.join("\n")).with_meta("count", names.len().to_string()))
            }
            "exists" => {
                let path = self.safe_path(call.require("path")?)?;
                Ok(ToolOutput::new(path.exists().to_string()))
            }
            "mkdir" => {
                let path = self.safe_path(call.require("path")?)?;
                fs::create_dir_all(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                Ok(ToolOutput::new("ok"))
            }
            "remove" => {
                let path = self.safe_path(call.require("path")?)?;
                if path.is_dir() {
                    fs::remove_dir_all(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                } else {
                    fs::remove_file(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                }
                Ok(ToolOutput::new("ok"))
            }
            other => Err(ToolError::UnknownAction {
                tool: "fs".to_string(),
                action: other.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_sandbox(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nsynth_fs_{}_{}",
            std::process::id(),
            tag
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_write_read_roundtrip() {
        let root = temp_sandbox("rw");
        let tool = FsTool::new(&root);
        tool.invoke(&ToolCall::new("write").arg("path", "a/b.txt").arg("content", "hi"))
            .unwrap();
        let out = tool
            .invoke(&ToolCall::new("read").arg("path", "a/b.txt"))
            .unwrap();
        assert_eq!(out.content, "hi");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_append_and_list() {
        let root = temp_sandbox("append");
        let tool = FsTool::new(&root);
        tool.invoke(&ToolCall::new("write").arg("path", "log.txt").arg("content", "a"))
            .unwrap();
        tool.invoke(&ToolCall::new("append").arg("path", "log.txt").arg("content", "b"))
            .unwrap();
        let read = tool
            .invoke(&ToolCall::new("read").arg("path", "log.txt"))
            .unwrap();
        assert_eq!(read.content, "ab");
        let list = tool.invoke(&ToolCall::new("list")).unwrap();
        assert!(list.content.contains("log.txt"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_sandbox_rejects_parent_traversal() {
        let root = temp_sandbox("escape");
        let tool = FsTool::new(&root);
        let err = tool
            .invoke(&ToolCall::new("read").arg("path", "../secret"))
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_sandbox_rejects_absolute() {
        let root = temp_sandbox("abs");
        let tool = FsTool::new(&root);
        let err = tool
            .invoke(&ToolCall::new("read").arg("path", "/etc/passwd"))
            .unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_exists_and_remove() {
        let root = temp_sandbox("rm");
        let tool = FsTool::new(&root);
        tool.invoke(&ToolCall::new("write").arg("path", "f.txt").arg("content", "x"))
            .unwrap();
        assert_eq!(
            tool.invoke(&ToolCall::new("exists").arg("path", "f.txt"))
                .unwrap()
                .content,
            "true"
        );
        tool.invoke(&ToolCall::new("remove").arg("path", "f.txt"))
            .unwrap();
        assert_eq!(
            tool.invoke(&ToolCall::new("exists").arg("path", "f.txt"))
                .unwrap()
                .content,
            "false"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
