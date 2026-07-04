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
        "Sandboxed filesystem access: read, read_range, write, append, list, glob, grep, exists, mkdir, remove, move"
    }

    fn actions(&self) -> Vec<&'static str> {
        vec![
            "read", "read_range", "write", "append", "list", "glob", "grep",
            "exists", "mkdir", "remove", "move",
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
            "read_range" => {
                // 1-indexed inclusive line range — partial read for large files.
                let path = self.safe_path(call.require("path")?)?;
                let start: usize = call
                    .require("start")?
                    .parse()
                    .map_err(|_| ToolError::Io("start must be a line number".to_string()))?;
                let end: usize = call
                    .require("end")?
                    .parse()
                    .map_err(|_| ToolError::Io("end must be a line number".to_string()))?;
                let content =
                    fs::read_to_string(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                let lines: Vec<&str> = content.lines().collect();
                let s = start.max(1);
                let e = end.min(lines.len());
                let slice = if s <= e { lines[s - 1..e].join("\n") } else { String::new() };
                Ok(ToolOutput::new(slice).with_meta("range", format!("{s}-{e}")))
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
                let entries = fs::read_dir(&path).map_err(|e| ToolError::Io(e.to_string()))?;
                for entry in entries {
                    let entry = entry.map_err(|e| ToolError::Io(e.to_string()))?;
                    names.push(entry.file_name().to_string_lossy().to_string());
                }
                names.sort();
                Ok(ToolOutput::new(names.join("\n")).with_meta("count", names.len().to_string()))
            }
            "glob" => {
                // Find files whose relative path matches a wildcard pattern (* and
                // ?). `*` spans directories, so `*.rs` finds every Rust file.
                let pattern = call.require("pattern")?;
                let mut all: Vec<String> = Vec::new();
                walk_files(&self.root, &self.root, &mut all, 20_000);
                let mut matched: Vec<String> = all
                    .into_iter()
                    .filter(|rel| wildcard_match(pattern.as_bytes(), rel.as_bytes()))
                    .collect();
                matched.sort();
                matched.truncate(500);
                Ok(ToolOutput::new(matched.join("\n"))
                    .with_meta("count", matched.len().to_string()))
            }
            "grep" => {
                // Content search: return `relpath:lineno:line` for lines containing
                // `query`, optionally under `path` and case-insensitively.
                let query = call.require("query")?;
                let ignore_case = matches!(call.optional("ignore_case"), Some("true") | Some("1"));
                let base = self.safe_path(call.optional("path").unwrap_or("."))?;
                let needle = if ignore_case { query.to_lowercase() } else { query.to_string() };
                let mut files: Vec<String> = Vec::new();
                walk_files(&self.root, &base, &mut files, 20_000);
                let mut hits: Vec<String> = Vec::new();
                'files: for rel in &files {
                    let abs = self.root.join(rel);
                    if fs::metadata(&abs).map(|m| m.len() > 1_000_000).unwrap_or(true) {
                        continue; // skip large/unstattable files
                    }
                    let Ok(text) = fs::read_to_string(&abs) else { continue };
                    for (i, line) in text.lines().enumerate() {
                        let hay = if ignore_case { line.to_lowercase() } else { line.to_string() };
                        if hay.contains(&needle) {
                            let shown: String = line.chars().take(200).collect();
                            hits.push(format!("{rel}:{}:{}", i + 1, shown.trim_end()));
                            if hits.len() >= 300 {
                                break 'files;
                            }
                        }
                    }
                }
                Ok(ToolOutput::new(hits.join("\n")).with_meta("count", hits.len().to_string()))
            }
            "move" => {
                let from = self.safe_path(call.require("from")?)?;
                let to = self.safe_path(call.require("to")?)?;
                if let Some(parent) = to.parent() {
                    fs::create_dir_all(parent).map_err(|e| ToolError::Io(e.to_string()))?;
                }
                fs::rename(&from, &to).map_err(|e| ToolError::Io(e.to_string()))?;
                Ok(ToolOutput::new("ok"))
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

/// Recursively collect file paths under `dir` (relative to `root`), skipping
/// hidden entries, `target`, and `node_modules`, deterministic + bounded by `max`.
fn walk_files(root: &Path, dir: &Path, out: &mut Vec<String>, max: usize) {
    if out.len() >= max {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut items: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    items.sort_by_key(|e| e.file_name());
    for entry in items {
        if out.len() >= max {
            return;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_files(root, &path, out, max);
        } else if path.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().to_string());
            }
        }
    }
}

/// Classic wildcard match: `*` matches any run of chars (incl. `/`), `?` matches
/// one. Iterative with backtracking — no regex dependency.
fn wildcard_match(pat: &[u8], text: &[u8]) -> bool {
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while t < text.len() {
        if p < pat.len() && (pat[p] == b'?' || pat[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pat.len() && pat[p] == b'*' {
            star = p;
            mark = t;
            p += 1;
        } else if star != usize::MAX {
            p = star + 1;
            mark += 1;
            t = mark;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == b'*' {
        p += 1;
    }
    p == pat.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_sandbox(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nsynth_fs_{}_{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_write_read_roundtrip() {
        let root = temp_sandbox("rw");
        let tool = FsTool::new(&root);
        tool.invoke(
            &ToolCall::new("write")
                .arg("path", "a/b.txt")
                .arg("content", "hi"),
        )
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
        tool.invoke(
            &ToolCall::new("write")
                .arg("path", "log.txt")
                .arg("content", "a"),
        )
        .unwrap();
        tool.invoke(
            &ToolCall::new("append")
                .arg("path", "log.txt")
                .arg("content", "b"),
        )
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
    fn test_glob_grep_read_range_move() {
        let root = temp_sandbox("nav");
        let tool = FsTool::new(&root);
        let w = |p: &str, c: &str| {
            tool.invoke(&ToolCall::new("write").arg("path", p).arg("content", c)).unwrap();
        };
        w("src/main.rs", "fn main() {\n    let x = 1;\n    println!(\"hi\");\n}\n");
        w("src/util.rs", "pub fn helper() -> i64 { 42 }\n");
        w("notes.txt", "todo: fix helper\n");

        // glob: * spans directories
        let rs = tool.invoke(&ToolCall::new("glob").arg("pattern", "*.rs")).unwrap();
        assert!(rs.content.contains("src/main.rs") && rs.content.contains("src/util.rs"));
        assert!(!rs.content.contains("notes.txt"));

        // grep: content search across the tree, path:line:text
        let hits = tool.invoke(&ToolCall::new("grep").arg("query", "helper")).unwrap();
        assert!(hits.content.contains("src/util.rs:1:"), "{}", hits.content);
        assert!(hits.content.contains("notes.txt:1:"), "{}", hits.content);

        // grep scoped to a subdir
        let scoped = tool
            .invoke(&ToolCall::new("grep").arg("query", "helper").arg("path", "src"))
            .unwrap();
        assert!(scoped.content.contains("src/util.rs:1:"));
        assert!(!scoped.content.contains("notes.txt"), "scoped out: {}", scoped.content);

        // read_range: 1-indexed inclusive
        let r = tool
            .invoke(&ToolCall::new("read_range").arg("path", "src/main.rs").arg("start", "2").arg("end", "3"))
            .unwrap();
        assert_eq!(r.content, "    let x = 1;\n    println!(\"hi\");");

        // move
        tool.invoke(&ToolCall::new("move").arg("from", "notes.txt").arg("to", "docs/notes.txt")).unwrap();
        assert_eq!(
            tool.invoke(&ToolCall::new("exists").arg("path", "notes.txt")).unwrap().content,
            "false"
        );
        assert_eq!(
            tool.invoke(&ToolCall::new("exists").arg("path", "docs/notes.txt")).unwrap().content,
            "true"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_wildcard_matcher() {
        assert!(wildcard_match(b"*.rs", b"src/a/b.rs"));
        assert!(wildcard_match(b"src/*/mod.rs", b"src/agent/mod.rs"));
        assert!(wildcard_match(b"*session*", b"src/agent/session.rs"));
        assert!(!wildcard_match(b"*.rs", b"src/a/b.txt"));
        assert!(wildcard_match(b"a?c", b"abc"));
        assert!(!wildcard_match(b"a?c", b"ac"));
    }

    #[test]
    fn test_exists_and_remove() {
        let root = temp_sandbox("rm");
        let tool = FsTool::new(&root);
        tool.invoke(
            &ToolCall::new("write")
                .arg("path", "f.txt")
                .arg("content", "x"),
        )
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
