use crate::agent::repo::guardrails::{GuardrailDecision, GuardrailPolicy};

#[derive(Debug, Clone, PartialEq)]
pub struct PatchGateResult {
    pub allowed: bool,
    pub rejected: Vec<String>,
    pub warnings: Vec<String>,
}

impl PatchGateResult {
    fn ok() -> Self {
        Self {
            allowed: true,
            rejected: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn reject(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            rejected: vec![reason.into()],
            warnings: Vec::new(),
        }
    }
}

pub struct PatchGate {
    policy: GuardrailPolicy,
}

impl Default for PatchGate {
    fn default() -> Self {
        Self {
            policy: GuardrailPolicy::default(),
        }
    }
}

impl PatchGate {
    pub fn new(policy: GuardrailPolicy) -> Self {
        Self { policy }
    }

    pub fn validate_paths<'a>(&self, paths: impl IntoIterator<Item = &'a str>) -> PatchGateResult {
        let mut result = PatchGateResult::ok();
        for path in paths {
            match self.policy.check_path(path, true) {
                GuardrailDecision::Allow => {}
                GuardrailDecision::Deny(reason) => {
                    result.allowed = false;
                    result.rejected.push(reason);
                }
                GuardrailDecision::Ask(reason) => {
                    result.warnings.push(reason);
                }
            }
        }
        result
    }

    pub fn validate_diff(&self, diff: &str, allowed_files: &[String]) -> PatchGateResult {
        let mut result = PatchGateResult::ok();
        let mut paths = Vec::new();
        for line in diff.lines() {
            if let Some(path) = diff_path(line) {
                paths.push(path);
            }
        }
        if paths.is_empty() {
            result.allowed = false;
            result
                .rejected
                .push("diff did not contain any file paths".to_string());
            return result;
        }
        for path in &paths {
            if !allowed_files
                .iter()
                .any(|pattern| pattern_matches(pattern, path))
            {
                result.allowed = false;
                result
                    .rejected
                    .push(format!("path is outside allowed file policy: {path}"));
            }
            match self.policy.check_path(path, true) {
                GuardrailDecision::Allow => {}
                GuardrailDecision::Deny(reason) => {
                    result.allowed = false;
                    result.rejected.push(reason);
                }
                GuardrailDecision::Ask(reason) => result.warnings.push(reason),
            }
        }
        if diff.contains("<<<<<<<") || diff.contains("=======") || diff.contains(">>>>>>>") {
            result.allowed = false;
            result
                .rejected
                .push("diff contains merge-conflict markers".to_string());
        }
        result
    }
}

fn diff_path(line: &str) -> Option<String> {
    if let Some(rest) = line.strip_prefix("+++ b/") {
        return Some(rest.trim().to_string());
    }
    if let Some(rest) = line.strip_prefix("--- a/") {
        return Some(rest.trim().to_string());
    }
    None
}

fn pattern_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.trim_start_matches("./").trim_end_matches('/');
    let path = path.trim_start_matches("./");
    if pattern.ends_with("/**") {
        let prefix = pattern.trim_end_matches("/**");
        return path == prefix || path.starts_with(&format!("{prefix}/"));
    }
    if pattern.ends_with('*') {
        let prefix = pattern.trim_end_matches('*');
        return path.starts_with(prefix);
    }
    path == pattern || path.starts_with(&format!("{pattern}/"))
}

// ---------------------------------------------------------------------------
// Safe manifest (Cargo.toml) editing
// ---------------------------------------------------------------------------
//
// Editing a build manifest is dangerous: a malicious or buggy patch could add a
// `build = "build.rs"` build script, a new `[[bin]]`, or arbitrary sections that
// execute code at compile time. We therefore refuse ALL manifest edits by
// default (see `RepairLoop::with_manifest_edits`), and even when the capability
// is enabled we only allow additive changes confined to the dependency and
// feature tables, validated by `validate_manifest_edit` below.
//
// The check is intentionally *fail-closed*: if the resulting manifest cannot be
// parsed by our (deliberately strict) mini-parser, or if any content outside the
// allowed-mutable tables changes, the edit is rejected.

/// Top-level tables whose contents a capability-enabled task may add to / change.
/// Everything else (notably `[package]`, `[lib]`, `[[bin]]`, `[build-dependencies]`)
/// must remain byte-for-byte identical.
const MANIFEST_MUTABLE_TABLES: &[&str] = &["dependencies", "dev-dependencies", "features"];

/// Returns true if `path` names a Cargo manifest at any directory depth.
pub fn is_manifest_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    normalized
        .rsplit('/')
        .next()
        .map(|file| file == "Cargo.toml")
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ManifestEntry {
    /// First dotted segment of the enclosing section header (e.g. `dependencies`
    /// for `[dependencies.serde]`). `None` for keys before any header (root table).
    top_key: Option<String>,
    is_header: bool,
    /// The trimmed logical line (multi-line values joined onto one line).
    text: String,
}

/// Validate that transforming `before` -> `after` is a *safe* manifest edit:
/// `after` must parse, and every change must be confined to the dependency /
/// feature tables. Returns `Err(reason)` describing the first violation.
pub fn validate_manifest_edit(before: &str, after: &str) -> Result<(), String> {
    let after_doc = parse_manifest(after)
        .map_err(|e| format!("resulting manifest does not parse as valid TOML: {e}"))?;
    // `before` should already be a valid manifest, but if a repo ships an exotic
    // manifest our strict parser cannot handle, fail closed by treating its fixed
    // content as empty — any surviving fixed content in `after` then trips the
    // length/equality check below.
    let before_doc = parse_manifest(before).unwrap_or_default();

    let is_mutable = |key: &Option<String>| {
        key.as_deref()
            .map(|name| MANIFEST_MUTABLE_TABLES.contains(&name))
            .unwrap_or(false)
    };

    let before_fixed: Vec<&ManifestEntry> =
        before_doc.iter().filter(|e| !is_mutable(&e.top_key)).collect();
    let after_fixed: Vec<&ManifestEntry> =
        after_doc.iter().filter(|e| !is_mutable(&e.top_key)).collect();

    if before_fixed.len() != after_fixed.len() {
        return Err(
            "manifest edit adds or removes content outside [dependencies]/[dev-dependencies]/[features]"
                .to_string(),
        );
    }
    for (b, a) in before_fixed.iter().zip(after_fixed.iter()) {
        if b.is_header != a.is_header || b.top_key != a.top_key || b.text != a.text {
            return Err(format!(
                "manifest edit alters disallowed section content: '{}' -> '{}'",
                b.text, a.text
            ));
        }
    }

    // Defense in depth: never allow a newly-introduced build script even if some
    // future change to the comparison above let it slip through.
    let package_build = |doc: &[ManifestEntry]| {
        doc.iter().any(|e| {
            !e.is_header
                && e.top_key.as_deref() == Some("package")
                && e.text
                    .split('=')
                    .next()
                    .map(|k| k.trim() == "build")
                    .unwrap_or(false)
        })
    };
    if package_build(&after_doc) && !package_build(&before_doc) {
        return Err("manifest edit introduces a build script, which is forbidden".to_string());
    }

    Ok(())
}

/// Strict, fail-closed structural parse of a Cargo manifest into tagged entries.
fn parse_manifest(text: &str) -> Result<Vec<ManifestEntry>, String> {
    let lines = manifest_logical_lines(text)?;
    let mut entries = Vec::new();
    let mut current: Option<String> = None;
    for line in lines {
        if line.starts_with('[') {
            let top = manifest_header_top_key(&line)?;
            current = top.clone();
            entries.push(ManifestEntry {
                top_key: top,
                is_header: true,
                text: line,
            });
        } else {
            let key = line.split('=').next().unwrap_or("").trim();
            if !line.contains('=') || key.is_empty() {
                return Err(format!(
                    "line is not a valid TOML header or assignment: {line}"
                ));
            }
            entries.push(ManifestEntry {
                top_key: current.clone(),
                is_header: false,
                text: line,
            });
        }
    }
    Ok(entries)
}

/// Segment `text` into logical TOML lines, honoring strings, comments, and
/// bracket/brace continuation (so multi-line arrays and inline tables collapse to
/// one logical line). Returns `Err` for unbalanced brackets or unterminated
/// single-line strings.
fn manifest_logical_lines(text: &str) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    let mut buf = String::new();
    let mut in_basic = false; // inside a "double-quoted" string
    let mut in_literal = false; // inside a 'single-quoted' string
    let mut in_comment = false;
    let mut escape = false;
    let mut depth: i32 = 0;

    for c in text.chars() {
        if c == '\n' {
            if in_basic || in_literal {
                return Err("unterminated string at end of line".to_string());
            }
            in_comment = false;
            if depth == 0 {
                let trimmed = buf.trim().to_string();
                if !trimmed.is_empty() {
                    lines.push(trimmed);
                }
                buf.clear();
            } else {
                buf.push(' ');
            }
            continue;
        }
        if in_comment {
            continue;
        }
        if in_basic {
            buf.push(c);
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_basic = false;
            }
            continue;
        }
        if in_literal {
            buf.push(c);
            if c == '\'' {
                in_literal = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_basic = true;
                buf.push(c);
            }
            '\'' => {
                in_literal = true;
                buf.push(c);
            }
            '#' => in_comment = true,
            '[' | '{' => {
                depth += 1;
                buf.push(c);
            }
            ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    return Err("unbalanced closing bracket".to_string());
                }
                buf.push(c);
            }
            _ => buf.push(c),
        }
    }

    if in_basic || in_literal {
        return Err("unterminated string".to_string());
    }
    if depth != 0 {
        return Err("unbalanced brackets".to_string());
    }
    let trimmed = buf.trim().to_string();
    if !trimmed.is_empty() {
        lines.push(trimmed);
    }
    Ok(lines)
}

/// Extract the first dotted segment (the top-level table name) of a header line
/// such as `[dependencies.serde]` or `[[bin]]`. Rejects malformed headers.
fn manifest_header_top_key(line: &str) -> Result<Option<String>, String> {
    let inner = if line.starts_with("[[") {
        if !line.ends_with("]]") {
            return Err(format!("malformed array-of-tables header: {line}"));
        }
        &line[2..line.len() - 2]
    } else {
        if !line.ends_with(']') {
            return Err(format!("malformed table header: {line}"));
        }
        &line[1..line.len() - 1]
    };
    let inner = inner.trim();
    if inner.is_empty() {
        return Err("empty section header".to_string());
    }
    let segment = manifest_first_segment(inner);
    let segment = segment
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string();
    if segment.is_empty() {
        return Err(format!("empty section name: {line}"));
    }
    Ok(Some(segment))
}

/// First `.`-delimited segment of a section path, respecting quotes.
fn manifest_first_segment(s: &str) -> String {
    let mut out = String::new();
    let mut in_basic = false;
    let mut in_literal = false;
    for c in s.chars() {
        match c {
            '"' if !in_literal => {
                in_basic = !in_basic;
                out.push(c);
            }
            '\'' if !in_basic => {
                in_literal = !in_literal;
                out.push(c);
            }
            '.' if !in_basic && !in_literal => break,
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_MANIFEST: &str = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nserde = \"1\"\n\n[lib]\npath = \"src/lib.rs\"\n";

    #[test]
    fn manifest_add_dependency_is_accepted() {
        let after = BASE_MANIFEST.replace(
            "[dependencies]\nserde = \"1\"\n",
            "[dependencies]\nserde = \"1\"\nregex = \"1\"\n",
        );
        assert!(validate_manifest_edit(BASE_MANIFEST, &after).is_ok());
    }

    #[test]
    fn manifest_add_new_dependency_table_is_accepted() {
        let before = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n";
        let after = format!("{before}\n[dependencies]\nserde = {{ version = \"1\", features = [\"derive\"] }}\n");
        assert!(validate_manifest_edit(before, &after).is_ok());
    }

    #[test]
    fn manifest_add_feature_is_accepted() {
        let before = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n\n[features]\ndefault = []\n";
        let after = "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n\n[features]\ndefault = []\nfast = [\n  \"serde\",\n]\n";
        assert!(validate_manifest_edit(before, after).is_ok());
    }

    #[test]
    fn manifest_build_script_is_rejected() {
        let after = BASE_MANIFEST.replace(
            "name = \"demo\"\n",
            "name = \"demo\"\nbuild = \"build.rs\"\n",
        );
        let err = validate_manifest_edit(BASE_MANIFEST, &after).unwrap_err();
        // Rejected because it mutates [package] (adds a build-script line); the
        // dedicated build-script guard is belt-and-suspenders behind this.
        assert!(
            err.contains("build script") || err.contains("outside") || err.contains("disallowed"),
            "got: {err}"
        );
    }

    #[test]
    fn manifest_new_bin_section_is_rejected() {
        let after = format!("{BASE_MANIFEST}\n[[bin]]\nname = \"evil\"\npath = \"x.rs\"\n");
        let err = validate_manifest_edit(BASE_MANIFEST, &after).unwrap_err();
        assert!(err.contains("outside") || err.contains("disallowed"), "got: {err}");
    }

    #[test]
    fn manifest_build_dependencies_addition_is_rejected() {
        let after = format!("{BASE_MANIFEST}\n[build-dependencies]\ncc = \"1\"\n");
        assert!(validate_manifest_edit(BASE_MANIFEST, &after).is_err());
    }

    #[test]
    fn manifest_modifying_package_is_rejected() {
        let after = BASE_MANIFEST.replace("version = \"0.1.0\"", "version = \"9.9.9\"");
        assert!(validate_manifest_edit(BASE_MANIFEST, &after).is_err());
    }

    #[test]
    fn manifest_malformed_toml_is_rejected() {
        // Unbalanced inline-table brace.
        let after = BASE_MANIFEST.replace(
            "[dependencies]\nserde = \"1\"\n",
            "[dependencies]\nserde = { version = \"1\"\n",
        );
        let err = validate_manifest_edit(BASE_MANIFEST, &after).unwrap_err();
        assert!(err.contains("does not parse"), "got: {err}");
    }

    #[test]
    fn is_manifest_path_matches_nested_manifests() {
        assert!(is_manifest_path("Cargo.toml"));
        assert!(is_manifest_path("crates/foo/Cargo.toml"));
        assert!(!is_manifest_path("src/lib.rs"));
        assert!(!is_manifest_path("Cargo.lock"));
    }

    #[test]
    fn rejects_paths_outside_allowed_policy() {
        let gate = PatchGate::default();
        let diff =
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n-pub fn a() {}\n+pub fn a() -> i32 { 1 }\n";
        let result = gate.validate_diff(diff, &["tests/**".to_string()]);
        assert!(!result.allowed);
        assert!(result
            .rejected
            .iter()
            .any(|reason| reason.contains("outside allowed")));
    }

    #[test]
    fn rejects_merge_markers() {
        let gate = PatchGate::default();
        let diff = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n+<<<<<<< HEAD\n";
        let result = gate.validate_diff(diff, &["src/**".to_string()]);
        assert!(!result.allowed);
        assert!(result
            .rejected
            .iter()
            .any(|reason| reason.contains("merge-conflict")));
    }

    #[test]
    fn validates_allowed_paths() {
        let gate = PatchGate::default();
        let diff =
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@\n-pub fn a() {}\n+pub fn a() -> i32 { 1 }\n";
        let result = gate.validate_diff(diff, &["src/**".to_string()]);
        assert!(result.allowed);
    }
}
