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

#[cfg(test)]
mod tests {
    use super::*;

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
