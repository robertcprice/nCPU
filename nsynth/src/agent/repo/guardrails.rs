use regex::Regex;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailDecision {
    Allow,
    Deny(String),
    Ask(String),
}

#[derive(Debug, Clone)]
pub struct GuardrailPolicy {
    ignored_patterns: Vec<String>,
    immutable_patterns: Vec<String>,
    unsafe_commands: Vec<(String, String)>,
    secret_patterns: Vec<(String, String)>,
}

impl Default for GuardrailPolicy {
    fn default() -> Self {
        Self {
            ignored_patterns: vec![
                ".env".to_string(),
                ".env.local".to_string(),
                ".env.production".to_string(),
                ".git/".to_string(),
                ".ssh/".to_string(),
                "secret/".to_string(),
                "private/".to_string(),
                "dist/".to_string(),
                "build/".to_string(),
            ],
            immutable_patterns: vec![
                ".claude/.immutable".to_string(),
                ".claude/.ignore".to_string(),
                ".claude/settings.json".to_string(),
                ".claude/hooks/".to_string(),
                ".claude/rules.md".to_string(),
            ],
            unsafe_commands: vec![
                (
                    "rm -rf".to_string(),
                    "recursive delete is unsafe".to_string(),
                ),
                (
                    "git reset".to_string(),
                    "git reset can discard work".to_string(),
                ),
                (
                    "git push --force".to_string(),
                    "force push can rewrite remote history".to_string(),
                ),
                (
                    "git clean".to_string(),
                    "git clean can delete untracked files".to_string(),
                ),
                (
                    "chmod".to_string(),
                    "permission changes require review".to_string(),
                ),
                (
                    "kill -9".to_string(),
                    "force kill requires review".to_string(),
                ),
                (
                    "drop database".to_string(),
                    "database drop is destructive".to_string(),
                ),
                (
                    "drop table".to_string(),
                    "table drop is destructive".to_string(),
                ),
                (
                    "cat .env".to_string(),
                    "reading .env can expose secrets".to_string(),
                ),
                ("source".to_string(), "source executes files".to_string()),
                ("eval".to_string(), "eval executes strings".to_string()),
            ],
            secret_patterns: vec![
                (r"(?i)api[_-]?key\s*[:=]".to_string(), "API key".to_string()),
                (r"(?i)password\s*[:=]".to_string(), "password".to_string()),
                (r"(?i)secret\s*[:=]".to_string(), "secret".to_string()),
                (r"(?i)token\s*[:=]".to_string(), "token".to_string()),
                (
                    r"sk-[A-Za-z0-9]{20,}".to_string(),
                    "OpenAI-style key".to_string(),
                ),
                (
                    r"ghp_[A-Za-z0-9]{36}".to_string(),
                    "GitHub token".to_string(),
                ),
                (
                    r"-----BEGIN (?:RSA |OPENSSH |DSA |EC )?PRIVATE KEY-----".to_string(),
                    "private key".to_string(),
                ),
            ],
        }
    }
}

impl GuardrailPolicy {
    pub fn new(ignored_patterns: Vec<String>, immutable_patterns: Vec<String>) -> Self {
        let mut policy = Self::default();
        policy.ignored_patterns.extend(ignored_patterns);
        policy.immutable_patterns.extend(immutable_patterns);
        policy
    }

    pub fn with_unsafe_command(
        mut self,
        token: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        self.unsafe_commands.push((token.into(), reason.into()));
        self
    }

    pub fn check_path(&self, path: impl AsRef<Path>, writable: bool) -> GuardrailDecision {
        let normalized = normalize_path(path.as_ref());
        if self
            .ignored_patterns
            .iter()
            .any(|pattern| pattern_matches(pattern, &normalized))
        {
            return GuardrailDecision::Deny(format!(
                "path is ignored by guardrail policy: {normalized}"
            ));
        }
        if writable
            && self
                .immutable_patterns
                .iter()
                .any(|pattern| pattern_matches(pattern, &normalized))
        {
            return GuardrailDecision::Deny(format!("path is immutable: {normalized}"));
        }
        GuardrailDecision::Allow
    }

    pub fn check_command(&self, command: &str) -> GuardrailDecision {
        for (token, reason) in &self.unsafe_commands {
            if command.contains(token) {
                return GuardrailDecision::Ask(format!("unsafe command flagged: {reason}"));
            }
        }
        if let Some(secret) = self.detect_secret(command) {
            return GuardrailDecision::Deny(format!(
                "command contains secret-like content: {secret}"
            ));
        }
        GuardrailDecision::Allow
    }

    pub fn check_text_for_secret(&self, text: &str) -> Option<String> {
        for (pattern, label) in &self.secret_patterns {
            if Regex::new(pattern).is_ok_and(|regex| regex.is_match(text)) {
                return Some(label.clone());
            }
        }
        None
    }

    pub fn detect_secret(&self, text: &str) -> Option<String> {
        self.check_text_for_secret(text)
    }
}

fn normalize_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string()
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
    fn denies_ignored_paths() {
        let policy = GuardrailPolicy::default();
        assert!(matches!(
            policy.check_path(".env", false),
            GuardrailDecision::Deny(_)
        ));
        assert!(matches!(
            policy.check_path(".ssh/id_rsa", false),
            GuardrailDecision::Deny(_)
        ));
    }

    #[test]
    fn denies_immutable_writes() {
        let policy = GuardrailPolicy::default();
        assert!(matches!(
            policy.check_path(".claude/rules.md", true),
            GuardrailDecision::Deny(_)
        ));
    }

    #[test]
    fn asks_for_unsafe_commands() {
        let policy = GuardrailPolicy::default();
        assert!(matches!(
            policy.check_command("rm -rf build"),
            GuardrailDecision::Ask(_)
        ));
    }

    #[test]
    fn detects_secret_like_text() {
        let policy = GuardrailPolicy::default();
        assert_eq!(
            policy.detect_secret("password = hunter2"),
            Some("password".to_string())
        );
    }
}
