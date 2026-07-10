use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureKind {
    CompileError,
    TestFailure,
    RuntimeError,
    TypeMismatch,
    MissingImport,
    Formatting,
    Lint,
    PermissionDenied,
    Timeout,
    NoChangesMade,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureAnalysis {
    pub kind: FailureKind,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub message: String,
    pub likely_cause: String,
    pub suggested_action: String,
}

pub struct FailureParser;

impl Default for FailureParser {
    fn default() -> Self {
        Self
    }
}

impl FailureParser {
    pub fn parse(&self, output: &str) -> FailureAnalysis {
        let lower = output.to_ascii_lowercase();
        let (kind, message, cause, action) = if lower.contains("permission denied") {
            (
                FailureKind::PermissionDenied,
                extract_line(output, "permission denied"),
                "policy or filesystem permission blocked the operation".to_string(),
                "review guardrail policy and request explicit permission if the operation is safe"
                    .to_string(),
            )
        } else if lower.contains("timed out") || lower.contains("timeout") {
            (
                FailureKind::Timeout,
                extract_line(output, "timeout"),
                "verification or execution exceeded its time budget".to_string(),
                "reduce scope, increase timeout, or split the task into smaller steps".to_string(),
            )
        } else if lower.contains("no changes") || lower.contains("no modification") {
            (
                FailureKind::NoChangesMade,
                extract_line(output, "no changes"),
                "agent did not produce an applicable patch".to_string(),
                "revisit localization and require a concrete diff".to_string(),
            )
        } else if lower.contains("rustfmt") || lower.contains("formatting") {
            (
                FailureKind::Formatting,
                extract_line(output, "rustfmt"),
                "patch violates formatting conventions".to_string(),
                "run formatter or adjust patch formatting".to_string(),
            )
        } else if lower.contains("clippy") || lower.contains("lint") {
            (
                FailureKind::Lint,
                extract_line(output, "clippy"),
                "lint rule violation".to_string(),
                "fix lint warning or document an intentional exception".to_string(),
            )
        } else if lower.contains("cannot find")
            || lower.contains("use of undeclared")
            || lower.contains("unresolved import")
        {
            (
                FailureKind::MissingImport,
                extract_line(output, "cannot find"),
                "symbol or import is missing".to_string(),
                "inspect references and add or correct imports".to_string(),
            )
        } else if lower.contains("type mismatch")
            || lower.contains("expected") && lower.contains("found")
        {
            (
                FailureKind::TypeMismatch,
                extract_line(output, "type mismatch"),
                "type annotation or inferred type mismatch".to_string(),
                "inspect nearby expressions and adjust types".to_string(),
            )
        } else if lower.contains("panicked at")
            || lower.contains("failures:")
            || lower.contains("test result: failed")
            || lower.contains("assertion `left == right` failed")
            || lower.contains("assertion failed:")
        {
            (
                FailureKind::TestFailure,
                extract_line(output, "panicked at"),
                "test assertion or runtime expectation failed".to_string(),
                "inspect failing test and adjust implementation or test hypothesis".to_string(),
            )
        } else if lower.contains("could not compile") || lower.contains("error[e") {
            (
                FailureKind::CompileError,
                extract_line(output, "error"),
                "compiler rejected the patch".to_string(),
                "inspect compiler diagnostics and repair the failing file".to_string(),
            )
        } else if lower.contains("test failed") {
            (
                FailureKind::TestFailure,
                extract_line(output, "test failed"),
                "test assertion or runtime expectation failed".to_string(),
                "inspect failing test and adjust implementation or test hypothesis".to_string(),
            )
        } else if lower.contains("error") {
            (
                FailureKind::RuntimeError,
                extract_line(output, "error"),
                "runtime error occurred".to_string(),
                "inspect stack trace and reproduce the failing path".to_string(),
            )
        } else {
            (
                FailureKind::Unknown,
                output
                    .lines()
                    .next()
                    .unwrap_or("unknown failure")
                    .to_string(),
                "failure output did not match a known classifier".to_string(),
                "inspect raw output and add a new classifier if needed".to_string(),
            )
        };

        let (file, line) = extract_file_and_line(output);
        FailureAnalysis {
            kind,
            file,
            line,
            message,
            likely_cause: cause,
            suggested_action: action,
        }
    }
}

fn extract_line(output: &str, needle: &str) -> String {
    let lower_needle = needle.to_ascii_lowercase();
    output
        .lines()
        .find(|line| line.to_ascii_lowercase().contains(&lower_needle))
        .unwrap_or(needle)
        .trim()
        .to_string()
}

fn extract_file_and_line(output: &str) -> (Option<String>, Option<u32>) {
    for line in output.lines() {
        let parts: Vec<&str> = line.split(':').collect();
        for index in 1..parts.len() {
            if let Ok(number) = parts[index].trim().parse::<u32>() {
                let joined = parts[..index].join(":");
                // Take only the last whitespace-separated token so a panic line
                // (`thread '..' panicked at src/lib.rs`) or a diagnostic arrow (`--> src/lib.rs`)
                // yields the clean PATH `src/lib.rs`, not the whole prefix — the repair proposers
                // treat `.file` as the file to mutate first.
                let file_candidate = joined
                    .rsplit(char::is_whitespace)
                    .next()
                    .unwrap_or(joined.trim())
                    .trim()
                    .to_string();
                if file_candidate.contains('.')
                    || file_candidate.contains('/')
                    || file_candidate.contains('\\')
                {
                    return (Some(file_candidate), Some(number));
                }
            }
        }
    }
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_compile_error() {
        let parser = FailureParser;
        let analysis = parser.parse("error[E0308]: mismatched types\n --> src/lib.rs:12:5");
        assert_eq!(analysis.kind, FailureKind::CompileError);
        assert_eq!(analysis.file.as_deref(), Some("src/lib.rs"));
        assert_eq!(analysis.line, Some(12));
    }

    #[test]
    fn classifies_timeout() {
        let parser = FailureParser;
        let analysis = parser.parse("test timed out after 30s");
        assert_eq!(analysis.kind, FailureKind::Timeout);
    }

    #[test]
    fn panic_line_yields_clean_path_and_line_not_the_whole_message() {
        // A real test panic line embeds the path AFTER prose; `.file` must be the bare path (repair
        // proposers mutate `.file` first and localize to `.line`), not `thread '..' panicked at ..`.
        let parser = FailureParser;
        let out = "running 1 test\nthread 'tests::t' panicked at src/lib.rs:140:78:\nassertion `left == right` failed";
        let analysis = parser.parse(out);
        assert_eq!(analysis.kind, FailureKind::TestFailure);
        assert_eq!(analysis.file.as_deref(), Some("src/lib.rs"));
        assert_eq!(analysis.line, Some(140));
    }
}
