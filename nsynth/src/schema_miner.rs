//! WP6 — schema mining over verified (task, code) corpora.
//!
//! Pure functions: normalize identifiers to holes, cluster by normalized
//! statement sequence, report top-k recurring templates. No lg-core dependency.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// One harvested solve row (JSONL `{task, code}` or `{prompt, program}`).
#[derive(Debug, Clone, Deserialize)]
pub struct HarvestRow {
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub program: Option<String>,
}

impl HarvestRow {
    pub fn task_text(&self) -> &str {
        self.task
            .as_deref()
            .or(self.prompt.as_deref())
            .unwrap_or("")
    }

    pub fn code_text(&self) -> &str {
        self.code
            .as_deref()
            .or(self.program.as_deref())
            .unwrap_or("")
    }
}

/// A mined template with occurrence count and example hole fillings.
#[derive(Debug, Clone)]
pub struct MinedTemplate {
    pub normalized: String,
    pub count: usize,
    pub example_task: String,
    pub example_code: String,
}

/// Replace snake_case / camelCase identifiers with `?v0`, `?v1`, … (stable per
/// occurrence order within one program). Keywords and short builtins are kept.
pub fn normalize_identifiers_to_holes(code: &str) -> String {
    const KEEP: &[&str] = &[
        "fn", "let", "mut", "if", "else", "while", "for", "in", "loop", "return", "true", "false",
        "i64", "i32", "u64", "bool", "str", "String", "Vec", "vec", "Some", "None", "Ok", "Err",
        "self", "Self", "pub", "mod", "use", "crate", "super", "as", "match", "break", "continue",
        "struct", "impl", "enum", "const", "static", "type", "where", "ref", "move", "async",
        "await", "dyn", "trait", "unsafe", "extern", "box", "Box", "Option", "Result",
    ];
    let mut map: HashMap<String, String> = HashMap::new();
    let mut next = 0usize;
    let mut out = String::with_capacity(code.len());
    let mut chars = code.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_alphabetic() || c == '_' {
            let mut ident = String::new();
            ident.push(c);
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphanumeric() || n == '_' {
                    ident.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if KEEP.contains(&ident.as_str()) {
                out.push_str(&ident);
            } else {
                let hole = map.entry(ident).or_insert_with(|| {
                    let h = format!("?v{next}");
                    next += 1;
                    h
                });
                out.push_str(hole);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Split a program into a normalized statement-sequence key (semicolon / newline
/// separated, whitespace-collapsed).
pub fn statement_sequence_key(normalized_code: &str) -> String {
    normalized_code
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("//") && !l.starts_with("//!"))
        .map(|l| {
            l.trim_end_matches(';')
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Cluster harvest rows by normalized statement sequence; return top-k by count.
pub fn cluster_templates(rows: &[HarvestRow], top_k: usize) -> Vec<MinedTemplate> {
    let mut clusters: HashMap<String, MinedTemplate> = HashMap::new();
    for row in rows {
        let code = row.code_text();
        if code.trim().is_empty() {
            continue;
        }
        let norm = normalize_identifiers_to_holes(code);
        let key = statement_sequence_key(&norm);
        if key.is_empty() {
            continue;
        }
        clusters
            .entry(key.clone())
            .and_modify(|t| t.count += 1)
            .or_insert(MinedTemplate {
                normalized: key,
                count: 1,
                example_task: row.task_text().to_string(),
                example_code: code.to_string(),
            });
    }
    let mut v: Vec<_> = clusters.into_values().collect();
    v.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.normalized.cmp(&b.normalized)));
    if top_k == 0 {
        v
    } else {
        v.into_iter().take(top_k).collect()
    }
}

/// Load JSONL harvest file into rows (skips blank / malformed lines).
pub fn load_harvest_jsonl(path: &Path) -> Result<Vec<HarvestRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<HarvestRow>(line) {
            Ok(r) => rows.push(r),
            Err(e) => eprintln!("schema_miner: skip line {}: {e}", i + 1),
        }
    }
    Ok(rows)
}

/// Format top templates for CLI stdout.
pub fn format_top_templates(templates: &[MinedTemplate]) -> String {
    let mut out = String::new();
    for (i, t) in templates.iter().enumerate() {
        out.push_str(&format!(
            "#{} count={} task={:?}\n  key: {}\n  eg: {}\n",
            i + 1,
            t.count,
            t.example_task.chars().take(80).collect::<String>(),
            t.normalized.chars().take(200).collect::<String>(),
            t.example_code
                .lines()
                .take(3)
                .collect::<Vec<_>>()
                .join(" / ")
                .chars()
                .take(160)
                .collect::<String>()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_replaces_snake_idents_with_holes() {
        let n = normalize_identifiers_to_holes("fn double(x: i64) -> i64 { return x * 2; }");
        assert!(n.contains("?v0") || n.contains("fn ?v"), "got {n}");
        assert!(n.contains("i64"));
        assert!(n.contains("return"));
        // same ident → same hole
        assert!(
            n.matches("?v").count() >= 1,
            "expected holes in {n}"
        );
        let n2 = normalize_identifiers_to_holes("let acc = acc + item;");
        assert!(n2.contains("?v0"));
        // acc appears twice → same hole
        let hole = n2.split_whitespace().find(|t| t.starts_with("?v")).unwrap();
        assert_eq!(n2.matches(hole).count(), 2);
    }

    #[test]
    fn clusters_identical_shapes() {
        let rows = vec![
            HarvestRow {
                task: Some("double".into()),
                code: Some("fn double(x: i64) -> i64 { return x * 2; }".into()),
                prompt: None,
                program: None,
            },
            HarvestRow {
                task: Some("triple".into()),
                code: Some("fn triple(y: i64) -> i64 { return y * 3; }".into()),
                prompt: None,
                program: None,
            },
            HarvestRow {
                task: Some("add".into()),
                code: Some("fn add(a: i64, b: i64) -> i64 { return a + b; }".into()),
                prompt: None,
                program: None,
            },
        ];
        let top = cluster_templates(&rows, 5);
        assert!(!top.is_empty());
        // double/triple share the unary *const shape after hole-norm
        let max = top.iter().map(|t| t.count).max().unwrap();
        assert!(max >= 2, "expected shared template, got {top:?}");
    }

    #[test]
    fn statement_sequence_collapses_whitespace() {
        let a = statement_sequence_key("return  x  *  2");
        let b = statement_sequence_key("return x * 2;");
        assert_eq!(a, b);
    }
}
