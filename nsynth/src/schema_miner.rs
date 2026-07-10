//! WP6 — schema mining over verified (task, code) corpora.
//!
//! Pure functions: normalize identifiers + integer literals to holes, cluster by
//! normalized statement sequence, report top-k recurring templates, and
//! instantiate a mined template against a `Problem` (const-hole fill + verify).
//! No lg-core dependency.

use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinedTemplate {
    pub normalized: String,
    pub count: usize,
    pub example_task: String,
    pub example_code: String,
}

/// Replace snake_case / camelCase identifiers with `?v0`, `?v1`, … and integer
/// literals with `?c0`, `?c1`, … (stable per occurrence order within one program).
/// Keywords and short builtins are kept. Floats are left alone (rare in harvest).
pub fn normalize_to_holes(code: &str) -> String {
    const KEEP: &[&str] = &[
        "fn", "let", "mut", "if", "else", "while", "for", "in", "loop", "return", "true", "false",
        "i64", "i32", "u64", "bool", "str", "String", "Vec", "vec", "Some", "None", "Ok", "Err",
        "self", "Self", "pub", "mod", "use", "crate", "super", "as", "match", "break", "continue",
        "struct", "impl", "enum", "const", "static", "type", "where", "ref", "move", "async",
        "await", "dyn", "trait", "unsafe", "extern", "box", "Box", "Option", "Result",
    ];
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut const_map: HashMap<String, String> = HashMap::new();
    let mut next_v = 0usize;
    let mut next_c = 0usize;
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
                let hole = id_map.entry(ident).or_insert_with(|| {
                    let h = format!("?v{next_v}");
                    next_v += 1;
                    h
                });
                out.push_str(hole);
            }
        } else if c == '-' || c.is_ascii_digit() {
            // Integer literal (optional leading `-`). Do not hole a digit that is
            // part of an identifier (already handled) or a float (`1.0`).
            let mut lit = String::new();
            if c == '-' {
                // Only a negative literal if next is a digit and prev token ended
                // on a non-ident (we are at a fresh token boundary here).
                match chars.peek() {
                    Some(d) if d.is_ascii_digit() => {
                        lit.push('-');
                    }
                    _ => {
                        out.push(c);
                        continue;
                    }
                }
            } else {
                lit.push(c);
            }
            while let Some(&n) = chars.peek() {
                if n.is_ascii_digit() {
                    lit.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            // Skip floats: if next is `.` + digit, emit the int part raw and continue.
            if chars.peek() == Some(&'.') {
                let mut look = chars.clone();
                look.next(); // '.'
                if look.peek().is_some_and(|d| d.is_ascii_digit()) {
                    out.push_str(&lit);
                    continue;
                }
            }
            let hole = const_map.entry(lit).or_insert_with(|| {
                let h = format!("?c{next_c}");
                next_c += 1;
                h
            });
            out.push_str(hole);
        } else {
            out.push(c);
        }
    }
    out
}

/// Back-compat alias used by older call sites / tests.
pub fn normalize_identifiers_to_holes(code: &str) -> String {
    normalize_to_holes(code)
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
        let norm = normalize_to_holes(code);
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

/// Persist mined templates as JSON (array) for `NSYNTH_MINED_TEMPLATES`.
pub fn write_templates_json(path: &Path, templates: &[MinedTemplate]) -> Result<(), String> {
    let text = serde_json::to_string_pretty(templates).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Load templates previously written by [`write_templates_json`] or `mine_schemas --out`.
pub fn load_templates_json(path: &Path) -> Result<Vec<MinedTemplate>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Load templates from `NSYNTH_MINED_TEMPLATES` env (JSON path), or fall back to
/// `.nsynth/mined_templates.json` in the current working directory / home.
pub fn load_templates_from_env() -> Option<Vec<MinedTemplate>> {
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(path) = std::env::var("NSYNTH_MINED_TEMPLATES") {
            if !path.is_empty() {
                v.push(std::path::PathBuf::from(path));
            }
        }
        v.push(std::path::PathBuf::from(".nsynth/mined_templates.json"));
        if let Some(home) = std::env::var_os("HOME") {
            v.push(std::path::PathBuf::from(home).join(".nsynth/mined_templates.json"));
        }
        v
    };
    for path in candidates {
        if path.is_file() {
            if let Ok(t) = load_templates_json(&path) {
                if !t.is_empty() {
                    return Some(t);
                }
            }
        }
    }
    None
}

/// Append one verified (task, code) row to `NSYNTH_HARVEST` JSONL (Phase-4 flywheel).
/// Best-effort; never fails the caller. No-op when the env var is unset.
pub fn append_harvest_row(task: &str, code: &str) {
    let Some(path) = std::env::var_os("NSYNTH_HARVEST") else {
        return;
    };
    if path.is_empty() || code.trim().is_empty() {
        return;
    }
    let row = serde_json::json!({
        "task": task,
        "code": code,
    });
    let Ok(line) = serde_json::to_string(&row) else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{line}");
    }
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

/// Substitute `?cN` holes in a normalized template with concrete integer consts.
/// Identifier holes `?vN` are renamed: entry fn → `fn_name`, params → `a0..`, locals kept as `vN`.
pub fn instantiate_template(normalized: &str, fn_name: &str, consts: &[i64]) -> String {
    let mut s = normalized.to_string();
    for (i, c) in consts.iter().enumerate() {
        s = s.replace(&format!("?c{i}"), &c.to_string());
    }
    // Rename first `fn ?v0` (or bare `fn ?vN`) to the problem entry name.
    // Map remaining ?v holes to stable rust idents.
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    let mut seen_fn = false;
    let mut v_map: HashMap<String, String> = HashMap::new();
    let mut next_param = 0usize;
    while let Some(c) = chars.next() {
        if c == '?' {
            let mut hole = String::from("?");
            while let Some(&n) = chars.peek() {
                if n.is_ascii_alphanumeric() {
                    hole.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            if hole.starts_with("?c") {
                // Unfilled const — leave as-is (caller should have filled).
                out.push_str(&hole);
            } else if hole.starts_with("?v") {
                if !seen_fn && out.trim_end().ends_with("fn") {
                    // Immediately after `fn` → entry name.
                    seen_fn = true;
                    out.push_str(fn_name);
                } else {
                    let name = v_map.entry(hole.clone()).or_insert_with(|| {
                        let p = format!("a{next_param}");
                        next_param += 1;
                        p
                    });
                    out.push_str(name);
                }
            } else {
                out.push_str(&hole);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Mine candidate integer constants from a problem's examples (inputs, outputs,
/// pairwise diffs/products/quotients when exact).
pub fn mine_const_candidates(examples: &[crate::benchmark::Example]) -> Vec<i64> {
    use crate::benchmark::Value;
    let mut set = std::collections::BTreeSet::new();
    set.insert(0);
    set.insert(1);
    set.insert(2);
    set.insert(-1);
    for ex in examples {
        for v in ex.inputs.iter().chain(std::iter::once(&ex.expected)) {
            if let Value::Int(i) = v {
                set.insert(*i);
            }
        }
        // Unary affine: out = in * k or in + k
        if ex.inputs.len() == 1 {
            if let (Value::Int(x), Value::Int(y)) = (&ex.inputs[0], &ex.expected) {
                if *x != 0 && y % x == 0 {
                    set.insert(y / x);
                }
                set.insert(y - x);
            }
        }
    }
    set.into_iter().filter(|c| c.abs() <= 10_000).collect()
}

/// Count how many `?cN` holes appear in a normalized template (max index + 1).
pub fn count_const_holes(normalized: &str) -> usize {
    let mut max = None;
    let mut rest = normalized;
    while let Some(i) = rest.find("?c") {
        let after = &rest[i + 2..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<usize>() {
            max = Some(max.map_or(n, |m: usize| m.max(n)));
        }
        rest = &after[digits.len().max(0)..];
        if digits.is_empty() {
            rest = &after[1.min(after.len())..];
        }
    }
    max.map(|m| m + 1).unwrap_or(0)
}

/// Try to instantiate mined templates against a problem; return first code that
/// reproduces every example. Templates come from `NSYNTH_MINED_TEMPLATES` or the
/// provided slice. Bounded: at most `max_templates` × small const product.
pub fn try_instantiate_templates(
    problem: &crate::benchmark::Problem,
    templates: &[MinedTemplate],
    max_templates: usize,
) -> Option<String> {
    let name = {
        let n = problem.function_name();
        if n.is_empty() {
            "f"
        } else {
            n
        }
    };
    let consts = mine_const_candidates(&problem.examples);
    let mut tried = 0usize;
    for t in templates.iter().filter(|t| t.count >= 1).take(max_templates) {
        let n_holes = count_const_holes(&t.normalized);
        if n_holes > 3 {
            continue; // combinatorial blow-up
        }
        // Prefer the example_code shape when it has no holes left after rename —
        // but primary path is normalized + const fill.
        let combos: Vec<Vec<i64>> = if n_holes == 0 {
            vec![vec![]]
        } else {
            bounded_const_product(&consts, n_holes, 64)
        };
        for combo in combos {
            tried += 1;
            if tried > 256 {
                return None;
            }
            let code = instantiate_template(&t.normalized, name, &combo);
            // Statement-sequence keys use ` | ` — restore newlines for Mog-ish bodies.
            let code = code.replace(" | ", "\n");
            if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
                return Some(code);
            }
        }
        // Also try the raw example_code with fn renamed (exact shape reuse).
        if !t.example_code.is_empty() {
            let renamed = rename_fn_roughly(&t.example_code, name);
            if crate::runtime::code_reproduces_examples(&renamed, &problem.examples) {
                return Some(renamed);
            }
        }
    }
    None
}

fn bounded_const_product(consts: &[i64], n: usize, cap: usize) -> Vec<Vec<i64>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = vec![vec![]];
    for _ in 0..n {
        let mut next = Vec::new();
        for prefix in &out {
            for &c in consts {
                if next.len() >= cap {
                    break;
                }
                let mut row = prefix.clone();
                row.push(c);
                next.push(row);
            }
            if next.len() >= cap {
                break;
            }
        }
        out = next;
    }
    out
}

fn rename_fn_roughly(code: &str, new_name: &str) -> String {
    // Replace first `fn <ident>` with `fn new_name`.
    if let Some(i) = code.find("fn ") {
        let after = &code[i + 3..];
        let ident_len = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .count();
        if ident_len > 0 {
            return format!("{}fn {}{}", &code[..i], new_name, &after[ident_len..]);
        }
    }
    code.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_replaces_snake_idents_with_holes() {
        let n = normalize_to_holes("fn double(x: i64) -> i64 { return x * 2; }");
        assert!(n.contains("?v0") || n.contains("fn ?v"), "got {n}");
        assert!(n.contains("i64"));
        assert!(n.contains("return"));
        assert!(n.contains("?c"), "expected const hole for 2, got {n}");
        let n2 = normalize_to_holes("let acc = acc + item;");
        assert!(n2.contains("?v0"));
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

    #[test]
    fn instantiate_fills_const_and_renames_fn() {
        let norm = "fn ?v0(?v1: i64) -> i64 { return ?v1 * ?c0; }";
        let code = instantiate_template(norm, "double", &[2]);
        assert!(code.contains("fn double"), "got {code}");
        assert!(code.contains("* 2"), "got {code}");
        assert!(!code.contains("?c"), "got {code}");
    }

    #[test]
    fn count_const_holes_reads_max_index() {
        assert_eq!(count_const_holes("return ?v1 * ?c0 + ?c1;"), 2);
        assert_eq!(count_const_holes("return ?v1;"), 0);
    }

    #[test]
    fn append_harvest_row_writes_jsonl() {
        let path = std::env::temp_dir().join(format!("nsynth_harvest_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("NSYNTH_HARVEST", &path);
        append_harvest_row("double", "fn double(x: i64) -> i64 { return x * 2; }");
        append_harvest_row("triple", "fn triple(x: i64) -> i64 { return x * 3; }");
        let text = std::fs::read_to_string(&path).expect("harvest file");
        assert_eq!(text.lines().count(), 2);
        assert!(text.contains("\"task\":\"double\""));
        std::env::remove_var("NSYNTH_HARVEST");
        let _ = std::fs::remove_file(&path);
    }
}
