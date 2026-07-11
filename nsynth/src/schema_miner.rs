//! WP6 — schema mining over verified (task, code) corpora.
//!
//! Pure functions: normalize identifiers + integer literals to holes, cluster by
//! normalized statement sequence, report top-k recurring templates, and
//! instantiate a mined template against a `Problem` (const-hole fill + verify).
//! Auto-mine: [`append_harvest_row`] refreshes `.nsynth/mined_templates.json`
//! (or `NSYNTH_MINED_TEMPLATES`) once the harvest has ≥2 rows — closes the
//! harvest→mine→instantiate flywheel without a manual `mine_schemas` step.
//! No lg-core dependency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    /// Hole-normalized program body (`?vN` / `?cN`) — used by
    /// [`try_instantiate_templates`] to fill consts and rename the entry fn.
    pub normalized: String,
    /// Statement-sequence cluster key (whitespace-collapsed, ` | `-joined).
    /// Kept separate from [`Self::normalized`] so instantiate does not run on the key.
    #[serde(default)]
    pub cluster_key: String,
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
                // Store the hole-normalized BODY (not the cluster key) so
                // instantiate_template can fill ?cN / rename ?vN.
                normalized: norm,
                cluster_key: key,
                count: 1,
                example_task: row.task_text().to_string(),
                example_code: code.to_string(),
            });
    }
    let mut v: Vec<_> = clusters.into_values().collect();
    v.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.cluster_key.cmp(&b.cluster_key))
            .then_with(|| a.normalized.cmp(&b.normalized))
    });
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

/// Extract the first top-level `fn` / `pub fn` body from a Rust `lib.rs`,
/// stripping `#[cfg(test)]` modules and module-level noise so harvest clusters
/// on the verified implementation, not the oracle tests.
pub fn extract_entry_fn_body(lib_rs: &str) -> String {
    // Drop test modules (and anything after the first #[cfg(test)]).
    let code = match lib_rs.find("#[cfg(test)]") {
        Some(i) => &lib_rs[..i],
        None => lib_rs,
    };
    let mut search = 0usize;
    while search < code.len() {
        let rest = &code[search..];
        let rel = rest
            .find("pub fn ")
            .map(|i| (i, 7))
            .or_else(|| rest.find("fn ").map(|i| (i, 3)));
        let Some((rel, kw_len)) = rel else {
            break;
        };
        let at = search + rel;
        // Skip if this `fn` is inside a comment line.
        let line_start = code[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
        if code[line_start..at].trim_start().starts_with("//") {
            search = at + kw_len;
            continue;
        }
        // Find opening brace of the body.
        let after_kw = at + kw_len;
        let Some(brace_rel) = code[after_kw..].find('{') else {
            search = after_kw;
            continue;
        };
        let open = after_kw + brace_rel;
        let mut depth = 0i32;
        let mut end = None;
        for (i, &c) in code.as_bytes().iter().enumerate().skip(open) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end {
            return code[at..end].trim().to_string();
        }
        search = open + 1;
    }
    // Fallback: return the pre-test slice trimmed.
    code.trim().to_string()
}

/// Append one verified (task, code) row to `NSYNTH_HARVEST` JSONL (Phase-4 flywheel).
/// Best-effort; never fails the caller. No-op when the env var is unset/empty.
/// When `code` looks like a full `lib.rs`, only the entry fn body is stored.
///
/// After a successful append, best-effort **auto-mines** templates into
/// [`mined_templates_out_path`] when the harvest has ≥2 rows (disable with
/// `NSYNTH_AUTO_MINE=0`; throttle with `NSYNTH_AUTO_MINE_EVERY=N`).
pub fn append_harvest_row(task: &str, code: &str) {
    let Ok(path) = std::env::var("NSYNTH_HARVEST") else {
        return;
    };
    if path.is_empty() || code.trim().is_empty() {
        return;
    }
    let body = if code.contains("fn ") || code.contains("pub fn ") {
        extract_entry_fn_body(code)
    } else {
        code.trim().to_string()
    };
    if body.is_empty() {
        return;
    }
    let row = serde_json::json!({
        "task": task,
        "code": body,
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
        if writeln!(f, "{line}").is_ok() {
            maybe_refresh_templates_after_harvest(&path);
        }
    }
}

/// One row in the parallel Rust learned store (`NSYNTH_RUST_LEARNED` JSONL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustLearnedRow {
    pub name: String,
    #[serde(default)]
    pub task: String,
    pub code: String,
}

/// Path for the Rust-body learned store (no Mog transpile required).
/// Prefers `NSYNTH_RUST_LEARNED`; else `.nsynth/rust_learned.jsonl`.
pub fn rust_learned_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("NSYNTH_RUST_LEARNED") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    // Default on when harvest flywheel is active, else still allow cwd default
    // only if the file already exists (don't create spuriously).
    let default = PathBuf::from(".nsynth/rust_learned.jsonl");
    if default.is_file() || std::env::var_os("NSYNTH_HARVEST").is_some() {
        Some(default)
    } else {
        None
    }
}

/// Append a verified Rust entry-fn body to the parallel learned store.
/// Best-effort; no-op when no path resolves. Strips test modules via
/// [`extract_entry_fn_body`].
pub fn append_rust_learned(name: &str, task: &str, code: &str) {
    let Some(path) = rust_learned_path() else {
        return;
    };
    if name.is_empty() || code.trim().is_empty() {
        return;
    }
    let body = if code.contains("fn ") || code.contains("pub fn ") {
        extract_entry_fn_body(code)
    } else {
        code.trim().to_string()
    };
    if body.is_empty() {
        return;
    }
    let row = RustLearnedRow {
        name: name.to_string(),
        task: task.chars().take(200).collect(),
        code: body,
    };
    let Ok(line) = serde_json::to_string(&row) else {
        return;
    };
    use std::io::Write;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Load all Rust learned rows (skips blank / malformed lines).
pub fn load_rust_learned(path: &Path) -> Vec<RustLearnedRow> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<RustLearnedRow>(line) {
            if !r.code.trim().is_empty() {
                rows.push(r);
            }
        }
    }
    rows
}

/// Most recent learned Rust body for `name` (case-insensitive), if any.
pub fn find_rust_learned_by_name(name: &str) -> Option<String> {
    let path = rust_learned_path()?;
    if !path.is_file() {
        return None;
    }
    let want = name.to_ascii_lowercase();
    load_rust_learned(&path)
        .into_iter()
        .rev()
        .find(|r| r.name.to_ascii_lowercase() == want)
        .map(|r| r.code)
}

/// Write a learned entry-fn body into `root/src/lib.rs`, preserving any
/// `#[cfg(test)]` module that follows. Best-effort recall before hole-fill.
pub fn apply_rust_learned_to_lib(root: &Path, body: &str) -> Result<(), String> {
    let lib_path = root.join("src/lib.rs");
    let existing = std::fs::read_to_string(&lib_path).unwrap_or_default();
    let tests = existing
        .find("#[cfg(test)]")
        .map(|i| existing[i..].to_string())
        .unwrap_or_default();
    let mut out = body.trim().to_string();
    out.push('\n');
    if !tests.is_empty() {
        out.push('\n');
        out.push_str(&tests);
    }
    if let Some(parent) = lib_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&lib_path, out).map_err(|e| format!("write {}: {e}", lib_path.display()))
}

/// Path where auto-mined / CLI-mined templates are written and later loaded.
/// Prefers `NSYNTH_MINED_TEMPLATES`; else `.nsynth/mined_templates.json` (cwd).
pub fn mined_templates_out_path() -> PathBuf {
    if let Ok(path) = std::env::var("NSYNTH_MINED_TEMPLATES") {
        if !path.is_empty() {
            return PathBuf::from(path);
        }
    }
    PathBuf::from(".nsynth/mined_templates.json")
}

fn auto_mine_enabled() -> bool {
    match std::env::var("NSYNTH_AUTO_MINE") {
        Ok(v)
            if v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no") =>
        {
            false
        }
        _ => true,
    }
}

/// Re-cluster a harvest JSONL into mined templates and write them.
/// Returns how many templates were written (`0` if fewer than 2 harvest rows).
pub fn refresh_templates_from_harvest(
    harvest_path: &Path,
    out_path: &Path,
    top_k: usize,
) -> Result<usize, String> {
    let rows = load_harvest_jsonl(harvest_path)?;
    if rows.len() < 2 {
        return Ok(0);
    }
    let templates = cluster_templates(&rows, top_k);
    write_templates_json(out_path, &templates)?;
    Ok(templates.len())
}

/// Best-effort auto-mine after a harvest append. Never panics / never fails caller.
fn maybe_refresh_templates_after_harvest(harvest_path: &str) {
    if !auto_mine_enabled() {
        return;
    }
    let path = Path::new(harvest_path);
    let Ok(rows) = load_harvest_jsonl(path) else {
        return;
    };
    let n = rows.len();
    if n < 2 {
        return;
    }
    let every: usize = std::env::var("NSYNTH_AUTO_MINE_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    // Always refresh at the first useful size (2), then every N rows.
    if every > 1 && n != 2 && n % every != 0 {
        return;
    }
    let top_k: usize = std::env::var("NSYNTH_AUTO_MINE_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
        .max(1);
    let out = mined_templates_out_path();
    let _ = refresh_templates_from_harvest(path, &out, top_k);
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
            t.cluster_key
                .chars()
                .take(200)
                .collect::<String>(),
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
/// passes examples **and** [`crate::runtime::verify_problem_code_strict`]
/// (holdouts / generated probes). Templates come from `NSYNTH_MINED_TEMPLATES`
/// or the provided slice. Bounded: at most `max_templates` × small const product.
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
            // Older templates may have stored a statement-sequence key with ` | `;
            // restore newlines. Fresh templates keep real newlines from normalize.
            let code = code.replace(" | ", "\n");
            if mined_template_accepts(problem, &code) {
                return Some(code);
            }
        }
        // Also try the raw example_code with fn renamed (exact shape reuse).
        if !t.example_code.is_empty() {
            let renamed = rename_fn_roughly(&t.example_code, name);
            if mined_template_accepts(problem, &renamed) {
                return Some(renamed);
            }
        }
    }
    None
}

/// Never-wrong gate for a mined-template candidate: examples + declared holdouts
/// must reproduce, then the shared strict oracle must accept.
fn mined_template_accepts(problem: &crate::benchmark::Problem, code: &str) -> bool {
    if !crate::runtime::code_reproduces_examples(code, &problem.examples) {
        return false;
    }
    if !problem.holdouts.is_empty()
        && !crate::runtime::code_reproduces_examples(code, &problem.holdouts)
    {
        return false;
    }
    crate::runtime::verify_problem_code_strict(problem, code).is_ok()
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
        let shared = top.iter().find(|t| t.count >= 2).expect("shared");
        assert!(
            shared.normalized.contains("?c"),
            "normalized must be hole-body not cluster key: {}",
            shared.normalized
        );
        assert!(
            !shared.cluster_key.is_empty(),
            "cluster_key should be set"
        );
    }

    #[test]
    fn extract_entry_fn_strips_test_module() {
        let lib = r#"
pub fn double(a0: i64) -> i64 { a0 * 2 }

#[cfg(test)]
mod characterization {
    use super::*;
    #[test]
    fn char_0() { assert_eq!(double(2), 4); }
}
"#;
        let body = extract_entry_fn_body(lib);
        assert!(body.contains("fn double"), "got {body}");
        assert!(body.contains("a0 * 2"), "got {body}");
        assert!(!body.contains("characterization"), "got {body}");
        assert!(!body.contains("assert_eq"), "got {body}");
    }

    #[test]
    fn cluster_then_instantiate_fills_const() {
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
        ];
        let top = cluster_templates(&rows, 5);
        let shared = top.iter().find(|t| t.count >= 2).expect("shared");
        let code = instantiate_template(&shared.normalized, "quadruple", &[4]);
        assert!(code.contains("fn quadruple"), "got {code}");
        assert!(code.contains("* 4"), "got {code}");
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
        let _guard = harvest_env_lock();
        let path = std::env::temp_dir().join(format!("nsynth_harvest_{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("NSYNTH_HARVEST", &path);
        std::env::set_var("NSYNTH_AUTO_MINE", "0"); // isolate: no template side-effect
        append_harvest_row("double", "fn double(x: i64) -> i64 { return x * 2; }");
        append_harvest_row("triple", "fn triple(x: i64) -> i64 { return x * 3; }");
        let text = std::fs::read_to_string(&path).expect("harvest file");
        assert_eq!(text.lines().count(), 2);
        assert!(text.contains("\"task\":\"double\""));
        std::env::remove_var("NSYNTH_HARVEST");
        std::env::remove_var("NSYNTH_AUTO_MINE");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_harvest_auto_mines_templates() {
        let _guard = harvest_env_lock();
        let id = std::process::id();
        let harvest = std::env::temp_dir().join(format!("nsynth_auto_harvest_{id}.jsonl"));
        let templates = std::env::temp_dir().join(format!("nsynth_auto_templates_{id}.json"));
        let _ = std::fs::remove_file(&harvest);
        let _ = std::fs::remove_file(&templates);
        std::env::set_var("NSYNTH_HARVEST", &harvest);
        std::env::set_var("NSYNTH_MINED_TEMPLATES", &templates);
        std::env::set_var("NSYNTH_AUTO_MINE", "1");
        std::env::remove_var("NSYNTH_AUTO_MINE_EVERY");

        append_harvest_row("double", "fn double(x: i64) -> i64 { return x * 2; }");
        assert!(
            !templates.is_file(),
            "single row should not yet write templates"
        );
        append_harvest_row("triple", "fn triple(y: i64) -> i64 { return y * 3; }");
        assert!(templates.is_file(), "≥2 rows should auto-mine templates");
        let loaded = load_templates_json(&templates).expect("load templates");
        assert!(!loaded.is_empty());
        let shared = loaded.iter().find(|t| t.count >= 2).expect("shared shape");
        assert!(
            shared.normalized.contains("?c"),
            "auto-mined body must keep const holes: {}",
            shared.normalized
        );

        // Direct API: refresh is idempotent and returns template count.
        let n = refresh_templates_from_harvest(&harvest, &templates, 10).expect("refresh");
        assert!(n >= 1);

        std::env::remove_var("NSYNTH_HARVEST");
        std::env::remove_var("NSYNTH_MINED_TEMPLATES");
        std::env::remove_var("NSYNTH_AUTO_MINE");
        let _ = std::fs::remove_file(&harvest);
        let _ = std::fs::remove_file(&templates);
    }

    #[test]
    fn rust_learned_store_round_trip() {
        let _guard = harvest_env_lock();
        let id = std::process::id();
        let path = std::env::temp_dir().join(format!("nsynth_rust_learned_{id}.jsonl"));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("NSYNTH_RUST_LEARNED", &path);
        append_rust_learned(
            "double",
            "double a number",
            "pub fn double(a0: i64) -> i64 { a0 * 2 }\n\n#[cfg(test)]\nmod t { }",
        );
        append_rust_learned("triple", "triple", "fn triple(x: i64) -> i64 { x * 3 }");
        let rows = load_rust_learned(&path);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].code.contains("fn double"), "{}", rows[0].code);
        assert!(!rows[0].code.contains("cfg(test)"));
        assert_eq!(
            find_rust_learned_by_name("DOUBLE").as_deref().map(|s| s.contains("a0 * 2")),
            Some(true)
        );
        let tmp_crate = std::env::temp_dir().join(format!("nsynth_rust_apply_{id}"));
        let _ = std::fs::remove_dir_all(&tmp_crate);
        std::fs::create_dir_all(tmp_crate.join("src")).unwrap();
        std::fs::write(
            tmp_crate.join("src/lib.rs"),
            "fn double(x: i64) -> i64 { 0 }\n\n#[cfg(test)]\nmod t { fn x() {} }\n",
        )
        .unwrap();
        let body = find_rust_learned_by_name("double").unwrap();
        apply_rust_learned_to_lib(&tmp_crate, &body).unwrap();
        let lib = std::fs::read_to_string(tmp_crate.join("src/lib.rs")).unwrap();
        assert!(lib.contains("a0 * 2"), "{lib}");
        assert!(lib.contains("#[cfg(test)]"), "{lib}");
        std::env::remove_var("NSYNTH_RUST_LEARNED");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&tmp_crate);
    }

    fn harvest_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }
}
