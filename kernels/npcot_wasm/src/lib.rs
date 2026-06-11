//! NPCoT WASM / embedded runtime (N2b).
//!
//! Minimal zero-dependency pure-Rust executor for `DiscreteArrayProgram`
//! libraries. Compiles to:
//!
//! * `cargo build --release` — native rlib/cdylib/staticlib.
//! * `cargo build --release --target wasm32-unknown-unknown --features wasm`
//!   — browser-ready WASM with `wasm-bindgen` JS interop.
//! * `cargo build --release --target thumbv7em-none-eabihf` — ARM Cortex-M
//!   (bare metal, no std needed beyond `core` + `alloc` — future work).
//!
//! The code is deliberately NOT shared with `ncpu_metal::npcot_exec` (which
//! carries heavy pyo3 + objc2 deps) — we duplicate the pure-Rust portion
//! so this crate stays at zero dependencies. Keeping the two in sync is
//! the responsibility of the integration-level tests in the Python repo,
//! which feed the same library JSON through both paths and compare
//! outputs.

#![allow(clippy::needless_range_loop)]

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Canonical semantics — MUST match ncpu_metal::npcot_exec exactly.
// ---------------------------------------------------------------------------

pub const NEG_LARGE: f32 = -20.0;
pub const LOG_EPS: f32 = 1e-6;
pub const PROGRAM_STRIDE: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct DiscreteProgram {
    pub init_idx: u32,
    pub transform_idx: u32,
    pub reduce_idx: u32,
    pub post_scale_idx: u32,
    pub offset: f32,
}

#[inline]
fn init_value(init_idx: u32) -> f32 {
    match init_idx {
        0 => 0.0,
        1 => 1.0,
        2 => NEG_LARGE,
        _ => 0.0,
    }
}

#[inline]
fn apply_transform(x: f32, idx: u32) -> f32 {
    match idx {
        0 => x,
        1 => x * x,
        2 => x.abs(),
        3 => 1.0,
        4 => if x > 0.0 { 1.0 } else { 0.0 },
        5 => (x.abs() + LOG_EPS).ln(),
        _ => x,
    }
}

#[inline]
fn apply_reduce(acc: f32, f: f32, idx: u32) -> f32 {
    match idx {
        0 => acc + f,
        1 => acc * f,
        2 => acc.max(f),
        3 => acc.min(f),
        _ => acc + f,
    }
}

pub fn execute_program(program: DiscreteProgram, array: &[f32], length: u32) -> f32 {
    let mut acc = init_value(program.init_idx);
    let effective_len = (length as usize).min(array.len());
    for i in 0..effective_len {
        let f_i = apply_transform(array[i], program.transform_idx);
        acc = apply_reduce(acc, f_i, program.reduce_idx);
    }
    let post = match program.post_scale_idx {
        0 => acc,
        1 => {
            let denom = (length as f32).max(1.0);
            acc / denom
        }
        _ => {
            let clamped = acc.max(-30.0).min(30.0);
            clamped.exp()
        }
    };
    post + program.offset
}

// ---------------------------------------------------------------------------
// Minimal lookup index — linear scan. Libraries shipped to browsers / edge
// devices are small (tens to low hundreds of entries); the sharded index
// from the heavy crate is overkill at that scale.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NativeEntry {
    pub signature: Vec<f32>,
    pub program: DiscreteProgram,
}

pub struct NativeIndex {
    pub entries: Vec<NativeEntry>,
    pub similarity_threshold: f32,
}

impl NativeIndex {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            similarity_threshold,
        }
    }
    pub fn insert(&mut self, entry: NativeEntry) {
        self.entries.push(entry);
    }
    pub fn lookup(&self, signature: &[f32]) -> Option<&NativeEntry> {
        if signature.is_empty() {
            return None;
        }
        let mut best: Option<&NativeEntry> = None;
        let mut best_score = -1.0f32;
        for entry in &self.entries {
            let score = cosine_similarity(signature, &entry.signature);
            if score > best_score {
                best_score = score;
                best = Some(entry);
            }
        }
        if best_score >= self.similarity_threshold {
            best
        } else {
            None
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return -1.0;
    }
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        a_norm += a[i] * a[i];
        b_norm += b[i] * b[i];
    }
    let denom = a_norm.sqrt() * b_norm.sqrt();
    if denom < 1e-9 {
        -1.0
    } else {
        dot / denom
    }
}

pub fn consult_native(
    index: &NativeIndex,
    hidden: &[f32],
    array: &[f32],
    length: u32,
) -> Option<f32> {
    let mut norm_sq = 0.0f32;
    for v in hidden {
        norm_sq += v * v;
    }
    let norm = norm_sq.sqrt();
    if norm < 1e-8 {
        return None;
    }
    let normalized: Vec<f32> = hidden.iter().map(|v| v / norm).collect();
    let entry = index.lookup(&normalized)?;
    Some(execute_program(entry.program, array, length))
}

// ---------------------------------------------------------------------------
// Minimal JSON loader — same hand-rolled parser as the heavy crate.
// ---------------------------------------------------------------------------

pub fn load_library_json(payload: &str) -> Result<(f32, NativeIndex), String> {
    let similarity_threshold = extract_similarity_threshold(payload)?;
    let entries = extract_entries(payload)?;
    let mut index = NativeIndex::new(similarity_threshold);
    for entry in entries {
        index.insert(entry);
    }
    Ok((similarity_threshold, index))
}

fn extract_similarity_threshold(text: &str) -> Result<f32, String> {
    let marker = "\"similarity_threshold\"";
    let start = text.find(marker).ok_or_else(|| {
        "missing similarity_threshold".to_string()
    })?;
    let rest = &text[start + marker.len()..];
    let colon = rest.find(':').ok_or("malformed similarity_threshold")?;
    let after = &rest[colon + 1..];
    let end = after.find(|c: char| c == ',' || c == '}').unwrap_or(after.len());
    after[..end]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("parse similarity_threshold: {e}"))
}

fn extract_entries(text: &str) -> Result<Vec<NativeEntry>, String> {
    let marker = "\"entries\"";
    let start = text.find(marker).ok_or("missing entries")?;
    let rest = &text[start + marker.len()..];
    let lb = rest.find('[').ok_or("entries not an array")?;
    let arr_rest = &rest[lb..];
    let rb = find_matching(arr_rest, '[', ']').ok_or("entries unclosed")?;
    let body = &arr_rest[1..rb];
    let mut entries = Vec::new();
    let mut cursor = 0usize;
    while let Some(obj_start) = body[cursor..].find('{') {
        let absolute = cursor + obj_start;
        let obj_rest = &body[absolute..];
        let obj_end = find_matching(obj_rest, '{', '}').ok_or("entry no close")?;
        let obj = &obj_rest[..=obj_end];
        entries.push(parse_entry(obj)?);
        cursor = absolute + obj_end + 1;
    }
    Ok(entries)
}

fn find_matching(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in text.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn parse_entry(obj: &str) -> Result<NativeEntry, String> {
    let signature = parse_float_array(obj, "\"signature\"")?;
    let program_start = find_object_value(obj, "\"program\"").ok_or("no program")?;
    let program = DiscreteProgram {
        init_idx: parse_int_field(program_start, "\"init_idx\"")? as u32,
        transform_idx: parse_int_field(program_start, "\"transform_idx\"")? as u32,
        reduce_idx: parse_int_field(program_start, "\"reduce_idx\"")? as u32,
        post_scale_idx: parse_int_field(program_start, "\"post_scale_idx\"")? as u32,
        offset: parse_float_field(program_start, "\"offset\"")?,
    };
    Ok(NativeEntry { signature, program })
}

fn find_object_value<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let start = obj.find(key)?;
    let after = &obj[start + key.len()..];
    let colon = after.find(':')?;
    let rest = &after[colon + 1..];
    let trimmed = rest.trim_start();
    let open_idx = rest.len() - trimmed.len();
    let brace = trimmed.find('{')?;
    let abs_brace = colon + 1 + open_idx + brace;
    let tail = &after[abs_brace..];
    let close = find_matching(tail, '{', '}')?;
    Some(&after[abs_brace..abs_brace + close + 1])
}

fn parse_float_array(obj: &str, key: &str) -> Result<Vec<f32>, String> {
    let start = obj.find(key).ok_or_else(|| format!("missing {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after.find(':').ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let lb = rest.find('[').ok_or_else(|| format!("{key} not array"))?;
    let arr = &rest[lb..];
    let rb = find_matching(arr, '[', ']').ok_or_else(|| format!("{key} unclosed"))?;
    let body = &arr[1..rb];
    let mut out = Vec::new();
    for item in body.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.parse::<f32>().map_err(|e| format!("{key} float: {e}"))?);
    }
    Ok(out)
}

fn parse_int_field(obj: &str, key: &str) -> Result<i64, String> {
    let start = obj.find(key).ok_or_else(|| format!("missing {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after.find(':').ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let end = rest.find(|c: char| c == ',' || c == '}' || c == ']').unwrap_or(rest.len());
    rest[..end].trim().parse::<i64>().map_err(|e| format!("{key} int: {e}"))
}

fn parse_float_field(obj: &str, key: &str) -> Result<f32, String> {
    let start = obj.find(key).ok_or_else(|| format!("missing {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after.find(':').ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let end = rest.find(|c: char| c == ',' || c == '}' || c == ']').unwrap_or(rest.len());
    rest[..end].trim().parse::<f32>().map_err(|e| format!("{key} float: {e}"))
}

// ---------------------------------------------------------------------------
// On-device program synthesis — exhaustive search over the canonical program
// space (3 inits × 6 transforms × 4 reduces × 3 post-scales = 216 discrete
// programs, each with a closed-form fitted offset). This composes the
// executor above without touching its semantics, so synthesized programs
// stay byte-compatible with ncpu_metal::npcot_exec and every other runtime.
//
// This is what lets a deployed library GROW in the field: a miss can be
// turned into a new verified entry from I/O examples alone — no Python, no
// GPU, no network.
// ---------------------------------------------------------------------------

pub const N_INITS: u32 = 3;
pub const N_TRANSFORMS: u32 = 6;
pub const N_REDUCES: u32 = 4;
pub const N_POST_SCALES: u32 = 3;
pub const SEARCH_SPACE_SIZE: u32 = N_INITS * N_TRANSFORMS * N_REDUCES * N_POST_SCALES;

#[derive(Debug, Clone, Copy)]
pub struct SynthesisResult {
    pub program: DiscreteProgram,
    /// Worst absolute error across the training examples after offset fitting.
    pub max_err: f32,
    /// How many of the 216 programs were consistent with all examples.
    pub n_consistent: u32,
    /// Total programs evaluated.
    pub n_searched: u32,
}

/// Structural complexity used to break ties among consistent programs —
/// prefer the simplest faithful explanation of the examples.
fn program_complexity(p: &DiscreteProgram) -> u32 {
    let mut c = 0;
    if p.init_idx != 0 {
        c += 1;
    }
    if p.transform_idx != 0 {
        c += 1;
    }
    if p.reduce_idx != 0 {
        c += 1;
    }
    if p.post_scale_idx != 0 {
        c += 2;
    }
    if p.offset != 0.0 {
        c += 2;
    }
    c
}

/// Exhaustively search the canonical program space for a program consistent
/// with every (array, target) example. The offset is fitted in closed form
/// (mean residual). Returns `None` when no program in the space explains the
/// examples within `tol` — the synthesizer refuses rather than approximates.
pub fn synthesize_program(examples: &[(&[f32], f32)], tol: f32) -> Option<SynthesisResult> {
    if examples.is_empty() {
        return None;
    }
    for (arr, target) in examples {
        if !target.is_finite() || arr.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }
    let target_scale = examples
        .iter()
        .map(|(_, t)| t.abs())
        .fold(1.0f32, f32::max);
    let accept = tol * target_scale;

    let mut best: Option<(DiscreteProgram, f32, u32)> = None; // (program, max_err, complexity)
    let mut n_consistent = 0u32;
    let mut n_searched = 0u32;

    for init_idx in 0..N_INITS {
        for transform_idx in 0..N_TRANSFORMS {
            for reduce_idx in 0..N_REDUCES {
                for post_scale_idx in 0..N_POST_SCALES {
                    n_searched += 1;
                    let base = DiscreteProgram {
                        init_idx,
                        transform_idx,
                        reduce_idx,
                        post_scale_idx,
                        offset: 0.0,
                    };
                    // Closed-form offset: mean residual against raw outputs.
                    let mut residual_sum = 0.0f32;
                    let mut raw_ok = true;
                    let mut raw_outputs: Vec<f32> = Vec::with_capacity(examples.len());
                    for (arr, target) in examples {
                        let raw = execute_program(base, arr, arr.len() as u32);
                        if !raw.is_finite() {
                            raw_ok = false;
                            break;
                        }
                        raw_outputs.push(raw);
                        residual_sum += target - raw;
                    }
                    if !raw_ok {
                        continue;
                    }
                    let mut offset = residual_sum / examples.len() as f32;
                    // Snap near-zero offsets so exact programs stay exact.
                    if offset.abs() < accept.max(1e-6) {
                        offset = 0.0;
                    }
                    let mut max_err = 0.0f32;
                    for (i, (_, target)) in examples.iter().enumerate() {
                        let err = (raw_outputs[i] + offset - target).abs();
                        max_err = max_err.max(err);
                    }
                    if max_err <= accept {
                        n_consistent += 1;
                        let candidate = DiscreteProgram { offset, ..base };
                        let complexity = program_complexity(&candidate);
                        let better = match &best {
                            None => true,
                            Some((_, best_err, best_cx)) => {
                                complexity < *best_cx
                                    || (complexity == *best_cx && max_err < *best_err)
                            }
                        };
                        if better {
                            best = Some((candidate, max_err, complexity));
                        }
                    }
                }
            }
        }
    }

    best.map(|(program, max_err, _)| SynthesisResult {
        program,
        max_err,
        n_consistent,
        n_searched,
    })
}

/// Render a program as readable Rust-style source (mirrors executor semantics).
pub fn program_source(name: &str, p: &DiscreteProgram) -> String {
    let init = match p.init_idx {
        1 => "1.0".to_string(),
        2 => "NEG_LARGE".to_string(),
        _ => "0.0".to_string(),
    };
    let transform = match p.transform_idx {
        1 => "x * x",
        2 => "x.abs()",
        3 => "1.0",
        4 => "if x > 0.0 { 1.0 } else { 0.0 }",
        5 => "(x.abs() + LOG_EPS).ln()",
        _ => "x",
    };
    let body = match p.reduce_idx {
        1 => format!("acc *= {transform};"),
        2 => format!("acc = acc.max({transform});"),
        3 => format!("acc = acc.min({transform});"),
        _ => format!("acc += {transform};"),
    };
    let post = match p.post_scale_idx {
        1 => "\n    acc /= arr.len().max(1) as f32;",
        2 => "\n    acc = acc.clamp(-30.0, 30.0).exp();",
        _ => "",
    };
    let offset = if p.offset != 0.0 {
        format!(" + {:.6}", p.offset)
    } else {
        String::new()
    };
    format!(
        "fn {name}(arr: &[f32]) -> f32 {{\n    let mut acc: f32 = {init};\n    for &x in arr {{\n        {body}\n    }}{post}\n    acc{offset}\n}}"
    )
}

/// Serialize an index back to the canonical library JSON format so a library
/// grown on-device can be exported, signed, and shipped to other runtimes.
pub fn library_to_json(index: &NativeIndex) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{{\n  \"config\": {{\"similarity_threshold\": {}, \"max_entries\": {}, \"normalize_epsilon\": 1e-08}},\n  \"entries\": [\n",
        index.similarity_threshold,
        index.entries.len().max(16)
    ));
    for (i, e) in index.entries.iter().enumerate() {
        let sig: Vec<String> = e.signature.iter().map(|v| format!("{v}")).collect();
        out.push_str(&format!(
            "    {{\"signature\": [{}], \"program\": {{\"init_idx\": {}, \"transform_idx\": {}, \"reduce_idx\": {}, \"post_scale_idx\": {}, \"offset\": {}}}, \"hit_count\": 0, \"task_name\": \"entry_{i}\", \"cached_at_step\": null, \"convergence_gap\": null}}{}\n",
            sig.join(", "),
            e.program.init_idx,
            e.program.transform_idx,
            e.program.reduce_idx,
            e.program.post_scale_idx,
            e.program.offset,
            if i + 1 == index.entries.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n}");
    out
}

// ---------------------------------------------------------------------------
// WASM-specific JS exports (only when built with --features wasm).
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct NpcotRuntime {
    index: NativeIndex,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl NpcotRuntime {
    #[wasm_bindgen(constructor)]
    pub fn new(library_json: &str) -> Result<NpcotRuntime, JsValue> {
        let (_thr, index) = load_library_json(library_json)
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(NpcotRuntime { index })
    }

    pub fn entry_count(&self) -> usize {
        self.index.entries.len()
    }

    pub fn consult(&self, hidden: Vec<f32>, array: Vec<f32>, length: u32) -> Option<f32> {
        consult_native(&self.index, &hidden, &array, length)
    }

    /// Synthesize a program from flattened I/O examples. `arrays` holds all
    /// example arrays back to back, `lens[i]` gives each example's length,
    /// `targets[i]` its expected output. Returns a JSON description of the
    /// discovered program (with search stats and rendered source), or `None`
    /// when nothing in the space fits — refusal, not approximation.
    pub fn synthesize(
        &self,
        arrays: Vec<f32>,
        lens: Vec<u32>,
        targets: Vec<f32>,
        name: String,
    ) -> Option<String> {
        if lens.len() != targets.len() || lens.is_empty() {
            return None;
        }
        let total: usize = lens.iter().map(|l| *l as usize).sum();
        if total != arrays.len() {
            return None;
        }
        let mut examples: Vec<(&[f32], f32)> = Vec::with_capacity(lens.len());
        let mut cursor = 0usize;
        for (i, len) in lens.iter().enumerate() {
            let end = cursor + *len as usize;
            examples.push((&arrays[cursor..end], targets[i]));
            cursor = end;
        }
        let result = synthesize_program(&examples, 1e-3)?;
        let p = result.program;
        let source = program_source(&name, &p).replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        Some(format!(
            "{{\"init_idx\": {}, \"transform_idx\": {}, \"reduce_idx\": {}, \"post_scale_idx\": {}, \"offset\": {}, \"max_err\": {}, \"n_consistent\": {}, \"n_searched\": {}, \"source\": \"{}\"}}",
            p.init_idx, p.transform_idx, p.reduce_idx, p.post_scale_idx, p.offset,
            result.max_err, result.n_consistent, result.n_searched, source
        ))
    }

    /// Insert a synthesized program into the live library under `signature`.
    pub fn insert_skill(
        &mut self,
        signature: Vec<f32>,
        init_idx: u32,
        transform_idx: u32,
        reduce_idx: u32,
        post_scale_idx: u32,
        offset: f32,
    ) {
        self.index.insert(NativeEntry {
            signature,
            program: DiscreteProgram {
                init_idx,
                transform_idx,
                reduce_idx,
                post_scale_idx,
                offset,
            },
        });
    }

    /// Export the current library (including skills learned this session) as
    /// canonical library JSON, loadable by every NPCoT runtime.
    pub fn export_library(&self) -> String {
        library_to_json(&self.index)
    }
}

// ---------------------------------------------------------------------------
// Native tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_sum() {
        let program = DiscreteProgram {
            init_idx: 0,
            transform_idx: 0,
            reduce_idx: 0,
            post_scale_idx: 0,
            offset: 0.0,
        };
        let array = [1.0, 2.0, 3.0, 0.0, 0.0];
        assert_eq!(execute_program(program, &array, 3), 6.0);
    }

    #[test]
    fn execute_log_product() {
        // log|x| + reduce=+ + post_scale=exp → abs(product)
        let program = DiscreteProgram {
            init_idx: 0,
            transform_idx: 5,
            reduce_idx: 0,
            post_scale_idx: 2,
            offset: 0.0,
        };
        let array = [2.0, -3.0, 4.0];
        let result = execute_program(program, &array, 3);
        assert!((result - 24.0).abs() < 0.01, "got {result}");
    }

    #[test]
    fn consult_e2e() {
        let library_json = r##"{
  "config": {"similarity_threshold": 0.85, "max_entries": 16, "normalize_epsilon": 1e-08},
  "entries": [
    {
      "signature": [1.0, 0.0, 0.0],
      "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0, "post_scale_idx": 0, "offset": 0.0, "program_text": "sum"},
      "hit_count": 0,
      "task_name": "sum",
      "cached_at_step": null,
      "convergence_gap": null
    }
  ]
}"##;
        let (_thr, index) = load_library_json(library_json).expect("parse");
        let hidden = vec![1.0, 0.0, 0.0];
        let array = vec![1.0, 2.0, 3.0, 4.0];
        let r = consult_native(&index, &hidden, &array, 4).expect("hit");
        assert!((r - 10.0).abs() < 1e-5);
    }

    #[test]
    fn synthesize_discovers_sum() {
        let a1 = [1.0, 2.0, 3.0];
        let a2 = [10.0, -4.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 6.0), (&a2, 6.0)];
        let r = synthesize_program(&examples, 1e-3).expect("sum should synthesize");
        let p = r.program;
        assert_eq!(
            (p.init_idx, p.transform_idx, p.reduce_idx, p.post_scale_idx),
            (0, 0, 0, 0)
        );
        assert_eq!(p.offset, 0.0);
        assert_eq!(r.n_searched, SEARCH_SPACE_SIZE);
    }

    #[test]
    fn synthesize_discovers_mean() {
        let a1 = [2.0, 4.0, 6.0];
        let a2 = [10.0, 20.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 4.0), (&a2, 15.0)];
        let r = synthesize_program(&examples, 1e-3).expect("mean should synthesize");
        assert_eq!(r.program.post_scale_idx, 1);
        assert_eq!(r.program.reduce_idx, 0);
    }

    #[test]
    fn synthesize_discovers_sum_of_squares() {
        let a1 = [1.0, 2.0, 3.0];
        let a2 = [0.0, 4.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 14.0), (&a2, 16.0)];
        let r = synthesize_program(&examples, 1e-3).expect("sum of squares");
        assert_eq!(r.program.transform_idx, 1);
        assert_eq!(r.program.reduce_idx, 0);
    }

    #[test]
    fn synthesize_discovers_abs_product_via_log_domain() {
        let a1 = [2.0, -3.0];
        let a2 = [4.0, 0.5, 2.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 6.0), (&a2, 4.0)];
        let r = synthesize_program(&examples, 1e-2).expect("abs product");
        // Two encodings exist in the space (direct |x| product and the
        // log-domain exp(Σ ln|x|) trick); either is acceptable — judge by
        // behavior on a held-out example.
        let held_out = [3.0, -2.0, 0.5];
        let pred = execute_program(r.program, &held_out, 3);
        assert!((pred - 3.0).abs() < 1e-2, "held-out abs product: got {pred}");
    }

    #[test]
    fn synthesize_fits_offset() {
        // sum + 7
        let a1 = [1.0, 2.0];
        let a2 = [5.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 10.0), (&a2, 12.0)];
        let r = synthesize_program(&examples, 1e-3).expect("sum + offset");
        assert_eq!(r.program.reduce_idx, 0);
        assert!((r.program.offset - 7.0).abs() < 1e-4, "offset {}", r.program.offset);
    }

    #[test]
    fn synthesize_refuses_outside_space() {
        // Median is not expressible in the 216-program space; with these
        // examples no candidate (incl. fitted offsets) explains all three.
        let a1 = [1.0, 100.0, 2.0];
        let a2 = [50.0, 1.0, 3.0];
        let a3 = [9.0, 7.0, 1000.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 2.0), (&a2, 3.0), (&a3, 9.0)];
        assert!(synthesize_program(&examples, 1e-3).is_none());
    }

    #[test]
    fn synthesize_rejects_nan_input() {
        let a1 = [1.0, f32::NAN];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 1.0)];
        assert!(synthesize_program(&examples, 1e-3).is_none());
    }

    #[test]
    fn synthesize_prefers_simplest_program() {
        // Single all-positive example: many programs fit, plain sum must win
        // over offset-fitted or post-scaled variants.
        let a1 = [3.0, -1.0, 4.0];
        let examples: Vec<(&[f32], f32)> = vec![(&a1, 6.0)];
        let r = synthesize_program(&examples, 1e-3).expect("fit");
        assert_eq!(
            (r.program.transform_idx, r.program.reduce_idx, r.program.post_scale_idx),
            (0, 0, 0)
        );
        assert!(r.n_consistent > 1, "expected multiple candidates, got {}", r.n_consistent);
    }

    #[test]
    fn exported_library_round_trips() {
        let mut index = NativeIndex::new(0.85);
        index.insert(NativeEntry {
            signature: vec![1.0, 0.0, 0.0],
            program: DiscreteProgram {
                init_idx: 0,
                transform_idx: 1,
                reduce_idx: 0,
                post_scale_idx: 0,
                offset: 2.5,
            },
        });
        let json = library_to_json(&index);
        let (thr, parsed) = load_library_json(&json).expect("round trip");
        assert!((thr - 0.85).abs() < 1e-6);
        assert_eq!(parsed.entries.len(), 1);
        let e = &parsed.entries[0];
        assert_eq!(e.program.transform_idx, 1);
        assert!((e.program.offset - 2.5).abs() < 1e-6);
        // Round-tripped library still answers consults.
        let r = consult_native(&parsed, &[1.0, 0.0, 0.0], &[1.0, 2.0, 3.0], 3).expect("hit");
        assert!((r - (14.0 + 2.5)).abs() < 1e-4);
    }

    #[test]
    fn program_source_renders_sum() {
        let p = DiscreteProgram {
            init_idx: 0,
            transform_idx: 0,
            reduce_idx: 0,
            post_scale_idx: 0,
            offset: 0.0,
        };
        let src = program_source("sum", &p);
        assert!(src.contains("fn sum(arr: &[f32]) -> f32"));
        assert!(src.contains("acc += x;"));
    }

    #[test]
    fn consult_miss_returns_none() {
        let library_json = r##"{
  "config": {"similarity_threshold": 0.99, "max_entries": 16, "normalize_epsilon": 1e-08},
  "entries": [
    {
      "signature": [1.0, 0.0, 0.0],
      "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0, "post_scale_idx": 0, "offset": 0.0, "program_text": "sum"},
      "hit_count": 0,
      "task_name": "sum",
      "cached_at_step": null,
      "convergence_gap": null
    }
  ]
}"##;
        let (_thr, index) = load_library_json(library_json).expect("parse");
        assert!(consult_native(&index, &[0.0, 1.0, 0.0], &[1.0], 1).is_none());
    }
}
