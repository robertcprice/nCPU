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
