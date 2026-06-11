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
// Format v2 — multi-field data points + predicate guards.
//
// A v2 data point is a record of `arity` floats laid out contiguously
// (x0,y0, x1,y1, ...). Execution adds two stages in front of the v1
// pipeline:
//
//   combine: record -> scalar   (field select / sum / product / diff / ...)
//   guard:   scalar -> include? (always / >t / <t / |v|>t / ==t)
//
// then transform/reduce/post/offset run exactly as v1. A v1 program is the
// special case arity=1, combine=field0, guard=always — `ProgramV2::from_v1`
// is exact, so v1 libraries execute identically under the v2 engine.
//
// v2 libraries serialize with `"format": 2` and a `program_v2` key. v1
// loaders fail closed on them (their parser requires a `program` object),
// so an old runtime can never silently mis-execute a guarded program.
// ---------------------------------------------------------------------------

pub const N_COMBINES: u32 = 8;
pub const N_GUARDS: u32 = 5;
pub const MAX_ARITY: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgramV2 {
    /// Fields per data point (1..=MAX_ARITY).
    pub arity: u32,
    /// 0=f0, 1=f1, 2=Σfields, 3=Πfields, 4=f0-f1, 5=|f0-f1|, 6=min, 7=max
    pub combine_idx: u32,
    /// 0=always, 1=v>t, 2=v<t, 3=|v|>t, 4=v==t (1e-4 tolerance)
    pub guard_idx: u32,
    pub guard_threshold: f32,
    pub init_idx: u32,
    pub transform_idx: u32,
    pub reduce_idx: u32,
    pub post_scale_idx: u32,
    pub offset: f32,
}

impl ProgramV2 {
    pub fn from_v1(p: DiscreteProgram) -> Self {
        ProgramV2 {
            arity: 1,
            combine_idx: 0,
            guard_idx: 0,
            guard_threshold: 0.0,
            init_idx: p.init_idx,
            transform_idx: p.transform_idx,
            reduce_idx: p.reduce_idx,
            post_scale_idx: p.post_scale_idx,
            offset: p.offset,
        }
    }

    /// True when expressible in the v1 format (so exports can stay
    /// backward-compatible whenever possible).
    pub fn is_v1(&self) -> bool {
        self.arity == 1 && self.combine_idx == 0 && self.guard_idx == 0
    }
}

#[inline]
fn apply_combine(fields: &[f32], idx: u32) -> f32 {
    match idx {
        1 => fields.get(1).copied().unwrap_or(0.0),
        2 => fields.iter().sum(),
        3 => fields.iter().product(),
        4 => fields.first().copied().unwrap_or(0.0) - fields.get(1).copied().unwrap_or(0.0),
        5 => (fields.first().copied().unwrap_or(0.0) - fields.get(1).copied().unwrap_or(0.0)).abs(),
        6 => fields.iter().copied().fold(f32::INFINITY, f32::min),
        7 => fields.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        _ => fields.first().copied().unwrap_or(0.0),
    }
}

#[inline]
fn guard_passes(v: f32, idx: u32, t: f32) -> bool {
    match idx {
        1 => v > t,
        2 => v < t,
        3 => v.abs() > t,
        4 => (v - t).abs() < 1e-4,
        _ => true,
    }
}

/// Execute a v2 program over `n_points` records of `arity` floats laid out
/// contiguously in `data`. The mean post-scale divides by the number of
/// guard-INCLUDED points (mean of what was aggregated), not raw length.
pub fn execute_program_v2(p: ProgramV2, data: &[f32], n_points: u32) -> f32 {
    let arity = p.arity.max(1) as usize;
    let usable = (n_points as usize).min(data.len() / arity);
    let mut acc = init_value(p.init_idx);
    let mut included = 0u32;
    for i in 0..usable {
        let fields = &data[i * arity..(i + 1) * arity];
        let v = apply_combine(fields, p.combine_idx);
        if !guard_passes(v, p.guard_idx, p.guard_threshold) {
            continue;
        }
        included += 1;
        acc = apply_reduce(acc, apply_transform(v, p.transform_idx), p.reduce_idx);
    }
    let post = match p.post_scale_idx {
        0 => acc,
        1 => acc / (included as f32).max(1.0),
        _ => acc.max(-30.0).min(30.0).exp(),
    };
    post + p.offset
}

// ---------------------------------------------------------------------------
// Format v3 — stateful skills: skill = (state, input) → (state', output).
//
// A v3 program is a v2-style per-step pipeline PLUS one persistent state
// cell `s: f32`. A v3 *example* is a TRACE: a sequence of (input record,
// expected output) steps; execution emits one output per step and state
// persists across steps (each trace restarts state). Per step:
//
//   v = combine(fields)                         (v2 combine vocabulary)
//   if include_guard passes (v2 guard vocab; synthesis keeps it `always`):
//       if reset_guard fires on v: s ← init, included ← 0   (full restart)
//       included += 1
//       s ← reduce(s, transform(v))             (v1 op vocabularies)
//   y = output_select(s, v)                     (small enumerable vocab)
//   y = post_scale(y)  + offset                 (v2 post vocab; synthesis
//                                                keeps it `identity`)
//
// The reset guard reuses the v2 guard comparison vocabulary with index 0
// meaning "never fire" (the stage's no-op, mirroring 0 = "always pass" for
// inclusion guards). Reset thresholds are MINED from trace data via
// `mine_thresholds` — no hardcoded vocabulary.
//
// `guard_idx`/`guard_threshold`/`post_scale_idx` exist so that EVERY v2
// program lifts exactly into v3 (`ProgramV3::from_v2` with reset=never,
// output=state reproduces `execute_program_v2` at the final step — v2 is
// the exact special case "state never resets, output is the fold"). The
// v3 SYNTHESIS space keeps both stages neutral; they are carried for
// entry-wise lifting when a mixed library exports as format 3.
//
// v3 libraries serialize with `"format": 3` and a `program_v3` key. v1/v2
// loaders fail closed on them (their parsers require a `program` /
// `program_v2` object key), so an old runtime can never mis-execute a
// stateful program as a stateless fold.
// ---------------------------------------------------------------------------

/// Output-select vocabulary size: 0=s, 1=v, 2=s+v, 3=s*v, 4=|s|.
pub const N_OUTPUTS_V3: u32 = 5;
/// Reset-guard vocabulary size (0=never, 1..=4 reuse v2 guard comparisons).
pub const N_RESET_GUARDS: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgramV3 {
    /// Fields per data point (1..=MAX_ARITY).
    pub arity: u32,
    /// v2 combine vocabulary (see `ProgramV2::combine_idx`).
    pub combine_idx: u32,
    /// Inclusion guard, exact v2 semantics (0=always). Kept for lossless
    /// v2→v3 lifting; the v3 synthesis space leaves this at 0.
    pub guard_idx: u32,
    pub guard_threshold: f32,
    /// Reset guard on the combined value `v`: 0=never, 1=v>t, 2=v<t,
    /// 3=|v|>t, 4=v==t (1e-4 tolerance). Fires BEFORE the state update.
    pub reset_guard_idx: u32,
    pub reset_threshold: f32,
    /// State initializer (v1 `init_value` vocabulary): 0.0 / 1.0 / NEG_LARGE.
    pub state_init_idx: u32,
    /// Transform applied to `v` before folding into state (v1 vocabulary).
    pub update_transform_idx: u32,
    /// Reduce folding the transformed value into state (v1 vocabulary).
    pub update_reduce_idx: u32,
    /// Post-scale on the selected output, exact v2 semantics (0=identity,
    /// 1=divide by included-so-far, 2=clamp-exp). Kept for lossless v2→v3
    /// lifting; the v3 synthesis space leaves this at 0.
    pub post_scale_idx: u32,
    /// Output select: 0=s, 1=v, 2=s+v, 3=s*v, 4=|s|.
    pub output_idx: u32,
    /// Closed-form fitted offset added to every step's output.
    pub offset: f32,
}

impl ProgramV3 {
    pub fn from_v2(p: ProgramV2) -> Self {
        ProgramV3 {
            arity: p.arity,
            combine_idx: p.combine_idx,
            guard_idx: p.guard_idx,
            guard_threshold: p.guard_threshold,
            reset_guard_idx: 0,
            reset_threshold: 0.0,
            state_init_idx: p.init_idx,
            update_transform_idx: p.transform_idx,
            update_reduce_idx: p.reduce_idx,
            post_scale_idx: p.post_scale_idx,
            output_idx: 0,
            offset: p.offset,
        }
    }

    pub fn from_v1(p: DiscreteProgram) -> Self {
        Self::from_v2(ProgramV2::from_v1(p))
    }

    /// True when expressible in the v2 format: state never resets and the
    /// output is the running fold, i.e. the final step's output equals the
    /// v2 aggregate. Lets exports stay at the lowest loadable format.
    pub fn is_v2(&self) -> bool {
        self.reset_guard_idx == 0 && self.output_idx == 0
    }

    pub fn to_v2(&self) -> Option<ProgramV2> {
        if !self.is_v2() {
            return None;
        }
        Some(ProgramV2 {
            arity: self.arity,
            combine_idx: self.combine_idx,
            guard_idx: self.guard_idx,
            guard_threshold: self.guard_threshold,
            init_idx: self.state_init_idx,
            transform_idx: self.update_transform_idx,
            reduce_idx: self.update_reduce_idx,
            post_scale_idx: self.post_scale_idx,
            offset: self.offset,
        })
    }
}

/// Reset guard: index 0 is "never"; 1..=4 reuse the v2 guard comparisons.
#[inline]
fn reset_fires(v: f32, idx: u32, t: f32) -> bool {
    idx != 0 && guard_passes(v, idx, t)
}

#[inline]
fn output_select_v3(p: &ProgramV3, s: f32, v: f32) -> f32 {
    match p.output_idx {
        1 => v,
        2 => s + v,
        3 => s * v,
        4 => s.abs(),
        _ => s,
    }
}

#[inline]
fn post_scale_v3(p: &ProgramV3, y: f32, included: u32) -> f32 {
    match p.post_scale_idx {
        0 => y,
        1 => y / (included as f32).max(1.0),
        _ => y.max(-30.0).min(30.0).exp(),
    }
}

/// Replay a v3 program over a trace of `n_steps` records of `arity` floats
/// laid out contiguously in `data`. Returns one output per step. State is
/// initialized at the start of the trace (each trace restarts state); a
/// reset-guard hit restores both the state cell and the included-counter
/// (a full restart of the running aggregate).
pub fn execute_program_v3(p: ProgramV3, data: &[f32], n_steps: u32) -> Vec<f32> {
    let arity = p.arity.max(1) as usize;
    let usable = (n_steps as usize).min(data.len() / arity);
    let mut s = init_value(p.state_init_idx);
    let mut included = 0u32;
    let mut outputs = Vec::with_capacity(usable);
    for i in 0..usable {
        let fields = &data[i * arity..(i + 1) * arity];
        let v = apply_combine(fields, p.combine_idx);
        if guard_passes(v, p.guard_idx, p.guard_threshold) {
            if reset_fires(v, p.reset_guard_idx, p.reset_threshold) {
                s = init_value(p.state_init_idx);
                included = 0;
            }
            included += 1;
            s = apply_reduce(s, apply_transform(v, p.update_transform_idx), p.update_reduce_idx);
        }
        let y = output_select_v3(&p, s, v);
        outputs.push(post_scale_v3(&p, y, included) + p.offset);
    }
    outputs
}

/// Final-step output of a v3 trace replay. For programs lifted from v2
/// (`ProgramV3::from_v2`) this equals `execute_program_v2` over the same
/// points exactly — including the empty trace, where it mirrors v2's
/// empty fold (state stays at its initializer, `v` is taken as 0).
pub fn execute_program_v3_final(p: ProgramV3, data: &[f32], n_steps: u32) -> f32 {
    match execute_program_v3(p, data, n_steps).last() {
        Some(&y) => y,
        None => {
            let s = init_value(p.state_init_idx);
            let y = output_select_v3(&p, s, 0.0);
            post_scale_v3(&p, y, 0) + p.offset
        }
    }
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
    /// Present when the entry uses v2 capabilities (records, guards). When
    /// `Some`, execution goes through the v2 engine; `program` is a
    /// placeholder kept for struct compatibility.
    pub program_v2: Option<ProgramV2>,
    /// Present when the entry uses v3 capabilities (persistent state,
    /// resets, non-fold outputs). When `Some`, consults go through the v3
    /// trace engine; `program`/`program_v2` are kept in sync for struct
    /// compatibility and lowest-format export.
    pub program_v3: Option<ProgramV3>,
}

impl NativeEntry {
    pub fn effective_program(&self) -> ProgramV2 {
        self.program_v2.unwrap_or_else(|| ProgramV2::from_v1(self.program))
    }

    /// The entry's program lifted to v3 (exact: lower formats are special
    /// cases of v3 — see `ProgramV3::from_v2`).
    pub fn effective_program_v3(&self) -> ProgramV3 {
        self.program_v3
            .unwrap_or_else(|| ProgramV3::from_v2(self.effective_program()))
    }
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
    // v3 entries replay the trace and answer with the final step's output
    // (for v2-lifted programs this is exactly the v2 aggregate). v1/v2
    // entries take the original v2-engine path, byte-identical to before.
    if let Some(p3) = entry.program_v3 {
        return Some(execute_program_v3_final(p3, array, length));
    }
    let p = entry.effective_program();
    // `length` counts data points; v1 entries have arity 1 so this is the
    // exact v1 semantics, and v2 entries interpret `array` as records.
    Some(execute_program_v2(p, array, length))
}

/// Consult path for stateful (v3) skills: look up by hidden-state
/// similarity, then replay the program over `n_steps` records of `arity`
/// floats, returning ALL per-step outputs. Works for v1/v2 entries too
/// (lifted exactly — their per-step outputs are the running fold). Returns
/// `None` on lookup miss, or when the entry's arity disagrees with the
/// caller's layout (honest refusal instead of misinterpreting records).
pub fn consult_native_v3(
    index: &NativeIndex,
    hidden: &[f32],
    data: &[f32],
    arity: u32,
    n_steps: u32,
) -> Option<Vec<f32>> {
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
    let p = entry.effective_program_v3();
    if p.arity.max(1) != arity.max(1) {
        return None;
    }
    Some(execute_program_v3(p, data, n_steps))
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
    // v3 entries carry a `program_v3` object; v2 entries `program_v2`; v1
    // entries `program`. Each loader generation requires its own key, so
    // older runtimes fail closed on newer files instead of mis-executing.
    if let Some(v3_obj) = find_object_value(obj, "\"program_v3\"") {
        let program_v3 = ProgramV3 {
            arity: (parse_int_field(v3_obj, "\"arity\"")? as u32).clamp(1, MAX_ARITY),
            combine_idx: parse_int_field(v3_obj, "\"combine_idx\"")? as u32,
            guard_idx: parse_int_field(v3_obj, "\"guard_idx\"")? as u32,
            guard_threshold: parse_float_field(v3_obj, "\"guard_threshold\"")?,
            reset_guard_idx: parse_int_field(v3_obj, "\"reset_guard_idx\"")? as u32,
            reset_threshold: parse_float_field(v3_obj, "\"reset_threshold\"")?,
            state_init_idx: parse_int_field(v3_obj, "\"state_init_idx\"")? as u32,
            update_transform_idx: parse_int_field(v3_obj, "\"update_transform_idx\"")? as u32,
            update_reduce_idx: parse_int_field(v3_obj, "\"update_reduce_idx\"")? as u32,
            post_scale_idx: parse_int_field(v3_obj, "\"post_scale_idx\"")? as u32,
            output_idx: parse_int_field(v3_obj, "\"output_idx\"")? as u32,
            offset: parse_float_field(v3_obj, "\"offset\"")?,
        };
        return Ok(NativeEntry {
            signature,
            program: DiscreteProgram {
                init_idx: program_v3.state_init_idx,
                transform_idx: program_v3.update_transform_idx,
                reduce_idx: program_v3.update_reduce_idx,
                post_scale_idx: program_v3.post_scale_idx,
                offset: program_v3.offset,
            },
            // Keep the v2 view in sync when expressible so a later export
            // can demote the entry back to the lowest loadable format.
            program_v2: program_v3.to_v2(),
            program_v3: Some(program_v3),
        });
    }
    if let Some(v2_obj) = find_object_value(obj, "\"program_v2\"") {
        let program_v2 = ProgramV2 {
            arity: (parse_int_field(v2_obj, "\"arity\"")? as u32).clamp(1, MAX_ARITY),
            combine_idx: parse_int_field(v2_obj, "\"combine_idx\"")? as u32,
            guard_idx: parse_int_field(v2_obj, "\"guard_idx\"")? as u32,
            guard_threshold: parse_float_field(v2_obj, "\"guard_threshold\"")?,
            init_idx: parse_int_field(v2_obj, "\"init_idx\"")? as u32,
            transform_idx: parse_int_field(v2_obj, "\"transform_idx\"")? as u32,
            reduce_idx: parse_int_field(v2_obj, "\"reduce_idx\"")? as u32,
            post_scale_idx: parse_int_field(v2_obj, "\"post_scale_idx\"")? as u32,
            offset: parse_float_field(v2_obj, "\"offset\"")?,
        };
        return Ok(NativeEntry {
            signature,
            program: DiscreteProgram {
                init_idx: program_v2.init_idx,
                transform_idx: program_v2.transform_idx,
                reduce_idx: program_v2.reduce_idx,
                post_scale_idx: program_v2.post_scale_idx,
                offset: program_v2.offset,
            },
            program_v2: Some(program_v2),
            program_v3: None,
        });
    }
    let program_start = find_object_value(obj, "\"program\"").ok_or("no program")?;
    let program = DiscreteProgram {
        init_idx: parse_int_field(program_start, "\"init_idx\"")? as u32,
        transform_idx: parse_int_field(program_start, "\"transform_idx\"")? as u32,
        reduce_idx: parse_int_field(program_start, "\"reduce_idx\"")? as u32,
        post_scale_idx: parse_int_field(program_start, "\"post_scale_idx\"")? as u32,
        offset: parse_float_field(program_start, "\"offset\"")?,
    };
    Ok(NativeEntry { signature, program, program_v2: None, program_v3: None })
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

/// Mine guard-threshold candidates from the examples themselves — the same
/// emergent-vocabulary idea as nsynth's `discover_useful_consts`: no
/// hardcoded magic numbers, the data proposes its own thresholds. Candidates
/// are every distinct field value plus 0, capped to keep the search bounded.
pub fn mine_thresholds(examples: &[(&[f32], f32)]) -> Vec<f32> {
    const CAP: usize = 12;
    let mut seen: Vec<f32> = vec![0.0];
    for (data, _) in examples {
        for &v in *data {
            if v.is_finite() && !seen.iter().any(|s| (s - v).abs() < 1e-6) {
                seen.push(v);
            }
        }
    }
    // Prefer small-magnitude thresholds (more likely structural).
    seen.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal));
    seen.truncate(CAP);
    seen
}

fn complexity_v2(p: &ProgramV2) -> u32 {
    let mut c = 0;
    if p.combine_idx != 0 {
        c += 1;
    }
    if p.guard_idx != 0 {
        c += 2;
    }
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

#[derive(Debug, Clone, Copy)]
pub struct SynthesisResultV2 {
    pub program: ProgramV2,
    pub max_err: f32,
    pub n_consistent: u32,
    pub n_searched: u32,
}

/// v2 synthesis: exhaustive search over combine × guard × threshold ×
/// transform × reduce × post × init with closed-form offset fitting.
/// `examples` hold flat records (`arity` floats per point). Same refusal
/// contract as v1: `None` when nothing in the space explains every example.
pub fn synthesize_program_v2(
    examples: &[(&[f32], f32)],
    arity: u32,
    tol: f32,
) -> Option<SynthesisResultV2> {
    if examples.is_empty() || arity == 0 || arity > MAX_ARITY {
        return None;
    }
    for (data, target) in examples {
        if !target.is_finite() || data.iter().any(|v| !v.is_finite()) {
            return None;
        }
        if data.len() % arity as usize != 0 {
            return None;
        }
    }
    let thresholds = mine_thresholds(examples);
    let target_scale = examples.iter().map(|(_, t)| t.abs()).fold(1.0f32, f32::max);
    let accept = tol * target_scale;

    // Combine ops that reference field 1 are meaningless at arity 1.
    let combines: Vec<u32> = if arity == 1 {
        vec![0]
    } else {
        (0..N_COMBINES).collect()
    };

    let mut best: Option<(ProgramV2, f32, u32)> = None;
    let mut n_consistent = 0u32;
    let mut n_searched = 0u32;

    for &combine_idx in &combines {
        for guard_idx in 0..N_GUARDS {
            let guard_thresholds: &[f32] = if guard_idx == 0 { &[0.0] } else { &thresholds };
            for &guard_threshold in guard_thresholds {
                for init_idx in 0..N_INITS {
                    for transform_idx in 0..N_TRANSFORMS {
                        for reduce_idx in 0..N_REDUCES {
                            for post_scale_idx in 0..N_POST_SCALES {
                                n_searched += 1;
                                let base = ProgramV2 {
                                    arity,
                                    combine_idx,
                                    guard_idx,
                                    guard_threshold,
                                    init_idx,
                                    transform_idx,
                                    reduce_idx,
                                    post_scale_idx,
                                    offset: 0.0,
                                };
                                let mut raw_outputs: Vec<f32> = Vec::with_capacity(examples.len());
                                let mut residual_sum = 0.0f32;
                                let mut ok = true;
                                for (data, target) in examples {
                                    let n_points = (data.len() / arity as usize) as u32;
                                    let raw = execute_program_v2(base, data, n_points);
                                    if !raw.is_finite() {
                                        ok = false;
                                        break;
                                    }
                                    raw_outputs.push(raw);
                                    residual_sum += target - raw;
                                }
                                if !ok {
                                    continue;
                                }
                                let mut offset = residual_sum / examples.len() as f32;
                                if offset.abs() < accept.max(1e-6) {
                                    offset = 0.0;
                                }
                                let mut max_err = 0.0f32;
                                for (i, (_, target)) in examples.iter().enumerate() {
                                    max_err = max_err.max((raw_outputs[i] + offset - target).abs());
                                }
                                if max_err <= accept {
                                    n_consistent += 1;
                                    let candidate = ProgramV2 { offset, ..base };
                                    let cx = complexity_v2(&candidate);
                                    let better = match &best {
                                        None => true,
                                        Some((_, be, bc)) => cx < *bc || (cx == *bc && max_err < *be),
                                    };
                                    if better {
                                        best = Some((candidate, max_err, cx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    best.map(|(program, max_err, _)| SynthesisResultV2 {
        program,
        max_err,
        n_consistent,
        n_searched,
    })
}

fn complexity_v3(p: &ProgramV3) -> u32 {
    let mut c = 0;
    if p.combine_idx != 0 {
        c += 1;
    }
    if p.guard_idx != 0 {
        c += 2;
    }
    if p.reset_guard_idx != 0 {
        c += 2;
    }
    if p.state_init_idx != 0 {
        c += 1;
    }
    if p.update_transform_idx != 0 {
        c += 1;
    }
    if p.update_reduce_idx != 0 {
        c += 1;
    }
    if p.post_scale_idx != 0 {
        c += 2;
    }
    if p.output_idx != 0 {
        c += 1;
    }
    if p.offset != 0.0 {
        c += 2;
    }
    c
}

#[derive(Debug, Clone, Copy)]
pub struct SynthesisResultV3 {
    pub program: ProgramV3,
    /// Worst absolute error across every step of every trace after offset
    /// fitting.
    pub max_err: f32,
    pub n_consistent: u32,
    pub n_searched: u32,
}

/// v3 synthesis: exhaustive search over the stateful program space, replaying
/// every candidate over every trace and accepting only when EVERY step's
/// output matches within tolerance. Refusal contract as v1/v2: `None` when
/// nothing in the space explains all traces.
///
/// Each trace is `(data, expected)`: `data` holds the trace's records back to
/// back (`arity` floats per step), `expected` one target output per step, so
/// `data.len() == expected.len() * arity`. Every trace restarts state.
///
/// Search-space size (the inclusion guard and post-scale stay neutral — they
/// exist only for v2 lifting):
///
///   combines (≤8, 1 at arity 1)
///   × reset options (1 never + 4 comparisons × ≤12 mined thresholds = ≤49)
///   × state inits (3) × update transforms (6) × update reduces (4)
///   × output selects (5)
///   = ≤ 8 × 49 × 3 × 6 × 4 × 5 = 141,120 candidates (17,640 at arity 1).
///
/// Each candidate replays all trace steps, so a typical task (3 traces ×
/// ~10 steps) costs ≤ ~4.2M step evaluations — well under the 10^7 budget.
pub fn synthesize_program_v3(
    traces: &[(&[f32], &[f32])],
    arity: u32,
    tol: f32,
) -> Option<SynthesisResultV3> {
    if traces.is_empty() || arity == 0 || arity > MAX_ARITY {
        return None;
    }
    let a = arity as usize;
    let mut total_steps = 0usize;
    for (data, expected) in traces {
        if expected.is_empty() || data.len() != expected.len() * a {
            return None;
        }
        if data.iter().any(|v| !v.is_finite()) || expected.iter().any(|v| !v.is_finite()) {
            return None;
        }
        total_steps += expected.len();
    }
    // Reset thresholds are mined from the traces' input records — the same
    // emergent-vocabulary discipline as v2 (reuse `mine_thresholds`, which
    // only reads the data side of its example pairs).
    let adapted: Vec<(&[f32], f32)> = traces.iter().map(|(d, _)| (*d, 0.0)).collect();
    let thresholds = mine_thresholds(&adapted);
    let target_scale = traces
        .iter()
        .flat_map(|(_, e)| e.iter())
        .map(|t| t.abs())
        .fold(1.0f32, f32::max);
    let accept = tol * target_scale;

    let combines: Vec<u32> = if arity == 1 {
        vec![0]
    } else {
        (0..N_COMBINES).collect()
    };
    // Reset options: (0, _) = never, plus every (comparison, mined t) pair.
    let mut reset_options: Vec<(u32, f32)> = vec![(0, 0.0)];
    for reset_guard_idx in 1..N_RESET_GUARDS {
        for &t in &thresholds {
            reset_options.push((reset_guard_idx, t));
        }
    }

    let mut best: Option<(ProgramV3, f32, u32)> = None;
    let mut n_consistent = 0u32;
    let mut n_searched = 0u32;

    for &combine_idx in &combines {
        // Hoist the combine stage: it is identical for every candidate that
        // shares `combine_idx`.
        let combined: Vec<Vec<f32>> = traces
            .iter()
            .map(|(data, expected)| {
                (0..expected.len())
                    .map(|i| apply_combine(&data[i * a..(i + 1) * a], combine_idx))
                    .collect()
            })
            .collect();
        for &(reset_guard_idx, reset_threshold) in &reset_options {
            for state_init_idx in 0..N_INITS {
                for update_transform_idx in 0..N_TRANSFORMS {
                    for update_reduce_idx in 0..N_REDUCES {
                        for output_idx in 0..N_OUTPUTS_V3 {
                            n_searched += 1;
                            let base = ProgramV3 {
                                arity,
                                combine_idx,
                                guard_idx: 0,
                                guard_threshold: 0.0,
                                reset_guard_idx,
                                reset_threshold,
                                state_init_idx,
                                update_transform_idx,
                                update_reduce_idx,
                                post_scale_idx: 0,
                                output_idx,
                                offset: 0.0,
                            };
                            // Replay traces; closed-form offset = mean
                            // residual over every step of every trace.
                            let mut raw: Vec<f32> = Vec::with_capacity(total_steps);
                            let mut residual_sum = 0.0f32;
                            let mut ok = true;
                            'traces: for (ti, (_, expected)) in traces.iter().enumerate() {
                                let vs = &combined[ti];
                                let mut s = init_value(state_init_idx);
                                for (i, &v) in vs.iter().enumerate() {
                                    if reset_fires(v, reset_guard_idx, reset_threshold) {
                                        s = init_value(state_init_idx);
                                    }
                                    s = apply_reduce(
                                        s,
                                        apply_transform(v, update_transform_idx),
                                        update_reduce_idx,
                                    );
                                    let y = output_select_v3(&base, s, v);
                                    if !y.is_finite() {
                                        ok = false;
                                        break 'traces;
                                    }
                                    raw.push(y);
                                    residual_sum += expected[i] - y;
                                }
                            }
                            if !ok {
                                continue;
                            }
                            let mut offset = residual_sum / total_steps as f32;
                            if offset.abs() < accept.max(1e-6) {
                                offset = 0.0;
                            }
                            let mut max_err = 0.0f32;
                            let mut cursor = 0usize;
                            for (_, expected) in traces {
                                for &target in *expected {
                                    max_err = max_err.max((raw[cursor] + offset - target).abs());
                                    cursor += 1;
                                }
                            }
                            if max_err <= accept {
                                n_consistent += 1;
                                let candidate = ProgramV3 { offset, ..base };
                                let cx = complexity_v3(&candidate);
                                let better = match &best {
                                    None => true,
                                    Some((_, be, bc)) => {
                                        cx < *bc || (cx == *bc && max_err < *be)
                                    }
                                };
                                if better {
                                    best = Some((candidate, max_err, cx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    best.map(|(program, max_err, _)| SynthesisResultV3 {
        program,
        max_err,
        n_consistent,
        n_searched,
    })
}

// ---------------------------------------------------------------------------
// Multi-language source rendering. One discovered program, five working
// implementations — the synthesized artifact is an IR, not a string.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lang {
    Rust,
    Python,
    JavaScript,
    C,
    TypeScript,
}

impl Lang {
    pub fn from_name(name: &str) -> Option<Lang> {
        match name {
            "rust" => Some(Lang::Rust),
            "python" => Some(Lang::Python),
            "javascript" | "js" => Some(Lang::JavaScript),
            "c" => Some(Lang::C),
            "typescript" | "ts" => Some(Lang::TypeScript),
            _ => None,
        }
    }
}

fn combine_expr(p: &ProgramV2, lang: Lang) -> String {
    let abs = |e: String| match lang {
        Lang::Rust => format!("({e}).abs()"),
        Lang::Python => format!("abs({e})"),
        Lang::C => format!("fabsf({e})"),
        _ => format!("Math.abs({e})"),
    };
    let minmax = |f: &str| match lang {
        Lang::Rust => format!("pt.iter().copied().fold(f32::{}, f32::{})", if f == "min" { "INFINITY" } else { "NEG_INFINITY" }, f),
        Lang::Python => format!("{f}(pt)"),
        Lang::C => format!("/* {f} over fields */ {f}_fields(pt, k)"),
        _ => format!("Math.{f}(...pt)"),
    };
    match p.combine_idx {
        1 => "pt[1]".to_string(),
        2 => match lang {
            Lang::Rust => "pt.iter().sum::<f32>()".to_string(),
            Lang::Python => "sum(pt)".to_string(),
            Lang::C => "sum_fields(pt, k)".to_string(),
            _ => "pt.reduce((a, b) => a + b, 0)".to_string(),
        },
        3 => match lang {
            Lang::Rust => "pt.iter().product::<f32>()".to_string(),
            Lang::Python => "math.prod(pt)".to_string(),
            Lang::C => "prod_fields(pt, k)".to_string(),
            _ => "pt.reduce((a, b) => a * b, 1)".to_string(),
        },
        4 => "pt[0] - pt[1]".to_string(),
        5 => abs("pt[0] - pt[1]".to_string()),
        6 => minmax("min"),
        7 => minmax("max"),
        _ => if p.arity == 1 { "pt[0]".to_string() } else { "pt[0]".to_string() },
    }
}

fn transform_expr(idx: u32, lang: Lang) -> String {
    match idx {
        1 => "v * v".to_string(),
        2 => match lang {
            Lang::Rust => "v.abs()".to_string(),
            Lang::Python => "abs(v)".to_string(),
            Lang::C => "fabsf(v)".to_string(),
            _ => "Math.abs(v)".to_string(),
        },
        3 => "1.0".to_string(),
        4 => match lang {
            Lang::Python => "(1.0 if v > 0 else 0.0)".to_string(),
            Lang::Rust => "if v > 0.0 { 1.0 } else { 0.0 }".to_string(),
            Lang::C => "(v > 0.0f ? 1.0f : 0.0f)".to_string(),
            _ => "(v > 0 ? 1.0 : 0.0)".to_string(),
        },
        5 => match lang {
            Lang::Rust => "(v.abs() + 1e-6).ln()".to_string(),
            Lang::Python => "math.log(abs(v) + 1e-6)".to_string(),
            Lang::C => "logf(fabsf(v) + 1e-6f)".to_string(),
            _ => "Math.log(Math.abs(v) + 1e-6)".to_string(),
        },
        _ => "v".to_string(),
    }
}

fn guard_expr(p: &ProgramV2, lang: Lang) -> Option<String> {
    let t = p.guard_threshold;
    let abs_v = match lang {
        Lang::Rust => "v.abs()".to_string(),
        Lang::Python => "abs(v)".to_string(),
        Lang::C => "fabsf(v)".to_string(),
        _ => "Math.abs(v)".to_string(),
    };
    match p.guard_idx {
        1 => Some(format!("v > {t}")),
        2 => Some(format!("v < {t}")),
        3 => Some(format!("{abs_v} > {t}")),
        4 => Some(match lang {
            Lang::Rust => format!("(v - {t}).abs() < 1e-4"),
            Lang::Python => format!("abs(v - {t}) < 1e-4"),
            Lang::C => format!("fabsf(v - {t}f) < 1e-4f"),
            _ => format!("Math.abs(v - {t}) < 1e-4"),
        }),
        _ => None,
    }
}

/// Render a discovered v2 program in any supported language. Output is a
/// complete, runnable function over `points` (each point = `arity` floats).
pub fn program_source_v2(name: &str, p: &ProgramV2, lang: Lang) -> String {
    let init = match p.init_idx {
        1 => "1.0".to_string(),
        2 => "-20.0".to_string(),
        _ => "0.0".to_string(),
    };
    let combine = combine_expr(p, lang);
    let transform = transform_expr(p.transform_idx, lang);
    let guard = guard_expr(p, lang);
    let reduce_stmt = |acc: &str, f: &str, lang: Lang| match p.reduce_idx {
        1 => format!("{acc} *= {f}"),
        2 => match lang {
            Lang::Rust => format!("{acc} = {acc}.max({f})"),
            Lang::Python => format!("{acc} = max({acc}, {f})"),
            Lang::C => format!("{acc} = fmaxf({acc}, {f})"),
            _ => format!("{acc} = Math.max({acc}, {f})"),
        },
        3 => match lang {
            Lang::Rust => format!("{acc} = {acc}.min({f})"),
            Lang::Python => format!("{acc} = min({acc}, {f})"),
            Lang::C => format!("{acc} = fminf({acc}, {f})"),
            _ => format!("{acc} = Math.min({acc}, {f})"),
        },
        _ => format!("{acc} += {f}"),
    };
    let offset_suffix = if p.offset != 0.0 {
        format!(" + {:.6}", p.offset)
    } else {
        String::new()
    };

    match lang {
        Lang::Rust => {
            let mut body = String::new();
            body.push_str(&format!("fn {name}(points: &[[f32; {}]]) -> f32 {{\n", p.arity));
            body.push_str(&format!("    let mut acc: f32 = {init};\n"));
            if p.post_scale_idx == 1 {
                body.push_str("    let mut n: f32 = 0.0;\n");
            }
            body.push_str("    for pt in points {\n");
            body.push_str(&format!("        let v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if !({g}) {{ continue; }}\n"));
            }
            if p.post_scale_idx == 1 {
                body.push_str("        n += 1.0;\n");
            }
            body.push_str(&format!("        {};\n", reduce_stmt("acc", &transform, lang)));
            body.push_str("    }\n");
            if p.post_scale_idx == 1 {
                body.push_str("    acc /= n.max(1.0);\n");
            } else if p.post_scale_idx == 2 {
                body.push_str("    acc = acc.clamp(-30.0, 30.0).exp();\n");
            }
            body.push_str(&format!("    acc{offset_suffix}\n}}"));
            body
        }
        Lang::Python => {
            let mut body = String::new();
            if p.transform_idx == 5 || p.combine_idx == 3 {
                body.push_str("import math\n\n");
            }
            body.push_str(&format!("def {name}(points):  # each point: {} value(s)\n", p.arity));
            body.push_str(&format!("    acc = {init}\n"));
            if p.post_scale_idx == 1 {
                body.push_str("    n = 0\n");
            }
            body.push_str("    for pt in points:\n");
            body.push_str(&format!("        v = {combine}\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if not ({g}):\n            continue\n"));
            }
            if p.post_scale_idx == 1 {
                body.push_str("        n += 1\n");
            }
            body.push_str(&format!("        {}\n", reduce_stmt("acc", &transform, lang)));
            if p.post_scale_idx == 1 {
                body.push_str("    acc /= max(n, 1)\n");
            } else if p.post_scale_idx == 2 {
                body.push_str("    acc = math.exp(max(-30.0, min(30.0, acc)))\n");
            }
            body.push_str(&format!("    return acc{offset_suffix}\n"));
            body
        }
        Lang::JavaScript | Lang::TypeScript => {
            let sig = if lang == Lang::TypeScript {
                format!("function {name}(points: number[][]): number {{")
            } else {
                format!("function {name}(points) {{")
            };
            let mut body = String::new();
            body.push_str(&format!("{sig}\n"));
            body.push_str(&format!("  let acc = {init};\n"));
            if p.post_scale_idx == 1 {
                body.push_str("  let n = 0;\n");
            }
            body.push_str("  for (const pt of points) {\n");
            body.push_str(&format!("    const v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("    if (!({g})) continue;\n"));
            }
            if p.post_scale_idx == 1 {
                body.push_str("    n += 1;\n");
            }
            body.push_str(&format!("    {};\n", reduce_stmt("acc", &transform, lang)));
            body.push_str("  }\n");
            if p.post_scale_idx == 1 {
                body.push_str("  acc /= Math.max(n, 1);\n");
            } else if p.post_scale_idx == 2 {
                body.push_str("  acc = Math.exp(Math.max(-30, Math.min(30, acc)));\n");
            }
            body.push_str(&format!("  return acc{offset_suffix};\n}}"));
            body
        }
        Lang::C => {
            let mut body = String::new();
            body.push_str("#include <math.h>\n\n");
            if p.combine_idx == 2 || p.combine_idx == 3 || p.combine_idx >= 6 {
                body.push_str("static float sum_fields(const float* pt, int k) { float s = 0; for (int j = 0; j < k; j++) s += pt[j]; return s; }\n");
                body.push_str("static float prod_fields(const float* pt, int k) { float s = 1; for (int j = 0; j < k; j++) s *= pt[j]; return s; }\n");
                body.push_str("static float min_fields(const float* pt, int k) { float s = pt[0]; for (int j = 1; j < k; j++) s = fminf(s, pt[j]); return s; }\n");
                body.push_str("static float max_fields(const float* pt, int k) { float s = pt[0]; for (int j = 1; j < k; j++) s = fmaxf(s, pt[j]); return s; }\n\n");
            }
            body.push_str(&format!(
                "float {name}(const float* data, int n_points) {{\n    const int k = {};\n    float acc = {init}f;\n",
                p.arity
            ));
            if p.post_scale_idx == 1 {
                body.push_str("    float n = 0.0f;\n");
            }
            body.push_str("    for (int i = 0; i < n_points; i++) {\n");
            body.push_str("        const float* pt = data + i * k;\n");
            body.push_str(&format!("        float v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if (!({g})) continue;\n"));
            }
            if p.post_scale_idx == 1 {
                body.push_str("        n += 1.0f;\n");
            }
            body.push_str(&format!("        {};\n", reduce_stmt("acc", &transform, lang)));
            body.push_str("    }\n");
            if p.post_scale_idx == 1 {
                body.push_str("    acc /= fmaxf(n, 1.0f);\n");
            } else if p.post_scale_idx == 2 {
                body.push_str("    acc = expf(fminf(fmaxf(acc, -30.0f), 30.0f));\n");
            }
            body.push_str(&format!("    return acc{offset_suffix};\n}}"));
            body
        }
    }
}

fn reduce_stmt_v3(reduce_idx: u32, acc: &str, f: &str, lang: Lang) -> String {
    match reduce_idx {
        1 => format!("{acc} *= {f}"),
        2 => match lang {
            Lang::Rust => format!("{acc} = {acc}.max({f})"),
            Lang::Python => format!("{acc} = max({acc}, {f})"),
            Lang::C => format!("{acc} = fmaxf({acc}, {f})"),
            _ => format!("{acc} = Math.max({acc}, {f})"),
        },
        3 => match lang {
            Lang::Rust => format!("{acc} = {acc}.min({f})"),
            Lang::Python => format!("{acc} = min({acc}, {f})"),
            Lang::C => format!("{acc} = fminf({acc}, {f})"),
            _ => format!("{acc} = Math.min({acc}, {f})"),
        },
        _ => format!("{acc} += {f}"),
    }
}

fn output_expr_v3(p: &ProgramV3, lang: Lang) -> String {
    match p.output_idx {
        1 => "v".to_string(),
        2 => "s + v".to_string(),
        3 => "s * v".to_string(),
        4 => match lang {
            Lang::Rust => "s.abs()".to_string(),
            Lang::Python => "abs(s)".to_string(),
            Lang::C => "fabsf(s)".to_string(),
            _ => "Math.abs(s)".to_string(),
        },
        _ => "s".to_string(),
    }
}

/// Render a discovered v3 program in any supported language: a loop with a
/// mutable state cell emitting one output per step. Full fidelity with
/// `execute_program_v3`, including the inclusion-guard / post-scale stages
/// carried for v2-lifted programs (synthesized v3 programs keep both
/// neutral, so the common rendering is a clean stateful loop).
pub fn program_source_v3(name: &str, p: &ProgramV3, lang: Lang) -> String {
    let init = match p.state_init_idx {
        1 => "1.0".to_string(),
        2 => "-20.0".to_string(),
        _ => "0.0".to_string(),
    };
    // Reuse the v2 expression renderers through a v2 view of the shared
    // stages (combine + inclusion guard); the reset guard reuses the same
    // comparison vocabulary via a second view.
    let v2_view = ProgramV2 {
        arity: p.arity,
        combine_idx: p.combine_idx,
        guard_idx: p.guard_idx,
        guard_threshold: p.guard_threshold,
        init_idx: p.state_init_idx,
        transform_idx: p.update_transform_idx,
        reduce_idx: p.update_reduce_idx,
        post_scale_idx: p.post_scale_idx,
        offset: p.offset,
    };
    let reset_view = ProgramV2 { guard_idx: p.reset_guard_idx, guard_threshold: p.reset_threshold, ..v2_view };
    let combine = combine_expr(&v2_view, lang);
    let guard = guard_expr(&v2_view, lang);
    let reset = guard_expr(&reset_view, lang);
    let transform = transform_expr(p.update_transform_idx, lang);
    let needs_n = p.post_scale_idx == 1;
    let y_expr = {
        let y = output_expr_v3(p, lang);
        match p.post_scale_idx {
            1 => match lang {
                Lang::Rust => format!("({y}) / n.max(1.0)"),
                Lang::Python => format!("({y}) / max(n, 1)"),
                Lang::C => format!("({y}) / fmaxf(n, 1.0f)"),
                _ => format!("({y}) / Math.max(n, 1)"),
            },
            2 => match lang {
                Lang::Rust => format!("({y}).clamp(-30.0, 30.0).exp()"),
                Lang::Python => format!("math.exp(max(-30.0, min(30.0, {y})))"),
                Lang::C => format!("expf(fminf(fmaxf({y}, -30.0f), 30.0f))"),
                _ => format!("Math.exp(Math.max(-30, Math.min(30, {y})))"),
            },
            _ => y,
        }
    };
    let offset_suffix = if p.offset != 0.0 {
        format!(" + {:.6}", p.offset)
    } else {
        String::new()
    };

    // The state-update block (reset check, included counter, fold), shared
    // shape across languages; wrapped in the inclusion guard when present.
    let update_lines = |indent: &str, lang: Lang| -> String {
        let mut lines = String::new();
        if let Some(r) = &reset {
            match lang {
                Lang::Python => {
                    lines.push_str(&format!("{indent}if {r}:\n"));
                    lines.push_str(&format!("{indent}    s = {init}\n"));
                    if needs_n {
                        lines.push_str(&format!("{indent}    n = 0\n"));
                    }
                }
                _ => {
                    let n_reset = if needs_n {
                        if lang == Lang::C { " n = 0.0f;" } else { " n = 0.0;" }
                    } else {
                        ""
                    };
                    let init_lit = if lang == Lang::C { format!("{init}f") } else { init.clone() };
                    lines.push_str(&format!("{indent}if ({r}) {{ s = {init_lit};{n_reset} }}\n"));
                }
            }
        }
        if needs_n {
            match lang {
                Lang::Python => lines.push_str(&format!("{indent}n += 1\n")),
                Lang::C => lines.push_str(&format!("{indent}n += 1.0f;\n")),
                _ => lines.push_str(&format!("{indent}n += 1.0;\n")),
            }
        }
        let stmt = reduce_stmt_v3(p.update_reduce_idx, "s", &transform, lang);
        match lang {
            Lang::Python => lines.push_str(&format!("{indent}{stmt}\n")),
            _ => lines.push_str(&format!("{indent}{stmt};\n")),
        }
        lines
    };

    match lang {
        Lang::Rust => {
            let mut body = String::new();
            body.push_str(&format!(
                "fn {name}(points: &[[f32; {}]]) -> Vec<f32> {{\n",
                p.arity
            ));
            body.push_str(&format!("    let mut s: f32 = {init};\n"));
            if needs_n {
                body.push_str("    let mut n: f32 = 0.0;\n");
            }
            body.push_str("    let mut out: Vec<f32> = Vec::with_capacity(points.len());\n");
            body.push_str("    for pt in points {\n");
            body.push_str(&format!("        let v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if {g} {{\n"));
                body.push_str(&update_lines("            ", lang));
                body.push_str("        }\n");
            } else {
                body.push_str(&update_lines("        ", lang));
            }
            body.push_str(&format!("        out.push({y_expr}{offset_suffix});\n"));
            body.push_str("    }\n    out\n}");
            body
        }
        Lang::Python => {
            let mut body = String::new();
            if p.update_transform_idx == 5 || p.combine_idx == 3 || p.post_scale_idx == 2 {
                body.push_str("import math\n\n");
            }
            body.push_str(&format!(
                "def {name}(points):  # each point: {} value(s)\n",
                p.arity
            ));
            body.push_str(&format!("    s = {init}\n"));
            if needs_n {
                body.push_str("    n = 0\n");
            }
            body.push_str("    out = []\n");
            body.push_str("    for pt in points:\n");
            body.push_str(&format!("        v = {combine}\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if {g}:\n"));
                body.push_str(&update_lines("            ", lang));
            } else {
                body.push_str(&update_lines("        ", lang));
            }
            body.push_str(&format!("        out.append({y_expr}{offset_suffix})\n"));
            body.push_str("    return out\n");
            body
        }
        Lang::JavaScript | Lang::TypeScript => {
            let sig = if lang == Lang::TypeScript {
                format!("function {name}(points: number[][]): number[] {{")
            } else {
                format!("function {name}(points) {{")
            };
            let mut body = String::new();
            body.push_str(&format!("{sig}\n"));
            body.push_str(&format!("  let s = {init};\n"));
            if needs_n {
                body.push_str("  let n = 0;\n");
            }
            let out_decl = if lang == Lang::TypeScript {
                "  const out: number[] = [];\n"
            } else {
                "  const out = [];\n"
            };
            body.push_str(out_decl);
            body.push_str("  for (const pt of points) {\n");
            body.push_str(&format!("    const v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("    if ({g}) {{\n"));
                body.push_str(&update_lines("      ", lang));
                body.push_str("    }\n");
            } else {
                body.push_str(&update_lines("    ", lang));
            }
            body.push_str(&format!("    out.push({y_expr}{offset_suffix});\n"));
            body.push_str("  }\n  return out;\n}");
            body
        }
        Lang::C => {
            let mut body = String::new();
            body.push_str("#include <math.h>\n\n");
            if p.combine_idx == 2 || p.combine_idx == 3 || p.combine_idx >= 6 {
                body.push_str("static float sum_fields(const float* pt, int k) { float s = 0; for (int j = 0; j < k; j++) s += pt[j]; return s; }\n");
                body.push_str("static float prod_fields(const float* pt, int k) { float s = 1; for (int j = 0; j < k; j++) s *= pt[j]; return s; }\n");
                body.push_str("static float min_fields(const float* pt, int k) { float s = pt[0]; for (int j = 1; j < k; j++) s = fminf(s, pt[j]); return s; }\n");
                body.push_str("static float max_fields(const float* pt, int k) { float s = pt[0]; for (int j = 1; j < k; j++) s = fmaxf(s, pt[j]); return s; }\n\n");
            }
            body.push_str(&format!(
                "void {name}(const float* data, int n_steps, float* out) {{\n    const int k = {};\n    float s = {init}f;\n",
                p.arity
            ));
            if needs_n {
                body.push_str("    float n = 0.0f;\n");
            }
            body.push_str("    for (int i = 0; i < n_steps; i++) {\n");
            body.push_str("        const float* pt = data + i * k;\n");
            body.push_str(&format!("        float v = {combine};\n"));
            if let Some(g) = &guard {
                body.push_str(&format!("        if ({g}) {{\n"));
                body.push_str(&update_lines("            ", lang));
                body.push_str("        }\n");
            } else {
                body.push_str(&update_lines("        ", lang));
            }
            body.push_str(&format!("        out[i] = {y_expr}{offset_suffix};\n"));
            body.push_str("    }\n}");
            body
        }
    }
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

/// Serialize an index back to canonical library JSON at the LOWEST loadable
/// format. Pure-v1 libraries are emitted in the v1 format so every existing
/// runtime loads them; libraries containing any v2 capability (records,
/// guards) are emitted as `"format": 2` with `program_v2` keys; libraries
/// containing any v3 capability (state resets, non-fold outputs) are emitted
/// as `"format": 3` with `program_v3` keys. In a higher-format file every
/// entry is lifted to that format entry-wise (lifting is exact — see
/// `ProgramV2::from_v1` / `ProgramV3::from_v2`), and lower-format loaders
/// reject the file cleanly instead of silently mis-executing.
pub fn library_to_json(index: &NativeIndex) -> String {
    let needs_v3 = index
        .entries
        .iter()
        .any(|e| e.program_v3.map(|p| !p.is_v2()).unwrap_or(false));
    let needs_v2 = !needs_v3
        && index
            .entries
            .iter()
            .any(|e| e.program_v2.map(|p| !p.is_v1()).unwrap_or(false));
    let mut out = String::new();
    out.push_str("{\n");
    if needs_v3 {
        out.push_str("  \"format\": 3,\n");
    } else if needs_v2 {
        out.push_str("  \"format\": 2,\n");
    }
    out.push_str(&format!(
        "  \"config\": {{\"similarity_threshold\": {}, \"max_entries\": {}, \"normalize_epsilon\": 1e-08}},\n  \"entries\": [\n",
        index.similarity_threshold,
        index.entries.len().max(16)
    ));
    for (i, e) in index.entries.iter().enumerate() {
        let sig: Vec<String> = e.signature.iter().map(|v| format!("{v}")).collect();
        let trailing = if i + 1 == index.entries.len() { "" } else { "," };
        if needs_v3 {
            let p = e.effective_program_v3();
            out.push_str(&format!(
                "    {{\"signature\": [{}], \"program_v3\": {{\"arity\": {}, \"combine_idx\": {}, \"guard_idx\": {}, \"guard_threshold\": {}, \"reset_guard_idx\": {}, \"reset_threshold\": {}, \"state_init_idx\": {}, \"update_transform_idx\": {}, \"update_reduce_idx\": {}, \"post_scale_idx\": {}, \"output_idx\": {}, \"offset\": {}}}, \"hit_count\": 0, \"task_name\": \"entry_{i}\", \"cached_at_step\": null, \"convergence_gap\": null}}{trailing}\n",
                sig.join(", "),
                p.arity, p.combine_idx, p.guard_idx, p.guard_threshold,
                p.reset_guard_idx, p.reset_threshold, p.state_init_idx,
                p.update_transform_idx, p.update_reduce_idx, p.post_scale_idx,
                p.output_idx, p.offset,
            ));
        } else if needs_v2 {
            let p = e.effective_program();
            out.push_str(&format!(
                "    {{\"signature\": [{}], \"program_v2\": {{\"arity\": {}, \"combine_idx\": {}, \"guard_idx\": {}, \"guard_threshold\": {}, \"init_idx\": {}, \"transform_idx\": {}, \"reduce_idx\": {}, \"post_scale_idx\": {}, \"offset\": {}}}, \"hit_count\": 0, \"task_name\": \"entry_{i}\", \"cached_at_step\": null, \"convergence_gap\": null}}{trailing}\n",
                sig.join(", "),
                p.arity, p.combine_idx, p.guard_idx, p.guard_threshold,
                p.init_idx, p.transform_idx, p.reduce_idx, p.post_scale_idx, p.offset,
            ));
        } else {
            out.push_str(&format!(
                "    {{\"signature\": [{}], \"program\": {{\"init_idx\": {}, \"transform_idx\": {}, \"reduce_idx\": {}, \"post_scale_idx\": {}, \"offset\": {}}}, \"hit_count\": 0, \"task_name\": \"entry_{i}\", \"cached_at_step\": null, \"convergence_gap\": null}}{trailing}\n",
                sig.join(", "),
                e.program.init_idx,
                e.program.transform_idx,
                e.program.reduce_idx,
                e.program.post_scale_idx,
                e.program.offset,
            ));
        }
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
            program_v2: None,
            program_v3: None,
        });
    }

    /// v2 synthesis over multi-field data points. `data` holds all examples'
    /// records back to back; `point_counts[i]` gives example i's number of
    /// points (each point = `arity` floats); `targets[i]` its expected
    /// output. Returns JSON with the program, search stats, and rendered
    /// source in all five supported languages.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_v2(
        &self,
        data: Vec<f32>,
        point_counts: Vec<u32>,
        targets: Vec<f32>,
        arity: u32,
        name: String,
    ) -> Option<String> {
        if point_counts.len() != targets.len() || point_counts.is_empty() {
            return None;
        }
        let total: usize = point_counts.iter().map(|l| (*l * arity) as usize).sum();
        if total != data.len() {
            return None;
        }
        let mut examples: Vec<(&[f32], f32)> = Vec::with_capacity(point_counts.len());
        let mut cursor = 0usize;
        for (i, n_points) in point_counts.iter().enumerate() {
            let end = cursor + (*n_points * arity) as usize;
            examples.push((&data[cursor..end], targets[i]));
            cursor = end;
        }
        let result = synthesize_program_v2(&examples, arity, 1e-3)?;
        let p = result.program;
        let esc = |s: String| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let sources: Vec<String> = [
            ("rust", Lang::Rust),
            ("python", Lang::Python),
            ("javascript", Lang::JavaScript),
            ("c", Lang::C),
            ("typescript", Lang::TypeScript),
        ]
        .iter()
        .map(|(label, lang)| {
            format!("\"{label}\": \"{}\"", esc(program_source_v2(&name, &p, *lang)))
        })
        .collect();
        Some(format!(
            "{{\"arity\": {}, \"combine_idx\": {}, \"guard_idx\": {}, \"guard_threshold\": {}, \"init_idx\": {}, \"transform_idx\": {}, \"reduce_idx\": {}, \"post_scale_idx\": {}, \"offset\": {}, \"max_err\": {}, \"n_consistent\": {}, \"n_searched\": {}, \"sources\": {{{}}}}}",
            p.arity, p.combine_idx, p.guard_idx, p.guard_threshold,
            p.init_idx, p.transform_idx, p.reduce_idx, p.post_scale_idx, p.offset,
            result.max_err, result.n_consistent, result.n_searched,
            sources.join(", ")
        ))
    }

    /// Insert a v2 skill (records + guards) into the live library.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_skill_v2(
        &mut self,
        signature: Vec<f32>,
        arity: u32,
        combine_idx: u32,
        guard_idx: u32,
        guard_threshold: f32,
        init_idx: u32,
        transform_idx: u32,
        reduce_idx: u32,
        post_scale_idx: u32,
        offset: f32,
    ) {
        let v2 = ProgramV2 {
            arity: arity.clamp(1, MAX_ARITY),
            combine_idx,
            guard_idx,
            guard_threshold,
            init_idx,
            transform_idx,
            reduce_idx,
            post_scale_idx,
            offset,
        };
        self.index.insert(NativeEntry {
            signature,
            program: DiscreteProgram {
                init_idx,
                transform_idx,
                reduce_idx,
                post_scale_idx,
                offset,
            },
            program_v2: Some(v2),
            program_v3: None,
        });
    }

    /// v3 synthesis over stateful TRACES.
    ///
    /// Flat trace encoding:
    /// * `data` — all traces' input records back to back. Trace `i`
    ///   contributes `point_counts[i]` records of `arity` floats each
    ///   (so trace `i` occupies `point_counts[i] * arity` floats).
    /// * `point_counts[i]` — number of steps in trace `i`.
    /// * `expected` — all traces' per-step expected outputs back to back:
    ///   trace `i` contributes `point_counts[i]` floats. Total length must
    ///   equal `sum(point_counts)`.
    ///
    /// Every trace restarts state. Returns JSON with the discovered
    /// program's fields, search stats, and rendered source in all five
    /// supported languages — or `None` when no program in the v3 space
    /// reproduces every step of every trace (refusal, not approximation).
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_v3(
        &self,
        data: Vec<f32>,
        point_counts: Vec<u32>,
        expected: Vec<f32>,
        arity: u32,
        name: String,
    ) -> Option<String> {
        if point_counts.is_empty() || arity == 0 {
            return None;
        }
        let total_steps: usize = point_counts.iter().map(|l| *l as usize).sum();
        let total_floats: usize = total_steps * arity as usize;
        if total_floats != data.len() || total_steps != expected.len() {
            return None;
        }
        let mut traces: Vec<(&[f32], &[f32])> = Vec::with_capacity(point_counts.len());
        let mut d_cursor = 0usize;
        let mut e_cursor = 0usize;
        for n_points in &point_counts {
            let d_end = d_cursor + (*n_points * arity) as usize;
            let e_end = e_cursor + *n_points as usize;
            traces.push((&data[d_cursor..d_end], &expected[e_cursor..e_end]));
            d_cursor = d_end;
            e_cursor = e_end;
        }
        let result = synthesize_program_v3(&traces, arity, 1e-3)?;
        let p = result.program;
        let esc = |s: String| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
        let sources: Vec<String> = [
            ("rust", Lang::Rust),
            ("python", Lang::Python),
            ("javascript", Lang::JavaScript),
            ("c", Lang::C),
            ("typescript", Lang::TypeScript),
        ]
        .iter()
        .map(|(label, lang)| {
            format!("\"{label}\": \"{}\"", esc(program_source_v3(&name, &p, *lang)))
        })
        .collect();
        Some(format!(
            "{{\"arity\": {}, \"combine_idx\": {}, \"reset_guard_idx\": {}, \"reset_threshold\": {}, \"state_init_idx\": {}, \"update_transform_idx\": {}, \"update_reduce_idx\": {}, \"output_idx\": {}, \"offset\": {}, \"max_err\": {}, \"n_consistent\": {}, \"n_searched\": {}, \"sources\": {{{}}}}}",
            p.arity, p.combine_idx, p.reset_guard_idx, p.reset_threshold,
            p.state_init_idx, p.update_transform_idx, p.update_reduce_idx,
            p.output_idx, p.offset,
            result.max_err, result.n_consistent, result.n_searched,
            sources.join(", ")
        ))
    }

    /// Insert a stateful (v3) skill from the synthesized v3 space into the
    /// live library. The inclusion-guard and post-scale stages stay neutral
    /// (they exist only for v2 lifting); to insert a stateless guarded fold
    /// use `insert_skill_v2`.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_skill_v3(
        &mut self,
        signature: Vec<f32>,
        arity: u32,
        combine_idx: u32,
        reset_guard_idx: u32,
        reset_threshold: f32,
        state_init_idx: u32,
        update_transform_idx: u32,
        update_reduce_idx: u32,
        output_idx: u32,
        offset: f32,
    ) {
        let v3 = ProgramV3 {
            arity: arity.clamp(1, MAX_ARITY),
            combine_idx,
            guard_idx: 0,
            guard_threshold: 0.0,
            reset_guard_idx,
            reset_threshold,
            state_init_idx,
            update_transform_idx,
            update_reduce_idx,
            post_scale_idx: 0,
            output_idx,
            offset,
        };
        self.index.insert(NativeEntry {
            signature,
            program: DiscreteProgram {
                init_idx: state_init_idx,
                transform_idx: update_transform_idx,
                reduce_idx: update_reduce_idx,
                post_scale_idx: 0,
                offset,
            },
            program_v2: v3.to_v2(),
            program_v3: Some(v3),
        });
    }

    /// Consult path for stateful skills: similarity lookup on `hidden`,
    /// then replay the matched program over `n_steps` records of `arity`
    /// floats in `inputs`, returning ALL per-step outputs (a `Float32Array`
    /// of length `n_steps`). v1/v2 entries answer too (lifted exactly; their
    /// per-step outputs are the running fold — the last element equals what
    /// `consult` returns). `None` on lookup miss or arity mismatch. For a
    /// single final output, `consult` also accepts v3 entries and returns
    /// the last step's output.
    pub fn consult_v3(
        &self,
        hidden: Vec<f32>,
        inputs: Vec<f32>,
        arity: u32,
        n_steps: u32,
    ) -> Option<Vec<f32>> {
        consult_native_v3(&self.index, &hidden, &inputs, arity, n_steps)
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
            program_v2: None,
            program_v3: None,
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

    // ------------------------------------------------------------- v2 tests

    #[test]
    fn v2_discovers_dot_product() {
        // points (x,y); target = Σ x*y
        let e1 = [1.0, 2.0, 3.0, 4.0]; // (1,2),(3,4) -> 2+12=14
        let e2 = [2.0, 5.0]; // (2,5) -> 10
        let examples: Vec<(&[f32], f32)> = vec![(&e1, 14.0), (&e2, 10.0)];
        let r = synthesize_program_v2(&examples, 2, 1e-3).expect("dot product");
        let p = r.program;
        assert_eq!(p.combine_idx, 3, "product of fields");
        assert_eq!(p.reduce_idx, 0, "sum reduce");
        // Held-out check.
        let held = [10.0, 0.5, 4.0, 0.25];
        assert!((execute_program_v2(p, &held, 2) - 6.0).abs() < 1e-3);
    }

    #[test]
    fn v2_discovers_guarded_sum_with_mined_threshold() {
        // sum of values above 10 — threshold 10 must be mined from the data.
        let e1 = [3.0, 12.0, 10.0, 20.0]; // 12+20=32
        let e2 = [11.0, 2.0]; // 11
        let e3 = [1.0, 4.0]; // 0
        let examples: Vec<(&[f32], f32)> = vec![(&e1, 32.0), (&e2, 11.0), (&e3, 0.0)];
        let r = synthesize_program_v2(&examples, 1, 1e-3).expect("guarded sum");
        let p = r.program;
        assert_eq!(p.guard_idx, 1, "v > t guard");
        assert!((p.guard_threshold - 10.0).abs() < 1e-5, "mined threshold {}", p.guard_threshold);
        let held = [9.0, 15.0, 30.0];
        assert!((execute_program_v2(p, &held, 3) - 45.0).abs() < 1e-3);
    }

    #[test]
    fn v2_discovers_manhattan_distance() {
        // Σ |x - y| over point pairs
        let e1 = [1.0, 4.0, 10.0, 7.0]; // 3 + 3 = 6
        let e2 = [0.0, 5.0]; // 5
        let examples: Vec<(&[f32], f32)> = vec![(&e1, 6.0), (&e2, 5.0)];
        let r = synthesize_program_v2(&examples, 2, 1e-3).expect("manhattan");
        let held = [2.0, 2.5, -1.0, 1.0];
        assert!((execute_program_v2(r.program, &held, 2) - 2.5).abs() < 1e-3);
    }

    #[test]
    fn v2_guarded_mean_divides_by_included() {
        let p = ProgramV2 {
            arity: 1,
            combine_idx: 0,
            guard_idx: 1,
            guard_threshold: 0.0,
            init_idx: 0,
            transform_idx: 0,
            reduce_idx: 0,
            post_scale_idx: 1,
            offset: 0.0,
        };
        // mean of positives: (4 + 6) / 2 = 5, NOT (4+6)/4
        let data = [4.0, -2.0, 6.0, -8.0];
        assert!((execute_program_v2(p, &data, 4) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn v2_arity1_unguarded_matches_v1_exactly() {
        for init_idx in 0..N_INITS {
            for transform_idx in 0..N_TRANSFORMS {
                for reduce_idx in 0..N_REDUCES {
                    for post_scale_idx in 0..N_POST_SCALES {
                        let v1 = DiscreteProgram {
                            init_idx,
                            transform_idx,
                            reduce_idx,
                            post_scale_idx,
                            offset: 0.25,
                        };
                        let data = [3.0, -1.5, 4.0, 0.0, 9.0];
                        let a = execute_program(v1, &data, 5);
                        let b = execute_program_v2(ProgramV2::from_v1(v1), &data, 5);
                        if a.is_finite() && b.is_finite() {
                            assert!(
                                (a - b).abs() < 1e-5,
                                "v1/v2 divergence at {init_idx}/{transform_idx}/{reduce_idx}/{post_scale_idx}: {a} vs {b}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn v2_library_round_trips_and_v1_loader_rejects_it() {
        let mut index = NativeIndex::new(0.85);
        index.insert(NativeEntry {
            signature: vec![0.0, 1.0, 0.0],
            program: DiscreteProgram {
                init_idx: 0,
                transform_idx: 0,
                reduce_idx: 0,
                post_scale_idx: 0,
                offset: 0.0,
            },
            program_v2: Some(ProgramV2 {
                arity: 2,
                combine_idx: 3,
                guard_idx: 0,
                guard_threshold: 0.0,
                init_idx: 0,
                transform_idx: 0,
                reduce_idx: 0,
                post_scale_idx: 0,
                offset: 0.0,
            }),
            program_v3: None,
        });
        let json = library_to_json(&index);
        assert!(json.contains("\"format\": 2"));
        assert!(json.contains("program_v2"));
        assert!(!json.contains("\"program\":"), "v2 export must not carry a v1 program key");
        // New loader round-trips it.
        let (_thr, parsed) = load_library_json(&json).expect("v2 round trip");
        let p = parsed.entries[0].effective_program();
        assert_eq!((p.arity, p.combine_idx), (2, 3));
        // Consult executes dot product through the v2 engine.
        let r = consult_native(&parsed, &[0.0, 1.0, 0.0], &[2.0, 3.0, 4.0, 0.5], 2).expect("hit");
        assert!((r - 8.0).abs() < 1e-4);
    }

    #[test]
    fn v1_export_stays_v1_for_pure_v1_library() {
        let mut index = NativeIndex::new(0.85);
        index.insert(NativeEntry {
            signature: vec![1.0, 0.0],
            program: DiscreteProgram {
                init_idx: 0,
                transform_idx: 1,
                reduce_idx: 0,
                post_scale_idx: 0,
                offset: 0.0,
            },
            program_v2: None,
            program_v3: None,
        });
        let json = library_to_json(&index);
        assert!(!json.contains("format"));
        assert!(json.contains("\"program\":"));
    }

    #[test]
    fn v2_refuses_outside_space() {
        // target = x*y only when y > x else 0, per-point conditional on a
        // cross-field comparison — not expressible.
        let e1 = [1.0, 5.0, 9.0, 2.0]; // 5 + 0 = 5
        let e2 = [3.0, 4.0, 8.0, 1.0, 2.0, 7.0]; // 12 + 0 + 14 = 26
        let e3 = [6.0, 1.0]; // 0
        let examples: Vec<(&[f32], f32)> = vec![(&e1, 5.0), (&e2, 26.0), (&e3, 0.0)];
        assert!(synthesize_program_v2(&examples, 2, 1e-3).is_none());
    }

    #[test]
    fn v2_sources_render_in_all_languages() {
        let p = ProgramV2 {
            arity: 2,
            combine_idx: 3,
            guard_idx: 1,
            guard_threshold: 10.0,
            init_idx: 0,
            transform_idx: 0,
            reduce_idx: 0,
            post_scale_idx: 0,
            offset: 0.0,
        };
        let rust = program_source_v2("weighted", &p, Lang::Rust);
        assert!(rust.contains("fn weighted") && rust.contains("v > 10"));
        let py = program_source_v2("weighted", &p, Lang::Python);
        assert!(py.contains("def weighted") && py.contains("math.prod(pt)"));
        let js = program_source_v2("weighted", &p, Lang::JavaScript);
        assert!(js.contains("function weighted") && js.contains("continue"));
        let c = program_source_v2("weighted", &p, Lang::C);
        assert!(c.contains("float weighted") && c.contains("prod_fields"));
        let ts = program_source_v2("weighted", &p, Lang::TypeScript);
        assert!(ts.contains("points: number[][]"));
    }

    #[test]
    fn mined_thresholds_anchor_zero_and_dedupe() {
        let e1 = [3.0, 12.0, 10.0, 12.0];
        let examples: Vec<(&[f32], f32)> = vec![(&e1, 0.0)];
        let t = mine_thresholds(&examples);
        assert!(t.contains(&0.0));
        assert_eq!(t.iter().filter(|v| (**v - 12.0).abs() < 1e-6).count(), 1);
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

    // ------------------------------------------------------------- v3 tests

    #[test]
    fn v3_discovers_running_counter() {
        // y_t = t+1 regardless of input values: transform=const-1 + reduce=add
        // + output=state. Inputs include a negative and a zero so the
        // indicator transform (x>0) cannot masquerade as const-1.
        let inputs = [5.0, -2.0, 0.0, 9.0];
        let expected = [1.0, 2.0, 3.0, 4.0];
        let traces: Vec<(&[f32], &[f32])> = vec![(&inputs, &expected)];
        let r = synthesize_program_v3(&traces, 1, 1e-3).expect("running counter");
        let p = r.program;
        assert_eq!(p.update_transform_idx, 3, "const-1 transform");
        assert_eq!(p.update_reduce_idx, 0, "add reduce");
        assert_eq!(p.output_idx, 0, "output = state");
        assert_eq!(p.reset_guard_idx, 0, "no reset needed");
        assert_eq!(p.offset, 0.0);
        // Held-out replay.
        let held = [0.5, 0.5, -3.0, 100.0, 0.0, 7.0];
        let out = execute_program_v3(p, &held, 6);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn v3_discovers_running_max_with_mined_reset() {
        // Running max that restarts when a sentinel (-10) arrives. The reset
        // threshold must come out of the trace data (mine_thresholds), and
        // the separating values (-8 / -10) are NOT the anchored zero — a
        // hardcoded-zero guard would fire on the ordinary negatives too.
        let in1 = [3.0, -5.0, 7.0, -10.0, 2.0, -8.0, 4.0];
        let ex1 = [3.0, 3.0, 7.0, -10.0, 2.0, 2.0, 4.0];
        // Second trace also proves state restarts between traces: carrying
        // trace 1's final state (4.0) would give max(4,1)=4, not 1.
        let in2 = [1.0, 3.0, -10.0, 2.0, -8.0];
        let ex2 = [1.0, 3.0, -10.0, 2.0, 2.0];
        let traces: Vec<(&[f32], &[f32])> = vec![(&in1, &ex1), (&in2, &ex2)];
        let r = synthesize_program_v3(&traces, 1, 1e-3).expect("running max with reset");
        let p = r.program;
        assert_eq!(p.update_reduce_idx, 2, "max reduce");
        assert_ne!(p.reset_guard_idx, 0, "reset guard must be active");
        // The threshold is mined from the data: the only consistent
        // separators are -8 (v < t) and -10 (v == t); zero cannot work.
        assert!(
            (p.reset_threshold - -8.0).abs() < 1e-5 || (p.reset_threshold - -10.0).abs() < 1e-5,
            "threshold {} must be a mined data value",
            p.reset_threshold
        );
        let mined = mine_thresholds(&[(&in1[..], 0.0), (&in2[..], 0.0)]);
        assert!(
            mined.iter().any(|t| (t - p.reset_threshold).abs() < 1e-6),
            "threshold {} not in mined set {:?}",
            p.reset_threshold,
            mined
        );
        // Held-out replay.
        let held = [5.0, 2.0, -10.0, 1.0, 0.0];
        let out = execute_program_v3(p, &held, 5);
        assert_eq!(out, vec![5.0, 5.0, -10.0, 1.0, 1.0]);
    }

    #[test]
    fn v3_refuses_two_step_delay() {
        // y_t = x_{t-2} needs a 2-deep shift register; one f32 state cell
        // under {add,mul,max,min} folds cannot express it. Honest refusal.
        let in1 = [2.0, 9.0, 4.0, 7.0, 1.0];
        let ex1 = [0.0, 0.0, 2.0, 9.0, 4.0];
        let in2 = [5.0, 3.0, 8.0];
        let ex2 = [0.0, 0.0, 5.0];
        let traces: Vec<(&[f32], &[f32])> = vec![(&in1, &ex1), (&in2, &ex2)];
        assert!(synthesize_program_v3(&traces, 1, 1e-3).is_none());
    }

    #[test]
    fn v3_lift_matches_v2_exactly_across_grid() {
        // v2 is the exact special case of v3 (reset=never, output=state):
        // the final step of the v3 replay must equal the v2 fold for EVERY
        // v2 program, including guards and post-scales.
        let data = [3.0, -1.5, 4.0, 0.0, 9.0, 2.0, -2.0, 5.0]; // 4 points, arity 2
        for combine_idx in 0..N_COMBINES {
            for guard_idx in 0..N_GUARDS {
                for &guard_threshold in &[0.0f32, 2.0] {
                    for init_idx in 0..N_INITS {
                        for transform_idx in 0..N_TRANSFORMS {
                            for reduce_idx in 0..N_REDUCES {
                                for post_scale_idx in 0..N_POST_SCALES {
                                    let v2 = ProgramV2 {
                                        arity: 2,
                                        combine_idx,
                                        guard_idx,
                                        guard_threshold,
                                        init_idx,
                                        transform_idx,
                                        reduce_idx,
                                        post_scale_idx,
                                        offset: 0.25,
                                    };
                                    let a = execute_program_v2(v2, &data, 4);
                                    let b = execute_program_v3_final(ProgramV3::from_v2(v2), &data, 4);
                                    if a.is_finite() && b.is_finite() {
                                        assert!(
                                            (a - b).abs() < 1e-5,
                                            "v2/v3 divergence at c{combine_idx} g{guard_idx} t{guard_threshold} i{init_idx} tr{transform_idx} r{reduce_idx} p{post_scale_idx}: {a} vs {b}"
                                        );
                                    }
                                    // Empty trace mirrors v2's empty fold.
                                    let ae = execute_program_v2(v2, &[], 0);
                                    let be = execute_program_v3_final(ProgramV3::from_v2(v2), &[], 0);
                                    if ae.is_finite() && be.is_finite() {
                                        assert!((ae - be).abs() < 1e-5, "empty-trace divergence: {ae} vs {be}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn v3_stateless_output_matches_v2_combine_stage() {
        // A v3 program whose output ignores state (output = v) reproduces
        // the v2 combine stage per step on the same base fields, whatever
        // the state machinery does.
        let p = ProgramV3 {
            arity: 2,
            combine_idx: 4, // f0 - f1
            guard_idx: 0,
            guard_threshold: 0.0,
            reset_guard_idx: 3, // arbitrary state machinery — must not matter
            reset_threshold: 1.0,
            state_init_idx: 1,
            update_transform_idx: 1,
            update_reduce_idx: 1,
            post_scale_idx: 0,
            output_idx: 1, // y = v
            offset: 0.5,
        };
        let data = [5.0, 2.0, 1.0, 4.0, -3.0, -3.0]; // (5,2),(1,4),(-3,-3)
        let out = execute_program_v3(p, &data, 3);
        for (i, y) in out.iter().enumerate() {
            let fields = &data[i * 2..(i + 1) * 2];
            let expected = apply_combine(fields, p.combine_idx) + p.offset;
            assert!((y - expected).abs() < 1e-6, "step {i}: {y} vs {expected}");
        }
    }

    #[test]
    fn v3_library_round_trips_and_v1_v2_loaders_reject_it() {
        let mut index = NativeIndex::new(0.85);
        // v1 entry (sum), v2 entry (dot product), v3 entry (running max with
        // reset) — the v3 entry forces format 3 and the lower entries lift.
        index.insert(NativeEntry {
            signature: vec![1.0, 0.0, 0.0],
            program: DiscreteProgram { init_idx: 0, transform_idx: 0, reduce_idx: 0, post_scale_idx: 0, offset: 0.0 },
            program_v2: None,
            program_v3: None,
        });
        index.insert(NativeEntry {
            signature: vec![0.0, 1.0, 0.0],
            program: DiscreteProgram { init_idx: 0, transform_idx: 0, reduce_idx: 0, post_scale_idx: 0, offset: 0.0 },
            program_v2: Some(ProgramV2 {
                arity: 2, combine_idx: 3, guard_idx: 0, guard_threshold: 0.0,
                init_idx: 0, transform_idx: 0, reduce_idx: 0, post_scale_idx: 0, offset: 0.0,
            }),
            program_v3: None,
        });
        let v3 = ProgramV3 {
            arity: 1, combine_idx: 0, guard_idx: 0, guard_threshold: 0.0,
            reset_guard_idx: 2, reset_threshold: -8.0, state_init_idx: 2,
            update_transform_idx: 0, update_reduce_idx: 2, post_scale_idx: 0,
            output_idx: 0, offset: 0.0,
        };
        index.insert(NativeEntry {
            signature: vec![0.0, 0.0, 1.0],
            program: DiscreteProgram { init_idx: 2, transform_idx: 0, reduce_idx: 2, post_scale_idx: 0, offset: 0.0 },
            program_v2: None,
            program_v3: Some(v3),
        });
        let json = library_to_json(&index);
        assert!(json.contains("\"format\": 3"));
        assert!(json.contains("program_v3"));
        // Entry-wise lifting: NO v1/v2 program keys remain, so v1 loaders
        // (which require `"program"`) and v2 loaders (which require
        // `"program_v2"`) both fail closed on this file. This is exactly
        // what the old parsers grep for:
        assert!(!json.contains("\"program\":"), "v3 export must not carry a v1 program key");
        assert!(!json.contains("\"program_v2\":"), "v3 export must not carry a v2 program key");
        assert!(find_object_value(&json, "\"program\"").is_none());
        assert!(find_object_value(&json, "\"program_v2\"").is_none());

        // New loader round-trips all three entries.
        let (_thr, parsed) = load_library_json(&json).expect("v3 round trip");
        assert_eq!(parsed.entries.len(), 3);
        // Lifted v1 entry still sums (consult = final fold output).
        let r = consult_native(&parsed, &[1.0, 0.0, 0.0], &[1.0, 2.0, 3.0, 4.0], 4).expect("hit");
        assert!((r - 10.0).abs() < 1e-4);
        // Lifted v2 entry still answers the dot product.
        let r = consult_native(&parsed, &[0.0, 1.0, 0.0], &[2.0, 3.0, 4.0, 0.5], 2).expect("hit");
        assert!((r - 8.0).abs() < 1e-4);
        // v3 entry replays per-step through consult_native_v3...
        let steps = consult_native_v3(&parsed, &[0.0, 0.0, 1.0], &[5.0, 2.0, -10.0, 1.0], 1, 4)
            .expect("v3 hit");
        assert_eq!(steps, vec![5.0, 5.0, -10.0, 1.0]);
        // ...refuses on arity mismatch...
        assert!(consult_native_v3(&parsed, &[0.0, 0.0, 1.0], &[5.0, 2.0], 2, 1).is_none());
        // ...and the scalar consult answers with the final step.
        let r = consult_native(&parsed, &[0.0, 0.0, 1.0], &[5.0, 2.0, -10.0, 1.0], 4).expect("hit");
        assert!((r - 1.0).abs() < 1e-4);
        // Re-export stays format 3 (stable round trip).
        assert!(library_to_json(&parsed).contains("\"format\": 3"));
    }

    #[test]
    fn v3_export_demotes_to_v2_when_no_state_used() {
        // A v3 program with reset=never and output=state IS a v2 fold; the
        // export must stay at the lowest loadable format.
        let v3 = ProgramV3 {
            arity: 2, combine_idx: 3, guard_idx: 1, guard_threshold: 0.5,
            reset_guard_idx: 0, reset_threshold: 0.0, state_init_idx: 0,
            update_transform_idx: 0, update_reduce_idx: 0, post_scale_idx: 0,
            output_idx: 0, offset: 0.0,
        };
        assert!(v3.is_v2());
        let mut index = NativeIndex::new(0.85);
        index.insert(NativeEntry {
            signature: vec![1.0, 0.0],
            program: DiscreteProgram { init_idx: 0, transform_idx: 0, reduce_idx: 0, post_scale_idx: 0, offset: 0.0 },
            program_v2: v3.to_v2(),
            program_v3: Some(v3),
        });
        let json = library_to_json(&index);
        assert!(json.contains("\"format\": 2"));
        assert!(json.contains("program_v2"));
        assert!(!json.contains("program_v3"));
    }

    #[test]
    fn v3_sources_render_rust_python_typescript() {
        // Running max with reset — the canonical stateful skill.
        let p = ProgramV3 {
            arity: 1, combine_idx: 0, guard_idx: 0, guard_threshold: 0.0,
            reset_guard_idx: 2, reset_threshold: -8.0, state_init_idx: 2,
            update_transform_idx: 0, update_reduce_idx: 2, post_scale_idx: 0,
            output_idx: 0, offset: 0.0,
        };
        let rust = program_source_v3("running_max", &p, Lang::Rust);
        assert!(rust.contains("fn running_max"), "{rust}");
        assert!(rust.contains("let mut s"), "{rust}");
        assert!(rust.contains("v < -8"), "{rust}");
        assert!(rust.contains("s = s.max(v)"), "{rust}");
        assert!(rust.contains("out.push(s)"), "{rust}");
        let py = program_source_v3("running_max", &p, Lang::Python);
        assert!(py.contains("def running_max"), "{py}");
        assert!(py.contains("if v < -8:"), "{py}");
        assert!(py.contains("s = max(s, v)"), "{py}");
        assert!(py.contains("out.append(s)"), "{py}");
        let ts = program_source_v3("running_max", &p, Lang::TypeScript);
        assert!(ts.contains("points: number[][]"), "{ts}");
        assert!(ts.contains("): number[]"), "{ts}");
        assert!(ts.contains("let s = -20.0"), "{ts}");
        assert!(ts.contains("s = Math.max(s, v)"), "{ts}");
        // JS and C render too (smoke).
        let js = program_source_v3("running_max", &p, Lang::JavaScript);
        assert!(js.contains("function running_max(points)"), "{js}");
        let c = program_source_v3("running_max", &p, Lang::C);
        assert!(c.contains("void running_max"), "{c}");
        assert!(c.contains("out[i] = s"), "{c}");
    }
}
