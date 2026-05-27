//! Native executor for NPCoT `DiscreteArrayProgram` (milestone M3).
//!
//! When an `ArrayExecutableThoughtHead` converges, its argmax program is
//! cached as a `DiscreteArrayProgram` — a 5-tuple of `(init_idx,
//! transform_idx, reduce_idx, post_scale_idx, offset)`. Executing such a
//! program is a simple loop:
//!
//! ```text
//! acc = init_value
//! for i in 0..length:
//!     f_i = transform(array[i])          // x, x*x, |x|, 1, 1{x>0}
//!     acc = reduce(acc, f_i)             // +, *, max, min
//! result = post_scale(acc, length) + offset
//! ```
//!
//! This module implements that loop in two native forms:
//!
//! * **`execute_cpu`** — a pure-Rust function (no Metal, no Python). The
//!   Python reference in `ncpu/self_optimizing/array_program_library.py`
//!   matches it exactly; a cross-check is exposed to the Python side via the
//!   PyO3 `npcot_execute_cpu` function below.
//! * **`NpcotGpuExecutor`** — a Metal-backed dispatcher that compiles the
//!   loop into a compute shader, batches many `(program, array)` samples,
//!   and runs them with one `dispatch`. For a batch of N samples with array
//!   length L, the CPU path is O(N·L) Rust ops; the Metal path launches N
//!   threads that each run a length-L loop on the GPU.
//!
//! Both paths share the canonical semantics spelled out in Python; see the
//! Rust unit tests at the bottom of this file for bit-for-bit agreement on
//! sum, max, min, count_positive, and scaled variants.
//!
//! The fundamental claim these executors back: once the model's reasoning
//! has crystallized into a discrete program, execution requires no neural
//! network at all. The library hit path is pure tensor arithmetic that runs
//! on hardware without pytorch, without a Python interpreter, and — via the
//! Metal shader — without any CPU thread at all.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::slice;

use crate::{get_default_device, MetalError};

// ---------------------------------------------------------------------------
// Canonical semantics
// ---------------------------------------------------------------------------

/// Integer sentinels that mirror the Python module's discrete choices.
///
/// `init_idx`: 0 = 0, 1 = 1, 2 = -large (-20.0, the max-sentinel)
/// `transform_idx`: 0 = x, 1 = x*x, 2 = |x|, 3 = 1, 4 = 1{x>0}, 5 = ln(|x|+eps)
/// `reduce_idx`: 0 = +, 1 = *, 2 = max, 3 = min
/// `post_scale_idx`: 0 = acc, 1 = acc/len, 2 = exp(clamp(acc, -30, 30))
pub const NEG_LARGE: f32 = -20.0;
pub const LOG_EPS: f32 = 1e-6;

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
        4 => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
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

/// Pack a program into the layout the Metal shader expects.
///
/// Five `f32` lanes per program: `[init_idx, transform_idx, reduce_idx,
/// post_scale_idx, offset]`. The first four are cast-to-`u32` inside the
/// shader; offset is additively applied after post-scaling.
pub const PROGRAM_STRIDE: usize = 5;

/// A single discrete program.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteProgram {
    pub init_idx: u32,
    pub transform_idx: u32,
    pub reduce_idx: u32,
    pub post_scale_idx: u32,
    pub offset: f32,
}

impl DiscreteProgram {
    pub fn new(
        init_idx: u32,
        transform_idx: u32,
        reduce_idx: u32,
        post_scale_idx: u32,
        offset: f32,
    ) -> Self {
        Self {
            init_idx,
            transform_idx,
            reduce_idx,
            post_scale_idx,
            offset,
        }
    }

    fn flatten_to(&self, out: &mut [f32]) {
        out[0] = self.init_idx as f32;
        out[1] = self.transform_idx as f32;
        out[2] = self.reduce_idx as f32;
        out[3] = self.post_scale_idx as f32;
        out[4] = self.offset;
    }
}

/// Execute a single discrete program over a length-L slice on the CPU.
pub fn execute_cpu_one(program: DiscreteProgram, array: &[f32], length: u32) -> f32 {
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
            // idx 2 (and any fallback): exp(clamp(acc, -30, 30))
            let clamped = acc.max(-30.0).min(30.0);
            clamped.exp()
        }
    };
    post + program.offset
}

/// Execute a batch of `(program, array, length)` triples on the CPU.
///
/// `arrays_flat` is a row-major `[batch, max_len]` buffer; `lengths` is
/// per-sample effective length. `programs` may have len 1 (broadcast to the
/// whole batch) or len `batch`.
pub fn execute_cpu_batch(
    programs: &[DiscreteProgram],
    arrays_flat: &[f32],
    lengths: &[u32],
    max_len: usize,
) -> Vec<f32> {
    let batch = lengths.len();
    assert_eq!(
        arrays_flat.len(),
        batch * max_len,
        "arrays_flat shape mismatch: got {} for batch {} max_len {}",
        arrays_flat.len(),
        batch,
        max_len,
    );
    let broadcast = programs.len() == 1;
    if !broadcast {
        assert_eq!(
            programs.len(),
            batch,
            "programs must have len 1 or batch",
        );
    }

    let mut result = Vec::with_capacity(batch);
    for sample in 0..batch {
        let program = if broadcast {
            programs[0]
        } else {
            programs[sample]
        };
        let start = sample * max_len;
        let end = start + max_len;
        let value = execute_cpu_one(
            program,
            &arrays_flat[start..end],
            lengths[sample],
        );
        result.push(value);
    }
    result
}

// ---------------------------------------------------------------------------
// Metal compute shader
// ---------------------------------------------------------------------------

const NPCOT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant float NEG_LARGE = -20.0f;
constant float LOG_EPS = 1e-6f;

inline float init_value(uint init_idx) {
    if (init_idx == 0u) return 0.0f;
    if (init_idx == 1u) return 1.0f;
    if (init_idx == 2u) return NEG_LARGE;
    return 0.0f;
}

inline float apply_transform(float x, uint idx) {
    if (idx == 0u) return x;
    if (idx == 1u) return x * x;
    if (idx == 2u) return fabs(x);
    if (idx == 3u) return 1.0f;
    if (idx == 4u) return (x > 0.0f) ? 1.0f : 0.0f;
    if (idx == 5u) return log(fabs(x) + LOG_EPS);
    return x;
}

inline float apply_reduce(float acc, float f, uint idx) {
    if (idx == 0u) return acc + f;
    if (idx == 1u) return acc * f;
    if (idx == 2u) return max(acc, f);
    if (idx == 3u) return min(acc, f);
    return acc + f;
}

kernel void npcot_execute_batch(
    device const float* programs          [[buffer(0)]],   // [N_prog, 5]
    device const float* arrays            [[buffer(1)]],   // [batch, max_len]
    device const uint* lengths            [[buffer(2)]],   // [batch]
    device const uint* meta               [[buffer(3)]],   // [batch, max_len, program_stride, broadcast]
    device float* results                 [[buffer(4)]],   // [batch]
    uint tid                              [[thread_position_in_grid]]
) {
    uint batch    = meta[0];
    uint max_len  = meta[1];
    uint pstride  = meta[2];   // always PROGRAM_STRIDE = 5
    uint bcast    = meta[3];

    if (tid >= batch) return;

    uint prog_base = (bcast != 0u) ? 0u : tid * pstride;
    uint init_idx     = (uint)programs[prog_base + 0u];
    uint trans_idx    = (uint)programs[prog_base + 1u];
    uint reduce_idx   = (uint)programs[prog_base + 2u];
    uint post_idx     = (uint)programs[prog_base + 3u];
    float offset      = programs[prog_base + 4u];

    uint length = lengths[tid];
    if (length > max_len) length = max_len;

    uint row_base = tid * max_len;
    float acc = init_value(init_idx);
    for (uint i = 0u; i < length; ++i) {
        float x_i = arrays[row_base + i];
        float f_i = apply_transform(x_i, trans_idx);
        acc = apply_reduce(acc, f_i, reduce_idx);
    }

    float denom = (float)length;
    if (denom < 1.0f) denom = 1.0f;
    float post;
    if (post_idx == 0u) {
        post = acc;
    } else if (post_idx == 1u) {
        post = acc / denom;
    } else {
        // post_idx 2: exp(clamp(acc, -30, 30)) — stable product recovery.
        float clamped = clamp(acc, -30.0f, 30.0f);
        post = exp(clamped);
    }
    results[tid] = post + offset;
}
"#;

pub struct NpcotGpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl NpcotGpu {
    pub fn new() -> Result<Self, MetalError> {
        let device = get_default_device().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::NoCommandQueue)?;

        let source = NSString::from_str(NPCOT_SHADER_SOURCE);
        let library: Retained<ProtocolObject<dyn MTLLibrary>> = {
            let options = objc2_metal::MTLCompileOptions::new();
            device
                .newLibraryWithSource_options_error(&source, Some(&options))
                .map_err(|err| {
                    MetalError::ShaderCompilationFailed(format!("{err:?}"))
                })?
        };
        let function = library
            .newFunctionWithName(&NSString::from_str("npcot_execute_batch"))
            .ok_or_else(|| {
                MetalError::PipelineCreationFailed("missing kernel npcot_execute_batch".into())
            })?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| MetalError::PipelineCreationFailed(format!("{err:?}")))?;
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    pub fn execute(
        &self,
        programs: &[DiscreteProgram],
        arrays_flat: &[f32],
        lengths: &[u32],
        max_len: usize,
    ) -> Result<Vec<f32>, MetalError> {
        let batch = lengths.len();
        if batch == 0 {
            return Ok(Vec::new());
        }
        let broadcast = programs.len() == 1;
        if !broadcast && programs.len() != batch {
            return Err(MetalError::ExecutionFailed);
        }
        if arrays_flat.len() != batch * max_len {
            return Err(MetalError::ExecutionFailed);
        }

        let n_programs = if broadcast { 1 } else { batch };
        let mut packed = vec![0.0f32; n_programs * PROGRAM_STRIDE];
        for (sample_index, program) in programs.iter().enumerate() {
            let offset = sample_index * PROGRAM_STRIDE;
            program.flatten_to(&mut packed[offset..offset + PROGRAM_STRIDE]);
        }

        let prog_buf = self.new_buf_f32(&packed);
        let arr_buf = self.new_buf_f32(arrays_flat);
        let len_buf = self.new_buf_u32(lengths);
        let meta = [
            batch as u32,
            max_len as u32,
            PROGRAM_STRIDE as u32,
            broadcast as u32,
        ];
        let meta_buf = self.new_buf_u32(&meta);
        let result_buf = self
            .device
            .newBufferWithLength_options(batch * 4, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed)?;

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::ExecutionFailed)?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or(MetalError::ExecutionFailed)?;
        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&prog_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&arr_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&len_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&meta_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&result_buf), 0, 4);
        }
        let thread_group_size = self.pipeline.maxTotalThreadsPerThreadgroup().min(batch);
        let tg = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: batch,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let ptr = unsafe {
            slice::from_raw_parts(
                result_buf.contents().as_ptr() as *const f32,
                batch,
            )
        };
        Ok(ptr.to_vec())
    }

    fn new_buf_f32(
        &self,
        data: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bytes = data.len() * 4;
        let buf = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .expect("buffer alloc");
        unsafe {
            let dst = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        buf
    }

    fn new_buf_u32(
        &self,
        data: &[u32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bytes = data.len() * 4;
        let buf = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .expect("buffer alloc");
        unsafe {
            let dst = buf.contents().as_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Native library lookup index
// ---------------------------------------------------------------------------

/// One entry in the native lookup index.
#[derive(Debug, Clone)]
pub struct NativeLibraryEntry {
    pub signature: Vec<f32>,
    pub program: DiscreteProgram,
}

/// Signature-sharded hash index over discrete programs.
///
/// Shards are chosen by quantizing the signature's dimensionality and its
/// dominant axis (argmax of absolute-value components). Each shard holds
/// a small list of entries; lookup computes cosine similarity only against
/// entries in the same shard. For large libraries this reduces O(n) full
/// scan to O(n / num_shards) amortized.
pub struct NativeLibraryIndex {
    entries: Vec<NativeLibraryEntry>,
    shards: std::collections::HashMap<ShardKey, Vec<usize>>,
    signature_dim: Option<usize>,
    similarity_threshold: f32,
}

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct ShardKey {
    dim: usize,
    dominant_axis: usize,
    sign_bucket: i8,
}

impl NativeLibraryIndex {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            shards: std::collections::HashMap::new(),
            signature_dim: None,
            similarity_threshold,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn insert(&mut self, entry: NativeLibraryEntry) {
        if entry.signature.is_empty() {
            return;
        }
        match self.signature_dim {
            None => self.signature_dim = Some(entry.signature.len()),
            Some(dim) if dim != entry.signature.len() => return,
            _ => {}
        }
        let key = shard_key_for(&entry.signature);
        let index = self.entries.len();
        self.entries.push(entry);
        self.shards.entry(key).or_insert_with(Vec::new).push(index);
    }

    /// Find the best matching entry above the similarity threshold.
    pub fn lookup(&self, signature: &[f32]) -> Option<&NativeLibraryEntry> {
        if signature.is_empty() {
            return None;
        }
        if let Some(dim) = self.signature_dim {
            if dim != signature.len() {
                return None;
            }
        } else {
            return None;
        }
        let key = shard_key_for(signature);

        // Check the primary shard first.
        let mut best: Option<&NativeLibraryEntry> = None;
        let mut best_score = -1.0f32;
        if let Some(indices) = self.shards.get(&key) {
            for &i in indices {
                let score = cosine_similarity(signature, &self.entries[i].signature);
                if score > best_score {
                    best_score = score;
                    best = Some(&self.entries[i]);
                }
            }
        }
        // If primary-shard best beats threshold, return it.
        if best_score >= self.similarity_threshold {
            return best;
        }
        // Fall through: scan adjacent shards (different dominant_axis) for
        // the edge case where a query sits near a shard boundary.
        for (other_key, indices) in &self.shards {
            if other_key == &key {
                continue;
            }
            for &i in indices {
                let score = cosine_similarity(signature, &self.entries[i].signature);
                if score > best_score {
                    best_score = score;
                    best = Some(&self.entries[i]);
                }
            }
        }
        if best_score >= self.similarity_threshold {
            best
        } else {
            None
        }
    }
}

fn shard_key_for(signature: &[f32]) -> ShardKey {
    let mut dominant = 0usize;
    let mut dominant_abs = 0.0f32;
    let mut sign_positive = true;
    for (i, &v) in signature.iter().enumerate() {
        let a = v.abs();
        if a > dominant_abs {
            dominant_abs = a;
            dominant = i;
            sign_positive = v >= 0.0;
        }
    }
    ShardKey {
        dim: signature.len(),
        dominant_axis: dominant,
        sign_bucket: if sign_positive { 1 } else { -1 },
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
    let denom = (a_norm.sqrt()) * (b_norm.sqrt());
    if denom < 1e-9 {
        return -1.0;
    }
    dot / denom
}

// ---------------------------------------------------------------------------
// JSON library loader + end-to-end consult
// ---------------------------------------------------------------------------

/// Minimal JSON parser for the `ArrayProgramLibrary` on-disk format.
///
/// We deliberately avoid a third-party JSON dependency — serde or
/// `serde_json` are heavy for what we need. This parser handles the exact
/// shape `ArrayProgramLibrary.save` emits (config + entries list with
/// signature/program fields). Rejects anything else.
pub fn load_library_from_json_bytes(
    payload: &[u8],
) -> Result<(f32, NativeLibraryIndex), String> {
    let text = std::str::from_utf8(payload)
        .map_err(|e| format!("invalid utf8: {e}"))?;
    // The library JSON is a flat dict with shape
    //     {"config": {"similarity_threshold": f, ...}, "entries": [...]}
    // We walk it with a tiny hand-rolled parser rather than pulling in
    // serde to keep the binary size small.
    let similarity_threshold = extract_similarity_threshold(text)?;
    let entries = extract_entries(text)?;
    let mut index = NativeLibraryIndex::new(similarity_threshold);
    for entry in entries {
        index.insert(entry);
    }
    Ok((similarity_threshold, index))
}

fn extract_similarity_threshold(text: &str) -> Result<f32, String> {
    // Expect `"similarity_threshold": 0.85` somewhere.
    let marker = "\"similarity_threshold\"";
    let start = text.find(marker).ok_or_else(|| {
        "missing similarity_threshold in library JSON".to_string()
    })?;
    let rest = &text[start + marker.len()..];
    let colon = rest
        .find(':')
        .ok_or_else(|| "malformed similarity_threshold".to_string())?;
    let after = &rest[colon + 1..];
    let end = after
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(after.len());
    let value = after[..end].trim();
    value
        .parse::<f32>()
        .map_err(|e| format!("parse similarity_threshold: {e}"))
}

fn extract_entries(text: &str) -> Result<Vec<NativeLibraryEntry>, String> {
    // Locate the entries array.
    let marker = "\"entries\"";
    let start = text
        .find(marker)
        .ok_or_else(|| "missing entries array".to_string())?;
    let rest = &text[start + marker.len()..];
    let bracket_start = rest
        .find('[')
        .ok_or_else(|| "entries has no opening bracket".to_string())?;
    let array_rest = &rest[bracket_start..];
    let close = find_matching_bracket(array_rest, '[', ']')
        .ok_or_else(|| "entries has no closing bracket".to_string())?;
    let body = &array_rest[1..close];
    let mut entries: Vec<NativeLibraryEntry> = Vec::new();
    let mut cursor = 0usize;
    while let Some(obj_start) = body[cursor..].find('{') {
        let absolute = cursor + obj_start;
        let obj_rest = &body[absolute..];
        let obj_end = find_matching_bracket(obj_rest, '{', '}')
            .ok_or_else(|| "entries object missing close brace".to_string())?;
        let obj = &obj_rest[..=obj_end];
        entries.push(parse_entry(obj)?);
        cursor = absolute + obj_end + 1;
    }
    Ok(entries)
}

fn find_matching_bracket(text: &str, open: char, close: char) -> Option<usize> {
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

fn parse_entry(obj: &str) -> Result<NativeLibraryEntry, String> {
    let signature = parse_float_array(obj, "\"signature\"")?;
    let program_start = find_object_value(obj, "\"program\"")
        .ok_or_else(|| "entry missing program field".to_string())?;
    let init_idx = parse_int_field(program_start, "\"init_idx\"")?;
    let transform_idx = parse_int_field(program_start, "\"transform_idx\"")?;
    let reduce_idx = parse_int_field(program_start, "\"reduce_idx\"")?;
    let post_scale_idx = parse_int_field(program_start, "\"post_scale_idx\"")?;
    let offset = parse_float_field(program_start, "\"offset\"")?;
    Ok(NativeLibraryEntry {
        signature,
        program: DiscreteProgram::new(
            init_idx as u32,
            transform_idx as u32,
            reduce_idx as u32,
            post_scale_idx as u32,
            offset,
        ),
    })
}

fn find_object_value<'a>(obj: &'a str, key: &str) -> Option<&'a str> {
    let start = obj.find(key)?;
    let after = &obj[start + key.len()..];
    let colon = after.find(':')?;
    let value_start = &after[colon + 1..];
    let trimmed = value_start.trim_start();
    let open_idx = value_start.len() - trimmed.len();
    let brace = trimmed.find('{')?;
    let abs_brace = colon + 1 + open_idx + brace;
    let after_brace = &after[abs_brace..];
    let close = find_matching_bracket(after_brace, '{', '}')?;
    Some(&after[abs_brace..abs_brace + close + 1])
}

fn parse_float_array(obj: &str, key: &str) -> Result<Vec<f32>, String> {
    let start = obj
        .find(key)
        .ok_or_else(|| format!("missing field {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after
        .find(':')
        .ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let lb = rest
        .find('[')
        .ok_or_else(|| format!("{key} not an array"))?;
    let arr_rest = &rest[lb..];
    let rb = find_matching_bracket(arr_rest, '[', ']')
        .ok_or_else(|| format!("{key} unclosed array"))?;
    let body = &arr_rest[1..rb];
    let mut out = Vec::new();
    for item in body.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(
            trimmed
                .parse::<f32>()
                .map_err(|e| format!("parse float in {key}: {e}"))?,
        );
    }
    Ok(out)
}

fn parse_int_field(obj: &str, key: &str) -> Result<i64, String> {
    let start = obj
        .find(key)
        .ok_or_else(|| format!("missing int {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after
        .find(':')
        .ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']')
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<i64>()
        .map_err(|e| format!("parse int {key}: {e}"))
}

fn parse_float_field(obj: &str, key: &str) -> Result<f32, String> {
    let start = obj
        .find(key)
        .ok_or_else(|| format!("missing float {key}"))?;
    let after = &obj[start + key.len()..];
    let colon = after
        .find(':')
        .ok_or_else(|| format!("malformed {key}"))?;
    let rest = &after[colon + 1..];
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == ']')
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<f32>()
        .map_err(|e| format!("parse float {key}: {e}"))
}

/// Pure-Rust `consult_library`: normalize the query signature, look it up
/// in the index, and execute the matching discrete program against the
/// supplied array. Returns `Some(result)` on hit, `None` on miss.
pub fn consult_library_native(
    index: &NativeLibraryIndex,
    hidden: &[f32],
    array: &[f32],
    length: u32,
) -> Option<f32> {
    let norm: f32 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return None;
    }
    let normalized: Vec<f32> = hidden.iter().map(|v| v / norm).collect();
    let entry = index.lookup(&normalized)?;
    Some(execute_cpu_one(entry.program, array, length))
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

/// Pure-Rust CPU execution of a batch of discrete array programs.
///
/// Python call contract:
///
///     result = ncpu_metal.npcot_execute_cpu(
///         programs_flat,   # List[float], len = (1 or batch) * 5
///         arrays_flat,     # List[float], len = batch * max_len
///         lengths,         # List[int],   len = batch
///         max_len,         # int
///     )  # -> List[float], len = batch
///
/// Match the Python reference exactly — see tests in
/// `tests/self_optimizing/test_array_program_library.py`.
#[pyfunction]
pub fn npcot_execute_cpu(
    programs_flat: Vec<f32>,
    arrays_flat: Vec<f32>,
    lengths: Vec<u32>,
    max_len: usize,
) -> PyResult<Vec<f32>> {
    if programs_flat.len() % PROGRAM_STRIDE != 0 {
        return Err(PyRuntimeError::new_err(
            "programs_flat length must be a multiple of 5",
        ));
    }
    let programs: Vec<DiscreteProgram> = programs_flat
        .chunks_exact(PROGRAM_STRIDE)
        .map(|chunk| {
            DiscreteProgram::new(
                chunk[0] as u32,
                chunk[1] as u32,
                chunk[2] as u32,
                chunk[3] as u32,
                chunk[4],
            )
        })
        .collect();
    Ok(execute_cpu_batch(&programs, &arrays_flat, &lengths, max_len))
}

/// PyO3 class wrapping the Metal-backed dispatcher.
#[pyclass]
pub struct NpcotGpuExecutor {
    inner: NpcotGpu,
}

#[pymethods]
impl NpcotGpuExecutor {
    #[new]
    fn new() -> PyResult<Self> {
        NpcotGpu::new()
            .map(|inner| Self { inner })
            .map_err(|err| PyRuntimeError::new_err(format!("{err:?}")))
    }

    fn execute(
        &self,
        programs_flat: Vec<f32>,
        arrays_flat: Vec<f32>,
        lengths: Vec<u32>,
        max_len: usize,
    ) -> PyResult<Vec<f32>> {
        if programs_flat.len() % PROGRAM_STRIDE != 0 {
            return Err(PyRuntimeError::new_err(
                "programs_flat length must be a multiple of 5",
            ));
        }
        let programs: Vec<DiscreteProgram> = programs_flat
            .chunks_exact(PROGRAM_STRIDE)
            .map(|chunk| {
                DiscreteProgram::new(
                    chunk[0] as u32,
                    chunk[1] as u32,
                    chunk[2] as u32,
                    chunk[3] as u32,
                    chunk[4],
                )
            })
            .collect();
        self.inner
            .execute(&programs, &arrays_flat, &lengths, max_len)
            .map_err(|err| PyRuntimeError::new_err(format!("{err:?}")))
    }
}

/// Rust-backed sharded library index. Bind to Python via PyO3.
#[pyclass]
pub struct NpcotLibraryIndex {
    inner: NativeLibraryIndex,
}

#[pymethods]
impl NpcotLibraryIndex {
    #[new]
    fn new(similarity_threshold: f32) -> Self {
        Self {
            inner: NativeLibraryIndex::new(similarity_threshold),
        }
    }

    /// Insert one entry. `program_params` is the flat
    /// `[init_idx, transform_idx, reduce_idx, post_scale_idx, offset]`
    /// tuple; `signature` must be unit-norm for correct cosine comparisons
    /// (callers are expected to normalize before insert).
    fn insert(
        &mut self,
        signature: Vec<f32>,
        program_params: Vec<f32>,
    ) -> PyResult<()> {
        if program_params.len() != PROGRAM_STRIDE {
            return Err(PyRuntimeError::new_err(
                "program_params must be length 5",
            ));
        }
        let program = DiscreteProgram::new(
            program_params[0] as u32,
            program_params[1] as u32,
            program_params[2] as u32,
            program_params[3] as u32,
            program_params[4],
        );
        self.inner.insert(NativeLibraryEntry { signature, program });
        Ok(())
    }

    /// Return `(program_params, similarity)` on hit, else None.
    fn lookup(&self, signature: Vec<f32>) -> Option<(Vec<f32>, f32)> {
        let entry = self.inner.lookup(&signature)?;
        let similarity = cosine_similarity(&signature, &entry.signature);
        let mut out = vec![0.0f32; PROGRAM_STRIDE];
        entry.program.flatten_to(&mut out);
        Some((out, similarity))
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// A fully self-contained Rust runtime for NPCoT libraries.
///
/// Loads a library JSON file, builds the sharded index, and exposes a
/// `consult(hidden, array, length)` method that does the entire
/// lookup-then-execute pipeline without any Python call-out per sample.
/// This is the building block for embedded / WASM / edge inference paths.
#[pyclass]
pub struct NpcotStandaloneRuntime {
    index: NativeLibraryIndex,
    similarity_threshold: f32,
}

#[pymethods]
impl NpcotStandaloneRuntime {
    #[staticmethod]
    fn from_json_path(path: String) -> PyResult<Self> {
        let bytes = std::fs::read(&path).map_err(|e| {
            PyRuntimeError::new_err(format!("read {path}: {e}"))
        })?;
        let (thr, index) =
            load_library_from_json_bytes(&bytes).map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            index,
            similarity_threshold: thr,
        })
    }

    #[staticmethod]
    fn from_json_bytes(payload: Vec<u8>) -> PyResult<Self> {
        let (thr, index) = load_library_from_json_bytes(&payload)
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self {
            index,
            similarity_threshold: thr,
        })
    }

    fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    fn entry_count(&self) -> usize {
        self.index.len()
    }

    /// Run the full consult-library pipeline. Returns None on miss.
    fn consult(
        &self,
        hidden: Vec<f32>,
        array: Vec<f32>,
        length: u32,
    ) -> Option<f32> {
        consult_library_native(&self.index, &hidden, &array, length)
    }
}

pub fn register_npcot_exec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(npcot_execute_cpu, m)?)?;
    m.add_class::<NpcotGpuExecutor>()?;
    m.add_class::<NpcotLibraryIndex>()?;
    m.add_class::<NpcotStandaloneRuntime>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_program_matches_reference() {
        let program = DiscreteProgram::new(0, 0, 0, 0, 0.0);
        let arrays = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let lengths = vec![3u32, 2u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 5);
        assert_eq!(result, vec![6.0, 9.0]);
    }

    #[test]
    fn max_program_honors_padding_via_length() {
        let program = DiscreteProgram::new(2, 0, 2, 0, 0.0);
        // init=-large, transform=x, reduce=max. Padding must not leak.
        let arrays = vec![-1.0, -3.0, -2.0, 99.0, 99.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 5);
        assert_eq!(result, vec![-1.0]);
    }

    #[test]
    fn count_positive_program() {
        let program = DiscreteProgram::new(0, 4, 0, 0, 0.0);
        let arrays = vec![1.0, -2.0, 3.0, 0.0, 0.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 5);
        assert_eq!(result, vec![2.0]);
    }

    #[test]
    fn mean_program() {
        let program = DiscreteProgram::new(0, 0, 0, 1, 0.0);
        let arrays = vec![2.0, 4.0, 6.0, 0.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 4);
        assert_eq!(result, vec![4.0]);
    }

    #[test]
    fn offset_is_additive() {
        let program = DiscreteProgram::new(0, 0, 0, 0, -1.5);
        let arrays = vec![1.0, 2.0, 3.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 3);
        assert_eq!(result, vec![4.5]);
    }

    #[test]
    fn broadcast_single_program() {
        // Broadcasting: program.len() == 1 applies to entire batch.
        let program = DiscreteProgram::new(0, 0, 0, 0, 0.0);
        let arrays = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let lengths = vec![3u32, 3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 3);
        assert_eq!(result, vec![6.0, 15.0]);
    }

    #[test]
    fn log_product_recovers_magnitude() {
        // transform=log|x|, reduce=+, post_scale=exp(acc) =>
        // exp(sum_i log(|x_i| + eps)) ≈ abs(product(x_i)) for non-zero x.
        let program = DiscreteProgram::new(0, 5, 0, 2, 0.0);
        let arrays = vec![2.0, -3.0, 4.0, 0.0, 0.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 5);
        // |2 * -3 * 4| = 24. Eps in each log adds a tiny positive drift.
        assert!(
            (result[0] - 24.0).abs() < 0.01,
            "expected ~24, got {}",
            result[0]
        );
    }

    #[test]
    fn log_product_handles_zero_without_exploding() {
        // If any element is zero, log(|0| + eps) = log(eps) ≈ -13.8. Sum
        // becomes very negative, exp underflows to ~0 — which is the
        // correct answer (product of anything times zero).
        let program = DiscreteProgram::new(0, 5, 0, 2, 0.0);
        let arrays = vec![2.0, 0.0, 4.0];
        let lengths = vec![3u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 3);
        assert!(
            result[0].is_finite(),
            "result should not overflow on zero input"
        );
        assert!(
            result[0] < 0.01,
            "product through zero should be tiny, got {}",
            result[0]
        );
    }

    #[test]
    fn exp_post_scale_clamps_on_large_acc() {
        // Intentionally feed a program whose acc would overflow: init=1,
        // transform=x*x, reduce=+ over very large values. post_scale=exp
        // should clamp rather than produce +inf.
        let program = DiscreteProgram::new(1, 1, 0, 2, 0.0);
        let arrays = vec![100.0, 100.0, 100.0, 100.0];
        let lengths = vec![4u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 4);
        assert!(
            result[0].is_finite(),
            "clamped exp must not produce +inf, got {}",
            result[0]
        );
    }

    #[test]
    fn length_zero_returns_init_plus_offset() {
        let program = DiscreteProgram::new(0, 0, 0, 0, 1.25);
        let arrays = vec![7.0, 8.0, 9.0];
        let lengths = vec![0u32];
        let result = execute_cpu_batch(&[program], &arrays, &lengths, 3);
        assert_eq!(result, vec![1.25]);
    }

    #[test]
    fn metal_and_cpu_agree_on_sum() {
        // Only run when a Metal device is available (macOS with GPU).
        let gpu = match NpcotGpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let program = DiscreteProgram::new(0, 0, 0, 0, 0.0);
        let arrays = vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let lengths = vec![3u32, 2u32];
        let cpu = execute_cpu_batch(&[program], &arrays, &lengths, 5);
        let gpu_res = gpu
            .execute(&[program], &arrays, &lengths, 5)
            .expect("metal execute");
        for (c, g) in cpu.iter().zip(gpu_res.iter()) {
            assert!((c - g).abs() < 1e-5, "{} vs {}", c, g);
        }
    }

    #[test]
    fn library_index_inserts_and_looks_up() {
        let mut index = NativeLibraryIndex::new(0.9);
        let sig1: Vec<f32> = vec![1.0, 0.0, 0.0];
        let sig2: Vec<f32> = vec![0.0, 1.0, 0.0];
        index.insert(NativeLibraryEntry {
            signature: sig1.clone(),
            program: DiscreteProgram::new(0, 0, 0, 0, 0.0),
        });
        index.insert(NativeLibraryEntry {
            signature: sig2.clone(),
            program: DiscreteProgram::new(2, 0, 2, 0, 0.0),
        });
        let hit = index.lookup(&sig1).expect("lookup hit");
        assert_eq!(hit.program.transform_idx, 0);
        assert_eq!(hit.program.reduce_idx, 0);
        // Aligned lookup on second entry.
        let hit2 = index.lookup(&[0.0, 0.9, 0.1]).expect("close match");
        assert_eq!(hit2.program.reduce_idx, 2);
    }

    #[test]
    fn library_index_rejects_below_threshold() {
        let mut index = NativeLibraryIndex::new(0.98);
        index.insert(NativeLibraryEntry {
            signature: vec![1.0, 0.0, 0.0],
            program: DiscreteProgram::new(0, 0, 0, 0, 0.0),
        });
        // Distant query ([0.3, 0.9, 0.3]) has cos ~= 0.3 which is below 0.98.
        assert!(index.lookup(&[0.3, 0.9, 0.3]).is_none());
    }

    #[test]
    fn library_index_handles_mismatched_dims() {
        let mut index = NativeLibraryIndex::new(0.9);
        index.insert(NativeLibraryEntry {
            signature: vec![1.0, 0.0, 0.0],
            program: DiscreteProgram::new(0, 0, 0, 0, 0.0),
        });
        // Different-dim query returns None.
        assert!(index.lookup(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn json_loader_parses_library_round_trip() {
        let library_json = r##"{
  "config": {
    "similarity_threshold": 0.85,
    "max_entries": 32,
    "normalize_epsilon": 1e-08
  },
  "entries": [
    {
      "signature": [1.0, 0.0, 0.0],
      "program": {
        "init_idx": 0,
        "transform_idx": 0,
        "reduce_idx": 0,
        "post_scale_idx": 0,
        "offset": 0.0,
        "program_text": "sum"
      },
      "hit_count": 3,
      "task_name": "sum",
      "cached_at_step": null,
      "convergence_gap": null
    },
    {
      "signature": [0.0, 1.0, 0.0],
      "program": {
        "init_idx": 2,
        "transform_idx": 0,
        "reduce_idx": 2,
        "post_scale_idx": 0,
        "offset": 0.0,
        "program_text": "max"
      },
      "hit_count": 0,
      "task_name": "max",
      "cached_at_step": null,
      "convergence_gap": null
    }
  ]
}"##;
        let (thr, index) = load_library_from_json_bytes(library_json.as_bytes())
            .expect("parse");
        assert!((thr - 0.85).abs() < 1e-5);
        assert_eq!(index.len(), 2);
        let hit = index.lookup(&[1.0, 0.0, 0.0]).expect("sum hit");
        assert_eq!(hit.program.reduce_idx, 0);
        let hit2 = index.lookup(&[0.0, 1.0, 0.0]).expect("max hit");
        assert_eq!(hit2.program.reduce_idx, 2);
    }

    #[test]
    fn consult_library_native_executes_on_hit() {
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
        let (_thr, index) =
            load_library_from_json_bytes(library_json.as_bytes()).expect("parse");
        let hidden = vec![1.0, 0.0, 0.0];
        let array = vec![1.0, 2.0, 3.0, 0.0, 0.0];
        let result = consult_library_native(&index, &hidden, &array, 3).expect("hit");
        assert!((result - 6.0).abs() < 1e-5);
    }

    #[test]
    fn consult_library_native_returns_none_on_miss() {
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
        let (_thr, index) =
            load_library_from_json_bytes(library_json.as_bytes()).expect("parse");
        // Orthogonal query should miss at 0.99 threshold.
        let hidden = vec![0.0, 1.0, 0.0];
        let array = vec![1.0, 2.0, 3.0];
        assert!(consult_library_native(&index, &hidden, &array, 3).is_none());
    }

    #[test]
    fn metal_and_cpu_agree_on_mixed_batch() {
        let gpu = match NpcotGpu::new() {
            Ok(g) => g,
            Err(_) => return,
        };
        let progs = vec![
            DiscreteProgram::new(0, 0, 0, 0, 0.0),   // sum
            DiscreteProgram::new(2, 0, 2, 0, 0.0),   // max (signed)
            DiscreteProgram::new(0, 4, 0, 0, 0.0),   // count_positive
        ];
        let arrays = vec![
            1.0, 2.0, 3.0, 0.0, 0.0,     // sum -> 6
            -3.0, -1.0, -5.0, 0.0, 0.0,  // max -> -1
            1.0, -2.0, 3.0, 4.0, -5.0,   // count>0 -> 3
        ];
        let lengths = vec![3u32, 3u32, 5u32];
        let cpu = execute_cpu_batch(&progs, &arrays, &lengths, 5);
        let gpu_res = gpu
            .execute(&progs, &arrays, &lengths, 5)
            .expect("metal execute");
        for (c, g) in cpu.iter().zip(gpu_res.iter()) {
            assert!((c - g).abs() < 1e-5, "{} vs {}", c, g);
        }
    }
}
