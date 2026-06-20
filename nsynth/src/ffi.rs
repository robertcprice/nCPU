//! C ABI for the native comprehension engine. Lets C (or any C-FFI-capable
//! language) build the engine once and query it. Every result is decided by a
//! synthesized, verified Mog program — see [`crate::comprehension`].
//!
//! ```c
//! NcpuEngine *e = ncpu_engine_new();
//! int blocked = ncpu_comprehend_roles(e, "the report writes the teacher"); // 0
//! int ok      = ncpu_check_agreement(e, "the captains watch the report");  // 1
//! int valid   = ncpu_judge_argument(e,
//!     "If the alarm rings, then the guard wakes. The alarm rings. Thus, the guard wakes."); // 1
//! ncpu_engine_free(e);
//! ```

use std::ffi::{c_char, CStr};

use crate::comprehension::Engine;
use crate::understanding::mind::Mind;

/// Opaque handle to a stateful understanding "mind" (an engine + a discourse it
/// reads into). Create with [`ncpu_mind_new`], release with [`ncpu_mind_free`].
pub struct NcpuMind(Mind);

/// Build a mind (synthesizes the lexicon/rules once; takes a few seconds).
///
/// # Safety
/// The returned pointer must be released exactly once with [`ncpu_mind_free`].
#[no_mangle]
pub extern "C" fn ncpu_mind_new() -> *mut NcpuMind {
    Box::into_raw(Box::new(NcpuMind(Mind::new())))
}

/// Release a mind created by [`ncpu_mind_new`]. Null is a no-op.
///
/// # Safety
/// `mind` must be a pointer from [`ncpu_mind_new`], not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn ncpu_mind_free(mind: *mut NcpuMind) {
    if !mind.is_null() {
        drop(Box::from_raw(mind));
    }
}

/// Read a sentence into the world model (resolving coreference, asserting facts).
/// Returns 0 on success, -1 on null/invalid input.
///
/// # Safety
/// `mind` must be a valid handle; `sentence` a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ncpu_read(mind: *mut NcpuMind, sentence: *const c_char) -> i32 {
    if mind.is_null() || sentence.is_null() {
        return -1;
    }
    let Ok(s) = CStr::from_ptr(sentence).to_str() else {
        return -1;
    };
    (*mind).0.read(s);
    0
}

/// Answer a question from what the mind has read, writing the answer
/// NUL-terminated into `out` (capacity `cap`). Returns bytes written (excl NUL),
/// -1 on null/invalid input, -2 if the buffer is too small.
///
/// # Safety
/// `mind` must be a valid handle; `question` a valid NUL-terminated UTF-8 string;
/// `out` must point to at least `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ncpu_ask(
    mind: *const NcpuMind,
    question: *const c_char,
    out: *mut c_char,
    cap: usize,
) -> i32 {
    if mind.is_null() || question.is_null() || out.is_null() || cap == 0 {
        return -1;
    }
    let Ok(q) = CStr::from_ptr(question).to_str() else {
        return -1;
    };
    let answer = (*mind).0.ask(q);
    let bytes = answer.as_bytes();
    if bytes.len() + 1 > cap {
        return -2;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
    *out.add(bytes.len()) = 0;
    bytes.len() as i32
}

/// Third-person-singular form of a verb base, written NUL-terminated into `out`
/// (capacity `cap` bytes). Regular verbs go through the synthesized rule;
/// irregulars (have→has, be→is, do→does, go→goes) through the synthesized
/// lexicon. Returns the byte length written (excluding NUL), -1 on null/invalid
/// input, or -2 if the buffer is too small.
///
/// # Safety
/// `engine` must be a valid handle; `base` a valid NUL-terminated UTF-8 string;
/// `out` must point to at least `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ncpu_verb_3sg(
    engine: *const NcpuEngine,
    base: *const c_char,
    out: *mut c_char,
    cap: usize,
) -> i32 {
    if engine.is_null() || base.is_null() || out.is_null() || cap == 0 {
        return -1;
    }
    let Ok(b) = CStr::from_ptr(base).to_str() else {
        return -1;
    };
    let form = (*engine).0.verb_3sg(b);
    let bytes = form.as_bytes();
    if bytes.len() + 1 > cap {
        return -2;
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
    *out.add(bytes.len()) = 0;
    bytes.len() as i32
}

/// Opaque handle to a built engine. Create with [`ncpu_engine_new`], release with
/// [`ncpu_engine_free`].
pub struct NcpuEngine(Engine);

/// Build the comprehension engine (synthesizes the lexicon and rules once; this
/// takes a few seconds). Returns an owned pointer the caller must free.
///
/// # Safety
/// The returned pointer must be released exactly once with [`ncpu_engine_free`].
#[no_mangle]
pub extern "C" fn ncpu_engine_new() -> *mut NcpuEngine {
    Box::into_raw(Box::new(NcpuEngine(Engine::new())))
}

/// Release an engine created by [`ncpu_engine_new`]. Passing null is a no-op.
///
/// # Safety
/// `engine` must be a pointer returned by [`ncpu_engine_new`] and not used after.
#[no_mangle]
pub unsafe extern "C" fn ncpu_engine_free(engine: *mut NcpuEngine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

/// Borrow the engine and decode a C string; returns `fallback` on null/invalid.
unsafe fn with_str<F: FnOnce(&Engine, &str) -> i32>(
    engine: *const NcpuEngine,
    text: *const c_char,
    fallback: i32,
    f: F,
) -> i32 {
    if engine.is_null() || text.is_null() {
        return fallback;
    }
    let Ok(s) = CStr::from_ptr(text).to_str() else {
        return fallback;
    };
    f(&(*engine).0, s)
}

/// 1 if the sentence's action is semantically licensed (animate subject acting on
/// an inanimate object), else 0. Returns -1 on null/invalid input.
///
/// # Safety
/// `engine` must be a valid handle; `sentence` a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ncpu_comprehend_roles(
    engine: *const NcpuEngine,
    sentence: *const c_char,
) -> i32 {
    with_str(engine, sentence, -1, |e, s| e.comprehend_roles(s) as i32)
}

/// 1 if the sentence is grammatical in subject-verb agreement, else 0. Returns -1
/// on null/invalid input.
///
/// # Safety
/// `engine` must be a valid handle; `sentence` a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ncpu_check_agreement(
    engine: *const NcpuEngine,
    sentence: *const c_char,
) -> i32 {
    with_str(engine, sentence, -1, |e, s| e.check_agreement(s) as i32)
}

/// 1 if the word is an animate noun (a "person"), else 0. Returns -1 on
/// null/invalid input.
///
/// # Safety
/// `engine` must be a valid handle; `word` a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ncpu_is_person(engine: *const NcpuEngine, word: *const c_char) -> i32 {
    with_str(engine, word, -1, |e, s| e.is_person(s) as i32)
}

/// Judge a conditional argument: 1 valid, 0 invalid, -1 unparseable/invalid input.
///
/// # Safety
/// `engine` must be a valid handle; `sentence` a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ncpu_judge_argument(
    engine: *const NcpuEngine,
    sentence: *const c_char,
) -> i32 {
    with_str(engine, sentence, -1, |e, s| e.judge_argument(s) as i32)
}
