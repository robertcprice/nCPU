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
    let Ok(s) = CStr::from_ptr(text).to_str() else { return fallback };
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
pub unsafe extern "C" fn ncpu_is_person(
    engine: *const NcpuEngine,
    word: *const c_char,
) -> i32 {
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
