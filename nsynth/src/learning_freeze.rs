//! Process-global "learning freeze" — a read-only evaluation mode.
//!
//! Normally a successful `solve_problem` has *learning side-effects*: it records
//! the program into [`crate::solved_cache`] and reinforces the
//! [`crate::meta_learner`] weight vector / cache success-counts on transfer
//! wins. Those side-effects are correct during normal operation but **corrupt
//! any process that needs to measure the solver as a pure function of its
//! parameters** — most importantly the bounded recursive self-improvement loop
//! (Phase 5.1), whose fitness gate requires that evaluating a candidate weight
//! vector neither (a) mutates a second surface (the cache) nor (b) mutates the
//! weights themselves mid-evaluation.
//!
//! While frozen, the three learning write sites no-op:
//! - `pipeline::solve_problem` skips `solved_cache::record`,
//! - `meta_learner::record_transfer_success` returns early,
//! - `solved_cache::note_transfer_success` returns early.
//!
//! The cache and weights remain fully *readable* (so teacher ranking still
//! reflects the candidate weights over the fixed donor set) — only writes are
//! suppressed. A global `AtomicBool` (not thread-local) is used so the freeze
//! covers any worker threads the solver may spawn during an evaluation.

use std::sync::atomic::{AtomicBool, Ordering};

static FROZEN: AtomicBool = AtomicBool::new(false);

/// Whether learning side-effects are currently suppressed.
pub fn is_frozen() -> bool {
    FROZEN.load(Ordering::SeqCst)
}

/// RAII guard: freezes learning side-effects for its lifetime, restoring the
/// prior state on drop (panic-safe, nesting-safe).
pub struct FreezeGuard {
    prev: bool,
}

/// Enter a frozen (read-only-learning) scope.
pub fn freeze() -> FreezeGuard {
    let prev = FROZEN.swap(true, Ordering::SeqCst);
    FreezeGuard { prev }
}

impl Drop for FreezeGuard {
    fn drop(&mut self) {
        FROZEN.store(self.prev, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeze_guard_sets_and_restores() {
        assert!(!is_frozen());
        {
            let _g = freeze();
            assert!(is_frozen());
            {
                let _g2 = freeze();
                assert!(is_frozen());
            }
            // Inner drop restores prior (still frozen), not unconditionally false.
            assert!(is_frozen());
        }
        assert!(!is_frozen());
    }
}
