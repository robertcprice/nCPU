//! The self-improvement safety substrate.
//!
//! Everything that lets the understanding layer change itself routes through
//! this module, and it exists for exactly one reason: a system that rewrites
//! its own behavior is only trustworthy if every change is *guarded* and every
//! attempt is *recorded*. Those are the two invariants this substrate enforces.
//!
//! * [`gate`] is the **guard**. No self-modification is allowed to take effect
//!   until it has passed the regression gate — a fixed corpus of golden cases
//!   (setup sentences, a question, an expected answer) replayed through a fresh
//!   discourse, plus a soundness check. If a candidate change makes any golden
//!   case regress, or makes the world model unsound, the gate rejects it and the
//!   change is discarded. The gate is a pure *consumer* of the existing
//!   understanding API (`Engine` + `Discourse` + `qa::answer`); it never mutates
//!   the engine, so it can be run before *and* after a proposed change to prove
//!   the change did no harm.
//!
//! * [`journal`] is the **record**. Every self-modification attempt — accepted
//!   or rejected — is appended to a durable, append-only reflection journal:
//!   what gap was being closed, what action was tried, by which method, whether
//!   the synthesized program verified, whether it passed the regression gate,
//!   and whether it was ultimately accepted. This is the audit trail: it makes
//!   the system's self-modification history inspectable after the fact and gives
//!   later phases a memory of what has already been tried (so a failed approach
//!   isn't blindly retried).
//!
//! Together: **propose → journal the attempt → run the gate → accept only on a
//! green gate → journal the outcome.** A self-modification that skips either the
//! gate or the journal is, by construction, not allowed. This module currently
//! holds the type scaffold and stubbed entry points; the real gating and
//! journaling logic lands in the next phase.

pub mod extend;
pub mod gate;
pub mod journal;
pub mod store;
