//! nsynth-as-a-tool — the RLVR entry point.
//!
//! THE ARCHITECTURE (see docs/AGENTIC_NL_PLAYBOOK.md): a small model is TRAINED to
//! USE nsynth, not have nsynth baked into its weights. The correctness argument
//! stays INDEPENDENT of the model — nsynth verifies at inference — so the
//! zero-false-positive guarantee survives (distilling nsynth into weights would
//! lose both the guarantee and, per the research, 3-10% capability).
//!
//! This module is the clean seam the RL loop drives:
//!   * [`ToolRequest`] — a structured proposal the model emits (a spec or a
//!     candidate program).
//!   * [`run_tool`] — routes the proposal through nsynth's REAL synthesis +
//!     strict verification + the consensus trust gate, returning a
//!     [`ToolResponse`] (`Verified` / `Tentative` / `Refused`).
//!   * [`rlvr_reward`] — the UN-HACKABLE reward: 1.0 iff nsynth returns a Verified
//!     program that ALSO passes the held-out `hidden` tests (actual-task
//!     correctness, not self-consistency). nsynth is a proof-carrying verifier, a
//!     strictly stronger RLVR reward than raw unit tests.
//!
//! The `nsynth_tool` bin exposes this over stdin/stdout JSON as the RL environment.

use crate::agent::consensus::{differential_consensus, ConsensusVerdict};
use crate::benchmark::{Example, Problem};
use crate::runtime::{code_reproduces_examples, verify_problem_code_strict};
use crate::solver::solve_problem;

/// A proposal the model hands to nsynth. All variants route to REAL synthesis +
/// verification; none can bypass the verifier.
#[derive(Debug, Clone)]
pub enum ToolRequest {
    /// The model proposes the I/O the target function must satisfy; nsynth
    /// SYNTHESIZES a program reproducing them (+ strict-verify + trust gate).
    Examples {
        signature: String,
        examples: Vec<Example>,
    },
    /// The model writes a reference implementation; nsynth manufactures examples by
    /// RUNNING it and synthesizes an equivalent program verified to agree on fresh
    /// inputs ("make a function like THIS").
    Reference {
        name: String,
        signature: String,
        code: String,
    },
    /// The model writes a candidate PROGRAM directly; nsynth acts as pure VERIFIER —
    /// strict-verify it against the given examples + corroborate. (nsynth's value
    /// here is the strong proof-carrying check beyond just running the tests.)
    VerifyProgram {
        signature: String,
        code: String,
        examples: Vec<Example>,
    },
}

/// nsynth's verdict on a proposal.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResponse {
    /// Reproduces the spec under strict verification AND is independently
    /// corroborated (or reference/holdout-backed). A confident solve.
    Verified { code: String, method: String },
    /// Reproduces the spec but could not be independently corroborated — present
    /// to the user as "confirm or add an example", never a confident solve.
    Tentative { code: String, method: String },
    /// nsynth could not verify a program for this proposal (no synthesis, failed
    /// strict verify, or a divergence witness proving the spec underdetermined).
    Refused { reason: String },
}

impl ToolResponse {
    pub fn code(&self) -> Option<&str> {
        match self {
            ToolResponse::Verified { code, .. } | ToolResponse::Tentative { code, .. } => Some(code),
            ToolResponse::Refused { .. } => None,
        }
    }
    pub fn is_verified(&self) -> bool {
        matches!(self, ToolResponse::Verified { .. })
    }
}

/// Route a model proposal through nsynth's synthesis + verification + trust gate.
pub fn run_tool(req: &ToolRequest) -> ToolResponse {
    match req {
        ToolRequest::Examples { examples, .. } => {
            if examples.is_empty() {
                return ToolResponse::Refused {
                    reason: "no examples proposed".into(),
                };
            }
            let problem = examples_problem(examples.clone(), Vec::new());
            let solved = solve_problem(&problem);
            if !solved.success {
                return ToolResponse::Refused {
                    reason: solved
                        .error
                        .unwrap_or_else(|| "nsynth could not synthesize a program".into()),
                };
            }
            gate(&problem, solved.code, solved.method)
        }
        ToolRequest::Reference {
            name,
            signature,
            code,
        } => {
            // problem_from_reference needs &'static strs — leak (tool-call lifetime).
            let sig: &'static str = Box::leak(signature.clone().into_boxed_str());
            let refc: &'static str = Box::leak(code.clone().into_boxed_str());
            match crate::benchmark::problem_from_reference(name, sig, refc) {
                Ok(problem) => {
                    let solved = solve_problem(&problem);
                    if !solved.success {
                        return ToolResponse::Refused {
                            reason: "nsynth could not synthesize an equivalent of the reference"
                                .into(),
                        };
                    }
                    gate(&problem, solved.code, solved.method)
                }
                Err(e) => ToolResponse::Refused {
                    reason: format!("unusable reference: {e}"),
                },
            }
        }
        ToolRequest::VerifyProgram { code, examples, .. } => {
            if examples.is_empty() {
                return ToolResponse::Refused {
                    reason: "no examples to verify against".into(),
                };
            }
            // nsynth as pure verifier: the model wrote `code`; `gate` strict-verifies
            // it + corroborates.
            let problem = examples_problem(examples.clone(), Vec::new());
            gate(&problem, code.clone(), "model-program".to_string())
        }
    }
}

/// The RLVR reward: 1.0 iff nsynth returns a VERIFIED program that also passes the
/// held-out `hidden` tests. A Tentative (uncorroborated) that happens to pass earns
/// partial credit; anything wrong or refused earns 0. `hidden` is the actual-task
/// ground truth held out from the model — this is what makes the reward reflect
/// real correctness, not self-consistency, and it cannot be reward-hacked because
/// nsynth's verification is proof-carrying.
pub fn rlvr_reward(req: &ToolRequest, hidden: &[Example]) -> f32 {
    match run_tool(req) {
        ToolResponse::Verified { code, .. } => {
            if code_reproduces_examples(&code, hidden) {
                1.0
            } else {
                0.0
            }
        }
        ToolResponse::Tentative { code, .. } => {
            if code_reproduces_examples(&code, hidden) {
                0.5
            } else {
                0.0
            }
        }
        ToolResponse::Refused { .. } => 0.0,
    }
}

/// Build an examples-only Problem (no reference — the pure NL/tool regime). The
/// signature is INFERRED from the example value types (the examples are ground
/// truth for types); a model-proposed signature string is only advisory and often
/// wrong, so we do not trust it.
fn examples_problem(examples: Vec<Example>, holdouts: Vec<Example>) -> Problem {
    let sig = crate::linguigenesis_bridge::infer_signature("f", &examples);
    Problem {
        name: "tool".to_string(),
        signature: Box::leak(sig.into_boxed_str()),
        examples,
        holdouts,
        ..Default::default()
    }
}

/// Strict-verify + consensus trust gate → Verified / Tentative / Refused.
fn gate(problem: &Problem, code: String, method: String) -> ToolResponse {
    if verify_problem_code_strict(problem, &code).is_err() {
        return ToolResponse::Refused {
            reason: "program failed strict verification against the examples".into(),
        };
    }
    match differential_consensus(problem, &code) {
        ConsensusVerdict::Ambiguous { .. } => ToolResponse::Refused {
            reason: "an independent solution fitting the same examples disagrees on \
                     another input — the examples do not determine the function"
                .into(),
        },
        ConsensusVerdict::Verified { .. } => ToolResponse::Verified { code, method },
        ConsensusVerdict::NoConsensus => ToolResponse::Tentative { code, method },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Value;

    fn ex(i: &[i64], o: i64) -> Example {
        Example {
            inputs: i.iter().map(|v| Value::Int(*v)).collect(),
            expected: Value::Int(o),
        }
    }

    #[test]
    fn examples_proposal_verifies_and_rewards() {
        // Model proposes the I/O for "add two numbers".
        let req = ToolRequest::Examples {
            signature: "fn add(a: i64, b: i64) -> i64".into(),
            examples: vec![ex(&[2, 3], 5), ex(&[10, 4], 14), ex(&[0, 0], 0)],
        };
        let resp = run_tool(&req);
        assert!(resp.code().is_some(), "should synthesize: {resp:?}");
        // Held-out tests it was NOT given — reward reflects TRUE correctness.
        let hidden = [ex(&[7, 8], 15), ex(&[100, 1], 101)];
        assert_eq!(rlvr_reward(&req, &hidden), 1.0, "add generalizes: {resp:?}");
    }

    #[test]
    fn reference_proposal_synthesizes_equivalent() {
        // Model writes a reference; nsynth builds a verified equivalent.
        let req = ToolRequest::Reference {
            name: "triple".into(),
            signature: "fn triple(x: i64) -> i64".into(),
            code: "fn triple(x: i64) -> i64 { return x * 3; }".into(),
        };
        let resp = run_tool(&req);
        assert!(resp.code().is_some(), "reference intake should solve: {resp:?}");
        let hidden = [ex(&[5], 15), ex(&[-2], -6)];
        assert!(rlvr_reward(&req, &hidden) >= 0.5, "triple correct: {resp:?}");
    }

    #[test]
    fn wrong_program_is_refused_zero_reward() {
        // Model writes a program that does NOT satisfy the examples → refused, 0.
        let req = ToolRequest::VerifyProgram {
            signature: "fn f(a: i64, b: i64) -> i64".into(),
            code: "fn f(a: i64, b: i64) -> i64 { return a - b; }".into(),
            examples: vec![ex(&[2, 3], 5), ex(&[10, 4], 14)],
        };
        assert!(matches!(run_tool(&req), ToolResponse::Refused { .. }));
        assert_eq!(rlvr_reward(&req, &[ex(&[7, 8], 15)]), 0.0);
    }
}
