//! nsynth-as-a-tool — the RLVR entry point.
//!
//! THE ARCHITECTURE (see docs/AGENTIC_NL_PLAYBOOK.md): a small model is TRAINED to
//! USE nsynth, not have nsynth baked into its weights. The correctness argument
//! stays INDEPENDENT of the model — nsynth verifies at inference — so the
//! zero-false-positive guarantee survives.
//!
//! TWO PATHS, and the FIRST is the powerful one:
//!   1. [`ToolRequest::VerifyProgram`] — the model WRITES Mog code; nsynth EXECUTES
//!      + strict-verifies it. This is the POWERFUL path: nsynth's runtime runs a
//!      broad Rust-subset (loops/conditionals/arrays/strings), so the model can
//!      write ALGORITHMS nsynth could never synthesize, and nsynth still guarantees
//!      correctness. Ceiling = the interpreter's execution breadth + the model's
//!      coding — NOT nsynth's synthesis reach. MOG DIALECT (a Rust subset, but NO
//!      `as` casts): declare `x: i64 = 0;` (no `let`); `for e in arr {}`,
//!      `while c {}`, `if c {} else {}`; index `a[i]` (i64 index, no `as usize`);
//!      `a.len()` returns i64 (no cast); `a.push(e)`; `&&`/`||`/`%`; `return e;`.
//!   2. [`ToolRequest::Examples`] / [`Reference`] — the model proposes a SPEC;
//!      nsynth SYNTHESIZES the verified program. Narrower (bounded by nsynth's
//!      synthesis = the PBE rate), but the model writes nothing.
//!
//! [`run_tool`] routes a proposal through synthesis/execution + strict verify + the
//! consensus trust gate → [`ToolResponse`] (`Verified`/`Tentative`/`Refused`).
//! [`rlvr_reward`] = 1.0 iff a Verified program ALSO passes the held-out `hidden`
//! tests (real correctness, un-hackable — nsynth's check is proof-carrying, a
//! strictly stronger RLVR reward than raw unit tests).
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
            // nsynth as pure verifier: the model WROTE `code`. Strict-verify it
            // against the examples. Unlike the synthesis paths, we do NOT refuse a
            // PASSING program just because an independent candidate also fits the
            // (possibly thin) examples — that is about the spec, not the model's
            // code. So: fail strict-verify → Refused; else Verified iff corroborated,
            // Tentative otherwise (never refuse a program that passes).
            let problem = examples_problem(examples.clone(), Vec::new());
            // The verifier calls the entry fn by `problem.function_name()` (inferred
            // as `f` here). A model names its function whatever it likes (`nth_prime`,
            // `solve`, …), so align the code's entry-fn name to what the verifier
            // looks up — otherwise a perfectly CORRECT program is false-Refused merely
            // for its name (the fn is never found). Correctness is name-independent.
            let code = &normalize_entry_fn(code, &problem.function_name());
            if verify_problem_code_strict(&problem, code).is_err() {
                return ToolResponse::Refused {
                    reason: "program failed strict verification against the examples".into(),
                };
            }
            match differential_consensus(&problem, code) {
                ConsensusVerdict::Verified { .. } => ToolResponse::Verified {
                    code: code.clone(),
                    method: "model-program".to_string(),
                },
                _ => ToolResponse::Tentative {
                    code: code.clone(),
                    method: "model-program".to_string(),
                },
            }
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

/// Rename a program's entry function (and any self-recursive calls to it) to
/// `target`, so a verifier that invokes the entry by a fixed name can run a
/// program the model named arbitrarily. Whole-word match only — never touches a
/// variable/substring that merely contains the name. No-op if the name already
/// matches or no `fn <name>` is found.
fn normalize_entry_fn(code: &str, target: &str) -> String {
    let Some((_, rest)) = code.split_once("fn ") else { return code.to_string() };
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.is_empty() || name == target {
        return code.to_string();
    }
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(code.len());
    let mut idx = 0;
    while let Some(pos) = code[idx..].find(&name) {
        let at = idx + pos;
        let before_ok = at == 0 || !code[..at].chars().next_back().is_some_and(is_ident);
        let after_ok = !code[at + name.len()..].chars().next().is_some_and(is_ident);
        out.push_str(&code[idx..at]);
        out.push_str(if before_ok && after_ok { target } else { &name });
        idx = at + name.len();
    }
    out.push_str(&code[idx..]);
    out
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
    fn model_written_algorithm_beyond_synthesis_is_verified() {
        // The POWERFUL path: the model writes a Mog ALGORITHM (a loop nsynth would
        // not synthesize from these examples); nsynth EXECUTES + verifies it. Must
        // NOT be refused for being one-of-many fits (that's the spec, not the code).
        let arr = |xs: &[i64]| Example {
            inputs: vec![Value::int_array(xs)],
            expected: Value::Int(xs.iter().sum()),
        };
        let req = ToolRequest::VerifyProgram {
            signature: "fn f(a: [i64]) -> i64".into(),
            // Mog dialect: no `let`; `a.len()` is i64; `a[i]` i64 index; no `as`.
            code: "fn f(a: [i64]) -> i64 { s: i64 = 0; i: i64 = 0; while i < a.len() { \
                   s = s + a[i]; i = i + 1; } return s; }"
                .into(),
            examples: vec![arr(&[1, 2, 3]), arr(&[10, 20]), arr(&[5])],
        };
        let resp = run_tool(&req);
        assert!(resp.code().is_some(), "model algorithm must be accepted, got {resp:?}");
        // Held-out oracle: it actually sums.
        assert!(rlvr_reward(&req, &[arr(&[7, 8, 9])]) >= 0.5, "sums correctly: {resp:?}");
    }

    #[test]
    fn correct_program_not_refused_for_its_fn_name() {
        // Regression: a CORRECT model program named anything but `f` must still be
        // verified. Before normalize_entry_fn the verifier looked up `f`, never
        // found the model's `nth_prime`, and false-Refused a perfect program — which
        // silently sank every model-taught op in the distillation flywheel.
        let e = |i: i64, o: i64| Example { inputs: vec![Value::Int(i)], expected: Value::Int(o) };
        let named = "fn nth_prime(n: i64) -> i64 { count: i64 = 0; candidate: i64 = 2; \
            while count < n { is_prime: i64 = 1; d: i64 = 2; while (d * d) <= candidate { \
            if (candidate % d) == 0 { is_prime = 0; } d = d + 1; } if is_prime == 1 { \
            count = count + 1; if count == n { return candidate; } } candidate = candidate + 1; } \
            return candidate; }";
        let examples = vec![e(1, 2), e(2, 3), e(3, 5), e(4, 7), e(5, 11), e(6, 13)];
        let req = ToolRequest::VerifyProgram {
            signature: "fn f(n: i64) -> i64".into(),
            code: named.into(),
            examples,
        };
        let resp = run_tool(&req);
        assert!(resp.code().is_some(), "correctly-named-`nth_prime` must be accepted, got {resp:?}");
        // Held-out oracle: it computes the real nth prime.
        assert!(rlvr_reward(&req, &[e(7, 17), e(8, 19)]) >= 0.5, "computes nth prime: {resp:?}");
    }

    #[test]
    fn normalize_entry_fn_is_whole_word_and_recursion_safe() {
        // Whole-word rename of the entry fn + its self-calls; leaves lookalikes alone.
        let src = "fn fib(n: i64) -> i64 { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }";
        let got = normalize_entry_fn(src, "f");
        assert_eq!(
            got,
            "fn f(n: i64) -> i64 { if n < 2 { return n; } return f(n - 1) + f(n - 2); }"
        );
        // A variable that merely CONTAINS the entry name (`ab` inside `abc`) is left
        // alone; only the whole-word entry fn is renamed.
        assert_eq!(normalize_entry_fn("fn ab(abc: i64) -> i64 { return abc + 1; }", "f"),
                   "fn f(abc: i64) -> i64 { return abc + 1; }");
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
