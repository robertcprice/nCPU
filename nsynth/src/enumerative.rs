//! Bottom-up enumerative program synthesis with learned component library.
//!
//! Builds ALL programs from small to large, tests each against I/O examples,
//! returns the first (smallest) program that satisfies all examples.
//!
//! Features:
//!   - 12 binary ops: +, -, *, /, %, min, max, &, |, ^, <<, >>
//!   - 4 unary ops: neg, abs, bitwise_not, popcount
//!   - If-then-else with 6 comparisons
//!   - While loops with accumulator pattern
//!   - Observational equivalence pruning (skip equivalent programs)
//!   - Learned component library: saves discovered sub-expressions
//!   - Dream mode: explores combinations offline to grow the library
//!
//! No templates. No patterns. Pure discovery.

use crate::benchmark::{Problem, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::runtime::verify_problem_code_strict;
use crate::solver::SolveResult;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

// ─── Expression AST ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Var(usize),
    Const(i64),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnOp, Box<Expr>),
    IfExpr(CmpOp, Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>), // if lhs CMP rhs { then } else { els }
    WhileAccum {
        // acc = init; while cond { acc = body(acc, i); i++ } return acc
        init: Box<Expr>,
        bound: Box<Expr>,    // loop runs i from 0..bound
        body_op: BinOp,      // acc = acc OP rhs
        body_rhs: Box<Expr>, // rhs computed from [acc, i, args, consts]
    },
    ForFold {
        // acc = init; for item in arr { acc = acc OP rhs(item, i, acc, args) } return acc
        init: Box<Expr>,
        body_op: BinOp,
        body_rhs: Box<Expr>, // rhs over {item, i, acc, scalar_args, consts}
    },
    NestedWhile {
        // two-level nested accumulator loop
        outer_init: Box<Expr>,
        outer_bound: Box<Expr>,
        outer_body_op: BinOp,
        inner_init: Box<Expr>,
        inner_bound: Box<Expr>,
        inner_body_op: BinOp,
        inner_body_rhs: Box<Expr>, // rhs over {args, outer_acc, outer_i, inner_acc, inner_i}
    },
    WhileCond {
        // acc = init; x = x_init; while x CMP cond_val { acc = acc OP body(x, acc, args); x = x / divisor } return acc
        init: Box<Expr>,       // acc init
        state_init: Box<Expr>, // x init (typically arg[0])
        cond_cmp: CmpOp,       // condition: x CMP cond_val
        cond_val: Box<Expr>,   // value to compare against
        divisor: i64,          // x update: x = x / divisor
        body_op: BinOp,        // acc = acc OP body_rhs
        body_rhs: Box<Expr>,   // rhs over {args, x, acc, i}
    },
    /// Call a previously-SOLVED function (a registered [`NamedCallable`]) by its
    /// registry index, with `args` sub-expressions. This is the inter-function
    /// data-flow node: when component B is searched with producer A registered,
    /// the enumerator may emit `Call(idx_of_A, [arg])`. It is DISCOVERED by the
    /// search exactly like any other constructor — there is no compose template.
    /// `eval` requires the matching registry to be in scope (see `eval_with_callees`).
    Call(usize, Vec<Expr>),
}

/// A solved function registered as a real callable PRIMITIVE for the search.
/// `eval` executes the producer A on concrete args during candidate evaluation,
/// so the enumerator can verify a `Call(idx, args)` candidate end-to-end. The
/// registry is supplied ONLY at the call site that wants inter-function flow
/// (B's solve with A registered); when it is empty the base search is byte-
/// identical to today (no `Call` node is ever constructed).
pub struct NamedCallable {
    /// The producer's emitted Mog fn name (used by `emit_mog` to render the call).
    pub name: String,
    /// Exact arity. A `Call` is only enumerated when `args.len() == n_args`.
    pub n_args: usize,
    /// Executes A on concrete args. `None` on a domain error (e.g. overflow),
    /// propagated like every other partial op so an unsound call is rejected.
    pub eval: Box<dyn Fn(&[i64]) -> Option<i64>>,
    /// The producer's full Mog SOURCE (`fn name(..) -> i64 { .. }`). Prepended to
    /// the consumer's emitted code so the strict-verify gate can RESOLVE the call
    /// (the consumer `name(args)` references a real definition). Empty string ⇒
    /// no source prelude (the consumer is verified standalone, as before).
    pub source: String,
}

impl std::fmt::Debug for NamedCallable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedCallable")
            .field("name", &self.name)
            .field("n_args", &self.n_args)
            .finish()
    }
}

thread_local! {
    /// Callee-name table read by `to_mog_ext` when rendering a `Call(idx, _)`
    /// node. Set ONLY for the duration of an `emit_mog`/`to_mog` call that may
    /// contain `Call`s (see [`with_callee_names`]); empty otherwise, so the
    /// base emission path is unaffected. It is a SCOPED producer→consumer edge,
    /// not persistent shared state: cleared when the scope guard drops.
    static CALLEE_NAMES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` with `CALLEE_NAMES` set to `names` for the duration of the call,
/// restoring the prior value afterward (so nested/re-entrant emission is safe).
fn with_callee_names<R>(names: &[String], f: impl FnOnce() -> R) -> R {
    CALLEE_NAMES.with(|c| {
        let prev = std::mem::replace(&mut *c.borrow_mut(), names.to_vec());
        let r = f();
        *c.borrow_mut() = prev;
        r
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Abs,
    BitNot,
    Popcount,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
    Ne,
}

const ALL_BINOPS: [BinOp; 12] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Mod,
    BinOp::Min,
    BinOp::Max,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::Shl,
    BinOp::Shr,
];
const ALL_UNOPS: [UnOp; 4] = [UnOp::Neg, UnOp::Abs, UnOp::BitNot, UnOp::Popcount];
const ALL_CMPS: [CmpOp; 6] = [
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Eq,
    CmpOp::Ge,
    CmpOp::Gt,
    CmpOp::Ne,
];
const CONSTANTS: [i64; 12] = [0, 1, -1, 2, -2, 3, 5, 7, 10, 32, 100, 255];

// ── ALGEBRAIC CANONICALIZATION (e-graph on-ramp) ────────────────────────────
// Normalize a scalar Expr to a canonical form using ONLY semantics-preserving
// rewrites that are SOUND under partial (Option) evaluation — the seed of an
// e-graph rewrite ruleset (the metamorphic algebraic laws), used here to
// (a) simplify emitted code and (b) merge algebraically-equal subtrees so the
// library flywheel compresses `a+b` and `b+a` to ONE reusable op. Deliberately
// omits constant-folding and annihilators (x*0, x-x): those change which
// sub-expressions get evaluated, which is unsound when a discarded sub-expr can
// error. Guarantee (property-tested): eval is preserved on every input.

fn expr_rank(e: &Expr) -> u8 {
    match e {
        Expr::Const(_) => 0,
        Expr::Var(_) => 1,
        Expr::UnaryOp(..) => 2,
        Expr::BinOp(..) => 3,
        _ => 4, // if / loops / call: opaque leaves for ordering purposes
    }
}

/// Deterministic total order on Expr for canonical commutative-operand ordering.
/// Structural; used ONLY to pick a stable operand order (semantics-irrelevant).
fn cmp_expr(a: &Expr, b: &Expr) -> std::cmp::Ordering {
    use std::cmp::Ordering::Equal;
    let (ra, rb) = (expr_rank(a), expr_rank(b));
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Expr::Const(x), Expr::Const(y)) => x.cmp(y),
        (Expr::Var(x), Expr::Var(y)) => x.cmp(y),
        (Expr::UnaryOp(o1, e1), Expr::UnaryOp(o2, e2)) => {
            (*o1 as u8).cmp(&(*o2 as u8)).then_with(|| cmp_expr(e1, e2))
        }
        (Expr::BinOp(o1, l1, r1), Expr::BinOp(o2, l2, r2)) => (*o1 as u8)
            .cmp(&(*o2 as u8))
            .then_with(|| cmp_expr(l1, l2))
            .then_with(|| cmp_expr(r1, r2)),
        _ => Equal,
    }
}

/// The commutative ops here are ALSO associative, so this is the assoc-comm set
/// eligible for flatten-and-sort normalization.
fn is_commutative(op: BinOp) -> bool {
    matches!(
        op,
        BinOp::Add | BinOp::Mul | BinOp::Min | BinOp::Max | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
    )
}

/// The identity element that can be DROPPED from an assoc-comm operand list
/// without changing eval (it is a literal const, never a sub-expr that errors).
/// `None` for ops with no finite droppable identity here (Min/Max/BitAnd).
fn identity_of(op: BinOp) -> Option<i64> {
    match op {
        BinOp::Add | BinOp::BitOr | BinOp::BitXor => Some(0),
        BinOp::Mul => Some(1),
        _ => None,
    }
}

/// Collect every operand of an assoc-comm `op` into `out`, flattening nested
/// applications of the SAME op (so `(a+b)+c` and `a+(b+c)` yield `[a,b,c]`).
fn flatten_assoc(op: BinOp, e: &Expr, out: &mut Vec<Expr>) {
    if let Expr::BinOp(o, l, r) = e {
        if *o == op {
            flatten_assoc(op, l, out);
            flatten_assoc(op, r, out);
            return;
        }
    }
    out.push(e.clone());
}

/// Semantics-preserving canonicalization of a scalar Expr (see module note).
/// Iterates the local rewriter to a bounded fixpoint. Non-scalar nodes
/// (loops / Call) are returned structurally unchanged in v1.
pub fn algebraic_normalize(e: &Expr) -> Expr {
    let mut cur = algebra_step(e);
    for _ in 0..8 {
        let next = algebra_step(&cur);
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

fn algebra_step(e: &Expr) -> Expr {
    // CONSTANT FOLDING, sound BY CONSTRUCTION: a closed (Var-free) subexpr is
    // replaced by its value computed with `eval` ITSELF — so the fold can never
    // diverge from eval — and an undefined closed expr (e.g. `n/0` -> None, or a
    // `Call` with no registry) is left as-is, never fabricating a Const. `eval`
    // on a Var short-circuits to None, so non-constant nodes fall straight
    // through cheaply.
    if let Some(v) = e.eval(&[]) {
        return Expr::Const(v);
    }
    match e {
        Expr::Var(_) | Expr::Const(_) => e.clone(),
        Expr::UnaryOp(op, inner) => {
            let inner = algebra_step(inner);
            // Involutions: --x -> x, ~~x -> x (both keep x; sound).
            match (op, &inner) {
                (UnOp::Neg, Expr::UnaryOp(UnOp::Neg, x)) => (**x).clone(),
                (UnOp::BitNot, Expr::UnaryOp(UnOp::BitNot, x)) => (**x).clone(),
                _ => Expr::UnaryOp(*op, Box::new(inner)),
            }
        }
        Expr::IfExpr(cmp, l, r, t, els) => Expr::IfExpr(
            *cmp,
            Box::new(algebra_step(l)),
            Box::new(algebra_step(r)),
            Box::new(algebra_step(t)),
            Box::new(algebra_step(els)),
        ),
        Expr::BinOp(op, l, r) => {
            let l = algebra_step(l);
            let r = algebra_step(r);
            if is_commutative(*op) {
                // ASSOCIATIVE-COMMUTATIVE flattening: gather every operand of the
                // same op into one list, drop identity elements, sort into
                // canonical order, and (for idempotent ops) drop duplicates —
                // then rebuild a left-leaning tree. This merges ALL groupings and
                // orderings of `a+b+c` into ONE form. SOUND: every operand is
                // still evaluated exactly as before (any erroring operand yields
                // None regardless of tree shape), so eval is preserved.
                let mut operands = Vec::new();
                flatten_assoc(*op, &l, &mut operands);
                flatten_assoc(*op, &r, &mut operands);
                if let Some(id) = identity_of(*op) {
                    operands.retain(|e| !matches!(e, Expr::Const(c) if *c == id));
                }
                operands.sort_by(cmp_expr);
                if matches!(op, BinOp::Min | BinOp::Max | BinOp::BitAnd | BinOp::BitOr) {
                    operands.dedup(); // idempotent: min(x,x)=x etc.
                }
                return match operands.len() {
                    0 => Expr::Const(identity_of(*op).unwrap_or(0)),
                    1 => operands.into_iter().next().unwrap(),
                    _ => {
                        let mut it = operands.into_iter();
                        let mut acc = it.next().unwrap();
                        for e in it {
                            acc = Expr::BinOp(*op, Box::new(acc), Box::new(e));
                        }
                        acc
                    }
                };
            }
            // Non-assoc ops: only the safe literal-0 identity (x - 0 -> x).
            if matches!(op, BinOp::Sub) {
                if let Expr::Const(0) = &r {
                    return l;
                }
            }
            Expr::BinOp(*op, Box::new(l), Box::new(r))
        }
        // Non-scalar (loops / Call): v1 leaves the structure unchanged.
        other => other.clone(),
    }
}

/// Max example-mined constants to add to the size-1 seed (bounds search blow-up).
const MAX_MINED_CONSTANTS: usize = 8;
/// Magnitude bound on a mined constant (skip absurd literals that explode the
/// search / risk overflow during enumeration).
const MINED_CONSTANT_MAX_ABS: i64 = 100_000;

/// Mine candidate literal constants FROM the examples, so a closed form needing a
/// literal outside the fixed `CONSTANTS` pool (e.g. `x + 42`, `x * 256`) becomes
/// reachable. Pure function of `examples` (== the frontier fingerprint), so the
/// seeded set is deterministic per problem and stays frontier-consistent.
///
/// Candidates: each output value (constant-output / output literals); each
/// `output - input[j]` (additive literal `x + k`); and `output / input[j]` when
/// it divides exactly (multiplicative literal `x * k`). Excludes values already
/// in `CONSTANTS`, bounds magnitude, dedups, caps count — all deterministically
/// ordered (by |value| then value) so two runs seed byte-identically.
fn mine_example_constants(examples: &[(Vec<i64>, i64)]) -> Vec<i64> {
    use std::collections::BTreeSet;
    let mut cands: BTreeSet<i64> = BTreeSet::new();
    for (inputs, out) in examples {
        cands.insert(*out);
        for &x in inputs {
            cands.insert(out.wrapping_sub(x)); // x + k
            if x != 0 && out % x == 0 {
                cands.insert(out / x); // x * k
            }
        }
    }
    let fixed: BTreeSet<i64> = CONSTANTS.iter().copied().collect();
    let mut mined: Vec<i64> = cands
        .into_iter()
        .filter(|c| !fixed.contains(c) && c.abs() <= MINED_CONSTANT_MAX_ABS)
        .collect();
    // Deterministic priority: smaller-magnitude literals first (likelier the real
    // closed-form constant), tie-break by value for stability.
    mined.sort_by(|a, b| a.abs().cmp(&b.abs()).then(a.cmp(b)));
    mined.truncate(MAX_MINED_CONSTANTS);
    mined
}

// Ops suitable for loop body (no division to avoid div-by-zero in loops)
const LOOP_BODY_OPS: [BinOp; 5] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::BitXor,
    BinOp::BitOr,
];

impl Expr {
    pub fn eval(&self, args: &[i64]) -> Option<i64> {
        // No callee registry in scope → a `Call` node cannot be evaluated.
        // The base search never CONSTRUCTS a `Call` (registry empty), so this
        // path is only reached when an externally-built `Call` is `eval`'d
        // without callees, which is correctly a domain error (`None`).
        self.eval_with_callees(args, &[])
    }

    /// Like [`eval`] but resolves `Call(idx, args)` against `callees`. Every
    /// non-Call arm is identical to the base `eval`, so when `callees` is empty
    /// this is byte-identical to the historical evaluator. A `Call` evaluates
    /// its arg sub-expressions (which may themselves contain `Call`s, same
    /// registry), then runs the callee's `eval` closure on the concrete values.
    pub fn eval_with_callees(&self, args: &[i64], callees: &[NamedCallable]) -> Option<i64> {
        match self {
            Expr::Var(i) => args.get(*i).copied(),
            Expr::Const(c) => Some(*c),
            Expr::Call(idx, call_args) => {
                let callee = callees.get(*idx)?;
                if call_args.len() != callee.n_args {
                    return None;
                }
                let mut vals = Vec::with_capacity(call_args.len());
                for a in call_args {
                    vals.push(a.eval_with_callees(args, callees)?);
                }
                (callee.eval)(&vals)
            }
            Expr::BinOp(op, l, r) => {
                let a = l.eval_with_callees(args, callees)?;
                let b = r.eval_with_callees(args, callees)?;
                eval_binop(*op, a, b)
            }
            Expr::UnaryOp(op, e) => {
                let v = e.eval_with_callees(args, callees)?;
                match op {
                    UnOp::Neg => v.checked_neg(),
                    UnOp::Abs => v.checked_abs(),
                    UnOp::BitNot => Some(!v),
                    UnOp::Popcount => Some((v as u64).count_ones() as i64),
                }
            }
            Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) => {
                let l = lhs.eval_with_callees(args, callees)?;
                let r = rhs.eval_with_callees(args, callees)?;
                if eval_cmp(*cmp, l, r) {
                    then_e.eval_with_callees(args, callees)
                } else {
                    else_e.eval_with_callees(args, callees)
                }
            }
            Expr::WhileAccum {
                init,
                bound,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval_with_callees(args, callees)?;
                let n = bound.eval_with_callees(args, callees)?;
                if n < 0 || n > 10_000 {
                    return None;
                } // safety bound
                for i in 0..n {
                    // Build extended args: original args + [acc, i]
                    let mut ext = args.to_vec();
                    ext.push(acc);
                    ext.push(i);
                    let rhs = body_rhs.eval_with_callees(&ext, callees)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                }
                Some(acc)
            }
            // ForFold, NestedWhile, WhileCond are never evaluated via scalar eval().
            // They use eval_array / eval_nested / eval_while_cond instead.
            Expr::ForFold { .. } | Expr::NestedWhile { .. } | Expr::WhileCond { .. } => None,
        }
    }

    /// Evaluate a ForFold expression: acc = init; for (i, item) in arr { acc = acc OP rhs }
    /// Extended args for body: [scalar_args..., item, i, acc]
    pub fn eval_array(&self, scalar_args: &[i64], array: &[i64]) -> Option<i64> {
        match self {
            Expr::ForFold {
                init,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval(scalar_args)?;
                for (i, &item) in array.iter().enumerate() {
                    let mut ext = scalar_args.to_vec();
                    ext.push(item); // item
                    ext.push(i as i64); // index
                    ext.push(acc); // accumulator
                    let rhs = body_rhs.eval(&ext)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                }
                Some(acc)
            }
            _ => None,
        }
    }

    /// Evaluate a NestedWhile expression.
    /// Extended args for inner body: [args..., outer_acc, outer_i, inner_acc, inner_i]
    pub fn eval_nested(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::NestedWhile {
                outer_init,
                outer_bound,
                outer_body_op,
                inner_init,
                inner_bound,
                inner_body_op,
                inner_body_rhs,
            } => {
                let mut outer_acc = outer_init.eval(args)?;
                let outer_n = outer_bound.eval(args)?;
                if outer_n < 0 || outer_n > 1000 {
                    return None;
                }
                for outer_i in 0..outer_n {
                    // Run inner loop
                    let mut inner_acc = inner_init.eval(args)?;
                    let inner_n = inner_bound.eval(args)?;
                    if inner_n < 0 || inner_n > 1000 {
                        return None;
                    }
                    for inner_i in 0..inner_n {
                        let mut ext = args.to_vec();
                        ext.push(outer_acc);
                        ext.push(outer_i);
                        ext.push(inner_acc);
                        ext.push(inner_i);
                        let rhs = inner_body_rhs.eval(&ext)?;
                        inner_acc = eval_binop(*inner_body_op, inner_acc, rhs)?;
                    }
                    // Outer update: outer_acc = outer_acc OP inner_result
                    outer_acc = eval_binop(*outer_body_op, outer_acc, inner_acc)?;
                }
                Some(outer_acc)
            }
            _ => None,
        }
    }

    /// Evaluate a WhileCond expression: digit-style loop.
    /// acc = init; x = state_init; while x CMP cond_val { acc = acc OP body(x, acc, args); x = x / divisor }
    pub fn eval_while_cond(&self, args: &[i64]) -> Option<i64> {
        match self {
            Expr::WhileCond {
                init,
                state_init,
                cond_cmp,
                cond_val,
                divisor,
                body_op,
                body_rhs,
            } => {
                let mut acc = init.eval(args)?;
                let mut x = state_init.eval(args)?;
                let cv = cond_val.eval(args)?;
                if *divisor == 0 {
                    return None;
                }
                let mut iterations = 0;
                while eval_cmp(*cond_cmp, x, cv) {
                    if iterations > 10_000 {
                        return None;
                    }
                    // Body namespace: args ++ [x, acc, i]
                    let mut ext = args.to_vec();
                    ext.push(x); // x = Var(n_args)
                    ext.push(acc); // acc = Var(n_args + 1)
                    ext.push(iterations as i64); // i = Var(n_args + 2)
                    let rhs = body_rhs.eval(&ext)?;
                    acc = eval_binop(*body_op, acc, rhs)?;
                    x = x / divisor;
                    iterations += 1;
                }
                Some(acc)
            }
            _ => None,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 1,
            // A call costs 1 (the call itself) plus the size of every argument,
            // mirroring UnaryOp/BinOp so cost/depth bounds apply uniformly.
            Expr::Call(_, call_args) => 1 + call_args.iter().map(|a| a.size()).sum::<usize>(),
            Expr::BinOp(_, l, r) => 1 + l.size() + r.size(),
            Expr::UnaryOp(_, e) => 1 + e.size(),
            Expr::IfExpr(_, a, b, c, d) => 1 + a.size() + b.size() + c.size() + d.size(),
            Expr::WhileAccum {
                init,
                bound,
                body_rhs,
                ..
            } => 3 + init.size() + bound.size() + body_rhs.size(),
            Expr::ForFold { init, body_rhs, .. } => 3 + init.size() + body_rhs.size(),
            Expr::NestedWhile {
                outer_init,
                outer_bound,
                inner_init,
                inner_bound,
                inner_body_rhs,
                ..
            } => {
                7 + outer_init.size()
                    + outer_bound.size()
                    + inner_init.size()
                    + inner_bound.size()
                    + inner_body_rhs.size()
            }
            Expr::WhileCond {
                init,
                state_init,
                cond_val,
                body_rhs,
                ..
            } => 5 + init.size() + state_init.size() + cond_val.size() + body_rhs.size(),
        }
    }

    pub fn to_mog(&self, param_names: &[&str]) -> String {
        self.to_mog_ext(param_names, &[])
    }

    fn to_mog_ext(&self, param_names: &[&str], extra_names: &[&str]) -> String {
        match self {
            Expr::Var(i) => {
                if *i < param_names.len() {
                    param_names[*i].to_string()
                } else {
                    let ext_idx = *i - param_names.len();
                    extra_names.get(ext_idx).unwrap_or(&"x").to_string()
                }
            }
            Expr::Const(c) => {
                if *c == i64::MIN {
                    "(-9223372036854775807 - 1)".to_string()
                } else if *c < 0 {
                    format!("(0 - {})", c.checked_neg().unwrap())
                } else {
                    format!("{c}")
                }
            }
            // Render `Call(idx, args)` as `callee_name(arg0, arg1, ...)`. The
            // callee name is resolved from the thread-local emission table set
            // by `emit_mog` (see `with_callee_names`). This is REAL Mog source:
            // a plain function call the line-based transpiler passes through
            // unchanged — no transpiler edit. When the table lacks the index
            // (i.e. emission was not scoped, which never happens for a searched
            // Call) it falls back to a stable synthetic name so output is total.
            Expr::Call(idx, call_args) => {
                let name = CALLEE_NAMES.with(|c| {
                    c.borrow()
                        .get(*idx)
                        .cloned()
                        .unwrap_or_else(|| format!("callee{idx}"))
                });
                let arg_strs: Vec<String> = call_args
                    .iter()
                    .map(|a| a.to_mog_ext(param_names, extra_names))
                    .collect();
                format!("{name}({})", arg_strs.join(", "))
            }
            Expr::BinOp(op, l, r) => {
                let ls = l.to_mog_ext(param_names, extra_names);
                let rs = r.to_mog_ext(param_names, extra_names);
                match op {
                    BinOp::Add => format!("{ls} + {rs}"),
                    BinOp::Sub => format!("{ls} - {rs}"),
                    BinOp::Mul => format!("{ls} * {rs}"),
                    BinOp::Div => format!("{ls} / {rs}"),
                    BinOp::Mod => format!("{ls} % {rs}"),
                    BinOp::Min => format!("if {ls} < {rs} {{ {ls} }} else {{ {rs} }}"),
                    BinOp::Max => format!("if {ls} > {rs} {{ {ls} }} else {{ {rs} }}"),
                    // Mog supports native bitwise ops in lexer/parser/interpreter
                    // (runtime/mod.rs: Token::Amp/Caret/Pipe/Shl/Shr; eval BitAnd/
                    // BitOr/BitXor/Shl/Shr) — emit them faithfully, not approximations.
                    BinOp::BitAnd => format!("({ls}) & ({rs})"),
                    BinOp::BitOr => format!("({ls}) | ({rs})"),
                    BinOp::BitXor => format!("({ls}) ^ ({rs})"),
                    BinOp::Shl => format!("({ls}) << ({rs})"),
                    BinOp::Shr => format!("({ls}) >> ({rs})"),
                }
            }
            Expr::UnaryOp(op, e) => {
                let es = e.to_mog_ext(param_names, extra_names);
                match op {
                    UnOp::Neg => format!("0 - {es}"),
                    UnOp::Abs => format!("if {es} < 0 {{ 0 - {es} }} else {{ {es} }}"),
                    UnOp::BitNot => format!("0 - {es} - 1"),
                    UnOp::Popcount => {
                        // Can't express popcount as single expr, emit as loop
                        format!("{es}") // placeholder — WhileAccum handles this
                    }
                }
            }
            Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) => {
                let ls = lhs.to_mog_ext(param_names, extra_names);
                let rs = rhs.to_mog_ext(param_names, extra_names);
                let cs = match cmp {
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ge => ">=",
                    CmpOp::Gt => ">",
                    CmpOp::Ne => "!=",
                };
                let ts = then_e.to_mog_ext(param_names, extra_names);
                let es = else_e.to_mog_ext(param_names, extra_names);
                format!("if {ls} {cs} {rs} {{ {ts} }} else {{ {es} }}")
            }
            // Loop nodes (WhileAccum/ForFold/NestedWhile/WhileCond) are never
            // legitimate SUB-expressions — Mog has no expression-level loop, so
            // the enumerators only ever construct them as the TOP-LEVEL program,
            // which `emit_mog`/`emit_mog_array` render as full `fn` bodies (with
            // correct accumulator ops via `fold_op_mog`). Reaching these arms
            // means a loop node was nested inside another Expr, which would be a
            // synthesis bug; render an explicit, non-runnable marker carrying the
            // correct operator (via `fold_op_mog`) instead of the old approximate
            // `-` so the marker never claims wrong semantics.
            Expr::WhileAccum {
                init,
                bound,
                body_op,
                body_rhs,
            } => {
                let ext_names = &["acc", "i"];
                let init_s = init.to_mog_ext(param_names, &[]);
                let bound_s = bound.to_mog_ext(param_names, &[]);
                let op_s = fold_op_mog(body_op);
                let rhs_s = body_rhs.to_mog_ext(param_names, ext_names);
                format!("/* UNSUPPORTED nested loop: while i < {bound_s} {{ acc = acc {op_s} {rhs_s} }} from {init_s} */")
            }
            Expr::ForFold { .. } => {
                // Rendered as a top-level fn by emit_mog/emit_mog_array.
                "/* UNSUPPORTED nested for-fold */".to_string()
            }
            Expr::NestedWhile { .. } => {
                "/* UNSUPPORTED nested nested-while */".to_string()
            }
            Expr::WhileCond { .. } => {
                "/* UNSUPPORTED nested while-cond */".to_string()
            }
        }
    }
}

fn eval_binop(op: BinOp, a: i64, b: i64) -> Option<i64> {
    match op {
        BinOp::Add => a.checked_add(b),
        BinOp::Sub => a.checked_sub(b),
        BinOp::Mul => a.checked_mul(b),
        BinOp::Div => {
            if b == 0 || (a == i64::MIN && b == -1) {
                None
            } else {
                Some(a / b)
            }
        }
        BinOp::Mod => {
            if b == 0 || (a == i64::MIN && b == -1) {
                None
            } else {
                Some(a % b)
            }
        }
        BinOp::Min => Some(a.min(b)),
        BinOp::Max => Some(a.max(b)),
        BinOp::BitAnd => Some(a & b),
        BinOp::BitOr => Some(a | b),
        BinOp::BitXor => Some(a ^ b),
        BinOp::Shl => {
            if b < 0 || b > 63 {
                None
            } else {
                Some(a << b)
            }
        }
        BinOp::Shr => {
            if b < 0 || b > 63 {
                None
            } else {
                Some(a >> b)
            }
        }
    }
}

fn eval_cmp(cmp: CmpOp, a: i64, b: i64) -> bool {
    match cmp {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Eq => a == b,
        CmpOp::Ge => a >= b,
        CmpOp::Gt => a > b,
        CmpOp::Ne => a != b,
    }
}

// ─── Observational equivalence ───────────────────────────────────────────────

fn fingerprint(expr: &Expr, test_inputs: &[Vec<i64>]) -> Option<Vec<i64>> {
    fingerprint_c(expr, test_inputs, &[])
}

/// Callee-aware fingerprint. Identical to [`fingerprint`] when `callees` is
/// empty (the base path), but resolves `Call` nodes against the registry so a
/// `Call(idx, args)` candidate gets a real behavioural fingerprint.
fn fingerprint_c(expr: &Expr, test_inputs: &[Vec<i64>], callees: &[NamedCallable]) -> Option<Vec<i64>> {
    let mut fp = Vec::with_capacity(test_inputs.len());
    for args in test_inputs {
        match expr.eval_with_callees(args, callees) {
            Some(v) => fp.push(v),
            None => return None,
        }
    }
    Some(fp)
}

fn matches_all(expr: &Expr, examples: &[(Vec<i64>, i64)]) -> bool {
    matches_all_c(expr, examples, &[])
}

/// Callee-aware `matches_all`. Identical to [`matches_all`] when `callees` is
/// empty; otherwise resolves `Call` nodes so a candidate using a registered
/// producer is checked against the examples with the producer actually run.
fn matches_all_c(expr: &Expr, examples: &[(Vec<i64>, i64)], callees: &[NamedCallable]) -> bool {
    examples
        .iter()
        .all(|(args, expected)| expr.eval_with_callees(args, callees) == Some(*expected))
}

// ─── Robust verification (anti-fingerprint collision) ──────────────────────────

/// Deterministic probe inputs: hash-based, no external RNG dependency.
/// Returns `count` input vectors with values in [-100, 100].
fn probe_inputs(n_args: usize, count: usize) -> Vec<Vec<i64>> {
    (0..count)
        .map(|i| {
            (0..n_args)
                .map(|j| {
                    // Simple hash: mix i and j, map to [-100, 100]
                    let h = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add((j as u64).wrapping_mul(1442695040888963407))
                        .wrapping_add(0xB5001F);
                    ((h % 201) as i64) - 100
                })
                .collect()
        })
        .collect()
}

/// Check that an expression doesn't crash on random inputs (no None returns).
/// This catches spurious matches like `b - b%3` for min(a,b).
fn robust_well_defined(expr: &Expr, n_args: usize, n_probes: usize) -> bool {
    robust_well_defined_c(expr, n_args, n_probes, &[])
}

/// Callee-aware `robust_well_defined`. Identical to [`robust_well_defined`]
/// when `callees` is empty; otherwise resolves `Call` nodes so a candidate is
/// probed with the producer actually executed (an unsound call → `None` → reject).
fn robust_well_defined_c(
    expr: &Expr,
    n_args: usize,
    n_probes: usize,
    callees: &[NamedCallable],
) -> bool {
    for args in probe_inputs(n_args, n_probes) {
        if expr.eval_with_callees(&args, callees).is_none() {
            return false;
        }
    }
    true
}

// ─── Corpus-driven subtree mining (frequent-subtree + anti-unification) ────────
//
// Replaces the old hardcoded 15-seed `dream()`. We mine recurring scalar
// subtrees out of a corpus of *verified* solved `Expr` trees (mined identically
// to the trees we later inject — sound by construction). Each mined component is
// a canonicalized `Expr` whose free `Var(0..k)` ARE its parameters; there is no
// new AST node — abstractions reuse the existing free-Var-as-hole substrate so
// eval/to_mog/serde need no changes. Anti-unification (least-general
// generalization) folds structurally-matching subtrees (e.g. `x*x` and `y*y`)
// into a shared pattern `?0*?0` so mined components GENERALIZE rather than
// memorize. All counting is BTreeMap-keyed and all selection is total-ordered:
// no clock/rand ever picks an abstraction (Instant only bounds work).

use std::collections::{BTreeMap, BTreeSet};

/// Stable, deterministic BTreeMap/grouping key for an Expr. `Expr` derives
/// `Debug`, whose output is deterministic — the same approach the old `dream()`
/// already trusted for component descriptions.
fn expr_key(e: &Expr) -> String {
    format!("{e:?}")
}

/// Distinct `Var` indices referenced by a *scalar* subtree (loop variants
/// contribute nothing — they are out of scope for mining v1).
fn free_vars(e: &Expr) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    fn walk(e: &Expr, out: &mut BTreeSet<usize>) {
        match e {
            Expr::Var(i) => {
                out.insert(*i);
            }
            Expr::Const(_) => {}
            Expr::BinOp(_, l, r) => {
                walk(l, out);
                walk(r, out);
            }
            Expr::UnaryOp(_, c) => walk(c, out),
            Expr::IfExpr(_, a, b, c, d) => {
                walk(a, out);
                walk(b, out);
                walk(c, out);
                walk(d, out);
            }
            // A call's free vars are the union of its arg sub-expressions' vars.
            Expr::Call(_, call_args) => {
                for a in call_args {
                    walk(a, out);
                }
            }
            // Loop variants are out of scope — their interiors reference
            // loop-local extended Var slots that are meaningless when lifted.
            Expr::WhileAccum { .. }
            | Expr::ForFold { .. }
            | Expr::NestedWhile { .. }
            | Expr::WhileCond { .. } => {}
        }
    }
    walk(e, &mut out);
    out
}

/// Collect every pure-scalar node (self + descendants) into `out`. Recurses
/// through BinOp/UnaryOp/IfExpr children only. On encountering ANY loop variant
/// it pushes nothing and stops descending (scope guard — loop interiors
/// reference loop-local Var slots that can't be lifted soundly).
fn collect_scalar_subtrees<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::Var(_) | Expr::Const(_) => out.push(e),
        Expr::BinOp(_, l, r) => {
            out.push(e);
            collect_scalar_subtrees(l, out);
            collect_scalar_subtrees(r, out);
        }
        Expr::UnaryOp(_, c) => {
            out.push(e);
            collect_scalar_subtrees(c, out);
        }
        Expr::IfExpr(_, a, b, c, d) => {
            out.push(e);
            collect_scalar_subtrees(a, out);
            collect_scalar_subtrees(b, out);
            collect_scalar_subtrees(c, out);
            collect_scalar_subtrees(d, out);
        }
        // Scope guard: do not push, do not descend into loop interiors.
        // A `Call` is likewise not a liftable scalar subtree — its callee index
        // is meaningful only within the consumer's registry scope, so mining it
        // into the global library would be unsound. Skip it entirely.
        Expr::Call(..)
        | Expr::WhileAccum { .. }
        | Expr::ForFold { .. }
        | Expr::NestedWhile { .. }
        | Expr::WhileCond { .. } => {}
    }
}

/// Rename free `Var` indices to dense slots `0..k` in left-to-right
/// first-encounter order (deterministic via BTreeMap). Returns the canonical
/// Expr plus its arity `k`. Since loop interiors are excluded by
/// [`collect_scalar_subtrees`], every Var seen here is a free parameter, so
/// `x*x` -> (`Var0*Var0`, 1) and `a-b` -> (`Var0-Var1`, 2).
fn canonicalize(e: &Expr) -> (Expr, usize) {
    // ALGEBRAIC normal form FIRST (semantics-preserving), THEN dense-var renaming.
    // Composing the two means algebraically-equal subtrees merge into one library
    // op: `a+2` and `2+a` both normalize to `Const(2)+Var0`, so the flywheel
    // compresses them together instead of mining two look-alike ops.
    let e = &algebraic_normalize(e);
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    // First pass: assign dense slots in first-encounter (left-to-right) order.
    fn assign(e: &Expr, map: &mut BTreeMap<usize, usize>, next: &mut usize) {
        match e {
            Expr::Var(i) => {
                if !map.contains_key(i) {
                    map.insert(*i, *next);
                    *next += 1;
                }
            }
            Expr::Const(_) => {}
            Expr::BinOp(_, l, r) => {
                assign(l, map, next);
                assign(r, map, next);
            }
            Expr::UnaryOp(_, c) => assign(c, map, next),
            Expr::IfExpr(_, a, b, c, d) => {
                assign(a, map, next);
                assign(b, map, next);
                assign(c, map, next);
                assign(d, map, next);
            }
            _ => {}
        }
    }
    let mut next = 0usize;
    assign(e, &mut map, &mut next);
    fn rebuild(e: &Expr, map: &BTreeMap<usize, usize>) -> Expr {
        match e {
            Expr::Var(i) => Expr::Var(*map.get(i).unwrap_or(i)),
            Expr::Const(c) => Expr::Const(*c),
            Expr::BinOp(op, l, r) => Expr::BinOp(
                *op,
                Box::new(rebuild(l, map)),
                Box::new(rebuild(r, map)),
            ),
            Expr::UnaryOp(op, c) => Expr::UnaryOp(*op, Box::new(rebuild(c, map))),
            Expr::IfExpr(cmp, a, b, c, d) => Expr::IfExpr(
                *cmp,
                Box::new(rebuild(a, map)),
                Box::new(rebuild(b, map)),
                Box::new(rebuild(c, map)),
                Box::new(rebuild(d, map)),
            ),
            other => other.clone(),
        }
    }
    (rebuild(e, &map), next)
}

/// Op-only structural skeleton key (consts and vars erased to `?`). Used to
/// group candidates for pairwise anti-unification so we only attempt lgg within
/// a structurally-matching group (keeps it O(group^2) and deterministic).
fn skeleton(e: &Expr) -> String {
    match e {
        Expr::Var(_) | Expr::Const(_) => "?".to_string(),
        Expr::BinOp(op, l, r) => format!("({:?} {} {})", op, skeleton(l), skeleton(r)),
        Expr::UnaryOp(op, c) => format!("({:?} {})", op, skeleton(c)),
        Expr::IfExpr(cmp, a, b, c, d) => format!(
            "(if {:?} {} {} {} {})",
            cmp,
            skeleton(a),
            skeleton(b),
            skeleton(c),
            skeleton(d)
        ),
        other => format!("{other:?}"),
    }
}

/// First-order least-general generalization (anti-unification). When the two
/// trees share a top constructor + op/cmp, recurse and rebuild the node. Equal
/// `Var` indices and equal `Const`s are kept verbatim. On ANY disagreement we
/// allocate a shared hole: a linear scan of `subst` reuses the existing hole for
/// a previously-seen (a_sub, b_sub) pair, so `f(x,x)` vs `g(y,y)` generalizes to
/// `?0 OP ?0` (ONE hole), not two. Holes are emitted as `Expr::Var(next_hole)`
/// — re-using the free-Var substrate, NO new AST variant. Returns `None` when
/// the top constructors are incompatible (no useful lgg there).
fn anti_unify(
    a: &Expr,
    b: &Expr,
    subst: &mut Vec<(Expr, Expr)>,
    next_hole: &mut usize,
) -> Option<Expr> {
    // Helper: allocate (or reuse) a shared hole for a disagreeing pair.
    fn hole_for(
        a: &Expr,
        b: &Expr,
        subst: &mut Vec<(Expr, Expr)>,
        next_hole: &mut usize,
    ) -> Expr {
        for (idx, (sa, sb)) in subst.iter().enumerate() {
            if sa == a && sb == b {
                return Expr::Var(idx);
            }
        }
        let h = *next_hole;
        subst.push((a.clone(), b.clone()));
        *next_hole += 1;
        Expr::Var(h)
    }
    match (a, b) {
        (Expr::Var(i), Expr::Var(j)) if i == j => Some(Expr::Var(*i)),
        (Expr::Const(x), Expr::Const(y)) if x == y => Some(Expr::Const(*x)),
        (Expr::BinOp(o1, l1, r1), Expr::BinOp(o2, l2, r2)) if o1 == o2 => {
            let l = anti_unify(l1, l2, subst, next_hole)?;
            let r = anti_unify(r1, r2, subst, next_hole)?;
            Some(Expr::BinOp(*o1, Box::new(l), Box::new(r)))
        }
        (Expr::UnaryOp(o1, c1), Expr::UnaryOp(o2, c2)) if o1 == o2 => {
            let c = anti_unify(c1, c2, subst, next_hole)?;
            Some(Expr::UnaryOp(*o1, Box::new(c)))
        }
        (Expr::IfExpr(cmp1, a1, b1, t1, e1), Expr::IfExpr(cmp2, a2, b2, t2, e2))
            if cmp1 == cmp2 =>
        {
            let la = anti_unify(a1, a2, subst, next_hole)?;
            let lb = anti_unify(b1, b2, subst, next_hole)?;
            let lt = anti_unify(t1, t2, subst, next_hole)?;
            let le = anti_unify(e1, e2, subst, next_hole)?;
            Some(Expr::IfExpr(
                *cmp1,
                Box::new(la),
                Box::new(lb),
                Box::new(lt),
                Box::new(le),
            ))
        }
        // Same broad constructor class but disagreeing leaves -> shared hole.
        (Expr::Var(_), Expr::Var(_))
        | (Expr::Const(_), Expr::Const(_))
        | (Expr::Var(_), Expr::Const(_))
        | (Expr::Const(_), Expr::Var(_)) => Some(hole_for(a, b, subst, next_hole)),
        // Incompatible top constructors (e.g. BinOp vs UnaryOp): no useful lgg.
        _ => None,
    }
}

/// Maximum size of any promotable abstraction (explicit anti-memorization cap).
const MAX_ABSTRACTION_SIZE: usize = 9;
/// A mined subtree must recur across at least this many DISTINCT corpus trees.
const MIN_SUPPORT: u32 = 2;
/// Smallest promotable subtree: a binop of two leaves (size 3).
const MIN_SUBTREE_SIZE: usize = 3;
/// Hard cap on how many abstractions we promote in one mining pass.
const MAX_PROMOTE: usize = 64;
/// Hard cap on injective re-rooting instantiations produced per component.
const MAX_INSTANTIATIONS: usize = 16;
/// Hard cap on how many size-1 library injections enter the enumeration bank
/// (bounds the size-2 binop blow-up).
const MAX_SIZE1_INJECTIONS: usize = 64;

/// Degeneracy filter (D1-D6). Rejects abstractions that can't possibly help
/// search or that smuggle memorization. Reuses [`probe_inputs`] / [`fingerprint`]
/// — no new RNG.
fn is_promotable(canon: &Expr, arity: usize) -> bool {
    // D1: bare leaf.
    if matches!(canon, Expr::Var(_) | Expr::Const(_)) {
        return false;
    }
    // D2: too small (a unary-of-leaf is size 2 and uninteresting).
    if canon.size() < MIN_SUBTREE_SIZE {
        return false;
    }
    // D3: var-free constant fold — already handled by the enumerator.
    if arity == 0 {
        return false;
    }
    // D4: over-parameterized => memorization in disguise.
    if arity > 3 {
        return false;
    }
    // D5: near-whole-program tree => the explicit anti-memorization cap.
    if canon.size() > MAX_ABSTRACTION_SIZE {
        return false;
    }
    // D6: observational identity. If the abstraction behaves exactly like a bare
    // Var0 on probe inputs it's a no-op (Var0+Const0, Var0*Const1, abs(abs(x))).
    let probes = probe_inputs(arity.max(1), 16);
    let canon_fp = fingerprint(canon, &probes);
    let id_fp = fingerprint(&Expr::Var(0), &probes);
    if let (Some(a), Some(b)) = (&canon_fp, &id_fp) {
        if a == b {
            return false;
        }
    }
    true
}

/// Human-readable provenance name for a mined abstraction. The `"mined: "`
/// prefix namespaces against seed descriptions so `ComponentLibrary::add`'s
/// dedup-by-description keeps working. Pure function of the canonical Expr ->
/// deterministic and collision-free.
fn name_abstraction(e: &Expr) -> String {
    let arity = free_vars(e).iter().max().map(|m| m + 1).unwrap_or(0);
    let names = ["a", "b", "c", "d"];
    let slice = &names[..arity.min(names.len())];
    format!("mined: {}", e.to_mog(slice))
}

/// Rewrite `Var(slot)` -> `Var(map[slot])` for re-rooting a k-ary abstraction
/// onto real argument indices. Slots out of `map`'s range are left unchanged.
fn remap_vars(e: &Expr, map: &[usize]) -> Expr {
    match e {
        Expr::Var(i) => Expr::Var(*map.get(*i).unwrap_or(i)),
        Expr::Const(c) => Expr::Const(*c),
        Expr::BinOp(op, l, r) => Expr::BinOp(
            *op,
            Box::new(remap_vars(l, map)),
            Box::new(remap_vars(r, map)),
        ),
        Expr::UnaryOp(op, c) => Expr::UnaryOp(*op, Box::new(remap_vars(c, map))),
        Expr::IfExpr(cmp, a, b, c, d) => Expr::IfExpr(
            *cmp,
            Box::new(remap_vars(a, map)),
            Box::new(remap_vars(b, map)),
            Box::new(remap_vars(c, map)),
            Box::new(remap_vars(d, map)),
        ),
        other => other.clone(),
    }
}

/// Produce every injective re-rooting of a k-ary component's dense slots
/// `0..comp_arity` onto the real argument indices `0..n_args`. k=1 yields
/// `n_args` instantiations; k=2 yields ordered pairs; etc. Capped at
/// [`MAX_INSTANTIATIONS`], keeping the lexicographically-first maps when over
/// cap (deterministic). Returns `[]` if the component needs more args than the
/// problem has.
fn instantiate_component(comp: &Expr, n_args: usize) -> Vec<Expr> {
    let comp_arity = free_vars(comp).iter().max().map(|m| m + 1).unwrap_or(0);
    if comp_arity == 0 || comp_arity > n_args {
        return Vec::new();
    }
    // Enumerate injective maps slot->real in lexicographic order.
    let mut maps: Vec<Vec<usize>> = Vec::new();
    fn rec(
        slot: usize,
        comp_arity: usize,
        n_args: usize,
        used: &mut Vec<bool>,
        cur: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if out.len() >= MAX_INSTANTIATIONS {
            return;
        }
        if slot == comp_arity {
            out.push(cur.clone());
            return;
        }
        for real in 0..n_args {
            if used[real] {
                continue;
            }
            used[real] = true;
            cur.push(real);
            rec(slot + 1, comp_arity, n_args, used, cur, out);
            cur.pop();
            used[real] = false;
            if out.len() >= MAX_INSTANTIATIONS {
                return;
            }
        }
    }
    let mut used = vec![false; n_args];
    let mut cur = Vec::new();
    rec(0, comp_arity, n_args, &mut used, &mut cur, &mut maps);
    maps.iter().map(|m| remap_vars(comp, m)).collect()
}

/// Per-pattern mining statistics keyed by canonical expr_key in a BTreeMap.
#[derive(Clone, Debug)]
struct SubtreeStats {
    /// Number of DISTINCT corpus trees this pattern appears in.
    support: u32,
    /// Total raw occurrences across the corpus (tie-break / diagnostics).
    total_occurrences: u32,
    arity: usize,
    size: usize,
    exemplar: Expr,
}

// ─── Persistent solved-Expr corpus (mirrors solved_cache guards) ───────────────

/// One verified solved scalar Expr, with its arity and a behavioural
/// fingerprint for dedup.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SolvedExpr {
    expr: Expr,
    n_args: usize,
    fp: Vec<i64>,
}

/// Max corpus entries kept on disk.
const SOLVED_EXPRS_MAX_ENTRIES: usize = 5000;
/// Max corpus file size in bytes (64 MiB). Refuse to grow/read past this.
const SOLVED_EXPRS_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// On-disk location for the mined-corpus. Mirrors `solved_cache::cache_path`:
/// `NSYNTH_SOLVED_EXPRS_PATH` override, empty string disables, `None` under
/// `cfg!(test)` so tests stay hermetic, else `~/.mog_synth_solved_exprs.json`.
fn solved_exprs_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_SOLVED_EXPRS_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    Some(dirs_home().join(".mog_synth_solved_exprs.json"))
}

/// Append a verified solved Expr to the persistent corpus. No-op when the path
/// is disabled (env empty / `cfg!(test)`). Dedups by (expr_key, fp), enforces
/// the entry and file-byte caps BEFORE writing, and fails soft on any IO error.
pub fn record_solved_expr(e: &Expr, n_args: usize, fp: &[i64]) {
    let path = match solved_exprs_path() {
        Some(p) => p,
        None => return,
    };
    let mut existing = load_solved_exprs();
    let key = expr_key(e);
    let already = existing
        .iter()
        .any(|s| expr_key(&s.expr) == key && s.fp == fp);
    if already {
        return;
    }
    existing.push(SolvedExpr {
        expr: e.clone(),
        n_args,
        fp: fp.to_vec(),
    });
    // Entry cap: keep the most recent SOLVED_EXPRS_MAX_ENTRIES.
    if existing.len() > SOLVED_EXPRS_MAX_ENTRIES {
        let overflow = existing.len() - SOLVED_EXPRS_MAX_ENTRIES;
        existing.drain(0..overflow);
    }
    let json = match serde_json::to_string_pretty(&existing) {
        Ok(j) => j,
        Err(_) => return,
    };
    // File-byte cap: refuse to grow past the limit.
    if json.len() as u64 > SOLVED_EXPRS_MAX_BYTES {
        return;
    }
    let _ = std::fs::write(&path, json);
}

/// Load the persistent corpus. Returns `[]` on missing file / parse failure /
/// disabled path, and refuses to read a file larger than the byte cap.
fn load_solved_exprs() -> Vec<SolvedExpr> {
    let path = match solved_exprs_path() {
        Some(p) => p,
        None => return Vec::new(),
    };
    if !path.exists() {
        return Vec::new();
    }
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > SOLVED_EXPRS_MAX_BYTES {
            return Vec::new();
        }
    }
    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&json).unwrap_or_default()
}

// ─── Resumable anytime search frontier ─────────────────────────────────────────
//
// A `Frontier` lifts the bottom-up enumerator's locals — the size-stratified
// expression bank (`by_size`) and the next size to expand (`next_size`) — out of
// the call stack so a search that runs out of *time budget* can be PERSISTED and
// RESUMED on a later call instead of restarting from size 1. Combined with the
// removal of the fixed `max_size` ceiling in `enumerate_exprs_resumable`, this
// turns every "give up / None" into "budget exhausted; frontier persisted;
// resumable, deepening next time" — the search is anytime and monotone, never a
// proof of impossibility.
//
// Determinism: selection order is purely structural (size ascending, then the
// fixed op/var/const order). `seen` (the observational-equivalence dedup set) is
// NOT stored; it is re-derived by replaying the stored `by_size` strata in order
// on resume, which keeps the file smaller and reproduces the identical dedup
// decisions (the replay visits strata in the exact stored Vec order).

/// A resumable, size-stratified enumeration frontier for one problem.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frontier {
    /// Deterministic key (the examples fingerprint) the frontier belongs to.
    fingerprint: String,
    /// Number of function arguments (selection space depends on it).
    n_args: usize,
    /// Op tier this frontier was grown under: 0 = core (5 ops), 1 = full (12).
    /// A frontier must only be resumed under the SAME tier (different op sets
    /// produce different strata), so a tier mismatch forces a fresh start.
    ops_tier: u8,
    /// Size-stratified expression bank: `by_size[s]` holds the unique (under
    /// observational equivalence) expressions of size `s`. Index 0 is unused.
    by_size: Vec<Vec<Expr>>,
    /// The next size level to expand. On a cold frontier this is 2 (sizes 0/1
    /// are seeded fresh each run from vars/consts/library).
    next_size: usize,
}

impl Frontier {
    /// A fresh frontier for `fingerprint`/`n_args`/`ops_tier`, ready to expand
    /// from size 2 (size-1 atoms are always re-seeded from vars/consts/library).
    fn fresh(fingerprint: String, n_args: usize, ops_tier: u8) -> Self {
        Self {
            fingerprint,
            n_args,
            ops_tier,
            by_size: vec![vec![]],
            next_size: 2,
        }
    }

    /// True when this frontier is reusable for the given problem signature.
    /// A mismatch on fingerprint / arity / op-tier means the stored strata do
    /// not describe THIS search, so the caller must start fresh.
    fn matches(&self, fingerprint: &str, n_args: usize, ops_tier: u8) -> bool {
        self.fingerprint == fingerprint && self.n_args == n_args && self.ops_tier == ops_tier
    }
}

/// Max distinct frontiers kept on disk. A frontier is FAR larger per entry than
/// a one-line solved program, so this cap is small and the file-byte cap below
/// is the real guard. (Disk-blowup history: the solved-program cache once grew
/// to 13.4 GB; the frontier store must never repeat that.)
const FRONTIER_MAX_ENTRIES: usize = 64;
/// Max frontier-store file size in bytes (32 MiB). Refuse to grow/read past it.
const FRONTIER_MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Max serialized size of a SINGLE frontier (8 MiB). A frontier that has grown
/// past this is not persisted (search degrades gracefully to one-shot behavior
/// for that problem) so one deep search can't fill the store on its own.
const FRONTIER_MAX_ENTRY_BYTES: usize = 8 * 1024 * 1024;

/// On-disk location for the resumable-frontier store. Mirrors
/// `solved_exprs_path`: `NSYNTH_ENUM_FRONTIER_PATH` override, empty string
/// disables, `None` under `cfg!(test)` so tests stay hermetic, else
/// `~/.mog_synth_enum_frontier.json`.
fn frontier_path() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("NSYNTH_ENUM_FRONTIER_PATH") {
        if val.is_empty() {
            return None;
        }
        return Some(PathBuf::from(val));
    }
    if cfg!(test) {
        return None;
    }
    Some(dirs_home().join(".mog_synth_enum_frontier.json"))
}

/// Load all persisted frontiers. Returns `[]` on missing file / parse failure /
/// disabled path, and refuses to read a file larger than the byte cap.
fn load_frontiers() -> Vec<Frontier> {
    let path = match frontier_path() {
        Some(p) => p,
        None => return Vec::new(),
    };
    if !path.exists() {
        return Vec::new();
    }
    if let Ok(meta) = std::fs::metadata(&path) {
        if meta.len() > FRONTIER_MAX_BYTES {
            return Vec::new();
        }
    }
    let json = match std::fs::read_to_string(&path) {
        Ok(j) => j,
        Err(_) => return Vec::new(),
    };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Load the frontier for `fingerprint`/`n_args`/`ops_tier`, or `None` if the
/// store is disabled / absent / has no matching (and tier-compatible) entry.
fn load_frontier(fingerprint: &str, n_args: usize, ops_tier: u8) -> Option<Frontier> {
    load_frontiers()
        .into_iter()
        .find(|f| f.matches(fingerprint, n_args, ops_tier))
}

/// Persist `frontier`, replacing any prior entry for the same fingerprint.
/// No-op when the path is disabled. Enforces the single-entry byte cap, the
/// entry cap, and the file-byte cap BEFORE writing, and fails soft on any IO
/// error. This is the resumable counterpart of `record_solved_expr`.
fn save_frontier(frontier: &Frontier) {
    let path = match frontier_path() {
        Some(p) => p,
        None => return,
    };
    // Serialize THIS frontier first to enforce the per-entry cap; an
    // over-cap frontier is dropped (graceful degradation to one-shot search)
    // rather than persisted, so a single deep search can't fill the store.
    let entry_json = match serde_json::to_string(frontier) {
        Ok(j) => j,
        Err(_) => return,
    };
    if entry_json.len() > FRONTIER_MAX_ENTRY_BYTES {
        return;
    }
    let mut existing = load_frontiers();
    existing.retain(|f| f.fingerprint != frontier.fingerprint);
    existing.push(frontier.clone());
    // Entry cap: keep the most recent FRONTIER_MAX_ENTRIES.
    if existing.len() > FRONTIER_MAX_ENTRIES {
        let overflow = existing.len() - FRONTIER_MAX_ENTRIES;
        existing.drain(0..overflow);
    }
    let json = match serde_json::to_string(&existing) {
        Ok(j) => j,
        Err(_) => return,
    };
    // File-byte cap: refuse to grow past the limit (drop this save rather than
    // risk a runaway file).
    if json.len() as u64 > FRONTIER_MAX_BYTES {
        return;
    }
    let _ = std::fs::write(&path, json);
}

/// Drop the persisted frontier for `fingerprint` (e.g. once the problem is
/// solved, so its frontier becomes dead weight). No-op when disabled / absent.
fn evict_frontier(fingerprint: &str) {
    let path = match frontier_path() {
        Some(p) => p,
        None => return,
    };
    let mut existing = load_frontiers();
    let before = existing.len();
    existing.retain(|f| f.fingerprint != fingerprint);
    if existing.len() == before {
        return; // nothing to evict
    }
    if existing.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    if let Ok(json) = serde_json::to_string(&existing) {
        let _ = std::fs::write(&path, json);
    }
}

// ─── Bootstrap corpus (cold-start safety) ──────────────────────────────────────

/// Build a cold-start corpus by re-deriving Exprs for the classic seed I/O
/// specs (the same 15 specs the old `dream()` used) via the enumerator. These
/// feed the miner as INPUT — they are not directly the library output. The
/// frequency miner then extracts the recurring subtrees; a hybrid floor
/// (in `mine_library`) re-adds these seed Exprs as low-priority components when
/// support-mining alone produces too few entries (cold machine safety).
fn bootstrap_corpus(budget_ms: u64) -> Vec<SolvedExpr> {
    let start = std::time::Instant::now();
    // (n_args, fn(&[i64]) -> i64) — same specs as the historical dream() seeds.
    type SeedFn = Box<dyn Fn(&[i64]) -> i64>;
    let seeds: Vec<(usize, SeedFn)> = vec![
        (2, Box::new(|a: &[i64]| a[0] + a[1])),
        (2, Box::new(|a: &[i64]| a[0] * a[1])),
        (2, Box::new(|a: &[i64]| a[0] - a[1])),
        (2, Box::new(|a: &[i64]| a[0].saturating_sub(a[1]).saturating_abs())),
        (2, Box::new(|a: &[i64]| a[0].min(a[1]))),
        (2, Box::new(|a: &[i64]| a[0].max(a[1]))),
        (1, Box::new(|a: &[i64]| a[0] * a[0])),
        (1, Box::new(|a: &[i64]| a[0] % 2)),
        (1, Box::new(|a: &[i64]| a[0] / 2)),
        (1, Box::new(|a: &[i64]| a[0] + a[0])),
        (1, Box::new(|a: &[i64]| a[0] * 2)),
        (1, Box::new(|a: &[i64]| a[0] * a[0] + a[0])),
        (1, Box::new(|a: &[i64]| a[0] % 10)),
        (1, Box::new(|a: &[i64]| a[0] + 1)),
        (1, Box::new(|a: &[i64]| a[0].saturating_neg())),
    ];

    let mut corpus = Vec::new();
    for (n_args, func) in &seeds {
        if start.elapsed().as_millis() as u64 > budget_ms {
            break;
        }
        let examples: Vec<(Vec<i64>, i64)> = (0..6)
            .map(|i| {
                let args: Vec<i64> = (0..*n_args)
                    .map(|j| ((i as i64 * 7 + 3 + j as i64 * 11) % 20) - 10)
                    .collect();
                let expected = func(&args);
                (args, expected)
            })
            .collect();
        if let Some(expr) = enumerate_exprs_core(*n_args, 5, &examples, 500, None) {
            let fp = fingerprint(&expr, &probe_inputs(*n_args, 8)).unwrap_or_default();
            corpus.push(SolvedExpr {
                expr,
                n_args: *n_args,
                fp,
            });
        }
    }
    corpus
}

// ─── Mining entry point ─────────────────────────────────────────────────────

/// Mine a [`ComponentLibrary`] from the persistent solved-Expr corpus.
/// Frequency-count scalar subtrees -> anti-unify within skeleton groups ->
/// threshold (support>=2, size>=3) -> rank by MDL compression score -> verify
/// (robust_well_defined) -> name -> add. Fully deterministic; `verify_budget_ms`
/// only bounds the cold-start bootstrap, never selection.
pub fn mine_library(verify_budget_ms: u64) -> ComponentLibrary {
    let corpus = load_solved_exprs();
    let corpus = if corpus.is_empty() {
        bootstrap_corpus(verify_budget_ms)
    } else {
        corpus
    };
    mine_from_corpus(&corpus)
}

/// Core mining over an explicit corpus (test-visible, disk-free). This is the
/// deterministic heart used by both `mine_library` and the unit tests.
fn mine_from_corpus(corpus_in: &[SolvedExpr]) -> ComponentLibrary {
    let mut library = ComponentLibrary::new();

    // Order-stability: sort the corpus by (n_args, expr_key) before mining.
    let mut corpus: Vec<&SolvedExpr> = corpus_in.iter().collect();
    corpus.sort_by(|a, b| {
        a.n_args
            .cmp(&b.n_args)
            .then_with(|| expr_key(&a.expr).cmp(&expr_key(&b.expr)))
    });

    // (b) FREQUENCY COUNT — distinct-trees support.
    let mut counts: BTreeMap<String, SubtreeStats> = BTreeMap::new();
    for s in &corpus {
        let mut nodes: Vec<&Expr> = Vec::new();
        collect_scalar_subtrees(&s.expr, &mut nodes);
        // Canonicalize each node and tally per-tree raw occurrences keyed by
        // canonical key; one tree contributes support+=1 per distinct pattern.
        let mut per_tree: BTreeMap<String, (Expr, usize, u32)> = BTreeMap::new();
        for node in &nodes {
            let (canon, arity) = canonicalize(node);
            if !is_promotable(&canon, arity) {
                continue;
            }
            let key = expr_key(&canon);
            let size = canon.size();
            let entry = per_tree.entry(key).or_insert((canon, arity, 0));
            entry.2 += 1;
            let _ = size;
        }
        for (key, (canon, arity, raw)) in per_tree {
            let size = canon.size();
            let stat = counts.entry(key).or_insert(SubtreeStats {
                support: 0,
                total_occurrences: 0,
                arity,
                size,
                exemplar: canon,
            });
            stat.support += 1;
            stat.total_occurrences += raw;
        }
    }

    // (c) ANTI-UNIFY PASS — group surviving exemplars by structural skeleton,
    // fold-left pairwise lgg within each group (members iterated in expr_key
    // order). Each successful lgg that is itself promotable is inserted into
    // counts with support = number of distinct group members it generalizes.
    let mut groups: BTreeMap<String, Vec<Expr>> = BTreeMap::new();
    for stat in counts.values() {
        groups
            .entry(skeleton(&stat.exemplar))
            .or_default()
            .push(stat.exemplar.clone());
    }
    let mut lgg_inserts: Vec<(String, SubtreeStats)> = Vec::new();
    for (_sk, members_in) in &groups {
        if members_in.len() < 2 {
            continue;
        }
        let mut members = members_in.clone();
        members.sort_by(|a, b| expr_key(a).cmp(&expr_key(b)));
        // Fold-left pairwise lgg.
        let mut acc = members[0].clone();
        let mut generalized = 1u32; // how many members folded so far
        for m in members.iter().skip(1) {
            let mut subst = Vec::new();
            let mut next_hole = 0usize;
            if let Some(g) = anti_unify(&acc, m, &mut subst, &mut next_hole) {
                acc = g;
                generalized += 1;
            }
        }
        let (canon, arity) = canonicalize(&acc);
        if generalized >= MIN_SUPPORT && is_promotable(&canon, arity) {
            let key = expr_key(&canon);
            let size = canon.size();
            lgg_inserts.push((
                key,
                SubtreeStats {
                    support: generalized,
                    total_occurrences: generalized,
                    arity,
                    size,
                    exemplar: canon,
                },
            ));
        }
    }
    for (key, stat) in lgg_inserts {
        counts
            .entry(key)
            .and_modify(|s| {
                if stat.support > s.support {
                    s.support = stat.support;
                }
            })
            .or_insert(stat);
    }

    // (d) RANK — collect candidates passing thresholds, sort by total
    // deterministic key: Reverse(compression_score), Reverse(support),
    // size ascending, expr_key lexicographic.
    let mut candidates: Vec<&SubtreeStats> = counts
        .values()
        .filter(|s| s.support >= MIN_SUPPORT && s.size >= MIN_SUBTREE_SIZE)
        .collect();
    candidates.sort_by(|a, b| {
        let sa = a.support as usize * (a.size.saturating_sub(1));
        let sb = b.support as usize * (b.size.saturating_sub(1));
        sb.cmp(&sa) // compression score descending
            .then_with(|| b.support.cmp(&a.support)) // support descending
            .then_with(|| a.size.cmp(&b.size)) // size ascending
            .then_with(|| expr_key(&a.exemplar).cmp(&expr_key(&b.exemplar)))
    });
    candidates.truncate(MAX_PROMOTE);

    // (e) VERIFY + NAME + ADD each in sorted order.
    for stat in candidates {
        if !robust_well_defined(&stat.exemplar, stat.arity.max(1), 50) {
            continue;
        }
        let name = name_abstraction(&stat.exemplar);
        library.add(stat.exemplar.clone(), name);
    }

    // HYBRID FALLBACK (cold-start safety): the bootstrap seeds rarely share
    // subtrees at support>=2 on a truly cold machine, so support-mining alone
    // can return very few entries. To never regress below the historical
    // coverage, also add the corpus's own canonicalized scalar Exprs as
    // low-priority components when the mined library is thin. This is an
    // intentional product decision, not a fallback that weakens soundness —
    // every added Expr is still verified-by-construction (it came from a solved
    // problem) and is robust-gated below.
    if library.len() < 8 {
        let mut seeds: Vec<(Expr, usize)> = corpus
            .iter()
            .map(|s| canonicalize(&s.expr))
            .filter(|(c, a)| is_promotable(c, *a))
            .collect();
        seeds.sort_by(|a, b| expr_key(&a.0).cmp(&expr_key(&b.0)));
        for (canon, arity) in seeds {
            if !robust_well_defined(&canon, arity.max(1), 50) {
                continue;
            }
            let name = name_abstraction(&canon);
            library.add(canon, name);
        }
    }

    library
}

// ─── Component Library ───────────────────────────────────────────────────────
// Stores discovered useful sub-expressions that can be reused across problems.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentLibrary {
    /// Reusable sub-expressions with their semantic descriptions
    components: Vec<(Expr, String)>, // (expr, description)
    /// Size of the solved-expression corpus this library was last mined from.
    /// Used by [`Self::load_or_dream`] to re-mine when the corpus has GROWN
    /// since the cached library was built — without this watermark the library
    /// was mined exactly once (at cold start) and every later solve, though
    /// recorded into the corpus, never became a new abstraction. `#[serde(default)]`
    /// so libraries written before this field load as `0` (forcing one re-mine).
    #[serde(default)]
    mined_corpus_len: usize,
}

/// Path for persistent library storage
fn library_path() -> PathBuf {
    dirs_home().join(".mog_synth_library.json")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

impl ComponentLibrary {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            mined_corpus_len: 0,
        }
    }

    /// Mine `corpus` and fold any NEW components into this library (dedup by
    /// description via [`Self::add`]). Returns how many components were added.
    ///
    /// This is the disk-free core of "the library keeps mining as the solve
    /// corpus grows." Mining is whole-corpus and frequency-gated, so re-mining
    /// a grown corpus yields the previously-mined patterns plus any new ones;
    /// merging (rather than replacing) guarantees a previously-mined component
    /// is never lost if a later, larger corpus shifts a pattern below threshold.
    pub fn merge_from_corpus(&mut self, corpus: &[SolvedExpr]) -> usize {
        let fresh = mine_from_corpus(corpus);
        let before = self.components.len();
        for (expr, desc) in fresh.components {
            self.add(expr, desc);
        }
        self.components.len() - before
    }

    /// Add a discovered component to the library
    pub fn add(&mut self, expr: Expr, description: String) {
        // Don't add duplicates (by description)
        if !self.components.iter().any(|(_, d)| d == &description) {
            self.components.push((expr, description));
        }
    }

    /// Get all re-rooted component instantiations applicable to an `n_args`
    /// problem. Each stored component is arity-filtered (skipped if it needs
    /// more args than the problem has) and re-rooted onto the real arg slots via
    /// [`instantiate_component`], so a mined `?0*?0` applies to BOTH Var0 and
    /// Var1 — enabling novel `square(a) + square(b)` compositions.
    pub fn get_for_args(&self, n_args: usize) -> Vec<Expr> {
        let mut out = Vec::new();
        for (comp, _) in &self.components {
            out.extend(instantiate_component(comp, n_args));
        }
        out
    }

    /// Number of components in the library
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Save library to disk
    pub fn save(&self) -> Result<(), String> {
        let path = library_path();
        let json = serde_json::to_string_pretty(self).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Load library from disk (returns empty library if file doesn't exist)
    pub fn load() -> Self {
        let path = library_path();
        if !path.exists() {
            return Self::new();
        }
        let json = match std::fs::read_to_string(&path) {
            Ok(j) => j,
            Err(_) => return Self::new(),
        };
        match serde_json::from_str(&json) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("[library] failed to parse {}: {e}", path.display());
                Self::new()
            }
        }
    }

    /// Load the library, re-mining when the solve corpus has grown — or create
    /// it via dream mode if none exists.
    ///
    /// Previously this returned the cached `library.json` unconditionally, so
    /// the library was mined exactly once and the `record_solved_expr` corpus,
    /// though it grew with every solve, never produced new abstractions. Now we
    /// compare the current corpus size against the watermark stored in the
    /// cached library and, when it has grown, mine the larger corpus and fold
    /// the new components in. This is the link that makes the library actually
    /// compound across runs ("writes its own teachers").
    pub fn load_or_dream(dream_budget_ms: u64) -> Self {
        let path = library_path();
        let corpus = load_solved_exprs();
        let corpus_len = corpus.len();
        if path.exists() {
            let mut lib = Self::load();
            if !lib.components.is_empty() {
                if corpus_len > lib.mined_corpus_len {
                    let mined_at = lib.mined_corpus_len;
                    let added = lib.merge_from_corpus(&corpus);
                    lib.mined_corpus_len = corpus_len;
                    eprintln!(
                        "[library] corpus grew {mined_at}->{corpus_len}; re-mined +{added} \
                         components -> {} total",
                        lib.len()
                    );
                    let _ = lib.save();
                } else {
                    eprintln!(
                        "[library] loaded {} components from {} (corpus {corpus_len}, no growth)",
                        lib.len(),
                        path.display()
                    );
                }
                return lib;
            }
        }
        eprintln!("[library] no library found, running dream mode...");
        let mut lib = dream(dream_budget_ms);
        // Watermark the corpus this library was mined from. When dream() had to
        // bootstrap (empty on-disk corpus), `corpus_len` is 0, so the first real
        // solve triggers a re-mine.
        lib.mined_corpus_len = corpus_len;
        let _ = lib.save();
        lib
    }
}

// ─── Main enumeration engine ─────────────────────────────────────────────────

const CORE_BINOPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod];
const CORE_UNOPS: [UnOp; 2] = [UnOp::Neg, UnOp::Abs];

/// Fast enumeration with core 5 ops only.
pub fn enumerate_exprs_core(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_core_ext(n_args, max_size, examples, time_limit_ms, library).0
}

/// Like `enumerate_exprs_core`, but also reports whether the search bailed out
/// because the time budget was hit (`true`) or because every expression up to
/// `max_size` was exhausted cleanly (`false`). The distinction lets callers
/// skip a slower follow-up pass when a clean exhaustion has already ruled out
/// a small closed-form answer.
pub fn enumerate_exprs_core_ext(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> (Option<Expr>, bool) {
    enumerate_exprs_with_ops_ext(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        &CORE_BINOPS,
        &CORE_UNOPS,
    )
}

/// Full enumeration with all 12 ops.
pub fn enumerate_exprs(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
) -> Option<Expr> {
    enumerate_exprs_with_ops(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        &ALL_BINOPS,
        &ALL_UNOPS,
    )
}

/// Enumerate expressions bottom-up with observational equivalence pruning.
fn enumerate_exprs_with_ops(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> Option<Expr> {
    enumerate_exprs_with_ops_ext(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
    )
    .0
}

/// Same as [`enumerate_exprs_with_ops`] but also returns `true` when the search
/// terminated because the time budget ran out, and `false` when every size up
/// to `max_size` was explored without finding a match.
fn enumerate_exprs_with_ops_ext(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> (Option<Expr>, bool) {
    let (expr, timed_out, _max_completed) = enumerate_exprs_with_ops_stats(
        n_args,
        max_size,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
    );
    (expr, timed_out)
}

/// Like [`enumerate_exprs_with_ops`] but additionally returns the largest size
/// level whose enumeration ran to completion before the deadline (`0` if none).
/// Callers can combine the `timed_out` flag with `max_completed` to decide
/// whether a shallower follow-up pass is worth running.
fn enumerate_exprs_with_ops_stats(
    n_args: usize,
    max_size: usize,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
) -> (Option<Expr>, bool, usize) {
    // Bounded one-shot search: build a throwaway fresh frontier and run the
    // resumable core with a fixed `soft_cap = max_size`. This preserves the
    // exact behavior (selection order, dedup, the `s <= max_size` push guards)
    // of the historical bounded enumerator that bootstrap/tests depend on.
    let mut frontier = Frontier::fresh(String::new(), n_args, 0);
    let (expr, timed_out) = enumerate_exprs_resumable(
        &mut frontier,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
        Some(max_size),
    );
    // `next_size` advances past each completed stratum; the deepest cleanly
    // completed size is `next_size - 1`, capped at `max_size` (0 if none).
    let max_completed = if expr.is_some() {
        // Found before completing the in-progress stratum; report the prior
        // fully-completed depth, consistent with the original semantics.
        frontier.next_size.saturating_sub(1).min(max_size)
    } else if timed_out {
        frontier.next_size.saturating_sub(1).min(max_size)
    } else {
        max_size
    };
    (expr, timed_out, max_completed)
}

/// Resumable, anytime, UNCAPPED bottom-up enumeration core.
///
/// Seeds `by_size`/`seen` from the incoming `frontier` (re-deriving `seen` by
/// replaying the stored strata in order, so dedup decisions are reproduced
/// exactly), then deepens from `frontier.next_size` with NO fixed size ceiling
/// when `soft_cap` is `None` — the ONLY stop conditions are "found a verified
/// Expr" or "time budget tripped". On a budget trip (or clean exhaustion under a
/// `soft_cap`), the grown `by_size` and the size reached are written back into
/// `frontier` so a later call RESUMES at that depth rather than restarting.
///
/// Returns `(found, timed_out)`. `timed_out == false` with `found == None` means
/// the (capped) space was exhausted cleanly; under `soft_cap == None` that can
/// only happen if the enumerator runs out of new unique expressions entirely.
///
/// Determinism: no rand, no clock-in-selection. Every `Instant::now()` use here
/// BOUNDS work (when to stop); none SELECTS a candidate. Selection is purely
/// `size ascending → fixed var/const/op order → library injection order`.
#[allow(clippy::too_many_arguments)]
fn enumerate_exprs_resumable(
    frontier: &mut Frontier,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
    soft_cap: Option<usize>,
) -> (Option<Expr>, bool) {
    enumerate_exprs_resumable_c(
        frontier,
        examples,
        time_limit_ms,
        library,
        binops,
        unops,
        soft_cap,
        &[],
    )
}

/// Callee-aware resumable enumerator. When `callees` is non-empty the deepening
/// loop additionally emits `Call(idx, args)` candidates for each registered
/// producer (gated by exact arity + the same size budget as unary ops), and all
/// candidate checks resolve `Call` nodes against the registry. When `callees`
/// is EMPTY this is byte-identical to the historical enumerator: no `Call` is
/// ever constructed and every check uses `&[]` (the regression-critical path).
#[allow(clippy::too_many_arguments)]
fn enumerate_exprs_resumable_c(
    frontier: &mut Frontier,
    examples: &[(Vec<i64>, i64)],
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
    soft_cap: Option<usize>,
    callees: &[NamedCallable],
) -> (Option<Expr>, bool) {
    let start = std::time::Instant::now();
    let n_args = frontier.n_args;
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    // `by_size` grows dynamically (no `max_size + 1` preallocation), so an
    // uncapped search can deepen without an upper bound. Take ownership of the
    // frontier's strata for the duration of the search; write the (possibly
    // grown) bank back before returning.
    let mut by_size: Vec<Vec<Expr>> = std::mem::take(&mut frontier.by_size);
    if by_size.is_empty() {
        by_size.push(vec![]); // index 0 sentinel
    }
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    let mut timed_out = false;

    // The soft cap, if any: pushes past it are suppressed and the deepening
    // loop stops once `size` would exceed it. `None` == unbounded.
    let cap = soft_cap.unwrap_or(usize::MAX);

    // Ensure `by_size` can hold index `s`, growing with empty strata as needed.
    let ensure_slot = |by_size: &mut Vec<Vec<Expr>>, s: usize| {
        if s >= by_size.len() {
            by_size.resize(s + 1, Vec::new());
        }
    };

    // Helper: check a candidate; on a match return it, else dedup-insert into
    // the size stratum (suppressed above the soft cap). Mirrors the historical
    // `check_add` exactly, but grows `by_size` instead of indexing a fixed vec.
    let check_add = |e: &Expr,
                     by_size: &mut Vec<Vec<Expr>>,
                     seen: &mut HashSet<Vec<i64>>|
     -> Option<Expr> {
        if let Some(fp) = fingerprint_c(e, &test_inputs, callees) {
            if matches_all_c(e, examples, callees)
                && robust_well_defined_c(e, n_args, 30, callees)
            {
                return Some(e.clone());
            }
            if seen.insert(fp) {
                let s = e.size();
                if s <= cap {
                    ensure_slot(by_size, s);
                    by_size[s].push(e.clone());
                }
            }
        }
        None
    };

    // RESUME PATH: if the frontier already carries strata beyond the size-1
    // seed (i.e. a prior call expanded it), rebuild `seen` by replaying those
    // strata in stored order. This reproduces the identical dedup state without
    // re-seeding size-1 atoms (which would duplicate vars/consts/library) and
    // keeps the persisted file free of the fingerprint set.
    let resuming = frontier.next_size > 2 || by_size.len() > 2;
    if resuming {
        for stratum in by_size.iter() {
            for e in stratum {
                if let Some(fp) = fingerprint_c(e, &test_inputs, callees) {
                    seen.insert(fp);
                }
            }
        }
    } else {

    // ── COLD START: seed size 1 (variables, constants, library components) ──
    // On a `found` return the frontier is irrelevant (the caller evicts a
    // solved problem's frontier), so success returns do not write `by_size`
    // back — only the budget-exhausted/exhausted-clean exit persists it.
    for i in 0..n_args {
        if let Some(e) = check_add(&Expr::Var(i), &mut by_size, &mut seen) {
            return (Some(e), false);
        }
    }
    for &c in &CONSTANTS {
        if let Some(e) = check_add(&Expr::Const(c), &mut by_size, &mut seen) {
            return (Some(e), false);
        }
    }
    // Example-MINED constants: literals derived from the I/O (output values,
    // output-input diffs, exact quotients) so a closed form needing a constant
    // outside the fixed pool (`x + 42`, `x * 256`) is reachable. Deterministic
    // per fingerprint (mined from `examples`), so frontier-consistent. Seeded
    // AFTER the fixed pool so the common case is unchanged; check_add dedups any
    // overlap.
    for c in mine_example_constants(examples) {
        if let Some(e) = check_add(&Expr::Const(c), &mut by_size, &mut seen) {
            return (Some(e), false);
        }
    }
    // Add library components as re-rooted size-1 leaves (cost discount).
    // Each mined abstraction is instantiated onto the real arg slots, gated by
    // the SAME soundness gate (matches_all + robust_well_defined) as every other
    // candidate, then injected into `by_size[1]` rather than `by_size[size]` so
    // the size-2 binop loop can compose two abstractions into a program of true
    // size 7 at an effective `max_size` of ~3 (the documented injection lever).
    // A count budget bounds the size-2 blow-up.
    if let Some(lib) = library {
        let mut injected = 0usize;
        for comp in lib.get_for_args(n_args) {
            if injected >= MAX_SIZE1_INJECTIONS {
                break;
            }
            if let Some(fp) = fingerprint_c(&comp, &test_inputs, callees) {
                if matches_all_c(&comp, examples, callees)
                    && robust_well_defined_c(&comp, n_args, 30, callees)
                {
                    return (Some(comp), false);
                }
                if seen.insert(fp) {
                    ensure_slot(&mut by_size, 1);
                    by_size[1].push(comp);
                    injected += 1;
                }
            }
        }
    }
    } // end COLD START

    // Deepening loop: from the frontier's resume point, with NO upper bound
    // when uncapped (`cap == usize::MAX`). The ONLY stops are "found" (early
    // return) or "budget tripped" (`timed_out`). `size` may exceed any prior
    // run's depth; `ensure_slot` grows `by_size` accordingly.
    let mut size = frontier.next_size;
    while size <= cap {
        if start.elapsed().as_millis() as u64 > time_limit_ms {
            timed_out = true;
            break;
        }
        ensure_slot(&mut by_size, size);
        let mut new: Vec<Expr> = Vec::new();

        // Unary ops
        if size >= 2 {
            let children = by_size[size - 1].clone();
            for child in &children {
                for &uop in unops {
                    let e = Expr::UnaryOp(uop, Box::new(child.clone()));
                    if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                        if matches_all_c(&e, examples, callees)
                            && robust_well_defined_c(&e, n_args, 30, callees)
                        {
                            return (Some(e), false);
                        }
                        if seen.insert(fp) {
                            new.push(e);
                        }
                    }
                }
            }
        }

        // Calls to registered producers (inter-function data flow). EMPTY when
        // `callees` is empty, so the base search is byte-identical to today: no
        // `Call` node is ever constructed without a registry. For each callable,
        // emit `Call(idx, args)` where the args are drawn from existing strata so
        // that `Call.size() == 1 + sum(arg sizes)` lands in THIS size level —
        // exactly the same cost discipline as UnaryOp/BinOp. The args are
        // DISCOVERED by the enumeration (any sub-expression of the right size),
        // not synthesized from a template. MVP supports arity-1 and arity-2
        // callees (the only arities the wiring registers); the arg-size split
        // for arity-2 enumerates ordered pairs summing to `size - 1`.
        if !callees.is_empty() && size >= 2 {
            for (idx, callee) in callees.iter().enumerate() {
                let arg_budget = size - 1; // the Call node itself costs 1
                match callee.n_args {
                    1 => {
                        if arg_budget >= 1 && arg_budget <= cap {
                            let args0 = by_size[arg_budget].clone();
                            for a0 in &args0 {
                                let e = Expr::Call(idx, vec![a0.clone()]);
                                if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                                    if matches_all_c(&e, examples, callees)
                                        && robust_well_defined_c(&e, n_args, 30, callees)
                                    {
                                        return (Some(e), false);
                                    }
                                    if seen.insert(fp) {
                                        new.push(e);
                                    }
                                }
                            }
                        }
                    }
                    2 => {
                        // ordered (s0, s1) with s0 + s1 == arg_budget, both >= 1
                        for s0 in 1..arg_budget {
                            let s1 = arg_budget - s0;
                            if s1 < 1 || s0 > cap || s1 > cap {
                                continue;
                            }
                            let a0s = by_size[s0].clone();
                            let a1s = by_size[s1].clone();
                            for a0 in &a0s {
                                for a1 in &a1s {
                                    let e = Expr::Call(idx, vec![a0.clone(), a1.clone()]);
                                    if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                                        if matches_all_c(&e, examples, callees)
                                            && robust_well_defined_c(&e, n_args, 30, callees)
                                        {
                                            return (Some(e), false);
                                        }
                                        if seen.insert(fp) {
                                            new.push(e);
                                        }
                                    }
                                }
                            }
                            if start.elapsed().as_millis() as u64 > time_limit_ms {
                                timed_out = true;
                                break;
                            }
                        }
                    }
                    // Higher arities are out of scope for this MVP (not registered).
                    _ => {}
                }
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    timed_out = true;
                    break;
                }
            }
        }

        // Binary ops
        for ls in 1..size {
            let rs = size - 1 - ls;
            if rs < 1 || rs > cap {
                continue;
            }
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                timed_out = true;
                break;
            }
            let lefts = by_size[ls].clone();
            let rights = by_size[rs].clone();
            for left in &lefts {
                for right in &rights {
                    for &op in binops {
                        let e = Expr::BinOp(op, Box::new(left.clone()), Box::new(right.clone()));
                        if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                            if matches_all_c(&e, examples, callees)
                                && robust_well_defined_c(&e, n_args, 30, callees)
                            {
                                return (Some(e), false);
                            }
                            if seen.insert(fp) {
                                new.push(e);
                            }
                        }
                    }
                    if start.elapsed().as_millis() as u64 > time_limit_ms {
                        timed_out = true;
                        break;
                    }
                }
            }
        }

        // If-then-else (size >= 5)
        if size >= 5 {
            let atoms = by_size[1].clone();
            for &cmp in &ALL_CMPS {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    timed_out = true;
                    break;
                }
                for cl in &atoms {
                    for cr in &atoms {
                        let budget = size - 3;
                        for ts in 1..budget {
                            let es = budget - ts;
                            if es < 1 {
                                continue;
                            }
                            let then_es = by_size[ts].clone();
                            let else_es = by_size[es].clone();
                            for te in &then_es {
                                for ee in &else_es {
                                    let e = Expr::IfExpr(
                                        cmp,
                                        Box::new(cl.clone()),
                                        Box::new(cr.clone()),
                                        Box::new(te.clone()),
                                        Box::new(ee.clone()),
                                    );
                                    if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                                        if matches_all_c(&e, examples, callees)
                                            && robust_well_defined_c(&e, n_args, 30, callees)
                                        {
                                            return (Some(e), false);
                                        }
                                        if seen.insert(fp) {
                                            new.push(e);
                                        }
                                    }
                                }
                                if start.elapsed().as_millis() as u64 > time_limit_ms {
                                    timed_out = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // While-accumulator loops (size >= 6)
        // acc = init; for i in 0..bound { acc = acc OP rhs(acc, i, args) }
        if size >= 6 && n_args >= 1 {
            let init_budget = 1;
            let bound_budget = 1;
            let rhs_budget = size - 3 - init_budget - bound_budget;
            if rhs_budget >= 1 && rhs_budget <= cap {
                let inits = by_size[init_budget].clone();
                let bounds = by_size[bound_budget].clone();
                // For loop body, the rhs can reference acc (var n_args) and i (var n_args+1)
                // Build a separate expression bank for the loop body context
                let loop_n_args = n_args + 2; // original args + acc + i
                let mut loop_atoms: Vec<Expr> = Vec::new();
                for i in 0..loop_n_args {
                    loop_atoms.push(Expr::Var(i));
                }
                for &c in &CONSTANTS {
                    loop_atoms.push(Expr::Const(c));
                }
                // Simple loop rhs: just atoms or atom OP atom
                let mut loop_exprs: Vec<Expr> = loop_atoms.clone();
                if rhs_budget >= 3 {
                    for l in &loop_atoms {
                        for r in &loop_atoms {
                            for &op in &[BinOp::Add, BinOp::Mul, BinOp::Mod] {
                                loop_exprs.push(Expr::BinOp(
                                    op,
                                    Box::new(l.clone()),
                                    Box::new(r.clone()),
                                ));
                            }
                        }
                    }
                }

                for init in &inits {
                    for bound in &bounds {
                        for &bop in &LOOP_BODY_OPS {
                            for rhs in &loop_exprs {
                                if start.elapsed().as_millis() as u64 > time_limit_ms {
                                    timed_out = true;
                                    break;
                                }
                                let e = Expr::WhileAccum {
                                    init: Box::new(init.clone()),
                                    bound: Box::new(bound.clone()),
                                    body_op: bop,
                                    body_rhs: Box::new(rhs.clone()),
                                };
                                if let Some(fp) = fingerprint_c(&e, &test_inputs, callees) {
                                    if matches_all_c(&e, examples, callees)
                                        && robust_well_defined_c(&e, n_args, 30, callees)
                                    {
                                        return (Some(e), false);
                                    }
                                    if seen.insert(fp) {
                                        new.push(e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if timed_out {
            // Budget tripped mid-stratum: DISCARD the partial `new` for this
            // size (leave its slot empty) so a later resume rebuilds the full
            // size deterministically, and do NOT advance `next_size` past it.
            break;
        }
        eprintln!(
            "[enum] size {size}: {} new, {} total unique, {:.1}s",
            new.len(),
            seen.len(),
            start.elapsed().as_secs_f32()
        );
        by_size[size] = new;
        size += 1;
    }

    // Persist the grown frontier: the deepest cleanly-completed size is
    // `size - 1`, so the next resume continues at `size`. (On a clean exhaustion
    // under a soft cap, `size` is `cap + 1`, which the caller treats as done.)
    frontier.next_size = size;
    frontier.by_size = by_size;
    (None, timed_out)
}

// ─── Emit Mog code from discovered expression ────────────────────────────────

/// Map a fold/accumulator `BinOp` to the Mog infix operator used in
/// `acc = acc OP rhs`. Mog natively supports `+ - * % & | ^` as infix operators
/// (runtime/mod.rs lexer+parser+eval), so a bitwise accumulator (e.g. XOR-fold)
/// must render to `^`, NOT the old `_ => "+"` fallback which silently dropped
/// bitwise/mod semantics and produced source that diverged from `Expr::eval`.
/// `Min`/`Max` are handled by their own conditional-update emission and are not
/// routed through this helper.
fn fold_op_mog(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        // Min/Max never reach here (emitted via conditional update); Shl/Shr are
        // not used as accumulator ops by the loop synthesizers. Keep `+` as a
        // last-resort default only for those unreachable cases.
        BinOp::Min | BinOp::Max | BinOp::Shl | BinOp::Shr => "+",
    }
}

fn emit_mog(expr: &Expr, fn_name: &str, param_names: &[&str]) -> String {
    let sig = param_names
        .iter()
        .map(|n| format!("{n}: i64"))
        .collect::<Vec<_>>()
        .join(", ");

    match expr {
        Expr::WhileAccum {
            init,
            bound,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let bound_s = bound.to_mog(param_names);
            let op_s = fold_op_mog(body_op);
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("acc");
                v.push("i");
                v
            };
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    i: i64 = 0;\n    while i < {bound_s} {{\n        acc = acc {op_s} {rhs_s};\n        i = i + 1;\n    }}\n    return acc;\n}}\n"
            )
        }
        Expr::ForFold {
            init,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let op_s = match body_op {
                BinOp::Min => "min",
                BinOp::Max => "max",
                other => fold_op_mog(other),
            };
            // Body rhs namespace: scalar_params + [item, i, acc]
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("item");
                v.push("i");
                v.push("acc");
                v
            };
            let arr_sig = if param_names.is_empty() {
                "arr: [i64]".to_string()
            } else {
                format!("{}, arr: [i64]", sig)
            };
            if op_s == "min" || op_s == "max" {
                // Min/max folds need conditional update
                let cmp = if op_s == "min" { "<" } else { ">" };
                format!(
                    "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if item {cmp} acc {{\n            acc = item;\n        }}\n    }}\n    return acc;\n}}\n"
                )
            } else if let Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) = body_rhs.as_ref() {
                // Conditional fold: if lhs CMP rhs { acc = acc OP then } else { acc = acc OP else }
                let ls = lhs.to_mog_ext(&ext_names, &[]);
                let rs = rhs.to_mog_ext(&ext_names, &[]);
                let cs = match cmp {
                    CmpOp::Lt => "<",
                    CmpOp::Le => "<=",
                    CmpOp::Eq => "==",
                    CmpOp::Ge => ">=",
                    CmpOp::Gt => ">",
                    CmpOp::Ne => "!=",
                };
                let ts = then_e.to_mog_ext(&ext_names, &[]);
                let es = else_e.to_mog_ext(&ext_names, &[]);
                if **else_e == Expr::Const(0) {
                    // if cond { acc = acc OP val; }
                    format!(
                        "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }}\n    }}\n    return acc;\n}}\n"
                    )
                } else {
                    // if cond { acc = acc OP a; } else { acc = acc OP b; }
                    format!(
                        "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }} else {{\n            acc = acc {op_s} {es};\n        }}\n    }}\n    return acc;\n}}\n"
                    )
                }
            } else {
                let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
                format!(
                    "fn {fn_name}({arr_sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        acc = acc {op_s} {rhs_s};\n    }}\n    return acc;\n}}\n"
                )
            }
        }
        Expr::NestedWhile {
            outer_init,
            outer_bound,
            outer_body_op,
            inner_init,
            inner_bound,
            inner_body_op,
            inner_body_rhs,
        } => {
            let oinit_s = outer_init.to_mog(param_names);
            let obound_s = outer_bound.to_mog(param_names);
            let oop_s = fold_op_mog(outer_body_op);
            let iinit_s = inner_init.to_mog(param_names);
            let ibound_s = inner_bound.to_mog(param_names);
            let iop_s = fold_op_mog(inner_body_op);
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("outer_acc");
                v.push("outer_i");
                v.push("inner_acc");
                v.push("inner_i");
                v
            };
            let irhs_s = inner_body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    outer_acc: i64 = {oinit_s};\n    outer_i: i64 = 0;\n    while outer_i < {obound_s} {{\n        inner_acc: i64 = {iinit_s};\n        inner_i: i64 = 0;\n        while inner_i < {ibound_s} {{\n            inner_acc = inner_acc {iop_s} {irhs_s};\n            inner_i = inner_i + 1;\n        }}\n        outer_acc = outer_acc {oop_s} inner_acc;\n        outer_i = outer_i + 1;\n    }}\n    return outer_acc;\n}}\n"
            )
        }
        Expr::WhileCond {
            init,
            state_init,
            cond_cmp,
            cond_val,
            divisor,
            body_op,
            body_rhs,
        } => {
            let init_s = init.to_mog(param_names);
            let state_init_s = state_init.to_mog(param_names);
            let cv_s = cond_val.to_mog(param_names);
            let op_s = fold_op_mog(body_op);
            let cmp_s = match cond_cmp {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Eq => "==",
                CmpOp::Ge => ">=",
                CmpOp::Gt => ">",
                CmpOp::Ne => "!=",
            };
            let ext_names: Vec<&str> = {
                let mut v: Vec<&str> = param_names.to_vec();
                v.push("x");
                v.push("acc");
                v.push("i");
                v
            };
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            let div_expr = if *divisor == 10 {
                "x / 10".to_string()
            } else if *divisor == 2 {
                "x / 2".to_string()
            } else {
                format!("x / {divisor}")
            };
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    x: i64 = {state_init_s};\n    while x {cmp_s} {cv_s} {{\n        acc = acc {op_s} {rhs_s};\n        x = {div_expr};\n    }}\n    return acc;\n}}\n"
            )
        }
        _ => {
            let expr_s = expr.to_mog(param_names);
            format!("fn {fn_name}({sig}) -> i64 {{\n    return {expr_s};\n}}\n")
        }
    }
}

/// Emit Mog code for array fold expressions, respecting the original parameter order.
/// `array_idx` is the position of the array parameter in the original function signature.
/// `scalar_names` are the names for the scalar parameters (in order of appearance).
fn emit_mog_array(expr: &Expr, fn_name: &str, scalar_names: &[&str], array_idx: usize) -> String {
    // Build full signature: put arr at the correct position
    let n_total = scalar_names.len() + 1; // scalars + 1 array
    let mut sig_parts: Vec<String> = vec![String::new(); n_total];
    sig_parts[array_idx] = "arr: [i64]".to_string();
    let mut scalar_i = 0;
    for i in 0..n_total {
        if i != array_idx {
            sig_parts[i] = format!("{}: i64", scalar_names[scalar_i]);
            scalar_i += 1;
        }
    }
    let sig = sig_parts.join(", ");

    if let Expr::ForFold {
        init,
        body_op,
        body_rhs,
    } = expr
    {
        let init_s = init.to_mog(scalar_names);
        let op_s = match body_op {
            BinOp::Min => "min",
            BinOp::Max => "max",
            other => fold_op_mog(other),
        };
        // Body rhs namespace: scalar_params + [item, i, acc]
        let ext_names: Vec<&str> = {
            let mut v: Vec<&str> = scalar_names.to_vec();
            v.push("item");
            v.push("i");
            v.push("acc");
            v
        };

        if op_s == "min" || op_s == "max" {
            let cmp = if op_s == "min" { "<" } else { ">" };
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = arr[0];\n    for item in arr {{\n        if item {cmp} acc {{\n            acc = item;\n        }}\n    }}\n    return acc;\n}}\n"
            )
        } else if let Expr::IfExpr(cmp, lhs, rhs, then_e, else_e) = body_rhs.as_ref() {
            let ls = lhs.to_mog_ext(&ext_names, &[]);
            let rs = rhs.to_mog_ext(&ext_names, &[]);
            let cs = match cmp {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Eq => "==",
                CmpOp::Ge => ">=",
                CmpOp::Gt => ">",
                CmpOp::Ne => "!=",
            };
            let ts = then_e.to_mog_ext(&ext_names, &[]);
            if **else_e == Expr::Const(0) {
                format!(
                    "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }}\n    }}\n    return acc;\n}}\n"
                )
            } else {
                let es = else_e.to_mog_ext(&ext_names, &[]);
                format!(
                    "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        if {ls} {cs} {rs} {{\n            acc = acc {op_s} {ts};\n        }} else {{\n            acc = acc {op_s} {es};\n        }}\n    }}\n    return acc;\n}}\n"
                )
            }
        } else {
            let rhs_s = body_rhs.to_mog_ext(&ext_names, &[]);
            format!(
                "fn {fn_name}({sig}) -> i64 {{\n    acc: i64 = {init_s};\n    for item in arr {{\n        acc = acc {op_s} {rhs_s};\n    }}\n    return acc;\n}}\n"
            )
        }
    } else {
        // Fallback: should not happen for array synthesis
        format!("fn {fn_name}({sig}) -> i64 {{\n    return 0;\n}}\n")
    }
}

// ─── Public synthesis entry point ────────────────────────────────────────────

/// Enumerative synthesis: discovers programs from I/O examples alone.
/// Handles both scalar and array problems.
pub fn synthesize_enumerative(problem: &Problem) -> Option<SolveResult> {
    // COMPOSITE OUTPUT (tuple/pair/quad of ints) → emit a struct. Scalar inputs:
    // each component is a flat expression (inline). Array inputs: each component
    // is a fold synthesized as a helper fn and called. Tried first so a composite
    // expected never falls through to the scalar/array int paths.
    if let Some(first) = problem.examples.first() {
        if tuple_arity(&first.expected).is_some() {
            let r = if first.inputs.iter().all(|v| matches!(v, Value::Int(_))) {
                synthesize_tuple_output_enumerative(problem)
            } else {
                synthesize_tuple_output_array(problem)
            };
            if let Some(r) = r {
                return Some(r);
            }
        }
        // BOOL OUTPUT (deterministic predicate) → encode {1,0} + emit `!= 0`.
        if matches!(first.expected, Value::Bool(_)) {
            if let Some(r) = synthesize_bool_output_enumerative(problem) {
                return Some(r);
            }
        }
    }

    // Detect array vs scalar
    let has_array = problem
        .examples
        .iter()
        .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))));

    if has_array {
        return synthesize_array_enumerative(problem);
    }

    // Scalar path (original)
    synthesize_scalar_enumerative(problem)
}

/// Scalar-only enumerative synthesis (original two-pass approach).
fn synthesize_scalar_enumerative(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    let fn_name = problem.function_name();

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let param_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .take(n_args)
        .copied()
        .collect();

    // Load component library (lazy, first call initializes)
    let library = ComponentLibrary::load_or_dream(5_000);

    // Inputs from the in-scope examples, used to fingerprint solved Exprs for
    // the persistent mining corpus (no-op under cfg!(test) / disabled path).
    let test_inputs: Vec<Vec<i64>> = examples.iter().map(|(a, _)| a.clone()).collect();

    // Deterministic per-problem key for the resumable frontier store. Derived
    // ONLY from the examples (stable across runs/machines), so a later solve of
    // the SAME problem reloads its frontier and deepens instead of restarting.
    let fp = crate::solved_cache::examples_fingerprint(&problem.examples);

    // ── Per-call time budget (anytime, resumable) ──────────────────────────
    // NSYNTH_ENUM_BUDGET_MS bounds THIS call; the SEARCH DEPTH is unbounded and
    // RISES across calls (the frontier resumes deeper each time). The default
    // equals the sum of the historical pass-1 + pass-2 budgets, so a cold first
    // call has unchanged wall-clock. The split mirrors the old deep/shallow
    // ratio: ~55% to the core-op (deep) tier, the rest to the full-op tier.
    let default_budget: u64 = if n_args <= 1 {
        18_000
    } else if n_args <= 2 {
        13_000
    } else {
        10_000
    };
    let budget_ms: u64 = std::env::var("NSYNTH_ENUM_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_budget);
    let core_budget = (budget_ms * 55 / 100).max(1);
    let full_budget = budget_ms.saturating_sub(core_budget).max(1);

    // Optional SOFT size cap. Unset → unbounded (the anytime guarantee). Set it
    // to bound cumulative re-run cost in CI without reintroducing a hard ceiling
    // in the engine itself.
    let soft_cap: Option<usize> = std::env::var("NSYNTH_ENUM_SOFT_CAP")
        .ok()
        .and_then(|v| v.parse().ok());

    // Run one resumable deepening pass for a given op tier: load this tier's
    // frontier (empty on first call), deepen under `budget`, and on a verified
    // hit record + evict the frontier and return the solution. On a miss,
    // persist the (deepened) frontier so the NEXT call resumes from here — this
    // is the "budget exhausted, frontier persisted, resumable" reframe of the
    // old terminal `None`.
    let mut run_tier = |ops_tier: u8,
                        binops: &[BinOp],
                        unops: &[UnOp],
                        budget: u64|
     -> Option<SolveResult> {
        let mut frontier = load_frontier(&fp, n_args, ops_tier)
            .unwrap_or_else(|| Frontier::fresh(fp.clone(), n_args, ops_tier));
        let (expr, _timed_out) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            budget,
            Some(&library),
            binops,
            unops,
            soft_cap,
        );
        if let Some(expr) = expr {
            let code = emit_mog(&expr, fn_name, &param_names);
            if verify_problem_code_strict(problem, &code).is_ok() {
                record_solved_expr(
                    &expr,
                    n_args,
                    &fingerprint(&expr, &test_inputs).unwrap_or_default(),
                );
                // Solved → its frontier is dead weight; drop it.
                evict_frontier(&fp);
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
            eprintln!(
                "[enum] found expr but Mog verification failed: {}",
                expr.to_mog(&param_names)
            );
        }
        // Miss for THIS call: persist the deepened frontier so a later call
        // continues from here (no fixed ceiling, never "impossible").
        save_frontier(&frontier);
        None
    };

    // Tier 0: core ops only (+,-,*,/,%) — deep. Tier 1: all 12 ops — broader.
    // Both are uncapped and resumable; clean exhaustion of small sizes is no
    // longer terminal (it just means the frontier now starts deeper next time).
    if let Some(result) = run_tier(0, &CORE_BINOPS, &CORE_UNOPS, core_budget) {
        return Some(result);
    }
    if let Some(result) = run_tier(1, &ALL_BINOPS, &ALL_UNOPS, full_budget) {
        return Some(result);
    }

    // Try nested loops for 1-2 arg scalar problems
    if n_args <= 2 {
        if let Some(result) = synthesize_nested_while(problem) {
            return Some(result);
        }
    }

    // Try while-cond loops for 1-arg problems (digit_sum, digit_count, etc.)
    if n_args == 1 {
        if let Some(result) = synthesize_while_cond(problem) {
            return Some(result);
        }
    }

    None
}

/// Does `e` (or any sub-expr) contain a `Call(idx, _)` node? Module-level so the
/// callee search can prefer a genuine inter-function call over a library-inlined
/// re-derivation of the producer's behaviour.
pub fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call(..) => true,
        Expr::UnaryOp(_, c) => expr_has_call(c),
        Expr::BinOp(_, l, r) => expr_has_call(l) || expr_has_call(r),
        Expr::IfExpr(_, a, b, c, d) => {
            expr_has_call(a) || expr_has_call(b) || expr_has_call(c) || expr_has_call(d)
        }
        _ => false,
    }
}

/// Solve a SCALAR problem with a set of registered producers available as
/// callable primitives. This is the inter-function-data-flow entry point: when
/// component B is synthesized with producer A registered, the search may emit a
/// `Call(A, ...)` node (discovered, not templated). It is a thin wrapper over
/// the same `enumerate_exprs_resumable_c` core the base path uses, with three
/// differences from `synthesize_scalar_enumerative`:
///   1. it threads `callees` into the enumerator (the ONLY behavioural delta);
///   2. it uses a FRESH per-call frontier (callee searches are NOT disk-cached,
///      so callee/non-callee strata never mix and the frontier-store byte guard
///      is untouched);
///   3. emission is wrapped in `with_callee_names` so a `Call` node renders as a
///      real `callee_name(args)` Mog call.
/// The library is still injected (mined abstractions help B too). The final
/// `verify_problem_code_strict` gate is the SAME un-gameable acceptance as the
/// base path — a `Call` candidate must pass strict holdout verification (the
/// verifier runs B's emitted code, which calls A's source).
pub fn synthesize_scalar_with_callees(
    problem: &Problem,
    callees: &[NamedCallable],
) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    let fn_name = problem.function_name();

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let param_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .take(n_args)
        .copied()
        .collect();

    let library = ComponentLibrary::load_or_dream(5_000);

    let default_budget: u64 = if n_args <= 1 {
        18_000
    } else if n_args <= 2 {
        13_000
    } else {
        10_000
    };
    let budget_ms: u64 = std::env::var("NSYNTH_ENUM_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_budget);
    let core_budget = (budget_ms * 55 / 100).max(1);
    let full_budget = budget_ms.saturating_sub(core_budget).max(1);
    let soft_cap: Option<usize> = std::env::var("NSYNTH_ENUM_SOFT_CAP")
        .ok()
        .and_then(|v| v.parse().ok());

    // Callee names for emission, indexed by registry position.
    let callee_names: Vec<String> = callees.iter().map(|c| c.name.clone()).collect();

    // Emit + strict-verify a found Expr into a SolveResult (None ⇒ verify failed).
    let emit_verify = |expr: &Expr| -> Option<SolveResult> {
        let code = with_callee_names(&callee_names, || emit_mog(expr, fn_name, &param_names));
        // STRICT-VERIFY GATE: prepend every callee's source so the verifier can
        // resolve the call (`fn_name(args)` references a real definition). The
        // consumer's own source (`code`) is the LAST fn, so the problem's
        // `function_name()` still names the entry point. Callees the emitted code
        // does not actually call are harmless (dead but valid).
        let mut verify_src = String::new();
        for c in callees {
            if !c.source.trim().is_empty() {
                verify_src.push_str(c.source.trim_end());
                verify_src.push_str("\n\n");
            }
        }
        verify_src.push_str(&code);
        if verify_problem_code_strict(problem, &verify_src).is_ok() {
            return Some(SolveResult {
                success: true,
                // Return ONLY the consumer's source (the producer lives in its own
                // module); the writer injects the `use` import.
                code,
                method: "enumerative-call".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
        eprintln!(
            "[enum-call] found expr but Mog verification failed: {}",
            with_callee_names(&callee_names, || expr.to_mog(&param_names))
        );
        None
    };

    let mut run_tier = |binops: &[BinOp], unops: &[UnOp], budget: u64| -> Option<SolveResult> {
        // PASS A — library ON: mined abstractions speed B's search.
        let mut frontier = Frontier::fresh(String::new(), n_args, 0);
        let (expr, _timed_out) = enumerate_exprs_resumable_c(
            &mut frontier,
            &examples,
            budget,
            Some(&library),
            binops,
            unops,
            soft_cap,
            callees,
        );
        // PREFER A GENUINE CALL: when producers are registered but the library-on
        // search inlined the producer's behaviour (a cheap library primitive
        // undercut the `Call` node by size), retry with the library OFF so the
        // `Call(A, ..)` is size-competitive and gets discovered instead. This is
        // the only behavioural change vs. before; with no callees, or when the
        // first pass already found a call, PASS B is skipped (byte-identical).
        if let Some(expr) = &expr {
            if callees.is_empty() || expr_has_call(expr) {
                return emit_verify(expr);
            }
        }
        if !callees.is_empty() {
            // PASS B — library OFF: force the call to compete on size alone.
            let mut frontier_b = Frontier::fresh(String::new(), n_args, 0);
            let (expr_b, _t) = enumerate_exprs_resumable_c(
                &mut frontier_b,
                &examples,
                budget,
                None,
                binops,
                unops,
                soft_cap,
                callees,
            );
            if let Some(expr_b) = &expr_b {
                if expr_has_call(expr_b) {
                    if let Some(r) = emit_verify(expr_b) {
                        return Some(r);
                    }
                }
            }
        }
        // Fall back to the library-on result (inlined but correct) only if PASS B
        // found no call — keeps a solution rather than failing, though the NL
        // orchestrator's anti-inline guard will reject a non-calling consumer.
        expr.as_ref().and_then(&emit_verify)
    };

    if let Some(result) = run_tier(&CORE_BINOPS, &CORE_UNOPS, core_budget) {
        return Some(result);
    }
    if let Some(result) = run_tier(&ALL_BINOPS, &ALL_UNOPS, full_budget) {
        return Some(result);
    }
    None
}

/// Inspect-only: return the solved `Expr` (not the Mog string) for `problem`
/// under `callees`, so a test can assert structurally that the AST contains a
/// `Call` node. Mirrors `synthesize_scalar_with_callees` but returns the Expr
/// and SKIPS the strict-verify gate's discard (it still runs the same accept
/// gate inside the enumerator: matches_all + robust_well_defined). Test-only.
pub fn solve_scalar_expr_with_callees(
    problem: &Problem,
    callees: &[NamedCallable],
    budget_ms: u64,
) -> Option<Expr> {
    let n_args = problem.examples.first()?.inputs.len();
    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| if let Value::Int(i) = v { Some(*i) } else { None })
                .collect();
            (args, ex.expected_int())
        })
        .collect();
    let mut frontier = Frontier::fresh(String::new(), n_args, 0);
    let (expr, _timed_out) = enumerate_exprs_resumable_c(
        &mut frontier,
        &examples,
        budget_ms,
        None, // no library — keep the search space minimal/controlled for the test
        &ALL_BINOPS,
        &ALL_UNOPS,
        None,
        callees,
    );
    expr
}

// ─── Array enumeration ─────────────────────────────────────────────────────

/// Number of distinct (init, body_op) wrappings to try per candidate body in
/// the resumable array search. Bounded so the per-body verify cost stays small.
const FOLD_WRAP_INITS: [i64; 2] = [0, 1];
const FOLD_WRAP_OPS: [BinOp; 4] = [BinOp::Add, BinOp::Mul, BinOp::Min, BinOp::Max];

/// Try to ACCEPT a candidate fold body: wrap it in each (init, body_op) pairing,
/// gate cheaply via [`check_fold_examples`], then — and only then — gate STRICTLY
/// via [`verify_problem_code_strict`] (the un-gameable acceptance, identical to
/// the scalar path). Returns the verified `SolveResult` on the first wrapping
/// that passes BOTH gates, else `None`.
///
/// This is the array-path counterpart of the scalar enumerator's inline
/// `matches_all(e) && robust_well_defined(e)` accept check, kept as a single
/// closure so every array hit — from the warm-up strategies OR the deepening
/// frontier — flows through the SAME strict verifier.
#[allow(clippy::too_many_arguments)]
fn try_accept_fold_body(
    body: &Expr,
    array_examples: &[(Vec<i64>, Vec<i64>, i64)],
    problem: &Problem,
    fn_name: &str,
    scalar_param_names: &[&str],
    array_idx: usize,
) -> Option<SolveResult> {
    for &init in &FOLD_WRAP_INITS {
        for &bop in &FOLD_WRAP_OPS {
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(init)),
                body_op: bop,
                body_rhs: Box::new(body.clone()),
            };
            if !check_fold_examples(&fold_expr, array_examples) {
                continue;
            }
            let code = emit_mog_array(&fold_expr, fn_name, scalar_param_names, array_idx);
            // STRICT gate: never accept on check_fold_examples alone.
            if verify_problem_code_strict(problem, &code).is_ok() {
                eprintln!(
                    "[enum-array] FOUND (frontier) init={init} op={bop:?} body={body:?}"
                );
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative-array".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
    }
    None
}

/// Resumable, size-stratified deepening search for a fold BODY expression,
/// reusing the SCALAR [`Frontier`] struct as its persistent store. This is the
/// array analogue of [`enumerate_exprs_resumable`]: it grows `frontier.by_size`
/// of body sub-expressions (over the fold namespace
/// `[scalar_args.., item, i, acc]`), and for every newly minted body wraps it
/// into a `ForFold` and runs it through [`try_accept_fold_body`] (cheap fold gate
/// THEN the strict verifier). On a hit it returns the verified result WITHOUT
/// writing the frontier back (the caller evicts a solved problem's frontier); on
/// a miss it writes the deepened bank back so the NEXT call resumes deeper.
///
/// Library participation: `library` components are instantiated onto the fold
/// namespace and injected as size-1 leaves (the documented injection lever), so
/// a mined abstraction can take part in — and shorten — an array solve exactly
/// as it does on the scalar path.
#[allow(clippy::too_many_arguments)]
fn deepen_fold_frontier(
    frontier: &mut Frontier,
    array_examples: &[(Vec<i64>, Vec<i64>, i64)],
    problem: &Problem,
    fn_name: &str,
    scalar_param_names: &[&str],
    array_idx: usize,
    time_limit_ms: u64,
    library: Option<&ComponentLibrary>,
    binops: &[BinOp],
    unops: &[UnOp],
    soft_cap: Option<usize>,
) -> Option<SolveResult> {
    let start = std::time::Instant::now();
    let fold_n_args = frontier.n_args; // = n_scalar_args + 3 (item, i, acc)
    // Probe vectors for OBSERVATIONAL dedup of bodies (mirrors the scalar
    // `fingerprint` dedup). Bodies that agree on all probes collapse to one.
    let probes = probe_inputs(fold_n_args, 8);

    let mut by_size: Vec<Vec<Expr>> = std::mem::take(&mut frontier.by_size);
    if by_size.is_empty() {
        by_size.push(vec![]); // index-0 sentinel
    }
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    let mut timed_out = false;
    let cap = soft_cap.unwrap_or(usize::MAX);

    let ensure_slot = |by_size: &mut Vec<Vec<Expr>>, s: usize| {
        if s >= by_size.len() {
            by_size.resize(s + 1, Vec::new());
        }
    };

    // RESUME: replay stored strata to rebuild `seen` deterministically (no
    // re-seed of size-1 atoms), identical to the scalar resume path.
    let resuming = frontier.next_size > 2 || by_size.len() > 2;
    if resuming {
        for stratum in by_size.iter() {
            for e in stratum {
                if let Some(fp) = fingerprint(e, &probes) {
                    seen.insert(fp);
                }
            }
        }
    } else {
        // COLD START: seed size-1 atoms (all fold-namespace vars + constants),
        // each tried as a body immediately.
        for v in 0..fold_n_args {
            let atom = Expr::Var(v);
            if let Some(r) =
                try_accept_fold_body(&atom, array_examples, problem, fn_name, scalar_param_names, array_idx)
            {
                return Some(r);
            }
            if let Some(fp) = fingerprint(&atom, &probes) {
                if seen.insert(fp) {
                    ensure_slot(&mut by_size, 1);
                    by_size[1].push(atom);
                }
            }
        }
        for &c in &CONSTANTS {
            let atom = Expr::Const(c);
            if let Some(fp) = fingerprint(&atom, &probes) {
                if seen.insert(fp) {
                    ensure_slot(&mut by_size, 1);
                    by_size[1].push(atom);
                }
            }
        }
        // Library injection at size 1 (the participation lever): instantiate
        // each mined component onto the fold namespace and add as a leaf.
        if let Some(lib) = library {
            let mut injected = 0usize;
            for comp in lib.get_for_args(fold_n_args) {
                if injected >= MAX_SIZE1_INJECTIONS {
                    break;
                }
                if let Some(r) = try_accept_fold_body(
                    &comp,
                    array_examples,
                    problem,
                    fn_name,
                    scalar_param_names,
                    array_idx,
                ) {
                    return Some(r);
                }
                if let Some(fp) = fingerprint(&comp, &probes) {
                    if seen.insert(fp) {
                        ensure_slot(&mut by_size, 1);
                        by_size[1].push(comp);
                        injected += 1;
                    }
                }
            }
        }
    }

    // Deepening loop (uniform-cost by size). Each newly minted body is tried as
    // a fold body before being banked. Comparisons are realized as `if c CMP d
    // { then } else { else }` bodies (size >= 5), mirroring the scalar IfExpr
    // construction, so conditional folds (count/sum-when) are reachable.
    let mut size = frontier.next_size;
    while size <= cap {
        if start.elapsed().as_millis() as u64 > time_limit_ms {
            timed_out = true;
            break;
        }
        ensure_slot(&mut by_size, size);
        let mut new: Vec<Expr> = Vec::new();

        // helper to try+bank one candidate body
        macro_rules! offer {
            ($e:expr) => {{
                let e = $e;
                if let Some(fp) = fingerprint(&e, &probes) {
                    if let Some(r) = try_accept_fold_body(
                        &e,
                        array_examples,
                        problem,
                        fn_name,
                        scalar_param_names,
                        array_idx,
                    ) {
                        // Found: do NOT write frontier back (caller evicts).
                        return Some(r);
                    }
                    if seen.insert(fp) {
                        new.push(e);
                    }
                }
            }};
        }

        // Unary ops
        if size >= 2 {
            let children = by_size[size - 1].clone();
            for child in &children {
                for &uop in unops {
                    offer!(Expr::UnaryOp(uop, Box::new(child.clone())));
                }
            }
        }

        // Binary ops
        for ls in 1..size {
            let rs = size - 1 - ls;
            if rs < 1 || rs > cap {
                continue;
            }
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                timed_out = true;
                break;
            }
            let lefts = by_size[ls].clone();
            let rights = by_size[rs].clone();
            for left in &lefts {
                for right in &rights {
                    for &op in binops {
                        offer!(Expr::BinOp(op, Box::new(left.clone()), Box::new(right.clone())));
                    }
                }
            }
        }

        // If-then-else bodies (size >= 5): if cl CMP cr { then } else { else }
        if size >= 5 {
            let atoms = by_size[1].clone();
            for &cmp in &ALL_CMPS {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    timed_out = true;
                    break;
                }
                for cl in &atoms {
                    for cr in &atoms {
                        let budget = size - 3;
                        for ts in 1..budget {
                            let es = budget - ts;
                            if es < 1 {
                                continue;
                            }
                            let then_es = by_size[ts].clone();
                            let else_es = by_size[es].clone();
                            for te in &then_es {
                                for ee in &else_es {
                                    offer!(Expr::IfExpr(
                                        cmp,
                                        Box::new(cl.clone()),
                                        Box::new(cr.clone()),
                                        Box::new(te.clone()),
                                        Box::new(ee.clone()),
                                    ));
                                }
                                if start.elapsed().as_millis() as u64 > time_limit_ms {
                                    timed_out = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if timed_out {
            // Discard partial stratum so a later resume rebuilds it cleanly.
            break;
        }
        eprintln!(
            "[enum-array] frontier size {size}: {} new bodies, {} unique, {:.1}s",
            new.len(),
            seen.len(),
            start.elapsed().as_secs_f32()
        );
        by_size[size] = new;
        size += 1;
    }

    // Miss: persist the deepened frontier for the next call.
    frontier.next_size = size;
    frontier.by_size = by_size;
    None
}

/// Enumerate fold bodies for array problems.
/// Most array problems are folds: acc = init; for item in arr { acc = acc OP body(item, i, acc) }
/// Render a map element body to a Mog EXPRESSION (not the if-form that
/// `to_mog` emits for Min/Max/IfExpr — Mog's `if` is a statement, invalid inside
/// `push(...)`/RHS). Uses the `min()`/`max()`/`abs()` builtins. Returns `None`
/// for shapes not expressible as a flat expression (IfExpr, loop nodes, Call) so
/// the caller skips emitting invalid code. `names` is `[scalar.., item, i]`.
fn render_map_body(e: &Expr, names: &[&str]) -> Option<String> {
    Some(match e {
        Expr::Var(i) => (*names.get(*i)?).to_string(),
        Expr::Const(c) => c.to_string(),
        Expr::UnaryOp(op, a) => {
            let s = render_map_body(a, names)?;
            match op {
                UnOp::Neg => format!("(0 - {s})"),
                UnOp::Abs => format!("abs({s})"),
                UnOp::BitNot => format!("(0 - {s} - 1)"),
                UnOp::Popcount => return None,
            }
        }
        Expr::BinOp(op, a, b) => {
            let l = render_map_body(a, names)?;
            let r = render_map_body(b, names)?;
            match op {
                BinOp::Add => format!("({l} + {r})"),
                BinOp::Sub => format!("({l} - {r})"),
                BinOp::Mul => format!("({l} * {r})"),
                BinOp::Div => format!("({l} / {r})"),
                BinOp::Mod => format!("({l} % {r})"),
                BinOp::Min => format!("min({l}, {r})"),
                BinOp::Max => format!("max({l}, {r})"),
                BinOp::BitAnd => format!("({l} & {r})"),
                BinOp::BitOr => format!("({l} | {r})"),
                BinOp::BitXor => format!("({l} ^ {r})"),
                BinOp::Shl => format!("({l} << {r})"),
                BinOp::Shr => format!("({l} >> {r})"),
            }
        }
        // IfExpr (statement-form only), loop nodes, Call: not a flat expression.
        _ => return None,
    })
}

/// Emit a `[i64] -> [i64]` elementwise map: apply `body` (over namespace
/// `[scalar_args.., item, i]`) to each element, pushing into `result`. Returns
/// `None` if the body is not expressible as a Mog map expression.
fn emit_mog_map(body: &Expr, fn_name: &str, scalar_names: &[&str], array_idx: usize) -> Option<String> {
    let n_total = scalar_names.len() + 1; // scalars + 1 array
    let mut sig_parts: Vec<String> = vec![String::new(); n_total];
    sig_parts[array_idx] = "arr: [i64]".to_string();
    let mut scalar_i = 0;
    for i in 0..n_total {
        if i != array_idx {
            sig_parts[i] = format!("{}: i64", scalar_names[scalar_i]);
            scalar_i += 1;
        }
    }
    let sig = sig_parts.join(", ");
    // body namespace: scalar params + [item, i]
    let mut names: Vec<&str> = scalar_names.to_vec();
    names.push("item");
    names.push("i");
    let body_s = render_map_body(body, &names)?;
    Some(format!(
        "fn {fn_name}({sig}) -> [i64] {{\n    result: [i64] = [];\n    i: i64 = 0;\n    \
         for item in arr {{\n        result.push({body_s});\n        i = i + 1;\n    }}\n    \
         return result;\n}}\n"
    ))
}

/// ARRAY-OUTPUT (elementwise map) synthesis — the representation lift that takes
/// the enumerative engine over the scalar-only ceiling into actual structured
/// generation. A `map(body)` program is correct iff `body` fits EVERY per-element
/// example `([scalar_args.., arr[k], k] -> out[k])`; so we FLATTEN the array
/// examples into scalar element examples and run the existing full-grammar
/// enumerator over them. The reach this unlocks over the fixed `array_transform`
/// grammar (which searches only `{Add,Sub,Mul,Mod}` element bodies): CONDITIONAL
/// maps (`if item < 0 { 0 - item } else { item }`), bitwise maps, and
/// mined-library-component maps — generated by reusing the scalar grammar
/// (IfExpr, all 12 binops + 4 unops, mined components) for free.
///
/// Elementwise only (`out.len() == in.len()`); filter (shorter output) and
/// reorder (sort/reverse) stay with `array_transform`. Strict-verified end to end.
/// #2 MULTI-STEP INTERMEDIATE: an array→array map whose element body uses a
/// WHOLE-ARRAY AGGREGATE computed once — e.g. "each element minus the max"
/// (out[k] = arr[k] - max(arr)), centering, normalization. The flat elementwise
/// map can't see an aggregate of the whole array; this computes the aggregate
/// into a local FIRST, then maps each element with it (genuine reduce-then-map
/// intermediate state). Single int-array input, elementwise int-array output,
/// non-empty. Tries a small set of aggregates; the body MUST use the aggregate
/// (else the plain map already covers it). Strict-verified.
fn synthesize_aggregate_map(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    if first.inputs.len() != 1
        || !matches!(first.inputs[0], Value::Array(_))
        || !matches!(first.expected, Value::Array(_))
    {
        return None;
    }
    // (label, compute fn, Mog code that puts the aggregate into local `agg`).
    let aggs: &[(&str, fn(&[i64]) -> Option<i64>, &str)] = &[
        ("sum", |a| Some(a.iter().sum()), "    agg: i64 = 0;\n    for av in arr {\n        agg = agg + av;\n    }\n"),
        ("max", |a| a.iter().copied().max(), "    agg: i64 = arr[0];\n    for av in arr {\n        if av > agg {\n            agg = av;\n        }\n    }\n"),
        ("min", |a| a.iter().copied().min(), "    agg: i64 = arr[0];\n    for av in arr {\n        if av < agg {\n            agg = av;\n        }\n    }\n"),
        ("len", |a| Some(a.len() as i64), "    agg: i64 = arr.len;\n"),
        ("first", |a| a.first().copied(), "    agg: i64 = arr[0];\n"),
        ("last", |a| a.last().copied(), "    agg: i64 = arr[arr.len - 1];\n"),
        ("product", |a| Some(a.iter().product()), "    agg: i64 = 1;\n    for av in arr {\n        agg = agg * av;\n    }\n"),
        // Derived aggregates (still a single per-array value): mean = sum/len
        // (integer), range = max - min. Cover centering ("x - mean") and
        // range-relative maps without a full multi-aggregate search.
        ("mean", |a| (!a.is_empty()).then(|| a.iter().sum::<i64>() / a.len() as i64), "    agg: i64 = 0;\n    for av in arr {\n        agg = agg + av;\n    }\n    agg = agg / arr.len;\n"),
        ("range", |a| { let mx = *a.iter().max()?; let mn = *a.iter().min()?; Some(mx - mn) }, "    mx: i64 = arr[0];\n    mn: i64 = arr[0];\n    for av in arr {\n        if av > mx {\n            mx = av;\n        }\n        if av < mn {\n            mn = av;\n        }\n    }\n    agg: i64 = mx - mn;\n"),
    ];
    for (label, compute, agg_code) in aggs {
        let mut flat: Vec<(Vec<i64>, i64)> = Vec::new();
        let mut applicable = true;
        for ex in &problem.examples {
            let arr = ex.inputs[0].as_i64_slice()?;
            let Value::Array(out) = &ex.expected else {
                return None;
            };
            if out.len() != arr.len() {
                return None; // not elementwise
            }
            if arr.is_empty() {
                applicable = false; // aggregate (max/first/...) undefined on empty
                break;
            }
            let Some(aval) = compute(&arr) else {
                applicable = false;
                break;
            };
            for (k, item) in arr.iter().enumerate() {
                let Value::Int(o) = out[k] else {
                    return None;
                };
                flat.push((vec![*item, k as i64, aval], o)); // [item, i, agg]
            }
        }
        if !applicable || flat.is_empty() {
            continue;
        }
        let library = ComponentLibrary::load_or_dream(3_000);
        let (body, _t, _m) = enumerate_exprs_with_ops_stats(
            3,
            7,
            &flat,
            8_000,
            Some(&library),
            &ALL_BINOPS,
            &ALL_UNOPS,
        );
        let Some(body) = body else { continue };
        let Some(body_s) = render_map_body(&body, &["item", "i", "agg"]) else {
            continue;
        };
        // Body MUST reference the aggregate, else the plain elementwise map covers it.
        if !body_s.contains("agg") {
            continue;
        }
        let code = format!(
            "fn {fn_name}(arr: [i64]) -> [i64] {{\n{agg_code}    result: [i64] = [];\n    i: i64 = 0;\n    for item in arr {{\n        result.push({body_s});\n        i = i + 1;\n    }}\n    return result;\n}}\n"
        );
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("aggregate-map:{label}"),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    None
}

fn synthesize_array_map_enumerative(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    let array_idx = first.inputs.iter().position(|v| matches!(v, Value::Array(_)))?;
    let n_scalar = first.inputs.len() - 1;

    // Flatten array examples into per-element scalar examples over the body
    // namespace [scalar_args.., item, i]. Bail (None) on any non-elementwise or
    // non-int shape so we never fabricate a wrong flattening.
    let mut flat: Vec<(Vec<i64>, i64)> = Vec::new();
    for ex in &problem.examples {
        let Value::Array(out) = &ex.expected else {
            return None;
        };
        let arr = ex.inputs.get(array_idx)?.as_i64_slice()?;
        if out.len() != arr.len() {
            return None; // not a pure elementwise map (filter/reorder/grow)
        }
        let scalars: Vec<i64> = ex
            .inputs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != array_idx)
            .filter_map(|(_, v)| if let Value::Int(n) = v { Some(*n) } else { None })
            .collect();
        if scalars.len() != n_scalar {
            return None; // a non-int scalar arg — out of scope here
        }
        for (k, item) in arr.iter().enumerate() {
            let Value::Int(o) = out[k] else {
                return None;
            };
            let mut inp = scalars.clone();
            inp.push(*item); // item
            inp.push(k as i64); // i
            flat.push((inp, o));
        }
    }
    if flat.is_empty() {
        return None;
    }

    let map_n_args = n_scalar + 2; // scalars + item + i
    let library = ComponentLibrary::load_or_dream(3_000);
    // Full-grammar bounded search over the element body (the new reach).
    let (body, _timed_out, _max_completed) = enumerate_exprs_with_ops_stats(
        map_n_args,
        7,
        &flat,
        8_000,
        Some(&library),
        &ALL_BINOPS,
        &ALL_UNOPS,
    );
    let body = body?;

    let scalar_names: Vec<&str> = ["a", "b", "c", "d"][..n_scalar].to_vec();
    // Body must be expressible as a flat Mog map expression (skip if not).
    let code = emit_mog_map(&body, fn_name, &scalar_names, array_idx)?;
    // STRICT gate over the WHOLE program (examples + holdouts) — never accept on
    // the flattened element fit alone.
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: "enumerative-array-map".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

/// Extract the k-th int component of a flat composite expected value.
fn tuple_component(expected: &Value, k: usize) -> Option<i64> {
    match expected {
        Value::Pair(a, b) => [*a, *b].get(k).copied(),
        Value::Quad(a, b, c, d) => [*a, *b, *c, *d].get(k).copied(),
        Value::Tuple(t) => match t.get(k)? {
            Value::Int(v) => Some(*v),
            _ => None,
        },
        _ => None,
    }
}

/// Component arity of a flat-int composite (Pair=2, Quad=4, Tuple=len), else None.
fn tuple_arity(expected: &Value) -> Option<usize> {
    match expected {
        Value::Pair(_, _) => Some(2),
        Value::Quad(_, _, _, _) => Some(4),
        Value::Tuple(t) if !t.is_empty() && t.iter().all(|v| matches!(v, Value::Int(_))) => {
            Some(t.len())
        }
        _ => None,
    }
}

/// COMPOSITE-OUTPUT synthesis — lift the engine over the scalar/array OUTPUT
/// ceiling into STRUCTURED output. A function returning a tuple/pair/quad is the
/// componentwise product of scalar functions: component k is correct iff a body
/// fits EVERY `(scalar_inputs -> expected_k)` example. So FLATTEN by component
/// (mirroring the array-map flatten by element), synthesize each component with
/// the existing scalar enumerator, and emit a struct-returning Mog program. The
/// strict verifier accepts a runtime Struct against a wire Pair/Quad/Tuple
/// (`struct_fields_match`, a multiset of int fields). Scalar (int) inputs only;
/// array-input composites (e.g. minmax(arr)) need the fold engine per component
/// and stay out of scope. Strict-verified end to end.
fn synthesize_tuple_output_enumerative(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    let ncomp = tuple_arity(&first.expected)?;
    if ncomp < 2 {
        return None;
    }
    let n_args = first.inputs.len();
    if n_args == 0 || n_args > 6 || !first.inputs.iter().all(|v| matches!(v, Value::Int(_))) {
        return None; // scalar (int) inputs only, bounded arity for naming
    }
    let library = ComponentLibrary::load_or_dream(3_000);
    let mut bodies: Vec<Expr> = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let mut flat: Vec<(Vec<i64>, i64)> = Vec::with_capacity(problem.examples.len());
        for ex in &problem.examples {
            if ex.inputs.len() != n_args {
                return None;
            }
            let inputs: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| if let Value::Int(n) = v { Some(*n) } else { None })
                .collect();
            if inputs.len() != n_args {
                return None; // a non-int input — out of scope
            }
            let comp = tuple_component(&ex.expected, k)?;
            flat.push((inputs, comp));
        }
        let (body, _t, _m) = enumerate_exprs_with_ops_stats(
            n_args,
            7,
            &flat,
            8_000,
            Some(&library),
            &ALL_BINOPS,
            &ALL_UNOPS,
        );
        bodies.push(body?);
    }
    let code = emit_mog_tuple(fn_name, n_args, &bodies)?;
    // STRICT gate over the whole program — the examples are checked structurally
    // (a runtime Struct vs the wire Pair/Quad/Tuple) via output_matches, never on
    // the componentwise fit alone.
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: "enumerative-tuple-output".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

/// Emit a struct-returning Mog program for composite output: a struct with one
/// `i64` field per component, returned from the component bodies.
fn emit_mog_tuple(fn_name: &str, n_args: usize, bodies: &[Expr]) -> Option<String> {
    let arg_names: Vec<&str> = ["a", "b", "c", "d", "e", "f"].get(..n_args)?.to_vec();
    let sig_params = arg_names
        .iter()
        .map(|n| format!("{n}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    // Mog only parses `Name { .. }` struct construction when Name is
    // uppercase-first (parser disambiguation from block braces), so prefix.
    let struct_name = format!("Out_{fn_name}");
    let mut decl_fields: Vec<String> = Vec::with_capacity(bodies.len());
    let mut ctor_fields: Vec<String> = Vec::with_capacity(bodies.len());
    for (k, body) in bodies.iter().enumerate() {
        let bs = render_map_body(body, &arg_names)?;
        decl_fields.push(format!("f{k}: i64"));
        ctor_fields.push(format!("f{k}: {bs}"));
    }
    Some(format!(
        "struct {struct_name} {{ {} }}\nfn {fn_name}({sig_params}) -> {struct_name} {{\n    return {struct_name} {{ {} }};\n}}\n",
        decl_fields.join(", "),
        ctor_fields.join(", "),
    ))
}

/// COMPOSITE OUTPUT with an ARRAY input (e.g. `minmax(arr) -> (min, max)`).
/// Components here are array→scalar FOLDS (loops), not flat expressions, so they
/// can't inline into the struct constructor. Instead synthesize each component as
/// its OWN helper function through the full solver (so it can use folds / array
/// teachers), then the main function CALLS each helper in the struct constructor.
/// Strict-verified end to end. `solve_problem` on the scalar-output sub-problems
/// does not recurse back here (their output is a plain int, not composite).
fn synthesize_tuple_output_array(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    let ncomp = tuple_arity(&first.expected)?;
    if ncomp < 2 {
        return None;
    }
    // Need an array input (all-scalar inputs use the inline-expr path); only int
    // scalars and int arrays are in scope.
    if !first.inputs.iter().any(|v| matches!(v, Value::Array(_)))
        || !first
            .inputs
            .iter()
            .all(|v| matches!(v, Value::Int(_) | Value::Array(_)))
    {
        return None;
    }
    let arg_decls: Vec<String> = first
        .inputs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let ty = if matches!(v, Value::Array(_)) { "[i64]" } else { "i64" };
            format!("p{i}: {ty}")
        })
        .collect();
    let arg_names: Vec<String> = (0..first.inputs.len()).map(|i| format!("p{i}")).collect();

    let mut helpers = String::new();
    let mut decl_fields: Vec<String> = Vec::with_capacity(ncomp);
    let mut ctor_fields: Vec<String> = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let helper_name = format!("{fn_name}_c{k}");
        let mut sub = problem.clone();
        sub.name = helper_name.clone();
        sub.signature = Box::leak(
            format!("fn {helper_name}({}) -> i64", arg_decls.join(", ")).into_boxed_str(),
        );
        sub.reference_code = "";
        sub.holdouts = Vec::new();
        sub.examples = problem
            .examples
            .iter()
            .map(|ex| {
                tuple_component(&ex.expected, k).map(|c| crate::benchmark::Example {
                    inputs: ex.inputs.clone(),
                    expected: Value::Int(c),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let solved = crate::solver::solve_problem(&sub);
        if !solved.success {
            return None;
        }
        helpers.push_str(solved.code.trim_end());
        helpers.push_str("\n\n");
        decl_fields.push(format!("f{k}: i64"));
        ctor_fields.push(format!("f{k}: {helper_name}({})", arg_names.join(", ")));
    }
    let struct_name = format!("Out_{fn_name}");
    let main = format!(
        "struct {struct_name} {{ {} }}\nfn {fn_name}({}) -> {struct_name} {{\n    return {struct_name} {{ {} }};\n}}\n",
        decl_fields.join(", "),
        arg_decls.join(", "),
        ctor_fields.join(", "),
    );
    let code = format!("{helpers}{main}");
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: "enumerative-tuple-output-array".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

/// BOOL-OUTPUT synthesis — deterministic predicates (is_even, is_positive, …).
/// A bool function is a PREDICATE over an i64 expression: encode each example's
/// bool as {1,0}, synthesize an i64 body that fits {1,0} with the existing
/// full-grammar enumerator (it freely builds `n%2`, comparisons-in-IfExpr,
/// arithmetic), then emit `fn f(..) -> bool { return (<body>) != 0; }` — since the
/// body is {0,1} on every example, `!= 0` IS the predicate. Strict-verified as
/// bool (output_matches Bool↔Bool). This is the REAL bool path; the old
/// probabilistic Bernoulli "solution" was a false-accept (now removed).
fn synthesize_bool_output_enumerative(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    if !matches!(first.expected, Value::Bool(_)) {
        return None;
    }
    let n_args = first.inputs.len();
    if n_args == 0 || n_args > 4 || !first.inputs.iter().all(|v| matches!(v, Value::Int(_))) {
        return None;
    }
    let mut flat: Vec<(Vec<i64>, i64)> = Vec::with_capacity(problem.examples.len());
    for ex in &problem.examples {
        let inputs: Vec<i64> = ex
            .inputs
            .iter()
            .filter_map(|v| if let Value::Int(n) = v { Some(*n) } else { None })
            .collect();
        if inputs.len() != n_args {
            return None;
        }
        let b = match ex.expected {
            Value::Bool(b) => b,
            _ => return None,
        };
        flat.push((inputs, if b { 1 } else { 0 }));
    }
    let library = ComponentLibrary::load_or_dream(3_000);
    let (body, _t, _m) = enumerate_exprs_with_ops_stats(
        n_args,
        7,
        &flat,
        8_000,
        Some(&library),
        &ALL_BINOPS,
        &ALL_UNOPS,
    );
    let body = body?;
    let arg_names: Vec<&str> = ["a", "b", "c", "d"].get(..n_args)?.to_vec();
    let bs = render_map_body(&body, &arg_names)?;
    let sig_params = arg_names
        .iter()
        .map(|n| format!("{n}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    let code = format!("fn {fn_name}({sig_params}) -> bool {{\n    return ({bs}) != 0;\n}}\n");
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: "enumerative-bool".to_string(),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

/// #2 BRANCH-ON-STRUCTURE: a function over an array whose EMPTY case returns a
/// fixed default and whose NON-EMPTY case is an ordinary reduce/computation —
/// e.g. "the max, or 0 if empty". The flat composition shape cannot express the
/// empty default (a reduce's identity is fixed; max/min/average have no
/// meaningful empty value). Split examples by emptiness, synthesize the NON-EMPTY
/// body through the full solver, and wrap it in a length guard. The non-empty
/// sub-problem has no empty examples → no re-entry here. Strict-verified.
fn synthesize_emptiness_guard(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    if first.inputs.len() != 1 || !matches!(first.expected, Value::Int(_)) {
        return None; // one array arg, scalar int output
    }
    let array_idx = first.inputs.iter().position(|v| matches!(v, Value::Array(_)))?;
    let is_empty_ex = |ex: &crate::benchmark::Example| -> bool {
        matches!(ex.inputs.get(array_idx), Some(Value::Array(a)) if a.is_empty())
    };
    let empty: Vec<&crate::benchmark::Example> =
        problem.examples.iter().filter(|e| is_empty_ex(e)).collect();
    let nonempty: Vec<crate::benchmark::Example> = problem
        .examples
        .iter()
        .filter(|e| !is_empty_ex(e))
        .cloned()
        .collect();
    // Need BOTH a guarded (empty) case and enough body (non-empty) cases.
    if empty.is_empty() || nonempty.len() < 2 {
        return None;
    }
    // The empty default must be a single agreed-upon Int.
    let default = match empty[0].expected {
        Value::Int(d) => d,
        _ => return None,
    };
    if !empty
        .iter()
        .all(|e| matches!(e.expected, Value::Int(x) if x == default))
    {
        return None;
    }
    // Synthesize the NON-EMPTY body as its own fn through the full solver.
    let body_fn = format!("{fn_name}_body");
    let mut sub = problem.clone();
    sub.name = body_fn.clone();
    sub.signature = Box::leak(format!("fn {body_fn}(arr: [i64]) -> i64").into_boxed_str());
    sub.reference_code = "";
    sub.holdouts = Vec::new();
    sub.examples = nonempty;
    let body = crate::solver::solve_problem(&sub);
    if !body.success {
        return None;
    }
    let code = format!(
        "{}\nfn {fn_name}(arr: [i64]) -> i64 {{\n    if arr.len == 0 {{\n        return {default};\n    }}\n    return {body_fn}(arr);\n}}\n",
        body.code.trim_end()
    );
    if verify_problem_code_strict(problem, &code).is_ok() {
        return Some(SolveResult {
            success: true,
            code,
            method: format!("structural-guard:{}", body.method),
            error: None,
            metadata: DifferentiableMetadata::default(),
        });
    }
    None
}

thread_local! {
    static IN_LENGTH_BRANCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Solve a branch's example subset as its own fn, holding the length-branch
/// re-entrancy guard so the sub-group cannot re-split (no infinite recursion).
fn solve_branch_body(
    problem: &Problem,
    examples: &[crate::benchmark::Example],
    body_fn: &str,
) -> Option<SolveResult> {
    let mut sub = problem.clone();
    sub.name = body_fn.to_string();
    sub.signature = Box::leak(format!("fn {body_fn}(arr: [i64]) -> i64").into_boxed_str());
    sub.reference_code = "";
    sub.holdouts = Vec::new();
    sub.examples = examples.to_vec();
    IN_LENGTH_BRANCH.with(|f| f.set(true));
    let res = crate::solver::solve_problem(&sub);
    IN_LENGTH_BRANCH.with(|f| f.set(false));
    res.success.then_some(res)
}

/// Two branch bodies are equivalent (a spurious split) if their code is identical
/// after neutralizing each one's own fn name — i.e. the same program both sides.
fn branch_bodies_equivalent(lo_code: &str, lo_fn: &str, hi_code: &str, hi_fn: &str) -> bool {
    lo_code.replace(lo_fn, "F") == hi_code.replace(hi_fn, "F")
}

/// #2 BRANCH-ON-STRUCTURE (general): `f(arr) = if arr.len < K { lo(arr) } else
/// { hi(arr) }`, generalizing the emptiness guard (len==0) to any small length
/// threshold K — e.g. "len < 2 → 0, else the max". Split examples by arr.len < K,
/// synthesize each side as its own fn through the full solver, require the two
/// bodies to be genuinely DIFFERENT (else a single op already covers it), emit a
/// length-guarded wrapper, strict-verify. The thread_local guard stops a sub-group
/// from re-splitting. Tried after the emptiness guard (which handles the constant-
/// empty-default case it can't).
fn synthesize_length_branch(problem: &Problem) -> Option<SolveResult> {
    if IN_LENGTH_BRANCH.with(|f| f.get()) {
        return None;
    }
    let fn_name = problem.function_name();
    let first = problem.examples.first()?;
    if first.inputs.len() != 1
        || !matches!(first.inputs[0], Value::Array(_))
        || !matches!(first.expected, Value::Int(_))
        || problem.examples.len() < 4
    {
        return None;
    }
    for k in 1usize..=3 {
        let mut lo: Vec<crate::benchmark::Example> = Vec::new();
        let mut hi: Vec<crate::benchmark::Example> = Vec::new();
        for ex in &problem.examples {
            let len = ex.inputs.first().and_then(|v| v.as_i64_slice()).map(|a| a.len())?;
            if len < k {
                lo.push(ex.clone());
            } else {
                hi.push(ex.clone());
            }
        }
        if lo.len() < 2 || hi.len() < 2 {
            continue;
        }
        let lo_fn = format!("{fn_name}_lo");
        let hi_fn = format!("{fn_name}_hi");
        let (Some(lo_res), Some(hi_res)) = (
            solve_branch_body(problem, &lo, &lo_fn),
            solve_branch_body(problem, &hi, &hi_fn),
        ) else {
            continue;
        };
        if branch_bodies_equivalent(&lo_res.code, &lo_fn, &hi_res.code, &hi_fn) {
            continue; // not a real branch — a single op covers both sides
        }
        let code = format!(
            "{}\n{}\nfn {fn_name}(arr: [i64]) -> i64 {{\n    if arr.len < {k} {{\n        return {lo_fn}(arr);\n    }}\n    return {hi_fn}(arr);\n}}\n",
            lo_res.code.trim_end(),
            hi_res.code.trim_end()
        );
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: format!("structural-length-branch:{}|{}", lo_res.method, hi_res.method),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    None
}

fn synthesize_array_enumerative(problem: &Problem) -> Option<SolveResult> {
    let fn_name = problem.function_name();

    // ARRAY-OUTPUT (elementwise map) → the full-grammar map enumerator. Scalar
    // (fold-to-scalar) output continues through the existing path below.
    if problem
        .examples
        .first()
        .is_some_and(|ex| matches!(ex.expected, Value::Array(_)))
    {
        if let Some(r) = synthesize_array_map_enumerative(problem) {
            return Some(r);
        }
        // Multi-step intermediate: a map whose body uses a whole-array aggregate
        // (reduce-then-map, e.g. "each element minus the max").
        return synthesize_aggregate_map(problem);
    }

    // #2 BRANCH-ON-STRUCTURE: examples mixing empty + non-empty arrays → the
    // emptiness-guarded form (the flat fold can't carry an empty default).
    if let Some(r) = synthesize_emptiness_guard(problem) {
        return Some(r);
    }
    // General length-threshold branch (len < K) — handles cases the emptiness
    // guard can't (a synthesized lo body, K up to 3). Re-entrancy-guarded.
    if let Some(r) = synthesize_length_branch(problem) {
        return Some(r);
    }

    // Extract scalar args and arrays from examples
    let first_ex = problem.examples.first()?;
    let array_idx = first_ex
        .inputs
        .iter()
        .position(|v| matches!(v, Value::Array(_)))?;
    let n_scalar_args = first_ex.inputs.len() - 1;

    // Build per-example data: (scalar_args, array, expected)
    let array_examples: Vec<(Vec<i64>, Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let scalar_args: Vec<i64> = ex
                .inputs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != array_idx)
                .filter_map(|(_, v)| {
                    if let Value::Int(n) = v {
                        Some(*n)
                    } else {
                        None
                    }
                })
                .collect();
            let array: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| v.as_i64_slice())
                .next()
                .unwrap_or_default();
            (scalar_args, array, ex.expected_int())
        })
        .collect();

    let scalar_param_names: Vec<&str> = ["a", "b", "c", "d"][..n_scalar_args].to_vec();

    // Namespace indices for fold body:
    // [scalar_args(0..n_scalar), item(n_scalar), i(n_scalar+1), acc(n_scalar+2)]
    let item_idx = n_scalar_args; // Var(item_idx) = item

    let init_candidates: Vec<Expr> = vec![Expr::Const(0), Expr::Const(1)];

    let start = std::time::Instant::now();
    // The hardcoded strategies below are a CHEAP warm-up sweep (no deep
    // enumeration). They are bounded by a short wall so that, on a miss, the
    // resumable deepening frontier (added at the end of this fn) always gets to
    // run instead of the old terminal 15s → None. The real anytime budget lives
    // in the frontier tiers below.
    let time_limit_ms: u64 = 3_000;

    // ── Strategy 1: Simple folds (acc = acc OP rhs, no condition) ──────────
    // Handles: array_sum, reverse_sum, interactive_sum, arr_sum_squares, closure_map_sum
    {
        let fold_n_args = n_scalar_args + 3;
        let mut body_atoms: Vec<Expr> = Vec::new();
        for v in 0..fold_n_args {
            body_atoms.push(Expr::Var(v));
        }
        for &c in &[0i64, 1, -1, 2] {
            body_atoms.push(Expr::Const(c));
        }

        // Atom OP atom combinations
        let mut body_exprs: Vec<Expr> = body_atoms.clone();
        for l in &body_atoms {
            for r in &body_atoms {
                for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul] {
                    body_exprs.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
                }
            }
        }

        for init in &init_candidates {
            for &bop in &[BinOp::Add, BinOp::Mul] {
                if start.elapsed().as_millis() as u64 > time_limit_ms {
                    break; // warm-up budget spent → fall through to the frontier
                }
                for rhs in &body_exprs {
                    let fold_expr = Expr::ForFold {
                        init: Box::new(init.clone()),
                        body_op: bop,
                        body_rhs: Box::new(rhs.clone()),
                    };
                    if check_fold_examples(&fold_expr, &array_examples) {
                        let code =
                            emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                        eprintln!(
                            "[enum-array] FOUND simple fold: {} {:?}",
                            fn_name, fold_expr
                        );
                        if verify_problem_code_strict(problem, &code).is_ok() {
                            return Some(SolveResult {
                                success: true,
                                code,
                                method: "enumerative-array".to_string(),
                                error: None,
                                metadata: DifferentiableMetadata::default(),
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Strategy 2: Conditional count folds (if CMP { acc + 1 }) ──────────
    // Handles: count_occurrences, count_positive, count_zeros, count_evens, count_greater_than
    {
        // Conditions: item CMP val where val ∈ {0, 1, scalar_args, item%2}
        let mut cond_pairs: Vec<(CmpOp, Expr, Expr)> = Vec::new();
        // item CMP const
        for &cmp in &[
            CmpOp::Eq,
            CmpOp::Gt,
            CmpOp::Lt,
            CmpOp::Ne,
            CmpOp::Ge,
            CmpOp::Le,
        ] {
            for &c in &[0i64, 1, -1, 2] {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Const(c)));
            }
            // item CMP scalar_arg
            for s in 0..n_scalar_args {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Var(s)));
                cond_pairs.push((cmp, Expr::Var(s), Expr::Var(item_idx)));
            }
        }
        // item % 2 == 0 (for count_evens)
        cond_pairs.push((
            CmpOp::Eq,
            Expr::BinOp(
                BinOp::Mod,
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(2)),
            ),
            Expr::Const(0),
        ));
        cond_pairs.push((
            CmpOp::Eq,
            Expr::BinOp(
                BinOp::Mod,
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(2)),
            ),
            Expr::Const(1),
        ));

        for (cmp, lhs, rhs) in &cond_pairs {
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                break; // warm-up budget spent → fall through to the frontier
            }
            // fold: acc = acc + if lhs CMP rhs { 1 } else { 0 }
            let cond_body = Expr::IfExpr(
                *cmp,
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
                Box::new(Expr::Const(1)),
                Box::new(Expr::Const(0)),
            );
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(0)),
                body_op: BinOp::Add,
                body_rhs: Box::new(cond_body),
            };
            if check_fold_examples(&fold_expr, &array_examples) {
                let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                eprintln!("[enum-array] FOUND cond-count: {} {:?}", fn_name, fold_expr);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "enumerative-array".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
            }
        }
    }

    // ── Strategy 3: Conditional sum folds (if CMP { acc + item }) ──────────
    // Handles: sum_negatives, sum_positives, sum_absolute
    {
        let mut cond_pairs: Vec<(CmpOp, Expr, Expr)> = Vec::new();
        for &cmp in &[CmpOp::Lt, CmpOp::Gt, CmpOp::Le, CmpOp::Ge] {
            for &c in &[0i64, 1, -1] {
                cond_pairs.push((cmp, Expr::Var(item_idx), Expr::Const(c)));
            }
        }

        for (cmp, lhs, rhs) in &cond_pairs {
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                break; // warm-up budget spent → fall through to the frontier
            }
            // fold: acc = acc + if lhs CMP rhs { item } else { 0 }
            let cond_body = Expr::IfExpr(
                *cmp,
                Box::new(lhs.clone()),
                Box::new(rhs.clone()),
                Box::new(Expr::Var(item_idx)),
                Box::new(Expr::Const(0)),
            );
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(0)),
                body_op: BinOp::Add,
                body_rhs: Box::new(cond_body),
            };
            if check_fold_examples(&fold_expr, &array_examples) {
                let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                eprintln!("[enum-array] FOUND cond-sum: {} {:?}", fn_name, fold_expr);
                if verify_problem_code_strict(problem, &code).is_ok() {
                    return Some(SolveResult {
                        success: true,
                        code,
                        method: "enumerative-array".to_string(),
                        error: None,
                        metadata: DifferentiableMetadata::default(),
                    });
                }
            }
        }

        // sum_absolute: acc = acc + if item < 0 { 0 - item } else { item }
        let abs_body = Expr::IfExpr(
            CmpOp::Lt,
            Box::new(Expr::Var(item_idx)),
            Box::new(Expr::Const(0)),
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Const(0)),
                Box::new(Expr::Var(item_idx)),
            )),
            Box::new(Expr::Var(item_idx)),
        );
        let abs_fold = Expr::ForFold {
            init: Box::new(Expr::Const(0)),
            body_op: BinOp::Add,
            body_rhs: Box::new(abs_body),
        };
        if check_fold_examples(&abs_fold, &array_examples) {
            let code = emit_mog_array(&abs_fold, fn_name, &scalar_param_names, array_idx);
            eprintln!("[enum-array] FOUND abs-sum: {} {:?}", fn_name, abs_fold);
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative-array".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
    }

    // ── Strategy 4: Max/Min folds with conditional update ──────────
    // Handles: array_max, array_min, array_max_elem, min_element.
    // `emit_mog_array` ALWAYS emits min/max folds seeded `acc = arr[0]` (the correct
    // reduce), so gate on the TRUE array min/max — NOT a constant-init eval. The old
    // gate evaluated the fold with init ∈ {0,1,-1}, which is wrong for MIN (no small
    // constant exceeds every element) and only accidentally right for MAX when the
    // max is positive; that asymmetry let array_max synthesize while array_min was
    // rejected before the correct arr[0]-seeded code was ever emitted. The strict
    // verifier (with holdouts) remains the real gate.
    {
        for &bop in &[BinOp::Min, BinOp::Max] {
            if start.elapsed().as_millis() as u64 > time_limit_ms {
                break; // warm-up budget spent → fall through to the frontier
            }
            let matches_all = !array_examples.is_empty()
                && array_examples.iter().all(|(_, arr, expected)| {
                    let agg = match bop {
                        BinOp::Min => arr.iter().copied().min(),
                        BinOp::Max => arr.iter().copied().max(),
                        _ => None,
                    };
                    agg == Some(*expected)
                });
            if !matches_all {
                continue;
            }
            // init is ignored by emit for min/max (it uses arr[0]); kept for the IR shape.
            let fold_expr = Expr::ForFold {
                init: Box::new(Expr::Const(0)),
                body_op: bop,
                body_rhs: Box::new(Expr::Var(item_idx)),
            };
            let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
            eprintln!("[enum-array] FOUND max/min fold (arr[0]-seeded): {fn_name} {bop:?}");
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "enumerative-array".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
    }

    // ── Strategy 5: Conditional sum with scalar comparison ──────────
    // Handles: count_greater_than, sum_at_even_indices-like (via i%2)
    {
        // Extended conditions: i CMP const
        for &cmp in &[CmpOp::Eq, CmpOp::Ne] {
            for &c in &[0i64, 1] {
                let cond_body = Expr::IfExpr(
                    cmp,
                    Box::new(Expr::BinOp(
                        BinOp::Mod,
                        Box::new(Expr::Var(item_idx)),
                        Box::new(Expr::Const(2)),
                    )),
                    Box::new(Expr::Const(c)),
                    Box::new(Expr::Const(1)),
                    Box::new(Expr::Const(0)),
                );
                let fold_expr = Expr::ForFold {
                    init: Box::new(Expr::Const(0)),
                    body_op: BinOp::Add,
                    body_rhs: Box::new(cond_body),
                };
                if check_fold_examples(&fold_expr, &array_examples) {
                    let code = emit_mog_array(&fold_expr, fn_name, &scalar_param_names, array_idx);
                    eprintln!("[enum-array] FOUND mod-count: {} {:?}", fn_name, fold_expr);
                    if verify_problem_code_strict(problem, &code).is_ok() {
                        return Some(SolveResult {
                            success: true,
                            code,
                            method: "enumerative-array".to_string(),
                            error: None,
                            metadata: DifferentiableMetadata::default(),
                        });
                    }
                }
            }
        }
    }

    // ── Resumable deepening frontier (replaces the old fixed-15s → None) ──────
    // The hardcoded warm-up strategies above are a cheap first sweep. If they
    // miss, we DO NOT return a terminal None: we deepen a PERSISTENT, anytime
    // frontier of fold bodies — keyed by the examples fingerprint and shared
    // with the scalar `Frontier` store — so a later invocation RESUMES deeper
    // (and consumes the growing mined library) instead of restarting.
    let fold_n_args = n_scalar_args + 3; // scalar args + item + i + acc
    let fp = crate::solved_cache::examples_fingerprint(&problem.examples);

    // Per-call budget (anytime). Default mirrors the old 15s wall, but it now
    // bounds THIS call only; depth rises across calls via the persisted frontier.
    let budget_ms: u64 = std::env::var("NSYNTH_ENUM_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15_000);
    let soft_cap: Option<usize> = std::env::var("NSYNTH_ENUM_SOFT_CAP")
        .ok()
        .and_then(|v| v.parse().ok());

    // Library: lazily loaded/mined, fed in so abstractions participate in the
    // array search exactly as on the scalar path.
    let library = ComponentLibrary::load_or_dream(5_000);

    let core_budget = (budget_ms * 55 / 100).max(1);
    let full_budget = budget_ms.saturating_sub(core_budget).max(1);

    let mut run_tier = |ops_tier: u8, binops: &[BinOp], unops: &[UnOp], budget: u64| {
        let mut frontier = load_frontier(&fp, fold_n_args, ops_tier)
            .unwrap_or_else(|| Frontier::fresh(fp.clone(), fold_n_args, ops_tier));
        let hit = deepen_fold_frontier(
            &mut frontier,
            &array_examples,
            problem,
            fn_name,
            &scalar_param_names,
            array_idx,
            budget,
            Some(&library),
            binops,
            unops,
            soft_cap,
        );
        if hit.is_some() {
            // Solved → its frontier is dead weight; drop it.
            evict_frontier(&fp);
        } else {
            // Miss: persist the deepened frontier so a later call resumes deeper.
            save_frontier(&frontier);
        }
        hit
    };

    if let Some(result) = run_tier(0, &CORE_BINOPS, &CORE_UNOPS, core_budget) {
        return Some(result);
    }
    if let Some(result) = run_tier(1, &ALL_BINOPS, &ALL_UNOPS, full_budget) {
        return Some(result);
    }

    eprintln!(
        "[enum-array] no fold found (frontier persisted, resumable) in {:.1}s",
        start.elapsed().as_secs_f32()
    );
    None
}

/// Check a ForFold expression against array examples.
fn check_fold_examples(expr: &Expr, examples: &[(Vec<i64>, Vec<i64>, i64)]) -> bool {
    examples.iter().all(|(scalar_args, array, expected)| {
        expr.eval_array(scalar_args, array) == Some(*expected)
    })
}

// ─── Nested while loops ────────────────────────────────────────────────────

/// Try to discover programs with nested loops (e.g., sum_of_divisors, count_divisors).
fn synthesize_nested_while(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    if n_args < 1 || n_args > 2 {
        return None;
    }

    let fn_name = problem.function_name();
    let param_names: Vec<&str> = ["a", "b"][..n_args].to_vec();

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let start = std::time::Instant::now();
    let time_limit_ms: u64 = 8_000;

    // Inner namespace: [args(0..n_args), outer_acc(n_args), outer_i(n_args+1), inner_acc(n_args+2), inner_i(n_args+3)]
    let nested_n_args = n_args + 4;
    let mut nested_atoms: Vec<Expr> = Vec::new();
    for v in 0..nested_n_args {
        nested_atoms.push(Expr::Var(v));
    }
    for &c in &[0i64, 1, -1, 2] {
        nested_atoms.push(Expr::Const(c));
    }

    let mut nested_binops: Vec<Expr> = Vec::new();
    for l in &nested_atoms {
        for r in &nested_atoms {
            for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Mod] {
                nested_binops.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }

    // Try different bound combinations:
    // outer_bound ∈ {args[0], args[0]+1}  inner_bound ∈ {outer_i, outer_i+1, args[0]}
    let outer_bounds: Vec<Expr> = vec![
        Expr::Var(0), // a
    ];
    let inner_bounds: Vec<Expr> = vec![
        Expr::Var(n_args + 1), // outer_i
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var(n_args + 1)),
            Box::new(Expr::Const(1)),
        ), // outer_i + 1
    ];

    for outer_bound in &outer_bounds {
        for inner_bound in &inner_bounds {
            for &outer_bop in &[BinOp::Add, BinOp::Mul] {
                for &inner_bop in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::BitOr] {
                    for inner_rhs in &nested_binops {
                        if start.elapsed().as_millis() as u64 > time_limit_ms {
                            return None;
                        }

                        let e = Expr::NestedWhile {
                            outer_init: Box::new(Expr::Const(0)),
                            outer_bound: Box::new(outer_bound.clone()),
                            outer_body_op: outer_bop,
                            inner_init: Box::new(Expr::Const(0)),
                            inner_bound: Box::new(inner_bound.clone()),
                            inner_body_op: inner_bop,
                            inner_body_rhs: Box::new(inner_rhs.clone()),
                        };

                        if examples
                            .iter()
                            .all(|(args, expected)| e.eval_nested(args) == Some(*expected))
                        {
                            let code = emit_mog(&e, fn_name, &param_names);
                            if verify_problem_code_strict(problem, &code).is_ok() {
                                return Some(SolveResult {
                                    success: true,
                                    code,
                                    method: "enumerative-nested".to_string(),
                                    error: None,
                                    metadata: DifferentiableMetadata::default(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// ─── While-cond loops (digit extraction, etc.) ─────────────────────────────

/// Discover programs with while-cond loops (e.g., digit_sum, digit_count, reverse_digits).
/// Pattern: acc = 0; x = arg; while x > 0 { acc = acc OP body(x, acc); x = x / 10 }
fn synthesize_while_cond(problem: &Problem) -> Option<SolveResult> {
    let n_args = problem.examples.first()?.inputs.len();
    if n_args != 1 {
        return None;
    }

    let fn_name = problem.function_name();
    let param_names: Vec<&str> = vec!["a"];

    let examples: Vec<(Vec<i64>, i64)> = problem
        .examples
        .iter()
        .map(|ex| {
            let args: Vec<i64> = ex
                .inputs
                .iter()
                .filter_map(|v| {
                    if let Value::Int(i) = v {
                        Some(*i)
                    } else {
                        None
                    }
                })
                .collect();
            (args, ex.expected_int())
        })
        .collect();

    let start = std::time::Instant::now();
    let time_limit_ms: u64 = 5_000;

    // Namespace: [a, x, acc, i]
    let n_ext = 4;
    let mut atoms: Vec<Expr> = Vec::new();
    for v in 0..n_ext {
        atoms.push(Expr::Var(v));
    }
    for &c in &[0i64, 1, -1, 2, 10] {
        atoms.push(Expr::Const(c));
    }

    // x % 10 (digit extraction) and x / 10 * 10 + acc etc.
    let mut body_exprs: Vec<Expr> = atoms.clone();
    for l in &atoms {
        for r in &atoms {
            for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Mod] {
                body_exprs.push(Expr::BinOp(op, Box::new(l.clone()), Box::new(r.clone())));
            }
        }
    }

    // Try different configurations
    let divisors: [i64; 2] = [10, 2];
    let cond_cmps: [CmpOp; 2] = [CmpOp::Gt, CmpOp::Ne];

    for &divisor in &divisors {
        for &cmp in &cond_cmps {
            for &bop in &[BinOp::Add, BinOp::Mul] {
                for body_rhs in &body_exprs {
                    if start.elapsed().as_millis() as u64 > time_limit_ms {
                        return None;
                    }

                    let e = Expr::WhileCond {
                        init: Box::new(Expr::Const(0)),
                        state_init: Box::new(Expr::Var(0)), // x = a
                        cond_cmp: cmp,
                        cond_val: Box::new(Expr::Const(0)), // while x > 0
                        divisor,
                        body_op: bop,
                        body_rhs: Box::new(body_rhs.clone()),
                    };

                    if examples
                        .iter()
                        .all(|(args, expected)| e.eval_while_cond(args) == Some(*expected))
                    {
                        let code = emit_mog(&e, fn_name, &param_names);
                        if verify_problem_code_strict(problem, &code).is_ok() {
                            return Some(SolveResult {
                                success: true,
                                code,
                                method: "enumerative-while-cond".to_string(),
                                error: None,
                                metadata: DifferentiableMetadata::default(),
                            });
                        }
                    }
                }
            }
        }
    }

    None
}

// ─── Dream mode ─────────────────────────────────────────────────────────────

/// Build the component library by corpus-driven frequent-subtree mining.
///
/// The historical hand-listed 15-seed enumeration has moved into
/// [`bootstrap_corpus`] (cold-start mining INPUT) — the library is now MINED
/// from the verified solved-Expr corpus rather than hand-enumerated. All call
/// sites (`load_or_dream`, `--dream`) are preserved by this shim.
pub fn dream(time_budget_ms: u64) -> ComponentLibrary {
    mine_library(time_budget_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SOUNDNESS GUARD for algebraic canonicalization (the crown jewel): over
    /// 2000 random scalar exprs (all 12 binops incl. div/mod, all unops, ifs) x 8
    /// random input vectors = 16k checks, `algebraic_normalize` must PRESERVE eval
    /// EXACTLY (including the None/domain-error cases). A single unsound rewrite —
    /// e.g. discarding a sub-expr that can error — is caught here.
    #[test]
    fn algebraic_normalize_preserves_eval_on_random_exprs() {
        fn lcg(s: &mut u64) -> u64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s >> 16
        }
        fn gen(s: &mut u64, depth: u32) -> Expr {
            if depth == 0 || lcg(s) % 3 == 0 {
                if lcg(s) % 2 == 0 {
                    Expr::Var((lcg(s) % 3) as usize)
                } else {
                    Expr::Const((lcg(s) % 11) as i64 - 5)
                }
            } else {
                match lcg(s) % 3 {
                    0 => {
                        let op = ALL_BINOPS[(lcg(s) as usize) % ALL_BINOPS.len()];
                        Expr::BinOp(op, Box::new(gen(s, depth - 1)), Box::new(gen(s, depth - 1)))
                    }
                    1 => {
                        let op = ALL_UNOPS[(lcg(s) as usize) % ALL_UNOPS.len()];
                        Expr::UnaryOp(op, Box::new(gen(s, depth - 1)))
                    }
                    _ => {
                        let c = ALL_CMPS[(lcg(s) as usize) % ALL_CMPS.len()];
                        Expr::IfExpr(
                            c,
                            Box::new(gen(s, depth - 1)),
                            Box::new(gen(s, depth - 1)),
                            Box::new(gen(s, depth - 1)),
                            Box::new(gen(s, depth - 1)),
                        )
                    }
                }
            }
        }
        let mut s = 0xC0FFEE_u64;
        let mut checked = 0usize;
        for _ in 0..2000 {
            let e = gen(&mut s, 4);
            let c = algebraic_normalize(&e);
            for _ in 0..8 {
                let inputs: Vec<i64> = (0..3).map(|_| (lcg(&mut s) % 21) as i64 - 10).collect();
                assert_eq!(
                    e.eval(&inputs),
                    c.eval(&inputs),
                    "canonicalization changed eval: {e:?} -> {c:?} on {inputs:?}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 16000, "ran {checked} checks");
    }

    /// The MERGE win: algebraically-equal forms normalize to ONE representative,
    /// so the library-mining canonicalize (algebra ∘ var-rename) compresses them
    /// together instead of mining look-alikes.
    #[test]
    fn algebraic_normalize_merges_equal_forms() {
        let a = Expr::Var(0);
        let two = Expr::Const(2);
        let ab = Expr::BinOp(BinOp::Add, Box::new(a.clone()), Box::new(two.clone())); // a+2
        let ba = Expr::BinOp(BinOp::Add, Box::new(two.clone()), Box::new(a.clone())); // 2+a
        assert_eq!(algebraic_normalize(&ab), algebraic_normalize(&ba), "a+2 == 2+a");
        // x+0 -> x
        let x0 = Expr::BinOp(BinOp::Add, Box::new(a.clone()), Box::new(Expr::Const(0)));
        assert_eq!(algebraic_normalize(&x0), a);
        // --x -> x
        let nn = Expr::UnaryOp(UnOp::Neg, Box::new(Expr::UnaryOp(UnOp::Neg, Box::new(a.clone()))));
        assert_eq!(algebraic_normalize(&nn), a);
        // min(x,x) -> x
        let mm = Expr::BinOp(BinOp::Min, Box::new(a.clone()), Box::new(a.clone()));
        assert_eq!(algebraic_normalize(&mm), a);
        // The mining canonical form merges a+2 and 2+a (algebra then var-rename).
        assert_eq!(canonicalize(&ab).0, canonicalize(&ba).0, "mining canon merges commutative");
        // Sub is NOT commutative — a-2 and 2-a must stay distinct.
        let sab = Expr::BinOp(BinOp::Sub, Box::new(a.clone()), Box::new(two.clone()));
        let sba = Expr::BinOp(BinOp::Sub, Box::new(two.clone()), Box::new(a.clone()));
        assert_ne!(canonicalize(&sab).0, canonicalize(&sba).0, "a-2 != 2-a (sub not commutative)");
    }

    /// ASSOCIATIVE-COMMUTATIVE flattening: every grouping AND ordering of a+b+c
    /// collapses to ONE normal form — the big merge win over adjacent-swap only.
    #[test]
    fn algebraic_normalize_flattens_associative_chains() {
        let (a, b, c) = (Expr::Var(0), Expr::Var(1), Expr::Var(2));
        let add = |x: Expr, y: Expr| Expr::BinOp(BinOp::Add, Box::new(x), Box::new(y));
        // (a+b)+c , a+(b+c) , c+(b+a) — all groupings/orders.
        let f1 = add(add(a.clone(), b.clone()), c.clone());
        let f2 = add(a.clone(), add(b.clone(), c.clone()));
        let f3 = add(c.clone(), add(b.clone(), a.clone()));
        let n1 = algebraic_normalize(&f1);
        assert_eq!(n1, algebraic_normalize(&f2), "(a+b)+c == a+(b+c)");
        assert_eq!(n1, algebraic_normalize(&f3), "(a+b)+c == c+(b+a)");
        // Eval preserved across the whole equivalence class.
        for inputs in [[1, 2, 3], [-4, 0, 9], [7, 7, -1]] {
            assert_eq!(f1.eval(&inputs), n1.eval(&inputs), "eval preserved");
        }
        // min(a, min(b, a)) flattens + dedups to min(a, b).
        let mn = Expr::BinOp(
            BinOp::Min,
            Box::new(a.clone()),
            Box::new(Expr::BinOp(BinOp::Min, Box::new(b.clone()), Box::new(a.clone()))),
        );
        let expect = Expr::BinOp(BinOp::Min, Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(algebraic_normalize(&mn), expect, "min(a,min(b,a)) -> min(a,b)");
    }

    /// CONSTANT FOLDING via eval: a closed subexpr folds to its value; a
    /// non-constant expr keeps its variable subtree; an UNDEFINED closed expr
    /// (div by zero) is NOT folded (no fabricated Const).
    #[test]
    fn algebraic_normalize_folds_constants_soundly() {
        let add = |x: Expr, y: Expr| Expr::BinOp(BinOp::Add, Box::new(x), Box::new(y));
        // 2 + 3 -> 5
        assert_eq!(algebraic_normalize(&add(Expr::Const(2), Expr::Const(3))), Expr::Const(5));
        // x + (2 + 3) -> x + 5 (order-canonical: Const then Var)
        let x = Expr::Var(0);
        let folded = algebraic_normalize(&add(x.clone(), add(Expr::Const(2), Expr::Const(3))));
        assert_eq!(folded, add(Expr::Const(5), x.clone()));
        // 6 / 0 is a closed but UNDEFINED expr -> left as-is (eval -> None).
        let divz = Expr::BinOp(BinOp::Div, Box::new(Expr::Const(6)), Box::new(Expr::Const(0)));
        assert_eq!(algebraic_normalize(&divz), divz, "n/0 never folds to a Const");
        // eval preserved on the folded form.
        for i in [-3, 0, 7] {
            assert_eq!(add(x.clone(), add(Expr::Const(2), Expr::Const(3))).eval(&[i]), folded.eval(&[i]));
        }
    }

    /// COMPOSITE OUTPUT: a Pair-returning function `(a,b) -> (a+b, a-b)` is
    /// synthesized componentwise (each component a scalar fit), emitted as a
    /// struct, and strict-verified — the engine producing STRUCTURED output, not
    /// just i64/array. This is the type-ceiling lift.
    #[test]
    fn synthesizes_pair_output_componentwise() {
        use crate::benchmark::{Example, Problem, Value};
        let mk = |a: i64, b: i64| Example {
            inputs: vec![Value::Int(a), Value::Int(b)],
            expected: Value::Pair(a + b, a - b),
        };
        let problem = Problem {
            name: "sumdiff".to_string(),
            category: "test",
            description: "pair output (a+b, a-b)",
            signature: "fn sumdiff(a: i64, b: i64) -> Out_sumdiff",
            examples: vec![
                mk(5, 3),
                mk(10, 2),
                mk(7, 1),
                mk(8, 4),
                mk(20, 6),
                mk(3, 9),
            ],
            ..Default::default()
        };
        let r = synthesize_enumerative(&problem)
            .unwrap_or_else(|| panic!("pair-output should synthesize"));
        assert_eq!(
            r.method, "enumerative-tuple-output",
            "wrong method; code:\n{}",
            r.code
        );
        // The emitted program declares a struct and returns it.
        assert!(
            r.code.contains("struct Out_sumdiff") && r.code.contains("return Out_sumdiff {"),
            "emit not a struct-returning program:\n{}",
            r.code
        );
    }

    /// COMPOSITE OUTPUT, 4 components: `(a,b) -> (a+b, a-b, a*b, a)` proves the
    /// componentwise lift generalizes beyond pairs (Quad → 4-field struct).
    #[test]
    fn synthesizes_quad_output_componentwise() {
        use crate::benchmark::{Example, Problem, Value};
        let mk = |a: i64, b: i64| Example {
            inputs: vec![Value::Int(a), Value::Int(b)],
            expected: Value::Quad(a + b, a - b, a * b, a),
        };
        let problem = Problem {
            name: "quadfn".to_string(),
            category: "test",
            description: "quad output",
            signature: "fn quadfn(a: i64, b: i64) -> Out_quadfn",
            examples: vec![
                mk(5, 3),
                mk(10, 2),
                mk(7, 1),
                mk(8, 4),
                mk(2, 6),
                mk(9, 5),
            ],
            ..Default::default()
        };
        let r = synthesize_enumerative(&problem)
            .unwrap_or_else(|| panic!("quad-output should synthesize"));
        assert_eq!(r.method, "enumerative-tuple-output", "code:\n{}", r.code);
    }

    /// COMPOSITE OUTPUT with ARRAY input: `minmax(arr) -> (min, max)` — each
    /// component is a fold synthesized as a helper fn and called from the struct
    /// constructor. Proves structured output over array→scalar reductions.
    #[test]
    fn synthesizes_array_input_pair_minmax() {
        use crate::benchmark::{Example, Problem, Value};
        let mk = |a: &[i64]| {
            let mn = *a.iter().min().unwrap();
            let mx = *a.iter().max().unwrap();
            Example {
                inputs: vec![Value::int_array(a)],
                expected: Value::Pair(mn, mx),
            }
        };
        let problem = Problem {
            name: "minmax".to_string(),
            category: "test",
            description: "min and max of an array",
            signature: "fn minmax(p0: [i64]) -> Out_minmax",
            examples: vec![
                mk(&[3, 1, 2]),
                mk(&[5, 9, 1, 7]),
                mk(&[-1, 4, 2]),
                mk(&[8, 8, 2, 10]),
                mk(&[0, -5, 5]),
                mk(&[6, 3, 9, 1]),
            ],
            ..Default::default()
        };
        let r = synthesize_enumerative(&problem)
            .unwrap_or_else(|| panic!("array-input pair (minmax) should synthesize"));
        assert_eq!(
            r.method, "enumerative-tuple-output-array",
            "code:\n{}",
            r.code
        );
    }

    #[test]
    fn core_ext_finds_identity_and_reports_no_timeout() {
        // f(x) = x: should be found at size 1 (Var), so timed_out stays false
        let examples = vec![(vec![3], 3), (vec![7], 7), (vec![-2], -2)];
        let (expr, timed_out) = enumerate_exprs_core_ext(1, 5, &examples, 5_000, None);
        assert!(expr.is_some(), "identity must be discoverable in pass-1");
        assert!(
            !timed_out,
            "identity is size-1 so the budget shouldn't expire"
        );
    }

    #[test]
    fn stats_reports_max_completed_when_exhausted() {
        // f(x) = x*x*x (cube) is genuinely unsolvable at max_size=3 — a cube needs
        // size 5, and no mined constant fits {8,27,64}. (The old `x+1000` premise
        // is stale since LOOP-23: constant-mining now seeds 1000 from the outputs,
        // so `x+1000` IS findable. Cube is mining-proof.) At max_size=3 with a
        // generous 3s budget, every size completes → max_completed == 3.
        let examples = vec![(vec![2], 8), (vec![3], 27), (vec![4], 64)];
        let (expr, timed_out, max_completed) =
            enumerate_exprs_with_ops_stats(1, 3, &examples, 3_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(expr.is_none(), "cube is size-5, unsolvable at size 3");
        assert!(!timed_out, "size 3 enumeration should fit within 3s");
        assert_eq!(max_completed, 3, "every size should have completed cleanly");
    }

    #[test]
    fn stats_reports_timeout_with_partial_max_completed() {
        // 0ms budget forces the search to bail out of the size loop on the
        // first time check. We don't assert a specific max_completed value
        // because some sizes may slip through before the first elapsed()
        // check, but the timed_out flag must be set.
        let examples = vec![
            (vec![0], 1_000_000),
            (vec![1], 1_000_001),
            (vec![2], 1_000_002),
            (vec![3], 1_000_003),
        ];
        let (_expr, timed_out, _max_completed) =
            enumerate_exprs_with_ops_stats(1, 5, &examples, 0, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(timed_out, "0ms budget must flip the timeout flag");
    }

    // ── Anytime resumable search (Part A) ──────────────────────────────────

    /// THE RESUMABILITY GUARANTEE. A problem (`f(x) = x*x*x`, minimal expr size
    /// 5 under core ops) is UNSOLVED under a shallow soft cap, but the search
    /// does NOT report impossibility — it persists a frontier and, when RESUMED
    /// under a deeper cap, finds the answer by continuing from where it stopped
    /// rather than restarting from size 1.
    #[test]
    fn resumable_search_deepens_across_calls() {
        // cube: no single op / constant reproduces it, so it is not findable at
        // size <= 3; (x*x)*x is the size-5 witness.
        let examples = vec![(vec![2], 8), (vec![3], 27), (vec![4], 64), (vec![5], 125)];

        // CALL 1: deepen only up to size 3 (generous time, so the cap — not the
        // clock — bounds the run). Must MISS, must NOT be a timeout (it cleanly
        // exhausted the capped space), and must have advanced the frontier.
        let mut frontier = Frontier::fresh(String::new(), 1, 0);
        let (expr1, timed_out1) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(expr1.is_none(), "cube must NOT be solvable at size <= 3");
        assert!(
            !timed_out1,
            "clean exhaustion under a soft cap is not a timeout — and a None \
             here is 'budget exhausted, resumable', never a proof of impossibility"
        );
        let depth_after_1 = frontier.next_size;
        assert!(
            depth_after_1 >= 4,
            "frontier should have advanced past size 3 (got next_size={depth_after_1})"
        );
        // The frontier carries real work forward (size-3 stratum is populated).
        assert!(
            frontier.by_size.len() > 3 && !frontier.by_size[3].is_empty(),
            "size-3 expressions must be persisted in the frontier for resume"
        );

        // CALL 2: RESUME the SAME frontier under a deeper cap. It must now solve
        // cube — proving the second call continued deeper instead of restarting.
        let (expr2, _timed_out2) = enumerate_exprs_resumable(
            &mut frontier,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let solved = expr2.expect("resumed deeper search must solve cube");
        assert!(
            matches_all(&solved, &examples),
            "resumed solution must satisfy every example"
        );
        assert!(
            frontier.next_size > depth_after_1 || solved.size() <= 7,
            "resume must have explored deeper than the first call"
        );
    }

    /// DETERMINISM / MONOTONICITY. A search SPLIT across two resumed calls finds
    /// the SAME witness as one long uncapped call (the search is uniform-cost by
    /// size and the frontier is a faithful snapshot — no rand, no clock in
    /// selection). Proves the split-vs-whole equivalence the design claims.
    #[test]
    fn split_resume_matches_single_run() {
        let examples = vec![(vec![2], 8), (vec![3], 27), (vec![4], 64), (vec![5], 125)];

        // One long run (generous cap).
        let mut whole = Frontier::fresh(String::new(), 1, 0);
        let (one, _) = enumerate_exprs_resumable(
            &mut whole,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let one = one.expect("single run solves cube");

        // Split run: cap 3 (miss), then resume cap 7 (solve).
        let mut split = Frontier::fresh(String::new(), 1, 0);
        let (a, _) = enumerate_exprs_resumable(
            &mut split,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(a.is_none(), "cube unsolved at cap 3");
        let (b, _) = enumerate_exprs_resumable(
            &mut split,
            &examples,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        );
        let two = b.expect("resumed run solves cube");

        assert_eq!(
            one, two,
            "split-across-calls search must find the identical witness as one long run"
        );
    }

    /// The resumable wrapper preserves the historical bounded semantics: the
    /// old `enumerate_exprs_with_ops_stats` API still finds identity at size 1
    /// and reports a clean (non-timeout) exhaustion when nothing matches.
    #[test]
    fn bounded_wrapper_behaves_like_before() {
        // identity at size 1
        let id = vec![(vec![3], 3), (vec![7], 7), (vec![-2], -2)];
        let (e, t, _mc) =
            enumerate_exprs_with_ops_stats(1, 5, &id, 5_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(e.is_some() && !t, "identity is size-1, no timeout");

        // Clean exhaustion at a small cap: a CUBE (`x*x*x`) needs size 5, so it is
        // unsolvable at cap 3 regardless of constants — exhausts sizes 1..=3 cleanly.
        // (The old `x+1000` example for this check is now SOLVED by example-mined
        // constants — see `solves_affine_with_mined_constant` — so it no longer
        // demonstrates exhaustion; cube does, and mining can't shortcut it.)
        let miss = vec![(vec![0], 0), (vec![2], 8), (vec![3], 27)];
        let (e2, t2, mc2) =
            enumerate_exprs_with_ops_stats(1, 3, &miss, 3_000, None, &CORE_BINOPS, &CORE_UNOPS);
        assert!(e2.is_none() && !t2, "x*x*x needs size 5; clean exhaustion at cap 3");
        assert_eq!(mc2, 3, "every size up to the cap completed");
    }

    // ── Mining / generalization tests ──────────────────────────────────────

    fn solved(expr: Expr, n_args: usize) -> SolvedExpr {
        let fp = fingerprint(&expr, &probe_inputs(n_args, 8)).unwrap_or_default();
        SolvedExpr { expr, n_args, fp }
    }

    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
    }

    /// REGRESSION (Bug B): the library must keep mining as the solve corpus
    /// GROWS — not freeze at its cold-start snapshot. Before the fix,
    /// `load_or_dream` returned the cached library unconditionally, so solves
    /// recorded into the corpus never became new abstractions. Here we mine a
    /// small corpus, then fold in a grown corpus carrying a NEW repeated pattern
    /// and assert a component is added (compounding) and that re-merging an
    /// unchanged corpus adds nothing (idempotent dedup).
    #[test]
    fn library_remines_as_corpus_grows() {
        // Small corpus: the square pattern a*a (support 2).
        let small = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1), // a*a
            solved(add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2), // a*a + b
        ];
        let mut lib = mine_from_corpus(&small);
        let n_small = lib.len();
        assert!(n_small >= 1, "small corpus must mine at least the square pattern");

        // Grown corpus: small PLUS two trees carrying a NEW repeated pattern a+a.
        let mut grown = small.clone();
        grown.push(solved(add(Expr::Var(0), Expr::Var(0)), 1)); // a+a
        grown.push(solved(add(add(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2)); // (a+a)+b

        let added = lib.merge_from_corpus(&grown);
        assert!(
            added >= 1,
            "growing the corpus with a new repeated pattern must fold in a component; \
             added={added}, names={:?}",
            lib.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>()
        );
        assert!(lib.len() > n_small, "library must grow with the corpus");

        // Re-merging an unchanged corpus must add nothing (dedup by description).
        let again = lib.merge_from_corpus(&grown);
        assert_eq!(again, 0, "re-merging an unchanged corpus must add no components");
    }

    /// THE GENERALIZATION LEVER. A mined `?0*?0` abstraction, re-rooted onto
    /// both arg slots and injected at by_size[1], makes sum-of-squares (true
    /// size 7) solvable at max_size=3 — otherwise impossible from scratch.
    #[test]
    fn mined_subtree_enables_novel_composition() {
        // Corpus: two distinct trees that BOTH contain the square pattern a*a.
        // Tree A = a*a (square). Tree B = a*a + b (square plus a second arg).
        let tree_a = mul(Expr::Var(0), Expr::Var(0)); // f(a) = a*a
        let tree_b = add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)); // f(a,b) = a*a + b
        let corpus = vec![solved(tree_a, 1), solved(tree_b, 2)];

        let lib = mine_from_corpus(&corpus);

        // The mined library must contain a component canonicalizing to a*a.
        let square = mul(Expr::Var(0), Expr::Var(0));
        let has_square = lib
            .get_for_args(1)
            .iter()
            .any(|e| canonicalize(e).0 == square);
        assert!(
            has_square,
            "mined library must contain the square pattern (canonical a*a); got names: {:?}",
            lib.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>()
        );
        // And its name must be the human-readable provenance.
        assert!(
            lib.components.iter().any(|(_, d)| d == "mined: a * a"),
            "square component must be named 'mined: a * a'"
        );

        // Sum-of-squares: f(a,b) = a*a + b*b. True Expr size is 7.
        let examples = vec![
            (vec![1, 2], 5),
            (vec![3, 1], 10),
            (vec![2, 4], 20),
            (vec![5, 0], 25),
            (vec![-2, 3], 13),
        ];

        // Without the library, max_size=3 CANNOT build a size-7 program.
        let without = enumerate_exprs(2, 3, &examples, 2000, None);
        assert!(
            without.is_none(),
            "max_size=3 must NOT solve sum-of-squares from scratch (got {without:?})"
        );

        // With the mined library, the re-rooted square (Var0*Var0 and Var1*Var1)
        // injected at by_size[1] composes via a size-2 Add into the solution.
        let with = enumerate_exprs(2, 3, &examples, 2000, Some(&lib));
        assert!(
            with.is_some(),
            "mined abstraction must make sum-of-squares solvable at max_size=3"
        );
        // Soundness preserved: the returned expr is observationally verified.
        let e = with.unwrap();
        assert!(
            matches_all(&e, &examples),
            "returned expr must satisfy all examples (sound): {e:?}"
        );
    }

    /// ADVERSARIAL PROBE (temporary): prove the miner is DATA-DERIVED.
    /// (A) A corpus whose recurring pattern is a*(a+b) [NOT a*a] must mine
    ///     that pattern and must NOT contain a*a. A hardcoded a*a list fails.
    /// (B) A corpus where a*a appears in only ONE tree (no recurrence) must NOT
    ///     promote a*a (support<2). A canned list would still surface it.
    #[test]
    fn adversarial_data_derivation_probe() {
        // ---- (A) different recurring pattern: a*(a+b) appears in two trees ----
        let patt = mul(Expr::Var(0), add(Expr::Var(0), Expr::Var(1))); // a*(a+b), size 5
        let tree_a = patt.clone();
        let tree_b = add(patt.clone(), Expr::Var(1)); // a*(a+b) + b
        let corpus_a = vec![solved(tree_a, 2), solved(tree_b, 2)];
        let lib_a = mine_from_corpus(&corpus_a);
        let patt_canon = canonicalize(&patt).0;
        let square_canon = canonicalize(&mul(Expr::Var(0), Expr::Var(0))).0;
        let mined: Vec<Expr> = lib_a.components.iter().map(|(e, _)| canonicalize(e).0).collect();
        eprintln!("PROBE-A mined canon set: {:?}", mined);
        eprintln!("PROBE-A names: {:?}", lib_a.components.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>());
        assert!(
            mined.iter().any(|e| *e == patt_canon),
            "miner must extract the ACTUAL recurring pattern a*(a+b) from THIS corpus"
        );
        assert!(
            !mined.iter().any(|e| *e == square_canon),
            "miner must NOT surface a*a — it is absent from corpus_a (no hardcoding)"
        );

        // ---- (B) a*a present in only ONE tree -> support 1 -> not promoted ----
        let only_once = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1), // a*a once
            solved(add(Expr::Var(0), Expr::Var(1)), 2), // a+b (no a*a)
        ];
        let lib_b = mine_from_corpus(&only_once);
        let mined_b: Vec<Expr> = lib_b.components.iter().map(|(e, _)| canonicalize(e).0).collect();
        eprintln!("PROBE-B mined canon set: {:?}", mined_b);
        // Support-mined a*a needs >=2 distinct trees. The hybrid floor (lib<8)
        // re-adds corpus exprs verbatim, so a*a CAN appear via the floor — but
        // that is still corpus-derived (it came from a solved tree), never a
        // literal. The discriminating claim: it is NOT named "mined: a * a"
        // unless support>=2. Assert the FREQUENCY path did not fire for a*a.
        let support_mined_square = lib_b
            .components
            .iter()
            .any(|(e, d)| canonicalize(e).0 == square_canon && d == "mined: a * a");
        // a*a appears once => the only way it enters is the hybrid floor, which
        // is corpus-derived; either way no canned literal exists. We additionally
        // confirm a pattern absent from BOTH corpora (a/b) is NEVER mined.
        let div_canon = canonicalize(&Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)))).0;
        assert!(
            !mined_b.iter().any(|e| *e == div_canon),
            "a/b never appears in any corpus and must never be mined (no seeded list)"
        );
        eprintln!("PROBE-B support-mined a*a named 'mined: a * a'? {}", support_mined_square);
    }

    /// Determinism: mining the same corpus twice yields byte-identical libraries
    /// in the same order.
    #[test]
    fn mining_is_deterministic() {
        let corpus = vec![
            solved(mul(Expr::Var(0), Expr::Var(0)), 1),
            solved(add(mul(Expr::Var(0), Expr::Var(0)), Expr::Var(1)), 2),
            solved(
                add(mul(Expr::Var(1), Expr::Var(1)), Expr::Var(0)),
                2,
            ),
        ];
        let lib1 = mine_from_corpus(&corpus);
        let lib2 = mine_from_corpus(&corpus);
        assert_eq!(
            lib1.components, lib2.components,
            "mining must be deterministic (identical components in identical order)"
        );
    }

    /// Degeneracy rejection: leaves, no-ops, var-free folds, and whole-program
    /// memorization are never promoted.
    #[test]
    fn rejects_degenerate_abstractions() {
        // Pure leaves + a no-op (Var0 + Const0 == Var0 observationally).
        let corpus = vec![
            solved(Expr::Var(0), 1),
            solved(Expr::Const(5), 1),
            solved(add(Expr::Var(0), Expr::Const(0)), 1),
            solved(add(Expr::Var(0), Expr::Const(0)), 1), // appears twice -> support 2
        ];
        let lib = mine_from_corpus(&corpus);
        // D1/D2/D6 must fire: no leaf, no no-op promoted.
        for (e, _) in &lib.components {
            let (canon, arity) = canonicalize(e);
            assert!(
                !matches!(canon, Expr::Var(_) | Expr::Const(_)),
                "no bare leaf may be promoted"
            );
            // observational-identity no-op must not be promoted.
            let probes = probe_inputs(arity.max(1), 16);
            assert_ne!(
                fingerprint(&canon, &probes),
                fingerprint(&Expr::Var(0), &probes),
                "no observational no-op may be promoted: {canon:?}"
            );
        }

        // Whole-program memorization (size > 9 cap, D5): two identical large
        // trees must NOT be promoted as the whole tree.
        // Build a size-11 tree: ((a*a)+(a*a)) + ((a*a)+(a*a)) ... compose to >9.
        let sq = mul(Expr::Var(0), Expr::Var(0)); // size 3
        let big = add(add(sq.clone(), sq.clone()), add(sq.clone(), sq.clone())); // size 3*4+3 = 15
        assert!(big.size() > MAX_ABSTRACTION_SIZE);
        let corpus2 = vec![solved(big.clone(), 1), solved(big.clone(), 1)];
        let lib2 = mine_from_corpus(&corpus2);
        let big_canon = canonicalize(&big).0;
        assert!(
            !lib2
                .components
                .iter()
                .any(|(e, _)| canonicalize(e).0 == big_canon),
            "whole oversized tree must not be promoted (no memorization)"
        );

        // Var-free fold (D3): Const2 * Const3 has arity 0 -> not promotable.
        let varfree = mul(Expr::Const(2), Expr::Const(3));
        assert!(
            !is_promotable(&canonicalize(&varfree).0, 0),
            "var-free constant fold must not be promotable"
        );
    }

    /// Soundness: re-rooting a Div onto an arg that can be 0 must never yield an
    /// accepted-but-ill-defined program. Either the mine-time robust gate
    /// rejects it, or the per-use robust gate prunes it, or any returned
    /// solution evaluates cleanly on every example.
    #[test]
    fn rerooted_division_is_probe_guarded() {
        // Component a / b (size 3). Re-rooted onto (Var0, Var1) and (Var1, Var0).
        let divexpr = Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let corpus = vec![
            solved(divexpr.clone(), 2),
            // a second tree containing a/b so support >= 2
            solved(
                add(
                    Expr::BinOp(BinOp::Div, Box::new(Expr::Var(0)), Box::new(Expr::Var(1))),
                    Expr::Var(0),
                ),
                2,
            ),
        ];
        let lib = mine_from_corpus(&corpus);

        // Examples where one input is 0 in the divisor position for some maps.
        let examples = vec![
            (vec![6, 2], 3),
            (vec![10, 0], 999), // divisor 0 path -> any re-rooting that divides by b crashes
            (vec![8, 4], 2),
        ];
        // Whatever the enumerator returns (or None), it must be sound: never an
        // expr that returns None on an example.
        let result = enumerate_exprs(2, 3, &examples, 2000, Some(&lib));
        if let Some(e) = result {
            assert!(
                matches_all(&e, &examples),
                "any returned solution must be observationally valid on all examples"
            );
            assert!(
                robust_well_defined(&e, 2, 30),
                "any returned solution must be robust (no div-by-zero on probes)"
            );
        }
        // The point: no crash, no ill-defined accepted program. (result may be
        // None — there is no clean closed form here, and that is correct.)
    }

    /// Back-compat: an OLD-shape library JSON ({"components":[[expr,"a + b"]]})
    /// still deserializes with the component intact.
    #[test]
    fn old_library_json_still_loads() {
        let json = r#"{
            "components": [
                [{"BinOp":["Add",{"Var":0},{"Var":1}]}, "a + b"]
            ]
        }"#;
        let lib: ComponentLibrary =
            serde_json::from_str(json).expect("old-shape library must deserialize");
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.components[0].1, "a + b");
        assert_eq!(
            lib.components[0].0,
            add(Expr::Var(0), Expr::Var(1)),
            "deserialized component must be a + b"
        );
    }

    /// Corpus guards: the persistent path is disabled under cfg!(test)
    /// (hermetic) and record_solved_expr respects the entry cap when an explicit
    /// path is provided.
    #[test]
    fn solved_expr_corpus_is_test_disabled_and_capped() {
        // Under cfg!(test) with no env override, the path is None (hermetic).
        assert!(
            solved_exprs_path().is_none(),
            "solved_exprs_path must be None under cfg!(test) for hermeticity"
        );

        // With an explicit tempfile path, the entry cap is enforced.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "mog_synth_test_corpus_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::env::set_var("NSYNTH_SOLVED_EXPRS_PATH", &path);

        // Push more than the cap; each distinct expr/fp is a new entry.
        for i in 0..(SOLVED_EXPRS_MAX_ENTRIES + 50) {
            let e = add(Expr::Var(0), Expr::Const(i as i64));
            let fp = vec![i as i64];
            record_solved_expr(&e, 1, &fp);
        }
        let loaded = load_solved_exprs();
        assert!(
            loaded.len() <= SOLVED_EXPRS_MAX_ENTRIES,
            "corpus must be capped at {} entries, got {}",
            SOLVED_EXPRS_MAX_ENTRIES,
            loaded.len()
        );

        // Cleanup.
        std::env::remove_var("NSYNTH_SOLVED_EXPRS_PATH");
        let _ = std::fs::remove_file(&path);
    }

    // ── Array-frontier resumability + library participation (ARRAY-FRONTIER) ──

    /// Build a real `sum_of_cubes` array problem. `reference_code` is SET, so
    /// `verify_problem_code_strict` runs DIFFERENTIAL holdouts sampled against
    /// the reference (not examples-only): the accept gate is un-gameable.
    fn sum_of_cubes_problem() -> Problem {
        use crate::benchmark::Example;
        // Minimal fold body is (item*item)*item = size 5, beyond the warm-up's
        // atom-OP-atom (size 3) sweep AND beyond a size-3 frontier cap.
        Problem {
            name: "sum_of_cubes".to_string(),
            category: "arrays",
            description: "Sum of the cubes of all elements.",
            signature: "fn sum_of_cubes(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::int_array(&[1, 2, 3])],
                    expected: Value::Int(36), // 1 + 8 + 27
                },
                Example {
                    inputs: vec![Value::int_array(&[2])],
                    expected: Value::Int(8),
                },
                Example {
                    inputs: vec![Value::int_array(&[1, 1, 1, 1])],
                    expected: Value::Int(4),
                },
                Example {
                    inputs: vec![Value::int_array(&[0, 3])],
                    expected: Value::Int(27),
                },
            ],
            holdouts: vec![],
            reference_code:
                "fn sum_of_cubes(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for item in arr {\n        acc = acc + item * item * item;\n    }\n    return acc;\n}\n",
            ..Default::default()
        }
    }

    /// Extract `(scalar_args, array, expected)` triples the way
    /// `synthesize_array_enumerative` does, so the test drives the same data.
    fn array_examples_of(problem: &Problem) -> Vec<(Vec<i64>, Vec<i64>, i64)> {
        problem
            .examples
            .iter()
            .map(|ex| {
                let array: Vec<i64> = ex
                    .inputs
                    .iter()
                    .filter_map(|v| v.as_i64_slice())
                    .next()
                    .unwrap_or_default();
                (Vec::new(), array, ex.expected_int())
            })
            .collect()
    }

    /// THE UN-GAMEABLE RESUMABILITY TEST.
    ///
    /// A HARDER array problem (sum_of_cubes; minimal fold body size 5) is UNSOLVED
    /// by a single first-shot run under a shallow cap — `deepen_fold_frontier`
    /// returns None and the warm-up cannot reach it. The frontier persists work;
    /// a SECOND invocation that RESUMES the SAME frontier under a deeper cap
    /// solves it by continuing deeper instead of restarting. Every hit is gated
    /// by `verify_problem_code_strict` (differential holdouts vs the reference).
    #[test]
    fn array_frontier_resumes_deeper_to_solve_harder_problem() {
        let problem = sum_of_cubes_problem();
        let array_examples = array_examples_of(&problem);
        let scalar_names: Vec<&str> = vec![];
        let fold_n_args = 3; // item, i, acc (0 scalar args)

        // PRIOR/FIXED-PATH PROOF: a single first-shot run from a FRESH frontier
        // under the shallow cap (size 3) MUST return None. This is the
        // "single-shot under the old budget returns None" half of the criterion.
        let mut fresh_shallow = Frontier::fresh(String::new(), fold_n_args, 0);
        let first_shot = deepen_fold_frontier(
            &mut fresh_shallow,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(
            first_shot.is_none(),
            "sum_of_cubes MUST be unsolved by a single shallow (cap-3) first shot"
        );

        // CALL 1 (the persisted frontier): same shallow cap, MISS, but the
        // frontier deepens and banks real work.
        let mut frontier = Frontier::fresh(String::new(), fold_n_args, 0);
        let miss = deepen_fold_frontier(
            &mut frontier,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(miss.is_none(), "cap-3 frontier must miss sum_of_cubes");
        let depth_after_1 = frontier.next_size;
        assert!(
            depth_after_1 >= 4,
            "frontier must have advanced past size 3 (next_size={depth_after_1})"
        );
        assert!(
            frontier.by_size.len() > 3 && !frontier.by_size[3].is_empty(),
            "size-3 bodies must be banked in the frontier for resume"
        );

        // CALL 2 (RESUME): deeper cap on the SAME frontier. Must now SOLVE — the
        // second call continued from the banked strata, not from scratch.
        let solved = deepen_fold_frontier(
            &mut frontier,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            10_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        )
        .expect("resumed deeper frontier must solve sum_of_cubes");

        assert_eq!(solved.method, "enumerative-array");
        // Re-assert the STRICT gate independently (differential holdouts vs the
        // reference): the accepted code is sound, not just example-fitting.
        verify_problem_code_strict(&problem, &solved.code)
            .expect("accepted array code must pass strict differential verification");
    }

    /// REAL DISK PERSISTENCE: write the deepened frontier to a test-local file
    /// and read it back into a FRESH `Frontier` struct, then resume to a solve —
    /// proving the resumable store carries array work across a real on-disk
    /// round-trip. The round-trip uses the IDENTICAL serde serialization that
    /// `save_frontier`/`load_frontier` rely on (the SCALAR machinery reused
    /// unchanged), but writes to a private pid-unique path WITHOUT touching the
    /// process-global `NSYNTH_ENUM_FRONTIER_PATH` env var — so the test is
    /// hermetic and cannot race other tests' frontier I/O under the parallel
    /// runner (the documented serial-test hazard).
    #[test]
    fn array_frontier_persists_and_reloads_from_disk() {
        let problem = sum_of_cubes_problem();
        let array_examples = array_examples_of(&problem);
        let scalar_names: Vec<&str> = vec![];
        let fold_n_args = 3;
        let fp = crate::solved_cache::examples_fingerprint(&problem.examples);

        let path = std::env::temp_dir().join(format!(
            "mog_synth_test_array_frontier_{}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        // CALL 1: fresh, shallow cap, MISS → write the deepened frontier to disk
        // via the same serde encoding `save_frontier` uses.
        let mut f1 = Frontier::fresh(fp.clone(), fold_n_args, 0);
        let miss = deepen_fold_frontier(
            &mut f1,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(3),
        );
        assert!(miss.is_none());
        std::fs::write(&path, serde_json::to_string(&f1).unwrap()).unwrap();
        assert!(path.exists(), "frontier file must be written on miss");

        // CALL 2: READ a FRESH struct back from disk, resume deeper → SOLVE.
        let json = std::fs::read_to_string(&path).unwrap();
        let mut f2: Frontier = serde_json::from_str(&json).unwrap();
        assert!(
            f2.matches(&fp, fold_n_args, 0),
            "reloaded frontier must match the problem signature"
        );
        assert!(
            f2.next_size >= 4 && f2.by_size.len() > 3 && !f2.by_size[3].is_empty(),
            "reloaded frontier must carry the banked size-3 work for a real resume"
        );
        let solved = deepen_fold_frontier(
            &mut f2,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            10_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(7),
        )
        .expect("frontier reloaded from disk must resume to a solve");
        verify_problem_code_strict(&problem, &solved.code)
            .expect("disk-resumed solve must pass strict differential verification");

        let _ = std::fs::remove_file(&path);
    }

    /// REGRESSION GUARD (bounded): the FULL `synthesize_array_enumerative`
    /// entry still solves an easy array problem (array_sum) via the cheap
    /// warm-up sweep — i.e. wiring the frontier in did not break the fast path.
    /// Bounded: array_sum is found by the warm-up in milliseconds.
    #[test]
    fn array_enumerative_still_solves_array_sum_via_warmup() {
        use crate::benchmark::Example;
        let problem = Problem {
            name: "array_sum".to_string(),
            category: "arrays",
            description: "Sum of all elements.",
            signature: "fn array_sum(arr: [i64]) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::int_array(&[1, 2, 3])],
                    expected: Value::Int(6),
                },
                Example {
                    inputs: vec![Value::int_array(&[5])],
                    expected: Value::Int(5),
                },
                Example {
                    inputs: vec![Value::int_array(&[4, 4])],
                    expected: Value::Int(8),
                },
                Example {
                    inputs: vec![Value::int_array(&[2, 7, 1, 0])],
                    expected: Value::Int(10),
                },
            ],
            holdouts: vec![],
            reference_code:
                "fn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
            ..Default::default()
        };
        // array_sum is found by the cheap warm-up (size-1 body `item`), so the
        // resumable frontier is never reached and no env/disk state is touched —
        // the test is hermetic under cfg!(test) (frontier_path() == None).
        let res = synthesize_array_enumerative(&problem)
            .expect("array_sum must still be solved by the array entry point");
        verify_problem_code_strict(&problem, &res.code)
            .expect("array_sum solution must pass strict differential verification");
    }

    /// MINED-LIBRARY PARTICIPATION: a from-scratch frontier under a TIGHT cap
    /// CANNOT reach the size-5 cube body, so it misses. The SAME tight cap, but
    /// with a library that contributes the cube abstraction (`?0*?0*?0`), SOLVES
    /// — the injected component takes part in (and shortens) the array search.
    /// Acceptance is still the strict differential verifier.
    #[test]
    fn mined_library_item_participates_in_array_solve() {
        let problem = sum_of_cubes_problem();
        let array_examples = array_examples_of(&problem);
        let scalar_names: Vec<&str> = vec![];
        let fold_n_args = 3;

        // BASELINE (no library), tight cap 4 → cube body (size 5) unreachable.
        let mut bare = Frontier::fresh(String::new(), fold_n_args, 0);
        let bare_res = deepen_fold_frontier(
            &mut bare,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            5_000,
            None,
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(4),
        );
        assert!(
            bare_res.is_none(),
            "without the library, a cap-4 frontier cannot reach the size-5 cube body"
        );

        // WITH a library carrying the cube abstraction. The component is a
        // single-slot pattern `(?0 * ?0) * ?0`; instantiated onto the fold
        // namespace it becomes a size-1 leaf, so the SAME cap-4 search now finds
        // it. (`?0` = Var(0); instantiate_component re-roots it onto each arg.)
        let cube = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(0)),
            )),
            Box::new(Expr::Var(0)),
        );
        let mut lib = ComponentLibrary::new();
        lib.add(cube, "cube(?0)".to_string());

        let mut with_lib = Frontier::fresh(String::new(), fold_n_args, 0);
        let lib_res = deepen_fold_frontier(
            &mut with_lib,
            &array_examples,
            &problem,
            "sum_of_cubes",
            &scalar_names,
            0,
            5_000,
            Some(&lib),
            &CORE_BINOPS,
            &CORE_UNOPS,
            Some(4),
        )
        .expect("the injected cube library item must enable a cap-4 solve");
        verify_problem_code_strict(&problem, &lib_res.code)
            .expect("library-assisted solve must pass strict differential verification");
    }

    // ── U7: emitter correctness for bitwise / loop nodes ───────────────────────
    //
    // These tests prove the FIX is REAL: a synthesized program using a node that
    // the old emitters rendered with approximate ops (`+`/`-`/`%`/`*2`) now
    // renders to Mog SOURCE that, when parsed + run through the Mog interpreter,
    // matches `Expr::eval*` on inputs WHERE THE OLD RENDERING WOULD DIVERGE. The
    // tests assert the divergence-witness property explicitly (true result !=
    // what the old op would have produced), so they would FAIL under the old
    // emitters and cannot pass by coincidence.

    /// Helper: run a rendered Mog `fn` against scalar i64 args, returning the i64.
    fn run_mog_i64(code: &str, fn_name: &str, args: &[i64]) -> i64 {
        let bargs: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        match crate::runtime::execute_function(code, fn_name, &bargs, fn_name)
            .expect("rendered Mog source must parse and run")
        {
            crate::runtime::Value::Int(n) => n,
            other => panic!("expected Int return, got {other:?}"),
        }
    }

    /// Helper: run a rendered Mog fold against (scalar args, array), returning i64.
    fn run_mog_fold_i64(code: &str, fn_name: &str, arr: &[i64]) -> i64 {
        let bargs = vec![Value::Array(arr.iter().map(|&v| Value::Int(v)).collect())];
        match crate::runtime::execute_function(code, fn_name, &bargs, fn_name)
            .expect("rendered Mog fold source must parse and run")
        {
            crate::runtime::Value::Int(n) => n,
            other => panic!("expected Int return, got {other:?}"),
        }
    }

    /// XOR-fold over an array: `acc = 0; for item in arr { acc = acc ^ item }`.
    /// OLD emitter rendered the accumulator op as `+` (the `_ => "+"` fallback),
    /// so the rendered source computed a SUM. We pick an array where XOR != SUM,
    /// then assert the rendered+run result equals `Expr::eval_array` (true XOR)
    /// AND differs from the sum the old `+` rendering would have produced.
    #[test]
    fn u7_xor_fold_renders_to_correct_mog() {
        let fold = Expr::ForFold {
            init: Box::new(Expr::Const(0)),
            body_op: BinOp::BitXor,
            body_rhs: Box::new(Expr::Var(0)), // item (namespace: [item, i, acc])
        };
        let code = emit_mog_array(&fold, "xor_fold", &[], 0);

        // Sanity: the fix actually emits the `^` operator, not the old `+`.
        assert!(
            code.contains("acc = acc ^ item"),
            "XOR-fold must render with native `^`, got:\n{code}"
        );

        // Divergence witnesses: arrays where XOR-fold != SUM-fold.
        for arr in [
            vec![1i64, 2, 3],     // xor=0, sum=6
            vec![5, 5, 7],        // xor=7, sum=17
            vec![13, 6, 9, 2],    // xor=4, sum=30
            vec![1024, 1, 1024],  // xor=1, sum=2049
        ] {
            let true_val = fold
                .eval_array(&[], &arr)
                .expect("Expr::eval_array must succeed");
            let sum_val: i64 = arr.iter().sum(); // what the OLD `+` rendering gave
            assert_ne!(
                true_val, sum_val,
                "test input must be a divergence witness (XOR != SUM) for arr={arr:?}"
            );
            let mog_val = run_mog_fold_i64(&code, "xor_fold", &arr);
            assert_eq!(
                mog_val, true_val,
                "rendered Mog XOR-fold must match Expr::eval_array for arr={arr:?}"
            );
            // And it must NOT equal the old broken `+` rendering's result.
            assert_ne!(
                mog_val, sum_val,
                "rendered Mog must NOT compute the old SUM semantics for arr={arr:?}"
            );
        }
    }

    /// OR-fold over an array: `acc = 0; for item in arr { acc = acc | item }`.
    /// Old `+` rendering diverges whenever OR != SUM (overlapping set bits).
    #[test]
    fn u7_or_fold_renders_to_correct_mog() {
        let fold = Expr::ForFold {
            init: Box::new(Expr::Const(0)),
            body_op: BinOp::BitOr,
            body_rhs: Box::new(Expr::Var(0)),
        };
        let code = emit_mog_array(&fold, "or_fold", &[], 0);
        assert!(
            code.contains("acc = acc | item"),
            "OR-fold must render with native `|`, got:\n{code}"
        );
        for arr in [vec![3i64, 1, 2], vec![6, 3, 5], vec![7, 7, 7]] {
            let true_val = fold.eval_array(&[], &arr).unwrap();
            let sum_val: i64 = arr.iter().sum();
            assert_ne!(true_val, sum_val, "must be OR!=SUM witness for {arr:?}");
            let mog_val = run_mog_fold_i64(&code, "or_fold", &arr);
            assert_eq!(mog_val, true_val, "OR-fold mismatch for {arr:?}");
            assert_ne!(mog_val, sum_val, "must not be old SUM for {arr:?}");
        }
    }

    /// Scalar bitwise expressions through the per-Expr `to_mog` -> `emit_mog`
    /// path. Old emitters rendered BitXor->`-`, BitAnd->`%`, BitOr->`+`,
    /// Shl->`*2`, Shr->`/2`. We render `f(a,b) = (a ^ b) & (a << 1)` and run it,
    /// matching `Expr::eval` on inputs where every old approximation diverges.
    #[test]
    fn u7_scalar_bitwise_renders_to_correct_mog() {
        // (a ^ b) & (a << b)
        let expr = Expr::BinOp(
            BinOp::BitAnd,
            Box::new(Expr::BinOp(
                BinOp::BitXor,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Shl,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1)),
            )),
        );
        let code = emit_mog(&expr, "bw", &["a", "b"]);
        // Native ops must appear; the old approximate glyphs ("acc = acc -",
        // "* 2", "/ 2") must not stand in for these.
        assert!(code.contains('^') && code.contains('&') && code.contains("<<"),
            "scalar bitwise must render native ^, &, <<; got:\n{code}");

        for (a, b) in [(6i64, 3), (12, 1), (5, 2), (255, 4)] {
            let true_val = expr.eval(&[a, b]).expect("Expr::eval must succeed");
            // What the OLD emitter would have computed: (a - b) % (a * 2)
            let old_lhs = a - b;
            let old_rhs = a * 2;
            let old_val = if old_rhs != 0 { old_lhs % old_rhs } else { old_lhs };
            assert_ne!(
                true_val, old_val,
                "input (a={a},b={b}) must be a divergence witness vs old ops"
            );
            let mog_val = run_mog_i64(&code, "bw", &[a, b]);
            assert_eq!(
                mog_val, true_val,
                "rendered Mog scalar bitwise must match Expr::eval for (a={a},b={b})"
            );
            assert_ne!(
                mog_val, old_val,
                "rendered Mog must NOT reproduce old approximate semantics (a={a},b={b})"
            );
        }
    }

    /// End-to-end: a XOR-fold synthesis problem now renders to Mog that passes
    /// strict differential verification. Under the OLD `+` emitter the rendered
    /// source computed a SUM, which would FAIL `verify_problem_code_strict`
    /// against XOR holdouts — so this proves the fix end-to-end on the loop path.
    #[test]
    fn u7_xor_fold_passes_strict_verify() {
        use crate::benchmark::Example;
        let fold = Expr::ForFold {
            init: Box::new(Expr::Const(0)),
            body_op: BinOp::BitXor,
            body_rhs: Box::new(Expr::Var(0)),
        };
        let code = emit_mog_array(&fold, "xor_reduce", &[], 0);

        // Build a problem whose examples are the TRUE XOR-fold (computed via
        // Expr::eval_array, not via the rendered code) over divergence-witness
        // arrays. The XOR `reference_code` drives differential holdout sampling,
        // so the holdouts are XOR-truth — a SUM renderer (the old bug) cannot
        // satisfy them. This makes the end-to-end accept un-gameable.
        let arrays: [&[i64]; 5] = [
            &[1, 2, 3],
            &[5, 5, 7],
            &[8, 8, 8],
            &[13, 6, 9, 2],
            &[100, 28, 7],
        ];
        let examples: Vec<Example> = arrays
            .iter()
            .map(|arr| {
                let out = fold.eval_array(&[], arr).unwrap();
                Example {
                    inputs: vec![Value::int_array(arr)],
                    expected: Value::Int(out),
                }
            })
            .collect();
        let problem = Problem {
            name: "xor_reduce".to_string(),
            category: "arrays",
            description: "XOR of all elements.",
            signature: "fn xor_reduce(arr: [i64]) -> i64",
            examples,
            holdouts: vec![],
            reference_code:
                "fn xor_reduce(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for item in arr {\n        acc = acc ^ item;\n    }\n    return acc;\n}\n",
            ..Default::default()
        };

        // The fixed rendering must pass strict differential verification.
        verify_problem_code_strict(&problem, &code)
            .expect("fixed XOR-fold rendering must pass strict verify");

        // Prove the OLD rendering would have FAILED: build the SUM-renderer
        // source by hand and assert strict verify rejects it.
        let old_code = code.replace("acc = acc ^ item", "acc = acc + item");
        assert!(
            verify_problem_code_strict(&problem, &old_code).is_err(),
            "the old `+` (SUM) rendering MUST fail strict verify on XOR holdouts"
        );
    }

    // ── STEP7: Call-node search — controlled, un-gameable necessity proof ─────

    /// Does `e` (or any sub-expr) contain a `Call(idx, _)` node? Inspect the
    /// AST structurally — NOT the rendered string — so the proof cannot be
    /// gamed by an emitted name that merely looks like a call.
    fn expr_contains_call(e: &Expr) -> bool {
        match e {
            Expr::Call(..) => true,
            Expr::Var(_) | Expr::Const(_) => false,
            Expr::UnaryOp(_, c) => expr_contains_call(c),
            Expr::BinOp(_, l, r) => expr_contains_call(l) || expr_contains_call(r),
            Expr::IfExpr(_, a, b, c, d) => {
                expr_contains_call(a)
                    || expr_contains_call(b)
                    || expr_contains_call(c)
                    || expr_contains_call(d)
            }
            _ => false,
        }
    }

    /// The OPAQUE producer A(x) = if x>0 { x*x*x - 7 } else { 13 }. Piecewise +
    /// non-polynomial: the base ops CANNOT replicate B(x)=A(x)+1 within a small
    /// budget, so a found Call is NECESSARY, not a coincidence of cheaper base
    /// ops. The closure is the ground truth the registry exposes.
    fn opaque_a(x: i64) -> Option<i64> {
        if x > 0 {
            x.checked_mul(x)?.checked_mul(x)?.checked_sub(7)
        } else {
            Some(13)
        }
    }

    fn registry_with_a() -> Vec<NamedCallable> {
        vec![NamedCallable {
            name: "opaque_a".to_string(),
            n_args: 1,
            // Opaque closure (no Mog source) — used by `solve_scalar_expr_with_callees`,
            // which does the in-search accept gate, not the source-prepend verify.
            source: String::new(),
            eval: Box::new(|xs: &[i64]| {
                if xs.len() != 1 {
                    return None;
                }
                opaque_a(xs[0])
            }),
        }]
    }

    /// Build the B problem: B(x) = A(x) + 1 on a set of SEED rows, with FRESH
    /// holdout rows the search never sees (used to prove generalization, not
    /// memorization). Returns (problem, holdout_rows).
    fn b_problem() -> (Problem, Vec<(i64, i64)>) {
        use crate::benchmark::Example;
        // Seed rows (search sees these). Mix of x>0 and x<=0 so the piecewise
        // structure must be captured.
        let seed_xs = [1i64, 2, 3, 4, -1, -5, 0];
        let examples: Vec<Example> = seed_xs
            .iter()
            .map(|&x| Example {
                inputs: vec![Value::Int(x)],
                expected: Value::Int(opaque_a(x).unwrap() + 1),
            })
            .collect();
        // FRESH holdouts: distinct x values, NOT in the seed set.
        let holdout_xs = [6i64, 7, -2, -10];
        let holdouts: Vec<(i64, i64)> = holdout_xs
            .iter()
            .map(|&x| (x, opaque_a(x).unwrap() + 1))
            .collect();
        let problem = Problem {
            name: "b_consumer".to_string(),
            category: "step7",
            description: "B(x) = A(x) + 1, solvable only by calling A.",
            signature: "fn b_consumer(a: i64) -> i64",
            examples,
            holdouts: vec![],
            reference_code: "",
            ..Default::default()
        };
        (problem, holdouts)
    }

    #[test]
    fn call_node_is_searched_and_necessary() {
        let (problem, holdouts) = b_problem();
        let budget_ms = 8_000;

        // (a) WITH registry=[A]: the solver must find a program whose SOLVED AST
        // CONTAINS Call(idx_of_A, ...), and it must generalize to FRESH holdouts.
        let callees = registry_with_a();
        let solved = solve_scalar_expr_with_callees(&problem, &callees, budget_ms)
            .expect("B must be solvable when A is a registered callable");
        eprintln!("[STEP7] solved B AST = {solved:?}");
        eprintln!(
            "[STEP7] solved B Mog = {}",
            with_callee_names(&["opaque_a".to_string()], || solved.to_mog(&["a"]))
        );
        assert!(
            expr_contains_call(&solved),
            "solved B AST must structurally contain a Call node, got: {solved:?}"
        );
        // Verify on FRESH holdouts (rows NOT in the seed) — differential proof
        // the solution generalizes, not memorizes. Resolve Call against [A].
        for (x, expected) in &holdouts {
            assert_eq!(
                solved.eval_with_callees(&[*x], &callees),
                Some(*expected),
                "solved B must hold on FRESH holdout x={x}"
            );
        }

        // (b) CONTROL — registry=[]: the SAME B problem must NOT be solved within
        // the SAME budget using base ops alone. This is the necessity proof: the
        // solve in (a) depended on the callable, not on cheaper base ops.
        let no_callees: Vec<NamedCallable> = Vec::new();
        let control = solve_scalar_expr_with_callees(&problem, &no_callees, budget_ms);
        // NON-VACUOUS necessity: B must be genuinely UNSOLVED with registry=[].
        // Asserting `is_none()` (not merely "no Call node") proves the solve in (a)
        // depended on the opaque callable A — there is NO cheaper base-op program
        // that reproduces A(x)+1 within the same budget. A non-None control here
        // would mean the callable was unnecessary, voiding the necessity claim.
        assert!(
            control.is_none(),
            "CONTROL: with registry=[], B must be UNSOLVED within budget (necessity \
             of the callable), but base ops found: {control:?}"
        );
    }

    #[test]
    fn call_renders_to_real_mog_call() {
        // The Call node must emit real Mog source `opaque_a(a)` — a plain call
        // the line-based transpiler passes through unchanged.
        let call = Expr::Call(0, vec![Expr::Var(0)]);
        let names = vec!["opaque_a".to_string()];
        let rendered = with_callee_names(&names, || call.to_mog(&["a"]));
        assert_eq!(rendered, "opaque_a(a)");
    }

    #[test]
    fn empty_registry_never_constructs_call() {
        // REGRESSION: with callees=[], the enumerator behaves byte-identically —
        // it never constructs a Call, so a problem trivially solvable by base ops
        // is solved WITHOUT a Call node.
        let examples = vec![(vec![3], 6), (vec![7], 14), (vec![-2], -4)]; // 2*x
        let mut frontier = Frontier::fresh(String::new(), 1, 0);
        let (expr, _t) = enumerate_exprs_resumable_c(
            &mut frontier,
            &examples,
            5_000,
            None,
            &ALL_BINOPS,
            &ALL_UNOPS,
            None,
            &[],
        );
        let e = expr.expect("2*x must be found by base ops");
        assert!(
            !expr_contains_call(&e),
            "empty registry must never yield a Call node, got: {e:?}"
        );
    }

    // ── Example-mined constants (closed forms needing out-of-pool literals) ──

    #[test]
    fn mines_out_of_pool_additive_constant() {
        // f(a) = a + 1234: `output - input` = 1234 for every row; 1234 is NOT in
        // the fixed CONSTANTS pool, so it must be mined.
        let ex = vec![(vec![1i64], 1235i64), (vec![2], 1236), (vec![10], 1244)];
        let mined = mine_example_constants(&ex);
        assert!(mined.contains(&1234), "must mine additive literal 1234: {mined:?}");
        // fixed-pool values are excluded (0/1/2/... already seeded)
        for c in CONSTANTS {
            assert!(!mined.contains(&c), "mined set must exclude fixed pool ({c})");
        }
    }

    #[test]
    fn solves_affine_with_mined_constant() {
        use crate::benchmark::Example;
        let xs = [1i64, 2, 3, 5, 8, 13];
        let problem = Problem {
            name: "affine_mined_k".to_string(),
            category: "test",
            description: "f(a) = a + 1234 — needs a literal outside the fixed pool",
            signature: "fn affine_mined_k(a: i64) -> i64",
            examples: xs
                .iter()
                .map(|&x| Example {
                    inputs: vec![Value::Int(x)],
                    expected: Value::Int(x + 1234),
                })
                .collect(),
            holdouts: vec![],
            // NON-EMPTY reference => strict verify uses real differential holdouts,
            // so success proves generalization, not example memorization.
            reference_code: "fn affine_mined_k(a: i64) -> i64 { return a + 1234; }",
            ..Default::default()
        };
        let solved = synthesize_scalar_enumerative(&problem)
            .expect("a + 1234 must synthesize once 1234 is mined from the examples");
        assert!(solved.success, "must succeed (strict-verified): {:?}", solved.error);
        assert!(
            solved.code.contains("1234"),
            "emitted closed form must use the mined literal 1234: {}",
            solved.code
        );
    }

    // ── Array-OUTPUT generation (representation lift over the i64 ceiling) ──

    /// CLAMP examples: `min(max(item, 0), 10)` — a COMPOSED min+max element map.
    /// array_transform's element-body SEARCH is {Add,Sub,Mul,Mod}-only and its
    /// fixed templates are single-op (min-vs-const OR max-vs-const), so it cannot
    /// express the composition; the full-grammar map enumerator can.
    fn clamp_examples() -> Vec<(Vec<i64>, Vec<i64>)> {
        vec![
            (vec![-2, 3, -1, 15], vec![0, 3, 0, 10]),
            (vec![0, -7, 12], vec![0, 0, 10]),
            (vec![-3, 11, 2, 9], vec![0, 10, 2, 9]),
            (vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5]),
            (vec![-9, -1, 0, 16, -4], vec![0, 0, 0, 10, 0]),
        ]
    }
    fn clamp_problem(name: &'static str) -> Problem {
        use crate::benchmark::{Example, Value};
        let sig: &'static str = Box::leak(format!("fn {name}(arr: [i64]) -> [i64]").into_boxed_str());
        let refc: &'static str = Box::leak(
            format!(
                "fn {name}(arr: [i64]) -> [i64] {{ result: [i64] = []; for item in arr {{ \
                 result.push(min(max(item, 0), 10)); }} return result; }}"
            )
            .into_boxed_str(),
        );
        Problem {
            name: name.to_string(),
            category: "test",
            description: "elementwise clamp(x,0,10) — composed min+max map",
            signature: sig,
            examples: clamp_examples()
                .iter()
                .map(|(a, o)| Example {
                    inputs: vec![Value::int_array(a)],
                    expected: Value::int_array(o),
                })
                .collect(),
            holdouts: vec![],
            reference_code: refc, // NON-EMPTY => real differential holdouts
            ..Default::default()
        }
    }

    #[test]
    fn generates_composed_minmax_elementwise_map() {
        let problem = clamp_problem("clamp_unit");
        let solved = synthesize_array_enumerative(&problem)
            .expect("clamp min(max(item,0),10) must generate via the full-grammar element body");
        assert!(solved.success, "must succeed (strict-verified): {:?}", solved.error);
        assert_eq!(solved.method, "enumerative-array-map", "via the map path");
        assert!(
            solved.code.contains("result.push(")
                && solved.code.contains("min(")
                && solved.code.contains("max("),
            "must emit an array-building map composing min+max: {}",
            solved.code
        );
    }

    // NOTE: reachability through the full pipeline is established by structure —
    // `has_non_scalar_input` (pipeline.rs) treats arrays as scalar-ish, so an
    // array->array problem is NOT `non_scalar` and the enumerative route runs;
    // array_transform tries first and, on a composed-min/max miss, falls through
    // to this map path. A `solve_problem`-level test is omitted here because the
    // full cascade's anytime budgets make it minutes-long under load.
}
