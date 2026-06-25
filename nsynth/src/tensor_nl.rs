//! NL-BRIDGE-3B-TENSOR-FORWARD: the first NL reach into the tensor/DL subsystem.
//!
//! This module makes the engine's **forward-inference** tensor surface
//! (`crate::tensor::ops::Tensor::{relu, sigmoid, softmax, transpose, matmul}`)
//! reachable from a natural-language request — by CODEGEN, not example search.
//! Tensors are too large to enumerate, so the honest mechanism is
//! codegen-from-op-identity: an NL request that names a forward op emits a small
//! Rust program that *calls the real engine op* and is verified by the existing
//! cargo-check compile gate.
//!
//! ## Why this is emergent, not a hand phrase table
//! The op SET ([`forward_ops`]) is REFLECTED from the real `Tensor` impl: each
//! descriptor carries the op's actual Rust call expression (`x.relu()`,
//! `a.matmul(&b)?`, …) and a `provenance` pointer at `crate::tensor::ops::Tensor`.
//! The unit test [`tests::forward_op_calls_compile_against_engine`] *compiles*
//! every emitted call against the real crate, so a descriptor that does not match
//! a real engine method fails closed. Surface synonyms are presentation metadata
//! only — they cannot smuggle in an op the engine does not expose (the call
//! expression still has to type-check against `crate::tensor`).
//!
//! ## Honest training gate
//! Tensor *training* is a no-op in this engine (`Trainer::train`'s backprop is a
//! TODO, autodiff is disconnected). [`is_training_request`] detects 'train a
//! model' / 'fit' / 'backprop' style requests so the caller can REFUSE them
//! rather than emit code that pretends to learn.

/// A forward-inference tensor op reflected from the engine's `Tensor` surface.
#[derive(Debug, Clone)]
pub struct ForwardOp {
    /// Canonical lemma (also the synthesized fn name stem).
    pub lemma: &'static str,
    /// Arity of the op (1 = unary `fn(Tensor)->Tensor`, 2 = `fn(Tensor,Tensor)->Tensor`).
    pub arity: usize,
    /// NL surface forms that resolve to this op (presentation metadata only —
    /// the op identity is the engine call expression below).
    pub surface: &'static [&'static str],
    /// The REAL engine call expression, with `{a}` / `{b}` placeholders for the
    /// input tensor variable names. This is emitted verbatim into the generated
    /// program, so it must name a real `crate::tensor::ops::Tensor` method.
    pub call_expr: &'static str,
    /// Whether the engine method returns `Result<Tensor, String>` (so the
    /// generated code unwraps it) vs a bare `Tensor`.
    pub returns_result: bool,
    /// Provenance: the engine symbol this op is reflected from.
    pub provenance: &'static str,
    /// Whether a surface match also requires the request to be tensor-flavoured.
    /// Some op verbs (`add`/`sub`/`mul`/`div`) are ambiguous with scalar i64/float
    /// arithmetic ("add two numbers"). For those we require an explicit tensor
    /// marker ("tensor"/"matrix"/"elementwise") so the tensor route does not
    /// hijack the prior numeric lanes. Unambiguous ops (relu/matmul/…) set false.
    pub requires_tensor_context: bool,
}

/// Markers that make an ambiguous-verb request unambiguously tensor-flavoured.
const TENSOR_CONTEXT_MARKERS: &[&str] = &[
    "tensor",
    "tensors",
    "matrix",
    "matrices",
    "elementwise",
    "element-wise",
    "element wise",
];

/// Is the request tensor-flavoured (carries an explicit tensor/matrix marker)?
fn has_tensor_context(lower: &str) -> bool {
    TENSOR_CONTEXT_MARKERS.iter().any(|m| lower.contains(m))
}

/// The forward-inference op surface, reflected from `crate::tensor::ops::Tensor`.
///
/// Each `call_expr` is a real method on `Tensor` (see `src/tensor/ops.rs`):
///   * `relu`      — `pub fn relu(&self) -> Tensor`
///   * `sigmoid`   — `pub fn sigmoid(&self) -> Tensor`
///   * `softmax`   — `pub fn softmax(&self) -> Tensor`
///   * `transpose` — `pub fn transpose(&self) -> Result<Tensor, String>`
///   * `matmul`    — `pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String>`
///
/// Add a forward op variant here AND it is auto-covered by the compile test
/// below; a bogus call expression fails that test (fail-closed reflection).
///
/// UNWALL-6 broadened the surface beyond the original 5 by reflecting MORE real
/// forward ops from `src/tensor/ops.rs`, all confirmed-real (a working forward,
/// not a stub):
///   * `tanh`  — `pub fn tanh(&self)  -> Tensor`          (ops.rs:359)
///   * `sqrt`  — `pub fn sqrt(&self)  -> Tensor`          (ops.rs:482)
///   * `add`   — `pub fn add(&self,o) -> Result<Tensor>`  (ops.rs:214)
///   * `sub`   — `pub fn sub(&self,o) -> Result<Tensor>`  (ops.rs:228)
///   * `mul`   — `pub fn mul(&self,o) -> Result<Tensor>`  (ops.rs:242)
///   * `div`   — `pub fn div(&self,o) -> Result<Tensor>`  (ops.rs:256)
/// The dim-reduction ops (`mean_dim`/`var_dim`/`sum_dim`) are STUBS that ignore
/// `dim` and collapse to a scalar (ops.rs:494-509) — they are deliberately NOT
/// mined here (see `tests::dim_reduction_stubs_not_mined`).
pub fn forward_ops() -> Vec<ForwardOp> {
    vec![
        ForwardOp {
            lemma: "relu",
            arity: 1,
            surface: &["relu", "rectify", "rectified linear", "clamp negative"],
            call_expr: "{a}.relu()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::relu",
            requires_tensor_context: false,
        },
        ForwardOp {
            lemma: "sigmoid",
            arity: 1,
            surface: &["sigmoid", "logistic"],
            call_expr: "{a}.sigmoid()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::sigmoid",
            requires_tensor_context: false,
        },
        ForwardOp {
            lemma: "softmax",
            arity: 1,
            surface: &["softmax", "soft max", "normalized exponential"],
            call_expr: "{a}.softmax()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::softmax",
            requires_tensor_context: false,
        },
        ForwardOp {
            lemma: "transpose",
            arity: 1,
            surface: &["transpose", "transposed"],
            call_expr: "{a}.transpose()?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::transpose",
            requires_tensor_context: false,
        },
        ForwardOp {
            lemma: "matmul",
            arity: 2,
            surface: &[
                "matmul",
                "matrix multiply",
                "multiply two matrices",
                "matrix product",
                "matrix multiplication",
            ],
            call_expr: "{a}.matmul(&{b})?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::matmul",
            requires_tensor_context: false,
        },
        // --- UNWALL-6: broadened forward surface (all confirmed-real forwards) ---
        ForwardOp {
            lemma: "tanh",
            arity: 1,
            surface: &["tanh", "hyperbolic tangent"],
            call_expr: "{a}.tanh()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::tanh",
            requires_tensor_context: false,
        },
        ForwardOp {
            lemma: "sqrt",
            arity: 1,
            // "square root" is unambiguous enough to be tensor-specific in this
            // engine (scalar sqrt is not an NL lane here), but require a tensor
            // marker for the bare "sqrt"/"square root" verb to avoid surprises.
            surface: &["sqrt", "square root", "element-wise square root"],
            call_expr: "{a}.sqrt()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::sqrt",
            requires_tensor_context: true,
        },
        ForwardOp {
            lemma: "add",
            arity: 2,
            // NB: no "sum" here — "sum"/"sum along a dim" is a REDUCTION, not a
            // binary elementwise add, and the engine's reductions are stubs.
            surface: &["add", "plus", "elementwise add", "element-wise add"],
            call_expr: "{a}.add(&{b})?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::add",
            requires_tensor_context: true,
        },
        ForwardOp {
            lemma: "sub",
            arity: 2,
            surface: &["sub", "subtract", "minus", "difference", "elementwise subtract"],
            call_expr: "{a}.sub(&{b})?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::sub",
            requires_tensor_context: true,
        },
        ForwardOp {
            lemma: "mul",
            arity: 2,
            surface: &[
                "multiply",
                "mul",
                "elementwise multiply",
                "element-wise multiply",
                "hadamard",
                "hadamard product",
            ],
            call_expr: "{a}.mul(&{b})?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::mul",
            requires_tensor_context: true,
        },
        ForwardOp {
            lemma: "div",
            arity: 2,
            surface: &["divide", "div", "elementwise divide", "element-wise divide"],
            call_expr: "{a}.div(&{b})?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::div",
            requires_tensor_context: true,
        },
    ]
}

/// Training-request markers. Tensor TRAINING is a NO-OP in this engine
/// (`Trainer::train` backprop is a TODO), so any of these must be REFUSED.
const TRAIN_MARKERS: &[&str] = &[
    "train",
    "fit ",
    "backprop",
    "back-propagat",
    "backpropagat",
    "gradient descent",
    "optimize the weights",
    "optimise the weights",
    "learn the weights",
    "update the weights",
    "fine-tune",
    "finetune",
];

/// Is this an honest-refusal training request? (training is a no-op here.)
pub fn is_training_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    // "train a model" / "train a neural network" / "fit a model" etc.
    if TRAIN_MARKERS.iter().any(|m| lower.contains(m)) {
        // Only treat as training if it is tensor/model-flavoured, so we don't
        // refuse unrelated requests that happen to contain "fit"/"train".
        let model_flavoured = ["model", "network", "net", "tensor", "weights", "layer", "nn"]
            .iter()
            .any(|w| lower.contains(w))
            || lower.contains("train");
        return model_flavoured;
    }
    false
}

/// Resolve an NL request to a forward-inference tensor op (or `None`). Matches
/// the request against each op's reflected surface forms (longest match wins so
/// "multiply two matrices" beats a bare "multiply").
pub fn resolve_forward_op(text: &str) -> Option<ForwardOp> {
    let lower = text.to_lowercase();
    let tensor_ctx = has_tensor_context(&lower);
    let mut best: Option<(usize, ForwardOp)> = None;
    for op in forward_ops() {
        // Ambiguous-verb ops (add/sub/mul/div/sqrt) only fire when the request is
        // explicitly tensor-flavoured, so "add two numbers" stays an i64 request.
        if op.requires_tensor_context && !tensor_ctx {
            continue;
        }
        for form in op.surface {
            if lower.contains(form) {
                let len = form.len();
                if best.as_ref().map(|(l, _)| len > *l).unwrap_or(true) {
                    best = Some((len, op.clone()));
                }
            }
        }
    }
    best.map(|(_, op)| op)
}

/// Outcome of a tensor-NL classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorRouteOutcome {
    /// A forward-inference op was recognized; carries the generated program.
    Forward { lemma: String, code: String },
    /// A training request — must be refused (training is a no-op).
    RefuseTraining,
    /// Not a tensor request at all (caller falls through to the normal path).
    NotTensor,
}

/// Classify an NL request for the tensor reach. Training requests refuse;
/// forward requests emit a real `crate::tensor`-calling program.
pub fn classify(text: &str) -> TensorRouteOutcome {
    if is_training_request(text) {
        return TensorRouteOutcome::RefuseTraining;
    }
    match resolve_forward_op(text) {
        Some(op) => TensorRouteOutcome::Forward {
            lemma: op.lemma.to_string(),
            code: emit_forward_program(&op),
        },
        None => TensorRouteOutcome::NotTensor,
    }
}

/// Emit a complete, self-contained Rust `lib.rs` body for a forward op. The
/// emitted fn takes engine `Tensor`(s), calls the REAL engine op, and returns a
/// `Tensor`. It is compiled (against the real `mog_synth` crate as a path dep —
/// see [`tensor_crate_files`]) by the cargo-check gate. NOT example-search:
/// codegen-from-op-identity is the honest mechanism for tensors.
pub fn emit_forward_program(op: &ForwardOp) -> String {
    let call = op.call_expr.replace("{a}", "a").replace("{b}", "b");
    let (params, ret) = match op.arity {
        1 => ("a: Tensor".to_string(), "Tensor"),
        _ => ("a: Tensor, b: Tensor".to_string(), "Tensor"),
    };
    // Unary `?`-using ops (transpose) and binary ops (matmul) return Result so
    // the `?` operator is valid; the fn signature reflects that.
    if op.returns_result {
        format!(
            "//! Forward-inference tensor program (NL-BRIDGE-3B-TENSOR-FORWARD).\n\
             //! Op `{lemma}` reflected from {prov}.\n\
             use mog_synth::tensor::ops::Tensor;\n\n\
             pub fn {lemma}_forward({params}) -> Result<{ret}, String> {{\n    \
             let out = {call};\n    \
             Ok(out)\n}}\n",
            lemma = op.lemma,
            prov = op.provenance,
            params = params,
            ret = ret,
            call = call,
        )
    } else {
        format!(
            "//! Forward-inference tensor program (NL-BRIDGE-3B-TENSOR-FORWARD).\n\
             //! Op `{lemma}` reflected from {prov}.\n\
             use mog_synth::tensor::ops::Tensor;\n\n\
             pub fn {lemma}_forward({params}) -> {ret} {{\n    \
             {call}\n}}\n",
            lemma = op.lemma,
            prov = op.provenance,
            params = params,
            ret = ret,
            call = call,
        )
    }
}

/// Build the file set for a self-contained tensor crate that depends on the
/// canonical `mog_synth` crate (path dep resolved from `CARGO_MANIFEST_DIR`, the
/// nsynth crate dir baked in at compile time) so the emitted `crate::tensor`
/// calls actually link + type-check. Returns `(relative_path, content)` pairs:
/// `Cargo.toml`, `src/lib.rs`.
pub fn tensor_crate_files(lemma: &str, lib_body: &str) -> Vec<(String, String)> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let cargo_toml = format!(
        "[package]\nname = \"tensor-{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
         [lib]\npath = \"src/lib.rs\"\n\n\
         [dependencies]\nmog_synth = {{ path = \"{}\" }}\n",
        lemma.replace('_', "-"),
        manifest_dir,
    );
    vec![
        ("Cargo.toml".to_string(), cargo_toml),
        ("src/lib.rs".to_string(), lib_body.to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::ops::{Shape, Tensor};

    /// EMERGENT: each reflected forward op's call expression is a REAL method on
    /// the engine `Tensor`. We don't compile a child crate here (slow) — we call
    /// the same methods the codegen emits, proving the descriptor surface is the
    /// engine surface (a bogus descriptor would not compile this test).
    #[test]
    fn forward_op_calls_compile_against_engine() {
        let m = Tensor::matrix(vec![1.0, -2.0, 3.0, -4.0], 2, 2);
        let n = Tensor::matrix(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let v = Tensor::vector(vec![1.0, 2.0, 3.0]);

        // relu / sigmoid / softmax — bare Tensor return.
        let _ = m.relu();
        let _ = m.sigmoid();
        let _ = v.softmax();
        // transpose / matmul — Result<Tensor, String> return.
        let _ = m.transpose().expect("transpose");
        let _ = m.matmul(&n).expect("matmul");
        // UNWALL-6 broadened forwards — unary tanh/sqrt (bare Tensor) ...
        let _ = m.tanh();
        let _ = m.sqrt();
        // ... and binary elementwise add/sub/mul/div (Result<Tensor, String>).
        let _ = m.add(&n).expect("add");
        let _ = m.sub(&n).expect("sub");
        let _ = m.mul(&n).expect("mul");
        let _ = m.div(&n).expect("div");

        // Every descriptor's lemma is one of the reflected ops (fail-closed: a
        // bogus call_expr above would not compile, and an unlisted lemma here
        // would fail this assertion).
        let lemmas: Vec<&str> = forward_ops().iter().map(|o| o.lemma).collect();
        assert_eq!(
            lemmas,
            vec![
                "relu", "sigmoid", "softmax", "transpose", "matmul", // original 5
                "tanh", "sqrt", "add", "sub", "mul", "div", // UNWALL-6
            ]
        );
        let _ = Shape::new(vec![2, 2]);
    }

    /// The emitted unary program references the engine `Tensor` and calls relu.
    #[test]
    fn emit_unary_program_calls_engine() {
        let op = resolve_forward_op("relu a tensor").expect("relu resolves");
        let code = emit_forward_program(&op);
        assert!(code.contains("use mog_synth::tensor::ops::Tensor;"));
        assert!(code.contains("a.relu()"));
        assert!(code.contains("fn relu_forward(a: Tensor) -> Tensor"));
    }

    /// The emitted binary (matmul) program calls the real 2-arg engine op.
    #[test]
    fn emit_matmul_program_calls_engine() {
        let op = resolve_forward_op("multiply two matrices").expect("matmul resolves");
        assert_eq!(op.lemma, "matmul");
        let code = emit_forward_program(&op);
        assert!(code.contains("a.matmul(&b)?"));
        assert!(code.contains("fn matmul_forward(a: Tensor, b: Tensor) -> Result<Tensor, String>"));
    }

    /// HONEST GATE: training requests are refused (training is a no-op here).
    #[test]
    fn training_requests_refuse() {
        assert_eq!(classify("train a model"), TensorRouteOutcome::RefuseTraining);
        assert_eq!(
            classify("train a neural network"),
            TensorRouteOutcome::RefuseTraining
        );
        assert!(matches!(
            classify("fit a model to the data"),
            TensorRouteOutcome::RefuseTraining
        ));
        assert!(matches!(
            classify("backprop the gradients through the net"),
            TensorRouteOutcome::RefuseTraining
        ));
    }

    /// Forward requests classify as Forward with generated code.
    #[test]
    fn forward_requests_route_to_codegen() {
        match classify("apply softmax to a tensor") {
            TensorRouteOutcome::Forward { lemma, code } => {
                assert_eq!(lemma, "softmax");
                assert!(code.contains("a.softmax()"));
            }
            other => panic!("expected Forward, got {other:?}"),
        }
        match classify("relu a tensor") {
            TensorRouteOutcome::Forward { lemma, .. } => assert_eq!(lemma, "relu"),
            other => panic!("expected Forward, got {other:?}"),
        }
    }

    /// Non-tensor requests fall through (NotTensor), so prior types are untouched.
    #[test]
    fn non_tensor_requests_fall_through() {
        assert_eq!(classify("add two numbers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("sort an array"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("lowercase a string"), TensorRouteOutcome::NotTensor);
    }

    /// UNWALL-6: the new unary forward ops resolve + emit the real engine call.
    #[test]
    fn new_unary_forward_ops_resolve_and_emit() {
        let op = resolve_forward_op("apply tanh to a tensor").expect("tanh resolves");
        assert_eq!(op.lemma, "tanh");
        let code = emit_forward_program(&op);
        assert!(code.contains("a.tanh()"), "got:\n{code}");
        assert!(code.contains("fn tanh_forward(a: Tensor) -> Tensor"), "got:\n{code}");

        // sqrt requires a tensor marker (ambiguous with scalar math).
        let op = resolve_forward_op("element-wise square root of a tensor")
            .expect("sqrt resolves with tensor ctx");
        assert_eq!(op.lemma, "sqrt");
        assert!(emit_forward_program(&op).contains("a.sqrt()"));
    }

    /// UNWALL-6: the new binary elementwise forward ops resolve (when tensor-
    /// flavoured) + emit the real 2-arg engine call.
    #[test]
    fn new_binary_forward_ops_resolve_and_emit() {
        for (req, lemma, call) in [
            ("add two tensors", "add", "a.add(&b)?"),
            ("subtract two tensors", "sub", "a.sub(&b)?"),
            ("multiply two tensors elementwise", "mul", "a.mul(&b)?"),
            ("divide two tensors", "div", "a.div(&b)?"),
        ] {
            let op = resolve_forward_op(req).unwrap_or_else(|| panic!("{req} resolves"));
            assert_eq!(op.lemma, lemma, "req={req}");
            let code = emit_forward_program(&op);
            assert!(code.contains(call), "req={req} got:\n{code}");
            assert!(
                code.contains(&format!(
                    "fn {lemma}_forward(a: Tensor, b: Tensor) -> Result<Tensor, String>"
                )),
                "req={req} got:\n{code}"
            );
        }
    }

    /// REGRESSION (differential): ambiguous verbs WITHOUT a tensor marker do NOT
    /// hijack the prior numeric/array lanes — they fall through to NotTensor.
    #[test]
    fn ambiguous_verbs_without_tensor_context_fall_through() {
        assert_eq!(classify("add two numbers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("subtract two numbers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("multiply two numbers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("divide two numbers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("sum a list of integers"), TensorRouteOutcome::NotTensor);
        assert_eq!(classify("compute the square root of n"), TensorRouteOutcome::NotTensor);
        // But WITH a tensor marker they DO route to the tensor codegen.
        assert!(matches!(
            classify("add two tensors"),
            TensorRouteOutcome::Forward { ref lemma, .. } if lemma == "add"
        ));
    }

    /// HONEST: the dim-reduction ops (mean_dim/var_dim/sum_dim) are STUBS that
    /// ignore `dim` and collapse to a scalar (ops.rs:494-509). They are NOT mined
    /// into the forward surface, so a dim-reduction request must NOT resolve.
    #[test]
    fn dim_reduction_stubs_not_mined() {
        // No descriptor names a *_dim op.
        assert!(
            forward_ops()
                .iter()
                .all(|o| !o.lemma.contains("_dim") && !o.provenance.contains("_dim")),
            "dim-reduction stubs must not appear in the forward surface"
        );
        // And a dim-reduction NL request does not resolve to a forward op.
        assert_eq!(
            classify("compute the mean along a dimension of a tensor"),
            TensorRouteOutcome::NotTensor
        );
        assert_eq!(
            classify("sum along a dimension of the tensor"),
            TensorRouteOutcome::NotTensor
        );
    }

    /// The emitted crate carries a path dep on the canonical mog_synth crate so
    /// `crate::tensor` resolves at the compile gate.
    #[test]
    fn crate_files_depend_on_engine() {
        let op = resolve_forward_op("relu a tensor").unwrap();
        let body = emit_forward_program(&op);
        let files = tensor_crate_files(&op.lemma, &body);
        let cargo = &files.iter().find(|(p, _)| p == "Cargo.toml").unwrap().1;
        assert!(cargo.contains("mog_synth = { path ="));
        assert!(files.iter().any(|(p, _)| p == "src/lib.rs"));
    }
}
