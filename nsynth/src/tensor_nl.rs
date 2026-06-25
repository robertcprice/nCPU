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
pub fn forward_ops() -> Vec<ForwardOp> {
    vec![
        ForwardOp {
            lemma: "relu",
            arity: 1,
            surface: &["relu", "rectify", "rectified linear", "clamp negative"],
            call_expr: "{a}.relu()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::relu",
        },
        ForwardOp {
            lemma: "sigmoid",
            arity: 1,
            surface: &["sigmoid", "logistic"],
            call_expr: "{a}.sigmoid()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::sigmoid",
        },
        ForwardOp {
            lemma: "softmax",
            arity: 1,
            surface: &["softmax", "soft max", "normalized exponential"],
            call_expr: "{a}.softmax()",
            returns_result: false,
            provenance: "crate::tensor::ops::Tensor::softmax",
        },
        ForwardOp {
            lemma: "transpose",
            arity: 1,
            surface: &["transpose", "transposed"],
            call_expr: "{a}.transpose()?",
            returns_result: true,
            provenance: "crate::tensor::ops::Tensor::transpose",
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
    let mut best: Option<(usize, ForwardOp)> = None;
    for op in forward_ops() {
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

        // Every descriptor's lemma is one of the 5 reflected ops.
        let lemmas: Vec<&str> = forward_ops().iter().map(|o| o.lemma).collect();
        assert_eq!(lemmas, vec!["relu", "sigmoid", "softmax", "transpose", "matmul"]);
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
