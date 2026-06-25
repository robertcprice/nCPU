//! NL-BRIDGE-3B-TENSOR-FORWARD accept-test: the engine's tensor/DL
//! forward-inference surface is now NL-reachable through the ACTUAL product path
//! (`CodingAgentSession::handle_query`, the same entry the `coding_agent` binary
//! uses), from a FRESH root each time.
//!
//! WHY IT CANNOT BE GAMED:
//!   * FORWARD COMPILE: a forward-inference tensor request (relu / matmul /
//!     softmax) must emit a program that CALLS `mog_synth::tensor` (the engine's
//!     real `crate::tensor` op) AND pass the cargo-check compile gate. The
//!     assertions require BOTH the `crate::tensor` call in the emitted source AND
//!     a `cargo.check -> ok` tool-trace entry. A stub that printed code without
//!     compiling it (the prior absence of any tensor reach) cannot produce the
//!     `cargo.check ok` trace; a program that did not call the real engine op
//!     would not compile against the `mog_synth` path-dep, so the gate would
//!     FAIL. (These tests build `mog_synth` as a dependency of the emitted crate,
//!     so they are slower than pure-logic tests — that is the real verification.)
//!   * EMERGENT VOCAB: the tensor op set is reflected from
//!     `crate::tensor::ops::Tensor` (see `tensor_nl::forward_ops`, whose unit test
//!     compiles every reflected call against the real engine). This suite asserts
//!     the emitted op identity (`relu`/`matmul`/`softmax`) matches the reflected
//!     surface, not a hand phrase→file table.
//!   * HONEST TRAIN REFUSAL: 'train a model' / 'train a neural network' must
//!     return success:false (training is a no-op — `Trainer::train` backprop is a
//!     TODO). The assertion forbids success and requires the no-op explanation.
//!   * REGRESSION (differential): a non-tensor i64 request ("add two numbers")
//!     must STILL synthesize a real i64 program, proving the tensor route did not
//!     hijack the prior lanes (it falls through on NotTensor).

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_tensor3b_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create root");
    root
}

fn run(root: &Path, query: &str) -> mog_synth::agent::AgentQueryResult {
    let mut session = CodingAgentSession::new(root, GuardrailPolicy::default());
    session.handle_query(query)
}

fn cargo_check_ok(r: &mog_synth::agent::AgentQueryResult) -> bool {
    r.tool_trace
        .iter()
        .any(|(k, v)| k == "cargo.check" && v == "ok")
}

/// FORWARD (unary, compile-verified): "relu a tensor" emits a program that calls
/// the real engine `crate::tensor` relu AND compiles via the cargo-check gate.
#[test]
fn relu_forward_emits_engine_call_and_compiles() {
    let root = fresh_root("relu");
    let r = run(&root, "relu a tensor");
    assert!(r.success, "relu forward must succeed; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    // Calls the REAL engine tensor op (not a fabricated DSL literal).
    assert!(
        r.response.contains("mog_synth::tensor::ops::Tensor"),
        "must call crate::tensor, got:\n{}",
        r.response
    );
    assert!(
        r.response.contains("a.relu()"),
        "must emit the reflected relu call, got:\n{}",
        r.response
    );
    // UN-GAMEABLE: the cargo-check compile gate actually passed.
    assert!(
        cargo_check_ok(&r),
        "cargo.check must be ok (real compile), tool_trace: {:?}",
        r.tool_trace
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:relu")
    );
    // The emitted crate was actually written.
    assert!(root.join("Cargo.toml").exists());
    assert!(root.join("src/lib.rs").exists());
}

/// FORWARD (binary matmul, compile-verified): "multiply two matrices" emits a
/// 2-arg `crate::tensor` matmul program that compiles.
#[test]
fn matmul_forward_emits_engine_call_and_compiles() {
    let root = fresh_root("matmul");
    let r = run(&root, "multiply two matrices");
    assert!(r.success, "matmul forward must succeed; got: {}", r.response);
    assert!(
        r.response.contains("a.matmul(&b)?"),
        "must emit the reflected matmul call, got:\n{}",
        r.response
    );
    assert!(
        r.response.contains("mog_synth::tensor::ops::Tensor"),
        "must call crate::tensor, got:\n{}",
        r.response
    );
    assert!(
        cargo_check_ok(&r),
        "cargo.check must be ok (real compile), tool_trace: {:?}",
        r.tool_trace
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:matmul")
    );
}

/// FORWARD (softmax, compile-verified): a third distinct forward op, proving the
/// reach is the reflected op SET, not a single hardcoded path.
#[test]
fn softmax_forward_emits_engine_call_and_compiles() {
    let root = fresh_root("softmax");
    let r = run(&root, "apply softmax to a tensor");
    assert!(r.success, "softmax forward must succeed; got: {}", r.response);
    assert!(
        r.response.contains("a.softmax()"),
        "must emit the reflected softmax call, got:\n{}",
        r.response
    );
    assert!(cargo_check_ok(&r), "cargo.check must be ok: {:?}", r.tool_trace);
}

/// UNWALL-6 FORWARD (new unary tanh, compile-verified): "apply tanh to a tensor"
/// emits a program that calls the real engine `crate::tensor` tanh AND compiles.
#[test]
fn tanh_forward_emits_engine_call_and_compiles() {
    let root = fresh_root("tanh");
    let r = run(&root, "apply tanh to a tensor");
    assert!(r.success, "tanh forward must succeed; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    assert!(
        r.response.contains("mog_synth::tensor::ops::Tensor"),
        "must call crate::tensor, got:\n{}",
        r.response
    );
    assert!(
        r.response.contains("a.tanh()"),
        "must emit the reflected tanh call, got:\n{}",
        r.response
    );
    // UN-GAMEABLE: the cargo-check compile gate actually passed.
    assert!(
        cargo_check_ok(&r),
        "cargo.check must be ok (real compile), tool_trace: {:?}",
        r.tool_trace
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:tanh")
    );
    assert!(root.join("src/lib.rs").exists());
}

/// UNWALL-6 FORWARD (new binary add, compile-verified): "add two tensors" emits a
/// 2-arg `crate::tensor` add program that compiles. The tensor marker disambiguates
/// it from the scalar i64 "add two numbers" lane (see `i64_lane_unaffected_*`).
#[test]
fn add_two_tensors_emits_engine_call_and_compiles() {
    let root = fresh_root("addt");
    let r = run(&root, "add two tensors");
    assert!(r.success, "tensor add forward must succeed; got: {}", r.response);
    assert!(
        r.response.contains("a.add(&b)?"),
        "must emit the reflected add call, got:\n{}",
        r.response
    );
    assert!(
        r.response.contains("mog_synth::tensor::ops::Tensor"),
        "must call crate::tensor, got:\n{}",
        r.response
    );
    assert!(
        cargo_check_ok(&r),
        "cargo.check must be ok (real compile), tool_trace: {:?}",
        r.tool_trace
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:add")
    );
}

/// UNWALL-6 FORWARD (new binary mul elementwise, compile-verified): a third NEW
/// op, proving the broadened reach is the reflected op SET, not one new path.
#[test]
fn mul_two_tensors_elementwise_emits_engine_call_and_compiles() {
    let root = fresh_root("mult");
    let r = run(&root, "multiply two tensors elementwise");
    assert!(r.success, "tensor mul forward must succeed; got: {}", r.response);
    assert!(
        r.response.contains("a.mul(&b)?"),
        "must emit the reflected mul call, got:\n{}",
        r.response
    );
    assert!(cargo_check_ok(&r), "cargo.check must be ok: {:?}", r.tool_trace);
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:mul")
    );
}

/// HONEST (dim-reduction stub NOT mined): a dim-reduction request must NOT route
/// to the tensor codegen — those engine ops (mean_dim/sum_dim) are stubs that
/// ignore `dim`. It falls through to the normal lane instead of emitting a
/// fake `crate::tensor` reduction.
#[test]
fn dim_reduction_request_not_mined_as_tensor_forward() {
    let root = fresh_root("dimred");
    let r = run(&root, "sum along a dimension of the tensor");
    // It must NOT have been solved by the tensor-forward codegen.
    assert_ne!(
        r.synthesis_method.as_deref(),
        Some("tensor-forward-codegen:add"),
        "dim-reduction must not be mined as elementwise add"
    );
    assert!(
        !r.response.contains("mog_synth::tensor::ops::Tensor"),
        "dim-reduction stub must not emit a crate::tensor forward, got:\n{}",
        r.response
    );
}

/// HONEST REFUSAL: 'train a model' is a no-op here, so it must REFUSE.
#[test]
fn train_a_model_refuses_as_noop() {
    let root = fresh_root("train1");
    let r = run(&root, "train a model");
    assert!(
        !r.success,
        "training is a no-op and must be refused; got success:\n{}",
        r.response
    );
    assert!(
        r.response.to_lowercase().contains("no-op")
            || r.response.to_lowercase().contains("unimplemented"),
        "must explain the no-op honestly, got:\n{}",
        r.response
    );
    assert_eq!(r.synthesis_method.as_deref(), Some("tensor-train-refused"));
}

/// HONEST REFUSAL (second phrasing): 'train a neural network' must also refuse.
#[test]
fn train_a_neural_network_refuses_as_noop() {
    let root = fresh_root("train2");
    let r = run(&root, "train a neural network");
    assert!(
        !r.success,
        "training is a no-op and must be refused; got success:\n{}",
        r.response
    );
    assert_eq!(r.synthesis_method.as_deref(), Some("tensor-train-refused"));
}

/// REGRESSION (differential): a non-tensor i64 request is UNAFFECTED — the
/// tensor route falls through on NotTensor and the prior i64 lane still solves.
#[test]
fn i64_lane_unaffected_by_tensor_route() {
    let root = fresh_root("add");
    let r = run(&root, "add two numbers");
    assert!(r.success, "i64 request must still succeed; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    // Real i64 program, NOT a tensor program.
    assert!(
        r.response.contains("-> i64"),
        "must be an i64 program, got:\n{}",
        r.response
    );
    assert!(
        !r.response.contains("mog_synth::tensor"),
        "i64 request must NOT route to the tensor codegen, got:\n{}",
        r.response
    );
}
