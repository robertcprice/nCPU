//! Regression guard: Stage 2 & Stage 3 search teachers
//!
//! Stage 2 (broadcast/dot-product/matmul templates):
//! - code_broadcast_pattern: broadcast operation across array dimensions
//! - code_dot_product_search: two-vector inner product computation
//! - code_matmul_template: matrix multiplication patterns
//!
//! Stage 3 (struct-based field manipulation):
//! - code_struct_field_reduction: aggregate a single struct field
//! - code_struct_coupled_fields: compute interdependent struct fields
//! - code_struct_conditional_fields: conditional struct field assembly
//!
//! Each is recovered via structured search + exact verification, so all
//! must be BOTH registered in SEARCH_CANDIDATES AND in the preemption
//! whitelist to avoid wasted re-distillation through the gradient path.

use super::super::post_enumerative::search_result_preempts_native_gradient;
use super::super::search::enumerate_search_candidate_keys;

fn candidate_methods() -> Vec<&'static str> {
    let mut keys = enumerate_search_candidate_keys();
    keys.sort();
    keys
}

fn make_solve_result(method: &str, code: &str) -> crate::solver::SolveResult {
    use crate::differentiable::DifferentiableMetadata;
    crate::solver::SolveResult {
        success: true,
        code: code.to_string(),
        method: method.to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    }
}

// ============================================================================
// Stage 2: Broadcast/Dot-Product/Matmul Templates
// ============================================================================

#[test]
fn search_broadcast_pattern_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_broadcast_pattern"),
        "search_broadcast_pattern missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_broadcast_pattern", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_broadcast_pattern not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_dot_product_search_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_dot_product_search"),
        "search_dot_product_search missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_dot_product_search", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_dot_product_search not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_matmul_template_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_matmul_template"),
        "search_matmul_template missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_matmul_template", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_matmul_template not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

// ============================================================================
// Stage 3: Struct Field Reduction/Coupling/Conditional Assembly
// ============================================================================

#[test]
fn search_struct_field_reduction_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_struct_field_reduction"),
        "search_struct_field_reduction missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_struct_field_reduction", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_struct_field_reduction not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_struct_coupled_fields_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_struct_coupled_fields"),
        "search_struct_coupled_fields missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_struct_coupled_fields", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_struct_coupled_fields not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}

#[test]
fn search_struct_conditional_fields_is_registered_and_preempts_gradient() {
    let candidates = candidate_methods();
    assert!(
        candidates.contains(&"search_struct_conditional_fields"),
        "search_struct_conditional_fields missing from SEARCH_CANDIDATES; \
         a teacher must be registered to run.",
    );
    let fake = make_solve_result("search_struct_conditional_fields", "");
    assert!(
        search_result_preempts_native_gradient(&fake),
        "search_struct_conditional_fields not in preemption whitelist; it would \
         be returned only to be re-distilled through the slow gradient path.",
    );
}
