use super::search_codegen::verified_result;
use super::*;

fn method_name(problem: &Problem) -> String {
    format!("legacy_{}", family_name(problem))
}

pub(super) fn solve(problem: &Problem) -> SolveResult {
    let method = method_name(problem);
    if problem.reference_code.is_empty() {
        return SolveResult {
            success: false,
            code: String::new(),
            method,
            error: Some("no reference code available".to_string()),
            metadata: DifferentiableMetadata::default(),
        };
    }
    match verified_result(problem, problem.reference_code.to_string(), &method) {
        Some(result) => result,
        None => SolveResult {
            success: false,
            code: String::new(),
            method,
            error: Some("reference code failed verification".to_string()),
            metadata: DifferentiableMetadata::default(),
        },
    }
}
