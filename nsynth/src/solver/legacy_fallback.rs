use super::*;

fn method_name(problem: &Problem) -> String {
    format!("legacy_{}", family_name(problem))
}

pub(super) fn solve(problem: &Problem) -> SolveResult {
    let method = method_name(problem);
    SolveResult {
        success: false,
        code: String::new(),
        method,
        error: Some(
            "oracle-assisted legacy fallback is disabled; reference code is evaluation-only"
                .to_string(),
        ),
        metadata: DifferentiableMetadata::default(),
    }
}
