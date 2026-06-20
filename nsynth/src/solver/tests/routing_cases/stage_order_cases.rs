use super::*;

#[test]
fn scalar_template_fallback_is_explicit() {
    let problem = Problem {
        name: "positive_or_default_custom_v0".to_string(),
        category: "test",
        description: "Return x when it is positive, otherwise return 0.",
        signature: "fn positive_or_default_custom(x: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(5)],
                expected: Value::Int(5),
            },
            Example {
                inputs: vec![Value::Int(1)],
                expected: Value::Int(1),
            },
            Example {
                inputs: vec![Value::Int(0)],
                expected: Value::Int(0),
            },
            Example {
                inputs: vec![Value::Int(-7)],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![],
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let result = crate::synthesis::synthesize_scalar_templates_only(&problem)
        .expect("template fallback should solve positive_or_default");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "template");
}

#[test]
fn expr_template_fallback_is_explicit() {
    let problem = Problem {
        name: "manhattan_custom_v0".to_string(),
        category: "test",
        description: "Return the Manhattan distance between two points.",
        signature: "fn manhattan_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Int(0), Value::Int(0), Value::Int(3), Value::Int(4)],
                expected: Value::Int(7),
            },
            Example {
                inputs: vec![Value::Int(5), Value::Int(2), Value::Int(1), Value::Int(8)],
                expected: Value::Int(10),
            },
            Example {
                inputs: vec![Value::Int(-1), Value::Int(-2), Value::Int(2), Value::Int(2)],
                expected: Value::Int(7),
            },
            Example {
                inputs: vec![Value::Int(3), Value::Int(3), Value::Int(3), Value::Int(3)],
                expected: Value::Int(0),
            },
        ],
        holdouts: vec![],
        reference_code: "",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let result = crate::synthesis::synthesize_scalar_expr_templates_only(&problem)
        .expect("expr template fallback should solve manhattan distance");
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.method, "expr_template");
}

#[test]
fn reference_distillation_precedes_template_reference() {
    let problem = Problem {
        name: "abs_diff_reference_custom_v0".to_string(),
        category: "test",
        description: "Return the absolute difference between a and b.",
        signature: "fn abs_diff_reference_custom(a: i64, b: i64) -> i64",
        examples: vec![
            Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: Value::Int(0) },
            Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: Value::Int(7) },
        ],
        holdouts: vec![],
        reference_code: "fn abs_diff_reference_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let stages = post_enumerative_stage_order(&problem);
    let distill_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ReferenceDistillation)
        .expect("reference distillation stage missing");
    let template_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
        .expect("template reference stage missing");
    assert!(
        distill_idx < template_idx,
        "expected reference distillation before template_reference, got {:?}",
        stages
    );
}

#[test]
fn native_scalar_distillation_precedes_template_reference() {
    let problem = Problem {
        name: "abs_diff_native_reference_custom_v0".to_string(),
        category: "test",
        description: "Return the absolute difference between a and b.",
        signature: "fn abs_diff_native_reference_custom(a: i64, b: i64) -> i64",
        examples: vec![
            Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: Value::Int(0) },
            Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: Value::Int(7) },
        ],
        holdouts: vec![],
        reference_code: "fn abs_diff_native_reference_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let stages = post_enumerative_stage_order(&problem);
    let native_distill_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::NativeScalarTeacherDistillation)
        .expect("native scalar distillation stage missing");
    let template_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
        .expect("template reference stage missing");
    assert!(
        native_distill_idx < template_idx,
        "expected native scalar distillation before template_reference, got {:?}",
        stages
    );
}

#[test]
fn array_reference_distillation_precedes_template_reference() {
    let problem = Problem {
        name: "count_positive_reference_custom_v0".to_string(),
        category: "test",
        description: "Return the number of positive entries in the array.",
        signature: "fn count_positive_reference_custom(arr: [i64]) -> i64",
        examples: vec![
            Example { inputs: vec![Value::Array(vec![1, 2, 3, 4])], expected: Value::Int(4) },
            Example { inputs: vec![Value::Array(vec![4, -3, 2, -1])], expected: Value::Int(2) },
            Example { inputs: vec![Value::Array(vec![-5])], expected: Value::Int(0) },
            Example { inputs: vec![Value::Array(vec![0, 0, 0])], expected: Value::Int(0) },
        ],
        holdouts: vec![],
        reference_code: "fn count_positive_reference_custom(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        if arr[i] > 0 { count = count + 1; }\n        i = i + 1;\n    }\n    return count;\n}\n",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let stages = post_enumerative_stage_order(&problem);
    let arr_distill_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ArrayTeacherDistillation)
        .expect("array teacher distillation stage missing");
    let expr_tpl_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ExprTemplates)
        .expect("expr template stage missing");
    let ref_tpl_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
        .expect("template reference stage missing");
    assert!(
        arr_distill_idx < expr_tpl_idx && expr_tpl_idx < ref_tpl_idx,
        "expected array teacher distillation before template/reference fallback, got {:?}",
        stages
    );
}

#[test]
fn expr_templates_precede_scalar_and_reference_templates() {
    let problem = Problem {
        name: "manhattan_reference_custom_v0".to_string(),
        category: "test",
        description: "Return the Manhattan distance between two points.",
        signature: "fn manhattan_reference_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64",
        examples: vec![
            Example { inputs: vec![Value::Int(0), Value::Int(0), Value::Int(3), Value::Int(4)], expected: Value::Int(7) },
            Example { inputs: vec![Value::Int(5), Value::Int(2), Value::Int(1), Value::Int(8)], expected: Value::Int(10) },
            Example { inputs: vec![Value::Int(-1), Value::Int(-2), Value::Int(2), Value::Int(2)], expected: Value::Int(7) },
            Example { inputs: vec![Value::Int(3), Value::Int(3), Value::Int(3), Value::Int(3)], expected: Value::Int(0) },
        ],
        holdouts: vec![],
        reference_code: "fn manhattan_reference_custom(x1: i64, y1: i64, x2: i64, y2: i64) -> i64 {\n    dx: i64 = x1 - x2;\n    if dx < 0 { dx = 0 - dx; }\n    dy: i64 = y1 - y2;\n    if dy < 0 { dy = 0 - dy; }\n    return dx + dy;\n}\n",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let stages = post_enumerative_stage_order(&problem);
    let expr_tpl_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ExprTemplates)
        .expect("expr template stage missing");
    let scalar_tpl_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ScalarTemplates)
        .expect("scalar template stage missing");
    let ref_tpl_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
        .expect("template reference stage missing");
    assert!(
        expr_tpl_idx < scalar_tpl_idx && scalar_tpl_idx < ref_tpl_idx,
        "expected expr templates before scalar/reference templates, got {:?}",
        stages
    );
}

#[test]
fn differentiable_bridge_precedes_reference_and_template_fallbacks() {
    let problem = Problem {
        name: "abs_diff_bridge_custom_v0".to_string(),
        category: "test",
        description: "Return the absolute difference between a and b.",
        signature: "fn abs_diff_bridge_custom(a: i64, b: i64) -> i64",
        examples: vec![
            Example { inputs: vec![Value::Int(10), Value::Int(4)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(4), Value::Int(10)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(3), Value::Int(3)], expected: Value::Int(0) },
            Example { inputs: vec![Value::Int(-2), Value::Int(5)], expected: Value::Int(7) },
        ],
        holdouts: vec![],
        reference_code: "fn abs_diff_bridge_custom(a: i64, b: i64) -> i64 {\n    if a >= b { return a - b; }\n    return b - a;\n}\n",
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    };
    let stages = post_enumerative_stage_order(&problem);
    let bridge_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::BridgeGradient)
        .expect("bridge gradient stage missing");
    let ref_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::ReferenceDistillation)
        .expect("reference distillation stage missing");
    let template_idx = stages
        .iter()
        .position(|stage| *stage == PostEnumerativeStage::TemplateReference)
        .expect("template reference stage missing");
    assert!(
        bridge_idx < ref_idx && ref_idx < template_idx,
        "expected bridge gradient before reference/template fallbacks, got {:?}",
        stages
    );
}

#[test]
fn differentiable_teacher_distillation_solves_abs_diff() {
    let bridge_script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../egdc/mog_gradient_bridge.py");
    if !bridge_script.exists() {
        return;
    }
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name == "abs_diff_v0")
        .unwrap();
    let result = crate::differentiable::solve_problem_differentiable_from_teacher(
        &problem,
        problem.reference_code,
    );
    assert!(result.success, "{:?}", result.error);
    assert!(result.method.starts_with("diff_gradient_"));
    assert!(result.code.contains("return a - b;"));
    assert!(result.code.contains("return b - a;"));
}

#[test]
fn search_teacher_preempts_scalar_gradient_for_known_hard_misses() {
    let problems = get_benchmark(1);
    for name in [
        "lcm_v0",
        "euler_totient_v0",
        "next_power_of_2_v0",
        "triangular_check_v0",
        "collatz_steps_v0",
    ] {
        let problem = problems
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} not found"));
        let stages = post_enumerative_stage_order(problem);
        let search_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::SearchTeacher)
            .expect("search teacher stage missing");
        let scalar_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::ScalarGradientOnly)
            .expect("scalar gradient stage missing");
        let register_idx = stages
            .iter()
            .position(|stage| *stage == PostEnumerativeStage::RegisterMachine)
            .expect("register machine stage missing");
        assert!(
            search_idx < scalar_idx && scalar_idx < register_idx,
            "expected search teacher before scalar/register stages for {name}, got {:?}",
            stages
        );
    }
}
