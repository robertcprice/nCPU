//! Integration test: Stage 2 & Stage 3 end-to-end synthesis verification
//!
//! Runs synthesis on representative Stage 2 (tensor/broadcast) and Stage 3
//! (struct-of-state) benchmarks, verifying that the solver returns code and
//! the verifier accepts it.

use mog_synth::{
    benchmark::{Example, Problem, Value},
    runtime::verify_problem_code_strict,
    solver::solve_problem,
};
use serde_json::json;

// ============================================================================
// Stage 2 Benchmarks: Tensor Operations (Broadcast, Dot Product, MatMul)
// ============================================================================

fn stage2_broadcast_vector() -> Problem {
    Problem {
        name: "broadcast_vector_4".to_string(),
        category: "stage2_tensor",
        description: "Broadcast scalar 5 to vector of length 4",
        signature: "fn broadcast_vec(x: i64) -> [i64]",
        examples: vec![
            Example {
                inputs: vec![Value::Int(5)],
                expected: Value::Array(vec![5, 5, 5, 5]),
            },
            Example {
                inputs: vec![Value::Int(3)],
                expected: Value::Array(vec![3, 3, 3, 3]),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Int(7)],
            expected: Value::Array(vec![7, 7, 7, 7]),
        }],
        reference_code: "fn broadcast_vec(x: i64) -> [i64] { [x, x, x, x] }",
        ..Default::default()
    }
}

fn stage2_dot_product() -> Problem {
    Problem {
        name: "dot_product_simple".to_string(),
        category: "stage2_tensor",
        description: "Dot product of two vectors",
        signature: "fn dot(a: [i64], b: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2, 3]), Value::Array(vec![4, 5, 6])],
                expected: Value::Int(1 * 4 + 2 * 5 + 3 * 6), // 32
            },
            Example {
                inputs: vec![Value::Array(vec![2, 3]), Value::Array(vec![4, 5])],
                expected: Value::Int(2 * 4 + 3 * 5), // 23
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![1, 1, 1]), Value::Array(vec![2, 2, 2])],
            expected: Value::Int(6),
        }],
        reference_code: "fn dot(a: [i64], b: [i64]) -> i64 { sum(zip(a, b, |x, y| x * y)) }",
        ..Default::default()
    }
}

fn stage2_matrix_row_sum() -> Problem {
    Problem {
        name: "matrix_row_sum".to_string(),
        category: "stage2_tensor",
        description: "Sum of first row in a matrix (represented as flat array + shape hint)",
        signature: "fn row_sum_first(matrix: [i64]) -> i64",
        examples: vec![
            Example {
                // Flat array [1, 2, 3, 4, 5, 6] with implicit shape 2x3
                inputs: vec![Value::Array(vec![1, 2, 3, 4, 5, 6])],
                expected: Value::Int(1 + 2 + 3), // 6
            },
            Example {
                inputs: vec![Value::Array(vec![10, 20, 30, 40, 50, 60])],
                expected: Value::Int(10 + 20 + 30), // 60
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![5, 5, 5, 5, 5, 5])],
            expected: Value::Int(15),
        }],
        reference_code:
            "fn row_sum_first(matrix: [i64]) -> i64 { matrix[0] + matrix[1] + matrix[2] }",
        ..Default::default()
    }
}

fn stage2_vector_add() -> Problem {
    Problem {
        name: "vector_add".to_string(),
        category: "stage2_tensor",
        description: "Element-wise addition of two vectors",
        signature: "fn vec_add(a: [i64], b: [i64]) -> [i64]",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2, 3]), Value::Array(vec![4, 5, 6])],
                expected: Value::Array(vec![5, 7, 9]),
            },
            Example {
                inputs: vec![Value::Array(vec![10, 20]), Value::Array(vec![1, 2])],
                expected: Value::Array(vec![11, 22]),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![3, 3, 3]), Value::Array(vec![1, 1, 1])],
            expected: Value::Array(vec![4, 4, 4]),
        }],
        reference_code:
            "fn vec_add(a: [i64], b: [i64]) -> [i64] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }",
        ..Default::default()
    }
}

fn stage2_scalar_times_vector() -> Problem {
    Problem {
        name: "scalar_times_vector".to_string(),
        category: "stage2_tensor",
        description: "Multiply each element of vector by scalar",
        signature: "fn scalar_mul(x: i64, v: [i64]) -> [i64]",
        examples: vec![
            Example {
                inputs: vec![Value::Int(3), Value::Array(vec![1, 2, 3])],
                expected: Value::Array(vec![3, 6, 9]),
            },
            Example {
                inputs: vec![Value::Int(2), Value::Array(vec![5, 10])],
                expected: Value::Array(vec![10, 20]),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Int(4), Value::Array(vec![1, 1, 1])],
            expected: Value::Array(vec![4, 4, 4]),
        }],
        reference_code: "fn scalar_mul(x: i64, v: [i64]) -> [i64] { [x*v[0], x*v[1], x*v[2]] }",
        ..Default::default()
    }
}

fn stage2_vector_max() -> Problem {
    Problem {
        name: "vector_max_element".to_string(),
        category: "stage2_tensor",
        description: "Find maximum element in vector",
        signature: "fn vec_max(v: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![3, 1, 4, 1, 5])],
                expected: Value::Int(5),
            },
            Example {
                inputs: vec![Value::Array(vec![10, 2, 7])],
                expected: Value::Int(10),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![2, 8, 1, 9])],
            expected: Value::Int(9),
        }],
        reference_code:
            "fn vec_max(v: [i64]) -> i64 { max(max(max(v[0], v[1]), v[2]), max(v[3], v[4])) }",
        ..Default::default()
    }
}

fn stage2_vector_norm_squared() -> Problem {
    Problem {
        name: "vector_norm_squared".to_string(),
        category: "stage2_tensor",
        description: "Sum of squares (L2 norm squared)",
        signature: "fn norm_sq(v: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![3, 4])],
                expected: Value::Int(9 + 16), // 25
            },
            Example {
                inputs: vec![Value::Array(vec![1, 1, 1])],
                expected: Value::Int(3),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![2, 2, 1])],
            expected: Value::Int(9),
        }],
        reference_code: "fn norm_sq(v: [i64]) -> i64 { v[0]*v[0] + v[1]*v[1] + v[2]*v[2] }",
        ..Default::default()
    }
}

fn stage2_pairwise_distances() -> Problem {
    Problem {
        name: "pairwise_distances".to_string(),
        category: "stage2_tensor",
        description: "Sum of absolute differences between two vectors",
        signature: "fn vec_dist(a: [i64], b: [i64]) -> i64",
        examples: vec![
            Example {
                inputs: vec![Value::Array(vec![1, 2]), Value::Array(vec![4, 6])],
                expected: Value::Int((4i64 - 1i64).abs() + (6i64 - 2i64).abs()), // 7
            },
            Example {
                inputs: vec![Value::Array(vec![0, 0, 0]), Value::Array(vec![3, 4, 0])],
                expected: Value::Int(7),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Array(vec![1, 1]), Value::Array(vec![3, 3])],
            expected: Value::Int(4),
        }],
        reference_code:
            "fn vec_dist(a: [i64], b: [i64]) -> i64 { abs(a[0]-b[0]) + abs(a[1]-b[1]) }",
        ..Default::default()
    }
}

// ============================================================================
// Stage 3 Benchmarks: Struct-of-State (Field Reduction, Coupled Fields)
// ============================================================================

fn stage3_struct_sum_count() -> Problem {
    Problem {
        name: "struct_sum_count".to_string(),
        category: "stage3_struct",
        description: "Maintain running sum and count in a struct",
        signature: "fn aggregate(state: (i64, i64), arr: [i64]) -> (i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Pair(0, 0), Value::Array(vec![1, 2, 3])],
                expected: Value::Pair(6, 3),
            },
            Example {
                inputs: vec![Value::Pair(10, 5), Value::Array(vec![4, 6])],
                expected: Value::Pair(20, 7),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Pair(0, 0), Value::Array(vec![2, 2, 2])],
            expected: Value::Pair(6, 3),
        }],
        reference_code: "fn aggregate(state: (i64, i64), arr: [i64]) -> (i64, i64) { (state.0 + sum(arr), state.1 + len(arr)) }",
        ..Default::default()
    }
}

fn stage3_struct_max_min() -> Problem {
    Problem {
        name: "struct_max_min_track".to_string(),
        category: "stage3_struct",
        description: "Track running max and min in a struct",
        signature: "fn track_bounds(state: (i64, i64), arr: [i64]) -> (i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Pair(0, 100), Value::Array(vec![5, 15, 3])],
                expected: Value::Pair(15, 3),
            },
            Example {
                inputs: vec![Value::Pair(10, 50), Value::Array(vec![20, 30])],
                expected: Value::Pair(30, 20),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Pair(-100, 100), Value::Array(vec![1, 2, 3])],
            expected: Value::Pair(3, 1),
        }],
        reference_code: "fn track_bounds(state: (i64, i64), arr: [i64]) -> (i64, i64) { (max(state.0, max(arr)), min(state.1, min(arr))) }",
        ..Default::default()
    }
}

fn stage3_struct_weighted_sum() -> Problem {
    Problem {
        name: "struct_weighted_sum".to_string(),
        category: "stage3_struct",
        description: "Maintain weighted sum with coupling (sum and weight_count)",
        signature: "fn weighted(state: (i64, i64), arr: [i64]) -> (i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Pair(0, 0), Value::Array(vec![2, 4, 6])],
                expected: Value::Pair(12, 3),
            },
            Example {
                inputs: vec![Value::Pair(10, 2), Value::Array(vec![5, 5])],
                expected: Value::Pair(20, 4),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Pair(0, 0), Value::Array(vec![1, 1, 1])],
            expected: Value::Pair(3, 3),
        }],
        reference_code: "fn weighted(state: (i64, i64), arr: [i64]) -> (i64, i64) { (state.0 + sum(arr), state.1 + len(arr)) }",
        ..Default::default()
    }
}

fn stage3_struct_quad_state() -> Problem {
    Problem {
        name: "struct_quad_accumulate".to_string(),
        category: "stage3_struct",
        description: "Four-field struct: sum, count, max, min",
        signature: "fn quad_agg(state: (i64, i64, i64, i64), arr: [i64]) -> (i64, i64, i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Quad(0, 0, -100, 100), Value::Array(vec![1, 5, 3])],
                expected: Value::Quad(9, 3, 5, 1),
            },
            Example {
                inputs: vec![Value::Quad(10, 2, 20, 0), Value::Array(vec![15, 25])],
                expected: Value::Quad(50, 4, 25, 0),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Quad(0, 0, -1000, 1000), Value::Array(vec![2, 2, 2])],
            expected: Value::Quad(6, 3, 2, 2),
        }],
        reference_code: "fn quad_agg(state: (i64, i64, i64, i64), arr: [i64]) -> (i64, i64, i64, i64) { (state.0 + sum(arr), state.1 + len(arr), max(state.2, max(arr)), min(state.3, min(arr))) }",
        ..Default::default()
    }
}

fn stage3_struct_conditional() -> Problem {
    Problem {
        name: "struct_conditional_count".to_string(),
        category: "stage3_struct",
        description: "Conditional accumulation: count positive and total",
        signature: "fn cond_count(state: (i64, i64), arr: [i64]) -> (i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Pair(0, 0), Value::Array(vec![1, -2, 3, -4, 5])],
                expected: Value::Pair(3, 5), // 3 positive, 5 total
            },
            Example {
                inputs: vec![Value::Pair(2, 0), Value::Array(vec![10, -5])],
                expected: Value::Pair(3, 2),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Pair(0, 0), Value::Array(vec![1, 2, 3])],
            expected: Value::Pair(3, 3),
        }],
        reference_code: "fn cond_count(state: (i64, i64), arr: [i64]) -> (i64, i64) { (state.0 + count_positive(arr), state.1 + len(arr)) }",
        ..Default::default()
    }
}

fn stage3_struct_cross_field() -> Problem {
    Problem {
        name: "struct_cross_field_delta".to_string(),
        category: "stage3_struct",
        description: "Cross-field coupling: maintain difference between accumulated values",
        signature: "fn delta_agg(state: (i64, i64), arr: [i64]) -> (i64, i64)",
        examples: vec![
            Example {
                inputs: vec![Value::Pair(0, 0), Value::Array(vec![2, 4, 6])],
                expected: Value::Pair(12, 3), // sum = 12, len = 3, implicit delta maintained
            },
            Example {
                inputs: vec![Value::Pair(10, 5), Value::Array(vec![1, 1])],
                expected: Value::Pair(12, 7),
            },
        ],
        holdouts: vec![Example {
            inputs: vec![Value::Pair(0, 0), Value::Array(vec![5, 5])],
            expected: Value::Pair(10, 2),
        }],
        reference_code: "fn delta_agg(state: (i64, i64), arr: [i64]) -> (i64, i64) { (state.0 + sum(arr), state.1 + len(arr)) }",
        ..Default::default()
    }
}

// ============================================================================
// Integration Test: Run Synthesis on All Stage 2 & 3 Benchmarks
// ============================================================================

#[test]
#[ignore] // Large integration test; run with: cargo test --test integration_stage2_stage3 -- --ignored
fn stage2_stage3_end_to_end_synthesis() {
    let stage2_benchmarks = vec![
        stage2_broadcast_vector(),
        stage2_dot_product(),
        stage2_matrix_row_sum(),
        stage2_vector_add(),
        stage2_scalar_times_vector(),
        stage2_vector_max(),
        stage2_vector_norm_squared(),
        stage2_pairwise_distances(),
    ];

    let stage3_benchmarks = vec![
        stage3_struct_sum_count(),
        stage3_struct_max_min(),
        stage3_struct_weighted_sum(),
        stage3_struct_quad_state(),
        stage3_struct_conditional(),
        stage3_struct_cross_field(),
    ];

    println!("\n================================================================================");
    println!("STAGE 2 END-TO-END SYNTHESIS VERIFICATION");
    println!("================================================================================\n");

    let mut stage2_passed = 0;
    let mut stage2_failed = 0;
    let mut stage2_times = Vec::new();

    for problem in &stage2_benchmarks {
        let start = std::time::Instant::now();
        let result = solve_problem(problem);
        let elapsed = start.elapsed().as_secs_f64();
        stage2_times.push(elapsed);

        if result.success {
            // Verify the synthesized code
            match verify_problem_code_strict(problem, &result.code) {
                Ok(_) => {
                    println!(
                        "✅ {} [{}s, method={}]",
                        problem.name,
                        format!("{:.3}", elapsed),
                        result.method
                    );
                    stage2_passed += 1;
                }
                Err(e) => {
                    println!("❌ {} [verification failed: {}]", problem.name, e);
                    stage2_failed += 1;
                }
            }
        } else {
            println!(
                "❌ {} [solve failed: {}]",
                problem.name,
                result.error.as_deref().unwrap_or("unknown")
            );
            stage2_failed += 1;
        }
    }

    println!("\n================================================================================");
    println!("STAGE 3 END-TO-END SYNTHESIS VERIFICATION");
    println!("================================================================================\n");

    let mut stage3_passed = 0;
    let mut stage3_failed = 0;
    let mut stage3_times = Vec::new();

    for problem in &stage3_benchmarks {
        let start = std::time::Instant::now();
        let result = solve_problem(problem);
        let elapsed = start.elapsed().as_secs_f64();
        stage3_times.push(elapsed);

        if result.success {
            // Verify the synthesized code
            match verify_problem_code_strict(problem, &result.code) {
                Ok(_) => {
                    println!(
                        "✅ {} [{}s, method={}]",
                        problem.name,
                        format!("{:.3}", elapsed),
                        result.method
                    );
                    stage3_passed += 1;
                }
                Err(e) => {
                    println!("❌ {} [verification failed: {}]", problem.name, e);
                    stage3_failed += 1;
                }
            }
        } else {
            println!(
                "❌ {} [solve failed: {}]",
                problem.name,
                result.error.as_deref().unwrap_or("unknown")
            );
            stage3_failed += 1;
        }
    }

    // ========================================================================
    // Summary Report
    // ========================================================================

    let stage2_total = stage2_passed + stage2_failed;
    let stage3_total = stage3_passed + stage3_failed;
    let stage2_mean_time = if !stage2_times.is_empty() {
        stage2_times.iter().sum::<f64>() / stage2_times.len() as f64
    } else {
        0.0
    };
    let stage3_mean_time = if !stage3_times.is_empty() {
        stage3_times.iter().sum::<f64>() / stage3_times.len() as f64
    } else {
        0.0
    };

    println!("\n================================================================================");
    println!("SYNTHESIS VERIFICATION SUMMARY");
    println!("================================================================================");
    println!(
        "Stage 2 (Tensor):  {}/{} passed ({:.1}%) | mean time: {:.3}s",
        stage2_passed,
        stage2_total,
        if stage2_total > 0 {
            (stage2_passed as f64 / stage2_total as f64) * 100.0
        } else {
            0.0
        },
        stage2_mean_time
    );
    println!(
        "Stage 3 (Struct):  {}/{} passed ({:.1}%) | mean time: {:.3}s",
        stage3_passed,
        stage3_total,
        if stage3_total > 0 {
            (stage3_passed as f64 / stage3_total as f64) * 100.0
        } else {
            0.0
        },
        stage3_mean_time
    );
    println!("────────────────────────────────────────────────────────────────────────────────");
    println!(
        "TOTAL:             {}/{} passed",
        stage2_passed + stage3_passed,
        stage2_total + stage3_total
    );
    println!("================================================================================\n");

    // JSON output for parsing by calling script
    let summary = json!({
        "stage2": {
            "passed": stage2_passed,
            "total": stage2_total,
            "mean_time_s": stage2_mean_time,
        },
        "stage3": {
            "passed": stage3_passed,
            "total": stage3_total,
            "mean_time_s": stage3_mean_time,
        },
        "overall": {
            "passed": stage2_passed + stage3_passed,
            "total": stage2_total + stage3_total,
        }
    });

    println!("JSON_RESULT: {}", summary.to_string());

    // Assert at least some passes (not blocking: diagnostic test)
    assert!(
        stage2_passed + stage3_passed > 0,
        "At least some benchmarks should synthesize successfully"
    );
}
