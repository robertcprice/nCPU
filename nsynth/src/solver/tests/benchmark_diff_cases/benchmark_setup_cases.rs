use super::*;

#[test]
fn benchmark_factory_count_matches_generated_benchmark() {
    assert_eq!(factory_count(), get_benchmark(1).len());
}

#[test]
fn benchmark_generated_holdouts_cover_full_benchmark() {
    for problem in get_benchmark(1) {
        assert!(
            !generated_holdouts(&problem).is_empty(),
            "missing generated holdouts for {}",
            problem.name
        );
    }
}

#[test]
fn python_warmstart_prefers_latest_available_model() {
    let root = temp_model_root();
    fs::write(root.join("models/metalearner_1arg_v3.pt"), b"v3").unwrap();
    fs::write(root.join("models/metalearner_1arg_v5.pt"), b"v5").unwrap();

    let selected = find_python_warmstart_model(&root).unwrap();
    assert_eq!(selected, root.join("models/metalearner_1arg_v5.pt"));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn python_warmstart_falls_back_when_latest_model_is_missing() {
    let root = temp_model_root();
    fs::write(root.join("models/metalearner_1arg_v3.pt"), b"v3").unwrap();

    let selected = find_python_warmstart_model(&root).unwrap();
    assert_eq!(selected, root.join("models/metalearner_1arg_v3.pt"));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn solves_count_positive() {
    let problem = get_benchmark(1)
        .into_iter()
        .find(|p| p.name.starts_with("count_positive"))
        .unwrap();
    let result = solve_problem_search_only(&problem);
    assert!(result.success);
    assert!(result.code.contains("for item in arr"));
}
