from pathlib import Path


def test_pathway_memory_can_retrieve_similar_successes(tmp_path: Path):
    from egdc.mog_pathways import PathwayMemory

    mem = PathwayMemory(root=tmp_path)
    mem.record_success(
        problem_name="array_sum_v0",
        family="array_sum_reduce",
        code="fn array_sum(arr: [i64]) -> i64 { total: i64 = 0; for item in arr { total = total + item; } return total; }",
        metadata={"description": "Return the sum of all elements in an array of i64 values.", "signature": "fn array_sum(arr: [i64]) -> i64"},
    )
    mem.record_success(
        problem_name="array_max_v0",
        family="array_max_reduce",
        code="fn array_max(arr: [i64]) -> i64 { best := arr[0]; for item in arr { if item > best { best = item; } } return best; }",
        metadata={"description": "Return the largest element in a non-empty array.", "signature": "fn array_max(arr: [i64]) -> i64"},
    )
    mem.save()

    hits = mem.retrieve_similar("Return the sum of array items.", "fn sum_items(arr: [i64]) -> i64", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["family"] == "array_sum_reduce"


def test_failure_memory_exposes_anti_patterns(tmp_path: Path):
    from egdc.mog_pathways import PathwayMemory

    mem = PathwayMemory(root=tmp_path)
    mem.record_failure(
        problem_name="bad_string_case",
        family="trimmed_len",
        error_type="compiler",
        error_message="Unexpected token: return at line 2",
        metadata={"code": "fn trimmed_len(s: string) -> i64 { return return s.len; }", "anti_pattern": "double_return"},
    )
    mem.record_failure(
        problem_name="bad_split_case",
        family="vowel_count",
        error_type="runtime",
        error_message="empty separator",
        metadata={"anti_pattern": "python_split_empty"},
    )
    mem.save()

    anti = mem.anti_patterns("vowel_count")
    assert "python_split_empty" in anti


def test_adaptive_router_uses_memory_and_solves(tmp_path: Path):
    from egdc.mog_benchmark import get_benchmark
    from egdc.mog_adaptive_router import AdaptiveMogRouter

    router = AdaptiveMogRouter(memory_root=tmp_path)
    problems = get_benchmark(seed=42, variants_per_factory=1)[:5]
    summary1 = router.evaluate(problems, use_real_compiler=True)
    assert summary1["num_solved"] == 5

    # Second pass should have non-empty retrieval/family score state.
    router2 = AdaptiveMogRouter(memory_root=tmp_path)
    suggestion = router2.suggest_families(problems[0], top_k=3)
    assert len(suggestion) >= 1
    summary2 = router2.evaluate(problems, use_real_compiler=True)
    assert summary2["num_solved"] == 5
