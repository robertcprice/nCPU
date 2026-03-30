from pathlib import Path


def test_pathway_memory_persists_success_and_failure(tmp_path: Path):
    from egdc.mog_pathways import PathwayMemory

    mem = PathwayMemory(root=tmp_path)
    mem.record_success(problem_name="add_two_v0", family="binary", code="fn add_two(...) -> i64 { return a + b; }", metadata={"loss": 0.0})
    mem.record_failure(problem_name="weird_case_v0", family="binary", error_type="compiler", error_message="parse error", metadata={"code": "bad"})
    mem.save()

    mem2 = PathwayMemory(root=tmp_path)
    mem2.load()
    assert mem2.success_count("binary") == 1
    assert mem2.failure_count("binary") == 1
    assert mem2.family_score("binary") < 1.0 and mem2.family_score("binary") > 0.0


def test_adaptive_router_records_and_solves(tmp_path: Path):
    from egdc.mog_benchmark import get_benchmark
    from egdc.mog_adaptive_router import AdaptiveMogRouter

    problems = get_benchmark(seed=42, variants_per_factory=1)[:10]
    router = AdaptiveMogRouter(memory_root=tmp_path)
    summary = router.evaluate(problems, use_real_compiler=True)

    assert summary["num_problems"] == 10
    assert summary["num_solved"] >= 10
    assert summary["pass_rate"] >= 1.0

    # Memory should have recorded successful families.
    router2 = AdaptiveMogRouter(memory_root=tmp_path)
    router2.memory.load()
    assert router2.memory.total_successes() >= 10
    assert len(router2.memory.successes_by_family()) > 0
