from pathlib import Path


def test_orchestrator_solves_with_direct_synth_first(tmp_path: Path):
    from egdc.mog.benchmark import get_benchmark
    from egdc.mog.routing.orchestrator import MogOrchestrator

    orch = MogOrchestrator(memory_root=tmp_path)
    problems = get_benchmark(seed=42, variants_per_factory=1)[:5]
    results = orch.solve_batch(problems)
    assert len(results) == 5
    for r in results:
        assert r.success, f"problem failed: {r.error}"
        assert r.method in ("search", "direct", "retrieval")


def test_orchestrator_records_and_retrieves_pathways(tmp_path: Path):
    from egdc.mog.benchmark import get_benchmark
    from egdc.mog.routing.orchestrator import MogOrchestrator

    orch = MogOrchestrator(memory_root=tmp_path)
    problems = get_benchmark(seed=42, variants_per_factory=1)[:10]
    orch.solve_batch(problems)

    # Memory should now have stored pathways.
    orch2 = MogOrchestrator(memory_root=tmp_path)
    assert orch2.memory.total_successes() >= 10

    # Retrieve similar for a new-ish problem.
    p = problems[0]
    hits = orch2.memory.retrieve_similar(p.description, p.signature, top_k=3)
    assert len(hits) >= 1


def test_orchestrator_enforces_anti_patterns(tmp_path: Path):
    from egdc.mog.routing.orchestrator import MogOrchestrator

    orch = MogOrchestrator(memory_root=tmp_path)
    # Inject a known anti-pattern.
    orch.memory.record_failure(
        "test_problem", "binary", "compiler",
        "double return statement",
        {"anti_pattern": "double_return", "code": "return return x;"},
    )
    orch.memory.save()

    blocked = orch.check_anti_patterns("binary", "fn foo() -> i64 { return return x; }")
    assert len(blocked) >= 1
    assert "double_return" in blocked


def test_orchestrator_full_benchmark_25(tmp_path: Path):
    from egdc.mog.benchmark import get_benchmark
    from egdc.mog.routing.orchestrator import MogOrchestrator

    orch = MogOrchestrator(memory_root=tmp_path)
    problems = get_benchmark(seed=42, variants_per_factory=1)[:25]
    results = orch.solve_batch(problems)
    solved = sum(1 for r in results if r.success)
    assert solved >= 25
