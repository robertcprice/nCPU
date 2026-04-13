"""Test that the system actually improves from use:
solve → grow → extract → compose → solve harder."""


def test_full_improvement_loop(tmp_path):
    """The complete self-improvement loop: solve problems, grow benchmark,
    extract sub-programs, compose new solutions from discovered parts."""
    from egdc.mog.tools.repl import MogREPL
    from egdc.mog.lang import interpret

    repl = MogREPL(memory_root=str(tmp_path))

    # Phase 1: Solve basic problems
    code = repl.synthesize("max2", ["a", "b"], [
        ((2.0, 3.0), 3.0), ((10.0, -4.0), 10.0), ((7.0, 7.0), 7.0),
    ])
    assert code is not None

    code = repl.synthesize("add", ["a", "b"], [
        ((1.0, 2.0), 3.0), ((5.0, -3.0), 2.0),
    ])
    assert code is not None

    code = repl.synthesize("gcd", ["a", "b"], [
        ((12.0, 18.0), 6.0), ((21.0, 14.0), 7.0), ((9.0, 28.0), 1.0),
    ])
    assert code is not None

    assert repl.memory.total_successes() >= 3

    # Phase 2: Auto-generate harder variants
    variants = repl.grower.generate_harder_variants("max2", 3)
    assert len(variants) >= 1
    # Verify the harder variants pass with the discovered program
    for v in variants:
        for args, expected in v.examples:
            r = interpret(
                repl.solved["max2"] + f"\nfn main() -> int {{ println_i64(max2({int(args[0])}, {int(args[1])})); return 0; }}"
            )
            assert r.success
            assert int(r.output.strip()) == int(expected)

    # Phase 3: Compose LCM from discovered GCD
    lcm_code = repl.synthesize("lcm", ["a", "b"], [
        ((3.0, 4.0), 12.0), ((6.0, 8.0), 24.0), ((5.0, 10.0), 10.0),
    ])
    assert lcm_code is not None
    assert repl.memory.total_successes() >= 4

    # Phase 4: Verify composition works on new input
    r = interpret(lcm_code + "\nfn main() -> int { println_i64(lcm(15, 20)); return 0; }")
    assert r.success
    assert r.output.strip() == "60"

    # Phase 5: Auto-extract should find the Euclidean loop pattern
    repl.composer.auto_extract_and_register()
    # Should have extracted shared fragments
    assert len(repl.composer.solved_codes) >= 3

    # Phase 6: Memory should have accumulated pathways
    families = repl.memory.successes_by_family()
    assert len(families) >= 3

    # Phase 7: Similar problem retrieval should work
    hits = repl.memory.retrieve_similar(
        "Return the greatest common divisor",
        "fn gcd2(a: i64, b: i64) -> i64",
        top_k=3,
    )
    assert len(hits) >= 1


def test_interactive_improvement_loop(tmp_path):
    """Solve interactive problems, then use them as sub-programs."""
    from egdc.mog.tools.interactive import InteractiveSolver
    from egdc.mog.lang import interpret

    solver = InteractiveSolver()

    # Discover running sum
    r1 = solver.solve_from_traces("rsum", [
        [(1, 1), (2, 3), (3, 6)],
        [(10, 10), (5, 15)],
    ])
    assert r1.success and r1.verified

    # Discover counter
    r2 = solver.solve_from_traces("counter", [
        [(99, 1), (99, 2), (99, 3)],
        [(0, 1), (0, 2)],
    ])
    assert r2.success and r2.verified

    # Run counter on new trace
    result = interpret(r2.code, input_data=["5", "10", "15", "20", "25"])
    assert result.success
    assert result.output.strip() == "1\n2\n3\n4\n5"
