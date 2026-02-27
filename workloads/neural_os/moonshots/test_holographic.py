"""
Comprehensive Tests for Holographic Program Representation

Tests the key claims of the holographic approach:
1. Programs can be encoded as fixed-size vectors
2. Similar programs have similar encodings
3. Pattern detection via interference works
4. Superposition preserves membership queries
5. Fourier analysis reveals structural patterns
6. Integration with SPNC/KVRM works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import List, Tuple
import unittest

from moonshots.holographic_programs import (
    HolographicConfig,
    HyperdimensionalVectorSpace,
    HolographicProgramEncoder,
    QuantumFourierAnalyzer,
    HolographicProgramSpace,
    InterferenceDiscoveryEngine,
    HolographicSearchEngine,
)


class TestHyperdimensionalVectorSpace(unittest.TestCase):
    """Test the core HDC operations."""

    def setUp(self):
        self.config = HolographicConfig(vector_dim=1000, seed=42)
        self.hd = HyperdimensionalVectorSpace(self.config)

    def test_random_vectors_nearly_orthogonal(self):
        """Random high-dim vectors should be nearly orthogonal."""
        v1 = self.hd._random_hypervector()
        v2 = self.hd._random_hypervector()

        sim = self.hd.similarity(v1, v2)

        # Should be close to 0 (orthogonal in high dimensions)
        self.assertLess(abs(sim), 0.2)

    def test_binding_is_self_inverse(self):
        """bind(bind(a, b), b) should recover a."""
        a = self.hd._random_hypervector()
        b = self.hd._random_hypervector()

        bound = self.hd.bind(a, b)
        recovered = self.hd.unbind(bound, b)

        sim = self.hd.similarity(a, recovered)
        self.assertGreater(sim, 0.99)

    def test_binding_dissimilar_to_inputs(self):
        """bind(a, b) should be dissimilar to both a and b."""
        a = self.hd._random_hypervector()
        b = self.hd._random_hypervector()

        bound = self.hd.bind(a, b)

        sim_a = self.hd.similarity(bound, a)
        sim_b = self.hd.similarity(bound, b)

        self.assertLess(abs(sim_a), 0.2)
        self.assertLess(abs(sim_b), 0.2)

    def test_bundle_preserves_membership(self):
        """Bundle should be similar to all its members."""
        vectors = [self.hd._random_hypervector() for _ in range(5)]
        bundle = self.hd.bundle(vectors)

        for v in vectors:
            sim = self.hd.similarity(bundle, v)
            self.assertGreater(sim, 0.3)

    def test_bundle_not_similar_to_non_members(self):
        """Bundle should not be similar to non-members."""
        members = [self.hd._random_hypervector() for _ in range(5)]
        non_member = self.hd._random_hypervector()

        bundle = self.hd.bundle(members)
        sim = self.hd.similarity(bundle, non_member)

        self.assertLess(abs(sim), 0.3)

    def test_permutation_creates_distinct_vectors(self):
        """Permutation should create distinct but related vectors."""
        v = self.hd._random_hypervector()

        p1 = self.hd.permute(v, shift=1)
        p2 = self.hd.permute(v, shift=2)

        # Should be distinct from original
        self.assertLess(abs(self.hd.similarity(v, p1)), 0.2)

        # Different permutations should be distinct
        self.assertLess(abs(self.hd.similarity(p1, p2)), 0.2)

    def test_inverse_permutation(self):
        """Inverse permutation should recover original."""
        v = self.hd._random_hypervector()

        permuted = self.hd.permute(v, shift=5)
        recovered = self.hd.inverse_permute(permuted, shift=5)

        sim = self.hd.similarity(v, recovered)
        self.assertGreater(sim, 0.99)


class TestHolographicProgramEncoding(unittest.TestCase):
    """Test program encoding into holograms."""

    def setUp(self):
        self.config = HolographicConfig(vector_dim=10000, seed=42)
        self.hd = HyperdimensionalVectorSpace(self.config)
        self.encoder = HolographicProgramEncoder(self.hd, self.config)

    def test_instruction_encoding_deterministic(self):
        """Same instruction should always produce same encoding."""
        enc1 = self.encoder.encode_instruction(0, 0, 1, 2)
        enc2 = self.encoder.encode_instruction(0, 0, 1, 2)

        sim = self.hd.similarity(enc1, enc2)
        self.assertGreater(sim, 0.99)

    def test_different_opcodes_different_encodings(self):
        """Different opcodes should produce different encodings."""
        add = self.encoder.encode_instruction(0, 0, 1, 2)  # ADD
        sub = self.encoder.encode_instruction(1, 0, 1, 2)  # SUB
        mul = self.encoder.encode_instruction(2, 0, 1, 2)  # MUL

        # Should be different
        self.assertLess(self.hd.similarity(add, sub), 0.9)
        self.assertLess(self.hd.similarity(add, mul), 0.9)
        self.assertLess(self.hd.similarity(sub, mul), 0.9)

    def test_program_encoding_fixed_size(self):
        """Programs of any length should produce same-size encoding."""
        short_prog = [(0, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]
        long_prog = [(0, 0, 0, 0, 0, False, 0)] * 10 + [(21, 0, 0, 0, 0, False, 0)]

        short_enc = self.encoder.encode_program(short_prog)
        long_enc = self.encoder.encode_program(long_prog)

        self.assertEqual(short_enc.shape, long_enc.shape)
        self.assertEqual(short_enc.shape[0], self.config.vector_dim)

    def test_identical_programs_identical_encodings(self):
        """Identical programs should have identical encodings."""
        prog = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]

        enc1 = self.encoder.encode_program(prog)
        enc2 = self.encoder.encode_program(prog)

        sim = self.hd.similarity(enc1, enc2)
        self.assertGreater(sim, 0.99)

    def test_similar_programs_similar_encodings(self):
        """Programs with similar structure should have similar encodings."""
        # Both compute something with MUL, but on different registers
        prog1 = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]
        prog2 = [(2, 1, 1, 1, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]

        enc1 = self.encoder.encode_program(prog1)
        enc2 = self.encoder.encode_program(prog2)

        sim = self.hd.similarity(enc1, enc2)
        # Should be somewhat similar (same opcode pattern)
        self.assertGreater(sim, 0.3)


class TestPatternDetection(unittest.TestCase):
    """Test pattern detection via interference."""

    def setUp(self):
        self.hps = HolographicProgramSpace(HolographicConfig(seed=42))

    def test_square_pattern_detected(self):
        """Square pattern should be detected in x*x programs."""
        square_prog = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]

        patterns = self.hps.detect_patterns(square_prog)

        self.assertEqual(max(patterns, key=patterns.get), 'square')

    def test_loop_pattern_detected(self):
        """Loop pattern should be detected in programs with loops."""
        loop_prog = [
            (12, 1, 0, 0, 0, True, 0),   # MOV x1, #0
            (11, 0, 0, 0, 10, True, 0),  # CMP x0, #10
            (17, 0, 0, 0, 0, False, 5),  # BGE to end
            (0, 1, 1, 0, 1, True, 0),    # ADD x1, x1, #1
            (13, 0, 0, 0, 0, False, 1),  # B to loop
            (21, 0, 0, 0, 0, False, 0),  # RET
        ]

        patterns = self.hps.detect_patterns(loop_prog)

        # Loop should be among top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        top_patterns = [p[0] for p in sorted_patterns[:3]]
        self.assertIn('loop', top_patterns)

    def test_factorial_pattern_detected(self):
        """Factorial pattern should be detected in factorial programs."""
        factorial_prog = [
            (12, 1, 0, 0, 1, True, 0),   # MOV x1, #1
            (12, 2, 0, 0, 1, True, 0),   # MOV x2, #1
            (11, 2, 2, 0, 0, False, 0),  # CMP x2, x0
            (18, 0, 0, 0, 0, False, 8),  # BGT to end
            (2, 1, 1, 2, 0, False, 0),   # MUL x1, x1, x2
            (0, 2, 2, 0, 1, True, 0),    # ADD x2, x2, #1
            (13, 0, 0, 0, 0, False, 2),  # B to loop
            (12, 0, 0, 1, 0, False, 0),  # MOV x0, x1
            (21, 0, 0, 0, 0, False, 0),  # RET
        ]

        patterns = self.hps.detect_patterns(factorial_prog)

        self.assertEqual(max(patterns, key=patterns.get), 'factorial')


class TestSuperposition(unittest.TestCase):
    """Test program superposition."""

    def setUp(self):
        self.hps = HolographicProgramSpace(HolographicConfig(seed=42))

    def test_superposition_preserves_all_members(self):
        """All bundled programs should be recoverable from superposition."""
        progs = [
            [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],  # square
            [(0, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],  # double
            [(1, 0, 0, 0, 1, True, 0), (21, 0, 0, 0, 0, False, 0)],   # subtract 1
        ]

        superposition = self.hps.superpose(progs)

        for prog in progs:
            prog_hologram = self.hps.encoder.encode_program(prog)
            sim = self.hps.hd_space.similarity(superposition, prog_hologram)
            self.assertGreater(sim, 0.5)

    def test_non_member_not_in_superposition(self):
        """Programs not in superposition should have low similarity."""
        members = [
            [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
            [(0, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
        ]

        non_member = [
            (3, 0, 0, 0, 2, True, 0),  # divide by 2
            (21, 0, 0, 0, 0, False, 0),
        ]

        superposition = self.hps.superpose(members)
        non_member_hologram = self.hps.encoder.encode_program(non_member)

        sim = self.hps.hd_space.similarity(superposition, non_member_hologram)

        # Should be lower than members
        for member in members:
            member_hologram = self.hps.encoder.encode_program(member)
            member_sim = self.hps.hd_space.similarity(superposition, member_hologram)
            self.assertLess(sim, member_sim)


class TestFourierAnalysis(unittest.TestCase):
    """Test quantum-inspired Fourier analysis."""

    def setUp(self):
        self.config = HolographicConfig(seed=42)
        self.fourier = QuantumFourierAnalyzer(self.config)
        self.hps = HolographicProgramSpace(self.config)

    def test_analysis_returns_expected_keys(self):
        """Fourier analysis should return all expected metrics."""
        prog = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]
        hologram = self.hps.encoder.encode_program(prog)

        analysis = self.fourier.analyze(hologram)

        expected_keys = ['spectrum', 'phases', 'band_energies', 'total_energy',
                        'peak_frequency', 'spectral_entropy', 'pattern_score']
        for key in expected_keys:
            self.assertIn(key, analysis)

    def test_spectral_entropy_bounded(self):
        """Spectral entropy should be between 0 and 1."""
        progs = [
            [(21, 0, 0, 0, 0, False, 0)],  # trivial
            [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],  # simple
            [(0, i, i, i, 0, False, 0) for i in range(10)] + [(21, 0, 0, 0, 0, False, 0)],  # complex
        ]

        for prog in progs:
            hologram = self.hps.encoder.encode_program(prog)
            analysis = self.fourier.analyze(hologram)

            self.assertGreaterEqual(analysis['spectral_entropy'], 0)
            self.assertLessEqual(analysis['spectral_entropy'], 1)

    def test_different_structures_different_spectra(self):
        """Programs with different structure should have different spectra."""
        simple = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]

        complex_prog = [
            (12, 1, 0, 0, 1, True, 0),
            (12, 2, 0, 0, 1, True, 0),
            (11, 2, 2, 0, 0, False, 0),
            (18, 0, 0, 0, 0, False, 8),
            (2, 1, 1, 2, 0, False, 0),
            (0, 2, 2, 0, 1, True, 0),
            (13, 0, 0, 0, 0, False, 2),
            (12, 0, 0, 1, 0, False, 0),
            (21, 0, 0, 0, 0, False, 0),
        ]

        simple_holo = self.hps.encoder.encode_program(simple)
        complex_holo = self.hps.encoder.encode_program(complex_prog)

        simple_analysis = self.fourier.analyze(simple_holo)
        complex_analysis = self.fourier.analyze(complex_holo)

        # Different peak frequencies
        self.assertNotEqual(simple_analysis['peak_frequency'],
                           complex_analysis['peak_frequency'])


class TestDiscoveryEngine(unittest.TestCase):
    """Test interference-based discovery engine."""

    def setUp(self):
        self.hps = HolographicProgramSpace(HolographicConfig(seed=42))
        self.discovery = InterferenceDiscoveryEngine(self.hps)

    def test_discover_from_examples(self):
        """Discovery should work with example programs."""
        examples = [
            [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
            [(12, 1, 0, 0, 0, False, 0), (2, 0, 1, 1, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
        ]

        result = self.discovery.discover_from_examples(examples, "test_task")

        self.assertTrue(result['success'])
        self.assertEqual(result['num_examples'], 2)
        self.assertIn('superposition', result)
        self.assertIn('pattern_scores', result)

    def test_find_hidden_algorithms_square(self):
        """Should detect square pattern from I/O pairs."""
        io_pairs = [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)]

        result = self.discovery.find_hidden_algorithms(io_pairs)

        self.assertIn('hypotheses', result)
        # Should identify square pattern
        hypotheses = [h[0] for h in result['hypotheses']]
        self.assertIn('square', hypotheses)

    def test_find_hidden_algorithms_identity(self):
        """Should detect identity pattern from I/O pairs."""
        io_pairs = [(0, 0), (1, 1), (5, 5), (100, 100)]

        result = self.discovery.find_hidden_algorithms(io_pairs)

        hypotheses = [h[0] for h in result['hypotheses']]
        self.assertIn('identity', hypotheses)


class TestSearchEngine(unittest.TestCase):
    """Test holographic search engine."""

    def setUp(self):
        self.hps = HolographicProgramSpace(HolographicConfig(seed=42))
        self.search = HolographicSearchEngine(self.hps)

        # Populate memory
        self.hps.store([(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
                      {'name': 'square'})
        self.hps.store([(0, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
                      {'name': 'double'})
        self.hps.store([(21, 0, 0, 0, 0, False, 0)],
                      {'name': 'identity'})

    def test_search_by_pattern(self):
        """Search by pattern should return relevant programs."""
        results = self.search.search_by_pattern('square', top_k=3)

        self.assertGreater(len(results), 0)
        # First result should be the square program
        first_idx = results[0][0]
        self.assertEqual(self.hps.memory_metadata[first_idx]['name'], 'square')

    def test_search_by_interference(self):
        """Search by interference should work with positive examples."""
        positive = [[(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]]

        results = self.search.search_by_interference(positive, top_k=3)

        self.assertGreater(len(results), 0)
        # Should find the square program
        first_idx = results[0][0]
        self.assertEqual(self.hps.memory_metadata[first_idx]['name'], 'square')

    def test_query_similar(self):
        """Similar query should return stored programs."""
        query = [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)]

        results = self.hps.query_similar(query, top_k=3)

        self.assertEqual(len(results), 3)
        # First should be exact match
        self.assertGreater(results[0][1], 0.99)


class TestIntegration(unittest.TestCase):
    """Integration tests with SPNC components."""

    def test_full_workflow(self):
        """Test complete workflow: store, analyze, discover, search."""
        hps = HolographicProgramSpace(HolographicConfig(seed=42))
        discovery = InterferenceDiscoveryEngine(hps)
        search = HolographicSearchEngine(hps)

        # 1. Store programs
        programs = [
            ([(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)], 'square'),
            ([(0, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)], 'double'),
            ([(1, 0, 0, 0, 1, True, 0), (21, 0, 0, 0, 0, False, 0)], 'decrement'),
        ]

        for prog, name in programs:
            hps.store(prog, {'name': name, 'program': prog})

        # 2. Analyze structure
        for prog, name in programs:
            analysis = hps.analyze_structure(prog)
            self.assertIn('dominant_pattern', analysis)

        # 3. Discover from examples
        square_variants = [
            [(2, 0, 0, 0, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
            [(12, 1, 0, 0, 0, False, 0), (2, 0, 1, 1, 0, False, 0), (21, 0, 0, 0, 0, False, 0)],
        ]
        discovery_result = discovery.discover_from_examples(square_variants, 'square')
        self.assertTrue(discovery_result['success'])

        # 4. Search for similar
        results = hps.query_similar(programs[0][0], top_k=3)
        self.assertEqual(len(results), 3)

        # 5. Search by pattern
        pattern_results = search.search_by_pattern('square')
        self.assertGreater(len(pattern_results), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHyperdimensionalVectorSpace))
    suite.addTests(loader.loadTestsFromTestCase(TestHolographicProgramEncoding))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestSuperposition))
    suite.addTests(loader.loadTestsFromTestCase(TestFourierAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestDiscoveryEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
