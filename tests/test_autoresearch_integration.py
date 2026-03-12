"""Tests for Autoresearch Integration"""

import unittest
import numpy as np
from ncpu.self_optimizing import (
    AutoresearchSOMEAgent,
    ExperimentCandidate,
    ExperimentResult,
    GradientGuidedSearch,
    MultiObjectiveSearch,
    SearchSuggestion,
    SearchPattern,
    SearchDirection,
)


class TestExperimentCandidate(unittest.TestCase):
    """Test ExperimentCandidate"""

    def test_create_candidate(self):
        """Test creating experiment candidate"""
        candidate = ExperimentCandidate(
            candidate_id=1,
            description="Increase layers",
            code_diff="# Increase n_layer",
            expected_change="Better val_bpb",
        )
        self.assertEqual(candidate.candidate_id, 1)
        self.assertEqual(candidate.description, "Increase layers")


class TestExperimentResult(unittest.TestCase):
    """Test ExperimentResult"""

    def test_create_result(self):
        """Test creating experiment result"""
        result = ExperimentResult(
            candidate_id=1,
            val_bpb=1.95,
            training_seconds=300.0,
            peak_vram_mb=2048.0,
            mfu_percent=0.25,
            total_tokens_M=10.5,
            num_steps=160,
            status="keep",
        )
        self.assertEqual(result.val_bpb, 1.95)
        self.assertEqual(result.status, "keep")


class TestGradientGuidedSearch(unittest.TestCase):
    """Test GradientGuidedSearch"""

    def test_create_search(self):
        """Test creating search"""
        search = GradientGuidedSearch()
        self.assertEqual(search.total_experiments, 0)

    def test_learn_from_result(self):
        """Test learning from results"""
        search = GradientGuidedSearch()

        # Successful experiment
        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.95,
            status="keep",
        )

        self.assertEqual(search.total_experiments, 1)
        self.assertEqual(search.successful_experiments, 1)

    def test_get_suggestion(self):
        """Test getting suggestion"""
        search = GradientGuidedSearch()
        suggestion = search.get_suggestion()

        self.assertIsInstance(suggestion, SearchSuggestion)

    def test_gradient_signal(self):
        """Test gradient signal generation"""
        search = GradientGuidedSearch()

        # Add some experiments
        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.95,
            status="keep",
        )

        gradient = search.get_gradient_signal()
        self.assertEqual(len(gradient), 10)

    def test_statistics(self):
        """Test statistics"""
        search = GradientGuidedSearch()

        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.95,
            status="keep",
        )

        stats = search.get_statistics()
        self.assertIn("total_experiments", stats)


class TestSearchSuggestion(unittest.TestCase):
    """Test SearchSuggestion"""

    def test_create_suggestion(self):
        """Test creating suggestion"""
        suggestion = SearchSuggestion(
            modification_type="layer",
            target="n_layer",
            direction=SearchDirection.INCREASE,
            magnitude=1,
            reason="Test",
            expected_improvement=0.1,
        )
        self.assertEqual(suggestion.modification_type, "layer")


class TestSearchPattern(unittest.TestCase):
    """Test SearchPattern"""

    def test_create_pattern(self):
        """Test creating pattern"""
        pattern = SearchPattern(
            pattern_id="layer_increase",
            modification_type="layer",
            direction=SearchDirection.INCREASE,
            magnitude=1.0,
            success_rate=0.75,
            count=4,
        )
        self.assertEqual(pattern.success_rate, 0.75)


class TestMultiObjectiveSearch(unittest.TestCase):
    """Test MultiObjectiveSearch"""

    def test_create_search(self):
        """Test creating multi-objective search"""
        search = MultiObjectiveSearch()
        self.assertIn("val_bpb", search.weights)

    def test_learn_multiobjective(self):
        """Test learning with multiple objectives"""
        search = MultiObjectiveSearch()

        search.learn_from_result(
            modification={"type": "layer"},
            val_bpb=1.95,
            training_seconds=280.0,
            peak_vram_mb=2048.0,
            status="keep",
        )

        self.assertEqual(search.total_experiments, 1)


if __name__ == "__main__":
    unittest.main()
