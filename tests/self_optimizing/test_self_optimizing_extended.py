"""
Expanded Test Suite for SOME

Comprehensive tests for all components.
"""

import unittest
import numpy as np
from ncpu.self_optimizing import (
    # Core
    SelfOptimizingEngine,
    Task,
    ExecutionResult,
    # Gradient feedback
    GradientFeedbackSystem,
    ExecutionSignal,
    FeedbackType,
    ImprovementDirection,
    # Model integration
    TensorDescriptor,
    ModelInterface,
    ExecutionVerifiedModel,
    # Autoresearch
    AutoresearchSOMEAgent,
    ExperimentCandidate,
    ExperimentResult,
    GradientFeedback,
    # Experiment runner
    SOMExperimentRunner,
    ExperimentConfig,
    Experiment,
    ModelWeights,
    # Gradient search
    GradientGuidedSearch,
    MultiObjectiveSearch,
    SearchSuggestion,
    SearchPattern,
    SearchDirection,
)


# =============================================================================
# CORE ENGINE TESTS
# =============================================================================

class TestSelfOptimizingEngineExtended(unittest.TestCase):
    """Extended tests for SelfOptimizingEngine"""

    def test_engine_initialization(self):
        """Test engine can be initialized with custom params"""
        engine = SelfOptimizingEngine(num_workers=8, max_iterations=50)
        self.assertEqual(engine.workers.num_workers, 8)
        self.assertEqual(engine.max_iterations, 50)

    def test_generate_creates_descriptor(self):
        """Test generate returns proper descriptor"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        task = Task(description="test")
        desc = engine.generate(task)

        self.assertIn("operation", desc)
        self.assertIn("task", desc)

        engine.shutdown()

    def test_execute_returns_execution_result(self):
        """Test execute returns proper result type"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        result = engine.execute({"operation": "test"})

        self.assertIsInstance(result, ExecutionResult)
        self.assertIn("success", result.__dict__)

        engine.shutdown()

    def test_verify_with_custom_function(self):
        """Test verify with custom verification function"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        task = Task(
            description="test",
            verification_fn=lambda x: x.get("computed") == True
        )

        result = ExecutionResult(success=True, output={"computed": True})
        self.assertTrue(engine.verify(task, result))

        engine.shutdown()


# =============================================================================
# GRADIENT FEEDBACK TESTS
# =============================================================================

class TestGradientFeedbackExtended(unittest.TestCase):
    """Extended tests for gradient feedback system"""

    def test_capture_execution_with_expected(self):
        """Test capturing execution with expected result"""
        gfs = GradientFeedbackSystem()

        signal = gfs.capture_execution(42, expected_result=42)
        self.assertEqual(signal.feedback_type, FeedbackType.SUCCESS)
        self.assertEqual(signal.improvement_direction, ImprovementDirection.POSITIVE)

    def test_capture_execution_mismatch(self):
        """Test capturing execution with wrong expected"""
        gfs = GradientFeedbackSystem()

        signal = gfs.capture_execution(100, expected_result=42)
        self.assertEqual(signal.feedback_type, FeedbackType.CORRECTNESS_ERROR)
        self.assertEqual(signal.improvement_direction, ImprovementDirection.NEGATIVE)

    def test_capture_error_with_context(self):
        """Test capturing error with context"""
        gfs = GradientFeedbackSystem()

        try:
            raise ValueError("test error")
        except ValueError as e:
            signal = gfs.capture_from_error(e, {"task": "test"})

        self.assertIsNotNone(signal)

    def test_pattern_recording_multiple(self):
        """Test recording multiple patterns"""
        gfs = GradientFeedbackSystem()

        gfs.record_pattern("hash1", "loop", success=True)
        gfs.record_pattern("hash1", "loop", success=True)
        gfs.record_pattern("hash1", "loop", success=False)
        gfs.record_pattern("hash1", "loop", success=True)

        # 3 successes, 1 failure = 75%
        self.assertAlmostEqual(gfs.patterns["hash1"].success_rate, 0.75, places=2)

    def test_get_improvement_suggestion(self):
        """Test getting improvement suggestions"""
        gfs = GradientFeedbackSystem()

        # No feedback yet
        suggestion = gfs.get_improvement_suggestion()
        self.assertEqual(suggestion, "No feedback yet")

        # Add some failures
        gfs.capture_execution(None, expected_result=42)
        gfs.capture_execution(None, expected_result=42)
        gfs.capture_execution(None, expected_result=42)

        suggestion = gfs.get_improvement_suggestion()
        self.assertIsInstance(suggestion, str)


# =============================================================================
# MODEL INTEGRATION TESTS
# =============================================================================

class TestModelInterfaceExtended(unittest.TestCase):
    """Extended tests for model interface"""

    def test_generate_multiple_candidates(self):
        """Test generating multiple candidates"""
        interface = ModelInterface()

        candidates = interface.generate_descriptors("test", num_candidates=5)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(interface.generation_history), 5)

    def test_apply_different_feedback_directions(self):
        """Test applying different feedback directions"""
        interface = ModelInterface()

        # Positive
        signal = ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS,
            improvement_direction=ImprovementDirection.POSITIVE,
        )
        interface.apply_gradient(signal)

        # Negative
        signal = ExecutionSignal(
            feedback_type=FeedbackType.FAILURE,
            improvement_direction=ImprovementDirection.NEGATIVE,
        )
        interface.apply_gradient(signal)

        # Refinement
        signal = ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS,
            improvement_direction=ImprovementDirection.REFINEMENT,
        )
        interface.apply_gradient(signal)

        self.assertEqual(len(interface.gfs.improvement_history), 3)


class TestExecutionVerifiedModelExtended(unittest.TestCase):
    """Extended tests for verified model"""

    def test_predict_with_retries(self):
        """Test prediction with multiple retries"""
        model = ExecutionVerifiedModel(max_retries=5)

        result = model.predict("test")
        self.assertIsNotNone(result)

    def test_predict_with_verification_function(self):
        """Test with custom verification"""
        model = ExecutionVerifiedModel(max_retries=3)

        result = model.predict(
            "test",
            verify_fn=lambda x: x is not None and x.get("executed") == True
        )
        self.assertIsNotNone(result)


# =============================================================================
# AUTORESEARCH INTEGRATION TESTS
# =============================================================================

class TestAutoresearchExtended(unittest.TestCase):
    """Extended tests for Autoresearch integration"""

    def test_experiment_candidate_with_priority(self):
        """Test candidate with priority"""
        candidate = ExperimentCandidate(
            candidate_id=1,
            description="test",
            code_diff="diff",
            expected_change="change",
            priority=0.8,
        )
        self.assertEqual(candidate.priority, 0.8)

    def test_experiment_result_with_error(self):
        """Test result with error"""
        result = ExperimentResult(
            candidate_id=1,
            val_bpb=2.5,
            training_seconds=300,
            peak_vram_mb=2048,
            mfu_percent=0.25,
            total_tokens_M=10,
            num_steps=160,
            status="crash",
            error="Out of memory",
        )
        self.assertEqual(result.error, "Out of memory")

    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = AutoresearchSOMEAgent(num_workers=4, experiment_minutes=5)
        self.assertEqual(agent.num_workers, 4)
        self.assertEqual(agent.experiment_minutes, 5)


# =============================================================================
# GRADIENT SEARCH TESTS
# =============================================================================

class TestGradientSearchExtended(unittest.TestCase):
    """Extended tests for gradient search"""

    def test_search_pattern_update(self):
        """Test pattern success rate updates"""
        search = GradientGuidedSearch()

        # Add successful experiments
        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.95,
            status="keep"
        )
        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.98,
            status="keep"
        )

        pattern = search.patterns.get("layer_increase")
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.success_rate, 1.0)

    def test_gradient_signal_shape(self):
        """Test gradient signal has correct shape"""
        search = GradientGuidedSearch()

        # Add some experiments
        for i in range(5):
            search.learn_from_result(
                {"type": f"type_{i}", "direction": "increase"},
                val_bpb=2.0 - i * 0.1,
                status="keep"
            )

        gradient = search.get_gradient_signal()
        self.assertEqual(gradient.shape, (10,))

    def test_search_statistics_complete(self):
        """Test statistics are complete"""
        search = GradientGuidedSearch()

        search.learn_from_result(
            {"type": "layer", "direction": "increase"},
            val_bpb=1.95,
            status="keep"
        )

        stats = search.get_statistics()

        self.assertIn("total_experiments", stats)
        self.assertIn("successful_experiments", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("best_val_bpb", stats)


class TestMultiObjectiveSearchExtended(unittest.TestCase):
    """Extended tests for multi-objective search"""

    def test_multi_objective_weights(self):
        """Test multi-objective weights"""
        search = MultiObjectiveSearch()

        self.assertIn("val_bpb", search.weights)
        self.assertIn("speed", search.weights)
        self.assertIn("memory", search.weights)

    def test_learn_with_all_metrics(self):
        """Test learning with all metrics"""
        search = MultiObjectiveSearch()

        search.learn_from_result(
            {"type": "test"},
            val_bpb=1.95,
            training_seconds=250,
            peak_vram_mb=2048,
            status="keep",
        )

        self.assertEqual(search.total_experiments, 1)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""

    def test_engine_to_feedback_pipeline(self):
        """Test full pipeline from engine to feedback"""
        # Create engine
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        # Create task
        task = Task(
            description="test task",
            verification_fn=lambda x: x.get("computed") == True
        )

        # Execute
        descriptor = engine.generate(task)
        result = engine.execute(descriptor)

        # Verify
        verified = engine.verify(task, result)
        self.assertTrue(verified)

        # Capture feedback
        feedback = engine.capture_feedback(result)
        self.assertIsInstance(feedback, dict)

        engine.shutdown()

    def test_model_interface_to_gradient_search(self):
        """Test pipeline from model to gradient search"""
        # Create model interface
        interface = ModelInterface()

        # Generate candidates
        candidates = interface.generate_descriptors("test", num_candidates=3)

        # Learn from results
        for i, c in enumerate(candidates):
            signal = ExecutionSignal(
                feedback_type=FeedbackType.SUCCESS if i < 2 else FeedbackType.FAILURE,
                improvement_direction=(
                    ImprovementDirection.POSITIVE if i < 2
                    else ImprovementDirection.NEGATIVE
                ),
            )
            interface.apply_gradient(signal)

        # Create gradient search
        search = GradientGuidedSearch()

        # Feed interface history into search
        for imp in interface.gfs.improvement_history:
            search.learn_from_result(
                imp,
                val_bpb=2.0,
                status="keep" if imp["direction"] == "positive" else "discard"
            )

        stats = search.get_statistics()
        self.assertGreater(stats["total_experiments"], 0)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases"""

    def test_empty_task(self):
        """Test with empty task"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        task = Task(description="")
        descriptor = engine.generate(task)
        self.assertIsNotNone(descriptor)

        engine.shutdown()

    def test_zero_max_iterations(self):
        """Test with zero max iterations"""
        engine = SelfOptimizingEngine(max_iterations=0)
        result = engine.run(Task(description="test"))

        # Should return failure after 0 iterations
        self.assertFalse(result.success)

    def test_large_number_of_patterns(self):
        """Test with many patterns"""
        gfs = GradientFeedbackSystem()

        for i in range(100):
            gfs.record_pattern(f"hash_{i}", f"type_{i}", success=(i % 2 == 0))

        self.assertEqual(len(gfs.patterns), 100)

    def test_gradient_signal_with_no_history(self):
        """Test gradient signal with no history"""
        search = GradientGuidedSearch()
        gradient = search.get_gradient_signal()

        # Should return zeros
        self.assertTrue(np.allclose(gradient, 0))


if __name__ == "__main__":
    unittest.main()
