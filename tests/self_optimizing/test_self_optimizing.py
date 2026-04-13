"""Tests for Self-Optimizing Machine Engine"""

import unittest
from ncpu.self_optimizing import (
    SelfOptimizingEngine,
    Task,
    ExecutionResult,
    GradientFeedbackSystem,
    ExecutionSignal,
    FeedbackType,
    ImprovementDirection,
    TensorDescriptor,
    ModelInterface,
    ExecutionVerifiedModel,
)


class TestTask(unittest.TestCase):
    """Test Task dataclass"""

    def test_create_task(self):
        """Test creating a task"""
        task = Task(description="Calculate fibonacci(100)")
        self.assertEqual(task.description, "Calculate fibonacci(100)")
        self.assertIsNone(task.expected_output)

    def test_task_with_verification(self):
        """Test task with verification function"""
        task = Task(
            description="Calculate fibonacci(100)",
            verification_fn=lambda x: x.get("result") == 42
        )
        self.assertIsNotNone(task.verification_fn)


class TestExecutionResult(unittest.TestCase):
    """Test ExecutionResult"""

    def test_create_result(self):
        """Test creating execution result"""
        result = ExecutionResult(
            success=True,
            output={"result": 42},
            execution_time_ms=1.5,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output["result"], 42)


class TestSelfOptimizingEngine(unittest.TestCase):
    """Test SelfOptimizingEngine"""

    def test_create_engine(self):
        """Test creating engine"""
        engine = SelfOptimizingEngine(num_workers=2)
        self.assertEqual(engine.max_iterations, 100)

    def test_initialize_shutdown(self):
        """Test initialize and shutdown"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()
        self.assertTrue(engine._initialized)
        engine.shutdown()
        self.assertFalse(engine._initialized)

    def test_generate(self):
        """Test generate creates descriptor"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        task = Task(description="test task")
        descriptor = engine.generate(task)

        self.assertIn("operation", descriptor)
        self.assertEqual(descriptor["task"], "test task")

        engine.shutdown()

    def test_execute_returns_result(self):
        """Test execute returns ExecutionResult"""
        engine = SelfOptimizingEngine(num_workers=2)
        engine.initialize()

        descriptor = {"operation": "test", "input": {}}
        result = engine.execute(descriptor)

        self.assertIsInstance(result, ExecutionResult)

        engine.shutdown()


class TestGradientFeedbackSystem(unittest.TestCase):
    """Test GradientFeedbackSystem"""

    def test_create_system(self):
        """Test creating gradient feedback system"""
        gfs = GradientFeedbackSystem()
        self.assertEqual(len(gfs.signals), 0)

    def test_capture_success(self):
        """Test capturing successful execution"""
        gfs = GradientFeedbackSystem()
        signal = gfs.capture_execution(42, expected_result=42)

        self.assertEqual(signal.feedback_type, FeedbackType.SUCCESS)
        self.assertEqual(signal.improvement_direction, ImprovementDirection.POSITIVE)

    def test_capture_failure(self):
        """Test capturing failed execution"""
        gfs = GradientFeedbackSystem()
        signal = gfs.capture_execution(100, expected_result=42)

        self.assertEqual(signal.feedback_type, FeedbackType.CORRECTNESS_ERROR)
        self.assertEqual(signal.improvement_direction, ImprovementDirection.NEGATIVE)

    def test_compute_gradient(self):
        """Test computing gradient from signals"""
        gfs = GradientFeedbackSystem()
        gfs.capture_execution(42, expected_result=42)
        gfs.capture_execution(42, expected_result=42)

        gradient = gfs.compute_gradient()
        self.assertEqual(len(gradient), 10)

    def test_pattern_recording(self):
        """Test recording code patterns"""
        gfs = GradientFeedbackSystem()
        gfs.record_pattern("hash123", "loop", success=True)
        gfs.record_pattern("hash123", "loop", success=True)
        gfs.record_pattern("hash123", "loop", success=False)

        self.assertIn("hash123", gfs.patterns)
        self.assertAlmostEqual(gfs.patterns["hash123"].success_rate, 0.666, places=2)


class TestExecutionSignal(unittest.TestCase):
    """Test ExecutionSignal"""

    def test_create_signal(self):
        """Test creating execution signal"""
        signal = ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS,
            improvement_direction=ImprovementDirection.POSITIVE,
        )
        self.assertEqual(signal.feedback_type, FeedbackType.SUCCESS)

    def test_to_gradient(self):
        """Test converting signal to gradient"""
        signal = ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS,
            improvement_direction=ImprovementDirection.POSITIVE,
            execution_time_ms=10.0,
            correctness_score=1.0,
        )

        gradient = signal.to_gradient()
        self.assertEqual(len(gradient), 10)


class TestTensorDescriptor(unittest.TestCase):
    """Test TensorDescriptor"""

    def test_create_descriptor(self):
        """Test creating tensor descriptor"""
        desc = TensorDescriptor(
            operation="matmul",
            inputs={"a": [1, 2], "b": [3, 4]},
            outputs={"c": None},
            shape=(2, 2),
        )
        self.assertEqual(desc.operation, "matmul")

    def test_serialization(self):
        """Test to_bytes and from_bytes"""
        desc = TensorDescriptor(
            operation="test",
            inputs={"x": 1},
            outputs={"y": 2},
        )

        data = desc.to_bytes()
        self.assertIsInstance(data, bytes)

        restored = TensorDescriptor.from_bytes(data)
        self.assertEqual(restored.operation, "test")


class TestModelInterface(unittest.TestCase):
    """Test ModelInterface"""

    def test_create_interface(self):
        """Test creating model interface"""
        interface = ModelInterface()
        self.assertIsNotNone(interface.gfs)

    def test_generate_descriptors(self):
        """Test generating descriptors"""
        interface = ModelInterface()
        candidates = interface.generate_descriptors("test query", num_candidates=3)

        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(interface.generation_history), 3)

    def test_apply_gradient(self):
        """Test applying gradient"""
        interface = ModelInterface()

        signal = ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS,
            improvement_direction=ImprovementDirection.POSITIVE,
        )
        interface.apply_gradient(signal)

        self.assertEqual(len(interface.gfs.improvement_history), 1)


class TestExecutionVerifiedModel(unittest.TestCase):
    """Test ExecutionVerifiedModel"""

    def test_create_model(self):
        """Test creating verified model"""
        model = ExecutionVerifiedModel(max_retries=2)
        self.assertEqual(model.max_retries, 2)

    def test_predict_success(self):
        """Test successful prediction"""
        model = ExecutionVerifiedModel()
        result = model.predict("test input")

        self.assertIsNotNone(result)

    def test_predict_with_verification(self):
        """Test prediction with custom verification"""
        model = ExecutionVerifiedModel()

        # Should succeed with verification
        result = model.predict(
            "test",
            verify_fn=lambda x: x is not None
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
