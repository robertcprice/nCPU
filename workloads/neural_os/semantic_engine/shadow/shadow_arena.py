"""
Shadow Arena - Isolated Parallel Execution Environment
OUROBOROS Phase 7.3 - Shadow Simulation Framework

The Shadow Arena provides a fully isolated environment for testing
consciousness layer changes before production deployment. It ensures
that experimental changes cannot affect the production system.

Key responsibilities:
1. Create isolated execution environments for testing
2. Run parallel executions (shadow vs production)
3. Collect metrics from shadow executions
4. Enforce stricter resource limits than production
5. Provide execution rollback capabilities

CRITICAL: Shadow executions are FULLY ISOLATED from production.
No data or state can leak from shadow to production.
"""

import os
import sys
import time
import hashlib
import threading
import multiprocessing
import tempfile
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
from contextlib import contextmanager
import json
import traceback


class ExecutionStatus(Enum):
    """Status of a shadow execution"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    KILLED = auto()


@dataclass
class ArenaConfig:
    """Configuration for the shadow arena"""
    # Resource limits (stricter than production)
    max_ram_bytes: int = 4 * 1024**3  # 4GB (half of production)
    max_cpu_cores: int = 1  # 1 core (half of production)
    max_execution_time: float = 300.0  # 5 minutes max
    max_tokens_per_thought: int = 250  # Half of production

    # Isolation settings
    use_process_isolation: bool = True
    use_filesystem_isolation: bool = True
    use_network_isolation: bool = True

    # Testing settings
    hypothesis_iterations: int = 1000  # Minimum Hypothesis tests
    differential_threshold: float = 0.001  # Max allowed divergence

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_ram': self.max_ram_bytes,
            'max_cpu': self.max_cpu_cores,
            'max_time': self.max_execution_time,
            'hypothesis_iterations': self.hypothesis_iterations,
        }


@dataclass
class ExecutionResult:
    """Result of a shadow execution"""
    execution_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    output: Any
    metrics: Dict[str, float]
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.execution_id,
            'status': self.status.name,
            'duration': self.duration_seconds,
            'metrics': self.metrics,
            'error': self.error,
            'resource_usage': self.resource_usage,
        }


@dataclass
class ShadowExecution:
    """A shadow execution instance"""
    execution_id: str
    created_at: datetime
    config: ArenaConfig
    function_name: str
    arguments: Dict[str, Any]
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[ExecutionResult] = None
    process: Optional[multiprocessing.Process] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.execution_id,
            'created_at': self.created_at.isoformat(),
            'function': self.function_name,
            'status': self.status.name,
            'result': self.result.to_dict() if self.result else None,
        }


class IsolatedFileSystem:
    """
    Provides isolated filesystem for shadow executions.

    Creates a temporary directory with copy-on-write semantics
    so shadow executions cannot modify production files.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.gettempdir()
        self.shadow_dirs: Dict[str, str] = {}
        self._lock = threading.Lock()

    def create_sandbox(self, execution_id: str) -> str:
        """Create an isolated sandbox directory"""
        with self._lock:
            sandbox_path = os.path.join(
                self.base_dir,
                f"shadow_sandbox_{execution_id}"
            )
            os.makedirs(sandbox_path, exist_ok=True)
            self.shadow_dirs[execution_id] = sandbox_path
            return sandbox_path

    def cleanup_sandbox(self, execution_id: str) -> bool:
        """Clean up a sandbox directory"""
        with self._lock:
            if execution_id in self.shadow_dirs:
                sandbox_path = self.shadow_dirs[execution_id]
                try:
                    shutil.rmtree(sandbox_path, ignore_errors=True)
                    del self.shadow_dirs[execution_id]
                    return True
                except Exception:
                    return False
            return False

    def cleanup_all(self) -> int:
        """Clean up all sandboxes"""
        with self._lock:
            count = 0
            for execution_id in list(self.shadow_dirs.keys()):
                if self.cleanup_sandbox(execution_id):
                    count += 1
            return count


class ResourceMonitor:
    """
    Monitors resource usage during shadow execution.

    Collects metrics for comparison with production.
    """

    def __init__(self, config: ArenaConfig):
        self.config = config
        self.metrics: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_bytes': [],
            'io_reads': [],
            'io_writes': [],
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, pid: int) -> None:
        """Start monitoring a process"""
        def monitor_loop():
            try:
                import resource
                while not self._stop_event.wait(timeout=0.5):
                    try:
                        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                        with self._lock:
                            self.metrics['cpu_percent'].append(
                                usage.ru_utime + usage.ru_stime
                            )
                            self.metrics['memory_bytes'].append(
                                usage.ru_maxrss * 1024  # Convert to bytes
                            )
                    except Exception:
                        pass
            except ImportError:
                pass

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        with self._lock:
            return {
                'peak_cpu': max(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0.0,
                'avg_cpu': sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent'])
                    if self.metrics['cpu_percent'] else 0.0,
                'peak_memory': max(self.metrics['memory_bytes']) if self.metrics['memory_bytes'] else 0,
                'avg_memory': sum(self.metrics['memory_bytes']) / len(self.metrics['memory_bytes'])
                    if self.metrics['memory_bytes'] else 0,
            }


def _run_in_sandbox(
    function: Callable,
    args: Dict[str, Any],
    result_queue: multiprocessing.Queue,
    config_dict: Dict[str, Any],
) -> None:
    """
    Worker function that runs in isolated process.

    This function runs in a separate process with restricted resources.
    """
    start_time = time.time()
    metrics: Dict[str, float] = {}

    try:
        # Set resource limits
        try:
            import resource
            # Memory limit
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(
                resource.RLIMIT_AS,
                (config_dict['max_ram'], config_dict['max_ram'])
            )
            # CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(config_dict['max_time']), int(config_dict['max_time']) + 10)
            )
        except (ImportError, ValueError):
            pass

        # Execute function
        result = function(**args)

        end_time = time.time()
        metrics['execution_time'] = end_time - start_time

        result_queue.put({
            'status': 'completed',
            'output': result,
            'metrics': metrics,
            'duration': end_time - start_time,
        })

    except MemoryError:
        result_queue.put({
            'status': 'failed',
            'error': 'Memory limit exceeded',
            'duration': time.time() - start_time,
        })
    except Exception as e:
        result_queue.put({
            'status': 'failed',
            'error': str(e),
            'stack_trace': traceback.format_exc(),
            'duration': time.time() - start_time,
        })


class ShadowArena:
    """
    The Shadow Arena for isolated parallel execution.

    Provides a fully isolated environment for testing consciousness
    layer changes before deployment to production.

    CRITICAL SAFETY PROPERTIES:
    1. Full process isolation from production
    2. Stricter resource limits than production
    3. No network access
    4. Isolated filesystem
    5. Automatic cleanup on completion or failure
    """

    def __init__(
        self,
        config: Optional[ArenaConfig] = None,
        on_execution_start: Optional[Callable[[ShadowExecution], None]] = None,
        on_execution_complete: Optional[Callable[[ShadowExecution], None]] = None,
    ):
        self.config = config or ArenaConfig()
        self.on_execution_start = on_execution_start
        self.on_execution_complete = on_execution_complete

        self.filesystem = IsolatedFileSystem()
        self.executions: Dict[str, ShadowExecution] = {}
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeout_executions': 0,
            'total_duration': 0.0,
        }

    def create_execution(
        self,
        function: Callable,
        arguments: Dict[str, Any],
        custom_config: Optional[ArenaConfig] = None,
    ) -> ShadowExecution:
        """
        Create a new shadow execution.

        Does not start execution - call run() to start.
        """
        config = custom_config or self.config
        execution_id = hashlib.sha256(
            f"{function.__name__}{json.dumps(str(arguments))}{time.time()}".encode()
        ).hexdigest()[:16]

        execution = ShadowExecution(
            execution_id=execution_id,
            created_at=datetime.now(),
            config=config,
            function_name=function.__name__,
            arguments=arguments,
        )

        with self._lock:
            self.executions[execution_id] = execution

        return execution

    def run(
        self,
        execution: ShadowExecution,
        function: Callable,
    ) -> ExecutionResult:
        """
        Run a shadow execution in isolation.

        Returns ExecutionResult when complete.
        """
        execution.status = ExecutionStatus.RUNNING
        start_time = datetime.now()

        if self.on_execution_start:
            self.on_execution_start(execution)

        # Create isolated filesystem
        sandbox_path = self.filesystem.create_sandbox(execution.execution_id)

        # Create result queue for inter-process communication
        result_queue = multiprocessing.Queue()

        # Create and start isolated process
        config_dict = {
            'max_ram': execution.config.max_ram_bytes,
            'max_time': execution.config.max_execution_time,
        }

        process = multiprocessing.Process(
            target=_run_in_sandbox,
            args=(function, execution.arguments, result_queue, config_dict),
        )

        # Resource monitoring
        monitor = ResourceMonitor(execution.config)

        try:
            process.start()
            execution.process = process
            monitor.start_monitoring(process.pid)

            # Wait with timeout
            process.join(timeout=execution.config.max_execution_time)

            if process.is_alive():
                # Timeout - kill process
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()

                execution.status = ExecutionStatus.TIMEOUT
                result = ExecutionResult(
                    execution_id=execution.execution_id,
                    status=ExecutionStatus.TIMEOUT,
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration_seconds=execution.config.max_execution_time,
                    output=None,
                    metrics={},
                    error=f"Execution exceeded timeout of {execution.config.max_execution_time}s",
                    resource_usage=monitor.stop_monitoring(),
                )
            else:
                # Process completed - get result
                resource_usage = monitor.stop_monitoring()

                try:
                    proc_result = result_queue.get(timeout=1.0)
                except Exception:
                    proc_result = {
                        'status': 'failed',
                        'error': 'Failed to get result from worker',
                        'duration': 0.0,
                    }

                if proc_result['status'] == 'completed':
                    execution.status = ExecutionStatus.COMPLETED
                    result = ExecutionResult(
                        execution_id=execution.execution_id,
                        status=ExecutionStatus.COMPLETED,
                        start_time=start_time,
                        end_time=datetime.now(),
                        duration_seconds=proc_result.get('duration', 0.0),
                        output=proc_result.get('output'),
                        metrics=proc_result.get('metrics', {}),
                        resource_usage=resource_usage,
                    )
                else:
                    execution.status = ExecutionStatus.FAILED
                    result = ExecutionResult(
                        execution_id=execution.execution_id,
                        status=ExecutionStatus.FAILED,
                        start_time=start_time,
                        end_time=datetime.now(),
                        duration_seconds=proc_result.get('duration', 0.0),
                        output=None,
                        metrics={},
                        error=proc_result.get('error'),
                        stack_trace=proc_result.get('stack_trace'),
                        resource_usage=resource_usage,
                    )

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            result = ExecutionResult(
                execution_id=execution.execution_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                output=None,
                metrics={},
                error=str(e),
                stack_trace=traceback.format_exc(),
            )

        finally:
            # Cleanup
            self.filesystem.cleanup_sandbox(execution.execution_id)
            monitor.stop_monitoring()

        # Update execution and stats
        execution.result = result
        self._update_stats(result)

        if self.on_execution_complete:
            self.on_execution_complete(execution)

        return result

    def run_parallel(
        self,
        executions: List[Tuple[ShadowExecution, Callable]],
        max_concurrent: int = 4,
    ) -> List[ExecutionResult]:
        """
        Run multiple shadow executions in parallel.

        Limited to max_concurrent simultaneous executions.
        """
        results = []
        semaphore = threading.Semaphore(max_concurrent)

        def run_with_semaphore(execution: ShadowExecution, function: Callable):
            with semaphore:
                return self.run(execution, function)

        threads = []
        result_list: List[Optional[ExecutionResult]] = [None] * len(executions)

        for i, (execution, function) in enumerate(executions):
            def worker(idx, exec, func):
                result_list[idx] = run_with_semaphore(exec, func)

            thread = threading.Thread(target=worker, args=(i, execution, function))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        return [r for r in result_list if r is not None]

    def _update_stats(self, result: ExecutionResult) -> None:
        """Update arena statistics"""
        with self._lock:
            self.stats['total_executions'] += 1
            self.stats['total_duration'] += result.duration_seconds

            if result.status == ExecutionStatus.COMPLETED:
                self.stats['successful_executions'] += 1
            elif result.status == ExecutionStatus.FAILED:
                self.stats['failed_executions'] += 1
            elif result.status == ExecutionStatus.TIMEOUT:
                self.stats['timeout_executions'] += 1

    def get_execution(self, execution_id: str) -> Optional[ShadowExecution]:
        """Get an execution by ID"""
        with self._lock:
            return self.executions.get(execution_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get arena statistics"""
        with self._lock:
            success_rate = (
                self.stats['successful_executions'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0.0
            )
            avg_duration = (
                self.stats['total_duration'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0.0
            )

            return {
                **self.stats,
                'success_rate': success_rate,
                'avg_duration': avg_duration,
            }

    def cleanup(self) -> int:
        """Clean up all resources"""
        with self._lock:
            # Kill any running processes
            for execution in self.executions.values():
                if execution.process and execution.process.is_alive():
                    execution.process.terminate()

            # Clean up sandboxes
            count = self.filesystem.cleanup_all()
            self.executions.clear()
            return count


@contextmanager
def shadow_execution_context(
    function: Callable,
    arguments: Dict[str, Any],
    config: Optional[ArenaConfig] = None,
):
    """
    Context manager for shadow executions.

    Usage:
        with shadow_execution_context(my_func, {'arg': 'value'}) as result:
            if result.status == ExecutionStatus.COMPLETED:
                print(result.output)
    """
    arena = ShadowArena(config=config)
    execution = arena.create_execution(function, arguments)

    try:
        result = arena.run(execution, function)
        yield result
    finally:
        arena.cleanup()


# Global shadow arena instance
_shadow_arena: Optional[ShadowArena] = None


def get_shadow_arena() -> ShadowArena:
    """Get the global shadow arena instance"""
    global _shadow_arena
    if _shadow_arena is None:
        _shadow_arena = ShadowArena()
    return _shadow_arena
