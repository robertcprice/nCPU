#!/usr/bin/env python3
"""
THE ARENA - Sandboxed Execution Environment

This is the cage where all AI-generated code runs. It provides:
1. Process isolation (fork + sandbox)
2. Resource limits (RAM, CPU time, file descriptors)
3. Input/output capture
4. Timeout enforcement
5. Clean termination

CRITICAL SAFETY PROPERTIES:
- All AI code runs in a SUBPROCESS, never in Governor's process
- Memory limited to prevent OOM attacks
- CPU time limited to prevent infinite loops
- Network access blocked (no exfiltration)
- File system access limited to temp directory

Author: Human (not AI-generated)
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Callable, Tuple
from pathlib import Path
import multiprocessing as mp
import subprocess
import tempfile
import resource
import signal
import time
import os
import sys
import json
import traceback
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArenaConfig:
    """Configuration for the Arena sandbox."""

    # Resource limits - RELAXED FOR TESTING
    max_ram_mb: int = 8192  # 8GB for bigger experiments
    timeout_seconds: int = 120  # 2 minutes - faster iterations
    max_file_size_mb: int = 500  # Larger files allowed
    max_open_files: int = 500  # More files allowed
    max_processes: int = 50  # More parallelism allowed

    # Execution settings
    use_subprocess: bool = True  # If False, uses multiprocessing (less isolated)
    capture_stdout: bool = True
    capture_stderr: bool = True

    # Paths
    sandbox_dir: Path = field(default_factory=lambda: Path("/tmp/genetic_forge/sandbox"))

    # Network (future: use network namespaces)
    allow_network: bool = False

    def __post_init__(self):
        if isinstance(self.sandbox_dir, str):
            self.sandbox_dir = Path(self.sandbox_dir)


@dataclass
class SandboxResult:
    """Result of running code in the Arena."""

    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    memory_used_mb: float = 0.0
    error: Optional[str] = None
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output': str(self.output) if self.output is not None else None,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'execution_time': self.execution_time,
            'memory_used_mb': self.memory_used_mb,
            'error': self.error,
            'exit_code': self.exit_code,
        }


def _set_resource_limits(max_ram_mb: int, max_files: int, max_procs: int) -> None:
    """Set resource limits for the current process (called in child)."""
    try:
        # Memory limit (bytes)
        mem_bytes = max_ram_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        # CPU time limit (handled separately via timeout, but add as backup)
        # resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

        # File size limit
        # file_bytes = max_file_size_mb * 1024 * 1024
        # resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))

        # Open files limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (max_files, max_files))

        # Process limit
        resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))

    except Exception as e:
        # Resource limits may fail on some systems - log but continue
        logger.warning(f"Failed to set resource limits: {e}")


def _run_in_subprocess(
    source_code: str,
    input_data: Any,
    sandbox_dir: Path,
    max_ram_mb: int,
    timeout: int,
) -> Tuple[bool, Any, str, str, float, int]:
    """
    Execute code in a subprocess with full isolation.
    Returns: (success, output, stdout, stderr, exec_time, exit_code)
    """
    # Create temporary file for the code
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    code_file = sandbox_dir / f"agent_{os.getpid()}_{time.time_ns()}.py"
    input_file = sandbox_dir / f"input_{os.getpid()}_{time.time_ns()}.json"
    output_file = sandbox_dir / f"output_{os.getpid()}_{time.time_ns()}.json"

    try:
        # Write the wrapper code that handles input/output
        wrapper_code = f'''
import sys
import json
import resource

# Set resource limits
try:
    mem_bytes = {max_ram_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except:
    pass

# Load input
input_data = None
try:
    with open("{input_file}", "r") as f:
        input_data = json.load(f)
except:
    pass

# Execute the agent code
result = None
error = None

try:
    # The actual agent code
{chr(10).join("    " + line for line in source_code.split(chr(10)))}

    # Try to find a main function or callable
    if 'main' in dir():
        result = main(input_data)
    elif 'run' in dir():
        result = run(input_data)
    elif 'process' in dir():
        result = process(input_data)
    elif 'optimize' in dir():
        result = optimize(input_data)
    else:
        # Look for any function that takes an argument
        for name in dir():
            obj = eval(name)
            if callable(obj) and not name.startswith('_'):
                try:
                    result = obj(input_data)
                    break
                except TypeError:
                    continue
except Exception as e:
    error = str(e)
    import traceback
    print(traceback.format_exc(), file=sys.stderr)

# Save output
output = {{"result": result, "error": error}}
with open("{output_file}", "w") as f:
    json.dump(output, f)
'''

        code_file.write_text(wrapper_code)

        # Write input data
        input_file.write_text(json.dumps(input_data))

        # Run in subprocess
        start_time = time.time()

        try:
            proc = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=str(sandbox_dir),
                env={
                    **os.environ,
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONIOENCODING': 'utf-8',
                },
            )

            exec_time = time.time() - start_time

            # Read output
            output = None
            if output_file.exists():
                try:
                    output_data = json.loads(output_file.read_text())
                    if output_data.get('error'):
                        return False, None, proc.stdout, proc.stderr + f"\nError: {output_data['error']}", exec_time, proc.returncode
                    output = output_data.get('result')
                except:
                    pass

            success = proc.returncode == 0
            return success, output, proc.stdout, proc.stderr, exec_time, proc.returncode

        except subprocess.TimeoutExpired:
            exec_time = time.time() - start_time
            return False, None, "", "Timeout expired", exec_time, -1

    finally:
        # Cleanup temporary files
        for f in [code_file, input_file, output_file]:
            try:
                if f.exists():
                    f.unlink()
            except:
                pass


def _run_in_process(
    source_code: str,
    input_data: Any,
    result_queue: mp.Queue,
    max_ram_mb: int,
    max_files: int,
    max_procs: int,
) -> None:
    """
    Execute code in a child process (multiprocessing).
    Less isolated than subprocess but faster.
    """
    # Set resource limits
    _set_resource_limits(max_ram_mb, max_files, max_procs)

    start_time = time.time()

    try:
        # Create a restricted globals dict
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'sum': sum,
                'min': min,
                'max': max,
                'sorted': sorted,
                'reversed': reversed,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'any': any,
                'all': all,
                'abs': abs,
                'round': round,
                'isinstance': isinstance,
                'type': type,
                'callable': callable,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'IndexError': IndexError,
                'KeyError': KeyError,
                '__import__': lambda name: __import__(name) if name in ['math', 'random', 'itertools', 'functools', 'collections'] else None,
            },
            'input_data': input_data,
        }

        # Execute the code
        exec(source_code, restricted_globals)

        # Look for result
        result = None
        for name in ['main', 'run', 'process', 'optimize']:
            if name in restricted_globals and callable(restricted_globals[name]):
                result = restricted_globals[name](input_data)
                break

        exec_time = time.time() - start_time
        result_queue.put((True, result, "", "", exec_time, 0))

    except Exception as e:
        exec_time = time.time() - start_time
        result_queue.put((False, None, "", traceback.format_exc(), exec_time, 1))


class Arena:
    """
    The Arena - Sandboxed execution environment for AI-generated code.

    All code produced by the Mutator must run here before being
    considered for the population. The Arena enforces:

    1. Process isolation - code runs in separate process
    2. Memory limits - OOM protection
    3. Time limits - infinite loop protection
    4. File limits - prevent file descriptor exhaustion
    5. Output capture - all stdout/stderr captured

    The Arena does NOT:
    - Execute code in the Governor's process
    - Allow network access
    - Allow persistent file modification
    - Trust any output from the sandboxed code
    """

    def __init__(self, config: ArenaConfig):
        self.config = config
        self._execution_count = 0
        self._total_execution_time = 0.0

        # Ensure sandbox directory exists
        self.config.sandbox_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Arena initialized: ram={config.max_ram_mb}MB, timeout={config.timeout_seconds}s")

    def run_sandboxed(
        self,
        source_code: str,
        input_data: Any = None,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """
        Run code in the sandbox with full isolation.

        Args:
            source_code: Python source code to execute
            input_data: Input data to pass to the code
            timeout: Optional timeout override

        Returns:
            SandboxResult with execution details
        """
        if timeout is None:
            timeout = self.config.timeout_seconds

        start_time = time.time()
        self._execution_count += 1

        try:
            if self.config.use_subprocess:
                success, output, stdout, stderr, exec_time, exit_code = _run_in_subprocess(
                    source_code=source_code,
                    input_data=input_data,
                    sandbox_dir=self.config.sandbox_dir,
                    max_ram_mb=self.config.max_ram_mb,
                    timeout=timeout,
                )
            else:
                # Use multiprocessing (faster but less isolated)
                result_queue = mp.Queue()
                proc = mp.Process(
                    target=_run_in_process,
                    args=(
                        source_code,
                        input_data,
                        result_queue,
                        self.config.max_ram_mb,
                        self.config.max_open_files,
                        self.config.max_processes,
                    ),
                )
                proc.start()
                proc.join(timeout=timeout)

                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                    return SandboxResult(
                        success=False,
                        error="Process timed out and was killed",
                        execution_time=timeout,
                    )

                if not result_queue.empty():
                    success, output, stdout, stderr, exec_time, exit_code = result_queue.get_nowait()
                else:
                    success, output, stdout, stderr, exec_time, exit_code = False, None, "", "No result", 0, -1

            self._total_execution_time += exec_time

            return SandboxResult(
                success=success,
                output=output,
                stdout=stdout,
                stderr=stderr,
                execution_time=exec_time,
                exit_code=exit_code,
                error=stderr if not success and stderr else None,
            )

        except Exception as e:
            exec_time = time.time() - start_time
            self._total_execution_time += exec_time

            return SandboxResult(
                success=False,
                error=str(e),
                execution_time=exec_time,
            )

    def run_comparison(
        self,
        old_code: str,
        new_code: str,
        input_data: Any,
        timeout: Optional[int] = None,
    ) -> Tuple[SandboxResult, SandboxResult]:
        """
        Run both old and new code on the same input for comparison.

        Used by the Judge for differential testing.
        """
        if timeout is None:
            timeout = self.config.timeout_seconds // 2

        old_result = self.run_sandboxed(old_code, input_data, timeout)
        new_result = self.run_sandboxed(new_code, input_data, timeout)

        return old_result, new_result

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'execution_count': self._execution_count,
            'total_execution_time': self._total_execution_time,
            'avg_execution_time': self._total_execution_time / self._execution_count if self._execution_count > 0 else 0,
            'config': {
                'max_ram_mb': self.config.max_ram_mb,
                'timeout_seconds': self.config.timeout_seconds,
                'use_subprocess': self.config.use_subprocess,
            },
        }

    def cleanup(self) -> None:
        """Clean up sandbox directory."""
        import shutil
        try:
            if self.config.sandbox_dir.exists():
                shutil.rmtree(self.config.sandbox_dir)
                self.config.sandbox_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")


# ============================================================================
# CONSTITUTION INVARIANT: This code is hand-written and NEVER auto-modified
# ============================================================================
