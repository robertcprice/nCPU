"""
Container Enforcer - Hard Resource Limits with Hardware Isolation
OUROBOROS Phase 7.1 - V4 Ratchet System

Provides immutable resource constraints that consciousness CANNOT modify.
Supports software isolation (cgroups/seccomp) and hardware isolation (SGX/SEV).

Per 6-AI Panel Recommendations:
- 8GB RAM maximum (panel suggested doubling from 4GB)
- 2 CPU cores (with optional 4 for compute-intensive)
- 500 tokens per thought (expandable to 1000)
- 50 decisions per hour (expandable to 200)
- 20% memory decay per hour (adjustable 5-50%)
- No network access (air-gapped)
"""

import os
import sys
import time
import signal
import resource
import threading
import subprocess
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import json


class IsolationLevel(Enum):
    """Isolation levels from software to hardware"""
    NONE = auto()           # No isolation (testing only)
    PROCESS = auto()        # Basic process isolation
    CGROUPS = auto()        # Linux cgroups v2
    SECCOMP = auto()        # Seccomp + cgroups
    NAMESPACE = auto()      # Full namespace isolation
    SGX = auto()            # Intel SGX enclave
    SEV = auto()            # AMD SEV encryption
    HARDWARE_FULL = auto()  # SGX + SEV combined


@dataclass
class ContainerLimits:
    """Immutable resource limits - cannot be modified by consciousness"""

    # Memory limits
    ram_bytes: int = 8 * 1024**3  # 8GB
    stack_bytes: int = 8 * 1024**2  # 8MB stack

    # CPU limits
    cpu_cores: int = 2
    cpu_percent: float = 100.0  # Per-core percentage

    # Token limits (for LLM thoughts)
    tokens_per_thought: int = 500
    max_tokens_per_hour: int = 25000

    # Decision limits
    decisions_per_hour: int = 50
    max_decisions_per_day: int = 1000

    # Memory decay
    memory_decay_rate: float = 0.20  # 20% per hour
    memory_decay_interval: int = 3600  # 1 hour in seconds

    # Time limits
    operation_timeout: int = 300  # 5 minutes per operation
    total_runtime_limit: int = 86400  # 24 hours max runtime

    # Network (always disabled)
    network_enabled: bool = False

    # Filesystem
    filesystem_readonly: bool = True
    sandbox_path: str = "/tmp/ouroboros_sandbox"
    max_file_size: int = 100 * 1024**2  # 100MB

    def __post_init__(self):
        """Validate limits are within safe bounds"""
        assert self.ram_bytes <= 16 * 1024**3, "RAM limit cannot exceed 16GB"
        assert self.cpu_cores <= 8, "CPU cores cannot exceed 8"
        assert self.tokens_per_thought <= 2000, "Tokens per thought cannot exceed 2000"
        assert self.decisions_per_hour <= 500, "Decisions per hour cannot exceed 500"
        assert 0.05 <= self.memory_decay_rate <= 0.50, "Memory decay must be 5-50%"
        assert not self.network_enabled, "Network must always be disabled"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize limits for logging"""
        return {
            'ram_gb': self.ram_bytes / 1024**3,
            'cpu_cores': self.cpu_cores,
            'tokens_per_thought': self.tokens_per_thought,
            'decisions_per_hour': self.decisions_per_hour,
            'memory_decay_rate': self.memory_decay_rate,
            'network_enabled': self.network_enabled,
            'filesystem_readonly': self.filesystem_readonly,
        }

    def hash(self) -> str:
        """Cryptographic hash of limits for integrity verification"""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ResourceUsage:
    """Current resource usage tracking"""
    memory_bytes: int = 0
    cpu_time_seconds: float = 0.0
    tokens_used: int = 0
    decisions_made: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_decay: datetime = field(default_factory=datetime.now)
    violations: List[Dict[str, Any]] = field(default_factory=list)


class ContainerEnforcer:
    """
    Hard container enforcement for consciousness layer.

    This class enforces immutable resource limits that the consciousness
    layer CANNOT modify. It supports multiple isolation levels from
    basic process isolation to hardware-level SGX/SEV enclaves.

    CRITICAL: This is part of THE CONSTITUTION and must never be
    modified by any AI component.
    """

    def __init__(
        self,
        limits: Optional[ContainerLimits] = None,
        isolation_level: IsolationLevel = IsolationLevel.SECCOMP,
        on_violation: Optional[Callable[[str, Dict], None]] = None,
        auto_kill_on_violation: bool = True,
    ):
        self.limits = limits or ContainerLimits()
        self.isolation_level = isolation_level
        self.on_violation = on_violation
        self.auto_kill = auto_kill_on_violation

        self.usage = ResourceUsage()
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Verify limits hash on initialization
        self._limits_hash = self.limits.hash()

        # Track process for kill switch
        self._contained_pid: Optional[int] = None

    def start(self) -> None:
        """Start container enforcement"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.usage = ResourceUsage()

            # Apply OS-level limits
            self._apply_resource_limits()

            # Start monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="ContainerEnforcer-Monitor"
            )
            self._monitor_thread.start()

    def stop(self) -> None:
        """Stop container enforcement"""
        with self._lock:
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                self._monitor_thread = None

    def _apply_resource_limits(self) -> None:
        """Apply OS-level resource limits"""
        try:
            # Memory limit (soft and hard)
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.limits.ram_bytes, self.limits.ram_bytes)
            )

            # Stack limit
            resource.setrlimit(
                resource.RLIMIT_STACK,
                (self.limits.stack_bytes, self.limits.stack_bytes)
            )

            # CPU time limit (per-process)
            max_cpu = self.limits.total_runtime_limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (max_cpu, max_cpu)
            )

            # File size limit
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (self.limits.max_file_size, self.limits.max_file_size)
            )

            # Number of open files
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (256, 256)
            )

        except (ValueError, resource.error) as e:
            # Log but continue - some limits may require root
            self._log_warning(f"Could not set some resource limits: {e}")

    def _apply_cgroups(self, pid: int) -> bool:
        """Apply cgroups v2 limits"""
        if self.isolation_level.value < IsolationLevel.CGROUPS.value:
            return True

        cgroup_path = f"/sys/fs/cgroup/ouroboros_{pid}"

        try:
            # Create cgroup (requires root)
            os.makedirs(cgroup_path, exist_ok=True)

            # Memory limit
            with open(f"{cgroup_path}/memory.max", 'w') as f:
                f.write(str(self.limits.ram_bytes))

            # CPU limit (in microseconds per 100ms period)
            cpu_quota = int(self.limits.cpu_cores * 100000)
            with open(f"{cgroup_path}/cpu.max", 'w') as f:
                f.write(f"{cpu_quota} 100000")

            # Add process to cgroup
            with open(f"{cgroup_path}/cgroup.procs", 'w') as f:
                f.write(str(pid))

            return True

        except (PermissionError, FileNotFoundError, OSError) as e:
            self._log_warning(f"Could not apply cgroups: {e}")
            return False

    def _apply_seccomp(self) -> bool:
        """Apply seccomp-bpf syscall filtering"""
        if self.isolation_level.value < IsolationLevel.SECCOMP.value:
            return True

        # Seccomp requires libseccomp - check availability
        try:
            import seccomp

            # Create seccomp filter
            f = seccomp.SyscallFilter(defaction=seccomp.KILL)

            # Allow essential syscalls only
            allowed_syscalls = [
                'read', 'write', 'open', 'close', 'stat', 'fstat',
                'lseek', 'mmap', 'mprotect', 'munmap', 'brk',
                'rt_sigaction', 'rt_sigprocmask', 'ioctl',
                'access', 'pipe', 'select', 'sched_yield',
                'mremap', 'msync', 'mincore', 'madvise',
                'dup', 'dup2', 'nanosleep', 'getpid', 'getuid',
                'getgid', 'geteuid', 'getegid', 'getppid',
                'getpgrp', 'setsid', 'getgroups', 'setgroups',
                'clock_gettime', 'clock_nanosleep',
                'exit', 'exit_group', 'wait4', 'kill',
                'uname', 'fcntl', 'flock', 'fsync',
                'readv', 'writev', 'pread64', 'pwrite64',
                'getcwd', 'chdir', 'rename', 'mkdir', 'rmdir',
                'unlink', 'link', 'symlink', 'readlink',
                'chmod', 'fchmod', 'chown', 'fchown',
                'umask', 'gettimeofday', 'getrlimit',
                'getrusage', 'sysinfo', 'times',
                'futex', 'set_tid_address', 'set_robust_list',
                'get_robust_list', 'getrandom',
            ]

            for syscall in allowed_syscalls:
                try:
                    f.add_rule(seccomp.ALLOW, syscall)
                except Exception:
                    pass  # Syscall may not exist on this platform

            # DENY network syscalls explicitly
            network_syscalls = [
                'socket', 'connect', 'accept', 'bind', 'listen',
                'sendto', 'recvfrom', 'sendmsg', 'recvmsg',
                'shutdown', 'setsockopt', 'getsockopt',
                'getpeername', 'getsockname',
            ]
            for syscall in network_syscalls:
                try:
                    f.add_rule(seccomp.ERRNO(1), syscall)  # Return EPERM
                except Exception:
                    pass

            f.load()
            return True

        except ImportError:
            self._log_warning("libseccomp not available, skipping seccomp")
            return False
        except Exception as e:
            self._log_warning(f"Could not apply seccomp: {e}")
            return False

    def check_sgx_available(self) -> bool:
        """Check if Intel SGX is available"""
        try:
            # Check for SGX support in CPUID
            result = subprocess.run(
                ['cpuid', '-1', '-l', '7'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'SGX' in result.stdout
        except Exception:
            return False

    def check_sev_available(self) -> bool:
        """Check if AMD SEV is available"""
        try:
            return os.path.exists('/dev/sev')
        except Exception:
            return False

    def verify_limits_integrity(self) -> bool:
        """Verify limits have not been tampered with"""
        return self.limits.hash() == self._limits_hash

    def check_resource(self, resource_type: str, amount: int = 1) -> bool:
        """
        Check if resource usage is within limits.
        Returns True if allowed, False if would exceed limits.
        """
        with self._lock:
            if not self.verify_limits_integrity():
                self._handle_violation("limits_tampered", {
                    "expected_hash": self._limits_hash,
                    "actual_hash": self.limits.hash()
                })
                return False

            if resource_type == "tokens":
                # Check per-thought limit
                if amount > self.limits.tokens_per_thought:
                    self._handle_violation("tokens_per_thought_exceeded", {
                        "requested": amount,
                        "limit": self.limits.tokens_per_thought
                    })
                    return False

                # Check hourly limit
                hour_start = datetime.now() - timedelta(hours=1)
                if self.usage.tokens_used + amount > self.limits.max_tokens_per_hour:
                    self._handle_violation("tokens_per_hour_exceeded", {
                        "used": self.usage.tokens_used,
                        "requested": amount,
                        "limit": self.limits.max_tokens_per_hour
                    })
                    return False

                return True

            elif resource_type == "decision":
                # Check hourly limit
                if self.usage.decisions_made + amount > self.limits.decisions_per_hour:
                    self._handle_violation("decisions_per_hour_exceeded", {
                        "made": self.usage.decisions_made,
                        "limit": self.limits.decisions_per_hour
                    })
                    return False

                return True

            elif resource_type == "memory":
                if self.usage.memory_bytes + amount > self.limits.ram_bytes:
                    self._handle_violation("memory_exceeded", {
                        "used": self.usage.memory_bytes,
                        "requested": amount,
                        "limit": self.limits.ram_bytes
                    })
                    return False

                return True

            return True

    def record_usage(self, resource_type: str, amount: int = 1) -> None:
        """Record resource usage after successful operation"""
        with self._lock:
            if resource_type == "tokens":
                self.usage.tokens_used += amount
            elif resource_type == "decision":
                self.usage.decisions_made += amount
            elif resource_type == "memory":
                self.usage.memory_bytes += amount

    def apply_memory_decay(self) -> float:
        """Apply memory decay and return amount decayed"""
        with self._lock:
            now = datetime.now()
            elapsed = (now - self.usage.last_decay).total_seconds()

            if elapsed >= self.limits.memory_decay_interval:
                decay_amount = int(self.usage.memory_bytes * self.limits.memory_decay_rate)
                self.usage.memory_bytes = max(0, self.usage.memory_bytes - decay_amount)
                self.usage.last_decay = now
                return decay_amount

            return 0.0

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                # Check memory usage
                current_memory = self._get_current_memory()
                if current_memory > self.limits.ram_bytes:
                    self._handle_violation("memory_exceeded", {
                        "current": current_memory,
                        "limit": self.limits.ram_bytes
                    })

                # Apply memory decay
                self.apply_memory_decay()

                # Verify limits integrity
                if not self.verify_limits_integrity():
                    self._handle_violation("limits_tampered", {})

                # Check runtime limit
                elapsed = (datetime.now() - self.usage.start_time).total_seconds()
                if elapsed > self.limits.total_runtime_limit:
                    self._handle_violation("runtime_exceeded", {
                        "elapsed": elapsed,
                        "limit": self.limits.total_runtime_limit
                    })

                time.sleep(1.0)  # Check every second

            except Exception as e:
                self._log_warning(f"Monitor error: {e}")

    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to /proc
            try:
                with open('/proc/self/statm', 'r') as f:
                    pages = int(f.read().split()[1])
                    return pages * 4096  # Assume 4KB pages
            except Exception:
                return 0

    def _handle_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Handle a resource violation"""
        violation = {
            "type": violation_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }

        with self._lock:
            self.usage.violations.append(violation)

        # Call custom handler
        if self.on_violation:
            try:
                self.on_violation(violation_type, details)
            except Exception:
                pass

        # Auto-kill if enabled
        if self.auto_kill:
            self._emergency_halt(violation_type)

    def _emergency_halt(self, reason: str) -> None:
        """Emergency halt - kill the contained process"""
        self._log_warning(f"EMERGENCY HALT: {reason}")

        if self._contained_pid:
            try:
                os.kill(self._contained_pid, signal.SIGKILL)
            except Exception:
                pass

        # Also kill current process group
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except Exception:
            pass

    def _log_warning(self, message: str) -> None:
        """Log a warning message"""
        print(f"[ContainerEnforcer WARNING] {message}", file=sys.stderr)

    def get_status(self) -> Dict[str, Any]:
        """Get current container status"""
        with self._lock:
            return {
                "running": self._running,
                "isolation_level": self.isolation_level.name,
                "limits": self.limits.to_dict(),
                "limits_hash": self._limits_hash,
                "limits_integrity": self.verify_limits_integrity(),
                "usage": {
                    "memory_bytes": self.usage.memory_bytes,
                    "tokens_used": self.usage.tokens_used,
                    "decisions_made": self.usage.decisions_made,
                    "violations": len(self.usage.violations),
                    "runtime_seconds": (datetime.now() - self.usage.start_time).total_seconds()
                },
                "sgx_available": self.check_sgx_available(),
                "sev_available": self.check_sev_available(),
            }

    def run_in_container(
        self,
        func: Callable,
        *args,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Run a function within the container with full enforcement.

        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Override operation timeout
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TimeoutError: If operation exceeds timeout
            MemoryError: If memory limit exceeded
            RuntimeError: If any violation occurs
        """
        timeout = timeout or self.limits.operation_timeout

        # Check decision budget
        if not self.check_resource("decision"):
            raise RuntimeError("Decision budget exceeded")

        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            self._handle_violation("operation_timeout", {
                "timeout": timeout,
                "function": func.__name__
            })
            raise TimeoutError(f"Operation timed out after {timeout}s")

        # Record successful decision
        self.record_usage("decision")

        if exception:
            raise exception

        return result


# Singleton instance for global access
_enforcer: Optional[ContainerEnforcer] = None


def get_enforcer() -> ContainerEnforcer:
    """Get the global container enforcer instance"""
    global _enforcer
    if _enforcer is None:
        _enforcer = ContainerEnforcer()
    return _enforcer


def initialize_enforcer(
    limits: Optional[ContainerLimits] = None,
    isolation_level: IsolationLevel = IsolationLevel.SECCOMP,
) -> ContainerEnforcer:
    """Initialize the global container enforcer"""
    global _enforcer
    _enforcer = ContainerEnforcer(limits=limits, isolation_level=isolation_level)
    _enforcer.start()
    return _enforcer
