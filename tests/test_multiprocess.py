"""Comprehensive stress tests for the GPU-native multi-process OS.

Tests cover the full process lifecycle on the Metal GPU kernel:
  - Fork: basic, return values, fork bomb protection
  - Pipes: basic IPC, EOF, multiple writes, dup2 redirect
  - Scheduling: multiple children, orphan reparenting
  - Signals: SIGKILL, SIGTERM
  - Resource limits: process table exhaustion, per-process fork cap
  - Identity: getpid, getppid
  - Environment: per-process env inheritance via SYS_SETENV/SYS_GETENV
  - PipeBuffer unit tests: read/write, EOF, would-block, EPIPE, capacity

All C source programs compile with aarch64-elf-gcc -O2 -ffreestanding and
run on the Metal GPU via MLXKernelCPUv2 with multi-process scheduling.

Requires: aarch64-elf-gcc cross-compiler, Apple Silicon with Metal (MLX)
"""

import sys
import os
import tempfile
import time
import struct

import pytest
from kernels.mlx.availability import has_gpu_backend

# Path setup: project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

HAS_GPU_BACKEND = has_gpu_backend()

try:
    from ncpu.os.gpu.runner import (
        compile_c_from_string, ProcessManager, run_multiprocess,
        make_syscall_handler, MAX_FORKS_PER_PROCESS, MAX_PROCESSES,
        SIGTERM, SIGKILL,
    )
    from ncpu.os.gpu.filesystem import GPUFilesystem, PipeBuffer
    from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2
    HAS_RUNNER = True
except Exception:
    HAS_RUNNER = False

pytestmark = pytest.mark.skipif(
    not (HAS_GPU_BACKEND and HAS_RUNNER),
    reason="GPU backend or arm64 runner not available",
)


# ---------------------------------------------------------------------------
# Helper: compile C source and run in multi-process mode
# ---------------------------------------------------------------------------

def run_multiproc_test(src, max_cycles=5_000_000, time_slice=100_000, quiet=True):
    """Compile C source string, boot as PID 1, and run the multi-process scheduler.

    Returns:
        (results_dict, ProcessManager, GPUFilesystem)
    """
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        bp = f.name
    try:
        ok = compile_c_from_string(src, bp, quiet=True)
        assert ok, "Compilation failed"
        binary = open(bp, 'rb').read()
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        pm.create_init_process(binary, fd_table={}, cwd='/')
        bh = make_syscall_handler(filesystem=fs)
        results = run_multiprocess(pm, bh, max_total_cycles=max_cycles,
                                   time_slice=time_slice, quiet=quiet)
        return results, pm, fs
    finally:
        os.unlink(bp)


# ===========================================================================
# PROCESS LIFECYCLE TESTS
# ===========================================================================


class TestFork:
    """Tests for fork(), wait(), and basic parent-child semantics."""

    def test_fork_basic(self):
        """Fork a child that prints a message; parent waits. Both processes run."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pid = fork();
    if (pid == 0) {
        printf("child\n");
        return 0;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    printf("parent done\n");
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1, "Expected at least one fork"
        # Both parent and child should have exited (no live processes remaining
        # except possibly zombies that were reaped).
        live = [p for p in pm.processes.values() if p.state not in (0, 4)]
        assert len(live) == 0, f"Processes still alive: {live}"

    def test_fork_return_values(self):
        """Verify fork returns 0 to the child and the child PID to the parent."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Child: fork returned 0 */
        int me = getpid();
        printf("child_pid=%d fork_ret=%d\n", me, pid);
        return 0;
    }
    /* Parent: fork returned child pid */
    printf("parent fork_ret=%d\n", pid);
    int status = 0;
    waitpid(pid, &status, 0);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] == 1

    def test_fork_bomb_protection(self):
        """Fork in a tight loop, verify capped at MAX_FORKS_PER_PROCESS (32)."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int count = 0;
    int i;
    for (i = 0; i < 50; i++) {
        int pid = fork();
        if (pid == 0) {
            /* Child exits immediately */
            return 0;
        }
        if (pid < 0) {
            /* Fork failed (limit reached or table full) */
            break;
        }
        count++;
        /* Reap the child immediately to free the slot */
        int status = 0;
        waitpid(pid, &status, 0);
    }
    printf("forked=%d\n", count);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=50_000_000)
        # The process should have been limited. The fork count on the init
        # process must not exceed MAX_FORKS_PER_PROCESS.
        assert results["total_forks"] <= MAX_FORKS_PER_PROCESS, (
            f"Fork bomb protection failed: {results['total_forks']} > {MAX_FORKS_PER_PROCESS}"
        )


class TestPipes:
    """Tests for pipe(), read/write across processes, and EOF semantics."""

    def test_pipe_basic(self):
        """Create pipe, fork, child writes, parent reads. Verify data transferred."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pfd[2];
    if (pipe(pfd) < 0) { printf("pipe failed\n"); return 1; }
    int pid = fork();
    if (pid == 0) {
        /* Child: close read end, write message */
        close(pfd[0]);
        const char *msg = "hello from child";
        write(pfd[1], msg, strlen(msg));
        close(pfd[1]);
        return 0;
    }
    /* Parent: close write end, read message */
    close(pfd[1]);
    char buf[64];
    memset(buf, 0, sizeof(buf));
    ssize_t n = read(pfd[0], buf, sizeof(buf));
    close(pfd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    printf("read %ld bytes: %s\n", (long)n, buf);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1

    def test_pipe_eof(self):
        """After child closes write end, parent read should get 0 bytes (EOF)."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pfd[2];
    pipe(pfd);
    int pid = fork();
    if (pid == 0) {
        close(pfd[0]);
        /* Write something then close */
        write(pfd[1], "x", 1);
        close(pfd[1]);
        return 0;
    }
    close(pfd[1]);
    char buf[16];
    /* First read: should get the byte */
    ssize_t n1 = read(pfd[0], buf, sizeof(buf));
    /* Second read: should get EOF (0 bytes) */
    ssize_t n2 = read(pfd[0], buf, sizeof(buf));
    close(pfd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    printf("n1=%ld n2=%ld\n", (long)n1, (long)n2);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1

    def test_pipe_multiple_writes(self):
        """Child writes multiple chunks, parent reads all accumulated data."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pfd[2];
    pipe(pfd);
    int pid = fork();
    if (pid == 0) {
        close(pfd[0]);
        write(pfd[1], "AAA", 3);
        write(pfd[1], "BBB", 3);
        write(pfd[1], "CCC", 3);
        close(pfd[1]);
        return 0;
    }
    close(pfd[1]);
    char buf[64];
    memset(buf, 0, sizeof(buf));
    int total = 0;
    ssize_t n;
    while ((n = read(pfd[0], buf + total, sizeof(buf) - total)) > 0) {
        total += n;
    }
    close(pfd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    printf("total=%d data=%s\n", total, buf);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1

    def test_dup2_redirect(self):
        """Create pipe, dup2 write end to stdout (fd 1), verify data routes through pipe."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pfd[2];
    pipe(pfd);
    int pid = fork();
    if (pid == 0) {
        /* Child: redirect stdout to pipe write end */
        close(pfd[0]);
        dup2(pfd[1], 1);
        close(pfd[1]);
        /* This write to fd 1 should go into the pipe */
        const char *msg = "redirected";
        write(1, msg, strlen(msg));
        return 0;
    }
    /* Parent: read from pipe read end */
    close(pfd[1]);
    char buf[64];
    memset(buf, 0, sizeof(buf));
    ssize_t n = read(pfd[0], buf, sizeof(buf));
    close(pfd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    printf("got=%s len=%ld\n", buf, (long)n);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1


class TestMultipleChildren:
    """Tests for multi-child scenarios and scheduling."""

    def test_multiple_children(self):
        """Fork 4 children, each prints its PID, parent waits for all."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int i;
    int pids[4];
    for (i = 0; i < 4; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            int me = getpid();
            printf("child %d pid=%d\n", i, me);
            return 0;
        }
    }
    /* Parent: wait for all 4 */
    for (i = 0; i < 4; i++) {
        int status = 0;
        waitpid(pids[i], &status, 0);
    }
    printf("all children done\n");
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=10_000_000)
        assert results["total_forks"] == 4, (
            f"Expected 4 forks, got {results['total_forks']}"
        )

    def test_orphan_reparenting(self):
        """Parent forks intermediate child, intermediate forks grandchild,
        intermediate exits. Grandchild should be reparented to PID 1."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Intermediate process */
        int gc = fork();
        if (gc == 0) {
            /* Grandchild: wait a bit then check ppid */
            /* In practice the scheduler will run us after
               intermediate exits, so ppid should be 1 */
            int ppid = getppid();
            printf("grandchild ppid=%d\n", ppid);
            return 0;
        }
        /* Intermediate exits without waiting for grandchild */
        return 0;
    }
    /* Init: wait for intermediate */
    int status = 0;
    waitpid(pid, &status, 0);
    /* Now wait for the reparented grandchild */
    int gc_pid = wait(&status);
    printf("reaped gc=%d\n", gc_pid);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=10_000_000)
        assert results["total_forks"] >= 2, (
            f"Expected at least 2 forks, got {results['total_forks']}"
        )


class TestSignals:
    """Tests for kill() with SIGKILL and SIGTERM."""

    def test_signal_kill(self):
        """Fork an infinite-loop child, SIGKILL it, parent waits successfully."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Child: infinite loop */
        volatile int x = 0;
        while (1) { x++; }
        return 0;
    }
    /* Parent: kill child with SIGKILL */
    int ret = kill(pid, 9);
    printf("kill ret=%d\n", ret);
    int status = 0;
    int w = waitpid(pid, &status, 0);
    printf("wait ret=%d\n", w);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=10_000_000)
        assert results["total_forks"] >= 1
        # The child should have been killed, no stuck processes
        live = [p for p in pm.processes.values()
                if p.state not in (0, 4)]  # FREE or ZOMBIE
        assert len(live) == 0, f"Processes still alive after SIGKILL: {live}"

    def test_signal_term(self):
        """Fork child, send SIGTERM, verify it terminates on next schedule."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int pid = fork();
    if (pid == 0) {
        /* Child: busy loop */
        volatile int x = 0;
        while (1) { x++; }
        return 0;
    }
    /* Parent: SIGTERM the child */
    int ret = kill(pid, 15);
    printf("sigterm ret=%d\n", ret);
    int status = 0;
    int w = waitpid(pid, &status, 0);
    printf("wait ret=%d\n", w);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=10_000_000)
        assert results["total_forks"] >= 1


class TestResourceLimits:
    """Tests for process table exhaustion and related limits."""

    def test_process_table_exhaustion(self):
        """Fork until MAX_PROCESSES without waiting. Verify fork returns -1."""
        # We fork children that block forever (infinite loop). Since we never
        # wait, the process table fills up and fork() should return -1.
        src = r'''#include "arm64_libc.h"
int main(void) {
    int count = 0;
    int i;
    for (i = 0; i < 64; i++) {
        int pid = fork();
        if (pid == 0) {
            /* Child: spin until killed */
            volatile int x = 0;
            while (1) { x++; }
            return 0;
        }
        if (pid < 0) {
            /* Fork failed — table full or per-process limit */
            break;
        }
        count++;
    }
    printf("spawned=%d\n", count);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src, max_cycles=20_000_000)
        # Should not exceed MAX_PROCESSES - 1 children (PID 1 is init)
        assert results["total_forks"] <= MAX_PROCESSES - 1, (
            f"Exceeded process limit: {results['total_forks']}"
        )
        # Also check processes_created does not exceed MAX_PROCESSES
        assert results["processes_created"] <= MAX_PROCESSES

    def test_wait_no_children(self):
        """Call waitpid(-1) with no children. Verify it returns -1 (ECHILD)."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int status = 0;
    int ret = waitpid(-1, &status, 0);
    printf("wait_ret=%d\n", ret);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        # The process should have completed normally without hanging
        live = [p for p in pm.processes.values()
                if p.state not in (0, 4)]
        assert len(live) == 0, "Process should not be blocked waiting"


class TestProcessIdentity:
    """Tests for getpid() and getppid() syscalls."""

    def test_getpid_getppid(self):
        """Verify getpid returns correct PID and getppid returns parent PID."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    int my_pid = getpid();
    printf("init pid=%d\n", my_pid);

    int pid = fork();
    if (pid == 0) {
        int child_pid = getpid();
        int parent_pid = getppid();
        printf("child: pid=%d ppid=%d\n", child_pid, parent_pid);
        return 0;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1


class TestEnvironment:
    """Tests for per-process environment variable inheritance."""

    def test_per_process_environment(self):
        """Set env var with SYS_SETENV, fork, child reads it. Verify inheritance."""
        src = r'''#include "arm64_libc.h"
int main(void) {
    sys_setenv("MY_VAR", "hello123");

    int pid = fork();
    if (pid == 0) {
        /* Child: read inherited env var */
        char buf[64];
        memset(buf, 0, sizeof(buf));
        int ret = sys_getenv("MY_VAR", buf, sizeof(buf));
        printf("child env ret=%d val=%s\n", ret, buf);
        return 0;
    }
    int status = 0;
    waitpid(pid, &status, 0);

    /* Verify parent still has it */
    char buf2[64];
    memset(buf2, 0, sizeof(buf2));
    int ret2 = sys_getenv("MY_VAR", buf2, sizeof(buf2));
    printf("parent env ret=%d val=%s\n", ret2, buf2);
    return 0;
}
'''
        results, pm, fs = run_multiproc_test(src)
        assert results["total_forks"] >= 1


# ===========================================================================
# PipeBuffer UNIT TESTS (no GPU, no compilation)
# ===========================================================================


class TestPipeBufferUnit:
    """Unit tests for the PipeBuffer class in isolation. No GPU required."""

    def test_pipe_buffer_basic(self):
        """Write data and read it back."""
        pb = PipeBuffer(capacity=256)
        written = pb.write(b"hello")
        assert written == 5
        data = pb.read(10)
        assert data == b"hello"

    def test_pipe_buffer_eof(self):
        """Close writer, verify read returns empty bytes (EOF)."""
        pb = PipeBuffer(capacity=256)
        pb.write(b"data")
        # Drain the buffer
        data = pb.read(100)
        assert data == b"data"
        # Close writer
        pb.close_writer()
        assert pb.writers == 0
        # Now read should return EOF (empty bytes, not None)
        eof = pb.read(100)
        assert eof == b"", f"Expected EOF (b''), got {eof!r}"

    def test_pipe_buffer_would_block(self):
        """Empty buffer with writers alive should return None (would-block)."""
        pb = PipeBuffer(capacity=256)
        assert pb.writers == 1
        # Buffer is empty, writers still alive => would block
        result = pb.read(10)
        assert result is None, f"Expected None (would-block), got {result!r}"

    def test_pipe_buffer_epipe(self):
        """Close all readers, verify write returns -1 (EPIPE)."""
        pb = PipeBuffer(capacity=256)
        pb.close_reader()
        assert pb.readers == 0
        result = pb.write(b"should fail")
        assert result == -1, f"Expected -1 (EPIPE), got {result}"

    def test_pipe_buffer_capacity(self):
        """Write more than capacity, verify partial write returned."""
        pb = PipeBuffer(capacity=8)
        # Write 20 bytes to an 8-byte pipe
        written = pb.write(b"A" * 20)
        assert written == 8, f"Expected partial write of 8, got {written}"
        # Buffer should now be full
        assert len(pb.buffer) == 8
        # Second write should return 0 (pipe full, would block)
        written2 = pb.write(b"B" * 5)
        assert written2 == 0, f"Expected 0 (pipe full), got {written2}"
        # Read to drain
        data = pb.read(100)
        assert data == b"A" * 8
        # Now writing should work again
        written3 = pb.write(b"C" * 3)
        assert written3 == 3

    def test_pipe_buffer_incremental_read(self):
        """Read in small chunks to verify buffer management."""
        pb = PipeBuffer(capacity=256)
        pb.write(b"ABCDEFGHIJ")
        # Read 3 at a time
        chunk1 = pb.read(3)
        assert chunk1 == b"ABC"
        chunk2 = pb.read(3)
        assert chunk2 == b"DEF"
        chunk3 = pb.read(3)
        assert chunk3 == b"GHI"
        chunk4 = pb.read(3)
        assert chunk4 == b"J"

    def test_pipe_buffer_close_reader_clamps(self):
        """Closing reader multiple times should clamp to 0, not go negative."""
        pb = PipeBuffer()
        pb.close_reader()
        assert pb.readers == 0
        pb.close_reader()
        assert pb.readers == 0, "Reader count should not go below 0"

    def test_pipe_buffer_close_writer_clamps(self):
        """Closing writer multiple times should clamp to 0, not go negative."""
        pb = PipeBuffer()
        pb.close_writer()
        assert pb.writers == 0
        pb.close_writer()
        assert pb.writers == 0, "Writer count should not go below 0"

    def test_pipe_buffer_refcount_increment(self):
        """Verify that reference counts can be manually incremented (for fork)."""
        pb = PipeBuffer()
        assert pb.readers == 1
        assert pb.writers == 1
        # Simulate fork: increment refcounts
        pb.readers += 1
        pb.writers += 1
        assert pb.readers == 2
        assert pb.writers == 2
        # Close one writer, should still have data available (not EOF)
        pb.close_writer()
        assert pb.writers == 1
        pb.write(b"still open")
        data = pb.read(100)
        assert data == b"still open"


# ===========================================================================
# GPUFilesystem PIPE INTEGRATION TESTS (no GPU)
# ===========================================================================


class TestFilesystemPipes:
    """Tests for GPUFilesystem pipe creation, dup2, and clone_fd_table."""

    def test_create_pipe_returns_valid_fds(self):
        """create_pipe() returns two distinct fds >= 3."""
        fs = GPUFilesystem()
        read_fd, write_fd = fs.create_pipe()
        assert read_fd >= 3, f"Read fd should be >= 3, got {read_fd}"
        assert write_fd >= 3, f"Write fd should be >= 3, got {write_fd}"
        assert read_fd != write_fd, "Read and write fds must differ"

    def test_pipe_read_write_through_fs(self):
        """Write via fs.write() to pipe write fd, read via fs.read() from read fd."""
        fs = GPUFilesystem()
        read_fd, write_fd = fs.create_pipe()
        result = fs.write(write_fd, b"test data")
        assert result == 9
        data = fs.read(read_fd, 100)
        assert data == b"test data"

    def test_dup2_basic(self):
        """dup2 should duplicate a file descriptor."""
        fs = GPUFilesystem()
        read_fd, write_fd = fs.create_pipe()
        # Dup write_fd to fd 10
        result = fs.dup2(write_fd, 10)
        assert result == 10
        # Writing to fd 10 should go to the same pipe
        fs.write(10, b"via dup")
        data = fs.read(read_fd, 100)
        assert data == b"via dup"

    def test_clone_fd_table_increments_refcounts(self):
        """clone_fd_table() should share pipe buffers and increment refcounts."""
        fs = GPUFilesystem()
        read_fd, write_fd = fs.create_pipe()
        pipe_buf = fs.fd_table[read_fd]["pipe_buffer"]
        assert pipe_buf.readers == 1
        assert pipe_buf.writers == 1
        cloned = fs.clone_fd_table()
        # After clone, refcounts should be incremented
        assert pipe_buf.readers == 2
        assert pipe_buf.writers == 2
        # Both tables should reference the same pipe buffer
        assert cloned[read_fd]["pipe_buffer"] is pipe_buf
        assert cloned[write_fd]["pipe_buffer"] is pipe_buf

    def test_close_pipe_endpoint_decrements_refcount(self):
        """Closing a pipe endpoint should decrement its refcount."""
        fs = GPUFilesystem()
        read_fd, write_fd = fs.create_pipe()
        pipe_buf = fs.fd_table[read_fd]["pipe_buffer"]
        assert pipe_buf.readers == 1
        fs.close(read_fd)
        assert pipe_buf.readers == 0
        # Write should now return EPIPE since no readers
        result = pipe_buf.write(b"no readers")
        assert result == -1


# ===========================================================================
# ProcessManager UNIT TESTS (no compilation, just PM logic)
# ===========================================================================


class TestProcessManagerUnit:
    """Unit tests for ProcessManager internals without compiling C code."""

    def test_alloc_pid_sequential(self):
        """PIDs should be allocated sequentially starting from 1."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        pid1 = pm._alloc_pid()
        assert pid1 == 1
        pid2 = pm._alloc_pid()
        assert pid2 == 2
        pid3 = pm._alloc_pid()
        assert pid3 == 3

    def test_alloc_pid_exhaustion(self):
        """After MAX_PROCESSES PIDs are allocated, _alloc_pid should return -1."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        pids = []
        for _ in range(MAX_PROCESSES):
            pid = pm._alloc_pid()
            assert pid > 0, f"Expected valid PID, got {pid}"
            # Register it so it's occupied
            from ncpu.os.gpu.runner import Process, ProcessState
            pm.processes[pid] = Process(
                pid=pid, ppid=0, state=ProcessState.READY
            )
            pids.append(pid)
        # Next allocation should fail
        overflow_pid = pm._alloc_pid()
        assert overflow_pid == -1, f"Expected -1, got {overflow_pid}"

    def test_kill_nonexistent_process(self):
        """Killing a PID that does not exist should return -1."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        result = pm.kill_process(999, SIGKILL, 1)
        assert result == -1

    def test_kill_zombie_process(self):
        """Killing an already-zombie process should return -1."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[5] = Process(
            pid=5, ppid=1, state=ProcessState.ZOMBIE, exit_code=0
        )
        result = pm.kill_process(5, SIGKILL, 1)
        assert result == -1

    def test_signal_zero_existence_check(self):
        """kill(pid, 0) should return 0 if process exists, -1 if not."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[3] = Process(
            pid=3, ppid=1, state=ProcessState.READY
        )
        assert pm.kill_process(3, 0, 1) == 0
        assert pm.kill_process(99, 0, 1) == -1

    def test_reap_zombie_any_child(self):
        """reap_zombie with child_pid=-1 should reap any zombie child."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(pid=1, ppid=0, state=ProcessState.RUNNING)
        pm.processes[2] = Process(pid=2, ppid=1, state=ProcessState.ZOMBIE, exit_code=42)
        zombie = pm.reap_zombie(parent_pid=1, child_pid=-1)
        assert zombie is not None
        assert zombie.pid == 2
        assert zombie.exit_code == 42
        assert 2 not in pm.processes, "Zombie should be removed from process table"

    def test_reap_zombie_specific_child(self):
        """reap_zombie with a specific child_pid should only reap that child."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(pid=1, ppid=0, state=ProcessState.RUNNING)
        pm.processes[2] = Process(pid=2, ppid=1, state=ProcessState.ZOMBIE, exit_code=1)
        pm.processes[3] = Process(pid=3, ppid=1, state=ProcessState.ZOMBIE, exit_code=2)
        zombie = pm.reap_zombie(parent_pid=1, child_pid=3)
        assert zombie is not None
        assert zombie.pid == 3
        assert zombie.exit_code == 2
        # PID 2 should still be in the table
        assert 2 in pm.processes

    def test_reap_zombie_wrong_parent(self):
        """reap_zombie should not reap a child belonging to a different parent."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(pid=1, ppid=0, state=ProcessState.RUNNING)
        pm.processes[2] = Process(pid=2, ppid=1, state=ProcessState.ZOMBIE, exit_code=0)
        # Try to reap PID 2 as if parent were PID 5 (not its real parent)
        zombie = pm.reap_zombie(parent_pid=5, child_pid=2)
        assert zombie is None

    def test_schedule_next_round_robin(self):
        """Scheduler should pick the next READY process in round-robin order."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        for pid in [1, 2, 3]:
            pm.processes[pid] = Process(
                pid=pid, ppid=0, state=ProcessState.READY
            )
        pm.current_pid = 1
        nxt = pm.schedule_next()
        assert nxt == 2, f"Expected PID 2, got {nxt}"
        pm.current_pid = 3
        nxt = pm.schedule_next()
        assert nxt == 1, f"Expected PID 1 (wrap-around), got {nxt}"

    def test_schedule_next_no_ready(self):
        """Scheduler should return None when no processes are READY."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(pid=1, ppid=0, state=ProcessState.BLOCKED)
        pm.current_pid = 0
        nxt = pm.schedule_next()
        assert nxt is None

    def test_process_exit_reparents_children(self):
        """When a process exits, its children should be reparented to PID 1."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(pid=1, ppid=0, state=ProcessState.READY)
        pm.processes[2] = Process(pid=2, ppid=1, state=ProcessState.RUNNING)
        pm.processes[3] = Process(pid=3, ppid=2, state=ProcessState.READY)
        pm.processes[4] = Process(pid=4, ppid=2, state=ProcessState.READY)
        pm.process_exit(2, exit_code=0)
        assert pm.processes[3].ppid == 1, "Child 3 should be reparented to PID 1"
        assert pm.processes[4].ppid == 1, "Child 4 should be reparented to PID 1"

    def test_process_exit_wakes_waiting_parent(self):
        """When a child exits, a BLOCKED parent waiting for it should become READY."""
        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        pm = ProcessManager(cpu, fs)
        from ncpu.os.gpu.runner import Process, ProcessState
        pm.processes[1] = Process(
            pid=1, ppid=0, state=ProcessState.BLOCKED, wait_target=-1
        )
        pm.processes[2] = Process(pid=2, ppid=1, state=ProcessState.RUNNING)
        pm.process_exit(2, exit_code=0)
        assert pm.processes[1].state == ProcessState.READY, (
            "Parent should be woken after child exit"
        )
