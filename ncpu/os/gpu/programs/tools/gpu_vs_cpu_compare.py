#!/usr/bin/env python3
"""
GPU vs CPU Execution Comparison Tool

Runs the same ARM64 binary on both GPU (nCPU with instruction tracing) and
native CPU (QEMU user-mode with -d in_asm), captures execution traces from
both, and diffs them to find the first instruction divergence.

This is a general-purpose debugging technique for finding ARM64 instruction
execution bugs in the Metal kernel.

Usage:
    python3 -m ncpu.os.gpu.programs.tools.gpu_vs_cpu_compare [command] [args...]

Example:
    python3 gpu_vs_cpu_compare.py echo hello
    python3 gpu_vs_cpu_compare.py grep -F root /etc/passwd
    python3 gpu_vs_cpu_compare.py basename /usr/local/bin/python3

Requires: qemu-aarch64 (brew install qemu)
"""

import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.gpu_cpu import GPUKernelCPU as create_gpu_cpu, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = PROJECT_ROOT / "demos" / "busybox.elf"


def find_qemu():
    """Find qemu-aarch64 binary."""
    for path in [
        "/opt/homebrew/bin/qemu-aarch64",
        "/usr/local/bin/qemu-aarch64",
        "/usr/bin/qemu-aarch64",
    ]:
        if os.path.exists(path):
            return path
    # Try PATH
    try:
        result = subprocess.run(["which", "qemu-aarch64"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def run_on_qemu(argv, fs_files=None, max_instructions=50000):
    """
    Run BusyBox command on QEMU with instruction tracing.

    Returns list of (pc, instruction_hex) tuples from QEMU trace.
    """
    qemu = find_qemu()
    if not qemu:
        print("WARNING: qemu-aarch64 not found. Install with: brew install qemu")
        return None

    # Create temp directory with filesystem files
    with tempfile.TemporaryDirectory() as tmpdir:
        if fs_files:
            for path, content in fs_files.items():
                full_path = os.path.join(tmpdir, path.lstrip('/'))
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)

        # Build QEMU command with instruction trace
        cmd = [
            qemu,
            "-d", "in_asm",  # Dump input assembly
            "-D", os.path.join(tmpdir, "qemu_trace.log"),
            "-singlestep",   # Trace every instruction
            str(BUSYBOX),
        ] + argv

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
                env={**os.environ, "QEMU_SET_ENV": "PATH=/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            print(f"QEMU timed out after 10s")
            return None
        except Exception as e:
            print(f"QEMU error: {e}")
            return None

        # Parse QEMU trace log
        trace_file = os.path.join(tmpdir, "qemu_trace.log")
        if not os.path.exists(trace_file):
            print("No QEMU trace file generated")
            return None

        entries = []
        # QEMU in_asm format: "0x0000000000400abc:  d2800540      movz x0, #0x2a"
        pc_pattern = re.compile(r'0x([0-9a-fA-F]+):\s+([0-9a-fA-F]+)\s+(.*)')

        with open(trace_file) as f:
            for line in f:
                m = pc_pattern.match(line.strip())
                if m:
                    pc = int(m.group(1), 16)
                    inst_hex = int(m.group(2), 16)
                    disasm = m.group(3).strip()
                    entries.append((pc, inst_hex, disasm))
                    if len(entries) >= max_instructions:
                        break

        return entries, result.stdout


def run_on_gpu(argv, fs, max_cycles=200000):
    """
    Run BusyBox command on GPU with instruction tracing.

    Returns list of (pc, inst, x0, x1, x2, x3) tuples from GPU trace.
    """
    cpu = create_gpu_cpu(quiet=True)

    entry = load_elf_into_memory(cpu, str(BUSYBOX), argv=argv, quiet=True)
    elf_data = BUSYBOX.read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    cpu.set_pc(entry)
    cpu.init_svc_buffer()
    cpu.enable_trace()

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    total_cycles = 0
    output_parts = []

    while total_cycles < max_cycles:
        result = cpu.execute(max_cycles=10000)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            output_parts.append(data.decode('ascii', errors='replace'))

        if result.stop_reason == StopReasonV2.HALT:
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)

    trace = cpu.read_trace()
    cpu.disable_trace()

    return trace, ''.join(output_parts), total_cycles


def compare_traces(gpu_trace, qemu_trace):
    """
    Compare GPU and QEMU traces to find first divergence.

    Returns (divergence_index, gpu_entry, qemu_entry) or None if identical.
    """
    if qemu_trace is None:
        print("Cannot compare — QEMU trace not available")
        return None

    min_len = min(len(gpu_trace), len(qemu_trace))
    print(f"\nComparing {min_len} instructions (GPU has {len(gpu_trace)}, QEMU has {len(qemu_trace)})")

    divergences = []
    for i in range(min_len):
        gpu_pc = gpu_trace[i][0]
        gpu_inst = gpu_trace[i][1]
        qemu_pc = qemu_trace[i][0]
        qemu_inst = qemu_trace[i][1]

        if gpu_pc != qemu_pc or gpu_inst != qemu_inst:
            divergences.append((i, gpu_trace[i], qemu_trace[i]))
            if len(divergences) >= 20:  # Show first 20
                break

    return divergences


def main():
    if len(sys.argv) < 2:
        print("Usage: gpu_vs_cpu_compare.py <command> [args...]")
        print("Example: gpu_vs_cpu_compare.py echo hello")
        print("         gpu_vs_cpu_compare.py basename /usr/local/bin/python3")
        sys.exit(1)

    argv = sys.argv[1:]

    print("=" * 70)
    print("  GPU vs CPU EXECUTION COMPARISON")
    print(f"  Command: {' '.join(argv)}")
    print("=" * 70)
    print()

    # Setup filesystem
    fs = GPUFilesystem()
    fs_files = {
        '/etc/passwd': "root:x:0:0:root:/root:/bin/sh\n"
                       "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
                       "hello:x:1000:1000:hello:/home/hello:/bin/sh\n",
    }
    for path, content in fs_files.items():
        fs.write_file(path, content)

    # Run on GPU
    print("[1/3] Running on GPU (nCPU Metal kernel)...")
    gpu_trace, gpu_output, gpu_cycles = run_on_gpu(argv, fs)
    print(f"  GPU: {len(gpu_trace)} trace entries, {gpu_cycles:,} cycles")
    if gpu_output.strip():
        print(f"  Output: {gpu_output.strip()[:100]}")

    # Run on QEMU
    print("\n[2/3] Running on QEMU (native aarch64)...")
    qemu_result = run_on_qemu(argv, fs_files)
    if qemu_result:
        qemu_trace, qemu_output = qemu_result
        print(f"  QEMU: {len(qemu_trace)} trace entries")
        if qemu_output.strip():
            print(f"  Output: {qemu_output.strip()[:100]}")
    else:
        qemu_trace = None

    # Compare
    print(f"\n[3/3] Comparing execution traces...")
    if qemu_trace is None:
        print("\n  QEMU not available — showing GPU trace analysis only")
        print(f"\n  GPU Trace: First 20 instructions")
        for i, (pc, inst, x0, x1, x2, x3, *_rest) in enumerate(gpu_trace[:20]):
            from ncpu.os.gpu.programs.tools.trace_grep_regex import decode_arm64
            decoded = decode_arm64(inst)
            print(f"    [{i:3d}] 0x{pc:08X}: {decoded}")
    else:
        divergences = compare_traces(gpu_trace, qemu_trace)
        if not divergences:
            print("\n  PERFECT MATCH — GPU and CPU execution are identical!")
        else:
            print(f"\n  DIVERGENCES FOUND: {len(divergences)}")
            print()
            print(f"  {'#':>3s}  {'GPU PC':>12s} {'GPU Inst':>12s}  {'QEMU PC':>12s} {'QEMU Inst':>12s}  QEMU Disasm")
            print(f"  {'─'*80}")
            for idx, gpu_entry, qemu_entry in divergences:
                gpu_pc, gpu_inst = gpu_entry[0], gpu_entry[1]
                qemu_pc, qemu_inst, qemu_disasm = qemu_entry
                match = "OK" if gpu_pc == qemu_pc and gpu_inst == qemu_inst else "**DIFF**"
                print(f"  [{idx:3d}] 0x{gpu_pc:08X} 0x{gpu_inst:08X}  "
                      f"0x{qemu_pc:08X} 0x{qemu_inst:08X}  {qemu_disasm}  {match}")

            # Context around first divergence
            first_div = divergences[0][0]
            print(f"\n  Context around first divergence (instruction #{first_div}):")
            start = max(0, first_div - 5)
            end = min(min(len(gpu_trace), len(qemu_trace)), first_div + 5)
            for i in range(start, end):
                marker = ">>>" if i == first_div else "   "
                gpu_pc, gpu_inst = gpu_trace[i][0], gpu_trace[i][1]
                qemu_pc, qemu_inst, qemu_disasm = qemu_trace[i]
                match_pc = "=" if gpu_pc == qemu_pc else "!"
                match_inst = "=" if gpu_inst == qemu_inst else "!"
                print(f"  {marker} [{i:3d}] GPU 0x{gpu_pc:08X}:{gpu_inst:08X} {match_pc}{match_inst} "
                      f"QEMU 0x{qemu_pc:08X}:{qemu_inst:08X} {qemu_disasm}")


if __name__ == "__main__":
    main()
