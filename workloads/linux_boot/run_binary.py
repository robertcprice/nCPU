#!/usr/bin/env python3
"""
Comprehensive binary runner with full Linux syscall emulation.

This integrates the GPU neural CPU with the full LinuxSyscallHandler
for running Linux ARM64 static binaries.

Usage:
    python run_binary.py binaries/alpine-echo "hello world"
    python run_binary.py binaries/alpine-hello
"""
import sys
import os
import torch
from neural_kernel import NeuralARM64Kernel

def run_binary(binary_path: str, args: list = None, max_batches: int = 10000, verbose: bool = False):
    """
    Run an ARM64 Linux binary with full syscall emulation.

    Args:
        binary_path: Path to the binary
        args: Command line arguments (binary name is automatically prepended)
        max_batches: Maximum execution batches
        verbose: Print execution trace

    Returns:
        (exit_code, total_instructions, output_text)
    """
    if args is None:
        args = []

    # Initialize kernel
    kernel = NeuralARM64Kernel()

    # Load binary
    with open(binary_path, 'rb') as f:
        binary = f.read()

    if verbose:
        print(f"Binary: {binary_path} ({len(binary):,} bytes)")

    device = kernel.cpu.device
    kernel.cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
    kernel.cpu.regs.zero_()
    result = kernel.load_elf(binary)

    entry = result['entry_point']
    if verbose:
        print(f"Entry point: 0x{entry:X}")

    kernel.cpu.pc = torch.tensor(entry, dtype=torch.int64, device=device)

    # Setup stack and arguments
    # Full argv: [binary_name, arg1, arg2, ...]
    binary_name = os.path.basename(binary_path)
    full_argv = [binary_name] + args

    stack_base = 0x100000
    string_area = 0x200000  # Far from stack

    # Write argument strings
    arg_ptrs = []
    str_ptr = string_area
    for arg in full_argv:
        arg_bytes = arg.encode('utf-8') + b'\x00'
        kernel.cpu.memory[str_ptr:str_ptr + len(arg_bytes)] = torch.tensor(
            list(arg_bytes), dtype=torch.uint8, device=device
        )
        arg_ptrs.append(str_ptr)
        str_ptr += len(arg_bytes)

    # Write environment strings (minimal)
    env_ptrs = []
    env_vars = ["PATH=/bin:/usr/bin", "HOME=/root", "USER=root"]
    for env in env_vars:
        env_bytes = env.encode('utf-8') + b'\x00'
        kernel.cpu.memory[str_ptr:str_ptr + len(env_bytes)] = torch.tensor(
            list(env_bytes), dtype=torch.uint8, device=device
        )
        env_ptrs.append(str_ptr)
        str_ptr += len(env_bytes)

    # Auxiliary vector with AT_RANDOM
    random_ptr = (str_ptr + 15) & ~15
    kernel.cpu.memory[random_ptr:random_ptr+16] = torch.randint(
        0, 256, (16,), dtype=torch.uint8, device=device
    )

    auxv_entries = [
        (25, random_ptr),    # AT_RANDOM - pointer to 16 random bytes
        (6, 4096),           # AT_PAGESZ
        (9, entry),          # AT_ENTRY
        (3, 0x40),           # AT_PHDR
        (4, 56),             # AT_PHENT
        (5, 4),              # AT_PHNUM
        (23, 0),             # AT_SECURE
        (0, 0),              # AT_NULL
    ]

    # Calculate stack frame size
    frame_size = (
        8 +                          # argc
        len(arg_ptrs) * 8 + 8 +      # argv + NULL
        len(env_ptrs) * 8 + 8 +      # envp + NULL
        len(auxv_entries) * 16       # auxv
    )
    frame_size = (frame_size + 15) & ~15  # 16-byte aligned
    sp = stack_base - frame_size

    # Zero stack region
    kernel.cpu.memory[0xFE000:stack_base] = torch.zeros(
        stack_base - 0xFE000, dtype=torch.uint8, device=device
    )

    # Write stack frame
    write_ptr = sp

    # argc
    kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(
        list(len(full_argv).to_bytes(8, 'little')), dtype=torch.uint8, device=device
    )
    write_ptr += 8

    # argv pointers
    for ptr in arg_ptrs:
        kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(
            list(ptr.to_bytes(8, 'little')), dtype=torch.uint8, device=device
        )
        write_ptr += 8
    kernel.cpu.memory[write_ptr:write_ptr+8] = torch.zeros(8, dtype=torch.uint8, device=device)
    write_ptr += 8

    # envp pointers
    for ptr in env_ptrs:
        kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(
            list(ptr.to_bytes(8, 'little')), dtype=torch.uint8, device=device
        )
        write_ptr += 8
    kernel.cpu.memory[write_ptr:write_ptr+8] = torch.zeros(8, dtype=torch.uint8, device=device)
    write_ptr += 8

    # auxv
    for a_type, a_val in auxv_entries:
        kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(
            list(a_type.to_bytes(8, 'little')), dtype=torch.uint8, device=device
        )
        kernel.cpu.memory[write_ptr+8:write_ptr+16] = torch.tensor(
            list(a_val.to_bytes(8, 'little')), dtype=torch.uint8, device=device
        )
        write_ptr += 16

    kernel.cpu.regs[31] = sp

    if verbose:
        print(f"Stack pointer: 0x{sp:X}")
        print(f"Arguments: {full_argv}")

    # Initialize syscall handler (uses kernel's LinuxSyscallHandler)
    syscall_handler = kernel.linux_syscalls

    # Execution loop
    total = 0
    output_text = []
    exit_code = 0

    for batch in range(max_batches):
        executed_t, _ = kernel.cpu.run_parallel_gpu(max_instructions=10000, batch_size=10000)
        executed = int(executed_t.item())
        total += executed

        if kernel.cpu._svc_t.item():
            kernel.cpu._svc_t.zero_()  # Reset SVC flag

            x8 = int(kernel.cpu.regs[8].item())  # syscall number

            # Use full syscall handler
            continue_exec, ret = syscall_handler.handle(x8)

            # Capture stdout/stderr output
            if x8 == 64:  # write
                x0_fd = int(kernel.cpu.regs[0].item())
                # Output already printed by handler, but track it

            if not continue_exec:
                exit_code = int(kernel.cpu.regs[0].item())
                if verbose:
                    print(f"Exit code: {exit_code}")
                break

        if executed == 0 or kernel.cpu.halted:
            if verbose:
                print(f"Halted after {total:,} instructions")
            break

    if verbose:
        print(f"Total instructions: {total:,}")

    return exit_code, total


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_binary.py <binary> [args...]")
        print("Examples:")
        print("  python run_binary.py binaries/alpine-hello")
        print("  python run_binary.py binaries/alpine-echo hello world")
        sys.exit(1)

    binary_path = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []

    exit_code, total = run_binary(binary_path, args, verbose=True)

    print(f"\n{'='*60}")
    print(f"Exit: {exit_code}, Instructions: {total:,}")
    print(f"{'='*60}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
