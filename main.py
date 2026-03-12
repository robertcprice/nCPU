#!/usr/bin/env python3
"""nCPU — Neural CPU Command Line Interface.

Run programs on a CPU where every ALU operation is a trained neural network.

Usage:
    python main.py --program programs/sum_1_to_10.asm
    python main.py --program programs/fibonacci.asm --trace
    python main.py --inline "MOV R0, 42; HALT"
    python main.py --binary firmware.bin --fast
    python main.py --program programs/fibonacci.asm --compute
"""

import argparse
import sys
from pathlib import Path


def run_neural(args):
    """Run a text assembly program with neural ALU (all ops through trained .pt models)."""
    from ncpu.model import CPU

    cpu = CPU(
        mock_mode=True,
        neural_execution=True,
        models_dir=args.models_dir,
        max_cycles=args.max_cycles,
    )

    if args.program:
        source = Path(args.program).read_text()
        if not args.quiet:
            print(f"Loading program: {args.program}")
    else:
        source = args.inline.replace(";", "\n")
        if not args.quiet:
            print("Running inline assembly")

    cpu.load_program(source)

    if not args.quiet:
        print("Neural ALU: all operations through trained .pt models")
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    try:
        cpu.run()
    except RuntimeError as e:
        print(f"Execution error: {e}")

    if args.trace:
        cpu.print_trace()
    elif not args.quiet:
        summary = cpu.get_summary()
        print(f"\nCycles: {summary['cycles']}")
        print(f"Halted: {summary['halted']}")
        print(f"Registers: {summary['registers']}")
        print(f"Flags: {summary['flags']}")
        if summary['errors']:
            print(f"Errors: {summary['errors']}")
    else:
        regs = cpu.dump_registers()
        for reg in sorted(regs.keys()):
            if regs[reg] != 0:
                print(f"{reg}={regs[reg]}")

    return 0 if cpu.is_halted() else 1


def run_compute(args):
    """Run on GPU compute shader (Metal) — qemu-style fetch-decode-execute on GPU."""
    from kernels.mlx.ncpu_kernel import NCPUComputeKernel

    kernel = NCPUComputeKernel()

    if args.binary:
        # ARM64 binary → use full 125-instruction MLX ARM64 kernel V2
        from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2
        cpu = MLXKernelCPUv2()
        binary_data = Path(args.binary).read_bytes()
        cpu.load_program(binary_data, address=0)
        cpu.set_pc(0)
        if not args.quiet:
            print(f"Loaded ARM64 binary: {args.binary} ({len(binary_data)} bytes)")
            print("GPU Compute: ARM64 Metal kernel V2 (125 instructions, qemu-style)")
            print("-" * 60)
            print("Executing...")
            print("-" * 60)

        result = cpu.execute(max_cycles=args.max_cycles)

        if not args.quiet:
            print(f"\nCycles: {result.cycles:,}")
            print(f"Elapsed: {result.elapsed_seconds * 1000:.2f} ms")
            print(f"IPS: {result.ips:,.0f}")
            print(f"Stop reason: {result.stop_reason_name}")
            for i in range(31):
                val = cpu.get_register(i)
                if val != 0:
                    print(f"  X{i} = {val}")
        return 0

    # nCPU ISA → assemble and run on Metal compute shader
    if args.program:
        source = Path(args.program).read_text()
        if not args.quiet:
            print(f"Loading program: {args.program}")
    elif args.inline:
        source = args.inline.replace(";", "\n")
        if not args.quiet:
            print("Running inline assembly")
    else:
        print("Error: --compute requires --program, --inline, or --binary")
        return 1

    kernel.load_program_from_asm(source)

    if not args.quiet:
        print("GPU Compute: nCPU ISA Metal kernel (qemu-style)")
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    result = kernel.execute(max_cycles=args.max_cycles)

    if not args.quiet:
        print(f"\nCycles: {result.cycles:,}")
        print(f"Elapsed: {result.elapsed_seconds * 1000:.3f} ms")
        print(f"IPS: {result.ips:,.0f}")
        print(f"Stop reason: {result.stop_reason_name}")
        regs = kernel.get_registers_dict()
        flags = kernel.get_flags()
        for name, val in sorted(regs.items()):
            if val != 0:
                print(f"  {name} = {val}")
        print(f"  Flags: ZF={int(flags['ZF'])} SF={int(flags['SF'])}")
    else:
        regs = kernel.get_registers_dict()
        for name in sorted(regs.keys()):
            if regs[name] != 0:
                print(f"{name}={regs[name]}")

    return 0


def run_fast(args):
    """Run on the GPU tensor CPU (NeuralCPU) — native tensor ops, maximum speed."""
    from ncpu.neural import NeuralCPU

    device = args.device or "cpu"  # auto-detect later if needed

    cpu = NeuralCPU(device_override=device, fast_mode=True)

    if args.binary:
        binary_data = Path(args.binary).read_bytes()
        cpu.load_binary(binary_data)
        if not args.quiet:
            print(f"Loaded binary: {args.binary} ({len(binary_data)} bytes)")
    elif args.program:
        # For text assembly in fast mode, assemble to ARM64 binary first
        print("Error: --fast mode requires --binary (ARM64 binary). Text assembly uses neural mode.")
        print("Usage: python main.py --program programs/fibonacci.asm  (neural mode)")
        print("       python main.py --binary firmware.bin --fast       (GPU tensor mode)")
        return 1
    else:
        print("Error: --fast mode requires --binary")
        return 1

    if not args.quiet:
        print(f"GPU Tensor CPU: native tensor ops on {device}")
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    # Run
    cycles = 0
    max_cycles = args.max_cycles
    while cycles < max_cycles:
        result = cpu.step()
        cycles += 1
        if not result:
            break

    if not args.quiet:
        print(f"\nCycles: {cycles}")
        # Show non-zero registers
        for i in range(31):
            val = int(cpu.regs[i].item())
            if val != 0:
                print(f"  X{i} = {val}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="nCPU: A CPU where every component is a trained neural network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with neural ALU (default — all ops through trained .pt models)
    python main.py --program programs/sum_1_to_10.asm

    # Run with trace output
    python main.py --program programs/fibonacci.asm --trace

    # Run inline assembly
    python main.py --inline "MOV R0, 42; HALT"

    # Run ARM64 binary on GPU tensor CPU (maximum speed)
    python main.py --binary firmware.bin --fast

    # Run on GPU compute shader (Metal) — qemu-style, millions of IPS
    python main.py --program programs/sum_1_to_10.asm --compute

    # ARM64 binary on Metal compute shader
    python main.py --binary firmware.bin --compute
        """
    )

    parser.add_argument("--program", "-p", type=str, help="Path to assembly program (.asm)")
    parser.add_argument("--inline", "-i", type=str, help="Inline assembly (separate with ;)")
    parser.add_argument("--binary", "-b", type=str, help="Path to ARM64 binary (for --fast mode)")
    parser.add_argument("--fast", action="store_true",
                        help="GPU tensor mode: native tensor ops, maximum speed (requires --binary)")
    parser.add_argument("--compute", action="store_true",
                        help="GPU compute mode: Metal shader execution (nCPU ISA or ARM64 binary)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps (default: auto-detect)")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Path to trained .pt models")
    parser.add_argument("--max-cycles", type=int, default=10000,
                        help="Max cycles (default: 10000)")
    parser.add_argument("--trace", "-t", action="store_true",
                        help="Print full execution trace")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    if not args.program and not args.inline and not args.binary:
        parser.error("One of --program, --inline, or --binary is required")

    if args.compute:
        return run_compute(args)
    elif args.fast:
        return run_fast(args)
    else:
        if args.binary:
            parser.error("--binary requires --fast or --compute flag")
        if not args.program and not args.inline:
            parser.error("Neural mode requires --program or --inline")
        return run_neural(args)


if __name__ == "__main__":
    sys.exit(main())
