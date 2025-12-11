#!/usr/bin/env python3
"""KVRM-CPU Command Line Interface.

Run assembly programs with the KVRM-CPU emulator.

Usage:
    python main.py --program programs/sum_1_to_10.asm
    python main.py --program programs/fibonacci.asm --mode real --model models/decode_llm
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kvrm_cpu import KVRMCPU


def main():
    parser = argparse.ArgumentParser(
        description="KVRM-CPU: Model-Native CPU Emulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run sum_1_to_10 program with mock decoder
    python main.py --program programs/sum_1_to_10.asm

    # Run fibonacci with full trace output
    python main.py --program programs/fibonacci.asm --trace

    # Run with trained decode model
    python main.py --program programs/multiply.asm --mode real --model models/decode_llm

    # Run inline assembly
    python main.py --inline "MOV R0, 42; HALT"
        """
    )

    parser.add_argument(
        "--program", "-p",
        type=str,
        help="Path to assembly program file (.asm)"
    )
    parser.add_argument(
        "--inline", "-i",
        type=str,
        help="Inline assembly (separate instructions with ;)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["mock", "real"],
        default="mock",
        help="Decoder mode: mock (rule-based) or real (LLM). Default: mock"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained decode model (required for real mode)"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10000,
        help="Maximum execution cycles (safety limit). Default: 10000"
    )
    parser.add_argument(
        "--trace", "-t",
        action="store_true",
        help="Print full execution trace"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (final registers only)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.program and not args.inline:
        parser.error("Either --program or --inline is required")

    if args.mode == "real" and not args.model:
        parser.error("--model is required when mode is 'real'")

    # Initialize CPU
    mock_mode = args.mode == "mock"
    cpu = KVRMCPU(
        mock_mode=mock_mode,
        model_path=args.model,
        max_cycles=args.max_cycles
    )

    # Load model for real mode
    if not mock_mode:
        print(f"Loading decode model from {args.model}...")
        cpu.load()

    # Load program
    if args.program:
        program_path = Path(args.program)
        if not program_path.exists():
            print(f"Error: Program file not found: {args.program}")
            sys.exit(1)
        source = program_path.read_text()
        if not args.quiet:
            print(f"Loading program: {args.program}")
    else:
        # Inline assembly
        source = args.inline.replace(";", "\n")
        if not args.quiet:
            print("Running inline assembly")

    cpu.load_program(source)

    # Run
    if not args.quiet:
        print("-" * 60)
        print("Executing...")
        print("-" * 60)

    try:
        cpu.run()
    except RuntimeError as e:
        print(f"Execution error: {e}")

    # Output
    if args.trace:
        cpu.print_trace()
    elif not args.quiet:
        print()
        summary = cpu.get_summary()
        print(f"Cycles: {summary['cycles']}")
        print(f"Halted: {summary['halted']}")
        print(f"Registers: {summary['registers']}")
        print(f"Flags: {summary['flags']}")
        if summary['errors']:
            print(f"Errors: {summary['errors']}")
    else:
        # Quiet mode - just print final registers
        regs = cpu.dump_registers()
        for reg in sorted(regs.keys()):
            if regs[reg] != 0:
                print(f"{reg}={regs[reg]}")

    # Unload model
    if not mock_mode:
        cpu.unload()

    # Return exit code based on halted state
    return 0 if cpu.is_halted() else 1


if __name__ == "__main__":
    sys.exit(main())
