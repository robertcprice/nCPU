#!/usr/bin/env python3
"""nCPU Neural Computer — One-command demo launcher (LEGACY PATH).

⚠️ Preferred entry point is now `python -m ncpu gpu` (the true hero experience)
   or `python -m ncpu lab` / `ncpu-lab`.

This module still works for the neural display demos, but new users should start with:
    python -m ncpu gpu                 # THE HERO: GPU as complete self-sufficient computer
    python -m ncpu gpu --native        # Raw maximum speed native binary

Boots the neural-enhanced GPU UNIX OS with live neural display rendering.
Every pixel on screen is produced by trained neural networks.

Usage (legacy):
    python -m ncpu demo              # Interactive shell with neural display
    ...
"""

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    old_argv = sys.argv[:]
    prog = sys.argv[0] if sys.argv else "ncpu"

    try:
        if "--help" in args or "-h" in args:
            print(__doc__.strip())
            return 0

        # ── Strict bottom-up full-neural path ────────────────────────────
        if "--full-neural" in args or "--bottom-up" in args:
            forwarded = [prog]
            for a in args:
                if a in {"--full-neural", "--bottom-up"}:
                    continue
                forwarded.append(a)
            sys.argv = forwarded

            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │       nCPU — Bottom-Up Full Neural Demo         │")
            print("  │                                                  │")
            print("  │  Decode: neural                                  │")
            print("  │  ALU:    neural specialist models                │")
            print("  │  Display: neural terminal renderer               │")
            print("  └─────────────────────────────────────────────────┘")
            print()

            runpy.run_path(
                str(PROJECT_ROOT / "demos" / "neural" / "full_neural_demo.py"),
                run_name="__main__",
            )
            return 0

        # ── Side-by-side comparison demo ────────────────────────────────
        if "--meta-compare" in args:
            forwarded = [prog]
            for a in args:
                if a == "--meta-compare":
                    continue
                forwarded.append(a)
            sys.argv = forwarded

            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │     nCPU — Neural Screen vs Meta Comparison     │")
            print("  │                                                  │")
            print("  │  Left:  real shell + neural-rendered pixels      │")
            print("  │  Right: reference comparison terminal            │")
            print("  │  Goal:  audit the full-neural screen claim       │")
            print("  └─────────────────────────────────────────────────┘")
            print()

            runpy.run_path(
                str(PROJECT_ROOT / "demos" / "neural" / "meta_comparison_demo.py"),
                run_name="__main__",
            )
            return 0

        # ── Live mode: pygame window with neural display ──────────────────
        if "--live" in args:
            live_args = [prog]
            for a in args:
                if a == "--live":
                    continue
                live_args.append(a)
            sys.argv = live_args

            from ncpu.os.gpu.neural_live import main as live_main

            live_main()
            return 0

        # ── Standard mode: terminal-based neural demo ─────────────────────
        demo_args = []

        if "--headless" in args:
            demo_args.append("--demo")
        elif "--script" in args:
            demo_args.append("--demo")

        if "--multiproc" in args or "-m" in args:
            demo_args.append("--multiproc")

        sys.argv = [prog] + demo_args

        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │         nCPU — Neural Computer v3.1              │")
        print("  │                                                  │")
        print("  │  Every OS decision: trained neural network       │")
        print("  │  Every pixel: neural glyph MLP (390K params)     │")
        print("  │  Execution: ARM64 on Apple Silicon Metal GPU     │")
        print("  │                                                  │")
        print("  │  9 neural models · 62K+ IPS · 29 dB display     │")
        print("  └─────────────────────────────────────────────────┘")
        print()

        from ncpu.os.gpu.neural_demo import main as neural_main

        neural_main()
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
