#!/usr/bin/env python3
"""Unified flagship launcher for nCPU interactive demos.

Preferred starting point for the hero experience (GPU as self-sufficient computer).
New users should almost always start with `python -m ncpu gpu` or `python -m ncpu lab`.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import runpy
import sys
import textwrap
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

DEMO_REGISTRY = {
    "discover": {
        "title": "Interactive Discovery REPL",
        "category": "Flagship interactive",
        "command": "python -m ncpu discover",
        "description": "Type examples and watch differentiable program synthesis discover executable programs live.",
        "highlights": [
            "Program by examples",
            "Live synthesis with gradient descent",
            "Immediate test-on-new-input workflow",
        ],
        "best_platform": "Cross-platform",
        "weight": "light",
        "script": DEMOS_DIR / "showcase" / "interactive_discovery.py",
        "default_argv": ["interactive_discovery.py"],
    },
    "text": {
        "title": "Neural Text Machine",
        "category": "Flagship interactive",
        "command": "python -m ncpu text --interactive",
        "description": "Discover ciphers, text transforms, and character-sequence programs from examples.",
        "highlights": [
            "Cipher discovery and cracking",
            "Character-sequence learning",
            "Text transformation through differentiable execution",
        ],
        "best_platform": "Cross-platform",
        "weight": "light",
        "script": DEMOS_DIR / "showcase" / "neural_text_machine.py",
        "default_argv": ["neural_text_machine.py", "--interactive"],
    },
    "busybox": {
        "title": "GPU BusyBox Shell",
        "category": "Systems wow",
        "command": "python -m ncpu systems busybox --interactive",
        "description": "Interactive BusyBox shell running through the GPU-native systems path.",
        "highlights": [
            "GPU shell experience",
            "Good next step after flagship demos",
            "Best on macOS / Apple Silicon",
        ],
        "best_platform": "macOS / Apple Silicon",
        "weight": "medium",
        "script": DEMOS_DIR / "gpu" / "busybox_gpu_demo.py",
        "default_argv": ["busybox_gpu_demo.py", "--interactive"],
    },
    "alpine": {
        "title": "Alpine Linux on GPU",
        "category": "Systems wow",
        "command": "python -m ncpu systems alpine --demo",
        "description": "Boot the Alpine Linux demo on the Metal GPU path.",
        "highlights": [
            "Fuller systems story",
            "Linux-on-GPU showcase",
            "Best on macOS / Apple Silicon",
        ],
        "best_platform": "macOS / Apple Silicon",
        "weight": "medium",
        "script": DEMOS_DIR / "gpu" / "alpine_gpu.py",
        "default_argv": ["alpine_gpu.py", "--demo"],
    },
    "full-neural": {
        "title": "Bottom-Up Full Neural Demo",
        "category": "Research depth",
        "command": "python -m ncpu full-neural",
        "description": "Strict bottom-up neural path: decode, ALU, memory, and terminal display all run through the neural stack rather than the Metal OS fast path.",
        "highlights": [
            "Best paper-facing proof of the full-neural claim",
            "Neural CPU woven execution plus neural terminal renderer",
            "Cross-platform with the trained model stack",
        ],
        "best_platform": "Cross-platform with model stack",
        "weight": "medium",
        "script": DEMOS_DIR / "neural" / "full_neural_demo.py",
        "default_argv": ["full_neural_demo.py"],
    },
    "meta-compare": {
        "title": "Neural Screen vs Meta Comparison",
        "category": "Research depth",
        "command": "python -m ncpu meta-compare",
        "description": "Side-by-side interactive comparison: a real PTY shell with neural-rendered pixels on the left and a neural-rendered reference panel comparing nCPU to the Neural Computers paper on the right.",
        "highlights": [
            "Left pane is a real interactive shell",
            "Visible content area is neural-rendered terminal pixels",
            "Paper-facing comparison against the Neural Computers framing",
        ],
        "best_platform": "Cross-platform with model stack",
        "weight": "medium",
        "script": DEMOS_DIR / "neural" / "meta_comparison_demo.py",
        "default_argv": ["meta_comparison_demo.py"],
    },
    "coprocessor": {
        "title": "Code in Brain / Coprocessor Demo",
        "category": "Research depth",
        "command": "python -m ncpu coprocessor --help-only",
        "description": "Arithmetic coprocessor injected into a language model forward pass.",
        "highlights": [
            "LLM + arithmetic integration",
            "Heavier dependencies",
            "Best after basic repo orientation",
        ],
        "best_platform": "Cross-platform with model stack",
        "weight": "heavy",
        "script": DEMOS_DIR / "showcase" / "demo_code_in_brain.py",
        "default_argv": ["demo_code_in_brain.py", "--help"],
    },
}

CURATED_ORDER = ["discover", "text", "busybox", "alpine", "full-neural", "meta-compare", "coprocessor"]
CATEGORY_ORDER = ["Flagship interactive", "Systems wow", "Research depth"]


def _grouped_demo_names() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    for key in CURATED_ORDER:
        grouped.setdefault(DEMO_REGISTRY[key]["category"], []).append(key)
    return grouped


def _require_demo_script(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Demo script not found: {path}\n"
            "The ncpu-lab launcher currently expects the repo checkout layout. "
            "Run from the repository root with an editable install."
        )


def _run_script(script: Path, argv: list[str]) -> None:
    _require_demo_script(script)
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def _run_demo(name: str, argv: list[str] | None = None) -> int:
    demo = DEMO_REGISTRY[name]
    _run_script(demo["script"], argv or demo["default_argv"])
    return 0


def _demo_exists(name: str) -> bool:
    return DEMO_REGISTRY[name]["script"].exists()


def _next_demo_name(name: str) -> str | None:
    try:
        idx = CURATED_ORDER.index(name)
    except ValueError:
        return None
    if idx + 1 < len(CURATED_ORDER):
        return CURATED_ORDER[idx + 1]
    return None


def _print_demo_summary(name: str) -> None:
    demo = DEMO_REGISTRY[name]
    print(f"{demo['title']} ({name})")
    print("-" * 72)
    print(f"Category:          {demo['category']}")
    print(f"Best platform:     {demo['best_platform']}")
    print(f"Dependency weight: {demo['weight']}")
    print(f"Command:           {demo['command']}")
    print(f"Script:            {demo['script']}")
    print(f"Available:         {'yes' if _demo_exists(name) else 'no'}")
    print()
    print(textwrap.fill(demo["description"], width=72))
    print()
    print("Highlights:")
    for item in demo["highlights"]:
        print(f"- {item}")
    print()
    print("Copy-paste run command:")
    print(f"  {demo['command']}")
    next_name = _next_demo_name(name)
    if next_name is not None:
        next_demo = DEMO_REGISTRY[next_name]
        print()
        print("Next suggested step:")
        print(f"  {next_demo['title']} -> python -m ncpu info {next_name}")


def cmd_discover(_args: argparse.Namespace) -> int:
    return _run_demo("discover")


def cmd_text(args: argparse.Namespace) -> int:
    argv = ["neural_text_machine.py"]
    if args.interactive:
        argv.append("--interactive")
    return _run_demo("text", argv)


def cmd_systems(args: argparse.Namespace) -> int:
    if args.demo == "busybox":
        argv = ["busybox_gpu_demo.py"]
        if args.interactive:
            argv.append("--interactive")
        return _run_demo("busybox", argv)
    if args.demo == "alpine":
        argv = ["alpine_gpu.py"]
        if args.demo_mode:
            argv.append("--demo")
        return _run_demo("alpine", argv)
    raise SystemExit(f"Unknown systems demo: {args.demo}")


def cmd_coprocessor(args: argparse.Namespace) -> int:
    argv = ["demo_code_in_brain.py"]
    if args.help_only:
        argv.append("--help")
    return _run_demo("coprocessor", argv)


def cmd_full_neural(args: argparse.Namespace) -> int:
    argv = ["full_neural_demo.py"]
    if args.device:
        argv.extend(["--device", args.device])
    if args.max_instructions is not None:
        argv.extend(["--max-instructions", str(args.max_instructions)])
    if args.output:
        argv.extend(["--output", args.output])
    if args.summary_json:
        argv.extend(["--summary-json", args.summary_json])
    return _run_demo("full-neural", argv)


def cmd_meta_compare(args: argparse.Namespace) -> int:
    argv = ["meta_comparison_demo.py"]
    if args.left_runtime:
        argv.extend(["--left-runtime", args.left_runtime])
    if args.shell:
        argv.extend(["--shell", args.shell])
    if args.device:
        argv.extend(["--device", args.device])
    if args.scale is not None:
        argv.extend(["--scale", str(args.scale)])
    for command in args.command or []:
        argv.extend(["--command", command])
    if args.capture_dir:
        argv.extend(["--capture-dir", args.capture_dir])
    if args.summary_json:
        argv.extend(["--summary-json", args.summary_json])
    if args.shell_log:
        argv.extend(["--shell-log", args.shell_log])
    if args.boot_delay_ms is not None:
        argv.extend(["--boot-delay-ms", str(args.boot_delay_ms)])
    if args.step_delay_ms is not None:
        argv.extend(["--step-delay-ms", str(args.step_delay_ms)])
    if args.final_hold_ms is not None:
        argv.extend(["--final-hold-ms", str(args.final_hold_ms)])
    if args.max_frames is not None:
        argv.extend(["--max-frames", str(args.max_frames)])
    if args.output:
        argv.extend(["--output", args.output])
    return _run_demo("meta-compare", argv)


def _print_hero_banner(use_neural: bool, backend_name: str, binary_path: Optional[str] = None):
    """Beautiful hero banner for the flagship GPU experience."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║            ★ nCPU HERO — THE GPU *IS* THE COMPUTER                       ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print("║  ~1.9M IPS  •  σ=0.0 determinism  •  26-command post-mortem debugger     ║")
    print("║  Real BusyBox + Alpine + self-hosting C on Metal GPU                     ")
    if use_neural:
        print("║  Neural ALU weights *inside* the shader (exact + verified)               ")
    print(f"║  Backend: {backend_name:<48}")
    if binary_path:
        print(f"║  Binary : {binary_path[-55:]:<55}")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Core thesis: The AI doesn't run *on* a computer. The AI *is* the computer.")
    print("Transcript: demos/HERO_GPU_DEMO_TRANSCRIPT.md")
    print("Next horizon (active research): Bottom-up full Neural CPU (JEPA/cross-JEPA) — see docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md + ncpu/jepa_neural_cpu/")
    print()


def cmd_gpu(args: argparse.Namespace) -> int:
    """Hero entry point: GPU as self-sufficient computer (Rust Metal primary)."""
    mode = getattr(args, "mode", "shell")
    use_neural = getattr(args, "neural_alu", False) or mode == "neural"

    # === Best-in-class backend detection ===
    rust_python_backend = False
    try:
        from ncpu.os.gpu.rust_backend import run_elf as _  # noqa: F401
        rust_python_backend = True
    except Exception:
        pass

    ncpu_run_path: Optional[str] = None
    try:
        from shutil import which
        ncpu_run_path = which("ncpu_run")
        if not ncpu_run_path:
            candidate = REPO_ROOT / "kernels" / "rust_metal" / "target" / "release" / "ncpu_run"
            if candidate.exists():
                ncpu_run_path = str(candidate)
    except Exception:
        pass

    # Determine the best available backend name for the banner
    if ncpu_run_path:
        backend_name = "Native Rust binary (maximum speed + determinism)"
    elif rust_python_backend:
        backend_name = "Rust Metal Python extension (full shell + VFS)"
    else:
        backend_name = "Pure Python fallback (limited performance)"

    _print_hero_banner(use_neural, backend_name, ncpu_run_path)

    # === Native binary path (raw speed / when user explicitly wants it) ===
    if mode == "native" and ncpu_run_path:
        print("★ Launching NATIVE ncpu_run binary (zero Python overhead — purest + fastest path)")
        print(f"   {ncpu_run_path}\n")
        import subprocess
        try:
            subprocess.run([ncpu_run_path, "--help"])
            return 0
        except Exception as e:
            print(f"Launch failed: {e}")
            return 1

    # Default hero path: always prefer the best *polished interactive* experience
    if mode in ("shell", "busybox"):
        if rust_python_backend:
            # Highest quality interactive experience (full VFS + nice setup)
            pass
        elif ncpu_run_path:
            print("★ Using native ncpu_run (Python Rust extension not installed).")
            print("   For the absolute best interactive GPU shell experience: maturin develop in kernels/rust_metal\n")
            import subprocess
            try:
                busybox_elf = str(DEMOS_DIR / "gpu" / "busybox.elf")
                cmd = [ncpu_run_path]
                if Path(busybox_elf).exists():
                    cmd += ["--elf", busybox_elf, "--rootfs", "--interactive"]
                else:
                    cmd += ["--help"]
                subprocess.run(cmd)
                print("\n(You are now experiencing the purest form of the nCPU hero thesis: the GPU *is* the computer.)")
                return 0
            except Exception as e:
                print(f"Native launch failed: {e}")
                return 1

    # === Full featured shell path (preferred for interactive hero experience) ===
    if mode in ("shell", "busybox"):
        if not rust_python_backend and not ncpu_run_path:
            print("No high-performance GPU backend available.")
            print("Install the Rust extension for the real experience:")
            print("  cd kernels/rust_metal && maturin develop --release")
            return 1

        script = DEMOS_DIR / "gpu" / "busybox_gpu_demo.py"
        argv = ["busybox_gpu_demo.py"]
        if args.interactive or mode == "shell":
            argv.append("--interactive")
        if use_neural:
            argv.extend(["--neural-alu", "1"])
            print("Neural ALU requested — using trained models for arithmetic where available.\n")

        _require_demo_script(script)
        old_argv = sys.argv[:]
        try:
            sys.argv = argv
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = old_argv
        return 0

    if mode == "alpine":
        script = DEMOS_DIR / "gpu" / "alpine_gpu.py"
        argv = ["alpine_gpu.py", "--demo"]
        if use_neural:
            argv.extend(["--neural-alu", "1"])
        print("Booting Alpine Linux on GPU...\n")
        _require_demo_script(script)
        old_argv = sys.argv[:]
        try:
            sys.argv = argv
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = old_argv
        return 0

    if mode in ("debug", "neural"):
        print("Deterministic GPU Post-Mortem Debugging Toolkit")
        print("────────────────────────────────────────────────")
        print("This is one of the strongest differentiators of the nCPU GPU computer:")
        print("  • σ=0.0 cycle variance → perfect replay and diffing")
        print("  • Full state survives process exit")
        print("  • Zero-overhead breakpoints/watchpoints + 26 analysis commands\n")

        if ncpu_run_path:
            print(f"Strongly recommended: {ncpu_run_path}")
            print("Inside the interactive shell you get commands like:")
            print("  gpu-trace, gpu-break, gpu-const-time, gpu-reverse, gpu-sanitize, gpu-fuzz, ...")
        else:
            print("Build the Rust Metal backend to unlock the full toolkit:")
            print("  cd kernels/rust_metal && maturin develop --release")
        print("\nReference: docs/gpu/gpu_debugging_toolkit.md  |  paper/gpu_debugging_toolkit_paper.md")
        return 0

    print(f"Unknown mode: {mode}")
    return 1

    if mode == "alpine":
        script = DEMOS_DIR / "gpu" / "alpine_gpu.py"
        argv = ["alpine_gpu.py", "--demo"]
        print("Launching Alpine Linux on GPU demo...")
        _require_demo_script(script)
        old_argv = sys.argv[:]
        try:
            sys.argv = argv
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = old_argv
        return 0

    if mode in ("debug", "neural"):
        print("GPU Deterministic Debugging Toolkit + Neural ALU mode")
        print()
        print("For the full 26-command post-mortem toolkit (trace, replay, reverse dataflow,")
        print("constant-time verification, memory sanitization, etc.):")
        print("  1. Build the Rust binary:  cd kernels/rust_metal && maturin develop --release")
        print("  2. Run:                    cargo run --bin ncpu_run -- --help   (or the installed ncpu_run)")
        print()
        print("The toolkit is integrated into the Alpine/BusyBox GPU shell when the Rust")
        print("backend is active (commands like gpu-trace, gpu-break, gpu-const-time, etc.).")
        print()
        print("See: docs/gpu/gpu_debugging_toolkit.md and paper/gpu_debugging_toolkit_paper.md")
        return 0

    print(f"Unknown gpu mode: {mode}")
    return 1


def cmd_run(args: argparse.Namespace) -> int:
    if args.name not in DEMO_REGISTRY:
        raise SystemExit(f"Unknown demo '{args.name}'. Use `python -m ncpu demos`.")
    demo = DEMO_REGISTRY[args.name]
    argv = [demo["script"].name, *args.demo_args]
    return _run_demo(args.name, argv)


def cmd_info(args: argparse.Namespace) -> int:
    _print_demo_summary(args.name)
    return 0


def cmd_demos(args: argparse.Namespace) -> int:
    print("nCPU curated demo list")
    print("=" * 72)
    grouped = _grouped_demo_names()
    for category in CATEGORY_ORDER:
        print(category)
        for key in grouped.get(category, []):
            demo = DEMO_REGISTRY[key]
            print(f"  - {key:12s} {demo['title']}")
            if args.verbose:
                print(f"    {demo['description']}")
                print(f"    best platform: {demo['best_platform']} | weight: {demo['weight']}")
        print()
    print("★ HERO (primary thesis — start here for the unique contribution):")
    print("  ncpu gpu                 # GPU as complete computer (Rust Metal, ~1.9M IPS, optional neural ALU in shader)")
    print("  ncpu gpu debug           # 26-command deterministic post-mortem debugging toolkit (impossible on CPU)")
    print("  See: demos/HERO_GPU_DEMO_TRANSCRIPT.md for the exact guided experience")
    print()
    print("High-novelty direction (fast latent world models of the machine):")
    print("  python -m ncpu.world_model.quickstart")
    print()
    print("Run this first (interactive demos):")
    for key in ("discover", "text"):
        demo = DEMO_REGISTRY[key]
        print(f"  {demo['title']}")
        print(f"  {key:12s} {demo['command']}")
        print(f"               {demo['description']}")
    print()
    print("Useful follow-ups:")
    print("  python -m ncpu path")
    print("  python -m ncpu walkthrough")
    print("  python -m ncpu info discover")
    print("  python -m ncpu systems busybox --interactive")
    print("  python -m ncpu systems alpine --demo")
    print("  python -m ncpu full-neural")
    print("  python -m ncpu meta-compare")
    print("  python -m ncpu coprocessor --help-only")
    print("  python -m ncpu run discover")
    return 0


def cmd_path(_args: argparse.Namespace) -> int:
    print("nCPU recommended path")
    print("=" * 72)
    print("★ 0. THE HERO EXPERIENCE (the unique contribution of this project)")
    print("     python -m ncpu gpu                 # GPU as complete self-sufficient computer")
    print("     python -m ncpu gpu --neural-alu    # Same, with trained neural ALU weights inside the Metal shader")
    print("     python -m ncpu gpu debug           # Launch the 26-command deterministic GPU debugging toolkit")
    print()
    print("1. Start with interactive program discovery")
    print("   python -m ncpu discover")
    print()
    print("2. Move to the neural text machine")
    print("   python -m ncpu text --interactive")
    print()
    print("3. If you want the systems story, try BusyBox or Alpine (via the unified gpu command above)")
    print()
    print("4. If you want the strict bottom-up full-neural path (research), run:")
    print("   python -m ncpu full-neural")
    print()
    print("5. If you want the screen-level comparison against Neural Computers, run:")
    print("   python -m ncpu meta-compare")
    print()
    print("6. If you want the LLM/computation story, inspect the coprocessor demo")
    print("   python -m ncpu coprocessor --help-only")
    print()
    print("7. For a guided first-time route, run:")
    print("   python -m ncpu walkthrough")
    return 0


def cmd_walkthrough(_args: argparse.Namespace) -> int:
    print("nCPU walkthrough")
    print("=" * 72)
    print("★ STEP 0 — THE HERO (unique contribution): GPU as the computer")
    print("  Run: python -m ncpu gpu")
    print("  Or for the neural-ALU-in-shader variant: python -m ncpu gpu --neural-alu")
    print("  This is where the 'CPU modeled on GPU with neural networks and kernels' thesis lives.")
    print("  Full OS, real Linux userspace, determinism superpowers, 26-command debugger.")
    print("  Guided transcript: demos/HERO_GPU_DEMO_TRANSCRIPT.md")
    print()
    print("Step 1: program-by-examples")
    print("  Run: python -m ncpu discover")
    print("  Then type:")
    print("    preset add")
    print("    synthesize")
    print("    summary")
    print("    test 15, 25")
    print("    export exports/add_program.asm")
    print()
    print("Step 2: text transformation")
    print("  Run: python -m ncpu text --interactive")
    print("  Then type:")
    print("    cipher hello khoor")
    print("    summary")
    print("    apply world")
    print("    save exports/text_summary.json")
    print()
    print("Step 3: systems story")
    print("  Run: python -m ncpu systems busybox --interactive")
    print("  If you are on Apple Silicon, also try:")
    print("    python -m ncpu systems alpine --demo")
    print()
    print("Step 4: strict full-neural stack")
    print("  Run: python -m ncpu full-neural")
    print("  Use this when you want the bottom-up paper path instead of the")
    print("  neural-enhanced Metal OS path.")
    print()
    print("Step 5: side-by-side comparison against Neural Computers")
    print("  Run: python -m ncpu meta-compare")
    print("  The left pane is a real shell; the visible content area is")
    print("  neural-rendered terminal pixels.")
    print()
    print("Step 6: research depth")
    print("  Run: python -m ncpu coprocessor --help-only")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("nCPU environment doctor")
    print("=" * 72)
    print(f"Python:   {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print(f"Repo:     {REPO_ROOT}")
    print(f"Demos:    {DEMOS_DIR}")
    print(f"TTY:      {sys.stdin.isatty()}")
    print()

    print("Dependency probes:")
    for module in ("torch", "transformers", "peft"):
        found = importlib.util.find_spec(module) is not None
        print(f"- {module:12s} installed: {'yes' if found else 'no'}")
    print()

    print("Mog toolchain:")
    try:
        from egdc.mog.execute import MOGC_BINARY, MOG_RUNTIME

        mogc_ok = MOGC_BINARY.is_file()
        runtime_ok = MOG_RUNTIME.is_file()
        print(f"- mogc        available: {'yes' if mogc_ok else 'no'}")
        print(f"  path: {MOGC_BINARY}")
        print(f"- runtime     available: {'yes' if runtime_ok else 'no'}")
        print(f"  path: {MOG_RUNTIME}")
        if not (mogc_ok and runtime_ok):
            print("  Build once for compiler-backed Mog tests and benchmarks:")
            print("    cargo build --release --manifest-path ../mog/compiler/Cargo.toml")
            print("    cargo build --release --manifest-path ../mog/runtime-rs/Cargo.toml")
            print("  Or point nCPU at a custom toolchain with MOG_ROOT / MOGC_BINARY / MOG_RUNTIME.")
    except Exception as exc:
        print(f"- unavailable: {exc}")
    print()

    print("Demo availability:")
    for key in CURATED_ORDER:
        demo = DEMO_REGISTRY[key]
        print(f"- {key:12s} script present: {'yes' if _demo_exists(key) else 'no'}")
    print()

    print("Platform guidance:")
    if sys.platform == "darwin":
        print("- macOS detected: best platform for Metal GPU demos.")
        if platform.machine().lower() not in {"arm64", "aarch64"}:
            print("- Apple Silicon not detected; some Metal workflows may be limited.")
    else:
        print("- Non-macOS detected: prioritize differentiable CPU, discovery, and text demos first.")
        print("- Metal GPU demos may be unavailable here.")
    print()

    print("Recommended install paths:")
    print("- Flagship demos: pip install -e '.[demo,dev]'")
    print("- Broader local environment: pip install -e '.[demo,model,train,dev]'")
    print()
    print("Recommended first run:")
    print("  python -m ncpu discover")
    print("  python -m ncpu text --interactive")
    return 0


def interactive_menu() -> int:
    print("nCPU lab")
    print("=" * 72)
    print()
    print("★ THE HERO EXPERIENCE (what makes this project unique)")
    print("   The GPU *is* the computer — with optional neural logic gates running inside the shader.")
    print("   ~1.9M IPS • Perfect determinism • 26-command super-debugger • Real Linux userspace")
    print()
    print("  0. gpu                 → Best interactive GPU UNIX shell")
    print("     gpu --native        → Raw native binary (maximum speed, zero Python)")
    print("     gpu debug           → 26-command deterministic post-mortem toolkit")
    print()
    print("Flagship interactive (lighter, cross-platform)")
    print("  1. Interactive Discovery REPL")
    print("  2. Neural Text Machine")
    print("Systems wow (GPU)")
    print("  3. GPU BusyBox shell")
    print("  4. Alpine Linux demo")
    print("Research depth")
    print("  5. Bottom-up full neural demo")
    print("  6. Neural screen vs Meta comparison")
    print("  7. Coprocessor demo help")
    print("Utility")
    print("  8. Show curated demos")
    print("  9. Demo info")
    print(" 10. Recommended path")
    print(" 11. Guided walkthrough")
    print(" 12. Environment doctor")
    print("q. Quit")
    print()
    print("Recommendation: Start with option 0 (gpu). That is the real story of this project.")
    print()
    while True:
        try:
            choice = input("lab> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in {"0", "gpu", "hero"}:
            return cmd_gpu(argparse.Namespace(mode="shell", interactive=True, neural_alu=False))
        if choice in {"native", "raw", "binary"}:
            return cmd_gpu(argparse.Namespace(mode="native", interactive=False, neural_alu=False))
        if choice == "1":
            return cmd_discover(argparse.Namespace())
        if choice == "2":
            return cmd_text(argparse.Namespace(interactive=True))
        if choice == "3":
            return cmd_systems(argparse.Namespace(demo="busybox", interactive=True, demo_mode=False))
        if choice == "4":
            return cmd_systems(argparse.Namespace(demo="alpine", interactive=False, demo_mode=True))
        if choice == "5":
            return cmd_full_neural(
                argparse.Namespace(device=None, max_instructions=None, output=None, summary_json=None)
            )
        if choice == "6":
            return cmd_meta_compare(
                argparse.Namespace(
                    shell=None,
                    device=None,
                    scale=1,
                    command=[],
                    capture_dir=None,
                    summary_json=None,
                    shell_log=None,
                    boot_delay_ms=1200,
                    step_delay_ms=1000,
                    final_hold_ms=800,
                    max_frames=None,
                    output=None,
                )
            )
        if choice == "7":
            return cmd_coprocessor(argparse.Namespace(help_only=True))
        if choice == "8":
            return cmd_demos(argparse.Namespace(verbose=True))
        if choice == "9":
            print("Available demos:", ", ".join(CURATED_ORDER))
            selected = input("demo name> ").strip().lower()
            if selected in DEMO_REGISTRY:
                return cmd_info(argparse.Namespace(name=selected))
            print(f"Unknown demo: {selected}")
            continue
        if choice == "10":
            return cmd_path(argparse.Namespace())
        if choice == "11":
            return cmd_walkthrough(argparse.Namespace())
        if choice == "12":
            return cmd_doctor(argparse.Namespace())
        if choice in {"q", "quit", "exit"}:
            return 0
        print("Choose 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, or q.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="nCPU flagship launcher for interactive discovery, systems, strict full-neural, side-by-side comparison, and coprocessor demos."
    )
    subparsers = parser.add_subparsers(dest="command")

    discover = subparsers.add_parser("discover", help="Run the interactive program discovery REPL")
    discover.set_defaults(func=cmd_discover)

    text = subparsers.add_parser("text", help="Run the neural text machine")
    text.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    text.set_defaults(func=cmd_text)

    systems = subparsers.add_parser("systems", help="Run a systems-focused demo")
    systems.add_argument("demo", choices=["busybox", "alpine"], help="Which systems demo to run")
    systems.add_argument("--interactive", action="store_true", help="Run BusyBox in interactive mode")
    systems.add_argument("--demo", dest="demo_mode", action="store_true", help="Run Alpine in demo mode")
    systems.set_defaults(func=cmd_systems)

    full_neural = subparsers.add_parser("full-neural", help="Run the strict bottom-up full-neural demo")
    full_neural.add_argument("--device", help="Override device passed to the demo (cpu, mps, cuda)")
    full_neural.add_argument("--max-instructions", type=int, default=None, help="Override the demo instruction budget")
    full_neural.add_argument("--output", help="Override the output PNG path")
    full_neural.add_argument("--summary-json", help="Write a machine-readable JSON summary for the run")
    full_neural.set_defaults(func=cmd_full_neural)

    meta_compare = subparsers.add_parser("meta-compare", help="Run the neural screen vs Meta comparison demo")
    meta_compare.add_argument(
        "--left-runtime",
        choices=("pty", "neural-os"),
        help="Left pane runtime: host PTY shell or nCPU GPU shell",
    )
    meta_compare.add_argument("--shell", help="Shell to spawn in the interactive pane")
    meta_compare.add_argument("--device", help="Override device passed to the demo (cpu, mps, cuda)")
    meta_compare.add_argument("--scale", type=int, default=1, help="Window scale factor")
    meta_compare.add_argument("--command", action="append", default=[], help="Scripted shell command to send; may be repeated")
    meta_compare.add_argument("--capture-dir", help="Directory for per-step PNG captures")
    meta_compare.add_argument("--summary-json", help="Write a machine-readable JSON summary for the run")
    meta_compare.add_argument("--shell-log", help="Write decoded shell output to this log path")
    meta_compare.add_argument("--boot-delay-ms", type=int, default=1200, help="Delay before first scripted capture/send")
    meta_compare.add_argument("--step-delay-ms", type=int, default=1000, help="Delay between scripted send and capture")
    meta_compare.add_argument("--final-hold-ms", type=int, default=800, help="Delay after the final scripted capture")
    meta_compare.add_argument("--max-frames", type=int, default=None, help="Exit after N frames")
    meta_compare.add_argument("--output", help="Save the composed neural frame as a PNG")
    meta_compare.set_defaults(func=cmd_meta_compare)

    coprocessor = subparsers.add_parser("coprocessor", help="Run the language-model coprocessor demo")
    coprocessor.add_argument("--help-only", action="store_true", help="Show demo help instead of running a full heavy command")
    coprocessor.set_defaults(func=cmd_coprocessor)

    demos = subparsers.add_parser("demos", help="Show curated demos and recommended commands")
    demos.add_argument("--verbose", action="store_true", help="Show descriptions, platform, and dependency weight")
    demos.set_defaults(func=cmd_demos)

    info = subparsers.add_parser("info", help="Show detailed information for one registered demo")
    info.add_argument("name", choices=sorted(DEMO_REGISTRY.keys()))
    info.set_defaults(func=cmd_info)

    path = subparsers.add_parser("path", help="Show the recommended newcomer path through the repo")
    path.set_defaults(func=cmd_path)

    walkthrough = subparsers.add_parser("walkthrough", help="Show a guided first-run walkthrough with exact commands")
    walkthrough.set_defaults(func=cmd_walkthrough)

    run = subparsers.add_parser("run", help="Run any registered demo by name with passthrough args")
    run.add_argument("name", choices=sorted(DEMO_REGISTRY.keys()))
    run.add_argument("demo_args", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    doctor = subparsers.add_parser("doctor", help="Show environment/platform guidance")
    doctor.set_defaults(func=cmd_doctor)

    # ── Hero: GPU as Self-Sufficient Computer (Rust Metal + optional neural ALU in shader)
    gpu = subparsers.add_parser(
        "gpu",
        help="★ THE HERO: GPU as complete self-sufficient computer (Rust Metal). Best performance, perfect determinism, full debugging toolkit, optional neural ALU in shader.",
        description=(
            "The flagship nCPU experience.\n\n"
            "A real multi-process UNIX OS (BusyBox + Alpine) running entirely on Apple Silicon GPU\n"
            "via Metal shaders at ~1.9M IPS, with optional trained neural networks executing the ALU\n"
            "directly inside the shader.\n\n"
            "This is the core thesis: the GPU *is* the computer, not an accelerator."
        ),
    )
    gpu.add_argument(
        "mode",
        nargs="?",
        choices=["shell", "busybox", "alpine", "debug", "neural", "native"],
        default="shell",
        help="shell (default): best interactive GPU UNIX shell; native: direct launch of the raw ncpu_run binary for max speed; debug: show deterministic toolkit guidance; neural: emphasize neural-ALU-in-shader path.",
    )
    gpu.add_argument("--interactive", action="store_true", help="Force interactive mode for BusyBox shell")
    gpu.add_argument("--neural-alu", action="store_true", help="Request neural ALU weights (in-shader when using Rust backend)")
    gpu.set_defaults(func=cmd_gpu)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if sys.stdin.isatty() and os.environ.get("NCPU_LAB_NO_MENU") != "1":
            return interactive_menu()
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
