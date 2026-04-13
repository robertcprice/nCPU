#!/usr/bin/env python3
"""Unified flagship launcher for nCPU interactive demos."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import runpy
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

DEMO_REGISTRY = {
    "discover": {
        "title": "Interactive Discovery REPL",
        "category": "Flagship interactive",
        "command": "PYTHONPATH=. python demos/interactive_discovery.py",
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
        "command": "PYTHONPATH=. python demos/neural_text_machine.py --interactive",
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
        "command": "PYTHONPATH=. python demos/busybox_gpu_demo.py --interactive",
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
        "command": "PYTHONPATH=. python demos/alpine_gpu.py --demo",
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
    "coprocessor": {
        "title": "Code in Brain / Coprocessor Demo",
        "category": "Research depth",
        "command": "PYTHONPATH=. python demos/demo_code_in_brain.py --help",
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

CURATED_ORDER = ["discover", "text", "busybox", "alpine", "coprocessor"]
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
        print(f"  {next_demo['title']} -> python -m ncpu.lab info {next_name}")


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


def cmd_run(args: argparse.Namespace) -> int:
    if args.name not in DEMO_REGISTRY:
        raise SystemExit(f"Unknown demo '{args.name}'. Use `python -m ncpu.lab demos`.")
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
    print("Run this first:")
    for key in ("discover", "text"):
        demo = DEMO_REGISTRY[key]
        print(f"  {demo['title']}")
        print(f"  {key:12s} {demo['command']}")
        print(f"               {demo['description']}")
    print()
    print("Useful follow-ups:")
    print("  python -m ncpu.lab path")
    print("  python -m ncpu.lab walkthrough")
    print("  python -m ncpu.lab info discover")
    print("  python -m ncpu.lab systems busybox --interactive")
    print("  python -m ncpu.lab systems alpine --demo")
    print("  python -m ncpu.lab coprocessor --help-only")
    print("  python -m ncpu.lab run discover")
    return 0


def cmd_path(_args: argparse.Namespace) -> int:
    print("nCPU recommended path")
    print("=" * 72)
    print("1. Start with interactive program discovery")
    print("   python -m ncpu.lab discover")
    print()
    print("2. Move to the neural text machine")
    print("   python -m ncpu.lab text --interactive")
    print()
    print("3. If you want the systems story, try BusyBox or Alpine")
    print("   python -m ncpu.lab systems busybox --interactive")
    print("   python -m ncpu.lab systems alpine --demo")
    print()
    print("4. If you want the LLM/computation story, inspect the coprocessor demo")
    print("   python -m ncpu.lab coprocessor --help-only")
    print()
    print("5. For a guided first-time route, run:")
    print("   python -m ncpu.lab walkthrough")
    return 0


def cmd_walkthrough(_args: argparse.Namespace) -> int:
    print("nCPU walkthrough")
    print("=" * 72)
    print("Step 1: program-by-examples")
    print("  Run: python -m ncpu.lab discover")
    print("  Then type:")
    print("    preset add")
    print("    synthesize")
    print("    summary")
    print("    test 15, 25")
    print("    export exports/add_program.asm")
    print()
    print("Step 2: text transformation")
    print("  Run: python -m ncpu.lab text --interactive")
    print("  Then type:")
    print("    cipher hello khoor")
    print("    summary")
    print("    apply world")
    print("    save exports/text_summary.json")
    print()
    print("Step 3: systems story")
    print("  Run: python -m ncpu.lab systems busybox --interactive")
    print("  If you are on Apple Silicon, also try:")
    print("    python -m ncpu.lab systems alpine --demo")
    print()
    print("Step 4: research depth")
    print("  Run: python -m ncpu.lab coprocessor --help-only")
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
    print("  python -m ncpu.lab discover")
    print("  python -m ncpu.lab text --interactive")
    return 0


def interactive_menu() -> int:
    print("nCPU lab")
    print("=" * 72)
    print("Flagship interactive")
    print("  1. Interactive Discovery REPL")
    print("  2. Neural Text Machine")
    print("Systems wow")
    print("  3. GPU BusyBox shell")
    print("  4. Alpine Linux demo")
    print("Research depth")
    print("  5. Coprocessor demo help")
    print("Utility")
    print("  6. Show curated demos")
    print("  7. Demo info")
    print("  8. Recommended path")
    print("  9. Guided walkthrough")
    print(" 10. Environment doctor")
    print("q. Quit")
    print()
    while True:
        try:
            choice = input("lab> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "1":
            return cmd_discover(argparse.Namespace())
        if choice == "2":
            return cmd_text(argparse.Namespace(interactive=True))
        if choice == "3":
            return cmd_systems(argparse.Namespace(demo="busybox", interactive=True, demo_mode=False))
        if choice == "4":
            return cmd_systems(argparse.Namespace(demo="alpine", interactive=False, demo_mode=True))
        if choice == "5":
            return cmd_coprocessor(argparse.Namespace(help_only=True))
        if choice == "6":
            return cmd_demos(argparse.Namespace(verbose=True))
        if choice == "7":
            print("Available demos:", ", ".join(CURATED_ORDER))
            selected = input("demo name> ").strip().lower()
            if selected in DEMO_REGISTRY:
                return cmd_info(argparse.Namespace(name=selected))
            print(f"Unknown demo: {selected}")
            continue
        if choice == "8":
            return cmd_path(argparse.Namespace())
        if choice == "9":
            return cmd_walkthrough(argparse.Namespace())
        if choice == "10":
            return cmd_doctor(argparse.Namespace())
        if choice in {"q", "quit", "exit"}:
            return 0
        print("Choose 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, or q.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="nCPU flagship launcher for interactive discovery, text, systems, and coprocessor demos."
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
