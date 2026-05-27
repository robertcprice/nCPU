"""Top-level module entrypoint for ``python -m ncpu``."""

from __future__ import annotations

from collections.abc import Sequence
import sys


_HELP_FLAGS = {"-h", "--help"}
_VERSION_FLAGS = {"--version"}
_LEGACY_DEMO_FLAGS = {"--live", "--headless", "--script", "--multiproc", "-m"}


def _run_lab(argv: Sequence[str]) -> int:
    from ncpu.lab import main as lab_main

    try:
        return lab_main(list(argv))
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1


def _run_demo(argv: Sequence[str]) -> int:
    from ncpu.demo import main as demo_main

    return demo_main(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] == "demo":
        return _run_demo(args[1:])

    if args and args[0] in _LEGACY_DEMO_FLAGS:
        return _run_demo(args)

    if args and args[0] in _VERSION_FLAGS:
        from ncpu import __version__

        print(__version__)
        return 0

    if args and args[0] in _HELP_FLAGS:
        return _run_lab(args)

    return _run_lab(args)


if __name__ == "__main__":
    raise SystemExit(main())
