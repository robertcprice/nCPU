"""Synthesis API — nsynth behind HTTP (cascade Rung 3).

See `ncpu.synthesis_api.server` for the stdlib-only HTTP server and the
embeddable `handle_synthesize_request` function.
"""

from ncpu.synthesis_api.server import (  # noqa: F401
    SynthConfig,
    handle_synthesize_request,
    read_bank_stats,
    start_server,
    validate_synthesize_request,
)
