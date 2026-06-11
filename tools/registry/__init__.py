"""Verified-skill registry — crates.io for synthesized NPCoT programs.

Trustless contribution: the server re-executes every submitted program
against its claimed examples with a deterministic pure-Python mirror of
the canonical executor (kernels/npcot_wasm/src/lib.rs). Wrong or spam
code physically cannot enter the registry.
"""
