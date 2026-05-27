"""JEPA-style World Models for the nCPU machine.

This package contains predictive world models of *computation itself*,
trained on top of nCPU's exact neural ALU substrate (and the fast Rust Metal path).

Primary goal: fast abstract prediction in latent space for
- speculation / mental simulation before expensive exact execution
- robustness via anomaly detection (divergence from predicted dynamics)
- better scheduling and phase detection for the hero GPU kernel
- richer training signals for the latent controller (SOME) stack

See docs/architecture/JEPA_MACHINE_WORLD_MODEL.md for the full design.
"""

from .je_world_model import JEWorldModel, JEWMConfig  # noqa: F401

__all__ = ["JEWorldModel", "JEWMConfig"]