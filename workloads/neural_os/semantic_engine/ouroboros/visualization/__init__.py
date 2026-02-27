"""
OUROBOROS Visualization System
================================
Data-driven visualizations from agent swarm behavior.

Generates:
- Override approval dashboard (human-in-the-loop)
- Real data from SharedKVMemory (not canned templates)

Note: Primary visualizations are in kvrm_integration/visualizations.py
"""

from .override_dashboard import OverrideDashboard

__all__ = [
    "OverrideDashboard",
]
