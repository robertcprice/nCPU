"""Shared isolation for the synthesis-API suite.

The live HTTP server inherits its process environment, and the Rust
backend it spawns reads every persistent nsynth memory bank — plus the
method-router state, which changes which solver families even run — from
``NSYNTH_*`` env vars that default to the user's real banks at
``~/.nsynth_*``. Without isolation, a populated
``~/.nsynth_method_router.json`` redirects the easy problems in this
suite onto slow solver paths and blows the per-request timeouts.

The env-var list is maintained in one place
(``tests/mcp_server/conftest.py``) and reused here so a new bank added
to the backend only needs registering once.
"""

from __future__ import annotations

import pytest

from tests.mcp_server.conftest import isolated_env


@pytest.fixture(scope="session")
def nsynth_isolated_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Full process environment with every nsynth bank under a tmp dir."""
    return isolated_env(tmp_path_factory.mktemp("nsynth_env_banks"))
