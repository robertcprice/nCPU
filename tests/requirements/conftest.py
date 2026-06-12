"""Isolate the nsynth memory banks for the requirements suite.

The pipeline runs the nsynth backend *in-process* (handle_synthesize_request
shells out to the binary with the test process's own environment). Without
isolation, a populated ``~/.nsynth_method_router.json`` on the dev machine
reroutes the easy problems in this suite onto slow solver paths, blowing the
per-request timeout and turning a clean solve into a false "refused". This
fixture points every ``NSYNTH_*`` bank at a fresh tmp dir for the whole
session (mirrors tests/mcp_server/conftest.py)."""

from __future__ import annotations

import pytest

from tests.mcp_server.conftest import _BANK_ENV


@pytest.fixture(autouse=True, scope="session")
def _isolate_nsynth_banks(tmp_path_factory):
    banks = tmp_path_factory.mktemp("requirements_nsynth_banks")
    import os

    saved = {k: os.environ.get(k) for k in _BANK_ENV}
    for var, fname in _BANK_ENV.items():
        os.environ[var] = str(banks / fname)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
