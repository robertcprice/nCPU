import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_metadata_parses():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["name"] == "ncpu"
    assert data["project"]["authors"][0]["name"] == "Robert Price"
    assert data["project"]["scripts"]["ncpu"] == "ncpu.__main__:main"
    assert data["project"]["scripts"]["ncpu-lab"] == "ncpu.lab:main"
    extras = data["project"]["optional-dependencies"]
    assert "demo" in extras
    assert "full" in extras


def test_package_import_exposes_version_and_author():
    sys.path.insert(0, str(ROOT))
    import ncpu

    assert ncpu.__version__ == "0.2.0"
    assert ncpu.__author__ == "Robert Price"
