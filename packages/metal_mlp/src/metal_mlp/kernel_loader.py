"""Dynamic loader for Metal GPU kernel shared libraries.

Handles the import mechanics for loading PyO3-compiled Rust/Metal shared
libraries (.so / .dylib) without polluting ``sys.path``. Supports multiple
search paths, cached module loading, and graceful fallback when Metal
hardware is unavailable.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

__all__ = ["MetalKernelLoader"]

logger = logging.getLogger(__name__)


class MetalKernelLoader:
    """Load kernel classes from a Rust/Metal shared library via importlib.

    Uses ``importlib.util.spec_from_file_location`` to load the .so file
    directly, avoiding sys.path pollution that can cause version conflicts
    (e.g., between system torch and venv torch).

    Parameters
    ----------
    so_name : str
        Filename of the shared library to search for.
        Common patterns: ``'ncpu_metal.abi3.so'``, ``'my_kernels.so'``.
    search_paths : list[Path] or None
        Directories to search for the .so file. Searched in order; the
        first directory containing ``so_name`` wins. If None, only the
        standard import mechanism is used.
    module_name : str
        The Python module name to register in ``sys.modules``. Defaults
        to the stem of ``so_name`` (e.g., ``'ncpu_metal'``).

    Examples
    --------
    >>> loader = MetalKernelLoader(
    ...     so_name="my_kernels.abi3.so",
    ...     search_paths=[Path("/path/to/build")],
    ... )
    >>> if loader.available:
    ...     kernel_cls = loader.get_class("MyKernel")
    ...     kernel = kernel_cls()
    """

    def __init__(
        self,
        so_name: str = "ncpu_metal.abi3.so",
        search_paths: Optional[list[Path]] = None,
        module_name: Optional[str] = None,
    ) -> None:
        self._so_name = so_name
        self._search_paths = search_paths or []
        self._module_name = module_name or Path(so_name).stem.split(".")[0]
        self._module: Optional[object] = None
        self._load_attempted = False
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        """Whether the Metal kernel module was successfully loaded."""
        self._ensure_loaded()
        return self._module is not None

    @property
    def load_error(self) -> Optional[str]:
        """Human-readable error message if loading failed, or None on success."""
        self._ensure_loaded()
        return self._load_error

    def get_class(self, class_name: str) -> Optional[type]:
        """Retrieve a kernel class from the loaded module by name.

        Parameters
        ----------
        class_name : str
            The class name exported by the shared library, e.g.
            ``'NeuralALUKernel'``, ``'NeuralDisplayKernel'``.

        Returns
        -------
        type or None
            The class object, or ``None`` if the module is unavailable or
            the class does not exist.
        """
        self._ensure_loaded()
        if self._module is None:
            return None
        cls = getattr(self._module, class_name, None)
        if cls is None:
            logger.warning(
                "Class '%s' not found in %s. Available: %s",
                class_name,
                self._module_name,
                self.list_classes(),
            )
        return cls

    def get_module(self) -> Optional[object]:
        """Return the raw loaded module, or None if unavailable."""
        self._ensure_loaded()
        return self._module

    def list_classes(self) -> list[str]:
        """List all public names exported by the loaded module.

        Returns
        -------
        list[str]
            Sorted list of public attribute names, or empty list if
            the module is unavailable.
        """
        self._ensure_loaded()
        if self._module is None:
            return []
        return sorted(a for a in dir(self._module) if not a.startswith("_"))

    # -- Private helpers --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the module on first access."""
        if self._load_attempted:
            return
        self._load_attempted = True

        # Check sys.modules first (may have been loaded by another codepath)
        if self._module_name in sys.modules:
            self._module = sys.modules[self._module_name]
            logger.debug(
                "Using already-imported %s from sys.modules", self._module_name
            )
            return

        # Search for the .so file in provided paths
        for search_dir in self._search_paths:
            so_path = search_dir / self._so_name
            if not so_path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    self._module_name, str(so_path)
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[self._module_name] = mod
                self._module = mod
                logger.debug("Loaded %s from %s", self._module_name, so_path)
                return
            except Exception as exc:
                logger.debug("Failed to load %s: %s", so_path, exc)
                continue

        # Last resort: standard import
        try:
            spec = importlib.util.find_spec(self._module_name)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[self._module_name] = mod
                self._module = mod
                logger.debug("Loaded %s via standard import", self._module_name)
                return
        except (ImportError, ModuleNotFoundError):
            pass

        self._load_error = (
            f"Shared library '{self._so_name}' not found. "
            f"Searched: {[str(p / self._so_name) for p in self._search_paths]}. "
            f"Also tried standard import of '{self._module_name}'."
        )
        logger.info(self._load_error)

    def __repr__(self) -> str:
        status = "available" if self.available else "unavailable"
        return f"MetalKernelLoader({self._so_name!r}, {status})"
