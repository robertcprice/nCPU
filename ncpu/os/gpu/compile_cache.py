#!/usr/bin/env python3
"""
Content-Addressed Compilation Cache for GPU programs.

Caches GCC cross-compilation results by SHA-256 of source + flags,
avoiding redundant recompilation of unchanged programs.

Cache location: ~/.ncpu/compile_cache/
Cache key: sha256(source_bytes + flags_string)[:16]

Usage:
    from ncpu.os.gpu.compile_cache import CompileCache
    cache = CompileCache()
    key = cache.cache_key(source_bytes, flags)
    binary = cache.get(key)
    if binary is None:
        binary = compile_it(source_bytes)
        cache.put(key, binary)
"""

import hashlib
from pathlib import Path
from typing import Optional


CACHE_DIR = Path.home() / ".ncpu" / "compile_cache"


class CompileCache:
    """Content-addressed binary cache for compiled GPU programs."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(source: bytes, flags: str = "") -> str:
        """Generate cache key from source bytes and compiler flags."""
        h = hashlib.sha256(source + flags.encode()).hexdigest()[:16]
        return h

    def get(self, key: str) -> Optional[bytes]:
        """Look up cached binary by key. Returns None on miss."""
        path = self._dir / key
        if path.exists():
            return path.read_bytes()
        return None

    def put(self, key: str, binary: bytes) -> None:
        """Store compiled binary in cache."""
        path = self._dir / key
        path.write_bytes(binary)

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        count = 0
        for f in self._dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count

    def stats(self) -> dict:
        """Return cache statistics."""
        files = list(self._dir.iterdir())
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "entries": len(files),
            "total_bytes": total_bytes,
            "cache_dir": str(self._dir),
        }
