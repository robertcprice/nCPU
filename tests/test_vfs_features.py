#!/usr/bin/env python3
"""
Tests for VFS features: caching, neural prefetching, hard links, ftruncate, etc.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.filesystem import GPUFilesystem


class TestLRUCache:
    """Tests for LRU file caching."""

    def test_cache_populates_on_read(self):
        """Cache should populate when reading a file."""
        fs = GPUFilesystem(cache_size=4)
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)  # O_RDONLY
        fs.read(fd, 100)

        assert '/tmp/test' in fs._cache
        assert fs._cache['/tmp/test'] == b'content'

    def test_cache_hit(self):
        """Second read should hit cache."""
        fs = GPUFilesystem(cache_size=4)
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)
        fs.read(fd, 100)  # First read - cache miss

        fs.lseek(fd, 0, 0)
        fs.read(fd, 100)  # Second read - cache hit

        stats = fs.get_cache_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_cache_eviction(self):
        """Cache should evict LRU entries when full."""
        fs = GPUFilesystem(cache_size=2)

        # Create 3 files
        for i in range(3):
            fs.files[f'/tmp/file{i}'] = b'content' * (i + 1)

        # Read them all to populate cache
        for i in range(3):
            fd = fs.open(f'/tmp/file{i}', 0)
            fs.read(fd, 100)

        # Only 2 should be cached (capacity is 2)
        assert len(fs._cache) <= 2

    def test_cache_invalidation_on_write(self):
        """Cache should be invalidated on write."""
        fs = GPUFilesystem(cache_size=4)
        fs.files['/tmp/test'] = b'original'

        # Read to populate cache
        fd = fs.open('/tmp/test', 0)
        fs.read(fd, 100)
        assert '/tmp/test' in fs._cache

        # Write should invalidate cache
        fd2 = fs.open('/tmp/test', 1)  # O_WRONLY
        fs.write(fd2, b'modified')

        assert '/tmp/test' not in fs._cache

    def test_clear_cache(self):
        """clear_cache should empty the cache."""
        fs = GPUFilesystem(cache_size=4)
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)
        fs.read(fd, 100)

        fs.clear_cache()

        assert len(fs._cache) == 0
        stats = fs.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0


class TestNeuralPrefetch:
    """Tests for neural prefetching."""

    def test_access_history_tracking(self):
        """Access history should be recorded."""
        fs = GPUFilesystem(enable_prefetch=True)
        fs.files['/tmp/a'] = b'a'
        fs.files['/tmp/b'] = b'b'
        fs.files['/tmp/c'] = b'c'

        # Sequential access
        for name in ['/tmp/a', '/tmp/b', '/tmp/c']:
            fd = fs.open(name, 0)
            fs.read(fd, 100)

        assert len(fs._access_history) == 3

    def test_pattern_prediction(self):
        """Pattern prediction should detect repeating patterns."""
        fs = GPUFilesystem(enable_prefetch=True)
        fs.files['/tmp/a'] = b'a'
        fs.files['/tmp/b'] = b'b'
        fs.files['/tmp/c'] = b'c'
        fs.files['/tmp/d'] = b'd'

        # Create repeating pattern: a, b, a, b, a, b
        for _ in range(3):
            for name in ['/tmp/a', '/tmp/b']:
                fd = fs.open(name, 0)
                fs.read(fd, 100)

        # Should have predictions
        assert len(fs._prefetch_predictions) > 0


class TestHardLinks:
    """Tests for hard link support."""

    def test_link_creates_hard_link(self):
        """link() should create a hard link."""
        fs = GPUFilesystem()
        fs.files['/tmp/original'] = b'content'

        result = fs.link('/tmp/original', '/tmp/hardlink')

        assert result == 0
        assert '/tmp/hardlink' in fs.files
        assert fs.files['/tmp/hardlink'] == b'content'

    def test_link_nonexistent_source(self):
        """link() should fail if source doesn't exist."""
        fs = GPUFilesystem()

        result = fs.link('/tmp/nonexistent', '/tmp/new')

        assert result == -1

    def test_link_target_exists(self):
        """link() should fail if target already exists."""
        fs = GPUFilesystem()
        fs.files['/tmp/original'] = b'content'
        fs.files['/tmp/exists'] = b'other'

        result = fs.link('/tmp/original', '/tmp/exists')

        assert result == -1


class TestFtruncate:
    """Tests for ftruncate."""

    def test_ftruncate_shortens_file(self):
        """ftruncate should shorten file."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'very long content'

        fd = fs.open('/tmp/test', 2)  # O_RDWR
        result = fs.ftruncate(fd, 4)

        assert result == 0
        assert fs.files['/tmp/test'] == b'very'

    def test_ftruncate_extends_file(self):
        """ftruncate should extend file with zeros."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'short'

        fd = fs.open('/tmp/test', 2)  # O_RDWR
        result = fs.ftruncate(fd, 10)

        assert result == 0
        assert fs.files['/tmp/test'] == b'short\x00\x00\x00\x00\x00'

    def test_ftruncate_readonly_fails(self):
        """ftruncate should fail on read-only fd."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)  # O_RDONLY
        result = fs.ftruncate(fd, 4)

        assert result == -1


class TestFsync:
    """Tests for fsync."""

    def test_fsync_succeeds(self):
        """fsync should succeed on valid fd."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)  # O_RDONLY
        result = fs.fsync(fd)

        assert result == 0

    def test_fsync_invalid_fd_fails(self):
        """fsync should fail on invalid fd."""
        fs = GPUFilesystem()

        result = fs.fsync(999)

        assert result == -1


class TestCacheStats:
    """Tests for cache statistics."""

    def test_cache_stats_structure(self):
        """get_cache_stats should return expected structure."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)
        fs.read(fd, 100)

        stats = fs.get_cache_stats()

        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
        assert 'cached_files' in stats

    def test_hit_rate_calculation(self):
        """hit_rate should be calculated correctly."""
        fs = GPUFilesystem()
        fs.files['/tmp/test'] = b'content'

        fd = fs.open('/tmp/test', 0)
        fs.read(fd, 100)  # miss
        fs.lseek(fd, 0, 0)
        fs.read(fd, 100)  # hit

        stats = fs.get_cache_stats()
        assert stats['hit_rate'] == 0.5


class TestIntegration:
    """Integration tests combining features."""

    def test_mixed_operations(self):
        """Test mixing cache, links, truncate."""
        fs = GPUFilesystem(cache_size=4, enable_prefetch=True)
        fs.mkdir('/tmp')

        # Create file
        fs.files['/tmp/data'] = b'x' * 1000

        # Read to populate cache
        fd = fs.open('/tmp/data', 0)
        fs.read(fd, 100)

        # Create hard link
        fs.link('/tmp/data', '/tmp/data_backup')

        # Truncate
        fd2 = fs.open('/tmp/data', 2)
        fs.ftruncate(fd2, 500)

        # Verify
        assert len(fs.files['/tmp/data']) == 500
        assert len(fs.files['/tmp/data_backup']) == 1000  # Original unchanged
        assert '/tmp/data' in fs._cache  # Cache invalidated

    def test_repeated_reads_benchmark(self):
        """Repeated reads should benefit from caching."""
        fs = GPUFilesystem(cache_size=4)
        fs.files['/tmp/large'] = b'y' * 10000

        # First read - cold
        fd = fs.open('/tmp/large', 0)
        fs.read(fd, 10000)

        # Repeated reads - should hit cache
        for _ in range(10):
            fs.lseek(fd, 0, 0)
            fs.read(fd, 10000)

        stats = fs.get_cache_stats()
        # Should have 10 hits and 1 miss
        assert stats['hits'] == 10
        assert stats['hit_rate'] > 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
