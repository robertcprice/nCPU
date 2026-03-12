"""
GPU Filesystem — Python-side filesystem for the GPU-native UNIX OS.

Provides a dict-based in-memory filesystem accessed via syscalls from
ARM64 programs running on the Metal GPU kernel. The GPU program issues
SVC traps which the Python syscall handler routes to this filesystem.

Supports: open, close, read, write, lseek, stat, mkdir, unlink, rmdir,
          getcwd, chdir, listdir, resolve_path

Hardened: fd bounds, read-only enforcement, directory read protection,
          max open files, path length limits, safe prefix matching.
"""

import struct
import copy
from typing import Optional


# Limits
MAX_FD = 1024
MAX_PATH_LEN = 4096
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB per file


# ═══════════════════════════════════════════════════════════════════════════════
# PIPE BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class PipeBuffer:
    """Shared buffer for inter-process pipe communication."""

    def __init__(self, capacity: int = 4096):
        self.buffer = bytearray()
        self.capacity = capacity
        self.readers = 1   # Reference count of read endpoints
        self.writers = 1   # Reference count of write endpoints

    def write(self, data: bytes) -> int:
        """Write data to pipe. Returns bytes written, or -1 for EPIPE."""
        if self.readers == 0:
            return -1  # EPIPE — no readers
        available = self.capacity - len(self.buffer)
        if available <= 0:
            return 0  # Would block — pipe full
        chunk = data[:available]
        self.buffer.extend(chunk)
        return len(chunk)

    def read(self, count: int) -> Optional[bytes]:
        """Read from pipe. Returns bytes, empty bytes for EOF, None for would-block."""
        if len(self.buffer) > 0:
            result = bytes(self.buffer[:count])
            del self.buffer[:count]
            return result
        # Buffer empty
        if self.writers == 0:
            return b""  # EOF — no writers left
        return None  # Would block — waiting for data

    def close_reader(self):
        self.readers = max(0, self.readers - 1)

    def close_writer(self):
        self.writers = max(0, self.writers - 1)


class GPUFilesystem:
    """In-memory filesystem for GPU programs with LRU caching and neural prefetching."""

    def __init__(self, cache_size: int = 64, enable_prefetch: bool = True):
        # Core storage
        self.files: dict[str, bytes] = {}         # path → content
        self.directories: set[str] = set()         # directory paths
        self.fd_table: dict[int, dict] = {}        # fd → {path, offset, flags, ...}
        self.cwd: str = "/"

        # ── LRU Cache ───────────────────────────────────────────────────────
        self._cache: dict[str, bytes] = {}        # LRU cache for file reads
        self._cache_order: list[str] = []          # Access order for LRU
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

        # ── Neural Prefetching ───────────────────────────────────────────
        self._enable_prefetch = enable_prefetch
        self._access_history: list[str] = []       # Recent file access pattern
        self._history_len = 32                     # Track last N accesses
        self._prefetch_predictions: list[str] = [] # Predicted next files

        # Bootstrap standard directories
        self._bootstrap()

    def _bootstrap(self):
        """Create standard UNIX directory structure."""
        for d in ["/", "/bin", "/home", "/tmp", "/etc", "/var", "/usr"]:
            self.directories.add(d)

        # Default files
        self.files["/etc/motd"] = b"Welcome to GPU-Native UNIX OS v1.0\nRunning on Apple Silicon Metal\n"
        self.files["/etc/hostname"] = b"gpu0\n"

    # ═══════════════════════════════════════════════════════════════════════
    # PATH RESOLUTION
    # ═══════════════════════════════════════════════════════════════════════

    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to cwd, normalize it."""
        if not path:
            return self.cwd

        # Clamp path length
        path = path[:MAX_PATH_LEN]

        if not path.startswith("/"):
            if self.cwd == "/":
                path = "/" + path
            else:
                path = self.cwd + "/" + path

        # Normalize: resolve . and ..
        parts = []
        for p in path.split("/"):
            if p == "" or p == ".":
                continue
            elif p == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(p)

        return "/" + "/".join(parts) if parts else "/"

    def _parent_dir(self, path: str) -> str:
        """Get parent directory of a path."""
        if path == "/":
            return "/"
        idx = path.rfind("/")
        return path[:idx] if idx > 0 else "/"

    def _is_child_of(self, child: str, parent: str) -> bool:
        """Check if child is a direct child of parent directory.

        Safe prefix matching that avoids /a matching /abc.
        """
        if parent == "/":
            prefix = "/"
        else:
            prefix = parent + "/"
        if not child.startswith(prefix):
            return False
        rest = child[len(prefix):]
        return len(rest) > 0 and "/" not in rest

    # ═══════════════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════

    O_RDONLY = 0
    O_WRONLY = 1
    O_RDWR   = 2
    O_CREAT  = 64
    O_TRUNC  = 512
    O_APPEND = 1024

    def _access_mode(self, flags: int) -> int:
        """Extract access mode (O_RDONLY=0, O_WRONLY=1, O_RDWR=2)."""
        return flags & 3

    def _allocate_fd(self) -> int:
        """Allocate lowest-available fd (for fd recycling)."""
        fd = 3
        while fd in self.fd_table:
            fd += 1
        return fd if fd < MAX_FD else -1

    def _follow_symlink(self, path: str, depth: int = 0) -> str:
        """Follow symlinks, returning the final target path. Max 8 hops."""
        if depth > 8:
            return path  # Prevent symlink loops
        if hasattr(self, 'symlinks') and path in self.symlinks:
            target = self.symlinks[path]
            resolved = self.resolve_path(target)
            return self._follow_symlink(resolved, depth + 1)
        return path

    def open(self, path: str, flags: int) -> int:
        """Open a file or directory, return fd or -1."""
        path = self.resolve_path(path)
        # Follow symlinks for open (except O_NOFOLLOW)
        path = self._follow_symlink(path)

        fd = self._allocate_fd()
        if fd < 0:
            return -1  # EMFILE

        # Allow opening directories (for ls/getdents64)
        if path in self.directories:
            self.fd_table[fd] = {"path": path, "offset": 0, "flags": flags,
                                 "is_dir": True, "type": "file"}
            return fd

        # Check parent directory exists
        parent = self._parent_dir(path)
        if parent != "/" and parent not in self.directories:
            return -1  # ENOENT

        # Create file if O_CREAT and doesn't exist
        if (flags & self.O_CREAT) and path not in self.files:
            self.files[path] = b""

        if path not in self.files:
            return -1  # ENOENT

        # Truncate if O_TRUNC
        if flags & self.O_TRUNC:
            self.files[path] = b""

        offset = len(self.files[path]) if (flags & self.O_APPEND) else 0
        self.fd_table[fd] = {"path": path, "offset": offset, "flags": flags,
                             "type": "file"}
        return fd

    def close(self, fd: int) -> int:
        """Close a file descriptor (pipe-aware)."""
        if fd not in self.fd_table:
            return -1
        entry = self.fd_table[fd]

        # Handle pipe endpoint cleanup
        fd_type = entry.get("type", "file")
        pipe_buf = entry.get("pipe_buffer")
        if pipe_buf:
            if fd_type == "pipe_read":
                pipe_buf.close_reader()
            elif fd_type == "pipe_write":
                pipe_buf.close_writer()

        del self.fd_table[fd]
        return 0

    def read(self, fd: int, count: int) -> Optional[bytes]:
        """Read from fd (pipe-aware), return bytes or None on error.

        Includes LRU caching and neural prefetching for performance.
        """
        if fd not in self.fd_table:
            return None
        entry = self.fd_table[fd]

        # Pipe read
        if entry.get("type") == "pipe_read":
            pipe_buf = entry.get("pipe_buffer")
            if pipe_buf:
                return pipe_buf.read(count)
            return None

        # Cannot read directories as files
        if entry.get("is_dir"):
            return None  # EISDIR

        # Check read permission
        mode = self._access_mode(entry["flags"])
        if mode == self.O_WRONLY:
            return None  # EBADF (write-only fd)

        path = entry["path"]

        # ── LRU Cache Lookup ─────────────────────────────────────────────
        if path in self._cache:
            # Cache hit
            self._cache_hits += 1
            data = self._cache[path]
            # Move to front of access order
            if path in self._cache_order:
                self._cache_order.remove(path)
            self._cache_order.insert(0, path)
        else:
            # Cache miss - read from files and populate cache
            self._cache_misses += 1
            data = self.files.get(path, b"")
            # Add to cache if small enough
            if len(data) < 1024 * 1024:  # Cache files < 1MB
                self._put_in_cache(path, data)

        # Record access for prefetching
        self._record_access(path)

        offset = entry["offset"]
        result = data[offset:offset + count]
        entry["offset"] += len(result)
        return result

    def _put_in_cache(self, path: str, data: bytes):
        """Add file to LRU cache, evicting oldest if necessary."""
        # Evict if cache is full
        while len(self._cache) >= self._cache_size and self._cache_order:
            oldest = self._cache_order.pop()
            if oldest in self._cache:
                del self._cache[oldest]

        # Add to cache
        self._cache[path] = data
        if path in self._cache_order:
            self._cache_order.remove(path)
        self._cache_order.insert(0, path)

    def _record_access(self, path: str):
        """Record file access for neural prefetching."""
        if not self._enable_prefetch:
            return

        self._access_history.append(path)
        if len(self._access_history) > self._history_len:
            self._access_history.pop(0)

        # Simple pattern prediction: repeat last few accesses
        if len(self._access_history) >= 4:
            # Look for repeating patterns
            last_2 = self._access_history[-2:]
            for i in range(len(self._access_history) - 4, -1, -1):
                if self._access_history[i:i+2] == last_2:
                    # Found pattern, predict next
                    next_idx = i + 2
                    if next_idx < len(self._access_history):
                        pred = self._access_history[next_idx]
                        if pred not in self._prefetch_predictions:
                            self._prefetch_predictions.append(pred)
                    break

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cached_files': len(self._cache),
            'predictions': self._prefetch_predictions[:5],
        }

    def clear_cache(self):
        """Clear the LRU cache."""
        self._cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def write(self, fd: int, data: bytes) -> int:
        """Write to fd (pipe-aware), return bytes written or -1."""
        if fd not in self.fd_table:
            return -1
        entry = self.fd_table[fd]

        # Pipe write
        if entry.get("type") == "pipe_write":
            pipe_buf = entry.get("pipe_buffer")
            if pipe_buf:
                return pipe_buf.write(data)
            return -1

        # Cannot write directories
        if entry.get("is_dir"):
            return -1  # EISDIR

        # Check write permission
        mode = self._access_mode(entry["flags"])
        if mode == self.O_RDONLY:
            return -1  # EBADF (read-only fd)

        path = entry["path"]
        content = self.files.get(path, b"")
        offset = entry["offset"]

        # Size limit
        if offset + len(data) > MAX_FILE_SIZE:
            return -1  # EFBIG

        # Extend file if needed
        if offset > len(content):
            content = content + b'\x00' * (offset - len(content))

        # Write data at offset
        content = content[:offset] + data + content[offset + len(data):]
        self.files[path] = content

        # Invalidate cache on write
        if path in self._cache:
            del self._cache[path]
            if path in self._cache_order:
                self._cache_order.remove(path)

        entry["offset"] = offset + len(data)
        return len(data)

    def lseek(self, fd: int, offset: int, whence: int) -> int:
        """Seek in file, return new offset or -1."""
        if fd not in self.fd_table:
            return -1
        entry = self.fd_table[fd]
        path = entry["path"]
        size = len(self.files.get(path, b""))

        if whence == 0:    # SEEK_SET
            new_off = offset
        elif whence == 1:  # SEEK_CUR
            new_off = entry["offset"] + offset
        elif whence == 2:  # SEEK_END
            new_off = size + offset
        else:
            return -1

        if new_off < 0:
            return -1
        entry["offset"] = new_off
        return new_off

    def stat(self, path: str) -> Optional[dict]:
        """Stat a file or directory."""
        path = self.resolve_path(path)
        if self.is_symlink(path):
            return {"type": "symlink", "size": len(self.files.get(path, b"")), "path": path}
        elif path in self.files:
            return {"type": "file", "size": len(self.files[path]), "path": path}
        elif path in self.directories:
            return {"type": "dir", "size": 0, "path": path}
        return None

    def fstat(self, fd: int) -> Optional[dict]:
        """Stat by file descriptor."""
        if fd not in self.fd_table:
            return None
        return self.stat(self.fd_table[fd]["path"])

    # ═══════════════════════════════════════════════════════════════════════
    # DIRECTORY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════

    def mkdir(self, path: str) -> int:
        """Create directory. Returns 0 or -1."""
        path = self.resolve_path(path)
        if path in self.directories or path in self.files:
            return -1  # EEXIST
        parent = self._parent_dir(path)
        if parent != "/" and parent not in self.directories:
            return -1  # ENOENT
        self.directories.add(path)
        return 0

    def unlink(self, path: str) -> int:
        """Remove a file. Returns 0 or -1."""
        path = self.resolve_path(path)
        if path in self.files:
            # Close any open fds for this file
            to_close = [fd for fd, e in self.fd_table.items() if e["path"] == path]
            for fd in to_close:
                del self.fd_table[fd]
            del self.files[path]
            return 0
        return -1  # ENOENT

    def rmdir(self, path: str) -> int:
        """Remove an empty directory. Returns 0 or -1."""
        path = self.resolve_path(path)
        if path not in self.directories:
            return -1
        if path == "/":
            return -1

        # Check if directory is empty — use safe child check
        for f in self.files:
            if self._is_child_of(f, path) or (f.startswith(path + "/") if path != "/" else f.startswith("/")):
                return -1  # ENOTEMPTY
        for d in self.directories:
            if d != path and self._is_child_of(d, path):
                return -1  # ENOTEMPTY

        self.directories.discard(path)
        return 0

    def rename(self, old_path: str, new_path: str) -> int:
        """Rename/move a file or directory. Returns 0 or -1."""
        old_path = self.resolve_path(old_path)
        new_path = self.resolve_path(new_path)

        # Ensure new parent exists
        new_parent = self._parent_dir(new_path)
        if new_parent != "/" and new_parent not in self.directories:
            return -1  # ENOENT

        if old_path in self.files:
            self.files[new_path] = self.files.pop(old_path)
            # Update any open fd entries
            for entry in self.fd_table.values():
                if entry.get("path") == old_path:
                    entry["path"] = new_path
            # Invalidate cache
            if old_path in self._cache:
                del self._cache[old_path]
            return 0
        elif old_path in self.directories:
            # Rename directory and all children
            self.directories.discard(old_path)
            self.directories.add(new_path)
            prefix = old_path + "/"
            # Move child files
            to_move = [(k, v) for k, v in self.files.items() if k.startswith(prefix)]
            for k, v in to_move:
                new_k = new_path + k[len(old_path):]
                self.files[new_k] = self.files.pop(k)
            # Move child directories
            to_move_dirs = [d for d in self.directories if d.startswith(prefix)]
            for d in to_move_dirs:
                self.directories.discard(d)
                self.directories.add(new_path + d[len(old_path):])
            return 0
        return -1  # ENOENT

    def link(self, old_path: str, new_path: str) -> int:
        """Create a hard link (old_path -> new_path). Returns 0 or -1."""
        old_path = self.resolve_path(old_path)
        new_path = self.resolve_path(new_path)

        # Source must exist and be a file
        if old_path not in self.files:
            return -1  # ENOENT

        # Target must not exist
        if new_path in self.files or new_path in self.directories:
            return -1  # EEXIST

        # Target parent must exist
        new_parent = self._parent_dir(new_path)
        if new_parent != "/" and new_parent not in self.directories:
            return -1  # ENOENT

        # Create hard link - same inode (we use same content)
        self.files[new_path] = self.files[old_path]
        return 0

    def ftruncate(self, fd: int, length: int) -> int:
        """Truncate file to specified length. Returns 0 or -1."""
        if fd not in self.fd_table:
            return -1  # EBADF

        entry = self.fd_table[fd]
        path = entry.get("path")
        if not path or path not in self.files:
            return -1  # EBADF

        # Check write permission
        mode = self._access_mode(entry.get("flags", 0))
        if mode == self.O_RDONLY:
            return -1  # EBADF

        # Truncate or extend
        content = self.files[path]
        if length < len(content):
            self.files[path] = content[:length]
        else:
            self.files[path] = content + b'\x00' * (length - len(content))

        return 0

    def fsync(self, fd: int) -> int:
        """Sync file to storage (no-op in memory FS). Returns 0 or -1."""
        if fd not in self.fd_table:
            return -1  # EBADF
        return 0  # Always succeeds for memory FS

    def getcwd(self) -> str:
        return self.cwd

    def chdir(self, path: str) -> int:
        """Change working directory. Returns 0 or -1."""
        path = self.resolve_path(path)
        if path in self.directories:
            self.cwd = path
            return 0
        return -1  # ENOENT

    def listdir(self, path: str) -> Optional[list[str]]:
        """List directory contents. Returns list of names or None."""
        path = self.resolve_path(path)
        if path not in self.directories:
            return None

        entries = set()

        # Find direct children (files and subdirectories)
        for f in self.files:
            if self._is_child_of(f, path):
                name = f.split("/")[-1]
                if name:
                    entries.add(name)

        for d in self.directories:
            if self._is_child_of(d, path):
                name = d.split("/")[-1]
                if name:
                    entries.add(name)

        return sorted(entries)

    # ═══════════════════════════════════════════════════════════════════════
    # SYMLINK SUPPORT
    # ═══════════════════════════════════════════════════════════════════════

    def symlink(self, target: str, link_path: str) -> int:
        """Create a symbolic link at link_path pointing to target. Returns 0 or -1."""
        link_path = self.resolve_path(link_path)
        if link_path in self.files or link_path in self.directories:
            return -1  # EEXIST
        parent = self._parent_dir(link_path)
        if parent != "/" and parent not in self.directories:
            return -1  # ENOENT
        # Store symlink as a special file with target stored separately
        if not hasattr(self, 'symlinks'):
            self.symlinks = {}
        self.symlinks[link_path] = target
        # Also store as a file so it shows up in listings and stat
        self.files[link_path] = target.encode('utf-8')
        return 0

    def readlink(self, path: str) -> Optional[str]:
        """Read the target of a symbolic link. Returns target string or None."""
        path = self.resolve_path(path)
        if hasattr(self, 'symlinks') and path in self.symlinks:
            return self.symlinks[path]
        return None

    def is_symlink(self, path: str) -> bool:
        """Check if a path is a symbolic link."""
        path = self.resolve_path(path)
        return hasattr(self, 'symlinks') and path in self.symlinks

    # ═══════════════════════════════════════════════════════════════════════
    # CONVENIENCE
    # ═══════════════════════════════════════════════════════════════════════

    def write_file(self, path: str, content: bytes | str):
        """Direct write (for bootstrapping). Creates parent dirs."""
        path = self.resolve_path(path)
        if isinstance(content, str):
            content = content.encode("ascii")

        # Ensure parent directory exists
        parent = self._parent_dir(path)
        if parent != "/":
            self.directories.add(parent)

        self.files[path] = content

    def read_file(self, path: str) -> Optional[bytes]:
        """Direct read (for Python-side access)."""
        path = self.resolve_path(path)
        return self.files.get(path)

    def exists(self, path: str) -> bool:
        path = self.resolve_path(path)
        return path in self.files or path in self.directories

    def tree(self, indent: int = 0) -> str:
        """Return a tree representation of the filesystem."""
        lines = []
        all_paths = sorted(set(list(self.files.keys()) + list(self.directories)))
        for p in all_paths:
            depth = p.count("/") - 1 if p != "/" else 0
            name = p.split("/")[-1] or "/"
            prefix = "  " * depth
            if p in self.directories:
                lines.append(f"{prefix}{name}/")
            else:
                size = len(self.files[p])
                lines.append(f"{prefix}{name} ({size}B)")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # PIPE SUPPORT
    # ═══════════════════════════════════════════════════════════════════════

    def create_pipe(self) -> tuple[int, int]:
        """Create a pipe, returning (read_fd, write_fd) or (-1, -1) on error."""
        read_fd = self._allocate_fd()
        if read_fd < 0:
            return (-1, -1)

        pipe_buf = PipeBuffer()

        self.fd_table[read_fd] = {
            "type": "pipe_read",
            "pipe_buffer": pipe_buf,
            "path": "<pipe>",
            "offset": 0,
            "flags": self.O_RDONLY,
        }

        write_fd = self._allocate_fd()
        if write_fd < 0:
            del self.fd_table[read_fd]
            return (-1, -1)

        self.fd_table[write_fd] = {
            "type": "pipe_write",
            "pipe_buffer": pipe_buf,
            "path": "<pipe>",
            "offset": 0,
            "flags": self.O_WRONLY,
        }

        return (read_fd, write_fd)

    def dup2(self, old_fd: int, new_fd: int) -> int:
        """Duplicate old_fd to new_fd. Closes new_fd first if open. Returns new_fd or -1."""
        if old_fd not in self.fd_table:
            # Handle virtual fds (stdin/stdout/stderr) — create synthetic entries
            if old_fd in (0, 1, 2):
                names = {0: "<stdin>", 1: "<stdout>", 2: "<stderr>"}
                self.fd_table[old_fd] = {
                    "type": "virtual", "path": names[old_fd],
                    "offset": 0, "flags": self.O_RDWR,
                }
            else:
                return -1

        # Close new_fd if already open
        if new_fd in self.fd_table:
            self.close(new_fd)

        # Shallow copy the fd entry (pipe buffers are shared by reference)
        entry = self.fd_table[old_fd].copy()

        # Increment pipe refcounts
        pipe_buf = entry.get("pipe_buffer")
        if pipe_buf:
            if entry.get("type") == "pipe_read":
                pipe_buf.readers += 1
            elif entry.get("type") == "pipe_write":
                pipe_buf.writers += 1

        self.fd_table[new_fd] = entry
        return new_fd

    def clone_fd_table(self) -> dict[int, dict]:
        """Deep clone the fd table for fork(). Pipe buffers are shared by reference
        with incremented refcounts."""
        cloned = {}
        for fd, entry in self.fd_table.items():
            new_entry = entry.copy()
            pipe_buf = new_entry.get("pipe_buffer")
            if pipe_buf:
                if new_entry.get("type") == "pipe_read":
                    pipe_buf.readers += 1
                elif new_entry.get("type") == "pipe_write":
                    pipe_buf.writers += 1
            cloned[fd] = new_entry
        return cloned
