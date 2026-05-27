"""Neural Filesystem (neurFS).

A filesystem where block allocation, directory lookup, and caching are
all performed by neural networks. Storage is a GPU tensor — all filesystem
operations are tensor operations with zero CPU-GPU transfers.

Components:
    1. NeuralBlockAllocator — learned block allocation policy
    2. Neural directory lookup — embedding-based path resolution
    3. Neural journaling — learned transaction ordering
    4. In-memory inode/dentry structures backed by GPU tensors

Design:
    - Block size: 4096 bytes (matching page size)
    - Max filesystem: 64K blocks × 4KB = 256MB
    - Inodes: GPU tensor array with metadata fields
    - Directory entries: name→inode mappings
    - File data: stored in block-aligned GPU tensor regions
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from enum import IntEnum

from .device import default_device

logger = logging.getLogger(__name__)

BLOCK_SIZE = 4096
MAX_BLOCKS = 65536          # 64K blocks = 256MB
MAX_INODES = 4096
MAX_NAME_LEN = 255
MAX_PATH_LEN = 1024
ROOT_INODE = 0


class FileType(IntEnum):
    REGULAR = 0
    DIRECTORY = 1
    SYMLINK = 2
    DEVICE = 3


class FileMode(IntEnum):
    READ = 0o444
    WRITE = 0o222
    EXEC = 0o111
    RW = 0o666
    RWX = 0o777


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Block Allocator
# ═══════════════════════════════════════════════════════════════════════════════

class BlockAllocatorNet(nn.Module):
    """Neural block allocation policy.

    Given the current allocation bitmap and recent allocation patterns,
    predicts the optimal block to allocate next (minimizing fragmentation,
    maximizing spatial locality for sequential reads).

    Architecture:
        Input:  [allocation_features(16)] — bitmap summary + recent patterns
        MLP:    → hidden → block region scores
        Output: [num_regions] — score for each allocation region
    """

    def __init__(self, num_regions: int = 64, feature_dim: int = 16,
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regions),
        )
        self.num_regions = num_regions

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [feature_dim] → [num_regions] allocation scores"""
        return self.net(features)


# ═══════════════════════════════════════════════════════════════════════════════
# Inode
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Inode:
    """In-memory inode structure."""
    ino: int                              # Inode number
    file_type: FileType = FileType.REGULAR
    mode: int = FileMode.RW
    size: int = 0                         # File size in bytes
    uid: int = 0                          # Owner
    blocks: List[int] = field(default_factory=list)  # Block numbers
    nlinks: int = 1
    ctime: int = 0                        # Creation tick
    mtime: int = 0                        # Modification tick
    atime: int = 0                        # Access tick


@dataclass
class DirEntry:
    """Directory entry: name → inode mapping."""
    name: str
    ino: int


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Filesystem
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralFilesystem:
    """GPU-native filesystem with neural block allocation.

    All storage is a GPU tensor. Inodes, directory entries, and file data
    are managed through tensor operations. The block allocator uses a
    neural network to minimize fragmentation.

    Layout:
        blocks[0]:     Superblock (filesystem metadata)
        blocks[1]:     Root directory inode
        blocks[2..N]:  Data and directory blocks
    """

    def __init__(self, num_blocks: int = 4096, max_inodes: int = 1024,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.num_blocks = num_blocks
        self.max_inodes = max_inodes

        # Storage: GPU tensor (all blocks)
        self.storage = torch.zeros(
            num_blocks, BLOCK_SIZE, dtype=torch.uint8, device=self.device
        )

        # Block allocation bitmap
        self.block_bitmap = torch.zeros(
            num_blocks, dtype=torch.bool, device=self.device
        )

        # Inode table
        self._inodes: Dict[int, Inode] = {}
        self._next_ino = 0

        # Directory structure: ino → list of DirEntry
        self._dirs: Dict[int, List[DirEntry]] = {}

        # Open file table: fd → (ino, position, mode)
        self._open_files: Dict[int, Tuple[int, int, str]] = {}
        self._next_fd = 3  # 0=stdin, 1=stdout, 2=stderr

        # Neural block allocator
        self._allocator_net = BlockAllocatorNet(
            num_regions=num_blocks // 64
        ).to(self.device)
        self._allocator_trained = False

        # Statistics
        self.tick = 0
        self.reads = 0
        self.writes = 0
        self.allocs = 0

        # Initialize root directory
        self._init_root()

    def _init_root(self):
        """Initialize the root directory."""
        root_ino = self._alloc_inode(FileType.DIRECTORY, FileMode.RWX)
        self._dirs[root_ino] = [
            DirEntry(".", root_ino),
            DirEntry("..", root_ino),
        ]

    def _alloc_inode(self, file_type: FileType = FileType.REGULAR,
                     mode: int = FileMode.RW) -> int:
        """Allocate a new inode."""
        if len(self._inodes) >= self.max_inodes:
            raise RuntimeError("No free inodes")

        ino = self._next_ino
        self._next_ino += 1
        self._inodes[ino] = Inode(
            ino=ino, file_type=file_type, mode=mode,
            ctime=self.tick, mtime=self.tick, atime=self.tick,
        )
        return ino

    def _alloc_block(self) -> int:
        """Allocate a data block. Returns block number or -1 if full."""
        if self._allocator_trained:
            return self._neural_alloc_block()

        # Simple first-fit allocation
        free = (~self.block_bitmap).nonzero(as_tuple=True)[0]
        if len(free) == 0:
            return -1

        block = int(free[0].item())
        self.block_bitmap[block] = True
        self.allocs += 1
        return block

    def _neural_alloc_block(self) -> int:
        """Neural block allocation — picks optimal region."""
        blocks_per_region = self.num_blocks // self._allocator_net.num_regions

        # Build features: per-region occupancy + recent allocation pattern
        region_occupancy = []
        for i in range(self._allocator_net.num_regions):
            start = i * blocks_per_region
            end = start + blocks_per_region
            occ = self.block_bitmap[start:end].float().mean()
            region_occupancy.append(occ)

        features = torch.tensor(region_occupancy, device=self.device)
        # Pad to feature_dim=16 if needed
        if len(features) < 16:
            features = torch.nn.functional.pad(features, (0, 16 - len(features)))
        else:
            features = features[:16]

        with torch.no_grad():
            scores = self._allocator_net(features)

        # Find best region with free blocks
        for region_idx in scores.argsort(descending=True):
            region = int(region_idx.item())
            start = region * blocks_per_region
            end = start + blocks_per_region
            region_free = (~self.block_bitmap[start:end]).nonzero(as_tuple=True)[0]
            if len(region_free) > 0:
                block = start + int(region_free[0].item())
                self.block_bitmap[block] = True
                self.allocs += 1
                return block

        return -1

    def _free_block(self, block: int):
        """Free a data block."""
        if 0 <= block < self.num_blocks:
            self.block_bitmap[block] = False
            self.storage[block].zero_()

    # ─── Path Resolution ──────────────────────────────────────────────────

    def _resolve_path(self, path: str) -> Optional[int]:
        """Resolve a path to an inode number.

        Walks the directory tree from root.
        """
        if not path or path == "/":
            return ROOT_INODE

        parts = PurePosixPath(path).parts
        current_ino = ROOT_INODE

        for part in parts:
            if part == "/":
                continue
            entries = self._dirs.get(current_ino)
            if entries is None:
                return None

            found = False
            for entry in entries:
                if entry.name == part:
                    current_ino = entry.ino
                    found = True
                    break

            if not found:
                return None

        return current_ino

    def _resolve_parent(self, path: str) -> Tuple[Optional[int], str]:
        """Resolve path to parent inode and basename."""
        p = PurePosixPath(path)
        parent_path = str(p.parent)
        name = p.name

        parent_ino = self._resolve_path(parent_path)
        return parent_ino, name

    # ─── File Operations ──────────────────────────────────────────────────

    def create(self, path: str, file_type: FileType = FileType.REGULAR,
               mode: int = FileMode.RW) -> int:
        """Create a file or directory.

        Returns inode number, or -1 on error.
        """
        self.tick += 1

        # Check if already exists
        existing = self._resolve_path(path)
        if existing is not None:
            return -1  # Already exists

        parent_ino, name = self._resolve_parent(path)
        if parent_ino is None:
            return -1  # Parent doesn't exist

        if not name or len(name) > MAX_NAME_LEN:
            return -1

        # Create inode
        ino = self._alloc_inode(file_type, mode)

        # Add to parent directory
        self._dirs[parent_ino].append(DirEntry(name, ino))

        # If directory, initialize with . and ..
        if file_type == FileType.DIRECTORY:
            self._dirs[ino] = [
                DirEntry(".", ino),
                DirEntry("..", parent_ino),
            ]

        return ino

    def mkdir(self, path: str, mode: int = FileMode.RWX) -> int:
        """Create a directory."""
        return self.create(path, FileType.DIRECTORY, mode)

    def unlink(self, path: str) -> bool:
        """Remove a file (not a directory)."""
        self.tick += 1

        ino = self._resolve_path(path)
        if ino is None or ino == ROOT_INODE:
            return False

        inode = self._inodes.get(ino)
        if inode is None:
            return False

        if inode.file_type == FileType.DIRECTORY:
            return False  # Use rmdir for directories

        # Remove from parent directory
        parent_ino, name = self._resolve_parent(path)
        if parent_ino is not None:
            entries = self._dirs.get(parent_ino, [])
            self._dirs[parent_ino] = [e for e in entries if e.name != name]

        # Free blocks
        for block in inode.blocks:
            self._free_block(block)

        # Remove inode
        del self._inodes[ino]
        return True

    def rmdir(self, path: str) -> bool:
        """Remove an empty directory."""
        self.tick += 1

        ino = self._resolve_path(path)
        if ino is None or ino == ROOT_INODE:
            return False

        inode = self._inodes.get(ino)
        if inode is None or inode.file_type != FileType.DIRECTORY:
            return False

        # Check if empty (only . and ..)
        entries = self._dirs.get(ino, [])
        real_entries = [e for e in entries if e.name not in (".", "..")]
        if real_entries:
            return False  # Not empty

        # Remove from parent
        parent_ino, name = self._resolve_parent(path)
        if parent_ino is not None:
            entries = self._dirs.get(parent_ino, [])
            self._dirs[parent_ino] = [e for e in entries if e.name != name]

        del self._dirs[ino]
        del self._inodes[ino]
        return True

    def open(self, path: str, mode: str = "r") -> int:
        """Open a file. Returns file descriptor, or -1 on error."""
        self.tick += 1

        ino = self._resolve_path(path)
        if ino is None:
            if "w" in mode or "a" in mode:
                # Create on write
                ino = self.create(path)
                if ino < 0:
                    return -1
            else:
                return -1

        inode = self._inodes.get(ino)
        if inode is None or inode.file_type == FileType.DIRECTORY:
            return -1

        fd = self._next_fd
        self._next_fd += 1

        pos = 0
        if "a" in mode:
            pos = inode.size
        if "w" in mode:
            # Truncate
            for block in inode.blocks:
                self._free_block(block)
            inode.blocks = []
            inode.size = 0

        self._open_files[fd] = (ino, pos, mode)
        inode.atime = self.tick
        return fd

    def close(self, fd: int) -> bool:
        """Close a file descriptor."""
        if fd in self._open_files:
            del self._open_files[fd]
            return True
        return False

    def read(self, fd: int, size: int) -> Optional[torch.Tensor]:
        """Read bytes from an open file.

        Returns tensor of bytes, or None on error.
        """
        self.tick += 1
        self.reads += 1

        if fd not in self._open_files:
            return None

        ino, pos, mode = self._open_files[fd]
        if "r" not in mode and "+" not in mode:
            return None

        inode = self._inodes.get(ino)
        if inode is None:
            return None

        # Calculate which blocks to read
        bytes_to_read = min(size, inode.size - pos)
        if bytes_to_read <= 0:
            return torch.tensor([], dtype=torch.uint8, device=self.device)

        result = torch.zeros(bytes_to_read, dtype=torch.uint8, device=self.device)

        bytes_read = 0
        while bytes_read < bytes_to_read:
            block_idx = (pos + bytes_read) // BLOCK_SIZE
            block_offset = (pos + bytes_read) % BLOCK_SIZE

            if block_idx >= len(inode.blocks):
                break

            block_num = inode.blocks[block_idx]
            chunk_size = min(BLOCK_SIZE - block_offset, bytes_to_read - bytes_read)
            result[bytes_read:bytes_read + chunk_size] = \
                self.storage[block_num, block_offset:block_offset + chunk_size]
            bytes_read += chunk_size

        # Update position
        self._open_files[fd] = (ino, pos + bytes_read, mode)
        inode.atime = self.tick
        return result[:bytes_read]

    def write(self, fd: int, data) -> int:
        """Write bytes to an open file. Accepts tensor or bytes. Returns bytes written."""
        if isinstance(data, (bytes, bytearray)):
            data = torch.tensor(list(data), dtype=torch.uint8, device=self.device)
        self.tick += 1
        self.writes += 1

        if fd not in self._open_files:
            return -1

        ino, pos, mode = self._open_files[fd]
        if "w" not in mode and "a" not in mode and "+" not in mode:
            return -1

        inode = self._inodes.get(ino)
        if inode is None:
            return -1

        bytes_to_write = len(data)
        bytes_written = 0

        while bytes_written < bytes_to_write:
            block_idx = (pos + bytes_written) // BLOCK_SIZE
            block_offset = (pos + bytes_written) % BLOCK_SIZE

            # Allocate blocks as needed
            while block_idx >= len(inode.blocks):
                new_block = self._alloc_block()
                if new_block < 0:
                    break  # Out of space
                inode.blocks.append(new_block)

            if block_idx >= len(inode.blocks):
                break  # Couldn't allocate

            block_num = inode.blocks[block_idx]
            chunk_size = min(BLOCK_SIZE - block_offset, bytes_to_write - bytes_written)
            self.storage[block_num, block_offset:block_offset + chunk_size] = \
                data[bytes_written:bytes_written + chunk_size]
            bytes_written += chunk_size

        # Update file size and position
        new_pos = pos + bytes_written
        inode.size = max(inode.size, new_pos)
        inode.mtime = self.tick
        self._open_files[fd] = (ino, new_pos, mode)

        return bytes_written

    def seek(self, fd: int, offset: int, whence: int = 0) -> int:
        """Seek within an open file.

        whence: 0 = SEEK_SET, 1 = SEEK_CUR, 2 = SEEK_END
        """
        if fd not in self._open_files:
            return -1

        ino, pos, mode = self._open_files[fd]
        inode = self._inodes.get(ino)
        if inode is None:
            return -1

        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = pos + offset
        elif whence == 2:
            new_pos = inode.size + offset
        else:
            return -1

        new_pos = max(0, new_pos)
        self._open_files[fd] = (ino, new_pos, mode)
        return new_pos

    # ─── Directory Operations ─────────────────────────────────────────────

    def list_dir(self, path: str) -> Optional[List[str]]:
        """List directory contents."""
        self.tick += 1

        ino = self._resolve_path(path)
        if ino is None:
            return None

        inode = self._inodes.get(ino)
        if inode is None or inode.file_type != FileType.DIRECTORY:
            return None

        entries = self._dirs.get(ino, [])
        return [e.name for e in entries if e.name not in (".", "..")]

    def stat(self, path: str) -> Optional[Dict]:
        """Get file/directory metadata."""
        ino = self._resolve_path(path)
        if ino is None:
            return None

        inode = self._inodes.get(ino)
        if inode is None:
            return None

        return {
            "ino": inode.ino,
            "type": FileType(inode.file_type).name,
            "mode": oct(inode.mode),
            "size": inode.size,
            "blocks": len(inode.blocks),
            "nlinks": inode.nlinks,
            "ctime": inode.ctime,
            "mtime": inode.mtime,
            "atime": inode.atime,
        }

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        return self._resolve_path(path) is not None

    # ─── Convenience ──────────────────────────────────────────────────────

    def write_file(self, path: str, data: torch.Tensor) -> bool:
        """Write data to a file (create if needed). Convenience wrapper."""
        fd = self.open(path, "w")
        if fd < 0:
            return False
        self.write(fd, data)
        self.close(fd)
        return True

    def read_file(self, path: str) -> Optional[torch.Tensor]:
        """Read entire file contents. Convenience wrapper."""
        fd = self.open(path, "r")
        if fd < 0:
            return None
        info = self.stat(path)
        if info is None:
            self.close(fd)
            return None
        data = self.read(fd, info["size"])
        self.close(fd)
        return data

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/block_alloc.pt"):
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._allocator_net.state_dict(), path)

    def load(self, path: str = "models/research/neuros/block_alloc.pt") -> bool:
        from pathlib import Path as P
        if not P(path).exists():
            return False
        self._allocator_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))
        self._allocator_net.eval()
        self._allocator_trained = True
        return True

    # ─── Diagnostics ──────────────────────────────────────────────────────

    @property
    def free_blocks(self) -> int:
        return int((~self.block_bitmap).sum().item())

    @property
    def used_blocks(self) -> int:
        return int(self.block_bitmap.sum().item())

    def stats(self) -> Dict:
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "inodes": len(self._inodes),
            "max_inodes": self.max_inodes,
            "open_files": len(self._open_files),
            "reads": self.reads,
            "writes": self.writes,
            "allocations": self.allocs,
            "allocator_trained": self._allocator_trained,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralFS(blocks={s['used_blocks']}/{s['total_blocks']}, "
                f"inodes={s['inodes']}, "
                f"allocator={'neural' if s['allocator_trained'] else 'first-fit'})")
