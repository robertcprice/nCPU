"""
Neural RTOS Enhanced Filesystem
================================
Persistent file storage with save/load capabilities.

Features:
- Persistent storage to JSON files
- File CRUD operations (Create, Read, Update, Delete)
- Directory support
- File metadata (permissions, timestamps)
- Import/export from host system
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class RTOSFile:
    """Represents a single file in the RTOS filesystem."""

    def __init__(self, name: str, content: str = "", permissions: int = 0o644):
        self.name = name
        self.content = content
        self.permissions = permissions
        self.created_at = time.time()
        self.modified_at = time.time()
        self.size = len(content)

    def update_content(self, new_content: str):
        """Update file content and timestamp."""
        self.content = new_content
        self.modified_at = time.time()
        self.size = len(new_content)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "content": self.content,
            "permissions": self.permissions,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "size": self.size
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RTOSFile":
        """Create from dictionary."""
        file = cls(data["name"], data["content"], data["permissions"])
        file.created_at = data["created_at"]
        file.modified_at = data["modified_at"]
        file.size = data["size"]
        return file


class RTOSFileSystem:
    """
    Enhanced filesystem for Neural RTOS with persistent storage.

    Memory layout (compatible with neural_rtos.c):
    - 0x30000 - 0x3FFFF: File data region
    - Metadata managed separately in Python
    """

    def __init__(self, storage_path: str = ".rtos_fs"):
        self.storage_path = storage_path
        self.files: Dict[str, RTOSFile] = {}
        self.max_file_size = 10240  # 10KB per file
        self.max_files = 32
        self.load()

    def create(self, name: str, content: str = "", permissions: int = 0o644) -> bool:
        """Create a new file."""
        if len(self.files) >= self.max_files:
            return False
        if name in self.files:
            return False
        if len(content) > self.max_file_size:
            return False

        self.files[name] = RTOSFile(name, content, permissions)
        self.save()
        return True

    def read(self, name: str) -> Optional[str]:
        """Read file content."""
        if name not in self.files:
            return None
        return self.files[name].content

    def write(self, name: str, content: str, append: bool = False) -> bool:
        """Write to file (create or update)."""
        if name in self.files:
            # Update existing
            if append:
                new_content = self.files[name].content + content
            else:
                new_content = content
            if len(new_content) > self.max_file_size:
                return False
            self.files[name].update_content(new_content)
        else:
            # Create new
            if len(content) > self.max_file_size:
                return False
            if len(self.files) >= self.max_files:
                return False
            self.files[name] = RTOSFile(name, content)

        self.save()
        return True

    def delete(self, name: str) -> bool:
        """Delete a file."""
        if name not in self.files:
            return False
        del self.files[name]
        self.save()
        return True

    def exists(self, name: str) -> bool:
        """Check if file exists."""
        return name in self.files

    def list_files(self) -> List[RTOSFile]:
        """List all files."""
        return list(self.files.values())

    def get_file(self, name: str) -> Optional[RTOSFile]:
        """Get file metadata."""
        return self.files.get(name)

    def import_from_host(self, host_path: str, rtos_name: str = None) -> bool:
        """Import a file from the host filesystem."""
        if not os.path.exists(host_path):
            return False

        if rtos_name is None:
            rtos_name = os.path.basename(host_path)

        try:
            with open(host_path, 'r') as f:
                content = f.read()

            if len(content) > self.max_file_size:
                # For large files, truncate
                content = content[:self.max_file_size]

            return self.create(rtos_name, content)
        except Exception as e:
            print(f"Error importing file: {e}")
            return False

    def export_to_host(self, rtos_name: str, host_path: str = None) -> bool:
        """Export a file to the host filesystem."""
        if rtos_name not in self.files:
            return False

        if host_path is None:
            host_path = rtos_name

        try:
            with open(host_path, 'w') as f:
                f.write(self.files[rtos_name].content)
            return True
        except Exception as e:
            print(f"Error exporting file: {e}")
            return False

    def save(self) -> bool:
        """Save filesystem to disk."""
        try:
            data = {
                "files": [f.to_dict() for f in self.files.values()],
                "metadata": {
                    "max_files": self.max_files,
                    "max_file_size": self.max_file_size,
                    "saved_at": time.time()
                }
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving filesystem: {e}")
            return False

    def load(self) -> bool:
        """Load filesystem from disk."""
        if not os.path.exists(self.storage_path):
            # Create default files
            self._create_defaults()
            return True

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.files = {}
            for file_data in data.get("files", []):
                file = RTOSFile.from_dict(file_data)
                self.files[file.name] = file

            return True
        except Exception as e:
            print(f"Error loading filesystem: {e}")
            self._create_defaults()
            return True

    def _create_defaults(self):
        """Create default files."""
        welcome = """Welcome to Neural RTOS!
========================

This OS runs entirely on the Neural CPU.
Every instruction is a neural network forward pass.

Commands:
  help      - Show this help
  ls        - List files
  cat <f>   - Show file contents
  echo <t>  - Print text
  calc      - Neural calculator
  mem       - Memory info
  regs      - Register state
  clear     - Clear screen
  info      - System info
  touch <f> - Create empty file
  rm <f>    - Delete file
  edit <f>  - Edit file (line by line)
  import <path> - Import file from host
  export <file> [path] - Export file to host
"""

        readme = """Neural RTOS Filesystem
=======================

Files are stored persistently in .rtos_fs
You can import/export files between RTOS and host.

Examples:
  import /etc/hostname
  export welcome.txt my_copy.txt
  edit myfile.txt

The filesystem supports:
- Up to 32 files
- Maximum 10KB per file
- Text files only
"""

        self.create("welcome.txt", welcome)
        self.create("readme.txt", readme)
        self.save()

    def sync_to_memory(self, cpu_memory) -> bool:
        """
        Sync filesystem to neural CPU memory.
        Writes file data to 0x30000 region.
        """
        try:
            offset = 0
            # File entries (simplified format)
            for file in self.files.values():
                if offset + len(file.content) >= 0x10000:
                    break

                # Write file content
                for i, char in enumerate(file.content):
                    cpu_memory.write8(0x30000 + offset + i, ord(char))

                offset += len(file.content) + 1  # +1 for null terminator

            return True
        except Exception as e:
            print(f"Error syncing to memory: {e}")
            return False

    def get_stats(self) -> dict:
        """Get filesystem statistics."""
        return {
            "total_files": len(self.files),
            "max_files": self.max_files,
            "total_size": sum(f.size for f in self.files.values()),
            "max_file_size": self.max_file_size,
            "storage_path": self.storage_path
        }


# Global filesystem instance
_fs_instance = None

def get_filesystem() -> RTOSFileSystem:
    """Get or create the global filesystem instance."""
    global _fs_instance
    if _fs_instance is None:
        _fs_instance = RTOSFileSystem()
    return _fs_instance
