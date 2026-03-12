"""Tests for Rust backend parity with Python implementation.

This verifies that the Rust run_elf produces the same results as the Python implementation.
"""

import pytest
from pathlib import Path

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")


class TestRustBackend:
    """Test the Rust backend implementation."""

    def test_echo_basic(self):
        """Test basic echo command."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["echo", "hello"],
            quiet=True
        )
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_echo_multiple_args(self):
        """Test echo with multiple arguments."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["echo", "a", "b", "c"],
            quiet=True
        )
        assert result["stdout"].strip() == "a b c"
        assert result["exit_code"] == 0

    def test_cat_file(self):
        """Test cat command with a file."""
        from ncpu.os.gpu.rust_backend import run_elf
        from ncpu.os.gpu.filesystem import GPUFilesystem

        fs = GPUFilesystem()
        fs.write_file("/test.txt", b"file content\n")

        result = run_elf(
            BUSYBOX,
            argv=["cat", "/test.txt"],
            quiet=True,
            filesystem=fs
        )
        assert result["stdout"].strip() == "file content"
        assert result["exit_code"] == 0

    def test_ls_root(self):
        """Test ls command on root directory."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["ls", "/"],
            quiet=True
        )
        assert "bin" in result["stdout"]
        assert result["exit_code"] == 0

    def test_true_command(self):
        """Test true command returns exit code 0."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["true"],
            quiet=True
        )
        assert result["exit_code"] == 0

    def test_false_command(self):
        """Test false command returns non-zero exit code."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["false"],
            quiet=True
        )
        assert result["exit_code"] != 0

    def test_stdin_data(self):
        """Test piped stdin data."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["cat"],
            quiet=True,
            stdin_data=b"stdin content\n"
        )
        assert result["stdout"] == "stdin content\n"
        assert result["exit_code"] == 0

    def test_filesystem_directories(self):
        """Test creating directories in VFS."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["ls", "/mydir"],
            quiet=True,
            files=[("/mydir/test.txt", b"content")],
            directories=["/mydir"]
        )
        # Should list the directory without error
        assert result["exit_code"] == 0

    def test_uname(self):
        """Test uname command."""
        from ncpu.os.gpu.rust_backend import run_elf

        result = run_elf(
            BUSYBOX,
            argv=["uname", "-s"],
            quiet=True
        )
        assert "Linux" in result["stdout"]
        assert result["exit_code"] == 0

    def test_get_backend(self):
        """Test backend detection."""
        from ncpu.os.gpu.rust_backend import get_backend

        # Should return 'rust' since the Rust module is available
        assert get_backend() == "rust"


class TestRustVsPythonParity:
    """Test parity between Rust and Python implementations."""

    def test_echo_parity(self):
        """Verify Rust path produces same output as Python path."""
        from ncpu.os.gpu.rust_backend import run_elf as rust_run_elf
        from ncpu.os.gpu.elf_loader import load_and_run_elf as python_run_elf

        rust_result = rust_run_elf(BUSYBOX, argv=["echo", "parity test"], quiet=True)
        py_result = python_run_elf(BUSYBOX, argv=["echo", "parity test"], quiet=True)

        assert rust_result["stdout"] == py_result["stdout"]
        assert rust_result["exit_code"] == py_result["exit_code"]

    def test_cat_parity(self):
        """Verify cat produces same output."""
        from ncpu.os.gpu.rust_backend import run_elf as rust_run_elf
        from ncpu.os.gpu.elf_loader import load_and_run_elf as python_run_elf
        from ncpu.os.gpu.filesystem import GPUFilesystem

        fs = GPUFilesystem()
        fs.write_file("/test.txt", b"test content")

        rust_result = rust_run_elf(BUSYBOX, argv=["cat", "/test.txt"], quiet=True, filesystem=fs)
        py_result = python_run_elf(BUSYBOX, argv=["cat", "/test.txt"], quiet=True, filesystem=fs)

        assert rust_result["stdout"] == py_result["stdout"]
        assert rust_result["exit_code"] == py_result["exit_code"]

    def test_stdin_parity(self):
        """Verify stdin produces same output."""
        from ncpu.os.gpu.rust_backend import run_elf as rust_run_elf
        from ncpu.os.gpu.elf_loader import load_and_run_elf as python_run_elf

        stdin_data = b"test input\n"

        rust_result = rust_run_elf(BUSYBOX, argv=["cat"], quiet=True, stdin_data=stdin_data)
        py_result = python_run_elf(BUSYBOX, argv=["cat"], quiet=True, stdin_data=stdin_data)

        assert rust_result["stdout"] == py_result["stdout"]
        assert rust_result["exit_code"] == py_result["exit_code"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
