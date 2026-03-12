"""Rust-backed ELF execution with automatic Python fallback.

This module provides a unified interface for running ELF binaries on the GPU,
preferring the Rust implementation for performance but falling back to the
Python implementation if Rust is unavailable.
"""

from pathlib import Path
from typing import Optional

# Type alias for the result dict (matches both Rust and Python implementations)
ExecutionResult = dict


def run_elf(
    elf_path: str,
    argv: Optional[list[str]] = None,
    envp: Optional[dict[str, str]] = None,
    handler=None,
    max_cycles: int = 500_000_000,
    quiet: bool = False,
    filesystem: Optional["GPUFilesystem"] = None,
    stdin_data: Optional[bytes] = None,
    cpu=None,
    on_framebuffer: Optional[callable] = None,
    files: Optional[list[tuple[str, bytes]]] = None,
    directories: Optional[list[str]] = None,
) -> ExecutionResult:
    """Rust-backed ELF execution with automatic Python fallback.

    Args:
        elf_path: Path to the ELF binary
        argv: Command-line arguments (default: [binary_name])
        envp: Environment variables (default: standard env)
        max_cycles: Maximum CPU cycles to execute
        quiet: Suppress initialization output
        filesystem: GPUFilesystem instance for virtual file contents
        stdin_data: Bytes to serve as stdin (for piped input)
        cpu: Optional pre-existing CPU instance (for tracing)
        on_framebuffer: Callback for framebuffer updates (width, height, data)
        files: Optional list of (path, content) tuples for VFS
        directories: Optional list of directory paths to create

    Returns:
        Execution results dict with: stdout, stderr, exit_code, total_cycles,
        elapsed_secs, stop_reason, total_forks, total_context_switches,
        processes_created
    """
    # Try Rust implementation first
    try:
        import ncpu_metal

        # Convert filesystem to Rust format
        file_list = []
        dir_list = []
        symlink_list = []

        # If files/directories passed directly, use those
        if files is not None:
            file_list = files
        elif filesystem is not None:
            for path, content in filesystem.files.items():
                file_list.append((path, content))

        if directories is not None:
            dir_list = directories
        elif filesystem is not None and hasattr(filesystem, 'directories'):
            dir_list = list(filesystem.directories)

        # Convert symlinks from filesystem
        if filesystem is not None and hasattr(filesystem, 'symlinks'):
            for link_path, target in filesystem.symlinks.items():
                symlink_list.append((link_path, target))

        # Call Rust run_elf
        result = ncpu_metal.run_elf(
            elf_path=elf_path,
            argv=argv,
            envp=[f"{k}={v}" for k, v in (envp or {}).items()],
            max_cycles=max_cycles,
            quiet=quiet,
            stdin_data=stdin_data,
            files=file_list if file_list else None,
            directories=dir_list if dir_list else None,
            symlinks=symlink_list if symlink_list else None,
            on_framebuffer=on_framebuffer,
        )

        # Apply VFS changes back to the filesystem for persistence
        if filesystem is not None:
            vfs_files = result.get('vfs_files', {})
            vfs_dirs = result.get('vfs_directories', [])
            vfs_symlinks = result.get('vfs_symlinks', {})

            # Apply files
            for path, content in vfs_files.items():
                filesystem.files[path] = content

            # Apply directories
            if hasattr(filesystem, 'directories'):
                if isinstance(filesystem.directories, set):
                    for d in vfs_dirs:
                        filesystem.directories.add(d)
                elif isinstance(filesystem.directories, dict):
                    for d in vfs_dirs:
                        filesystem.directories.setdefault(d, None)

            # Apply symlinks
            if hasattr(filesystem, 'symlinks'):
                for target, link_name in vfs_symlinks.items():
                    filesystem.symlinks[link_name] = target

        # Convert to match Python API format
        # Normalize stop_reason: Rust returns EXIT for clean exit, Python returns SYSCALL
        stop_reason = result.get('stop_reason', 'UNKNOWN')
        if stop_reason == 'EXIT' and result.get('exit_code', 0) == 0:
            stop_reason = 'SYSCALL'  # Match Python behavior for clean exit

        return {
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'exit_code': result.get('exit_code', 0),
            'total_cycles': result.get('total_cycles', 0),
            'elapsed': result.get('elapsed_secs', 0),
            'stop_reason': stop_reason,
            'ips': result.get('total_cycles', 0) / max(result.get('elapsed_secs', 0.001), 0.001),
            '_rust': True,
            '_cpu': None,
        }
    except ImportError as e:
        # Fallback to Python implementation
        from ncpu.os.gpu.elf_loader import load_and_run_elf

        return load_and_run_elf(
            elf_path=elf_path,
            argv=argv,
            envp=envp,
            handler=None,
            max_cycles=max_cycles,
            quiet=quiet,
            filesystem=filesystem,
            stdin_data=stdin_data,
            cpu=None,
        )
    except Exception as e:
        # If Rust fails for any other reason, also try Python fallback
        # but propagate if both fail
        try:
            from ncpu.os.gpu.elf_loader import load_and_run_elf

            return load_and_run_elf(
                elf_path=elf_path,
                argv=argv,
                envp=envp,
                handler=None,
                max_cycles=max_cycles,
                quiet=quiet,
                filesystem=filesystem,
                stdin_data=stdin_data,
                cpu=None,
            )
        except Exception:
            # Re-raise original Rust error if Python also fails
            raise e from None


def get_backend() -> str:
    """Return the backend being used: 'rust' or 'python'."""
    try:
        import ncpu_metal
        return "rust"
    except ImportError:
        return "python"
