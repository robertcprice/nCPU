"""
Neural RTOS Enhanced Shell
==========================
Extended shell with file operations and Linux-like commands.

New commands:
- touch <file>     - Create empty file
- rm <file>        - Delete file
- edit <file>      - Edit file line by line
- import <path>    - Import file from host
- export <file>    - Export file to host
- cp <src> <dst>   - Copy file
- mv <src> <dst>   - Move/rename file
- stat <file>      - Show file metadata
- df               - Show filesystem stats
- date             - Show current date/time
- uname            - Show system info
- pwd              - Print working directory
- env              - Show environment variables
"""

from rtos_filesystem import get_filesystem, RTOSFileSystem
import time


class RTOSShell:
    """Enhanced shell for Neural RTOS."""

    def __init__(self, cpu=None, fb=None):
        self.fs = get_filesystem()
        self.cpu = cpu
        self.fb = fb
        self.env = {
            "PATH": "/bin:/usr/bin",
            "HOME": "/root",
            "USER": "root",
            "SHELL": "/bin/neuralsh",
            "TERM": "vt100",
            "HOSTNAME": "neural-rtos",
            "PS1": "\\u@\\h:\\w\\$ "
        }

    def execute(self, cmd: str) -> str:
        """Execute a shell command and return output."""
        cmd = cmd.strip()
        if not cmd:
            return ""

        parts = cmd.split()
        cmd_name = parts[0].lower()
        args = parts[1:]

        # Route to appropriate handler
        handler = getattr(self, f"cmd_{cmd_name}", None)
        if handler:
            try:
                return handler(args)
            except Exception as e:
                return f"Error: {e}\n"
        else:
            return f"{cmd_name}: command not found\n"

    def cmd_help(self, args):
        """Show help."""
        return """
=== NEURAL RTOS SHELL ===

File Operations:
  ls              - List files
  cat <file>      - Show file contents
  touch <file>    - Create empty file
  rm <file>       - Delete file
  edit <file>     - Edit file line by line
  cp <src> <dst>  - Copy file
  mv <src> <dst>  - Move/rename file

Import/Export:
  import <path>   - Import file from host system
  export <file> [path] - Export file to host system

File Info:
  stat <file>     - Show file metadata
  df              - Show filesystem statistics

System:
  echo <text>     - Print text
  calc <a> <op> <b> - Calculator (+, -, *, /, &, |, ^)
  mem             - Memory info
  date            - Show date/time
  uname [-a]      - System information
  hostname        - Show hostname
  pwd             - Print working directory
  env             - Show environment variables
  clear           - Clear screen
  reboot          - Reboot the system

"""

    def cmd_ls(self, args):
        """List files."""
        files = self.fs.list_files()
        if not files:
            return "No files.\n"

        output = []
        for f in sorted(files, key=lambda x: x.name):
            perms = self._format_perms(f.permissions)
            size = f.size
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.modified_at))
            output.append(f"{perms} {size:6d} {mtime} {f.name}")

        return "\n".join(output) + "\n"

    def cmd_cat(self, args):
        """Show file contents."""
        if not args:
            return "Usage: cat <file>\n"

        name = args[0]
        content = self.fs.read(name)
        if content is None:
            return f"cat: {name}: No such file\n"

        return content + "\n"

    def cmd_touch(self, args):
        """Create empty file."""
        if not args:
            return "Usage: touch <file>\n"

        name = args[0]
        if self.fs.exists(name):
            return f"touch: {name}: File exists\n"

        if self.fs.create(name):
            return f"Created: {name}\n"
        else:
            return f"touch: {name}: Cannot create file\n"

    def cmd_rm(self, args):
        """Delete file."""
        if not args:
            return "Usage: rm <file>\n"

        name = args[0]
        if not self.fs.exists(name):
            return f"rm: {name}: No such file\n"

        if self.fs.delete(name):
            return f"Removed: {name}\n"
        else:
            return f"rm: {name}: Cannot delete\n"

    def cmd_edit(self, args):
        """Edit file line by line."""
        if not args:
            return "Usage: edit <file>\n"

        name = args[0]
        current = self.fs.read(name)
        if current is None:
            current = ""

        lines = current.split('\n') if current else ['']

        output = [f"Editing: {name}"]
        output.append("Lines: " + str(len(lines)))
        output.append("---")
        for i, line in enumerate(lines):
            output.append(f"{i+1:3d}: {line}")
        output.append("---")
        output.append("Use 'echo <line> >> <file>' to append")
        output.append("Use 'cat <file> > <newfile>' to copy")

        return "\n".join(output) + "\n"

    def cmd_cp(self, args):
        """Copy file."""
        if len(args) < 2:
            return "Usage: cp <src> <dst>\n"

        src, dst = args[0], args[1]
        content = self.fs.read(src)
        if content is None:
            return f"cp: {src}: No such file\n"

        if self.fs.create(dst, content):
            return f"Copied: {src} -> {dst}\n"
        else:
            return f"cp: Cannot create {dst}\n"

    def cmd_mv(self, args):
        """Move/rename file."""
        if len(args) < 2:
            return "Usage: mv <src> <dst>\n"

        src, dst = args[0], args[1]
        content = self.fs.read(src)
        if content is None:
            return f"mv: {src}: No such file\n"

        if self.fs.create(dst, content):
            self.fs.delete(src)
            return f"Moved: {src} -> {dst}\n"
        else:
            return f"mv: Cannot create {dst}\n"

    def cmd_import(self, args):
        """Import file from host."""
        if not args:
            return "Usage: import <host_path> [rtos_name]\n"

        host_path = args[0]
        rtos_name = args[1] if len(args) > 1 else None

        if self.fs.import_from_host(host_path, rtos_name):
            name = rtos_name or host_path
            return f"Imported: {host_path} -> {name}\n"
        else:
            return f"import: Failed to import {host_path}\n"

    def cmd_export(self, args):
        """Export file to host."""
        if not args:
            return "Usage: export <rtos_file> [host_path]\n"

        rtos_name = args[0]
        host_path = args[1] if len(args) > 1 else None

        if not self.fs.exists(rtos_name):
            return f"export: {rtos_name}: No such file\n"

        if self.fs.export_to_host(rtos_name, host_path):
            path = host_path or rtos_name
            return f"Exported: {rtos_name} -> {path}\n"
        else:
            return f"export: Failed to export\n"

    def cmd_stat(self, args):
        """Show file metadata."""
        if not args:
            return "Usage: stat <file>\n"

        name = args[0]
        f = self.fs.get_file(name)
        if f is None:
            return f"stat: {name}: No such file\n"

        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.created_at))
        modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.modified_at))
        perms = self._format_perms(f.permissions)

        return f"""File: {name}
Size: {f.size} bytes
Permissions: {perms}
Created: {created}
Modified: {modified}
"""

    def cmd_df(self, args):
        """Show filesystem statistics."""
        stats = self.fs.get_stats()
        return f"""Filesystem: Neural RTOS
Files: {stats['total_files']}/{stats['max_files']}
Used: {stats['total_size']} bytes
Available: {stats['max_file_size'] * stats['max_files'] - stats['total_size']} bytes
Storage: {stats['storage_path']}
"""

    def cmd_echo(self, args):
        """Print text."""
        text = " ".join(args)
        return text + "\n"

    def cmd_calc(self, args):
        """Calculator."""
        if len(args) < 3:
            return "Usage: calc <a> <op> <b>\nOperators: + - * / & | ^\n"

        try:
            a = float(args[0])
            op = args[1]
            b = float(args[2])

            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            elif op == '*':
                result = a * b
            elif op == '/':
                result = a / b if b != 0 else "Error: Division by zero"
            elif op == '&':
                result = int(a) & int(b)
            elif op == '|':
                result = int(a) | int(b)
            elif op == '^':
                result = int(a) ^ int(b)
            else:
                return f"Unknown operator: {op}\n"

            return f"{a} {op} {b} = {result}\n"
        except ValueError:
            return "Error: Invalid numbers\n"

    def cmd_mem(self, args):
        """Memory info."""
        if self.cpu:
            return f"""Memory Map:
  Kernel:  0x10000 - 0x1FFFF
  Data:    0x20000 - 0x2FFFF
  FS:      0x30000 - 0x3FFFF
  FB:      0x40000 - 0x4FFFF
  Kbd:     0x50000 - 0x500FF
  User:    0x60000 - 0x6FFFF
  Stack:   0x70000 - 0x7FFFF
Instructions executed: {self.cpu.inst_count:,}
"""
        else:
            return "Memory info not available (no CPU attached)\n"

    def cmd_date(self, args):
        """Show date/time."""
        return time.strftime("%Y-%m-%d %H:%M:%S UTC\n") + "\n"

    def cmd_uname(self, args):
        """System information."""
        if args and '-a' in args:
            return f"""NeuralRTOS 1.0.0 NeuralCPU neural-rtos-gcc
Architecture: AArch64
CPU: Neural Transformer
Byte Order: Little Endian
"""
        else:
            return "NeuralRTOS 1.0.0\n"

    def cmd_hostname(self, args):
        """Show hostname."""
        return self.env.get("HOSTNAME", "neural-rtos") + "\n"

    def cmd_pwd(self, args):
        """Print working directory."""
        return self.env.get("PWD", "/root") + "\n"

    def cmd_env(self, args):
        """Show environment variables."""
        output = []
        for key, value in sorted(self.env.items()):
            output.append(f"{key}={value}")
        return "\n".join(output) + "\n"

    def cmd_clear(self, args):
        """Clear screen."""
        return "\033[2J\033[H"

    def cmd_reboot(self, args):
        """Reboot system."""
        return "Rebooting...\n"  # In real implementation, would trigger reboot

    def _format_perms(self, perms):
        """Format permissions like ls -l."""
        if isinstance(perms, int):
            p = perms
        else:
            p = 0o644

        modes = "rwxrwxrwx"
        result = ""
        for i in range(9):
            if p & (0x100 >> i):
                result += modes[i]
            else:
                result += "-"
        return result


# Convenience function
def execute_shell_command(cmd: str, cpu=None, fb=None) -> str:
    """Execute a shell command."""
    shell = RTOSShell(cpu, fb)
    return shell.execute(cmd)
