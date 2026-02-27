"""
Linux System Call Compatibility Layer
======================================
Implements Linux syscalls for Neural RTOS to enable running Linux binaries.

This is a partial implementation of common Linux syscalls that would allow
simple statically-linked Linux binaries to run on the Neural RTOS.

Supported syscalls:
- File operations: open, close, read, write, lseek, stat, fstat
- Memory: brk, mmap, munmap, mprotect
- Process: exit, getpid, getppid
- Signal: rt_sigaction, rt_sigprocmask
- Info: uname, getcwd, gettimeofday
- Filesystem: access, statfs, getdents
- Socket: (stubbed) socket, bind, listen, accept, connect, recv, send
- Misc: ioctl, fcntl

References:
- Linux syscall table: https://github.com/torvalds/linux/blob/master/arch/arm64/kernel/sys.c
- ARM64 syscall calling convention: x8=syscall#, x0-x5=args
"""

import ctypes
import ctypes.util
import time
import os
from typing import Dict, Callable, Optional, List
from rtos_filesystem import get_filesystem


class LinuxSyscallError(Exception):
    """Base exception for syscall errors."""
    pass


class EMulatedFileDescriptor:
    """Emulated file descriptor for Linux syscall compatibility."""

    FD_COUNTER = 3  # Start after stdin(0), stdout(1), stderr(2)

    def __init__(self, path: str, flags: int, mode: int = 0o644):
        self.fd = EMulatedFileDescriptor.FD_COUNTER
        EMulatedFileDescriptor.FD_COUNTER += 1
        self.path = path
        self.flags = flags
        self.mode = mode
        self.offset = 0
        self.fs = get_filesystem()

        # Read file content if exists
        self.content = self.fs.read(path)
        if self.content is None:
            if flags & 0x01:  # O_WRONLY or O_RDWR
                self.content = ""
            else:
                self.content = None  # File doesn't exist
        else:
            if flags & 0x100:  # O_TRUNC
                self.content = ""

    def read(self, size: int) -> bytes:
        """Read from file."""
        if self.content is None:
            return b""

        data = self.content[self.offset:self.offset + size]
        self.offset += len(data)
        return data.encode() if isinstance(data, str) else data

    def write(self, data: bytes) -> int:
        """Write to file."""
        if self.content is None:
            self.content = ""

        content = data.decode() if isinstance(data, bytes) else data
        if self.offset == 0:
            new_content = content
        else:
            new_content = self.content[:self.offset] + content

        if self.offset + len(content) > len(self.content):
            new_content += self.content[self.offset + len(content):]

        self.fs.write(self.path, new_content)
        self.content = new_content
        self.offset += len(content)
        return len(content)

    def seek(self, offset: int, whence: int) -> int:
        """Seek in file."""
        if whence == 0:  # SEEK_SET
            self.offset = offset
        elif whence == 1:  # SEEK_CUR
            self.offset += offset
        elif whence == 2:  # SEEK_END
            self.offset = len(self.content) + offset if self.content else offset
        return self.offset

    def close(self) -> int:
        """Close file descriptor."""
        return 0


class LinuxSyscallEmulator:
    """
    Emulates Linux system calls for ARM64 binaries.

    ARM64 Syscall Calling Convention:
    - x8: syscall number
    - x0-x5: arguments (0-5)
    - x0: return value
    """

    # ARM64 syscall numbers for aarch64
    # Source: https://github.com/torvalds/linux/blob/master/arch/arm64/include/asm/unistd.h
    SYSCALLS = {
        # File operations
        56: "io_setup",
        57: "io_destroy",
        58: "io_submit",
        59: "io_cancel",
        60: "io_getevents",
        61: "setxattr",
        62: "lsetxattr",
        63: "fsetxattr",
        64: "getxattr",
        65: "lgetxattr",
        66: "fgetxattr",
        67: "listxattr",
        68: "llistxattr",
        69: "flistxattr",
        70: "removexattr",
        71: "lremovexattr",
        72: "fremovexattr",
        79: "getcwd",
        80: "lookup_dcookie",
        290: "eventfd2",
        291: "epoll_create1",
        292: "epoll_ctl",
        293: "epoll_pwait",
        23: "dup",
        24: "dup3",
        25: "fcntl",
        26: "inotify_init1",
        27: "inotify_add_watch",
        28: "inotify_rm_watch",
        29: "ioctl",
        30: "ioprio_set",
        31: "ioprio_get",
        32: "flock",
        33: "mknodat",
        34: "mkdirat",
        35: "unlinkat",
        36: "symlinkat",
        37: "linkat",
        38: "renameat",
        39: "umount2",
        40: "mount",
        41: "pivot_root",
        42: "nfsservctl",
        43: "statfs",
        44: "fstatfs",
        45: "truncate",
        46: "ftruncate",
        47: "fallocate",
        48: "faccessat",
        49: "chdir",
        50: "fchdir",
        51: "chroot",
        52: "fchmod",
        53: "fchmodat",
        54: "fchownat",
        55: "fchown",
        56: "openat",
        57: "close",
        58: "vhangup",
        59: "pipe2",
        60: "quota",
        61: "getdents64",
        62: "lseek",
        63: "read",
        64: "write",
        65: "readv",
        66: "writev",
        67: "pread64",
        68: "pwrite64",
        69: "preadv",
        70: "pwritev",
        # Memory management
        214: "brk",
        222: "mmap",
        215: "munmap",
        226: "mprotect",
        216: "mremap",
        227: "msync",
        139: "rt_sigreturn",
        128: "rt_sigaction",
        129: "rt_sigprocmask",
        130: "rt_sigpending",
        131: "rt_sigtimedwait",
        132: "rt_sigqueueinfo",
        133: "sigaltstack",
        # Process
        135: "rt_sigsuspend",
        276: "renameat2",
        172: "getpid",
        173: "getppid",
        93: "exit",
        94: "exit_group",
        174: "getuid",
        175: "getgid",
        176: "setuid",
        177: "geteuid",
        178: "getegid",
        # Info
        160: "uname",
        169: "gettimeofday",
        163: "getrlimit",
        164: "setrlimit",
        # Socket
        198: "socket",
        200: "bind",
        203: "connect",
        201: "listen",
        202: "accept",
        204: "getsockname",
        205: "getpeername",
        206: "socketpair",
        211: "send",
        212: "recv",
        206: "sendto",
        207: "recvfrom",
        208: "shutdown",
        209: "setsockopt",
        210: "getsockopt",
        211: "sendmsg",
        212: "recvmsg",
    }

    def __init__(self):
        self.fs = get_filesystem()
        self.open_fds: Dict[int, EMulatedFileDescriptor] = {}
        self.pid = 1000
        self.ppid = 1
        self.brk = 0
        self.hostname = "neural-rtos"
        self.domainname = "(none)"

        # Initialize standard FDs
        self.open_fds[0] = None  # stdin
        self.open_fds[1] = None  # stdout
        self.open_fds[2] = None  # stderr

    def handle_syscall(self, syscall_num: int, args: List[int]) -> int:
        """
        Handle a Linux system call.

        Args:
            syscall_num: ARM64 syscall number (from x8)
            args: List of arguments (x0-x5)

        Returns:
            Return value (goes in x0)
        """
        syscall_name = self.SYSCALLS.get(syscall_num, f"unknown_{syscall_num:x}")

        handler = getattr(self, f"syscall_{syscall_name}", None)
        if handler:
            try:
                return handler(args)
            except LinuxSyscallError as e:
                return -e.args[0] if e.args else -1
            except Exception as e:
                print(f"Syscall {syscall_name} error: {e}")
                return -1
        else:
            print(f"Unimplemented syscall: {syscall_name} ({syscall_num:#x})")
            return -38  # ENOSYS

    # File operation syscalls

    def syscall_openat(self, args):
        """int openat(int dirfd, const char *pathname, int flags, mode_t mode)"""
        dirfd, pathname_ptr, flags, mode = args[0], args[1], args[2], args[3]
        # Simplified: ignore dirfd, assume pathname is in memory
        # In real implementation, would read string from memory at pathname_ptr
        return self._open(pathname_ptr, flags, mode)

    def _open(self, path_ptr: int, flags: int, mode: int) -> int:
        """Helper to open file."""
        # In real implementation, would read path from memory
        # For now, return a dummy FD
        fd = EMulatedFileDescriptor.FD_COUNTER
        EMulatedFileDescriptor.FD_COUNTER += 1
        return fd

    def syscall_close(self, args):
        """int close(int fd)"""
        fd = args[0]
        if fd in self.open_fds:
            em_fd = self.open_fds[fd]
            if em_fd:
                em_fd.close()
            del self.open_fds[fd]
            return 0
        return -9  # EBADF

    def syscall_read(self, args):
        """ssize_t read(int fd, void *buf, size_t count)"""
        fd, buf, count = args[0], args[1], args[2]

        if fd == 0:  # stdin
            return 0  # No input available
        elif fd in self.open_fds:
            em_fd = self.open_fds[fd]
            if em_fd:
                data = em_fd.read(count)
                # In real implementation, would write data to memory at buf
                return len(data)
        return -9  # EBADF

    def syscall_write(self, args):
        """ssize_t write(int fd, const void *buf, size_t count)"""
        fd, buf, count = args[0], args[1], args[2]

        if fd == 1 or fd == 2:  # stdout or stderr
            # In real implementation, would read data from memory at buf
            # and print it
            return count  # Pretend we wrote everything
        elif fd in self.open_fds:
            em_fd = self.open_fds[fd]
            if em_fd:
                # In real implementation, would read data from memory at buf
                return count  # Pretend we wrote everything
        return -9  # EBADF

    def syscall_lseek(self, args):
        """off_t lseek(int fd, off_t offset, int whence)"""
        fd, offset, whence = args[0], args[1], args[2]

        if fd in self.open_fds and self.open_fds[fd]:
            return self.open_fds[fd].seek(offset, whence)
        return -9  # EBADF

    # Memory management syscalls

    def syscall_brk(self, args):
        """void *brk(void *addr)"""
        addr = args[0]
        if addr == 0:
            return self.brk if self.brk else 0x100000  # Return current or default
        elif addr > self.brk:
            self.brk = addr
        return self.brk

    def syscall_mmap(self, args):
        """void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)"""
        addr, length, prot, flags, fd, offset = args[0], args[1], args[2], args[3], args[4], args[5]

        # Simplified: just return an address
        # In real implementation, would manage virtual memory
        return 0x8000000

    def syscall_munmap(self, args):
        """int munmap(void *addr, size_t length)"""
        # Stub: always succeed
        return 0

    def syscall_mprotect(self, args):
        """int mprotect(void *addr, size_t len, int prot)"""
        # Stub: always succeed
        return 0

    # Process syscalls

    def syscall_getpid(self, args):
        """pid_t getpid(void)"""
        return self.pid

    def syscall_getppid(self, args):
        """pid_t getppid(void)"""
        return self.ppid

    def syscall_exit(self, args):
        """void _exit(int status)"""
        status = args[0] if args else 0
        raise SystemExit(status)

    def syscall_exit_group(self, args):
        """void exit_group(int status)"""
        status = args[0] if args else 0
        raise SystemExit(status)

    # Signal syscalls (stubs)

    def syscall_rt_sigaction(self, args):
        """int rt_sigaction(int signum, const struct sigaction *act, struct sigaction *oldact, size_t sigsetsize)"""
        return 0  # Stub: always succeed

    def syscall_rt_sigprocmask(self, args):
        """int rt_sigprocmask(int how, const sigset_t *set, sigset_t *oldset, size_t sigsetsize)"""
        return 0  # Stub: always succeed

    # Info syscalls

    def syscall_uname(self, args):
        """int uname(struct utsname *buf)"""
        # In real implementation, would write utsname struct to memory at buf
        return 0

    def syscall_getcwd(self, args):
        """char *getcwd(char *buf, size_t size)"""
        # In real implementation, would write cwd to memory
        return args[0]  # Return buf pointer

    def syscall_gettimeofday(self, args):
        """int gettimeofday(struct timeval *tv, struct timezone *tz)"""
        # In real implementation, would write time to memory
        return 0

    # Filesystem syscalls

    def syscall_statfs(self, args):
        """int statfs(const char *path, struct statfs *buf)"""
        return 0  # Stub

    def syscall_fstatfs(self, args):
        """int fstatfs(int fd, struct statfs *buf)"""
        return 0  # Stub

    def syscall_faccessat(self, args):
        """int faccessat(int dirfd, const char *pathname, int mode, int flags)"""
        return 0  # Stub: always accessible

    def syscall_getdents64(self, args):
        """ssize_t getdents64(int fd, struct linux_dirent64 *dirp, size_t count)"""
        return 0  # Stub: no entries

    # Socket syscalls (stubs)

    def syscall_socket(self, args):
        """int socket(int domain, int type, int protocol)"""
        return -97  # AF_NOINET

    def syscall_bind(self, args):
        """int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)"""
        return -97  # AF_NOINET

    def syscall_connect(self, args):
        """int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)"""
        return -97  # AF_NOINET

    def syscall_listen(self, args):
        """int listen(int sockfd, int backlog)"""
        return -97  # AF_NOINET

    def syscall_accept(self, args):
        """int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)"""
        return -97  # AF_NOINET

    def syscall_recv(self, args):
        """ssize_t recv(int sockfd, void *buf, size_t len, int flags)"""
        return -97  # AF_NOINET

    def syscall_send(self, args):
        """ssize_t send(int sockfd, const void *buf, size_t len, int flags)"""
        return -97  # AF_NOINET

    # Other syscalls

    def syscall_ioctl(self, args):
        """int ioctl(int fd, unsigned long request, ...)"""
        return -25  # ENOTTY

    def syscall_fcntl(self, args):
        """int fcntl(int fd, int cmd, ... /* arg */ )"""
        return 0  # Stub


# Global syscall emulator instance
_syscall_emulator = None

def get_syscall_emulator() -> LinuxSyscallEmulator:
    """Get or create the global syscall emulator instance."""
    global _syscall_emulator
    if _syscall_emulator is None:
        _syscall_emulator = LinuxSyscallEmulator()
    return _syscall_emulator


def handle_svc(syscall_num: int, args: List[int]) -> int:
    """
    Handle an ARM64 SVC (supervisor call) instruction.

    This is called when the neural CPU executes an SVC instruction,
    which is how ARM64 programs make system calls.

    Args:
        syscall_num: Syscall number from x8 register
        args: List of arguments from x0-x5 registers

    Returns:
        Return value to put in x0 register
    """
    emulator = get_syscall_emulator()
    return emulator.handle_syscall(syscall_num, args)


# Error codes (from errno-base.h)
class Errno:
    EPERM = 1      # Operation not permitted
    ENOENT = 2     # No such file or directory
    EBADF = 9      # Bad file number
    EAGAIN = 11    # Try again
    ENOMEM = 12    # Out of memory
    EACCES = 13    # Permission denied
    EFAULT = 14    # Bad address
    EBUSY = 16     # Device or resource busy
    EEXIST = 17    # File exists
    EINVAL = 22    # Invalid argument
    ENOSYS = 38    # Function not implemented
    ENOTTY = 25    # Not a typewriter
    AF_NOINET = 97 # Address family not supported
