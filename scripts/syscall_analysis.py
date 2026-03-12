#!/usr/bin/env python3
"""
Syscall coverage analysis - check which Linux syscalls are implemented.
"""

# Common Linux syscalls by category
SYSCALLS = {
    # File operations
    "openat": 56,
    "close": 57,
    "read": 63,
    "write": 64,
    "lseek": 62,
    "fstat": 80,
    "fstatat": 79,
    "dup": 23,
    "dup3": 24,
    "pipe2": 59,

    # File system
    "mkdirat": 34,
    "unlinkat": 35,
    "symlinkat": 36,
    "linkat": 37,
    "renameat": 38,
    "getdents64": 61,
    "readlinkat": 78,
    "faccessat": 48,
    "chdir": 49,
    "getcwd": 17,

    # Memory
    "brk": 214,
    "mmap": 222,
    "munmap": 215,
    "mprotect": 226,

    # Process
    "fork": 220,
    "vfork": 221,
    "execve": 221,  # Note: execve is 221 but handled specially
    "exit": 93,
    "wait4": 260,
    "getpid": 172,
    "getppid": 173,
    "getuid": 174,
    "getgid": 175,
    "setuid": 146,
    "setgid": 147,
    "gettid": 178,
    "prctl": 157,

    # Signals
    "kill": 62,
    "sigaction": 67,
    "sigprocmask": 68,
    "sigreturn": 139,

    # Time
    "clock_gettime": 113,
    "clock_getres": 114,
    "nanosleep": 101,
    "gettimeofday": 169,

    # Socket
    "socket": 198,
    "bind": 200,
    "listen": 201,
    "accept": 202,
    "connect": 203,
    "sendto": 205,
    "recvfrom": 206,
    "sendmsg": 211,
    "recvmsg": 212,
    "setsockopt": 208,
    "getsockopt": 209,
    "shutdown": 210,

    # IO
    "ioctl": 29,
    "fcntl": 25,
    "flock": 73,

    # Info
    "uname": 160,
    "sysinfo": 179,
    "getpriority": 141,
    "setpriority": 140,
    "capget": 90,
    "capset": 91,

    # Advanced
    "perf_event_open": 241,
    "bpf": 280,
}

# Check which syscalls have handlers in the Rust launcher
IMPLEMENTED = [
    # File operations
    56, 57, 63, 64, 62, 80, 79, 23, 24, 59,  # File ops (openat, close, read, write, lseek, fstat, etc.)
    25, 29,  # fcntl, ioctl
    # File system
    34, 35, 36, 37, 38, 61, 78, 48, 49, 17,  # mkdirat, unlinkat, symlinkat, renameat, etc.
    73, 74, 75, 76, 77,  # flock, fsync, fdatasync, truncate, ftruncate
    83, 84, 85, 86, 87,  # mkdir, rmdir, creat, link, unlink
    132, 133,  # utime, mknod
    245,  # mkfifo
    269, 276,  # futimesat, renameat2
    # Memory
    214, 222, 215, 226, 216, 227, 233,  # brk, mmap, munmap, mprotect, mremap, msync, madvise
    # Process
    220, 93, 94, 260, 172, 173, 178,  # fork, exit, exit_group, wait4, getpid, getppid, gettid
    146, 147, 149, 150, 120, 121,  # setuid, setgid, setreuid, setregid, setresuid, setresgid
    115, 113,  # getgroups, setgroups
    123, 132, 148, 149,  # setpgid, getpgid, setsid, getsid
    136, 135,  # setreuid, setregid
    96, 99,  # set_tid_address, set_robust_list
    221,  # vfork
    # Identity
    174, 175, 102, 176, 207, 208, 104,  # getuid, geteuid, getuid32, getegid, etc.
    163,  # getppid (also at 112)
    # Signal
    134, 135, 139,  # rt_sigaction, rt_sigprocmask, rt_sigreturn
    106, 107,  # pause, alarm
    102, 103,  # getitimer, setitimer
    131,  # sigaltstack
    136, 137, 138,  # rt_sigpending, rt_sigtimedwait, rt_sigqueueinfo
    # Time
    160,  # uname
    113, 114, 112, 115,  # clock_gettime, clock_getres, clock_settime, clock_nanosleep
    101, 169,  # nanosleep, gettimeofday
    # Socket/network
    198, 200, 201, 202, 203, 205, 206, 209, 210, 211, 212, 232,  # socket, bind, listen, accept, connect, getsockopt, etc.
    # Resource
    163, 165, 151,  # getrlimit, getrusage, umask
    140, 141,  # getpriority, setpriority
    # System
    179,  # sysinfo
    170, 171,  # gethostname, sethostname
    161, 166, 162,  # chroot, umount, mount
    232,  # socketpair
    213, 214, 215,  # epoll_create, epoll_ctl, epoll_wait
    # IPC
    194, 195, 196, 197,  # shmget, shmctl, shmat, shmdt
    217, 218, 219, 220,  # semget, semctl, semop, semtimedop
    218, 219, 220, 221,  # msgget, msgsnd, msgrcv, msgctl
    # Capabilities & Performance
    90, 91,  # capget, capset
    241,  # perf_event_open
    280,  # bpf
]

# Missing that we might need
MISSING = [x for x in SYSCALLS.values() if x not in IMPLEMENTED]

print("=== Syscall Coverage ===")
print(f"Total common syscalls tracked: {len(SYSCALLS)}")
print(f"Implemented: {len(IMPLEMENTED)}")
print(f"Missing: {len(MISSING)}")

print("\n=== Missing Syscalls ===")
for name, num in sorted(SYSCALLS.items(), key=lambda x: x[1]):
    if num in MISSING:
        print(f"  {name}: {num}")

# Critical missing ones for Linux compatibility
CRITICAL = ["ioctl", "fcntl", "prctl", "sigaction", "sigprocmask", "setuid", "setgid",
            "sysinfo", "gettimeofday", "setsockopt", "getsockopt", "nanosleep"]
print("\n=== Critical Missing ===")
for name in CRITICAL:
    if name in SYSCALLS:
        print(f"  {name}: {SYSCALLS[name]}")
