"""
Alpine Linux Root Filesystem — Build a complete Alpine v3.20 rootfs for GPU.

Creates a GPUFilesystem populated with the Alpine Linux v3.20 FHS directory
hierarchy, identity files, user databases, system configuration, and
synthetic /proc filesystem entries.

The resulting filesystem is served to a real BusyBox binary (Alpine's core
userspace) running on the Metal GPU compute shader via SVC-trapped syscalls.

Author: Robert Price
Date: March 2026
"""

from ncpu.os.gpu.filesystem import GPUFilesystem


def create_alpine_rootfs() -> GPUFilesystem:
    """Build a complete Alpine Linux v3.20 filesystem for GPU execution.

    Creates the standard Alpine FHS directory layout, populates identity
    files (/etc/os-release, /etc/alpine-release), user databases
    (/etc/passwd, /etc/group, /etc/shadow), system configuration, and
    synthetic /proc entries.

    Returns:
        GPUFilesystem populated with Alpine Linux v3.20 rootfs.
    """
    fs = GPUFilesystem()

    # ===================================================================
    # DIRECTORY HIERARCHY (Alpine FHS)
    # ===================================================================

    alpine_dirs = [
        "/bin", "/sbin",
        "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/share",
        "/etc", "/etc/init.d", "/etc/apk",
        "/home", "/root",
        "/tmp",
        "/var/log", "/var/run",
        "/proc", "/sys", "/dev",
        "/lib",
        "/run",
        "/mnt", "/media",
        "/opt", "/srv",
    ]

    for d in alpine_dirs:
        fs.directories.add(d)

    # ===================================================================
    # IDENTITY FILES
    # ===================================================================

    fs.write_file("/etc/alpine-release", "3.20.0\n")

    fs.write_file("/etc/os-release", (
        'NAME="Alpine Linux"\n'
        'ID=alpine\n'
        'VERSION_ID=3.20.0\n'
        'PRETTY_NAME="Alpine Linux v3.20 (nCPU GPU)"\n'
        'HOME_URL="https://alpinelinux.org/"\n'
        'BUG_REPORT_URL="https://gitlab.alpinelinux.org/alpine/aports/-/issues"\n'
    ))

    fs.write_file("/etc/hostname", "ncpu-gpu\n")

    fs.write_file("/etc/hosts", "127.0.0.1\tlocalhost ncpu-gpu\n")

    # ===================================================================
    # USER DATABASE
    # ===================================================================

    fs.write_file("/etc/passwd", (
        "root:x:0:0:root:/root:/bin/ash\n"
        "nobody:x:65534:65534:nobody:/:/sbin/nologin\n"
    ))

    fs.write_file("/etc/group", (
        "root:x:0:\n"
        "nogroup:x:65534:\n"
    ))

    fs.write_file("/etc/shadow", (
        "root:!::0:::::\n"
        "nobody:!::0:::::\n"
    ))

    fs.write_file("/etc/shells", (
        "/bin/ash\n"
        "/bin/sh\n"
    ))

    # ===================================================================
    # SYSTEM CONFIGURATION
    # ===================================================================

    fs.write_file("/etc/profile", (
        "export PATH=/usr/sbin:/usr/bin:/sbin:/bin\n"
        "export HOME=/root\n"
        "export PS1='\\u@\\h:\\w# '\n"
    ))

    fs.write_file("/etc/resolv.conf", "nameserver 127.0.0.1\n")

    fs.write_file("/etc/motd", (
        "\n"
        "Welcome to Alpine Linux v3.20 (nCPU GPU)\n"
        "Running on Apple Silicon Metal compute shader\n"
        "\n"
    ))

    # ===================================================================
    # PROC FILESYSTEM (synthetic)
    # ===================================================================

    fs.write_file("/proc/version",
        "Linux version 6.1.0-ncpu (gcc) #1 SMP aarch64 GNU/Linux\n"
    )

    fs.write_file("/proc/cpuinfo", (
        "processor\t: 0\n"
        "BogoMIPS\t: 48.00\n"
        "Features\t: fp asimd\n"
        "CPU implementer\t: 0x61\n"
        "CPU architecture: 8\n"
        "CPU variant\t: 0x1\n"
        "CPU part\t: 0x000\n"
        "CPU revision\t: 0\n"
        "\n"
        "Hardware\t: Apple Silicon (nCPU GPU)\n"
        "model name\t: Apple Silicon (nCPU GPU)\n"
    ))

    fs.write_file("/proc/meminfo", (
        "MemTotal:       16777216 kB\n"
        "MemFree:        16000000 kB\n"
        "MemAvailable:   15800000 kB\n"
        "Buffers:           65536 kB\n"
        "Cached:           524288 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n"
    ))

    fs.write_file("/proc/uptime", "3600.00 3600.00\n")

    fs.write_file("/proc/loadavg", "0.00 0.00 0.00 1/1 1\n")

    # ===================================================================
    # APK PACKAGE MANAGER
    # ===================================================================

    fs.write_file("/etc/apk/world", "busybox\nmusl\n")

    fs.write_file("/etc/apk/repositories",
        "https://dl-cdn.alpinelinux.org/alpine/v3.20/main\n"
    )

    # ===================================================================
    # ADDITIONAL SYSTEM FILES
    # ===================================================================

    fs.write_file("/etc/fstab", "# /etc/fstab: static file system information\n")

    fs.write_file("/etc/inittab", (
        "::sysinit:/sbin/openrc sysinit\n"
        "::sysinit:/sbin/openrc boot\n"
        "::wait:/sbin/openrc default\n"
        "tty1::respawn:/sbin/getty 38400 tty1\n"
    ))

    fs.write_file("/etc/issue", "Alpine Linux v3.20 (nCPU GPU) \\n \\l\n\n")

    return fs
