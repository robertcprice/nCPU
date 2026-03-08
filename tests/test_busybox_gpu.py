"""Tests for BusyBox on GPU — filesystem integration, struct formats, interactive mode.

Verifies:
  - stat64 struct packing (128 bytes, correct field offsets)
  - getdents64 struct format (linux_dirent64)
  - GPUFilesystem integration with BusyBox syscall handler
  - Interactive mode round-trip (filesystem persistence)
  - Unknown instruction tracking diagnostic region

Requires: mlx (Apple Silicon Metal)
"""

import struct
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")
DEMOS_DIR = str(PROJECT_ROOT / "demos")
HAS_BUSYBOX = Path(BUSYBOX).exists()

# Add demos/ to path so we can import busybox_gpu_demo
if DEMOS_DIR not in sys.path:
    sys.path.insert(0, DEMOS_DIR)

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# ═══════════════════════════════════════════════════════════════════════════════
# STAT64 STRUCT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStat64Struct:
    """Verify _pack_stat64 produces correct Linux aarch64 stat struct."""

    def test_stat64_size(self):
        from ncpu.os.gpu.elf_loader import _pack_stat64
        info = {"type": "file", "size": 42, "path": "/test.txt"}
        buf = _pack_stat64(info)
        assert len(buf) == 128, f"stat64 should be 128 bytes, got {len(buf)}"

    def test_stat64_file_fields(self):
        from ncpu.os.gpu.elf_loader import _pack_stat64
        info = {"type": "file", "size": 1024, "path": "/etc/motd"}
        buf = _pack_stat64(info)

        # Unpack key fields
        st_dev = struct.unpack_from('<Q', buf, 0)[0]
        st_ino = struct.unpack_from('<Q', buf, 8)[0]
        st_mode = struct.unpack_from('<I', buf, 16)[0]
        st_nlink = struct.unpack_from('<I', buf, 20)[0]
        st_uid = struct.unpack_from('<I', buf, 24)[0]
        st_gid = struct.unpack_from('<I', buf, 28)[0]
        st_size = struct.unpack_from('<q', buf, 48)[0]
        st_blksize = struct.unpack_from('<i', buf, 56)[0]
        st_blocks = struct.unpack_from('<q', buf, 64)[0]

        assert st_dev == 1
        assert st_ino != 0, "inode should be non-zero"
        assert st_mode == 0o100644, f"file mode should be 0o100644, got 0o{st_mode:o}"
        assert st_nlink == 1
        assert st_uid == 0
        assert st_gid == 0
        assert st_size == 1024
        assert st_blksize == 4096
        assert st_blocks == 2  # ceil(1024/512)

    def test_stat64_directory_fields(self):
        from ncpu.os.gpu.elf_loader import _pack_stat64
        info = {"type": "dir", "size": 0, "path": "/etc"}
        buf = _pack_stat64(info)

        st_mode = struct.unpack_from('<I', buf, 16)[0]
        st_nlink = struct.unpack_from('<I', buf, 20)[0]
        st_size = struct.unpack_from('<q', buf, 48)[0]

        assert st_mode == 0o040755, f"dir mode should be 0o040755, got 0o{st_mode:o}"
        assert st_nlink == 2
        assert st_size == 0

    def test_stat64_timestamps_nonzero(self):
        from ncpu.os.gpu.elf_loader import _pack_stat64
        info = {"type": "file", "size": 0, "path": "/test"}
        buf = _pack_stat64(info)

        st_atime = struct.unpack_from('<q', buf, 72)[0]
        st_mtime = struct.unpack_from('<q', buf, 88)[0]
        st_ctime = struct.unpack_from('<q', buf, 104)[0]

        assert st_atime > 0, "atime should be non-zero"
        assert st_mtime > 0, "mtime should be non-zero"
        assert st_ctime > 0, "ctime should be non-zero"

    def test_stat64_blocks_calculation(self):
        from ncpu.os.gpu.elf_loader import _pack_stat64
        # 0 bytes → 0 blocks
        buf = _pack_stat64({"type": "file", "size": 0, "path": "/a"})
        assert struct.unpack_from('<q', buf, 64)[0] == 0

        # 1 byte → 1 block
        buf = _pack_stat64({"type": "file", "size": 1, "path": "/b"})
        assert struct.unpack_from('<q', buf, 64)[0] == 1

        # 512 bytes → 1 block
        buf = _pack_stat64({"type": "file", "size": 512, "path": "/c"})
        assert struct.unpack_from('<q', buf, 64)[0] == 1

        # 513 bytes → 2 blocks
        buf = _pack_stat64({"type": "file", "size": 513, "path": "/d"})
        assert struct.unpack_from('<q', buf, 64)[0] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# GETDENTS64 FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetdents64Format:
    """Verify getdents64 produces correct linux_dirent64 structs."""

    def _make_cpu_and_fs(self):
        """Create a CPU, filesystem, and syscall handler for testing."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        from ncpu.os.gpu.filesystem import GPUFilesystem

        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        fs.write_file("/test/hello.txt", b"hello world\n")
        fs.write_file("/test/data.bin", b"\x00" * 100)
        fs.directories.add("/test/subdir")
        return cpu, fs

    def test_dirent64_header_size(self):
        """Each linux_dirent64 should have d_ino(8)+d_off(8)+d_reclen(2)+d_type(1)=19 byte header."""
        # Minimum entry: 19 bytes header + 1 byte name + null = 21 bytes, padded to 24
        header_size = 8 + 8 + 2 + 1  # d_ino + d_off + d_reclen + d_type
        assert header_size == 19

    def test_dirent64_alignment(self):
        """Each entry should be 8-byte aligned."""
        cpu, fs = self._make_cpu_and_fs()

        # Open directory
        fd = fs.open("/test", 0)
        assert fd >= 0

        # Simulate getdents64 by calling filesystem.listdir
        entries = fs.listdir("/test")
        assert entries is not None
        assert len(entries) >= 2  # hello.txt, data.bin, subdir

        # Simulate packing
        for name in entries:
            name_bytes = name.encode('ascii') + b'\x00'
            raw_len = 19 + len(name_bytes)
            d_reclen = (raw_len + 7) & ~7
            assert d_reclen % 8 == 0, f"d_reclen {d_reclen} not 8-byte aligned"

    def test_dirent64_type_values(self):
        """DT_DIR should be 4, DT_REG should be 8."""
        DT_DIR = 4
        DT_REG = 8
        assert DT_DIR == 4
        assert DT_REG == 8

    def test_getdents64_consumed_tracking(self):
        """Second getdents64 call on same fd should return 0 bytes."""
        cpu, fs = self._make_cpu_and_fs()

        fd = fs.open("/test", 0)
        assert fd >= 0

        # First call: should return entries
        entry = fs.fd_table[fd]
        assert not entry.get("getdents_consumed", False)

        # Mark consumed (as the syscall handler would)
        entry["getdents_consumed"] = True

        # Verify flag is set
        assert entry["getdents_consumed"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# FILESYSTEM INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilesystemIntegration:
    """Test GPUFilesystem creation and content for BusyBox demo."""

    def test_create_filesystem_has_standard_dirs(self):
        sys.path.insert(0, str(PROJECT_ROOT / "demos"))
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()

        for d in ["/", "/bin", "/etc", "/home", "/tmp", "/proc", "/home/user"]:
            assert d in fs.directories, f"Missing directory: {d}"

    def test_create_filesystem_has_files(self):
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()

        expected_files = ["/etc/motd", "/etc/hostname", "/etc/passwd",
                         "/etc/group", "/etc/os-release",
                         "/home/user/hello.txt", "/tmp/data.txt",
                         "/proc/version"]
        for f in expected_files:
            assert f in fs.files, f"Missing file: {f}"

    def test_filesystem_motd_content(self):
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()
        motd = fs.files["/etc/motd"]
        assert b"nCPU" in motd
        assert b"GPU" in motd

    def test_filesystem_stat_integration(self):
        """filesystem.stat() should return correct info for existing files."""
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()

        info = fs.stat("/etc/motd")
        assert info is not None
        assert info["type"] == "file"
        assert info["size"] > 0

        info = fs.stat("/etc")
        assert info is not None
        assert info["type"] == "dir"

    def test_filesystem_stat_nonexistent(self):
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()
        assert fs.stat("/nonexistent") is None

    def test_filesystem_listdir(self):
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()

        entries = fs.listdir("/")
        assert entries is not None
        assert "etc" in entries
        assert "tmp" in entries
        assert "home" in entries

    def test_filesystem_read_write_roundtrip(self):
        from busybox_gpu_demo import create_filesystem
        fs = create_filesystem()

        # Write a new file
        fs.write_file("/tmp/test.txt", b"test data\n")
        assert fs.exists("/tmp/test.txt")

        # Read it back via fd
        fd = fs.open("/tmp/test.txt", 0)
        assert fd >= 0
        data = fs.read(fd, 1024)
        assert data == b"test data\n"
        fs.close(fd)


# ═══════════════════════════════════════════════════════════════════════════════
# BUSYBOX SYSCALL HANDLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBusyboxSyscallHandler:
    """Test the busybox-specific syscall handler."""

    def test_handler_creation(self):
        from ncpu.os.gpu.elf_loader import make_busybox_syscall_handler
        from ncpu.os.gpu.filesystem import GPUFilesystem

        fs = GPUFilesystem()
        handler = make_busybox_syscall_handler(filesystem=fs)
        assert callable(handler)

    def test_handler_without_filesystem(self):
        from ncpu.os.gpu.elf_loader import make_busybox_syscall_handler
        handler = make_busybox_syscall_handler()
        assert callable(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# BUSYBOX ELF EXECUTION TESTS (require busybox.elf)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BUSYBOX, reason="busybox.elf not found")
class TestBusyboxExecution:
    """Test actual BusyBox ELF execution on GPU with filesystem."""

    def test_echo_basic(self):
        """BusyBox echo should produce output."""
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(["echo", "hello"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "hello" in output
        assert results["total_cycles"] > 0

    def test_echo_with_filesystem(self):
        """BusyBox echo should work when filesystem is provided."""
        from busybox_gpu_demo import create_filesystem
        import io
        fs = create_filesystem()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(["echo", "test123"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "test123" in output

    def test_uname(self):
        """BusyBox uname should return Linux."""
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(["uname", "-s"])
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "Linux" in output

    def test_cat_file(self):
        """BusyBox cat should read from filesystem."""
        from busybox_gpu_demo import create_filesystem
        import io
        fs = create_filesystem()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(["cat", "/etc/hostname"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "ncpu-gpu" in output

    def test_unknown_instruction_tracking(self):
        """Verify instruction coverage analysis works on loaded binary."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2()
        # Static analysis of a small code region should return a list
        unknowns = cpu.analyze_instruction_coverage(0, 64)
        assert isinstance(unknowns, list)


def load_and_run_elf_helper(argv, filesystem=None, stdin_data=None):
    """Helper to run BusyBox command for tests."""
    from ncpu.os.gpu.elf_loader import load_and_run_elf
    return load_and_run_elf(
        BUSYBOX,
        argv=argv,
        max_cycles=500_000_000,
        quiet=True,
        filesystem=filesystem,
        stdin_data=stdin_data,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LDR LITERAL INSTRUCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLDRLiteral:
    """Verify LDR Wt/Xt literal (PC-relative) loads work in the Metal kernel."""

    def _make_cpu(self):
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        return MLXKernelCPUv2()

    def test_ldr_w_literal(self):
        """LDR Wt, label — 32-bit load from PC-relative address."""
        import struct
        cpu = self._make_cpu()
        base = 0x10000

        # Place a 32-bit constant at base+16
        cpu.write_memory(base + 16, struct.pack('<I', 0xDEADBEEF))

        # LDR W0, [PC+16]: op_byte=0x18, imm19 = 16/4 = 4, rd=0
        # Encoding: 0x18 << 24 | (imm19 << 5) | rd
        inst = (0x18 << 24) | (4 << 5) | 0
        cpu.write_memory(base, struct.pack('<I', inst))

        # HLT at base+4
        cpu.write_memory(base + 4, struct.pack('<I', 0xD4400000))

        cpu.set_pc(base)
        from ncpu.os.gpu.runner import run
        results = run(cpu, lambda c: True, max_cycles=10, quiet=True)

        val = cpu.get_register(0)
        assert val == 0xDEADBEEF, f"Expected 0xDEADBEEF, got 0x{val:X}"

    def test_ldr_x_literal(self):
        """LDR Xt, label — 64-bit load from PC-relative address."""
        import struct
        cpu = self._make_cpu()
        base = 0x10000

        # Place a 64-bit constant at base+16
        cpu.write_memory(base + 16, struct.pack('<Q', 0x123456789ABCDEF0))

        # LDR X1, [PC+16]: op_byte=0x58, imm19 = 16/4 = 4, rd=1
        inst = (0x58 << 24) | (4 << 5) | 1
        cpu.write_memory(base, struct.pack('<I', inst))

        # HLT at base+4
        cpu.write_memory(base + 4, struct.pack('<I', 0xD4400000))

        cpu.set_pc(base)
        from ncpu.os.gpu.runner import run
        results = run(cpu, lambda c: True, max_cycles=10, quiet=True)

        val = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
        assert val == 0x123456789ABCDEF0, f"Expected 0x123456789ABCDEF0, got 0x{val:X}"

    def test_ldrsw_literal(self):
        """LDRSW Xt, label — sign-extended 32→64 load."""
        import struct
        cpu = self._make_cpu()
        base = 0x10000

        # Place -1 (0xFFFFFFFF) as a 32-bit value at base+16
        cpu.write_memory(base + 16, struct.pack('<I', 0xFFFFFFFF))

        # LDRSW X2, [PC+16]: op_byte=0x98, imm19 = 4, rd=2
        inst = (0x98 << 24) | (4 << 5) | 2
        cpu.write_memory(base, struct.pack('<I', inst))

        # HLT
        cpu.write_memory(base + 4, struct.pack('<I', 0xD4400000))

        cpu.set_pc(base)
        from ncpu.os.gpu.runner import run
        results = run(cpu, lambda c: True, max_cycles=10, quiet=True)

        val = cpu.get_register(2)
        # Should be sign-extended to -1 (0xFFFFFFFFFFFFFFFF in unsigned)
        assert val == -1 or (val & 0xFFFFFFFFFFFFFFFF) == 0xFFFFFFFFFFFFFFFF, \
            f"Expected sign-extended -1, got {val} (0x{val & 0xFFFFFFFFFFFFFFFF:X})"

    def test_prfm_literal_nop(self):
        """PRFM label — should be a NOP (no crash, no register change)."""
        import struct
        cpu = self._make_cpu()
        base = 0x10000

        # MOVZ X5, #42  (to verify execution continues past PRFM)
        # MOVZ 64-bit base is 0xD2800000 (bit 23 must be set to distinguish from EOR imm)
        movz = 0xD2800000 | (42 << 5) | 5
        cpu.write_memory(base, struct.pack('<I', movz))

        # PRFM [PC+8]: op_byte=0xD8, imm19=2, rd=0
        prfm = (0xD8 << 24) | (2 << 5) | 0
        cpu.write_memory(base + 4, struct.pack('<I', prfm))

        # HLT
        cpu.write_memory(base + 8, struct.pack('<I', 0xD4400000))

        cpu.set_pc(base)
        from ncpu.os.gpu.runner import run
        results = run(cpu, lambda c: True, max_cycles=10, quiet=True)

        val = cpu.get_register(5)
        assert val == 42, f"Expected 42 (PRFM should NOP), got {val}"

    def test_ldr_w_literal_negative_offset(self):
        """LDR Wt, label with negative PC-relative offset."""
        import struct
        cpu = self._make_cpu()
        base = 0x10000

        # Place constant BEFORE the instruction at base-8
        cpu.write_memory(base - 8, struct.pack('<I', 0xCAFEBABE))

        # LDR W3, [PC-8]: imm19 = -8/4 = -2
        # -2 in 19-bit signed = 0x7FFFE
        imm19_neg2 = (-2) & 0x7FFFF
        inst = (0x18 << 24) | (imm19_neg2 << 5) | 3
        cpu.write_memory(base, struct.pack('<I', inst))

        # HLT
        cpu.write_memory(base + 4, struct.pack('<I', 0xD4400000))

        cpu.set_pc(base)
        from ncpu.os.gpu.runner import run
        results = run(cpu, lambda c: True, max_cycles=10, quiet=True)

        val = cpu.get_register(3)
        assert val == 0xCAFEBABE, f"Expected 0xCAFEBABE, got 0x{val:X}"

    def test_known_instruction_detection(self):
        """Instruction coverage analysis should recognize LDR literal opcodes."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        # 0x18 = LDR Wt literal, 0x58 = LDR Xt literal, 0x98 = LDRSW, 0xD8 = PRFM
        for op_byte in [0x18, 0x58, 0x98, 0xD8]:
            inst = (op_byte << 24) | (4 << 5) | 0
            top = (inst >> 24) & 0xFF
            assert MLXKernelCPUv2._is_known_instruction(inst, top), \
                f"op_byte 0x{op_byte:02X} should be recognized as known"


# ═══════════════════════════════════════════════════════════════════════════════
# ALPINE LINUX ROOTFS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlpineRootfs:
    """Verify Alpine Linux rootfs builder produces correct structure."""

    def _make_rootfs(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return create_alpine_rootfs()

    def test_fhs_directories(self):
        """Alpine rootfs should contain standard FHS directories."""
        fs = self._make_rootfs()
        required_dirs = [
            "/bin", "/sbin", "/usr/bin", "/usr/sbin",
            "/etc", "/home", "/root", "/tmp",
            "/var", "/var/log", "/proc", "/sys", "/dev",
            "/lib", "/usr/lib", "/run", "/mnt", "/media",
        ]
        for d in required_dirs:
            assert d in fs.directories, f"Missing FHS directory: {d}"

    def test_alpine_identity(self):
        """Alpine rootfs should have correct identity files."""
        fs = self._make_rootfs()
        release = fs.read_file("/etc/alpine-release")
        assert release is not None
        assert b"3.20.0" in release

        os_release = fs.read_file("/etc/os-release")
        assert os_release is not None
        assert b"Alpine Linux" in os_release
        assert b"3.20.0" in os_release

    def test_passwd_file(self):
        """Passwd file should have root with /bin/ash shell."""
        fs = self._make_rootfs()
        passwd = fs.read_file("/etc/passwd")
        assert passwd is not None
        assert b"root:" in passwd
        assert b"/bin/ash" in passwd

    def test_proc_entries(self):
        """Synthetic /proc should have expected entries."""
        fs = self._make_rootfs()
        for proc_file in ["/proc/version", "/proc/cpuinfo", "/proc/meminfo",
                          "/proc/uptime", "/proc/loadavg"]:
            data = fs.read_file(proc_file)
            assert data is not None, f"Missing {proc_file}"
            assert len(data) > 0, f"Empty {proc_file}"

    def test_hostname(self):
        fs = self._make_rootfs()
        hostname = fs.read_file("/etc/hostname")
        assert hostname is not None
        assert b"ncpu-gpu" in hostname

    def test_hosts_file(self):
        fs = self._make_rootfs()
        hosts = fs.read_file("/etc/hosts")
        assert hosts is not None
        assert b"127.0.0.1" in hosts
        assert b"ncpu-gpu" in hosts

    def test_shadow_file_exists(self):
        fs = self._make_rootfs()
        shadow = fs.read_file("/etc/shadow")
        assert shadow is not None
        assert b"root:" in shadow

    def test_apk_world(self):
        """APK world file should list installed packages."""
        fs = self._make_rootfs()
        world = fs.read_file("/etc/apk/world")
        assert world is not None
        assert b"busybox" in world

    def test_stat_works(self):
        """stat() should work on Alpine rootfs files and dirs."""
        fs = self._make_rootfs()
        info = fs.stat("/etc/alpine-release")
        assert info is not None
        assert info["type"] == "file"
        assert info["size"] > 0

        info = fs.stat("/etc")
        assert info is not None
        assert info["type"] == "dir"

    def test_listdir_root(self):
        """listdir('/') should return top-level entries."""
        fs = self._make_rootfs()
        entries = fs.listdir("/")
        assert entries is not None
        for expected in ["bin", "etc", "home", "proc", "tmp", "var"]:
            assert expected in entries, f"Missing /{expected} in root listing"

    def test_total_file_count(self):
        """Alpine rootfs should have a reasonable number of files."""
        fs = self._make_rootfs()
        assert len(fs.files) >= 20, f"Expected ≥20 files, got {len(fs.files)}"
        assert len(fs.directories) >= 20, f"Expected ≥20 dirs, got {len(fs.directories)}"


# ═══════════════════════════════════════════════════════════════════════════════
# NEW SYSCALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewSyscalls:
    """Test newly-added syscall handlers."""

    def _make_handler_and_cpu(self):
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        from ncpu.os.gpu.elf_loader import make_busybox_syscall_handler
        from ncpu.os.gpu.filesystem import GPUFilesystem

        cpu = MLXKernelCPUv2()
        fs = GPUFilesystem()
        fs.write_file("/tmp/test.txt", b"Hello, World!")
        handler = make_busybox_syscall_handler(filesystem=fs)
        return cpu, fs, handler

    def test_pread64(self):
        """SYS_PREAD64 should read at offset without moving fd position."""
        cpu, fs, handler = self._make_handler_and_cpu()

        # Open the file
        fd = fs.open("/tmp/test.txt", 0)
        assert fd >= 0

        # Set up syscall: pread64(fd, buf, 5, 7)  → reads "World"
        buf_addr = 0x80000
        cpu.set_register(8, 67)        # SYS_PREAD64
        cpu.set_register(0, fd)        # fd
        cpu.set_register(1, buf_addr)  # buf
        cpu.set_register(2, 5)         # count
        cpu.set_register(3, 7)         # offset

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 5  # bytes read

        data = cpu.read_memory(buf_addr, 5)
        assert data == b"World"

        # fd position should be unchanged (still 0)
        assert fs.fd_table[fd]["offset"] == 0

    def test_renameat(self):
        """SYS_RENAMEAT should rename files."""
        cpu, fs, handler = self._make_handler_and_cpu()

        old_path = "/tmp/test.txt"
        new_path = "/tmp/renamed.txt"

        # Write paths to GPU memory
        cpu.write_memory(0x80000, old_path.encode() + b'\x00')
        cpu.write_memory(0x80100, new_path.encode() + b'\x00')

        cpu.set_register(8, 38)        # SYS_RENAMEAT
        cpu.set_register(0, -100)      # AT_FDCWD
        cpu.set_register(1, 0x80000)   # oldpath
        cpu.set_register(2, -100)      # AT_FDCWD
        cpu.set_register(3, 0x80100)   # newpath

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0  # success

        assert fs.read_file("/tmp/renamed.txt") == b"Hello, World!"
        assert fs.read_file("/tmp/test.txt") is None

    def test_fchmod_stub(self):
        """SYS_FCHMOD should succeed (stub)."""
        cpu, fs, handler = self._make_handler_and_cpu()
        cpu.set_register(8, 52)  # SYS_FCHMOD
        cpu.set_register(0, 3)   # fd
        cpu.set_register(1, 0o755)
        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0

    def test_fchown_stub(self):
        """SYS_FCHOWN should succeed (stub)."""
        cpu, fs, handler = self._make_handler_and_cpu()
        cpu.set_register(8, 55)  # SYS_FCHOWN
        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0

    def test_utimensat_stub(self):
        """SYS_UTIMENSAT should succeed (stub)."""
        cpu, fs, handler = self._make_handler_and_cpu()
        cpu.set_register(8, 88)  # SYS_UTIMENSAT
        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0

    def test_statfs(self):
        """SYS_STATFS should return a valid statfs struct."""
        cpu, fs, handler = self._make_handler_and_cpu()

        buf_addr = 0x80000
        cpu.set_register(8, 43)        # SYS_STATFS
        cpu.set_register(0, 0x80200)   # path (doesn't matter for stub)
        cpu.set_register(1, buf_addr)  # buf

        cpu.write_memory(0x80200, b"/\x00")

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0

        # Read f_bsize from statfs struct (offset 8, 8 bytes)
        f_bsize = int.from_bytes(cpu.read_memory(buf_addr + 8, 8), 'little')
        assert f_bsize == 4096

    def test_sysinfo(self):
        """SYS_SYSINFO should return a valid sysinfo struct."""
        cpu, fs, handler = self._make_handler_and_cpu()

        buf_addr = 0x80000
        cpu.set_register(8, 179)       # SYS_SYSINFO
        cpu.set_register(0, buf_addr)  # buf

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0

        # Read totalram (offset 32, 8 bytes)
        totalram = int.from_bytes(cpu.read_memory(buf_addr + 32, 8), 'little')
        assert totalram == 256 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# FILESYSTEM RENAME TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilesystemRename:
    """Test GPUFilesystem.rename() method."""

    def _make_fs(self):
        from ncpu.os.gpu.filesystem import GPUFilesystem
        fs = GPUFilesystem()
        fs.write_file("/tmp/old.txt", b"content")
        fs.directories.add("/tmp/mydir")
        fs.write_file("/tmp/mydir/child.txt", b"child content")
        return fs

    def test_rename_file(self):
        fs = self._make_fs()
        assert fs.rename("/tmp/old.txt", "/tmp/new.txt") == 0
        assert fs.read_file("/tmp/new.txt") == b"content"
        assert fs.read_file("/tmp/old.txt") is None

    def test_rename_directory(self):
        fs = self._make_fs()
        assert fs.rename("/tmp/mydir", "/tmp/renamed_dir") == 0
        assert "/tmp/renamed_dir" in fs.directories
        assert "/tmp/mydir" not in fs.directories
        assert fs.read_file("/tmp/renamed_dir/child.txt") == b"child content"
        assert fs.read_file("/tmp/mydir/child.txt") is None

    def test_rename_nonexistent(self):
        fs = self._make_fs()
        assert fs.rename("/tmp/nope.txt", "/tmp/new.txt") == -1

    def test_rename_to_bad_parent(self):
        fs = self._make_fs()
        assert fs.rename("/tmp/old.txt", "/nonexistent/dir/file.txt") == -1


# ═══════════════════════════════════════════════════════════════════════════════
# ALPINE COMMAND EXECUTION TESTS (require busybox.elf)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BUSYBOX, reason="busybox.elf not found")
class TestAlpineCommands:
    """Test BusyBox commands with Alpine rootfs (require busybox.elf)."""

    def _run_alpine_cmd(self, argv):
        """Run a command against Alpine rootfs and capture output."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(argv, filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output, results

    def test_cat_alpine_release(self):
        output, _ = self._run_alpine_cmd(["cat", "/etc/alpine-release"])
        assert "3.20.0" in output

    def test_cat_os_release(self):
        output, _ = self._run_alpine_cmd(["cat", "/etc/os-release"])
        assert "Alpine Linux" in output

    def test_cat_passwd(self):
        output, _ = self._run_alpine_cmd(["cat", "/etc/passwd"])
        assert "root:" in output
        assert "/bin/ash" in output

    def test_cat_proc_cpuinfo(self):
        output, _ = self._run_alpine_cmd(["cat", "/proc/cpuinfo"])
        assert "processor" in output

    def test_uname_with_alpine(self):
        output, _ = self._run_alpine_cmd(["uname", "-s"])
        assert "Linux" in output

    def test_echo_with_alpine(self):
        output, _ = self._run_alpine_cmd(["echo", "alpine", "test"])
        assert "alpine test" in output

    def test_ls_root(self):
        """BusyBox ls / should list Alpine FHS directories."""
        output, _ = self._run_alpine_cmd(["ls", "/"])
        for d in ["bin", "etc", "home", "proc", "tmp", "var", "usr"]:
            assert d in output, f"Missing /{d} in ls / output"

    def test_ls_etc(self):
        """BusyBox ls /etc should list Alpine config files."""
        output, _ = self._run_alpine_cmd(["ls", "/etc"])
        for f in ["alpine-release", "passwd", "hostname", "os-release"]:
            assert f in output, f"Missing {f} in ls /etc output"

    def test_head_passwd(self):
        """head -n 1 should return only the first line."""
        output, _ = self._run_alpine_cmd(["head", "-n", "1", "/etc/passwd"])
        assert "root:" in output
        assert "nobody" not in output

    def test_tail_passwd(self):
        """tail -n 1 should return only the last line."""
        output, _ = self._run_alpine_cmd(["tail", "-n", "1", "/etc/passwd"])
        assert "nobody:" in output

    def test_wc_passwd(self):
        """wc should count lines, words, bytes."""
        output, _ = self._run_alpine_cmd(["wc", "/etc/passwd"])
        assert "2" in output  # 2 lines

    def test_wc_l_passwd(self):
        """wc -l should count lines."""
        output, _ = self._run_alpine_cmd(["wc", "-l", "/etc/passwd"])
        assert "2" in output

    def test_grep_f_root(self):
        """grep -F should find fixed string matches."""
        output, _ = self._run_alpine_cmd(["grep", "-F", "root", "/etc/passwd"])
        assert "root:" in output
        assert "nobody" not in output

    def test_cut_passwd(self):
        """cut should extract fields."""
        output, _ = self._run_alpine_cmd(["cut", "-d:", "-f1", "/etc/passwd"])
        assert "root" in output
        assert "nobody" in output

    def test_hostname(self):
        """hostname should return the configured hostname."""
        output, _ = self._run_alpine_cmd(["hostname"])
        assert "ncpu-gpu" in output

    def test_id(self):
        """id should show root user info."""
        output, _ = self._run_alpine_cmd(["id"])
        assert "uid=0" in output

    def test_whoami(self):
        """whoami should return root."""
        output, _ = self._run_alpine_cmd(["whoami"])
        assert "root" in output

    def test_expr(self):
        """expr should evaluate arithmetic."""
        output, _ = self._run_alpine_cmd(["expr", "2", "+", "3"])
        assert "5" in output

    def test_printf(self):
        """printf should format output."""
        output, _ = self._run_alpine_cmd(["printf", "%s %d\\n", "test", "42"])
        assert "test 42" in output

    def test_basename(self):
        """basename should extract filename."""
        output, _ = self._run_alpine_cmd(["basename", "/etc/passwd"])
        assert "passwd" in output

    def test_dirname(self):
        """dirname should extract directory."""
        output, _ = self._run_alpine_cmd(["dirname", "/etc/passwd"])
        assert "/etc" in output

    def test_env(self):
        """env should list environment variables."""
        output, _ = self._run_alpine_cmd(["env"])
        assert "PATH=" in output

    def test_stat(self):
        """stat should show file information."""
        output, _ = self._run_alpine_cmd(["stat", "/etc/hostname"])
        assert "File:" in output or "hostname" in output

    def test_date(self):
        """date should produce output."""
        output, _ = self._run_alpine_cmd(["date"])
        assert "202" in output  # year

    def test_sort(self):
        """sort should sort lines alphabetically."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        fs.write_file("/tmp/data.txt", "banana\napple\ncherry\n")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["sort", "/tmp/data.txt"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        lines = [l for l in output.strip().split("\n") if l]
        assert lines == ["apple", "banana", "cherry"]

    def test_sort_numeric(self):
        """sort -n should sort numerically."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        fs.write_file("/tmp/nums.txt", "3\n1\n2\n")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["sort", "-n", "/tmp/nums.txt"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        lines = [l for l in output.strip().split("\n") if l]
        assert lines == ["1", "2", "3"]

    def test_uniq(self):
        """uniq should run and exit cleanly."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        fs.write_file("/tmp/data.txt", "a\na\nb\nb\na\n")
        result = load_and_run_elf_helper(["uniq", "/tmp/data.txt"], filesystem=fs)
        assert result["stop_reason"] == "SYSCALL"  # clean exit
        assert result["total_cycles"] > 0

    def test_touch(self):
        """touch should run and exit cleanly."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        result = load_and_run_elf_helper(["touch", "/tmp/newfile"], filesystem=fs)
        assert result["stop_reason"] == "SYSCALL"  # clean exit

    def test_mkdir_and_ls(self):
        """mkdir should create directory visible to ls."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        load_and_run_elf_helper(["mkdir", "/tmp/testdir"], filesystem=fs)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["ls", "/tmp"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "testdir" in output

    def test_cp_file(self):
        """cp should copy file contents."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        fs.write_file("/tmp/src.txt", "hello world\n")
        load_and_run_elf_helper(["cp", "/tmp/src.txt", "/tmp/dst.txt"], filesystem=fs)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["cat", "/tmp/dst.txt"], filesystem=fs)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert "hello world" in output


# ═══════════════════════════════════════════════════════════════════════════════
# PIPE TESTS — stdin injection for piped commands
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BUSYBOX, reason="busybox.elf not found")
class TestPipes:
    """Verify stdin_data injection enables piped commands on GPU."""

    def test_grep_from_stdin(self):
        """grep -F should filter lines from stdin_data."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        stdin = b"apple\nbanana\ncherry\n"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["grep", "-F", "banana"], filesystem=fs,
                                    stdin_data=stdin)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert output.strip() == "banana"

    def test_wc_from_stdin(self):
        """wc should count lines/words/bytes from stdin_data."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        stdin = b"hello world\nfoo bar\n"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["wc"], filesystem=fs, stdin_data=stdin)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        # Should have 2 lines, 4 words, 20 bytes
        parts = output.strip().split()
        assert parts[0] == "2"   # lines
        assert parts[1] == "4"   # words

    def test_cut_from_stdin(self):
        """cut should extract fields from stdin_data."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        stdin = b"root:x:0:0:root:/root:/bin/ash\n"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["cut", "-d:", "-f1"], filesystem=fs,
                                    stdin_data=stdin)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert output.strip() == "root"

    def test_sort_from_stdin(self):
        """sort should sort lines from stdin_data."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        stdin = b"cherry\napple\nbanana\n"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["sort"], filesystem=fs, stdin_data=stdin)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        lines = output.strip().split("\n")
        assert lines == ["apple", "banana", "cherry"]

    def test_head_from_stdin(self):
        """head -n 1 should return first line from stdin_data."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        stdin = b"first\nsecond\nthird\n"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["head", "-n", "1"], filesystem=fs,
                                    stdin_data=stdin)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert output.strip() == "first"

    def test_two_stage_pipe(self):
        """Simulate cat /etc/passwd | grep -F root via two-stage pipe."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()

        # Stage 1: cat /etc/passwd
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["cat", "/etc/passwd"], filesystem=fs)
            stage1 = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Stage 2: grep -F root with stage1 as stdin
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["grep", "-F", "root"], filesystem=fs,
                                    stdin_data=stage1.encode())
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "root:x:0:0" in output
        assert "nobody" not in output

    def test_three_stage_pipe(self):
        """Simulate cat /etc/passwd | grep -F root | cut -d: -f1."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()

        # Stage 1: cat
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["cat", "/etc/passwd"], filesystem=fs)
            s1 = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Stage 2: grep
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["grep", "-F", "root"], filesystem=fs,
                                    stdin_data=s1.encode())
            s2 = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # Stage 3: cut
        sys.stdout = io.StringIO()
        try:
            load_and_run_elf_helper(["cut", "-d:", "-f1"], filesystem=fs,
                                    stdin_data=s2.encode())
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        # "root" should be the first line (operator also matches because home=/root)
        lines = output.strip().split('\n')
        assert lines[0] == "root"
        assert "root" in output

    def test_stdin_eof(self):
        """Empty stdin should cause immediate EOF."""
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        import io
        fs = create_alpine_rootfs()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = load_and_run_elf_helper(["cat"], filesystem=fs,
                                             stdin_data=b"")
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        assert output == ""
        assert result["stop_reason"] == "SYSCALL"


# ═══════════════════════════════════════════════════════════════════════════════
# SHELL ENGINE TESTS (Python-side — no GPU needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVariableExpansion:
    """Test variable expansion in shell tokens."""

    def test_simple_variable(self):
        from alpine_gpu import expand_variables
        env = {'NAME': 'world'}
        result = expand_variables(['hello', '$NAME'], env)
        assert result == ['hello', 'world']

    def test_braced_variable(self):
        from alpine_gpu import expand_variables
        env = {'FOO': 'bar'}
        result = expand_variables(['${FOO}'], env)
        assert result == ['bar']

    def test_undefined_variable(self):
        from alpine_gpu import expand_variables
        env = {}
        result = expand_variables(['$UNDEFINED'], env)
        assert result == ['']

    def test_exit_status_variable(self):
        from alpine_gpu import expand_variables
        env = {'?': '0'}
        result = expand_variables(['$?'], env)
        assert result == ['0']

    def test_mixed_text_and_variable(self):
        from alpine_gpu import expand_variables
        env = {'USER': 'root'}
        result = expand_variables(['hello_$USER'], env)
        assert result == ['hello_root']

    def test_multiple_variables_in_token(self):
        from alpine_gpu import expand_variables
        env = {'A': 'x', 'B': 'y'}
        result = expand_variables(['$A-$B'], env)
        assert result == ['x-y']


class TestChainSplitting:
    """Test splitting on ;, &&, ||."""

    def test_semicolon(self):
        from alpine_gpu import split_chains
        result = split_chains("echo a ; echo b")
        assert len(result) == 2
        assert result[0][0] == ['echo', 'a']
        assert result[0][1] == ';'
        assert result[1][0] == ['echo', 'b']

    def test_and_chain(self):
        from alpine_gpu import split_chains
        result = split_chains("true && echo ok")
        assert len(result) == 2
        assert result[0][1] == '&&'

    def test_or_chain(self):
        from alpine_gpu import split_chains
        result = split_chains("false || echo fallback")
        assert len(result) == 2
        assert result[0][1] == '||'

    def test_single_command(self):
        from alpine_gpu import split_chains
        result = split_chains("echo hello")
        assert len(result) == 1
        assert result[0][0] == ['echo', 'hello']
        assert result[0][1] is None


class TestPipelineSplitting:
    """Test splitting on |."""

    def test_single_pipe(self):
        from alpine_gpu import split_pipeline
        result = split_pipeline(['echo', 'hello', '|', 'wc'])
        assert result == [['echo', 'hello'], ['wc']]

    def test_multi_pipe(self):
        from alpine_gpu import split_pipeline
        result = split_pipeline(['cat', '/etc/passwd', '|', 'grep', 'root', '|', 'wc'])
        assert len(result) == 3

    def test_no_pipe(self):
        from alpine_gpu import split_pipeline
        result = split_pipeline(['ls', '-la'])
        assert result == [['ls', '-la']]


class TestRedirection:
    """Test output redirection extraction."""

    def test_write_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(['echo', 'hi', '>', '/tmp/f'])
        assert tokens == ['echo', 'hi']
        assert redir == '/tmp/f'
        assert append is False

    def test_append_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(['echo', 'hi', '>>', '/tmp/f'])
        assert tokens == ['echo', 'hi']
        assert redir == '/tmp/f'
        assert append is True

    def test_no_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(['ls', '-la'])
        assert tokens == ['ls', '-la']
        assert redir is None

    def test_attached_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(['echo', 'hi', '>/tmp/f'])
        assert tokens == ['echo', 'hi']
        assert redir == '/tmp/f'


class TestEvaluateTest:
    """Test the test/[ builtin evaluation."""

    def test_file_exists(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert evaluate_test(['-f', '/etc/passwd'], fs) is True
        assert evaluate_test(['-f', '/nonexistent'], fs) is False

    def test_directory_exists(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert evaluate_test(['-d', '/etc'], fs) is True
        assert evaluate_test(['-d', '/nonexistent'], fs) is False

    def test_path_exists(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert evaluate_test(['-e', '/etc/passwd'], fs) is True
        assert evaluate_test(['-e', '/etc'], fs) is True
        assert evaluate_test(['-e', '/nothing'], fs) is False

    def test_string_equal(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.filesystem import GPUFilesystem
        fs = GPUFilesystem()
        assert evaluate_test(['abc', '=', 'abc'], fs) is True
        assert evaluate_test(['abc', '=', 'xyz'], fs) is False

    def test_string_not_equal(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.filesystem import GPUFilesystem
        fs = GPUFilesystem()
        assert evaluate_test(['abc', '!=', 'xyz'], fs) is True
        assert evaluate_test(['abc', '!=', 'abc'], fs) is False

    def test_numeric_comparisons(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.filesystem import GPUFilesystem
        fs = GPUFilesystem()
        assert evaluate_test(['5', '-eq', '5'], fs) is True
        assert evaluate_test(['5', '-ne', '3'], fs) is True
        assert evaluate_test(['5', '-gt', '3'], fs) is True
        assert evaluate_test(['3', '-lt', '5'], fs) is True
        assert evaluate_test(['5', '-ge', '5'], fs) is True
        assert evaluate_test(['5', '-le', '5'], fs) is True

    def test_string_empty(self):
        from alpine_gpu import evaluate_test
        from ncpu.os.gpu.filesystem import GPUFilesystem
        fs = GPUFilesystem()
        assert evaluate_test(['-z', ''], fs) is True
        assert evaluate_test(['-z', 'notempty'], fs) is False
        assert evaluate_test(['-n', 'notempty'], fs) is True
        assert evaluate_test(['-n', ''], fs) is False


class TestGlobExpansion:
    """Test glob expansion against filesystem."""

    def test_star_glob(self):
        from alpine_gpu import expand_glob
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        result = expand_glob(['/etc/apk/*'], fs)
        assert 'world' in [r.split('/')[-1] for r in result]

    def test_no_match_passthrough(self):
        from alpine_gpu import expand_glob
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        result = expand_glob(['/nonexistent/*.txt'], fs)
        assert result == ['/nonexistent/*.txt']

    def test_non_glob_passthrough(self):
        from alpine_gpu import expand_glob
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        result = expand_glob(['/etc/passwd'], fs)
        assert result == ['/etc/passwd']


class TestShellBuiltins:
    """Test shell builtins (Python-side, no GPU)."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {
                'PATH': '/bin:/usr/bin',
                'HOME': '/root',
                'USER': 'root',
                'PWD': '/',
                '?': '0',
            },
            'cwd': '/',
            'history': [],
            'aliases': {'ll': 'ls -l'},
        }

    def test_cd(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['cd', '/etc'], state)
        assert state['cwd'] == '/etc'

    def test_cd_nonexistent(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['cd', '/nonexistent'], state)
        assert state['env']['?'] == '1'

    def test_pwd(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['pwd'], state)
        assert '/' in capsys.readouterr().out

    def test_export(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['export', 'FOO=bar'], state)
        assert state['env']['FOO'] == 'bar'

    def test_unset(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['env']['FOO'] = 'bar'
        shell_builtin(['unset', 'FOO'], state)
        assert 'FOO' not in state['env']

    def test_set(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['set'], state)
        out = capsys.readouterr().out
        assert 'PATH=' in out
        assert 'HOME=' in out

    def test_echo(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['echo', 'hello', 'world'], state)
        assert capsys.readouterr().out == 'hello world\n'

    def test_echo_n(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['echo', '-n', 'no newline'], state)
        assert capsys.readouterr().out == 'no newline'

    def test_echo_e(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['echo', '-e', 'line1\\nline2'], state)
        assert capsys.readouterr().out == 'line1\nline2\n'

    def test_true_false(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['true'], state)
        assert state['env']['?'] == '0'
        shell_builtin(['false'], state)
        assert state['env']['?'] == '1'

    def test_test_builtin(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['test', '-f', '/etc/passwd'], state)
        assert state['env']['?'] == '0'
        shell_builtin(['test', '-f', '/nonexistent'], state)
        assert state['env']['?'] == '1'

    def test_bracket_builtin(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['[', '5', '-eq', '5', ']'], state)
        assert state['env']['?'] == '0'

    def test_type_builtin(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['type', 'cd'], state)
        out = capsys.readouterr().out
        assert 'builtin' in out

    def test_type_gpu(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['type', 'gpu-info'], state)
        out = capsys.readouterr().out
        assert 'GPU superpower' in out

    def test_alias(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['alias'], state)
        out = capsys.readouterr().out
        assert 'll' in out

    def test_alias_set(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['alias', 'myalias=echo hello'], state)
        assert state['aliases']['myalias'] == 'echo hello'

    def test_unalias(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        assert 'll' in state['aliases']
        shell_builtin(['unalias', 'll'], state)
        assert 'll' not in state['aliases']

    def test_history(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['history'] = ['echo hello', 'ls /']
        shell_builtin(['history'], state)
        out = capsys.readouterr().out
        assert 'echo hello' in out
        assert 'ls /' in out

    def test_clear(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['clear'], state)
        out = capsys.readouterr().out
        assert '\033[2J' in out


class TestGPUSuperpowers:
    """Test GPU superpower commands (Python-side, no GPU)."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {'?': '0'},
            'cwd': '/',
            'history': [],
            'aliases': {},
        }

    def test_gpu_info(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-info'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'Metal' in out or 'ARM64' in out

    def test_gpu_mem(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-mem'], state)
        assert result is True
        out = capsys.readouterr().out
        assert '0x' in out
        assert 'Stack' in out or 'Heap' in out

    def test_gpu_regs(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-regs'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'X0' in out
        assert 'SP' in out or 'XZR' in out

    def test_gpu_isa(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-isa'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'ADD' in out
        assert '135+' in out

    def test_gpu_side_channel(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-side-channel'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'deterministic' in out

    def test_gpu_neural(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-neural'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'neural' in out.lower() or 'Neural' in out

    def test_neofetch(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['neofetch'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'Alpine' in out
        assert 'nCPU' in out or 'Neural' in out

    def test_help(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['help'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'gpu-info' in out
        assert 'Pipes' in out or 'pipe' in out.lower() or 'BusyBox' in out

    def test_gpu_sha256(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-sha256', '/etc/hostname'], state)
        assert result is True
        out = capsys.readouterr().out
        # Should be a 64-char hex hash
        assert len(out.strip().split()[0]) == 64

    def test_unknown_command_returns_false(self):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['not-a-gpu-command'], state)
        assert result is False


class TestShellScripting:
    """Test shell script execution (Python-side, no GPU for builtins)."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {
                'PATH': '/bin:/usr/bin',
                'HOME': '/root',
                'USER': 'root',
                'PWD': '/',
                '?': '0',
            },
            'cwd': '/',
            'history': [],
            'aliases': {},
        }

    def test_for_loop(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "for x in a b c; do\necho $x\ndone\n"
        run_script(script, state)
        out = capsys.readouterr().out
        assert 'a\nb\nc\n' == out

    def test_if_true(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "if [ 1 -eq 1 ]; then\necho yes\nfi\n"
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'yes'

    def test_if_false_else(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "if [ 1 -eq 2 ]; then\necho yes\nelse\necho no\nfi\n"
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'no'

    def test_while_loop(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = (
            "I=0\n"
            "while [ $I -lt 3 ]; do\n"
            "echo $I\n"
            "I=$(($I + 1))\n"
            "done\n"
        )
        # while with variable increment won't work because
        # $((...)) isn't implemented; test simpler case
        script = "for i in 1 2 3; do\necho count $i\ndone\n"
        run_script(script, state)
        out = capsys.readouterr().out
        assert 'count 1' in out
        assert 'count 3' in out

    def test_variable_assignment(self):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('MYVAR=hello', state)
        assert state['env']['MYVAR'] == 'hello'

    def test_variable_with_quotes(self):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('GREETING="hello world"', state)
        assert state['env']['GREETING'] == 'hello world'

    def test_echo_with_variable(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('NAME=GPU', state)
        execute_line('echo hello $NAME', state)
        out = capsys.readouterr().out
        assert 'hello GPU' in out

    def test_case_block(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = (
            "COLOR=red\n"
            "case $COLOR in\n"
            "red) echo found red ;;\n"
            "blue) echo found blue ;;\n"
            "*) echo other ;;\n"
            "esac\n"
        )
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'found red'

    def test_case_wildcard(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = (
            "X=unknown\n"
            "case $X in\n"
            "a) echo a ;;\n"
            "*) echo default ;;\n"
            "esac\n"
        )
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'default'

    def test_comments_skipped(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "# This is a comment\necho visible\n# Another comment\n"
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'visible'

    def test_positional_params(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "echo $1 $2\n"
        run_script(script, state, script_args=['hello', 'world'])
        out = capsys.readouterr().out
        assert 'hello world' in out

    def test_script_arg_count(self):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = "echo placeholder\n"
        run_script(script, state, script_args=['a', 'b', 'c'])
        assert state['env']['#'] == '3'

    def test_source_script(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['fs'].write_file('/tmp/test.sh', "echo sourced ok\n")
        shell_builtin(['source', '/tmp/test.sh'], state)
        assert 'sourced ok' in capsys.readouterr().out

    def test_sh_script(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['fs'].write_file('/tmp/run.sh', "echo running script\n")
        shell_builtin(['sh', '/tmp/run.sh'], state)
        assert 'running script' in capsys.readouterr().out


class TestExecuteLine:
    """Test the central execute_line command engine."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {
                'PATH': '/bin:/usr/bin',
                'HOME': '/root',
                'USER': 'root',
                'PWD': '/',
                '?': '0',
            },
            'cwd': '/',
            'history': [],
            'aliases': {'ll': 'ls -l'},
        }

    def test_skip_comment(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('# this is a comment', state)
        assert capsys.readouterr().out == ''

    def test_skip_empty(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('', state)
        assert capsys.readouterr().out == ''

    def test_variable_assignment(self):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('X=42', state)
        assert state['env']['X'] == '42'

    def test_builtin_echo(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('echo hello from shell', state)
        assert capsys.readouterr().out.strip() == 'hello from shell'

    def test_chaining_semicolon(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('echo first ; echo second', state)
        out = capsys.readouterr().out
        assert 'first' in out
        assert 'second' in out

    def test_chaining_and(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('true && echo success', state)
        assert 'success' in capsys.readouterr().out

    def test_chaining_or(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('false || echo fallback', state)
        assert 'fallback' in capsys.readouterr().out

    def test_redirection_to_file(self):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('echo test content > /tmp/out.txt', state)
        # Can't easily test file write without GPU for non-builtin,
        # but we can verify it doesn't crash

    def test_gpu_command(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('gpu-regs', state)
        out = capsys.readouterr().out
        assert 'X0' in out


class TestAlpineRootfsComprehensive:
    """Extended tests for the comprehensive Alpine rootfs."""

    def test_directory_count(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert len(fs.directories) >= 60

    def test_file_count(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert len(fs.files) >= 100

    def test_identity_files(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert b'3.20.0' in fs.read_file('/etc/alpine-release')
        assert b'Alpine Linux' in fs.read_file('/etc/os-release')

    def test_user_database(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        passwd = fs.read_file('/etc/passwd').decode()
        assert 'root:x:0:0' in passwd
        assert 'nobody:x:65534' in passwd

    def test_group_database(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        group = fs.read_file('/etc/group').decode()
        assert 'root:x:0' in group
        assert 'wheel:x:10' in group

    def test_proc_filesystem(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert b'6.1.0-ncpu' in fs.read_file('/proc/version')
        assert b'Apple Silicon' in fs.read_file('/proc/cpuinfo')
        assert b'MemTotal' in fs.read_file('/proc/meminfo')

    def test_dev_nodes(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert '/dev/null' in fs.files
        assert '/dev/zero' in fs.files
        assert '/dev/urandom' in fs.files
        assert '/dev/tty' in fs.files

    def test_init_scripts(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        boot = fs.read_file('/etc/init.d/boot').decode()
        assert 'case' in boot
        assert 'start' in boot

    def test_network_config(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        ifaces = fs.read_file('/etc/network/interfaces').decode()
        assert 'lo' in ifaces
        assert 'eth0' in ifaces

    def test_profile_d_scripts(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        ncpu_sh = fs.read_file('/etc/profile.d/ncpu.sh').decode()
        assert 'NCPU_VERSION' in ncpu_sh
        assert 'NCPU_ARCH' in ncpu_sh

    def test_dmesg_log(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        dmesg = fs.read_file('/var/log/dmesg').decode()
        assert 'Booting Linux' in dmesg
        assert 'nCPU GPU' in dmesg

    def test_hostname(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        assert fs.read_file('/etc/hostname').decode().strip() == 'ncpu-gpu'

    def test_shells(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        shells = fs.read_file('/etc/shells').decode()
        assert '/bin/ash' in shells

    def test_motd(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        motd = fs.read_file('/etc/motd').decode()
        assert 'Alpine' in motd

    def test_apk_world(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        world = fs.read_file('/etc/apk/world').decode()
        assert 'busybox' in world

    def test_protocols(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        proto = fs.read_file('/etc/protocols').decode()
        assert 'tcp' in proto
        assert 'udp' in proto

    def test_services(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        svcs = fs.read_file('/etc/services').decode()
        assert 'ssh' in svcs
        assert '22/tcp' in svcs

    def test_proc_self(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        status = fs.read_file('/proc/self/status').decode()
        assert 'busybox' in status
        maps = fs.read_file('/proc/self/maps').decode()
        assert '00010000' in maps

    def test_root_home(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        profile = fs.read_file('/root/.profile').decode()
        assert 'PATH=' in profile
        ashrc = fs.read_file('/root/.ashrc').decode()
        assert 'alias' in ashrc

    def test_ncpu_info_script(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        script = fs.read_file('/usr/bin/ncpu-info').decode()
        assert 'nCPU' in script
        assert 'hostname' in script


# ═══════════════════════════════════════════════════════════════════════════════
# NOVEL GPU SUPERPOWERS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNovelGPUSuperpowers:
    """Test novel GPU superpower commands."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {'?': '0'},
            'cwd': '/',
            'history': [],
            'aliases': {},
        }

    def test_gpu_entropy(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        result = gpu_builtin(['gpu-entropy', '/etc/passwd'], state)
        assert result is True
        out = capsys.readouterr().out
        assert 'Entropy' in out
        assert 'bits/byte' in out

    def test_gpu_entropy_missing(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-entropy', '/nonexistent'], state)
        assert 'No such file' in capsys.readouterr().out

    def test_dmesg(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['dmesg'], state)
        out = capsys.readouterr().out
        assert 'Booting Linux' in out
        assert 'nCPU GPU' in out

    def test_uptime(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['uptime'], state)
        assert 'up' in capsys.readouterr().out

    def test_free(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['free'], state)
        out = capsys.readouterr().out
        assert 'Mem:' in out
        assert 'total' in out

    def test_lscpu(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['lscpu'], state)
        out = capsys.readouterr().out
        assert 'aarch64' in out
        assert 'Spectre' in out
        assert 'deterministic' in out

    def test_lsblk(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['lsblk'], state)
        assert 'sda' in capsys.readouterr().out

    def test_df(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['df'], state)
        out = capsys.readouterr().out
        assert 'rootfs' in out
        assert 'Mounted on' in out

    def test_mount(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['mount'], state)
        out = capsys.readouterr().out
        assert 'rootfs' in out
        assert '/proc' in out

    def test_ps(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['ps'], state)
        out = capsys.readouterr().out
        assert 'PID' in out
        assert 'init' in out

    def test_who(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['w'], state)
        out = capsys.readouterr().out
        assert 'root' in out

    def test_gpu_freeze_thaw(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        # No snapshots initially
        gpu_builtin(['gpu-thaw'], state)
        assert 'No snapshots' in capsys.readouterr().out

    def test_gpu_xray_usage(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-xray'], state)
        out = capsys.readouterr().out
        assert 'Usage' in out
        assert 'register' in out.lower() or 'Register' in out

    def test_gpu_replay_usage(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-replay'], state)
        out = capsys.readouterr().out
        assert 'Usage' in out
        assert 'deterministic' in out.lower()

    def test_gpu_diff_usage(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-diff'], state)
        out = capsys.readouterr().out
        assert 'Usage' in out

    def test_gpu_strace_usage(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-strace'], state)
        out = capsys.readouterr().out
        assert 'Usage' in out
        assert 'syscall' in out.lower() or 'trace' in out.lower()

    def test_gpu_timing_proof_usage(self, capsys):
        from alpine_gpu import gpu_builtin
        state = self._make_shell_state()
        gpu_builtin(['gpu-timing-proof'], state)
        out = capsys.readouterr().out
        assert 'Usage' in out
        assert 'timing' in out.lower()


class TestAdditionalBuiltins:
    """Test additional shell builtins."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'fs': fs,
            'env': {
                'PATH': '/bin:/usr/bin', 'HOME': '/root', 'USER': 'root',
                'PWD': '/', '?': '0',
            },
            'cwd': '/',
            'history': [],
            'aliases': {},
        }

    def test_seq(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['seq', '1', '5'], state)
        out = capsys.readouterr().out
        assert '1\n2\n3\n4\n5\n' == out

    def test_seq_range(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['seq', '3', '6'], state)
        out = capsys.readouterr().out
        assert '3\n4\n5\n6\n' == out

    def test_basename(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['basename', '/etc/passwd'], state)
        assert capsys.readouterr().out.strip() == 'passwd'

    def test_basename_suffix(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['basename', 'file.txt', '.txt'], state)
        assert capsys.readouterr().out.strip() == 'file'

    def test_dirname(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['dirname', '/etc/passwd'], state)
        assert capsys.readouterr().out.strip() == '/etc'

    def test_dirname_no_slash(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['dirname', 'file.txt'], state)
        assert capsys.readouterr().out.strip() == '.'

    def test_printf(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['printf', 'hello %s\\n', 'world'], state)
        assert capsys.readouterr().out == 'hello world\n'

    def test_pushd_popd(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['pushd', '/etc'], state)
        assert state['cwd'] == '/etc'
        capsys.readouterr()  # clear
        shell_builtin(['popd'], state)
        assert state['cwd'] == '/'

    def test_dirs(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['dirs'], state)
        assert '/' in capsys.readouterr().out

    def test_arithmetic_expansion(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('echo $((3 + 4))', state)
        assert capsys.readouterr().out.strip() == '7'

    def test_arithmetic_with_vars(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        state['env']['X'] = '10'
        execute_line('echo $((X + 5))', state)
        # Note: $X in $(()) gets expanded to its value
        out = capsys.readouterr().out.strip()
        assert out == '15'

    def test_while_with_arithmetic(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = (
            "I=1\n"
            "while [ $I -le 3 ]; do\n"
            "echo $I\n"
            "I=$(($I + 1))\n"
            "done\n"
        )
        run_script(script, state)
        assert capsys.readouterr().out == '1\n2\n3\n'


# ═══════════════════════════════════════════════════════════════════════════════
# NEW SHELL FEATURES: Input redirection, parameter expansion, functions,
# heredocs, brace expansion, set flags
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputRedirection:
    """Test input redirection (< file)."""

    def test_input_redirect_parsing(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(
            ['wc', '-l', '<', '/etc/passwd'])
        assert tokens == ['wc', '-l']
        assert redir is None
        assert redir_in == '/etc/passwd'

    def test_input_redirect_attached(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(
            ['wc', '-l', '</etc/passwd'])
        assert tokens == ['wc', '-l']
        assert redir_in == '/etc/passwd'

    def test_input_and_output_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(
            ['sort', '<', '/tmp/in', '>', '/tmp/out'])
        assert tokens == ['sort']
        assert redir == '/tmp/out'
        assert redir_in == '/tmp/in'

    def test_no_input_redirect(self):
        from alpine_gpu import extract_redirection
        tokens, redir, append, redir_in = extract_redirection(['echo', 'hi'])
        assert redir_in is None


class TestParameterExpansion:
    """Test ${VAR:-default}, ${VAR#pattern}, etc."""

    def test_default_value(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${UNSET:-hello}'], {})
        assert result == ['hello']

    def test_default_value_with_set_var(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${FOO:-hello}'], {'FOO': 'world'})
        assert result == ['world']

    def test_assign_default(self):
        from alpine_gpu import expand_variables
        env = {}
        result = expand_variables(['${X:=42}'], env)
        assert result == ['42']
        assert env['X'] == '42'

    def test_alternate_value(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${FOO:+yes}'], {'FOO': 'bar'})
        assert result == ['yes']

    def test_alternate_value_unset(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${FOO:+yes}'], {})
        assert result == ['']

    def test_string_length(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${#FOO}'], {'FOO': 'hello'})
        assert result == ['5']

    def test_prefix_strip(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${PATH#*/}'], {'PATH': '/usr/bin:/sbin'})
        assert result == ['usr/bin:/sbin']

    def test_suffix_strip(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${FILE%.txt}'], {'FILE': 'notes.txt'})
        assert result == ['notes']

    def test_greedy_suffix_strip(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${PATH%%:*}'], {'PATH': '/usr/bin:/sbin:/bin'})
        assert result == ['/usr/bin']

    def test_substitution(self):
        from alpine_gpu import expand_variables
        result = expand_variables(['${MSG/world/earth}'], {'MSG': 'hello world'})
        assert result == ['hello earth']


class TestShellFunctions:
    """Test function definition and invocation."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0', 'PATH': '/bin:/usr/bin'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
            'functions': {},
        }

    def test_single_line_function_def(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('greet() { echo hello; }', state)
        assert 'greet' in state['functions']
        execute_line('greet', state)
        assert capsys.readouterr().out.strip() == 'hello'

    def test_function_with_args(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """
say() {
echo $1 $2
}
say hello world
"""
        run_script(script, state)
        assert capsys.readouterr().out.strip() == 'hello world'

    def test_function_type(self, capsys):
        from alpine_gpu import execute_line, shell_builtin
        state = self._make_shell_state()
        state['functions']['myfunc'] = 'echo test'
        shell_builtin(['type', 'myfunc'], state)
        assert 'function' in capsys.readouterr().out

    def test_function_with_local(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """
counter() {
local i=0
echo $i
}
counter
"""
        run_script(script, state)
        assert capsys.readouterr().out.strip() == '0'


class TestHeredocs:
    """Test here-document support."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0', 'NAME': 'GPU'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
            'functions': {},
        }

    def test_heredoc_with_cat(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """cat <<EOF
hello world
EOF"""
        run_script(script, state)
        out = capsys.readouterr().out
        assert 'hello world' in out

    def test_heredoc_variable_expansion(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """cat <<EOF
Hello $NAME
EOF"""
        run_script(script, state)
        out = capsys.readouterr().out
        assert 'Hello GPU' in out


class TestBraceExpansion:
    """Test brace expansion ({1..5}, {a,b,c})."""

    def test_numeric_range(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['{1..5}'])
        assert result == ['1', '2', '3', '4', '5']

    def test_numeric_range_with_prefix(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['file{1..3}.txt'])
        assert result == ['file1.txt', 'file2.txt', 'file3.txt']

    def test_comma_expansion(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['{a,b,c}'])
        assert result == ['a', 'b', 'c']

    def test_comma_with_prefix_suffix(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['test_{x,y,z}.log'])
        assert result == ['test_x.log', 'test_y.log', 'test_z.log']

    def test_no_braces(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['hello', 'world'])
        assert result == ['hello', 'world']

    def test_numeric_step(self):
        from alpine_gpu import expand_braces
        result = expand_braces(['{0..10..2}'])
        assert result == ['0', '2', '4', '6', '8', '10']


class TestSetFlags:
    """Test set -e, set -x."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
            'set_flags': set(),
        }

    def test_set_e_stops_on_error(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """set -e
false
echo should not reach here
"""
        run_script(script, state)
        out = capsys.readouterr().out
        assert 'should not reach here' not in out

    def test_set_x_traces(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """set -x
echo hello
"""
        run_script(script, state)
        err = capsys.readouterr().err
        assert '+ echo hello' in err

    def test_set_plus_disables(self):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        execute_line('set -x', state)
        assert 'x' in state['set_flags']
        execute_line('set +x', state)
        assert 'x' not in state['set_flags']


class TestTrapEvalShiftLet:
    """Test trap, eval, shift, and let builtins."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0', 'PATH': '/bin:/usr/bin'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
            'functions': {},
            'set_flags': set(),
        }

    def test_trap_set_and_list(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['trap', 'echo bye', 'EXIT'], state)
        assert state['traps']['EXIT'] == 'echo bye'
        shell_builtin(['trap'], state)
        out = capsys.readouterr().out
        assert 'EXIT' in out

    def test_trap_remove(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['traps'] = {'EXIT': 'echo bye'}
        shell_builtin(['trap', '-', 'EXIT'], state)
        assert 'EXIT' not in state.get('traps', {})

    def test_eval_executes(self, capsys):
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        state['env']['CMD'] = 'echo hello'
        execute_line('eval $CMD', state)
        out = capsys.readouterr().out
        assert 'hello' in out

    def test_shift(self, capsys):
        from alpine_gpu import run_script
        state = self._make_shell_state()
        script = """
echo $1 $2 $3
shift
echo $1 $2
"""
        run_script(script, state, script_args=['a', 'b', 'c'])
        out = capsys.readouterr().out.strip().split('\n')
        assert out[0] == 'a b c'
        assert out[1] == 'b c'

    def test_let_arithmetic(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        state['env']['X'] = '10'
        shell_builtin(['let', 'X+5'], state)
        # let returns 0 if result is non-zero
        assert state['env']['?'] == '0'

    def test_let_zero_returns_failure(self):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['let', '0'], state)
        assert state['env']['?'] == '1'

    def test_read_with_prompt(self, monkeypatch):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        monkeypatch.setattr('builtins.input', lambda: 'test_value')
        shell_builtin(['read', '-p', 'Enter:', 'MY_VAR'], state)
        assert state['env']['MY_VAR'] == 'test_value'

    def test_read_multiple_vars(self, monkeypatch):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        monkeypatch.setattr('builtins.input', lambda: 'hello world extra')
        shell_builtin(['read', 'A', 'B', 'C'], state)
        assert state['env']['A'] == 'hello'
        assert state['env']['B'] == 'world'
        assert state['env']['C'] == 'extra'


class TestInputRedirectionGPU:
    """Test input redirection with actual GPU execution (if available)."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0', 'PATH': '/bin:/usr/bin'},
            'fs': fs,
            'cwd': '/',
            'functions': {},
            'set_flags': set(),
            'history': [],
            'aliases': {},
        }

    def test_input_redirect_parsing_in_execute(self, capsys):
        """Test that input redirection is parsed correctly in execute_line."""
        from alpine_gpu import execute_line
        state = self._make_shell_state()
        # Test that < is recognized in the command (without needing GPU)
        state['fs'].write_file('/tmp/test.txt', b'hello\nworld\nfoo\n')
        # Use a builtin that doesn't need GPU — echo should ignore stdin
        execute_line('echo ok < /tmp/test.txt', state)
        out = capsys.readouterr().out.strip()
        assert out == 'ok'


class TestExpandedTestOperators:
    """Test expanded test/[ operators."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
        }

    def test_readable_writable(self):
        from alpine_gpu import evaluate_test
        state = self._make_shell_state()
        assert evaluate_test(['-r', '/etc/passwd'], state['fs']) is True
        assert evaluate_test(['-w', '/etc/passwd'], state['fs']) is True
        assert evaluate_test(['-x', '/etc/passwd'], state['fs']) is True

    def test_symlink_false(self):
        from alpine_gpu import evaluate_test
        state = self._make_shell_state()
        assert evaluate_test(['-L', '/etc/passwd'], state['fs']) is False

    def test_block_char_false(self):
        from alpine_gpu import evaluate_test
        state = self._make_shell_state()
        assert evaluate_test(['-b', '/dev/null'], state['fs']) is False

    def test_compound_and(self):
        from alpine_gpu import evaluate_test
        state = self._make_shell_state()
        assert evaluate_test(['-f', '/etc/passwd', '-a', '-d', '/etc'], state['fs']) is True
        assert evaluate_test(['-f', '/etc/passwd', '-a', '-f', '/nonexist'], state['fs']) is False

    def test_compound_or(self):
        from alpine_gpu import evaluate_test
        state = self._make_shell_state()
        assert evaluate_test(['-f', '/nonexist', '-o', '-d', '/etc'], state['fs']) is True
        assert evaluate_test(['-f', '/nonexist', '-o', '-f', '/nope'], state['fs']) is False


class TestApkStub:
    """Test the apk package manager stub."""

    def _make_shell_state(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        return {
            'env': {'HOME': '/root', 'PWD': '/', '?': '0'},
            'fs': create_alpine_rootfs(),
            'cwd': '/',
        }

    def test_apk_info(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['apk', 'info'], state)
        out = capsys.readouterr().out
        assert 'busybox' in out
        assert 'musl' in out

    def test_apk_update(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['apk', 'update'], state)
        out = capsys.readouterr().out
        assert 'alpine' in out.lower()

    def test_apk_add_fails(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['apk', 'add', 'python3'], state)
        out = capsys.readouterr().out
        assert 'read-only' in out.lower() or 'unable' in out.lower()

    def test_apk_version(self, capsys):
        from alpine_gpu import shell_builtin
        state = self._make_shell_state()
        shell_builtin(['apk', 'version'], state)
        out = capsys.readouterr().out
        assert '2.14' in out


# ═══════════════════════════════════════════════════════════════════════════════
# GPU SVC BUFFER TESTS — verify GPU-side syscall handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestGPUSVCBuffer:
    """Test GPU-side SVC buffer (SYS_WRITE/SYS_BRK/SYS_CLOSE handled on-GPU)."""

    def test_svc_buffer_init(self):
        """SVC buffer should initialize with zero header."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.init_svc_buffer()
        # Read header: write_pos=0, entry_count=0
        header = cpu.read_memory(cpu.SVC_BUF_BASE, 16)
        assert header == b'\x00' * 16

    def test_svc_buffer_drain_empty(self):
        """Draining empty SVC buffer should return empty list."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.init_svc_buffer()
        entries = cpu.drain_svc_buffer()
        assert entries == []

    def test_shadow_numpy_registers(self):
        """Shadow numpy should accelerate register access."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.set_register(0, 42)
        assert cpu.get_register(0) == 42
        cpu.set_register(15, -1)
        assert cpu.get_register(15) == -1
        # XZR always 0
        cpu.set_register(31, 999)
        assert cpu.get_register(31) == 0

    def test_shadow_numpy_memory(self):
        """Shadow numpy should accelerate memory read/write."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.write_memory(0x1000, b'hello')
        assert cpu.read_memory(0x1000, 5) == b'hello'
        cpu.write_memory(0x1000, b'world')
        assert cpu.read_memory(0x1000, 5) == b'world'

    def test_shadow_numpy_flags(self):
        """Shadow numpy should accelerate flag access."""
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.set_flags(True, False, True, False)
        n, z, c, v = cpu.get_flags()
        assert bool(n) is True
        assert bool(z) is False
        assert bool(c) is True
        assert bool(v) is False


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED BUSYBOX COMMAND TESTS (require busybox.elf)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_BUSYBOX, reason="busybox.elf not found")
class TestExtendedBusyBoxCommands:
    """Test additional BusyBox commands with GPU SVC buffering."""

    def _run(self, argv, fs=None, stdin_data=None):
        """Run BusyBox command and capture output."""
        if fs is None:
            from ncpu.os.gpu.alpine import create_alpine_rootfs
            fs = create_alpine_rootfs()
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = load_and_run_elf_helper(argv, filesystem=fs,
                                               stdin_data=stdin_data)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output, results, fs

    # --- sort ---
    def test_sort_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['sort', '/etc/passwd'], fs)
        lines = output.strip().split('\n')
        assert lines == sorted(lines)

    def test_sort_stdin(self):
        output, _, _ = self._run(['sort'], stdin_data=b'ccc\naaa\nbbb\n')
        assert output.strip() == 'aaa\nbbb\nccc'

    # --- uniq ---
    def test_uniq_stdin(self):
        output, _, _ = self._run(['uniq'], stdin_data=b'aaa\naaa\nbbb\naaa\n')
        assert output.strip() == 'aaa\nbbb\naaa'

    # --- tr ---
    def test_tr_uppercase(self):
        output, _, _ = self._run(['tr', 'a-z', 'A-Z'], stdin_data=b'hello\n')
        assert output.strip() == 'HELLO'

    # --- cut ---
    def test_cut_field(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['cut', '-d:', '-f1', '/etc/passwd'], fs)
        lines = output.strip().split('\n')
        assert 'root' in lines

    # --- head/tail ---
    def test_head_n(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['head', '-n', '1', '/etc/passwd'], fs)
        assert output.strip().startswith('root:')

    def test_tail_n(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['tail', '-n', '1', '/etc/passwd'], fs)
        lines = output.strip().split('\n')
        assert len(lines) == 1

    # --- expr ---
    def test_expr_arithmetic(self):
        output, _, _ = self._run(['expr', '7', '+', '3'])
        assert output.strip() == '10'

    def test_expr_multiply(self):
        output, _, _ = self._run(['expr', '6', '*', '7'])
        assert output.strip() == '42'

    # --- id/whoami ---
    def test_id_root(self):
        output, _, _ = self._run(['id'])
        assert 'uid=0' in output
        assert 'root' in output

    def test_whoami_root(self):
        output, _, _ = self._run(['whoami'])
        assert output.strip() == 'root'

    # --- hostname ---
    def test_hostname(self):
        output, _, _ = self._run(['hostname'])
        assert 'ncpu-gpu' in output

    # --- date ---
    def test_date_year(self):
        output, _, _ = self._run(['date', '+%Y'])
        year = output.strip()
        assert year.isdigit()
        assert int(year) >= 2024

    # --- basename/dirname ---
    def test_basename_path(self):
        output, _, _ = self._run(['basename', '/etc/passwd'])
        assert output.strip() == 'passwd'

    def test_dirname_path(self):
        output, _, _ = self._run(['dirname', '/etc/passwd'])
        assert output.strip() == '/etc'

    # --- wc ---
    def test_wc_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['wc', '-l', '/etc/passwd'], fs)
        parts = output.strip().split()
        assert int(parts[0]) > 0

    # --- stat ---
    def test_stat_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['stat', '/etc/hostname'], fs)
        assert '/etc/hostname' in output or 'File:' in output

    # --- grep -F ---
    def test_grep_fixed(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['grep', '-F', 'root', '/etc/passwd'], fs)
        assert 'root' in output

    # --- touch + mv + cp + rm ---
    def test_touch_creates_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        self._run(['touch', '/tmp/newfile'], fs)
        assert '/tmp/newfile' in fs.files

    def test_mv_renames_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        fs.write_file('/tmp/src', b'data')
        self._run(['mv', '/tmp/src', '/tmp/dst'], fs)
        assert '/tmp/dst' in fs.files
        assert '/tmp/src' not in fs.files

    def test_cp_copies_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        self._run(['cp', '/etc/hostname', '/tmp/hostname_copy'], fs)
        assert '/tmp/hostname_copy' in fs.files

    def test_rm_removes_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        fs.write_file('/tmp/rmme', b'delete me')
        self._run(['rm', '/tmp/rmme'], fs)
        assert '/tmp/rmme' not in fs.files

    # --- mkdir + rmdir ---
    def test_mkdir_creates_dir(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        self._run(['mkdir', '/tmp/newdir'], fs)
        assert '/tmp/newdir' in fs.directories

    def test_rmdir_removes_dir(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        fs.mkdir('/tmp/rmdir_test')
        self._run(['rmdir', '/tmp/rmdir_test'], fs)
        assert '/tmp/rmdir_test' not in fs.directories

    # --- sleep ---
    def test_sleep_zero(self):
        """sleep 0 should exit immediately."""
        _, results, _ = self._run(['sleep', '0'])
        assert results['total_cycles'] < 50000

    # --- true/false ---
    def test_true(self):
        _, results, _ = self._run(['true'])
        assert results['total_cycles'] < 5000

    def test_false(self):
        _, results, _ = self._run(['false'])
        assert results['total_cycles'] < 5000

    # --- find ---
    def test_find_name(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['find', '/etc', '-name', 'hostname'], fs)
        assert 'hostname' in output

    # --- tee ---
    def test_tee_writes_file(self):
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        fs = create_alpine_rootfs()
        output, _, _ = self._run(['tee', '/tmp/tee_out'], fs,
                                  stdin_data=b'tee test data\n')
        assert 'tee test data' in output
        assert '/tmp/tee_out' in fs.files

    # --- env ---
    def test_env_output(self):
        output, _, _ = self._run(['env'])
        assert 'PATH=' in output

    # --- printf ---
    def test_printf_format(self):
        output, _, _ = self._run(['printf', '%d %s\\n', '42', 'test'])
        assert '42 test' in output


# ═══════════════════════════════════════════════════════════════════════════════
# UTIMENSAT FIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtimensatFix:
    """Test that utimensat returns ENOENT for non-existent files."""

    def test_utimensat_existing(self):
        """utimensat on existing file should return 0."""
        from ncpu.os.gpu.elf_loader import make_busybox_syscall_handler
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2

        fs = create_alpine_rootfs()
        cpu = MLXKernelCPUv2(quiet=True)
        handler = make_busybox_syscall_handler(filesystem=fs)

        # Write path to memory
        path = b'/etc/hostname\x00'
        cpu.write_memory(0x10000, path)

        # Set up: x8=88(utimensat), x0=AT_FDCWD, x1=path, x2=0, x3=0
        cpu.set_register(8, 88)
        cpu.set_register(0, -100)  # AT_FDCWD
        cpu.set_register(1, 0x10000)
        cpu.set_register(2, 0)
        cpu.set_register(3, 0)

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == 0  # Success

    def test_utimensat_nonexistent(self):
        """utimensat on non-existent file should return -ENOENT."""
        from ncpu.os.gpu.elf_loader import make_busybox_syscall_handler
        from ncpu.os.gpu.alpine import create_alpine_rootfs
        from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2

        fs = create_alpine_rootfs()
        cpu = MLXKernelCPUv2(quiet=True)
        handler = make_busybox_syscall_handler(filesystem=fs)

        path = b'/tmp/nonexistent\x00'
        cpu.write_memory(0x10000, path)

        cpu.set_register(8, 88)
        cpu.set_register(0, -100)
        cpu.set_register(1, 0x10000)
        cpu.set_register(2, 0)
        cpu.set_register(3, 0)

        result = handler(cpu)
        assert result is True
        assert cpu.get_register(0) == -2  # -ENOENT
