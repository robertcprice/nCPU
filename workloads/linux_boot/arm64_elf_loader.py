#!/usr/bin/env python3
"""
ARM64 ELF Loader for Neural CPU
================================

Loads ARM64 ELF binaries and runs them on the Neural ARM64 CPU.
Provides syscall emulation for basic Linux/OS operations.

Usage:
    from arm64_elf_loader import ARM64ELFLoader
    from neural_cpu import NeuralCPU

    cpu = NeuralCPU()
    loader = ARM64ELFLoader(cpu)
    loader.load_elf("my_arm64_binary")
    loader.run()
"""

import struct
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import IntEnum

# ELF Constants
class ELFType(IntEnum):
    ET_NONE = 0
    ET_REL = 1
    ET_EXEC = 2
    ET_DYN = 3
    ET_CORE = 4

class ELFMachine(IntEnum):
    EM_NONE = 0
    EM_AARCH64 = 183  # ARM64

class PHType(IntEnum):
    PT_NULL = 0
    PT_LOAD = 1
    PT_DYNAMIC = 2
    PT_INTERP = 3
    PT_NOTE = 4
    PT_SHLIB = 5
    PT_PHDR = 6
    PT_TLS = 7

class PHFlags(IntEnum):
    PF_X = 0x1  # Execute
    PF_W = 0x2  # Write
    PF_R = 0x4  # Read


@dataclass
class ELF64Header:
    """ELF64 File Header"""
    e_ident: bytes      # Magic number and info (16 bytes)
    e_type: int         # Object file type
    e_machine: int      # Architecture
    e_version: int      # Object file version
    e_entry: int        # Entry point virtual address
    e_phoff: int        # Program header table offset
    e_shoff: int        # Section header table offset
    e_flags: int        # Processor-specific flags
    e_ehsize: int       # ELF header size
    e_phentsize: int    # Program header entry size
    e_phnum: int        # Program header entry count
    e_shentsize: int    # Section header entry size
    e_shnum: int        # Section header entry count
    e_shstrndx: int     # Section name string table index


@dataclass
class ELF64ProgramHeader:
    """ELF64 Program Header"""
    p_type: int         # Segment type
    p_flags: int        # Segment flags
    p_offset: int       # File offset
    p_vaddr: int        # Virtual address
    p_paddr: int        # Physical address
    p_filesz: int       # Size in file
    p_memsz: int        # Size in memory
    p_align: int        # Alignment


class SyscallEmulator:
    """Emulates Linux syscalls for the Neural CPU.

    Provides basic syscall handling for running userspace programs.
    """

    # ARM64 Linux syscall numbers
    SYS_READ = 63
    SYS_WRITE = 64
    SYS_EXIT = 93
    SYS_EXIT_GROUP = 94
    SYS_BRK = 214
    SYS_MMAP = 222
    SYS_MUNMAP = 215
    SYS_GETUID = 174
    SYS_GETGID = 176
    SYS_GETEUID = 175
    SYS_GETEGID = 177
    SYS_GETPID = 172
    SYS_GETPPID = 173
    SYS_CLOCK_GETTIME = 113
    SYS_NANOSLEEP = 101

    def __init__(self, cpu, loader):
        self.cpu = cpu
        self.loader = loader
        self.brk = 0x10000000  # Heap start
        self.heap_end = self.brk
        self.stdout_buffer = []
        self.stdin_buffer = []

    def handle_syscall(self, syscall_num):
        """Handle a syscall based on X8 register value.

        Args:
            syscall_num: Syscall number (from X8)

        Returns:
            Syscall return value (to be placed in X0)
        """
        if syscall_num == self.SYS_EXIT:
            return self._sys_exit()
        elif syscall_num == self.SYS_EXIT_GROUP:
            return self._sys_exit()
        elif syscall_num == self.SYS_WRITE:
            return self._sys_write()
        elif syscall_num == self.SYS_READ:
            return self._sys_read()
        elif syscall_num == self.SYS_BRK:
            return self._sys_brk()
        elif syscall_num == self.SYS_MMAP:
            return self._sys_mmap()
        elif syscall_num == self.SYS_GETUID:
            return 1000  # Mock UID
        elif syscall_num == self.SYS_GETGID:
            return 1000  # Mock GID
        elif syscall_num == self.SYS_GETEUID:
            return 1000  # Mock EUID
        elif syscall_num == self.SYS_GETEGID:
            return 1000  # Mock EGID
        elif syscall_num == self.SYS_GETPID:
            return 1  # Mock PID
        elif syscall_num == self.SYS_GETPPID:
            return 0  # Mock PPID (init)
        elif syscall_num == self.SYS_CLOCK_GETTIME:
            return self._sys_clock_gettime()
        elif syscall_num == self.SYS_NANOSLEEP:
            return 0  # Success (no actual sleep)
        else:
            print(f"  ⚠️ Unknown syscall {syscall_num}")
            return -1  # ENOSYS

    def _sys_exit(self):
        """Handle exit/exit_group syscall."""
        exit_code = self.cpu.get_reg(0)
        print(f"  🛑 Program exited with code {exit_code}")
        self.loader.running = False
        return exit_code

    def _sys_write(self):
        """Handle write syscall.

        Args in registers:
            X0: fd (file descriptor)
            X1: buf (pointer to buffer)
            X2: count (number of bytes)
        """
        fd = self.cpu.get_reg(0)
        buf_ptr = self.cpu.get_reg(1)
        count = self.cpu.get_reg(2)

        if fd in [1, 2]:  # stdout or stderr
            # Read characters from neural memory
            output = ""
            for i in range(min(count, 1024)):  # Limit to 1KB
                char = self.cpu.load(buf_ptr + i) & 0xFF
                if char == 0:
                    break
                output += chr(char)

            print(output, end='')
            self.stdout_buffer.append(output)
            return len(output)

        return -1  # EBADF

    def _sys_read(self):
        """Handle read syscall."""
        fd = self.cpu.get_reg(0)
        buf_ptr = self.cpu.get_reg(1)
        count = self.cpu.get_reg(2)

        if fd == 0:  # stdin
            # For now, return 0 (EOF)
            return 0

        return -1  # EBADF

    def _sys_brk(self):
        """Handle brk syscall (heap management)."""
        new_brk = self.cpu.get_reg(0)

        if new_brk == 0:
            return self.brk
        elif new_brk > self.brk:
            self.heap_end = new_brk
            return new_brk

        return self.brk

    def _sys_mmap(self):
        """Handle mmap syscall (simplified)."""
        addr = self.cpu.get_reg(0)
        length = self.cpu.get_reg(1)
        # prot = self.cpu.get_reg(2)
        # flags = self.cpu.get_reg(3)

        # Simple allocation from heap
        if addr == 0:
            result = self.heap_end
            self.heap_end += length
            return result

        return addr

    def _sys_clock_gettime(self):
        """Handle clock_gettime syscall."""
        # clock_id = self.cpu.get_reg(0)
        tp = self.cpu.get_reg(1)

        # Store fake time (1000 seconds)
        import time
        t = int(time.time())
        self.cpu.store(tp, t)      # tv_sec
        self.cpu.store(tp + 8, 0)  # tv_nsec

        return 0


class ARM64ELFLoader:
    """Loads and runs ARM64 ELF binaries on the Neural CPU.

    This loader handles:
    - Parsing ELF64 headers for ARM64
    - Loading program segments into neural memory
    - Setting up the initial stack and registers
    - Running the program with syscall emulation
    """

    def __init__(self, cpu):
        """Initialize the ELF loader.

        Args:
            cpu: NeuralCPU instance to run programs on
        """
        self.cpu = cpu
        self.syscall = SyscallEmulator(cpu, self)
        self.header: Optional[ELF64Header] = None
        self.program_headers: List[ELF64ProgramHeader] = []
        self.entry_point = 0
        self.running = False
        self.loaded_segments = []

        # Memory layout
        self.stack_top = 0x7FFFFF000000  # Top of stack
        self.stack_size = 0x100000       # 1MB stack

    def load_elf(self, filepath: str) -> bool:
        """Load an ELF binary from file.

        Args:
            filepath: Path to the ELF file

        Returns:
            True if loaded successfully
        """
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False

        with open(filepath, 'rb') as f:
            data = f.read()

        return self.load_elf_data(data)

    def load_elf_data(self, data: bytes) -> bool:
        """Load an ELF binary from bytes.

        Args:
            data: Raw ELF binary data

        Returns:
            True if loaded successfully
        """
        # Parse ELF header
        if len(data) < 64:
            print("❌ File too small for ELF header")
            return False

        # Check ELF magic
        if data[:4] != b'\x7fELF':
            print("❌ Not an ELF file")
            return False

        # Check 64-bit
        if data[4] != 2:
            print("❌ Not a 64-bit ELF")
            return False

        # Parse header
        self.header = self._parse_header(data)

        # Verify ARM64
        if self.header.e_machine != ELFMachine.EM_AARCH64:
            print(f"❌ Not an ARM64 binary (machine={self.header.e_machine})")
            return False

        # Parse program headers
        self._parse_program_headers(data)

        # Load segments into memory
        self._load_segments(data)

        # Set up entry point
        self.entry_point = self.header.e_entry

        print(f"✅ Loaded ARM64 ELF: entry=0x{self.entry_point:X}")
        print(f"   Type: {ELFType(self.header.e_type).name}")
        print(f"   Segments: {len(self.program_headers)}")

        return True

    def _parse_header(self, data: bytes) -> ELF64Header:
        """Parse ELF64 file header."""
        return ELF64Header(
            e_ident=data[:16],
            e_type=struct.unpack_from('<H', data, 16)[0],
            e_machine=struct.unpack_from('<H', data, 18)[0],
            e_version=struct.unpack_from('<I', data, 20)[0],
            e_entry=struct.unpack_from('<Q', data, 24)[0],
            e_phoff=struct.unpack_from('<Q', data, 32)[0],
            e_shoff=struct.unpack_from('<Q', data, 40)[0],
            e_flags=struct.unpack_from('<I', data, 48)[0],
            e_ehsize=struct.unpack_from('<H', data, 52)[0],
            e_phentsize=struct.unpack_from('<H', data, 54)[0],
            e_phnum=struct.unpack_from('<H', data, 56)[0],
            e_shentsize=struct.unpack_from('<H', data, 58)[0],
            e_shnum=struct.unpack_from('<H', data, 60)[0],
            e_shstrndx=struct.unpack_from('<H', data, 62)[0],
        )

    def _parse_program_headers(self, data: bytes):
        """Parse program headers."""
        self.program_headers = []

        offset = self.header.e_phoff
        for _ in range(self.header.e_phnum):
            ph = ELF64ProgramHeader(
                p_type=struct.unpack_from('<I', data, offset)[0],
                p_flags=struct.unpack_from('<I', data, offset + 4)[0],
                p_offset=struct.unpack_from('<Q', data, offset + 8)[0],
                p_vaddr=struct.unpack_from('<Q', data, offset + 16)[0],
                p_paddr=struct.unpack_from('<Q', data, offset + 24)[0],
                p_filesz=struct.unpack_from('<Q', data, offset + 32)[0],
                p_memsz=struct.unpack_from('<Q', data, offset + 40)[0],
                p_align=struct.unpack_from('<Q', data, offset + 48)[0],
            )
            self.program_headers.append(ph)
            offset += self.header.e_phentsize

    def _load_segments(self, data: bytes):
        """Load program segments into neural memory and instruction list."""
        self.instructions = []  # List of 32-bit ARM64 instructions
        self.instruction_base = None

        for ph in self.program_headers:
            if ph.p_type != PHType.PT_LOAD:
                continue

            # Load segment data
            seg_data = data[ph.p_offset:ph.p_offset + ph.p_filesz]

            # If executable, extract as instructions
            if ph.p_flags & PHFlags.PF_X:
                if self.instruction_base is None:
                    self.instruction_base = ph.p_vaddr

                # Extract 32-bit instructions
                for i in range(0, len(seg_data), 4):
                    chunk = seg_data[i:i+4]
                    if len(chunk) < 4:
                        chunk = chunk + b'\x00' * (4 - len(chunk))
                    inst = struct.unpack('<I', chunk)[0]
                    self.instructions.append(inst)

            # Also store in neural memory for data access (8 bytes at a time)
            addr = ph.p_vaddr
            for i in range(0, len(seg_data), 8):
                chunk = seg_data[i:i+8]
                if len(chunk) < 8:
                    chunk = chunk + b'\x00' * (8 - len(chunk))
                val = struct.unpack('<Q', chunk)[0]
                self.cpu.store(addr + i, val)

            self.loaded_segments.append({
                'vaddr': ph.p_vaddr,
                'size': ph.p_memsz,
                'flags': ph.p_flags,
            })

            flags = ""
            if ph.p_flags & PHFlags.PF_R: flags += "R"
            if ph.p_flags & PHFlags.PF_W: flags += "W"
            if ph.p_flags & PHFlags.PF_X: flags += "X"

            print(f"   Loaded segment: 0x{ph.p_vaddr:X}-0x{ph.p_vaddr + ph.p_memsz:X} [{flags}]")
            if ph.p_flags & PHFlags.PF_X:
                print(f"   Extracted {len(self.instructions)} instructions")

    def setup_stack(self, argv: List[str] = None, envp: Dict[str, str] = None):
        """Set up the initial stack.

        Args:
            argv: Command line arguments
            envp: Environment variables
        """
        if argv is None:
            argv = ["program"]
        if envp is None:
            envp = {"HOME": "/", "PATH": "/bin"}

        sp = self.stack_top - 8

        # For simplicity, just set up basic stack
        # In a full implementation, we'd push:
        # - argc, argv pointers, NULL
        # - envp pointers, NULL
        # - auxv entries
        # - strings

        self.cpu.set_reg(29, sp)  # Frame pointer
        self.cpu.set_reg(31, 0)   # XZR (will read as 0)

        # Set stack pointer
        # Note: SP is separate from X31 in real ARM64
        # For our neural CPU, we'll track it differently
        self.cpu.neural_memory.sp.fill_(sp // 8 % self.cpu.neural_memory.memory_size)

        # Set argc
        self.cpu.set_reg(0, len(argv))

        print(f"   Stack set up at 0x{sp:X}")

    def run(self, max_instructions: int = 100000, verbose: bool = False) -> int:
        """Run the loaded program.

        Args:
            max_instructions: Maximum instructions to execute
            verbose: Print each instruction as it executes

        Returns:
            Exit code
        """
        if self.header is None:
            print("❌ No program loaded")
            return -1

        if not self.instructions:
            print("❌ No executable instructions found")
            return -1

        # Set up stack
        self.setup_stack()

        # Use CPU's run_program for simple execution
        # This uses the instruction list directly
        print(f"\n🚀 Starting execution")
        print(f"   Entry point: 0x{self.entry_point:X}")
        print(f"   Instructions: {len(self.instructions)}")
        print("="*60)

        self.running = True
        self.cpu.pc = 0  # Start at beginning of instruction list

        # For simple programs without syscalls, just use run_program
        # For complex programs with syscalls, we need manual execution
        if len(self.instructions) <= 10:
            # Simple program - use direct execution with verbose output
            for i, inst in enumerate(self.instructions):
                if verbose or len(self.instructions) <= 10:
                    print(f"  [{i}] 0x{inst:08X}")

                # Check for RET (0xD65F03C0)
                if inst == 0xD65F03C0:
                    print(f"  RET - returning with X0={self.cpu.get_reg(0)}")
                    break

                # Check for SVC
                if (inst & 0xFFE0001F) == 0xD4000001:
                    syscall_num = self.cpu.get_reg(8)
                    result = self.syscall.handle_syscall(syscall_num)
                    self.cpu.set_reg(0, result)
                    if not self.running:
                        break
                    continue

                # Execute instruction
                self.cpu.execute_instruction(inst)

        else:
            # Use run_program for larger programs
            executed = self.cpu.run_program(self.instructions, max_instructions)
            print(f"  Executed {executed} instructions")

        print("="*60)
        result = self.cpu.get_reg(0)
        print(f"📊 Execution complete")
        print(f"   Return value (X0): {result}")

        return result


def create_simple_arm64_program():
    """Create a simple ARM64 program for testing.

    This creates a minimal ELF with:
    - MOV X0, #42
    - RET
    """
    # ARM64 instructions:
    # MOV X0, #42  -> 0xD2800540
    # RET          -> 0xD65F03C0

    code = struct.pack('<II', 0xD2800540, 0xD65F03C0)

    # Build minimal ELF
    entry = 0x400000

    # ELF header
    elf_header = bytearray(64)
    elf_header[0:4] = b'\x7fELF'  # Magic
    elf_header[4] = 2             # 64-bit
    elf_header[5] = 1             # Little endian
    elf_header[6] = 1             # ELF version
    struct.pack_into('<H', elf_header, 16, ELFType.ET_EXEC)
    struct.pack_into('<H', elf_header, 18, ELFMachine.EM_AARCH64)
    struct.pack_into('<I', elf_header, 20, 1)  # Version
    struct.pack_into('<Q', elf_header, 24, entry)  # Entry point
    struct.pack_into('<Q', elf_header, 32, 64)  # Program header offset
    struct.pack_into('<H', elf_header, 52, 64)  # Header size
    struct.pack_into('<H', elf_header, 54, 56)  # Program header entry size
    struct.pack_into('<H', elf_header, 56, 1)   # Number of program headers

    # Program header
    prog_header = bytearray(56)
    struct.pack_into('<I', prog_header, 0, PHType.PT_LOAD)
    struct.pack_into('<I', prog_header, 4, PHFlags.PF_R | PHFlags.PF_X)
    struct.pack_into('<Q', prog_header, 8, 120)   # File offset
    struct.pack_into('<Q', prog_header, 16, entry)  # Virtual address
    struct.pack_into('<Q', prog_header, 24, entry)  # Physical address
    struct.pack_into('<Q', prog_header, 32, len(code))  # File size
    struct.pack_into('<Q', prog_header, 40, len(code))  # Memory size
    struct.pack_into('<Q', prog_header, 48, 0x1000)     # Alignment

    return bytes(elf_header) + bytes(prog_header) + code


if __name__ == "__main__":
    from neural_cpu import NeuralCPU

    print("="*70)
    print("🧪 ARM64 ELF LOADER TEST")
    print("="*70)

    # Create CPU
    cpu = NeuralCPU(quiet=True)

    # Create loader
    loader = ARM64ELFLoader(cpu)

    # Test with simple program
    print("\n📝 Creating simple ARM64 test program...")
    print("   Code: MOV X0, #42; RET")

    elf_data = create_simple_arm64_program()

    print(f"\n📦 Loading ELF ({len(elf_data)} bytes)...")

    if loader.load_elf_data(elf_data):
        # Run the program
        exit_code = loader.run(max_instructions=1000)
        print(f"\n✅ Program returned: {exit_code}")
    else:
        print("❌ Failed to load ELF")
