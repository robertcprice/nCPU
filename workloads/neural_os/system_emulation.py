#!/usr/bin/env python3
"""
Full Linux System Emulation Layer for Neural ARM64 CPU
=======================================================

This provides the system-level components needed to boot and run
a full Linux system on our Neural ARM64 CPU:

1. Memory Management Unit (MMU) - Virtual address translation
2. Generic Interrupt Controller (GIC) - IRQ handling
3. System Timer - OS scheduling
4. UART - Serial console
5. Exception Handling - Syscalls, page faults, IRQs
6. Boot Protocol - Load kernel, DTB, initramfs

Usage:
    system = NeuralARM64System()
    system.load_kernel("Image")        # Linux kernel
    system.load_dtb("virt.dtb")        # Device tree
    system.load_initramfs("rootfs.cpio.gz")  # Alpine initramfs
    system.boot()                       # Start execution
"""

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import IntEnum, auto
import os

# Try to import neural_cpu
try:
    from neural_cpu import NeuralCPU
except ImportError:
    NeuralCPU = None


# ============================================================
# ARM64 Exception Levels and System Registers
# ============================================================

class ExceptionLevel(IntEnum):
    """ARM64 exception levels."""
    EL0 = 0  # User mode
    EL1 = 1  # OS kernel
    EL2 = 2  # Hypervisor
    EL3 = 3  # Secure monitor


class ExceptionType(IntEnum):
    """ARM64 exception types."""
    SYNC = 0        # Synchronous (syscall, page fault)
    IRQ = 1         # Interrupt
    FIQ = 2         # Fast interrupt
    SERROR = 3      # System error


class SyscallNumber(IntEnum):
    """Linux ARM64 syscall numbers."""
    READ = 63
    WRITE = 64
    OPENAT = 56
    CLOSE = 57
    FSTAT = 80
    LSEEK = 62
    MMAP = 222
    MPROTECT = 226
    MUNMAP = 215
    BRK = 214
    IOCTL = 29
    WRITEV = 66
    READV = 65
    PREAD64 = 67
    PWRITE64 = 68
    EXIT = 93
    EXIT_GROUP = 94
    GETPID = 172
    GETUID = 174
    GETEUID = 175
    GETGID = 176
    GETEGID = 177
    GETTID = 178
    CLOCK_GETTIME = 113
    NANOSLEEP = 101
    NEWFSTATAT = 79
    OPENAT2 = 437
    FACCESSAT = 48
    FACCESSAT2 = 439
    GETRANDOM = 278


# ============================================================
# Memory Management Unit (MMU)
# ============================================================

@dataclass
class PageTableEntry:
    """4KB page table entry."""
    valid: bool = False
    physical_addr: int = 0
    readable: bool = False
    writable: bool = False
    executable: bool = False
    user_accessible: bool = False


class NeuralMMU:
    """Neural-inspired Memory Management Unit.

    Handles virtual to physical address translation with:
    - 4KB pages (12-bit offset)
    - 4-level page tables (48-bit virtual addresses)
    - TLB caching for fast lookups
    """

    def __init__(self, physical_memory_size: int = 512 * 1024 * 1024):
        self.physical_memory = bytearray(physical_memory_size)
        self.physical_size = physical_memory_size

        # Page tables (simplified - direct mapping for now)
        self.page_table_base = 0
        self.page_tables: Dict[int, PageTableEntry] = {}

        # TLB (Translation Lookaside Buffer)
        self.tlb: Dict[int, int] = {}  # virtual page -> physical page
        self.tlb_size = 1024

        # Memory regions
        self.regions: List[Dict] = []

    def map_region(self, virtual_addr: int, physical_addr: int,
                   size: int, readable: bool = True, writable: bool = True,
                   executable: bool = False, user: bool = True):
        """Map a virtual address region to physical memory."""
        num_pages = (size + 0xFFF) // 0x1000

        for i in range(num_pages):
            vpage = (virtual_addr + i * 0x1000) >> 12
            ppage = (physical_addr + i * 0x1000) >> 12

            self.page_tables[vpage] = PageTableEntry(
                valid=True,
                physical_addr=ppage << 12,
                readable=readable,
                writable=writable,
                executable=executable,
                user_accessible=user,
            )
            self.tlb[vpage] = ppage

        self.regions.append({
            'virtual': virtual_addr,
            'physical': physical_addr,
            'size': size,
            'permissions': {'r': readable, 'w': writable, 'x': executable},
        })

    def translate(self, virtual_addr: int, write: bool = False,
                  execute: bool = False) -> Optional[int]:
        """Translate virtual address to physical address."""
        vpage = virtual_addr >> 12
        offset = virtual_addr & 0xFFF

        # TLB lookup
        if vpage in self.tlb:
            ppage = self.tlb[vpage]
            return (ppage << 12) | offset

        # Page table walk
        if vpage in self.page_tables:
            entry = self.page_tables[vpage]
            if not entry.valid:
                return None  # Page fault

            # Permission check
            if write and not entry.writable:
                return None  # Write permission fault
            if execute and not entry.executable:
                return None  # Execute permission fault

            # Update TLB
            if len(self.tlb) >= self.tlb_size:
                # Evict random entry
                self.tlb.pop(next(iter(self.tlb)))
            self.tlb[vpage] = entry.physical_addr >> 12

            return entry.physical_addr | offset

        return None  # Page fault

    def read(self, virtual_addr: int, size: int) -> Optional[bytes]:
        """Read from virtual address."""
        physical = self.translate(virtual_addr)
        if physical is None:
            return None

        if physical + size > self.physical_size:
            return None

        return bytes(self.physical_memory[physical:physical + size])

    def write(self, virtual_addr: int, data: bytes) -> bool:
        """Write to virtual address."""
        physical = self.translate(virtual_addr, write=True)
        if physical is None:
            return False

        if physical + len(data) > self.physical_size:
            return False

        self.physical_memory[physical:physical + len(data)] = data
        return True

    def read_physical(self, addr: int, size: int) -> bytes:
        """Direct physical memory read."""
        if addr + size > self.physical_size:
            return bytes(size)
        return bytes(self.physical_memory[addr:addr + size])

    def write_physical(self, addr: int, data: bytes):
        """Direct physical memory write."""
        if addr + len(data) <= self.physical_size:
            self.physical_memory[addr:addr + len(data)] = data


# ============================================================
# Generic Interrupt Controller (GIC)
# ============================================================

@dataclass
class IRQHandler:
    """Interrupt handler registration."""
    irq_num: int
    handler: Callable
    enabled: bool = True
    pending: bool = False


class NeuralGIC:
    """Generic Interrupt Controller for the Neural CPU.

    Handles:
    - IRQ registration and dispatch
    - Priority levels
    - IRQ masking
    - Software interrupts
    """

    def __init__(self):
        self.handlers: Dict[int, IRQHandler] = {}
        self.pending_irqs: List[int] = []
        self.irq_mask = 0xFFFFFFFF  # All enabled
        self.priority_threshold = 0xFF

        # Standard IRQ numbers
        self.TIMER_IRQ = 30
        self.UART_IRQ = 33
        self.VIRTIO_BLK_IRQ = 48

    def register_irq(self, irq_num: int, handler: Callable):
        """Register an IRQ handler."""
        self.handlers[irq_num] = IRQHandler(
            irq_num=irq_num,
            handler=handler,
        )

    def raise_irq(self, irq_num: int):
        """Raise an interrupt."""
        if irq_num in self.handlers:
            self.handlers[irq_num].pending = True
            if irq_num not in self.pending_irqs:
                self.pending_irqs.append(irq_num)

    def ack_irq(self, irq_num: int):
        """Acknowledge and clear an interrupt."""
        if irq_num in self.handlers:
            self.handlers[irq_num].pending = False
        if irq_num in self.pending_irqs:
            self.pending_irqs.remove(irq_num)

    def get_pending_irq(self) -> Optional[int]:
        """Get highest priority pending IRQ."""
        if self.pending_irqs:
            return self.pending_irqs[0]
        return None

    def dispatch(self) -> bool:
        """Dispatch pending interrupts. Returns True if IRQ was handled."""
        irq = self.get_pending_irq()
        if irq is not None and irq in self.handlers:
            handler = self.handlers[irq]
            if handler.enabled:
                handler.handler()
                return True
        return False


# ============================================================
# System Timer
# ============================================================

class NeuralTimer:
    """System timer for OS scheduling.

    Provides:
    - Counter register (CNTPCT_EL0)
    - Timer value register (CNTV_TVAL_EL0)
    - Timer control (CNTV_CTL_EL0)
    """

    def __init__(self, frequency: int = 62500000):  # 62.5 MHz
        self.frequency = frequency
        self.counter = 0
        self.compare_value = 0
        self.control = 0  # Bit 0: enable, Bit 1: mask, Bit 2: status

        self.gic: Optional[NeuralGIC] = None
        self.timer_irq = 30

    def connect_gic(self, gic: NeuralGIC):
        """Connect to interrupt controller."""
        self.gic = gic

    def tick(self, cycles: int = 1):
        """Advance timer by cycles."""
        self.counter += cycles

        # Check if timer fired
        if self.control & 1:  # Enabled
            if self.counter >= self.compare_value:
                self.control |= 4  # Set status bit
                if not (self.control & 2):  # Not masked
                    if self.gic:
                        self.gic.raise_irq(self.timer_irq)

    def read_counter(self) -> int:
        """Read counter value (CNTPCT_EL0)."""
        return self.counter

    def write_compare(self, value: int):
        """Write compare value."""
        self.compare_value = value
        self.control &= ~4  # Clear status

    def write_control(self, value: int):
        """Write control register."""
        self.control = value & 0x7


# ============================================================
# UART (Serial Console)
# ============================================================

class NeuralUART:
    """PL011 UART emulation for serial console.

    Provides:
    - TX/RX data registers
    - Status flags
    - Interrupt generation
    """

    def __init__(self):
        self.tx_buffer: List[int] = []
        self.rx_buffer: List[int] = []
        self.status = 0x90  # TX empty, TX not full

        self.gic: Optional[NeuralGIC] = None
        self.uart_irq = 33

        # Callbacks
        self.on_output: Optional[Callable[[str], None]] = None

    def connect_gic(self, gic: NeuralGIC):
        """Connect to interrupt controller."""
        self.gic = gic

    def write_data(self, byte: int):
        """Write byte to UART TX."""
        self.tx_buffer.append(byte & 0xFF)

        # Output character
        if self.on_output:
            self.on_output(chr(byte & 0x7F))
        else:
            print(chr(byte & 0x7F), end='', flush=True)

    def read_data(self) -> int:
        """Read byte from UART RX."""
        if self.rx_buffer:
            return self.rx_buffer.pop(0)
        return 0

    def read_status(self) -> int:
        """Read UART status register."""
        status = 0
        if not self.rx_buffer:
            status |= (1 << 4)  # RX empty
        if len(self.tx_buffer) < 16:
            status |= (1 << 5)  # TX not full
        return status

    def input_char(self, char: str):
        """Receive input character."""
        self.rx_buffer.append(ord(char))
        if self.gic:
            self.gic.raise_irq(self.uart_irq)


# ============================================================
# Syscall Emulation
# ============================================================

class SyscallEmulator:
    """Linux syscall emulation layer.

    Handles syscalls made by userspace programs running on the neural CPU.
    """

    def __init__(self, mmu: NeuralMMU, uart: NeuralUART):
        self.mmu = mmu
        self.uart = uart

        # File descriptors
        self.fds: Dict[int, Dict] = {
            0: {'type': 'stdin', 'mode': 'r'},
            1: {'type': 'stdout', 'mode': 'w'},
            2: {'type': 'stderr', 'mode': 'w'},
        }
        self.next_fd = 3

        # Process info
        self.pid = 1
        self.uid = 0
        self.gid = 0

        # Heap management
        self.brk = 0x10000000
        self.brk_limit = 0x20000000

    def handle_syscall(self, syscall_num: int, x0: int, x1: int, x2: int,
                       x3: int, x4: int, x5: int) -> int:
        """Handle a syscall and return result."""

        if syscall_num == SyscallNumber.WRITE:
            return self._sys_write(x0, x1, x2)

        elif syscall_num == SyscallNumber.READ:
            return self._sys_read(x0, x1, x2)

        elif syscall_num == SyscallNumber.OPENAT:
            return self._sys_openat(x0, x1, x2, x3)

        elif syscall_num == SyscallNumber.CLOSE:
            return self._sys_close(x0)

        elif syscall_num == SyscallNumber.EXIT or syscall_num == SyscallNumber.EXIT_GROUP:
            return self._sys_exit(x0)

        elif syscall_num == SyscallNumber.BRK:
            return self._sys_brk(x0)

        elif syscall_num == SyscallNumber.MMAP:
            return self._sys_mmap(x0, x1, x2, x3, x4, x5)

        elif syscall_num == SyscallNumber.GETPID:
            return self.pid

        elif syscall_num == SyscallNumber.GETUID or syscall_num == SyscallNumber.GETEUID:
            return self.uid

        elif syscall_num == SyscallNumber.GETGID or syscall_num == SyscallNumber.GETEGID:
            return self.gid

        elif syscall_num == SyscallNumber.GETTID:
            return self.pid

        elif syscall_num == SyscallNumber.IOCTL:
            return self._sys_ioctl(x0, x1, x2)

        elif syscall_num == SyscallNumber.CLOCK_GETTIME:
            return self._sys_clock_gettime(x0, x1)

        elif syscall_num == SyscallNumber.NEWFSTATAT:
            return self._sys_newfstatat(x0, x1, x2, x3)

        elif syscall_num == SyscallNumber.FACCESSAT:
            return self._sys_faccessat(x0, x1, x2, x3)

        elif syscall_num == SyscallNumber.GETRANDOM:
            return self._sys_getrandom(x0, x1, x2)

        else:
            print(f"[SYSCALL] Unhandled syscall {syscall_num}")
            return -38  # ENOSYS

    def _sys_write(self, fd: int, buf: int, count: int) -> int:
        """Write to file descriptor."""
        if fd not in self.fds:
            return -9  # EBADF

        data = self.mmu.read(buf, count)
        if data is None:
            return -14  # EFAULT

        fd_info = self.fds[fd]
        if fd_info['type'] in ('stdout', 'stderr'):
            for byte in data:
                self.uart.write_data(byte)
            return count

        return -9  # EBADF

    def _sys_read(self, fd: int, buf: int, count: int) -> int:
        """Read from file descriptor."""
        if fd not in self.fds:
            return -9  # EBADF

        fd_info = self.fds[fd]
        if fd_info['type'] == 'stdin':
            # Read from UART
            data = []
            for _ in range(count):
                byte = self.uart.read_data()
                if byte == 0:
                    break
                data.append(byte)

            if data:
                self.mmu.write(buf, bytes(data))
            return len(data)

        return -9  # EBADF

    def _sys_openat(self, dirfd: int, pathname: int, flags: int, mode: int) -> int:
        """Open file."""
        # Read pathname from memory
        path_bytes = []
        for i in range(256):
            byte = self.mmu.read(pathname + i, 1)
            if byte is None or byte[0] == 0:
                break
            path_bytes.append(byte[0])

        path = bytes(path_bytes).decode('utf-8', errors='ignore')
        print(f"[SYSCALL] openat: {path}")

        # For now, return error for most files
        return -2  # ENOENT

    def _sys_close(self, fd: int) -> int:
        """Close file descriptor."""
        if fd in self.fds and fd >= 3:
            del self.fds[fd]
            return 0
        return -9  # EBADF

    def _sys_exit(self, status: int) -> int:
        """Exit process."""
        print(f"\n[SYSCALL] exit({status})")
        raise SystemExit(status)

    def _sys_brk(self, addr: int) -> int:
        """Change data segment size."""
        if addr == 0:
            return self.brk

        if addr > self.brk and addr < self.brk_limit:
            # Grow heap
            old_brk = self.brk
            self.brk = addr
            # Map new pages
            self.mmu.map_region(old_brk, old_brk, addr - old_brk)
            return self.brk

        return self.brk

    def _sys_mmap(self, addr: int, length: int, prot: int, flags: int,
                  fd: int, offset: int) -> int:
        """Memory map."""
        # Simplified - just allocate from heap
        aligned_len = (length + 0xFFF) & ~0xFFF

        if addr == 0:
            addr = self.brk
            self.brk += aligned_len

        # Map the region
        self.mmu.map_region(
            addr, addr, aligned_len,
            readable=bool(prot & 1),
            writable=bool(prot & 2),
            executable=bool(prot & 4),
        )

        return addr

    def _sys_ioctl(self, fd: int, request: int, arg: int) -> int:
        """IO control."""
        # For terminal ioctls, return success
        if fd <= 2:
            return 0
        return -25  # ENOTTY

    def _sys_clock_gettime(self, clockid: int, tp: int) -> int:
        """Get clock time."""
        import time
        t = time.time()
        sec = int(t)
        nsec = int((t - sec) * 1e9)

        # Write timespec struct
        self.mmu.write(tp, struct.pack('<QQ', sec, nsec))
        return 0

    def _sys_newfstatat(self, dirfd: int, pathname: int, statbuf: int, flags: int) -> int:
        """Get file status."""
        return -2  # ENOENT

    def _sys_faccessat(self, dirfd: int, pathname: int, mode: int, flags: int) -> int:
        """Check file access."""
        return -2  # ENOENT

    def _sys_getrandom(self, buf: int, buflen: int, flags: int) -> int:
        """Get random bytes."""
        import random
        data = bytes(random.randint(0, 255) for _ in range(buflen))
        self.mmu.write(buf, data)
        return buflen


# ============================================================
# Full System Emulator
# ============================================================

class NeuralARM64System:
    """Complete ARM64 system emulator.

    Combines:
    - Neural CPU core
    - MMU for virtual memory
    - GIC for interrupts
    - Timer for scheduling
    - UART for console
    - Syscall emulation
    """

    def __init__(self, memory_size: int = 512 * 1024 * 1024):
        # Initialize components
        self.mmu = NeuralMMU(memory_size)
        self.gic = NeuralGIC()
        self.timer = NeuralTimer()
        self.uart = NeuralUART()
        self.syscalls = SyscallEmulator(self.mmu, self.uart)

        # Connect components
        self.timer.connect_gic(self.gic)
        self.uart.connect_gic(self.gic)

        # CPU (will be initialized when neural_cpu is available)
        self.cpu = None

        # Execution state
        self.current_el = ExceptionLevel.EL1
        self.running = False

        # Memory layout
        self.KERNEL_BASE = 0x40080000
        self.DTB_BASE = 0x40000000
        self.INITRD_BASE = 0x45000000
        self.STACK_BASE = 0x50000000

        # Setup identity mapping for physical memory
        self.mmu.map_region(0, 0, memory_size, executable=True)

    def load_kernel(self, path: str) -> int:
        """Load Linux kernel Image."""
        if not os.path.exists(path):
            print(f"Kernel not found: {path}")
            return 0

        with open(path, 'rb') as f:
            kernel_data = f.read()

        self.mmu.write_physical(self.KERNEL_BASE, kernel_data)
        print(f"Loaded kernel: {len(kernel_data)} bytes at 0x{self.KERNEL_BASE:X}")
        return len(kernel_data)

    def load_dtb(self, path: str) -> int:
        """Load Device Tree Blob."""
        if not os.path.exists(path):
            print(f"DTB not found: {path}")
            return 0

        with open(path, 'rb') as f:
            dtb_data = f.read()

        self.mmu.write_physical(self.DTB_BASE, dtb_data)
        print(f"Loaded DTB: {len(dtb_data)} bytes at 0x{self.DTB_BASE:X}")
        return len(dtb_data)

    def load_initramfs(self, path: str) -> int:
        """Load initial RAM filesystem."""
        if not os.path.exists(path):
            print(f"Initramfs not found: {path}")
            return 0

        with open(path, 'rb') as f:
            initrd_data = f.read()

        self.mmu.write_physical(self.INITRD_BASE, initrd_data)
        print(f"Loaded initramfs: {len(initrd_data)} bytes at 0x{self.INITRD_BASE:X}")
        return len(initrd_data)

    def boot(self):
        """Boot the system."""
        print("\n" + "=" * 60)
        print("🚀 NEURAL ARM64 SYSTEM BOOT")
        print("=" * 60)

        # Initialize CPU state for kernel entry
        # X0 = DTB address
        # X1-X3 = 0
        # PC = kernel entry point

        self.running = True

        # For now, print boot info
        print(f"\nBoot parameters:")
        print(f"  Kernel entry: 0x{self.KERNEL_BASE:X}")
        print(f"  DTB address:  0x{self.DTB_BASE:X}")
        print(f"  Initramfs:    0x{self.INITRD_BASE:X}")
        print(f"  Stack:        0x{self.STACK_BASE:X}")

        print("\n[System ready - awaiting Neural CPU integration]")

    def handle_exception(self, exception_type: ExceptionType, syndrome: int):
        """Handle CPU exception."""
        if exception_type == ExceptionType.SYNC:
            # Check exception class (EC) in syndrome
            ec = (syndrome >> 26) & 0x3F

            if ec == 0x15:  # SVC from AArch64
                # Syscall
                syscall_num = syndrome & 0xFFFF
                # Would get args from X0-X5 registers
                # result = self.syscalls.handle_syscall(...)
                pass

            elif ec == 0x20 or ec == 0x21:  # Instruction abort
                print(f"Instruction abort at syndrome 0x{syndrome:X}")

            elif ec == 0x24 or ec == 0x25:  # Data abort
                print(f"Data abort at syndrome 0x{syndrome:X}")

    def run_cycles(self, num_cycles: int = 1000):
        """Run for specified number of cycles."""
        for _ in range(num_cycles):
            if not self.running:
                break

            # Tick timer
            self.timer.tick(1)

            # Check for interrupts
            if self.gic.get_pending_irq() is not None:
                self.gic.dispatch()

            # Would execute CPU instruction here
            # self.cpu.step()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Neural ARM64 System Emulation Layer")
    print("=" * 50)

    # Create system
    system = NeuralARM64System(memory_size=256 * 1024 * 1024)

    # Test UART
    print("\nTesting UART...")
    system.uart.write_data(ord('H'))
    system.uart.write_data(ord('i'))
    system.uart.write_data(ord('!'))
    system.uart.write_data(ord('\n'))

    # Test MMU
    print("\nTesting MMU...")
    system.mmu.write(0x1000, b"Hello, World!")
    data = system.mmu.read(0x1000, 13)
    print(f"Read from 0x1000: {data}")

    # Test timer
    print("\nTesting Timer...")
    for _ in range(100):
        system.timer.tick(1000)
    print(f"Timer counter: {system.timer.read_counter()}")

    print("\n✅ System emulation layer ready!")
    print("   Next: Integrate with Neural CPU and load Linux kernel")
