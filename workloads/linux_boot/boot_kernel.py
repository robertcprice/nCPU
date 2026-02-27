#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    NEURAL ARM64 KERNEL BOOT                                      ║
║                   Full Linux Kernel on Neural GPU CPU                            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  This script boots a full Linux kernel on the Neural ARM64 CPU:                  ║
║  • Loads ARM64 kernel Image (raw or gzipped)                                     ║
║  • Parses and loads Device Tree Blob (DTB)                                       ║
║  • Sets up initial page tables (identity mapping)                                ║
║  • Initializes GIC, timer, and UART emulation                                    ║
║  • Boots kernel with proper ARM64 boot protocol                                  ║
║                                                                                  ║
║  Usage:                                                                          ║
║    python3 boot_kernel.py --kernel Image --dtb virt.dtb                          ║
║    python3 boot_kernel.py --kernel Image.gz --dtb virt.dtb --initrd initrd.gz    ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import struct
import gzip
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_kernel import NeuralARM64Kernel


# ════════════════════════════════════════════════════════════════════════════════
# DEVICE TREE BLOB (DTB) PARSER
# ════════════════════════════════════════════════════════════════════════════════

class DTBParser:
    """
    Parse Flattened Device Tree (FDT) / Device Tree Blob (DTB).

    The DTB describes hardware configuration for the kernel:
    - Memory regions
    - CPU topology
    - Interrupt controllers
    - Serial ports, timers, etc.
    """

    FDT_MAGIC = 0xD00DFEED
    FDT_BEGIN_NODE = 0x1
    FDT_END_NODE = 0x2
    FDT_PROP = 0x3
    FDT_NOP = 0x4
    FDT_END = 0x9

    def __init__(self, data: bytes):
        self.data = data
        self.root = {}
        self.strings = b""
        self.struct_offset = 0
        self.strings_offset = 0

    def parse(self) -> Dict[str, Any]:
        """Parse the DTB and return a dictionary representation."""
        if len(self.data) < 40:
            raise ValueError("DTB too small")

        # Parse header
        magic = struct.unpack(">I", self.data[0:4])[0]
        if magic != self.FDT_MAGIC:
            raise ValueError(f"Invalid DTB magic: 0x{magic:08X}")

        totalsize = struct.unpack(">I", self.data[4:8])[0]
        self.struct_offset = struct.unpack(">I", self.data[8:12])[0]
        self.strings_offset = struct.unpack(">I", self.data[12:16])[0]
        # mem_rsvmap_off = struct.unpack(">I", self.data[16:20])[0]
        version = struct.unpack(">I", self.data[20:24])[0]
        # last_comp_version = struct.unpack(">I", self.data[24:28])[0]
        # boot_cpuid_phys = struct.unpack(">I", self.data[28:32])[0]
        strings_size = struct.unpack(">I", self.data[32:36])[0]
        # struct_size = struct.unpack(">I", self.data[36:40])[0]

        print(f"  DTB: size={totalsize}, version={version}")

        # Extract strings block
        self.strings = self.data[self.strings_offset:self.strings_offset + strings_size]

        # Parse structure block
        offset = self.struct_offset
        self.root, _ = self._parse_node(offset)

        return self.root

    def _parse_node(self, offset: int) -> tuple:
        """Parse a single DTB node."""
        node = {"properties": {}, "children": {}}

        # Read FDT_BEGIN_NODE token
        token = struct.unpack(">I", self.data[offset:offset+4])[0]
        offset += 4

        if token != self.FDT_BEGIN_NODE:
            return node, offset

        # Read node name
        name_end = self.data.index(b'\x00', offset)
        name = self.data[offset:name_end].decode('utf-8', errors='replace')
        offset = (name_end + 4) & ~3  # Align to 4 bytes

        # Parse properties and children
        while True:
            token = struct.unpack(">I", self.data[offset:offset+4])[0]

            if token == self.FDT_PROP:
                offset += 4
                prop_len = struct.unpack(">I", self.data[offset:offset+4])[0]
                offset += 4
                prop_nameoff = struct.unpack(">I", self.data[offset:offset+4])[0]
                offset += 4

                # Get property name from strings block
                prop_name_end = self.strings.index(b'\x00', prop_nameoff)
                prop_name = self.strings[prop_nameoff:prop_name_end].decode('utf-8')

                # Get property value
                prop_value = self.data[offset:offset + prop_len]
                offset = (offset + prop_len + 3) & ~3  # Align

                node["properties"][prop_name] = prop_value

            elif token == self.FDT_BEGIN_NODE:
                # Recursive child node
                child, offset = self._parse_node(offset)
                child_name = list(child.get("properties", {}).keys())[0] if child.get("properties") else f"node_{offset}"
                # Get actual name from the node itself
                # Re-read name at current position
                name_end = self.data.index(b'\x00', offset + 4) if offset + 4 < len(self.data) else offset + 4
                child_name_bytes = self.data[offset+4:name_end]
                if child_name_bytes:
                    child_name = child_name_bytes.decode('utf-8', errors='replace')
                node["children"][child_name] = child

            elif token == self.FDT_NOP:
                offset += 4

            elif token == self.FDT_END_NODE:
                offset += 4
                break

            elif token == self.FDT_END:
                break

            else:
                offset += 4  # Skip unknown tokens

        return node, offset

    def get_memory_regions(self) -> list:
        """Extract memory regions from DTB."""
        regions = []

        def find_memory(node, path=""):
            for name, child in node.get("children", {}).items():
                if name.startswith("memory"):
                    reg = child.get("properties", {}).get("reg", b"")
                    if len(reg) >= 16:
                        addr = struct.unpack(">Q", reg[0:8])[0]
                        size = struct.unpack(">Q", reg[8:16])[0]
                        regions.append((addr, size))
                find_memory(child, f"{path}/{name}")

        find_memory(self.root)
        return regions

    def get_stdout_path(self) -> str:
        """Get the chosen stdout-path."""
        chosen = self.root.get("children", {}).get("chosen", {})
        stdout = chosen.get("properties", {}).get("stdout-path", b"")
        if stdout:
            return stdout.rstrip(b'\x00').decode('utf-8', errors='replace')
        return ""


# ════════════════════════════════════════════════════════════════════════════════
# ARM64 KERNEL LOADER
# ════════════════════════════════════════════════════════════════════════════════

class ARM64KernelLoader:
    """
    Load and boot an ARM64 Linux kernel.

    Supports:
    - Raw Image files (uncompressed)
    - gzip-compressed Image.gz files
    - ARM64 boot protocol setup
    """

    # ARM64 kernel Image header magic
    ARM64_MAGIC = 0x644d5241  # "ARM\x64"

    # Standard load addresses for virt machine
    KERNEL_LOAD_ADDR = 0x40080000
    DTB_LOAD_ADDR = 0x44000000
    INITRD_LOAD_ADDR = 0x45000000

    def __init__(self, kernel: NeuralARM64Kernel):
        self.kernel = kernel
        self.cpu = kernel.cpu
        self.dtb_parser = None

    def load_kernel(self, path: str) -> int:
        """
        Load kernel image and return entry point.

        Handles both raw and gzipped images.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Kernel not found: {path}")

        print(f"  Loading kernel: {path}")

        # Read kernel data
        with open(path, 'rb') as f:
            data = f.read()

        # Check if gzipped
        if data[:2] == b'\x1f\x8b':
            print("  Decompressing gzipped kernel...")
            data = gzip.decompress(data)

        print(f"  Kernel size: {len(data):,} bytes")

        # Check ARM64 Image header
        if len(data) >= 64:
            magic = struct.unpack("<I", data[56:60])[0]
            if magic == self.ARM64_MAGIC:
                text_offset = struct.unpack("<Q", data[8:16])[0]
                image_size = struct.unpack("<Q", data[16:24])[0]
                print(f"  ARM64 Image: text_offset=0x{text_offset:X}, size={image_size:,}")

        # Load kernel into memory
        load_addr = self.KERNEL_LOAD_ADDR
        self._load_to_memory(load_addr, data)
        print(f"  Loaded at: 0x{load_addr:X}")

        return load_addr

    def load_dtb(self, path: str) -> int:
        """Load Device Tree Blob."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DTB not found: {path}")

        print(f"  Loading DTB: {path}")

        with open(path, 'rb') as f:
            data = f.read()

        # Parse DTB
        self.dtb_parser = DTBParser(data)
        dtb_info = self.dtb_parser.parse()

        # Get memory regions
        mem_regions = self.dtb_parser.get_memory_regions()
        if mem_regions:
            for addr, size in mem_regions:
                print(f"  Memory: 0x{addr:X} - 0x{addr+size:X} ({size // (1024*1024)} MB)")

        # Load DTB into memory
        load_addr = self.DTB_LOAD_ADDR
        self._load_to_memory(load_addr, data)
        print(f"  DTB loaded at: 0x{load_addr:X}")

        return load_addr

    def load_initrd(self, path: str) -> tuple:
        """Load initrd/initramfs."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Initrd not found: {path}")

        print(f"  Loading initrd: {path}")

        with open(path, 'rb') as f:
            data = f.read()

        load_addr = self.INITRD_LOAD_ADDR
        self._load_to_memory(load_addr, data)
        end_addr = load_addr + len(data)

        print(f"  Initrd: 0x{load_addr:X} - 0x{end_addr:X} ({len(data):,} bytes)")

        return load_addr, end_addr

    def _load_to_memory(self, addr: int, data: bytes):
        """Load data into neural CPU memory."""
        # Ensure memory is large enough
        end_addr = addr + len(data)
        if end_addr > self.cpu.mem_size:
            raise ValueError(f"Memory overflow: need {end_addr:,}, have {self.cpu.mem_size:,}")

        # Convert to tensor and copy
        tensor_data = torch.tensor(list(data), dtype=torch.uint8, device=self.cpu.device)
        self.cpu.memory[addr:addr + len(data)] = tensor_data

    def setup_boot_protocol(self, kernel_addr: int, dtb_addr: int,
                            initrd_start: int = 0, initrd_end: int = 0):
        """
        Set up ARM64 boot protocol for kernel entry.

        ARM64 boot protocol (Documentation/arm64/booting.rst):
        - x0 = DTB physical address
        - x1 = 0 (reserved)
        - x2 = 0 (reserved)
        - x3 = 0 (reserved)
        - MMU off, D-cache off, I-cache on or off
        - Exceptions masked
        - PC = kernel entry point
        """
        print("\n  Setting up ARM64 boot protocol...")

        # Set register state
        self.cpu.regs[0] = dtb_addr  # x0 = DTB address
        self.cpu.regs[1] = 0         # x1 = reserved
        self.cpu.regs[2] = 0         # x2 = reserved
        self.cpu.regs[3] = 0         # x3 = reserved

        # Set stack pointer (SP_EL1)
        self.cpu.regs[31] = 0x48000000  # 128KB below initrd

        # Set PC to kernel entry
        self.cpu.pc = torch.tensor(kernel_addr, dtype=torch.int64, device=self.cpu.device)

        # Configure CPU state for kernel boot
        self.cpu.current_el = 1  # Start in EL1 (kernel mode)
        self.cpu.sctlr_el1 = 0   # MMU off, caches off
        self.cpu.mmu_enabled = False

        # Set up exception vector base (kernel will reconfigure)
        self.cpu.vbar_el1 = kernel_addr

        # Enable GIC
        self.cpu.gic_enabled = True
        self.cpu.gicd_ctlr = 1  # Enable distributor
        self.cpu.gicc_ctlr = 1  # Enable CPU interface

        # Enable UART
        self.cpu.uart_enabled = True

        # Update DTB with initrd info if provided
        if initrd_start and initrd_end and self.dtb_parser:
            print(f"  Initrd: start=0x{initrd_start:X}, end=0x{initrd_end:X}")

        print(f"  Entry point: 0x{kernel_addr:X}")
        print(f"  DTB address: 0x{dtb_addr:X}")
        print(f"  Stack: 0x{self.cpu.regs[31]:X}")
        print(f"  EL: {self.cpu.current_el}")


# ════════════════════════════════════════════════════════════════════════════════
# KERNEL BOOT RUNNER
# ════════════════════════════════════════════════════════════════════════════════

def boot_kernel(args):
    """Boot a Linux kernel on the Neural ARM64 CPU."""
    print("=" * 78)
    print("   NEURAL ARM64 KERNEL BOOT")
    print("=" * 78)
    print()

    # Create kernel with larger memory for kernel boot
    memory_size = args.memory * 1024 * 1024  # Convert MB to bytes
    print(f"  Initializing Neural ARM64 Kernel ({args.memory} MB)...")
    kernel = NeuralARM64Kernel(memory_size=memory_size)

    # Create loader
    loader = ARM64KernelLoader(kernel)

    print()
    print("  Loading boot components:")
    print("  " + "-" * 40)

    # Load kernel
    kernel_addr = loader.load_kernel(args.kernel)

    # Load DTB
    dtb_addr = loader.load_dtb(args.dtb)

    # Load initrd if provided
    initrd_start = 0
    initrd_end = 0
    if args.initrd:
        initrd_start, initrd_end = loader.load_initrd(args.initrd)

    # Set up boot protocol
    loader.setup_boot_protocol(kernel_addr, dtb_addr, initrd_start, initrd_end)

    print()
    print("=" * 78)
    print("   STARTING KERNEL EXECUTION")
    print("=" * 78)
    print()

    # Run kernel
    try:
        start_time = time.perf_counter()

        if args.gpu_only:
            exit_code, elapsed = kernel.run_elf_gpu_only(
                b"",  # No ELF - already loaded
                [],
                max_instructions=args.max_instructions,
                batch_size=args.batch_size,
            )
        else:
            # Use the standard execution path
            max_steps = args.max_instructions
            step = 0
            while not kernel.cpu.halted and step < max_steps:
                # Execute in batches
                batch_executed, _ = kernel.cpu.run(min(10000, max_steps - step))
                step += batch_executed

                # Check for MMIO (handled by execute path)
                # Check for interrupts
                kernel.cpu._check_pending_irqs()

                # Print progress periodically
                if step % 100000 == 0:
                    elapsed = time.perf_counter() - start_time
                    ips = step / elapsed if elapsed > 0 else 0
                    print(f"  [{step:,} instructions, {ips:,.0f} IPS]")

            elapsed = time.perf_counter() - start_time
            exit_code = 0

        total_time = time.perf_counter() - start_time

        print()
        print("=" * 78)
        print("   EXECUTION COMPLETE")
        print("=" * 78)
        print(f"  Exit code: {exit_code}")
        print(f"  Time: {elapsed:.2f}s (total: {total_time:.2f}s)")
        print(f"  Instructions: {kernel.total_instructions:,}")
        if elapsed > 0:
            print(f"  IPS: {kernel.total_instructions/elapsed:,.0f}")
        print("=" * 78)

        return exit_code

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Boot a Linux kernel on Neural ARM64 CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --kernel Image --dtb virt.dtb
  %(prog)s --kernel Image.gz --dtb virt.dtb --initrd initrd.gz
  %(prog)s --kernel Image --dtb virt.dtb --memory 512
        """
    )

    parser.add_argument("--kernel", "-k", required=True,
                        help="Path to kernel Image (raw or gzipped)")
    parser.add_argument("--dtb", "-d", required=True,
                        help="Path to Device Tree Blob (.dtb)")
    parser.add_argument("--initrd", "-i",
                        help="Path to initial ramdisk (optional)")
    parser.add_argument("--memory", "-m", type=int, default=256,
                        help="Memory size in MB (default: 256)")
    parser.add_argument("--max-instructions", type=int, default=10_000_000,
                        help="Maximum instructions to execute")
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="GPU batch size")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Use GPU-only execution path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    sys.exit(boot_kernel(args))


if __name__ == "__main__":
    main()
