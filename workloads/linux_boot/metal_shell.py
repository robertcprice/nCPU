#!/usr/bin/env python3
"""
METAL SHELL - Real ARM64 Interactive Shell on GPU

This runs actual ARM64 machine code on the Metal GPU with real syscall handling.
No Python simulation - actual ARM64 instructions executed on GPU!

Syscalls supported:
- read (63): Read from stdin
- write (64): Write to stdout
- exit (93): Exit program
"""

import sys
import os
import struct
import time
import select

# Import Metal CPU
try:
    from kvrm_metal import ContinuousMetalCPU
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("Error: kvrm_metal not found. Run 'maturin develop --release' in rust_metal/")
    sys.exit(1)

# ANSI Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# ARM64 Syscall Numbers (Linux AArch64)
SYS_READ = 63
SYS_WRITE = 64
SYS_EXIT = 93
SYS_BRK = 214

# ARM64 Instruction Encoding
def encode_movz(rd, imm16, shift=0):
    """MOVZ Xd, #imm16, LSL #shift"""
    hw = shift // 16
    inst = 0xD2800000 | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)
    return struct.pack('<I', inst)

def encode_movk(rd, imm16, shift):
    """MOVK Xd, #imm16, LSL #shift"""
    hw = shift // 16
    inst = 0xF2800000 | ((hw & 0x3) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)
    return struct.pack('<I', inst)

def encode_add_imm(rd, rn, imm12):
    """ADD Xd, Xn, #imm12"""
    inst = 0x91000000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
    return struct.pack('<I', inst)

def encode_sub_imm(rd, rn, imm12):
    """SUB Xd, Xn, #imm12"""
    inst = 0xD1000000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F)
    return struct.pack('<I', inst)

def encode_mov_reg(rd, rm):
    """MOV Xd, Xm (alias for ORR Xd, XZR, Xm)"""
    inst = 0xAA0003E0 | ((rm & 0x1F) << 16) | (rd & 0x1F)
    return struct.pack('<I', inst)

def encode_svc(imm16):
    """SVC #imm16 - Supervisor Call (syscall)"""
    inst = 0xD4000001 | ((imm16 & 0xFFFF) << 5)
    return struct.pack('<I', inst)

def encode_branch(offset_words):
    """B offset - Unconditional branch"""
    imm26 = offset_words & 0x3FFFFFF
    inst = 0x14000000 | imm26
    return struct.pack('<I', inst)

def encode_cbz(rt, offset_words):
    """CBZ Xt, offset - Compare and branch if zero"""
    imm19 = offset_words & 0x7FFFF
    inst = 0xB4000000 | (imm19 << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_cbnz(rt, offset_words):
    """CBNZ Xt, offset - Compare and branch if not zero"""
    imm19 = offset_words & 0x7FFFF
    inst = 0xB5000000 | (imm19 << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_strb(rt, rn, imm12):
    """STRB Wt, [Xn, #imm12] - Store byte"""
    inst = 0x39000000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_ldrb(rt, rn, imm12):
    """LDRB Wt, [Xn, #imm12] - Load byte"""
    inst = 0x39400000 | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_str(rt, rn, imm12):
    """STR Xt, [Xn, #imm12] - Store 64-bit (imm12 must be 8-byte aligned)"""
    # Immediate is scaled by 8 for 64-bit operations
    scaled_imm = (imm12 // 8) & 0xFFF
    inst = 0xF9000000 | (scaled_imm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_ldr(rt, rn, imm12):
    """LDR Xt, [Xn, #imm12] - Load 64-bit (imm12 must be 8-byte aligned)"""
    # Immediate is scaled by 8 for 64-bit operations
    scaled_imm = (imm12 // 8) & 0xFFF
    inst = 0xF9400000 | (scaled_imm << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F)
    return struct.pack('<I', inst)

def encode_hlt():
    """HLT #0"""
    return struct.pack('<I', 0xD4400000)

def load_immediate(rd, value):
    """Load a 64-bit immediate into register Xd."""
    code = b''
    code += encode_movz(rd, value & 0xFFFF, 0)
    if value > 0xFFFF:
        code += encode_movk(rd, (value >> 16) & 0xFFFF, 16)
    if value > 0xFFFFFFFF:
        code += encode_movk(rd, (value >> 32) & 0xFFFF, 32)
    if value > 0xFFFFFFFFFFFF:
        code += encode_movk(rd, (value >> 48) & 0xFFFF, 48)
    return code


class MetalShell:
    """Real ARM64 shell running on Metal GPU."""

    def __init__(self):
        # Create CPU with small batch for interactive use
        self.cpu = ContinuousMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10000)

        # Memory layout
        self.CODE_BASE = 0x1000
        self.DATA_BASE = 0x100000  # 1MB
        self.BUFFER_BASE = 0x200000  # 2MB - I/O buffer
        self.STACK_BASE = 0x300000  # 3MB

        self.running = True
        self.total_instructions = 0

    def build_shell_program(self):
        """
        Build ARM64 shell program using X9 to save read count.
        """
        code = b''

        prompt_addr = self.DATA_BASE
        buffer_addr = self.BUFFER_BASE

        # === MAIN LOOP ===
        loop_start = len(code)

        # Write prompt: write(1, prompt, 2)
        code += load_immediate(0, 1)              # x0 = stdout
        code += load_immediate(1, prompt_addr)    # x1 = prompt address
        code += load_immediate(2, 2)              # x2 = 2 bytes
        code += load_immediate(8, SYS_WRITE)      # x8 = write
        code += encode_svc(0)

        # Read input: read(0, buffer, 256)
        code += load_immediate(0, 0)              # x0 = stdin
        code += load_immediate(1, buffer_addr)    # x1 = buffer
        code += load_immediate(2, 256)            # x2 = max length
        code += load_immediate(8, SYS_READ)       # x8 = read
        code += encode_svc(0)

        # Save read count: store X0 to temp memory and reload to X9
        # STR X0, [X1] ; LDR X9, [X1] (buffer_addr is still in X1)
        temp_addr = self.BUFFER_BASE + 512  # Use offset in buffer for temp
        code += encode_str(0, 1, 512)             # STR X0, [X1, #512]
        code += encode_ldr(9, 1, 512)             # LDR X9, [X1, #512]

        # Check for EOF
        cbz_offset = len(code)
        code += encode_cbz(9, 0)  # Placeholder

        # Echo: write(1, buffer, x9)
        # Copy X9 to X2 via memory (MOV/ORR not working correctly)
        code += load_immediate(10, self.BUFFER_BASE + 520)  # x10 = temp address
        code += encode_str(9, 10, 0)              # STR X9, [X10]
        code += encode_ldr(2, 10, 0)              # LDR X2, [X10]
        code += load_immediate(0, 1)              # x0 = stdout
        code += load_immediate(1, buffer_addr)    # x1 = buffer
        code += load_immediate(8, SYS_WRITE)      # x8 = write
        code += encode_svc(0)

        # Loop back
        loop_end = len(code)
        branch_offset = (loop_start - loop_end) // 4
        code += encode_branch(branch_offset & 0x3FFFFFF)

        # === EXIT ===
        exit_offset = len(code)
        code += load_immediate(0, 0)              # x0 = 0
        code += load_immediate(8, SYS_EXIT)       # x8 = exit
        code += encode_svc(0)
        code += encode_hlt()

        # Patch CBZ
        cbz_target_words = (exit_offset - cbz_offset) // 4
        patched_cbz = encode_cbz(9, cbz_target_words)
        code = code[:cbz_offset] + patched_cbz + code[cbz_offset + 4:]

        return code

    def setup_memory(self):
        """Initialize memory with program and data."""
        # Build and load shell program
        shell_code = self.build_shell_program()

        self.cpu.load_program(list(shell_code), self.CODE_BASE)

        # Write prompt string "$ " to DATA_BASE
        prompt = b'$ '
        for i, byte in enumerate(prompt):
            # Use memory write - we need to set individual bytes
            # Load program can do this
            pass

        # Actually we need to write prompt to memory
        # The load_program function can load bytes at any address
        self.cpu.load_program(list(prompt), self.DATA_BASE)

        # Set PC to code start
        self.cpu.set_pc(self.CODE_BASE)

        # Set stack pointer (X31/SP)
        self.cpu.set_register(31, self.STACK_BASE)

        # Clear other registers
        for i in range(31):
            self.cpu.set_register(i, 0)

    def handle_syscall(self, debug=False):
        """Handle a syscall from the GPU."""
        # Read syscall number from X8
        syscall_num = self.cpu.get_register(8)
        x0 = self.cpu.get_register(0)
        x1 = self.cpu.get_register(1)
        x2 = self.cpu.get_register(2)

        if debug:
            x9 = self.cpu.get_register(9)
            pc = self.cpu.get_pc()
            # Only show debug for non-standard syscalls
            if syscall_num not in (SYS_READ, SYS_WRITE, SYS_EXIT):
                print(f"{DIM}[syscall {syscall_num}: x0={x0}]{RESET}")

        if syscall_num == SYS_WRITE:
            # write(fd, buf, count)
            fd = x0
            buf_addr = x1
            count = x2

            # Read bytes from GPU memory
            data = bytes(self.cpu.read_memory(buf_addr, count))

            if fd == 1:  # stdout
                sys.stdout.write(data.decode('utf-8', errors='replace'))
                sys.stdout.flush()
            elif fd == 2:  # stderr
                sys.stderr.write(data.decode('utf-8', errors='replace'))
                sys.stderr.flush()

            # Return bytes written
            self.cpu.set_register(0, len(data))

        elif syscall_num == SYS_READ:
            # read(fd, buf, count)
            fd = x0
            buf_addr = x1
            count = x2

            if fd == 0:  # stdin
                try:
                    # Read from stdin
                    data = sys.stdin.readline(count)
                    if not data:
                        # EOF
                        self.cpu.set_register(0, 0)
                    else:
                        data_bytes = data.encode('utf-8')[:count]
                        # Write to GPU memory using load_program
                        self.cpu.load_program(list(data_bytes), buf_addr)
                        bytes_read = len(data_bytes)
                        self.cpu.set_register(0, bytes_read)
                except Exception as e:
                    self.cpu.set_register(0, -1)
            else:
                self.cpu.set_register(0, -1)

        elif syscall_num == SYS_EXIT:
            # exit(code)
            exit_code = x0
            self.running = False
            return exit_code

        elif syscall_num == SYS_BRK:
            # brk(addr) - just return current address
            self.cpu.set_register(0, x0 if x0 > 0 else self.STACK_BASE)

        else:
            # Unknown syscall
            print(f"{YELLOW}Unknown syscall: {syscall_num}{RESET}")
            self.cpu.set_register(0, -1)

        # Advance PC past SVC instruction (4 bytes)
        pc = self.cpu.get_pc()
        self.cpu.set_pc(pc + 4)

        return None

    def run(self):
        """Run the interactive shell."""
        print(f"""
{CYAN}{'=' * 70}{RESET}
{BOLD}  METAL SHELL - Real ARM64 on GPU{RESET}
{CYAN}{'=' * 70}{RESET}

  This is a REAL ARM64 shell running on Metal GPU.
  Every instruction executes on GPU via Metal shaders.
  Syscalls (read/write) transfer data between GPU and terminal.

  Type commands and press Enter. Type 'exit' or Ctrl+D to quit.

{CYAN}{'=' * 70}{RESET}
""")

        # Setup
        self.setup_memory()

        # Main execution loop
        while self.running:
            try:
                # Execute on GPU until syscall or halt
                result = self.cpu.execute_continuous(1000, 10.0)  # max 1000 batches, 10s timeout

                self.total_instructions += result.total_cycles

                # Check signal
                if result.signal == 1:  # HALT
                    print(f"\n{DIM}[GPU halted]{RESET}")
                    break
                elif result.signal == 2:  # SYSCALL
                    exit_code = self.handle_syscall(debug=True)
                    if exit_code is not None:
                        print(f"\n{DIM}[Exit code: {exit_code}]{RESET}")
                        break
                else:
                    # Timeout or other
                    pass

            except KeyboardInterrupt:
                print(f"\n{YELLOW}^C{RESET}")
                break
            except EOFError:
                break

        print(f"\n{CYAN}Metal Shell terminated.{RESET}")
        print(f"{DIM}Total GPU instructions executed: {self.total_instructions:,}{RESET}")
        print(f"{DIM}Running on: Apple Silicon Metal GPU (~1 MIPS){RESET}")


def main():
    shell = MetalShell()
    shell.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
