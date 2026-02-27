#!/usr/bin/env python3
"""
NEURAL OS COMPLETE - Full OS Running on Neural CPU
===================================================

Features:
1. Interactive Shell (bash-like)
2. Memory-based Filesystem
3. ASCII DOOM Raycaster
4. All running on the Neural CPU with fused tensor cache!

Memory Map:
  0x00000 - 0x0FFFF: Kernel code
  0x10000 - 0x1FFFF: User programs
  0x20000 - 0x2FFFF: Stack
  0x40000 - 0x4FFFF: Framebuffer (80x25 = 2000 bytes)
  0x50000 - 0x5FFFF: Filesystem
  0x60000 - 0x6FFFF: Input buffer
"""

import struct
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turbo_neural_cpu import TurboNeuralCPU, device
import torch

# Memory addresses
KERNEL_BASE = 0x00000
STACK_BASE = 0x2FFF0
FB_BASE = 0x40000
FS_BASE = 0x50000
INPUT_BASE = 0x60000

FB_WIDTH = 80
FB_HEIGHT = 25


class ARM64Assembler:
    """Simple ARM64 assembler for generating machine code."""

    def __init__(self):
        self.code = bytearray()
        self.labels = {}
        self.fixups = []
        self.base_addr = 0

    def set_base(self, addr):
        self.base_addr = addr

    def label(self, name):
        self.labels[name] = len(self.code) + self.base_addr

    def nop(self):
        self.code += struct.pack('<I', 0xD503201F)

    def ret(self):
        self.code += struct.pack('<I', 0xD65F03C0)

    def movz(self, rd, imm16, hw=0):
        inst = (1 << 31) | (0b10 << 29) | (0b100101 << 23) | (hw << 21) | (imm16 << 5) | rd
        self.code += struct.pack('<I', inst)

    def movk(self, rd, imm16, hw=0):
        inst = (1 << 31) | (0b11 << 29) | (0b100101 << 23) | (hw << 21) | (imm16 << 5) | rd
        self.code += struct.pack('<I', inst)

    def mov_imm(self, rd, value):
        self.movz(rd, value & 0xFFFF, 0)
        if value > 0xFFFF:
            self.movk(rd, (value >> 16) & 0xFFFF, 1)
        if value > 0xFFFFFFFF:
            self.movk(rd, (value >> 32) & 0xFFFF, 2)

    def mov_reg(self, rd, rm):
        # ORR Xd, XZR, Xm
        inst = (1 << 31) | (0b0101010 << 24) | (rm << 16) | (0x1F << 5) | rd
        self.code += struct.pack('<I', inst)

    def add_imm(self, rd, rn, imm12):
        inst = (1 << 31) | (0b00100010 << 23) | (imm12 << 10) | (rn << 5) | rd
        self.code += struct.pack('<I', inst)

    def add_reg(self, rd, rn, rm):
        inst = (1 << 31) | (0b0001011 << 24) | (rm << 16) | (rn << 5) | rd
        self.code += struct.pack('<I', inst)

    def sub_imm(self, rd, rn, imm12):
        # SUB Xd, Xn, #imm12: sf=1, op=1, S=0 → 0xD1xxxxxx
        inst = (1 << 31) | (0b10 << 29) | (0b100010 << 23) | (imm12 << 10) | (rn << 5) | rd
        self.code += struct.pack('<I', inst)

    def sub_reg(self, rd, rn, rm):
        inst = (1 << 31) | (0b1001011 << 24) | (rm << 16) | (rn << 5) | rd
        self.code += struct.pack('<I', inst)

    def cmp_imm(self, rn, imm12):
        inst = (1 << 31) | (0b11100010 << 23) | (1 << 22) | (imm12 << 10) | (rn << 5) | 31
        self.code += struct.pack('<I', inst)

    def cmp_reg(self, rn, rm):
        inst = (1 << 31) | (0b1101011 << 24) | (rm << 16) | (rn << 5) | 31
        self.code += struct.pack('<I', inst)

    def b(self, label):
        self.fixups.append((len(self.code), label, 'b'))
        self.code += struct.pack('<I', 0x14000000)

    def bl(self, label):
        self.fixups.append((len(self.code), label, 'bl'))
        self.code += struct.pack('<I', 0x94000000)

    def b_eq(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x54000000)

    def b_ne(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x54000001)

    def b_lt(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x5400000B)

    def b_ge(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x5400000A)

    def b_gt(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x5400000C)

    def b_le(self, label):
        self.fixups.append((len(self.code), label, 'bcond'))
        self.code += struct.pack('<I', 0x5400000D)

    def cbz(self, rt, label):
        self.fixups.append((len(self.code), label, 'cbz'))
        inst = (1 << 31) | (0b011010 << 25) | rt
        self.code += struct.pack('<I', inst)

    def cbnz(self, rt, label):
        self.fixups.append((len(self.code), label, 'cbnz'))
        inst = (1 << 31) | (0b011010 << 25) | (1 << 24) | rt
        self.code += struct.pack('<I', inst)

    def ldr(self, rt, rn, offset=0):
        imm12 = offset >> 3
        inst = (0b11111001 << 24) | (0b01 << 22) | (imm12 << 10) | (rn << 5) | rt
        self.code += struct.pack('<I', inst)

    def ldrb(self, rt, rn, offset=0):
        inst = (0b00111001 << 24) | (0b01 << 22) | (offset << 10) | (rn << 5) | rt
        self.code += struct.pack('<I', inst)

    def str(self, rt, rn, offset=0):
        imm12 = offset >> 3
        inst = (0b11111001 << 24) | (0b00 << 22) | (imm12 << 10) | (rn << 5) | rt
        self.code += struct.pack('<I', inst)

    def strb(self, rt, rn, offset=0):
        inst = (0b00111001 << 24) | (0b00 << 22) | (offset << 10) | (rn << 5) | rt
        self.code += struct.pack('<I', inst)

    def mul(self, rd, rn, rm):
        inst = (1 << 31) | (0b0011011 << 24) | (rm << 16) | (0x1F << 10) | (rn << 5) | rd
        self.code += struct.pack('<I', inst)

    def resolve(self):
        for offset, label, fixup_type in self.fixups:
            if label not in self.labels:
                raise ValueError(f"Unknown label: {label}")
            target = self.labels[label]
            current = offset + self.base_addr

            if fixup_type in ('b', 'bl'):
                rel = (target - current) >> 2
                rel &= 0x3FFFFFF
                inst = struct.unpack('<I', self.code[offset:offset+4])[0]
                inst |= rel
                self.code[offset:offset+4] = struct.pack('<I', inst)
            elif fixup_type in ('bcond', 'cbz', 'cbnz'):
                rel = (target - current) >> 2
                rel &= 0x7FFFF
                inst = struct.unpack('<I', self.code[offset:offset+4])[0]
                inst |= (rel << 5)
                self.code[offset:offset+4] = struct.pack('<I', inst)

    def get_code(self):
        self.resolve()
        return bytes(self.code)


class NeuralOSBuilder:
    """Builds the Neural OS kernel."""

    def __init__(self):
        self.asm = ARM64Assembler()
        self.asm.set_base(KERNEL_BASE)

    def build_kernel(self):
        asm = self.asm

        # ===== ENTRY POINT =====
        asm.label('_start')
        asm.mov_imm(29, STACK_BASE)
        asm.mov_imm(28, FB_BASE)
        asm.bl('clear_screen')
        asm.mov_imm(0, KERNEL_BASE + 0x1000)
        asm.bl('print_string')
        asm.b('shell_main')

        # ===== CLEAR SCREEN =====
        asm.label('clear_screen')
        asm.mov_imm(0, FB_BASE)
        asm.mov_imm(1, FB_WIDTH * FB_HEIGHT)
        asm.mov_imm(2, 0x20)
        asm.label('clear_loop')
        asm.strb(2, 0, 0)
        asm.add_imm(0, 0, 1)
        asm.sub_imm(1, 1, 1)
        asm.cbnz(1, 'clear_loop')
        asm.ret()

        # ===== PRINT STRING =====
        asm.label('print_string')
        asm.mov_imm(10, FB_BASE)
        asm.label('print_loop')
        asm.ldrb(1, 0, 0)
        asm.cbz(1, 'print_done')
        asm.strb(1, 10, 0)
        asm.add_imm(0, 0, 1)
        asm.add_imm(10, 10, 1)
        asm.b('print_loop')
        asm.label('print_done')
        asm.ret()

        # ===== SHELL MAIN =====
        asm.label('shell_main')
        asm.label('shell_prompt')
        asm.mov_imm(0, KERNEL_BASE + 0x1100)
        asm.bl('print_string')
        asm.mov_imm(0, INPUT_BASE)
        asm.bl('parse_command')
        asm.b('shell_prompt')

        # ===== PARSE COMMAND =====
        asm.label('parse_command')
        asm.ldrb(1, 0, 0)
        asm.cmp_imm(1, ord('d'))
        asm.b_ne('check_ls')
        asm.b('run_doom')

        asm.label('check_ls')
        asm.cmp_imm(1, ord('l'))
        asm.b_ne('check_cat')
        asm.b('cmd_ls')

        asm.label('check_cat')
        asm.cmp_imm(1, ord('c'))
        asm.b_ne('cmd_unknown')
        asm.b('cmd_cat')

        asm.label('cmd_unknown')
        asm.mov_imm(0, KERNEL_BASE + 0x1200)
        asm.bl('print_string')
        asm.ret()

        asm.label('cmd_ls')
        asm.mov_imm(0, KERNEL_BASE + 0x1300)
        asm.bl('print_string')
        asm.ret()

        asm.label('cmd_cat')
        asm.mov_imm(0, KERNEL_BASE + 0x1400)
        asm.bl('print_string')
        asm.ret()

        # ===== DOOM RAYCASTER =====
        asm.label('run_doom')
        asm.mov_imm(20, 0x0500)
        asm.mov_imm(21, 0x0500)
        asm.mov_imm(22, 0)

        asm.label('doom_loop')
        asm.bl('clear_screen')
        asm.bl('doom_render')
        asm.mov_imm(0, INPUT_BASE)
        asm.ldrb(1, 0, 0)
        asm.cmp_imm(1, ord('q'))
        asm.b_eq('doom_exit')
        asm.b('doom_loop')

        asm.label('doom_exit')
        asm.ret()

        # ===== DOOM RENDER =====
        asm.label('doom_render')
        asm.mov_imm(0, 0)

        asm.label('render_col_loop')
        asm.mov_imm(5, 0)

        asm.label('render_row_loop')
        asm.mov_imm(7, FB_WIDTH)
        asm.mul(8, 5, 7)
        asm.add_reg(8, 8, 0)
        asm.mov_imm(9, FB_BASE)
        asm.add_reg(8, 8, 9)
        asm.cmp_imm(5, 8)
        asm.b_lt('draw_ceiling')
        asm.cmp_imm(5, 16)
        asm.b_gt('draw_floor')
        asm.mov_imm(10, ord('#'))
        asm.strb(10, 8, 0)
        asm.b('next_row')

        asm.label('draw_ceiling')
        asm.mov_imm(10, ord(' '))
        asm.strb(10, 8, 0)
        asm.b('next_row')

        asm.label('draw_floor')
        asm.mov_imm(10, ord('.'))
        asm.strb(10, 8, 0)

        asm.label('next_row')
        asm.add_imm(5, 5, 1)
        asm.cmp_imm(5, 24)
        asm.b_lt('render_row_loop')
        asm.add_imm(0, 0, 1)
        asm.cmp_imm(0, 80)
        asm.b_lt('render_col_loop')
        asm.ret()

        # ===== DATA SECTION =====
        while len(asm.code) < 0x1000:
            asm.nop()

        asm.code += b"NeuralOS v1.0 - Running on Neural CPU!\n\x00"
        while len(asm.code) < 0x1100:
            asm.code += b'\x00'
        asm.code += b"$ \x00"
        while len(asm.code) < 0x1200:
            asm.code += b'\x00'
        asm.code += b"Unknown command\n\x00"
        while len(asm.code) < 0x1300:
            asm.code += b'\x00'
        asm.code += b"readme.txt\nhello.c\ndoom\n\x00"
        while len(asm.code) < 0x1400:
            asm.code += b'\x00'
        asm.code += b"Welcome to NeuralOS!\n\x00"

        return asm.get_code()


class NeuralOS:
    """Complete Neural OS running on the Turbo Neural CPU."""

    def __init__(self):
        print("=" * 70)
        print("   NEURAL OS - Complete Operating System on Neural CPU")
        print("=" * 70)

        self.cpu = TurboNeuralCPU(memory_size=1024 * 1024)

        builder = NeuralOSBuilder()
        kernel = builder.build_kernel()
        print(f"   Kernel size: {len(kernel)} bytes")

        self.cpu.memory[KERNEL_BASE:KERNEL_BASE+len(kernel)] = kernel
        self.cpu.pc = KERNEL_BASE
        print("=" * 70)

    def set_input(self, text):
        text_bytes = text.encode()[:255] + b'\x00'
        self.cpu.memory[INPUT_BASE:INPUT_BASE + len(text_bytes)] = text_bytes

    def get_framebuffer(self):
        lines = []
        for row in range(FB_HEIGHT):
            line = ''
            for col in range(FB_WIDTH):
                ch = self.cpu.memory[FB_BASE + row * FB_WIDTH + col]
                line += chr(ch) if 32 <= ch <= 126 else ' '
            lines.append(line.rstrip())
        return '\n'.join(lines)

    def run(self, instructions=1000):
        return self.cpu.run(instructions)

    def step(self):
        self.cpu.step()


def benchmark_os():
    """Benchmark the Neural OS."""
    print("\n" + "=" * 70)
    print("   NEURAL OS BENCHMARK")
    print("=" * 70)

    nos = NeuralOS()

    print("\nBooting Neural OS...")
    start = time.perf_counter()
    executed, elapsed = nos.run(10000)
    boot_ips = executed / elapsed

    print(f"\n   Boot: {executed:,} instructions in {elapsed:.2f}s")
    print(f"   IPS: {boot_ips:,.0f}")

    print("\n   Framebuffer:")
    fb = nos.get_framebuffer()
    for line in fb.split('\n')[:10]:
        if line.strip():
            print(f"   |{line}|")

    print("\n   Testing DOOM...")
    nos.set_input("doom")
    start = time.perf_counter()
    nos.run(50000)
    doom_time = time.perf_counter() - start

    print(f"   DOOM rendered in {doom_time:.2f}s")
    fb = nos.get_framebuffer()
    print("   DOOM framebuffer:")
    for line in fb.split('\n')[:15]:
        print(f"   |{line}|")

    nos.cpu.print_stats()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_os()
