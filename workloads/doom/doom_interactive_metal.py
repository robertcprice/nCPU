#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          INTERACTIVE DOOM ON NEURAL METAL CPU WITH SYSCALL HANDLING          ║
║                                                                              ║
║  Features:                                                                   ║
║  - Syscall handling for write(), read(), exit(), brk()                        ║
║  - Interactive keyboard controls                                              ║
║  - Real-time framebuffer display                                              ║
║  - 640K+ IPS performance                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from kvrm_metal import MetalCPU
import time
import sys
import termios
import tty
import select
import os

# Linux ARM64 syscall numbers
SYS_READ = 63
SYS_WRITE = 64
SYS_EXIT = 93
SYS_GETPID = 172
SYS_IOCTL = 29
SYS_MUNMAP = 215
SYS_MMAP = 222

# File descriptors
STDIN_FILENO = 0
STDOUT_FILENO = 1
STDERR_FILENO = 2

# DOOM/Raycast display constants
FB_WIDTH = 80
FB_HEIGHT = 25
FB_ADDR = 0x40000
PLAYER_ADDR = 0x20000
MAP_ADDR = 0x30000
KEY_ADDR = 0x50000
CODE_ADDR = 0x10000


class SyscallMetalCPU:
    """MetalCPU with Linux syscall handling"""

    def __init__(self, memory_size=16*1024*1024):
        self.cpu = MetalCPU(memory_size=memory_size)
        self.memory_size = memory_size
        self.running = True
        self.output_buffer = []

        # Key state
        self.key_state = 0  # 0=none, 1=w, 2=s, 3=a, 4=d, 5=q

    def load_program(self, data, address):
        self.cpu.load_program(bytearray(data), address)

    def set_pc(self, pc):
        self.cpu.set_pc(pc)

    def set_register(self, reg, value):
        self.cpu.set_register(reg, value)

    def get_register(self, reg):
        return self.cpu.get_register(reg)

    def read_memory(self, address, size):
        return self.cpu.read_memory(address, size)

    def write_memory(self, address, data):
        self.cpu.write_memory(address, bytearray(data))

    def handle_syscall(self):
        """Handle ARM64 SVC syscall"""
        # Get syscall number from x8
        syscall_num = self.cpu.get_register(8)

        # Handle different syscalls
        if syscall_num == SYS_WRITE:
            return self.handle_write()
        elif syscall_num == SYS_READ:
            return self.handle_read()
        elif syscall_num == SYS_EXIT:
            return self.handle_exit()
        elif syscall_num == SYS_IOCTL:
            return self.handle_ioctl()
        elif syscall_num == SYS_GETPID:
            return self.handle_getpid()
        elif syscall_num == SYS_BRK:
            return self.handle_brk()
        else:
            # Unknown syscall - just continue
            self.cpu.set_pc(self.cpu.get_pc() + 4)
            return True

    def handle_write(self):
        """Handle write(fd, buf, count)"""
        fd = self.cpu.get_register(0)
        buf = self.cpu.get_register(1)
        count = self.cpu.get_register(2)

        if fd == STDOUT_FILENO or fd == STDERR_FILENO:
            # Read from memory and output
            data = self.cpu.read_memory(buf, count)
            try:
                text = bytes(data).decode('utf-8', errors='ignore')
                print(text, end='', flush=True)
            except:
                pass

        # Return bytes written in x0
        self.cpu.set_register(0, count)
        self.cpu.set_pc(self.cpu.get_pc() + 4)
        return True

    def handle_read(self):
        """Handle read(fd, buf, count)"""
        fd = self.cpu.get_register(0)
        buf = self.cpu.get_register(1)
        count = self.cpu.get_register(2)

        if fd == STDIN_FILENO:
            # Non-blocking read - check for key
            key_data = self.get_key_input()
            if key_data:
                self.cpu.write_memory(buf, key_data[:count])
                self.cpu.set_register(0, len(key_data[:count]))
            else:
                self.cpu.set_register(0, 0)  # No data available

        self.cpu.set_pc(self.cpu.get_pc() + 4)
        return True

    def handle_exit(self):
        """Handle exit(code)"""
        exit_code = self.cpu.get_register(0)
        self.running = False
        return False

    def handle_ioctl(self):
        """Handle ioctl() - just return success"""
        self.cpu.set_register(0, 0)
        self.cpu.set_pc(self.cpu.get_pc() + 4)
        return True

    def handle_getpid(self):
        """Handle getpid() - return fake PID"""
        self.cpu.set_register(0, 1)
        self.cpu.set_pc(self.cpu.get_pc() + 4)
        return True

    def handle_brk(self):
        """Handle brk() - simple memory management"""
        addr = self.cpu.get_register(0)
        # For simplicity, just return the address
        self.cpu.set_register(0, addr if addr > 0 else 0x1000000)
        self.cpu.set_pc(self.cpu.get_pc() + 4)
        return True

    def get_key_input(self):
        """Get keyboard input (non-blocking)"""
        if self.key_state == 5:  # q = quit
            return b'q'
        elif self.key_state == 1:  # w = up
            return b'w'
        elif self.key_state == 2:  # s = down
            return b's'
        elif self.key_state == 3:  # a = left
            return b'a'
        elif self.key_state == 4:  # d = right
            return b'd'
        return b''

    def execute_with_syscalls(self, max_cycles=100000):
        """Execute with automatic syscall handling"""
        total_cycles = 0
        batch_size = 50000

        while total_cycles < max_cycles and self.running:
            cycles = min(batch_size, max_cycles - total_cycles)
            result = self.cpu.execute(max_cycles=cycles)
            total_cycles += result.cycles

            # Check if we hit a syscall (SVC instruction)
            # SVC is detected by stop_reason = 2 (SYSCALL)
            if result.stop_reason == 2:
                # Handle syscall
                if not self.handle_syscall():
                    break
            elif result.stop_reason == 1:
                # HALT
                break

        return {
            'cycles': total_cycles,
            'final_pc': result.final_pc,
            'running': self.running
        }


def set_nonblocking(fd):
    """Set file descriptor to non-blocking mode"""
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


import fcntl


class InteractiveDOOM:
    """Interactive DOOM with keyboard controls"""

    def __init__(self):
        self.cpu = SyscallMetalCPU(memory_size=16*1024*1024)

        # Load raycast binary
        with open('arm64_doom/raycast.elf', 'rb') as f:
            self.binary = f.read()

        # Player state
        self.player_x = 8 << 16  # Fixed point 16.16
        self.player_y = 8 << 16
        self.player_angle = 0

        # Simple 16x16 map (1 = wall, 0 = empty)
        self.map_data = bytearray([
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,
            1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,
            1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,
            1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,
            1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,
            1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        ])

        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

    def init_game(self):
        """Initialize the game"""
        # Load program
        self.cpu.load_program(self.binary, CODE_ADDR)

        # Set up memory map
        self.cpu.write_memory(MAP_ADDR, self.map_data)

        # Set player position (fixed point 16.16)
        self.cpu.write_memory(PLAYER_ADDR,
            (self.player_x & 0xFF).to_bytes(1, 'little') +
            ((self.player_x >> 8) & 0xFF).to_bytes(1, 'little') +
            ((self.player_x >> 16) & 0xFF).to_bytes(1, 'little') +
            ((self.player_x >> 24) & 0xFF).to_bytes(1, 'little'))

        self.cpu.write_memory(PLAYER_ADDR + 4,
            (self.player_y & 0xFF).to_bytes(1, 'little') +
            ((self.player_y >> 8) & 0xFF).to_bytes(1, 'little') +
            ((self.player_y >> 16) & 0xFF).to_bytes(1, 'little') +
            ((self.player_y >> 24) & 0xFF).to_bytes(1, 'little'))

        self.cpu.write_memory(PLAYER_ADDR + 8,
            (self.player_angle & 0xFF).to_bytes(1, 'little') +
            ((self.player_angle >> 8) & 0xFF).to_bytes(1, 'little') +
            ((self.player_angle >> 16) & 0xFF).to_bytes(1, 'little') +
            ((self.player_angle >> 24) & 0xFF).to_bytes(1, 'little'))

        # Set stack
        self.cpu.set_register(31, 0x1000000)

        # Set entry point (ELF entry is 0x102E0, so load at 0x10000 + 0x102E0 = 0x202E0)
        self.cpu.set_pc(CODE_ADDR + 0x102E0)

    def render_framebuffer(self):
        """Render the ASCII framebuffer"""
        fb_data = self.cpu.read_memory(FB_ADDR, FB_WIDTH * FB_HEIGHT)

        print("\n" + "=" * 82)
        print("  NEURAL DOOM - FPS: {:.1f} | Angle: {} | Pos: ({}, {})".format(
            self.fps, self.player_angle,
            self.player_x >> 16, self.player_y >> 16))
        print("=" * 82)

        for y in range(FB_HEIGHT):
            line = ""
            for x in range(FB_WIDTH):
                offset = y * FB_WIDTH + x
                char = chr(fb_data[offset])
                line += char
            print(line)

        print("=" * 82)
        print("  W/S: Move | A/D: Turn | Q: Quit")
        print("=" * 82)

    def check_input(self):
        """Non-blocking keyboard input check"""
        # Check for available input
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'w' or key == 'W':
                self.cpu.key_state = 1
            elif key == 's' or key == 'S':
                self.cpu.key_state = 2
            elif key == 'a' or key == 'A':
                self.cpu.key_state = 3
            elif key == 'd' or key == 'D':
                self.cpu.key_state = 4
            elif key == 'q' or key == 'Q':
                self.cpu.key_state = 5
            else:
                self.cpu.key_state = 0
        else:
            self.cpu.key_state = 0

    def run_frame(self):
        """Run one frame of DOOM"""
        # Execute one frame
        result = self.cpu.execute_with_syscalls(max_cycles=500000)

        # Update FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = current_time

        self.frame_count += 1

        return result['running']

    def run(self):
        """Main game loop"""
        # Set up non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            set_nonblocking(sys.stdin.fileno())

            print("\n" + "=" * 82)
            print("  NEURAL DOOM - INTERACTIVE RAYCASTING ENGINE")
            print("  Running on Metal GPU (Apple M4 Pro)")
            print("=" * 82)
            print("\nInitializing...")

            self.init_game()

            print("Ready! Use W/S/A/D to move, Q to quit\n")

            while True:
                # Check input
                self.check_input()

                if self.cpu.key_state == 5:  # Q = quit
                    print("\nQuitting...")
                    break

                # Run frame
                if not self.run_frame():
                    break

                # Render
                self.render_framebuffer()

                # Small delay to control frame rate
                time.sleep(0.05)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print(f"\nGame Over! Frames rendered: {self.frame_count}")


if __name__ == "__main__":
    game = InteractiveDOOM()
    game.run()
