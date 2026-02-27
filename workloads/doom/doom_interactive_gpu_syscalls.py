#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          INTERACTIVE DOOM - FULLY ON GPU WITH SYSCALLS                    ║
║                                                                              ║
║  - Syscalls handled entirely on GPU (write, read, exit, ioctl, etc)         ║
║  - Interactive keyboard controls                                              ║
║  - Real-time raycasting rendering                                             ║
║  - No Python syscall overhead - GPU autonomous!                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

from kvrm_metal import MetalCPU
import time
import sys
import termios
import tty
import select

# Framebuffer settings
FB_WIDTH = 80
FB_HEIGHT = 25
FB_ADDR = 0x40000

# Player settings
PLAYER_ADDR = 0x20000
MAP_ADDR = 0x30000
CODE_ADDR = 0x10000

# Wall characters for different distances (ASCII only)
WALL_CHARS = "@#%*+=-:. "

# Simple 16x16 map (1 = wall, 0 = empty)
MAP_DATA = bytes([
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


class InteractiveDOOMGPU:
    """Interactive DOOM using GPU-side syscalls"""

    def __init__(self):
        print("Initializing Neural DOOM with GPU-side syscalls...")
        self.cpu = MetalCPU(memory_size=16*1024*1024)
        self.running = True

        # Player state
        self.player_x = 8.0
        self.player_y = 8.0
        self.player_angle = 0  # 0-255 (256 = 360 degrees)

        # Frame timing
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0

    def init_game(self):
        """Initialize the game"""
        # Load raycast binary
        with open('arm64_doom/raycast.elf', 'rb') as f:
            binary = f.read()

        self.cpu.load_program(bytearray(binary), CODE_ADDR)

        # Load map data to memory
        self.cpu.write_memory(MAP_ADDR, MAP_DATA)

        # Initialize player position (fixed point 16.16 format for raycast)
        player_x_fp = int(self.player_x * 65536)
        player_y_fp = int(self.player_y * 65536)

        # Write player state
        self.cpu.write_memory(PLAYER_ADDR, player_x_fp.to_bytes(4, 'little'))
        self.cpu.write_memory(PLAYER_ADDR + 4, player_y_fp.to_bytes(4, 'little'))
        self.cpu.write_memory(PLAYER_ADDR + 8, self.player_angle.to_bytes(4, 'little'))

        # Set stack pointer
        self.cpu.set_register(31, 0x1000000)

        # Set entry point (ELF entry is 0x102E0)
        self.cpu.set_pc(CODE_ADDR + 0x102E0)

        # Reset key state
        self.cpu.set_key_state(0)

    def render_framebuffer(self):
        """Render the ASCII framebuffer"""
        fb_data = self.cpu.read_memory(FB_ADDR, FB_WIDTH * FB_HEIGHT)

        print(f"\n{'=' * 82}")
        print(f"  NEURAL DOOM - FPS: {self.fps:.1f} | Angle: {self.player_angle % 256} | Pos: ({self.player_x:.1f}, {self.player_y:.1f})")
        print(f"{'=' * 82}")

        for y in range(FB_HEIGHT):
            line = ""
            for x in range(FB_WIDTH):
                offset = y * FB_WIDTH + x
                char = chr(fb_data[offset])
                line += char
            print(line)

        print(f"{'=' * 82}")
        print("  W/S: Move | A/D: Turn | Q: Quit")
        print(f"{'=' * 82}")

    def check_input(self):
        """Non-blocking keyboard input check"""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'w' or key == 'W':
                self.cpu.set_key_state(1)
            elif key == 's' or key == 'S':
                self.cpu.set_key_state(2)
            elif key == 'a' or key == 'A':
                self.cpu.set_key_state(3)
            elif key == 'd' or key == 'D':
                self.cpu.set_key_state(4)
            elif key == 'q' or key == 'Q':
                self.cpu.set_key_state(5)
            else:
                self.cpu.set_key_state(0)
        else:
            self.cpu.set_key_state(0)

    def run_frame(self):
        """Run one frame of DOOM"""
        # Execute one frame with GPU-side syscalls
        result = self.cpu.execute(max_cycles=500000)

        # Check for syscall write output
        write_output = self.cpu.get_syscall_write_output()
        if write_output:
            text = bytes(write_output).decode('utf-8', errors='ignore')
            if text and text.strip():
                print(text, end='', flush=True)

        # Update FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = current_time

        self.frame_count += 1

        return result.stop_reason != 1  # Continue if not HALT

    def run(self):
        """Main game loop"""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())

            # Set non-blocking
            import fcntl
            flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            print(f"\n{'=' * 82}")
            print("  NEURAL DOOM - GPU-SIDE SYSCALLS")
            print("  Running on Metal GPU (Apple M4 Pro)")
            print("  Syscalls handled ENTIRELY on GPU - no Python overhead!")
            print(f"{'=' * 82}")
            print("\nInitializing...")

            self.init_game()

            print("Ready! Use W/S/A/D to move, Q to quit\n")

            while self.running:
                # Check input
                self.check_input()

                # Check key state for quit
                syscall_info = self.cpu.get_syscall_info()
                if syscall_info[5] == 5:  # Q = quit
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


import os
import fcntl

if __name__ == "__main__":
    game = InteractiveDOOMGPU()
    game.run()
