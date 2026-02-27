#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          INTERACTIVE DOOM - TRUE GPU EXECUTION                               ║
║                                                                              ║
║  ALL rendering and game logic runs on GPU in ARM64!                          ║
║  - ARM64 raycast.elf binary compiled from raycast.c                          ║
║  - GPU-side syscalls for write() output                                      ║
║  - GPU handles all raycasting, framebuffer rendering                         ║
║  - Python only handles keyboard input and display                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvrm_metal import MetalCPU
import time
import select
import termios
import tty
import fcntl

# Memory map from raycast.c
CODE_ADDR = 0x10000
PLAYER_ADDR = 0x20000  # x, y, angle (each 4 bytes)
MAP_ADDR = 0x30000     # 16x16 map
FB_ADDR = 0x40000      # 80x25 framebuffer
KEY_ADDR = 0x50000     # Keyboard input

# Display settings
FB_WIDTH = 80
FB_HEIGHT = 25

# 16x16 map (1 = wall, 0 = empty) - same as raycast.c expects
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


class TrueGPUDOOM:
    """DOOM running ENTIRELY on GPU with ARM64 raycasting code"""

    def __init__(self):
        print("\033[96m" + "╔══════════════════════════════════════════════════════════════════════════════╗" + "\033[0m")
        print("\033[96m" + "║          INITIALIZE NEURAL GPU CPU - TRUE GPU EXECUTION                      ║" + "\033[0m")
        print("\033[96m" + "╚══════════════════════════════════════════════════════════════════════════════╝" + "\033[0m")
        print()

        self.cpu = MetalCPU(memory_size=16*1024*1024)
        self.running = True

        # Frame timing
        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # Print GPU status
        print("\033[92m" + "✓ GPU initialized" + "\033[0m")
        print("\033[93m" + "  Mode: StorageModeManaged with staging buffers" + "\033[0m")
        print("\033[93m" + "  All raycasting and rendering runs on GPU!" + "\033[0m")
        print()

    def init_game(self):
        """Load ARM64 binary and initialize game state"""
        # Load raycast.elf binary
        with open('arm64_doom/raycast.elf', 'rb') as f:
            binary = f.read()

        # Load program to GPU memory
        self.cpu.load_program(bytearray(binary), CODE_ADDR)

        # Load map data
        self.cpu.write_memory(MAP_ADDR, MAP_DATA)

        # Initialize keyboard input to 0
        self.cpu.write_memory(KEY_ADDR, bytes(4))

        # Set stack pointer
        self.cpu.set_register(31, 0x1000000)

        # Set PC to entry point (0x102E0 is already the virtual address)
        entry_point = 0x102E0
        self.cpu.set_pc(entry_point)

        print("\033[92m" + f"✓ Loaded raycast.elf ({len(binary)} bytes)" + "\033[0m")
        print("\033[93m" + f"  Entry point: 0x{entry_point:X}" + "\033[0m")
        print()

    def handle_input(self, key):
        """Handle keyboard input by writing to GPU memory"""
        # Map key to value and write to KEY_ADDR
        key_value = 0
        if key in ('w', 'W'):
            key_value = ord('w')
        elif key in ('s', 'S'):
            key_value = ord('s')
        elif key in ('a', 'A'):
            key_value = ord('a')
        elif key in ('d', 'D'):
            key_value = ord('d')
        elif key in ('q', 'Q'):
            self.running = False
            return

        # Write key to GPU memory (ARM64 code reads from here)
        self.cpu.write_memory(KEY_ADDR, key_value.to_bytes(4, 'little'))

    def render_framebuffer(self):
        """Render the GPU-generated framebuffer to terminal"""
        # Read framebuffer from GPU memory
        fb_data = self.cpu.read_memory(FB_ADDR, FB_WIDTH * FB_HEIGHT)

        # Clear screen and move cursor to home
        print("\033[H", end="", flush=True)

        # Display header
        print("\033[92m" + f"  ╔══════════════════════════════════════════════════════════════════════════════╗" + "\033[0m")
        print("\033[92m" + "  ║" + "\033[97m" + "  NEURAL DOOM - TRUE GPU EXECUTION - ALL RAYCASTING ON GPU                    " + "\033[92m" + "║" + "\033[0m")
        print("\033[92m" + "  ║" + "\033[93m" + f"  FPS: {self.fps:.1f} | GPU Metal CPU | ARM64 raycast.elf running on GPU             " + "\033[92m" + "║" + "\033[0m")
        print("\033[92m" + f"  ║" + "\033[96m" + "  W/S: Move | A/D: Turn | Q: Quit                                                        " + "\033[92m" + "║" + "\033[0m")
        print("\033[92m" + "  ╚══════════════════════════════════════════════════════════════════════════════╝" + "\033[0m")
        print()

        # Display framebuffer
        for y in range(FB_HEIGHT):
            line = ""
            for x in range(FB_WIDTH):
                offset = y * FB_WIDTH + x
                char = chr(fb_data[offset])

                # Colorize based on character
                if char == '@' or char == 'È':  # È = 0xC8 from ARM64 code
                    line += "\033[97m"  # Bright white (close wall)
                elif char == '#':
                    line += "\033[37m"  # White
                elif char == '%':
                    line += "\033[96m"  # Cyan
                elif char == '*':
                    line += "\033[36m"  # Cyan
                elif char == '+':
                    line += "\033[95m"  # Magenta
                elif char == '=':
                    line += "\033[35m"  # Magenta
                elif char == '-':
                    line += "\033[34m"  # Blue
                elif char == ':':
                    line += "\033[90m"  # Gray
                elif char == '.' or fb_data[offset] == 0:
                    line += "\033[0m"   # Reset (floor or empty)
                else:
                    line += "\033[0m"   # Reset

                line += char

            line += "\033[0m"  # Reset color at end of line
            print(line)

        # Display footer
        print()
        print("\033[92m" + "  ╚══════════════════════════════════════════════════════════════════════════════╝" + "\033[0m")

        # Check for GPU syscall output
        write_output = self.cpu.get_syscall_write_output()
        if write_output:
            text = bytes(write_output).decode('utf-8', errors='ignore')
            if text and text.strip():
                print(text, end='', flush=True)

    def run_frame(self):
        """Execute GPU code for one frame"""
        # Execute ARM64 code on GPU
        # The ARM64 code runs its render_frame() function
        result = self.cpu.execute(max_cycles=500000)

        # Check if program stopped
        if result.stop_reason == 1:  # HALT
            return False

        # Update FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = current_time

        self.frame_count += 1

        return True

    def run(self):
        """Main game loop"""
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set raw mode for immediate input
            tty.setraw(sys.stdin.fileno())

            # NOTE: Don't set O_NONBLOCK - use select with timeout instead
            # This avoids BlockingIOError on stdout writes

            # Clear screen
            print("\033[2J\033[H", end="", flush=True)

            print("\033[96m" + "╔══════════════════════════════════════════════════════════════════════════════╗" + "\033[0m")
            print("\033[96m" + "║                    DOOM ON NEURAL GPU CPU                                  ║" + "\033[0m")
            print("\033[96m" + "║                                                                              ║" + "\033[0m")
            print("\033[96m" + "║  This is TRUE GPU execution!                                                ║" + "\033[0m")
            print("\033[96m" + "║  - ARM64 raycast.elf binary runs on Metal GPU                               ║" + "\033[0m")
            print("\033[96m" + "║  - All raycasting calculations done by GPU                                  ║" + "\033[0m")
            print("\033[96m" + "║  - GPU writes framebuffer directly to memory                                ║" + "\033[0m")
            print("\033[96m" + "║  - GPU-side syscalls handle I/O                                             ║" + "\033[0m")
            print("\033[96m" + "║                                                                              ║" + "\033[0m")
            print("\033[96m" + "║  Controls: W/S/A/D to move and turn, Q to quit                              ║" + "\033[0m")
            print("\033[96m" + "╚══════════════════════════════════════════════════════════════════════════════╝" + "\033[0m")
            print()

            self.init_game()

            print("\033[92m" + "✓ Ready! Starting GPU execution..." + "\033[0m")
            print()
            time.sleep(1)

            while self.running:
                # Check for keyboard input (with short timeout)
                if select.select([sys.stdin], [], [], 0.001)[0]:
                    key = sys.stdin.read(1)
                    self.handle_input(key)

                    if not self.running:
                        break

                # Run GPU frame (ARM64 code does raycasting)
                if not self.run_frame():
                    break

                # Render framebuffer from GPU memory
                self.render_framebuffer()

                # Small delay for frame rate
                time.sleep(0.033)  # ~30 FPS

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\033[2J\033[H", end="", flush=True)
            print("\033[96m" + "Thanks for playing!" + "\033[0m")
            print(f"\033[93m" + f"Frames rendered: {self.frame_count}" + "\033[0m")
            print()


if __name__ == "__main__":
    game = TrueGPUDOOM()
    game.run()
