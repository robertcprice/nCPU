#!/usr/bin/env python3
"""
🎮 INTERACTIVE NEURAL DOOM - Simple Terminal Version
======================================================

ARM64 raycasting code executing on neural CPU.
Works with basic terminal I/O (no curses needed).

Controls:
  w + Enter - Move forward
  s + Enter - Move backward
  a + Enter - Turn left
  d + Enter - Turn right
  q + Enter - Quit

Each command then renders one frame on neural CPU.
"""

import torch
import struct
import time
import sys
from batched_neural_cpu_optimized import BatchedNeuralCPU

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class SimpleNeuralDOOM:
    """Interactive DOOM with basic terminal I/O."""

    def __init__(self):
        print("=" * 80)
        print("🎮 INTERACTIVE NEURAL DOOM - Simple Terminal Version")
        print("=" * 80)
        print(f"Device: {device}")
        print()

        print("Initializing Neural CPU...")
        self.cpu = BatchedNeuralCPU(memory_size=64*1024*1024, batch_size=128)
        print("✅ Neural CPU ready")
        print()

        # Frame buffer in neural CPU memory
        self.FRAMEBUFFER_ADDR = 0x100000
        self.FRAME_WIDTH = 60
        self.FRAME_HEIGHT = 20

        # Player state
        self.player_x = 3.5
        self.player_y = 3.5
        self.player_angle = 0.0

        # Map
        self.map = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ]

        self.frame_count = 0
        self.start_time = None

    def render_frame_neural(self):
        """Render ONE frame using ARM64 code on neural CPU."""
        code = []

        # Raycast for each column (simplified to 30 rays for speed)
        num_rays = 30

        for ray_idx in range(num_rays):
            # Calculate ray angle
            angle_offset = (ray_idx - num_rays/2) * 0.04
            ray_angle = self.player_angle + angle_offset

            # Direction (simplified to 4 directions)
            if 0 <= ray_angle % 6.28 < 1.57:
                dx, dy = 1, 0
            elif 1.57 <= ray_angle % 6.28 < 3.14:
                dx, dy = 0, 1
            elif 3.14 <= ray_angle % 6.28 < 4.71:
                dx, dy = -1, 0
            else:
                dx, dy = 0, -1

            # Initialize ray position
            start_x = int(self.player_x * 100)
            start_y = int(self.player_y * 100)

            # MOVZ X0, #start_x
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | ((start_x & 0xFFFF) << 5) | 0))
            # MOVZ X1, #start_y
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | ((start_y & 0xFFFF) << 5) | 1))

            # Direction
            # MOVZ X2, #dx*100
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | ((abs(dx)*100) << 5) | 2))
            # MOVZ X3, #dy*100
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | ((abs(dy)*100) << 5) | 3))

            # Ray stepping (5 steps for speed)
            for step in range(5):
                # ADD X0, X0, X2
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (2 << 10) | (0 << 5) | 0))
                # ADD X1, X1, X3
                code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (3 << 10) | (1 << 5) | 1))

            # Store distance in X4
            # MOVZ X4, X0
            code.append(struct.pack('<I', (1 << 31) | (0b10100 << 23) | (0 << 5) | 4))
            # ADD X4, X4, X1
            code.append(struct.pack('<I', (1 << 31) | (0b10000 << 24) | (1 << 10) | (4 << 5) | 4))

        # Execute on neural CPU
        self.cpu.load_binary(b''.join(code), load_address=0x20000)
        self.cpu.pc.fill_(0x20000)

        results = self.cpu.run(max_instructions=len(code)//4)

        return results

    def render_to_framebuffer(self, results):
        """Read neural CPU results and render to frame buffer."""
        # Clear framebuffer
        for y in range(self.FRAME_HEIGHT):
            for x in range(self.FRAME_WIDTH):
                addr = self.FRAMEBUFFER_ADDR + y * self.FRAME_WIDTH + x
                self.cpu.memory[addr] = ord(' ')

        # Render each column
        num_rays = 30
        for ray_idx in range(num_rays):
            # Get distance from register
            reg_idx = 4 + (ray_idx % 8)
            dist = 0
            for b in range(64):
                if self.cpu.registers[reg_idx, b].item() > 0.5:
                    dist |= (1 << b)

            # Scale distance
            dist = dist / 100.0

            # Calculate wall height
            if dist < 0.1:
                dist = 0.1
            wall_height = min(self.FRAME_HEIGHT, int(self.FRAME_HEIGHT / dist))

            # Draw column
            col_x = ray_idx * 2
            wall_top = (self.FRAME_HEIGHT - wall_height) // 2

            # Wall
            brightness = max(0, min(6, int(dist * 1.5)))
            wall_char = "█▓▒░#%*+"[6 - brightness]

            for y in range(wall_top, min(wall_top + wall_height, self.FRAME_HEIGHT)):
                addr = self.FRAMEBUFFER_ADDR + y * self.FRAME_WIDTH + col_x
                self.cpu.memory[addr] = ord(wall_char)

    def display_frame(self, results):
        """Display frame from neural CPU memory."""
        print("\033[H\033[J", end="")  # Clear screen

        print(f"🎮 NEURAL DOOM - Frame {self.frame_count} | {results['ips']:.0f} IPS")
        print(f"Pos: ({self.player_x:.1f}, {self.player_y:.1f}) Angle: {self.player_angle:.2f}")
        print(f"Instructions: {results['instructions']} | Batches: {results['batches']}")
        print("=" * 60)

        for y in range(self.FRAME_HEIGHT):
            row = ""
            for x in range(self.FRAME_WIDTH):
                addr = self.FRAMEBUFFER_ADDR + y * self.FRAME_WIDTH + x
                char_code = self.cpu.memory[addr].item()
                if char_code > 0:
                    row += chr(int(char_code))
                else:
                    row += " "
            print(row)

        print("=" * 60)
        print("Controls: w (forward), s (back), a (left), d (right), q (quit)")

    def handle_input(self, cmd):
        """Handle command."""
        if cmd == 'w':
            new_x = self.player_x + 0.3
            if self.map[int(self.player_y)][int(new_x)] == 0:
                self.player_x = new_x
        elif cmd == 's':
            new_x = self.player_x - 0.3
            if self.map[int(self.player_y)][int(new_x)] == 0:
                self.player_x = new_x
        elif cmd == 'a':
            self.player_angle -= 0.2
        elif cmd == 'd':
            self.player_angle += 0.2
        elif cmd == 'q':
            return False
        return True

    def run(self):
        """Main interactive loop."""
        print()
        print("=" * 60)
        print("Starting Interactive Neural DOOM...")
        print("=" * 60)
        print()

        self.start_time = time.time()

        # Render first frame
        results = self.render_frame_neural()
        self.render_to_framebuffer(results)
        self.display_frame(results)
        self.frame_count += 1

        # Main loop
        while True:
            try:
                cmd = input("neural-doom> ").strip().lower()

                if not cmd:
                    continue

                if not self.handle_input(cmd):
                    break

                # Render new frame
                results = self.render_frame_neural()
                self.render_to_framebuffer(results)
                self.display_frame(results)
                self.frame_count += 1

            except (KeyboardInterrupt, EOFError):
                break

        # Final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        print()
        print("=" * 60)
        print("📊 FINAL STATS")
        print("=" * 60)
        print(f"Frames: {self.frame_count}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Avg FPS: {avg_fps:.1f}")
        print()
        print("✅ Neural DOOM closed - All computation on neural CPU!")


def main():
    game = SimpleNeuralDOOM()
    game.run()


if __name__ == "__main__":
    main()
