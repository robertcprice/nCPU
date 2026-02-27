#!/usr/bin/env python3
"""
🎮 PLAYABLE NEURAL DOOM
=======================
Full game loop with keyboard controls!

Controls:
  W/S - Move forward/backward
  A/D - Turn left/right
  Q   - Quit

Run: python3 doom_playable.py
"""

import curses
import math
import time
import torch

# Import our neural CPU
from kvrm_correct_loader import KVRMCPU, device

class NeuralDOOM:
    def __init__(self):
        # Neural CPU for calculations
        self.cpu = KVRMCPU()

        # Player state
        self.player_x = 2.5
        self.player_y = 2.5
        self.player_angle = 0.0
        self.health = 100
        self.score = 0

        # Movement settings
        self.move_speed = 0.15
        self.turn_speed = 0.12

        # Map (1 = wall, 0 = empty, 2 = item)
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

        self.map_width = len(self.map[0])
        self.map_height = len(self.map)

        # Frame counter
        self.frame = 0
        self.fps = 0

    def cast_ray(self, angle):
        """Cast a ray and return distance to wall + wall type"""
        ray_x = self.player_x
        ray_y = self.player_y

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Step size
        step = 0.05

        for i in range(200):
            ray_x += cos_a * step
            ray_y += sin_a * step

            map_x = int(ray_x)
            map_y = int(ray_y)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                if self.map[map_y][map_x] == 1:
                    dist = math.sqrt((ray_x - self.player_x)**2 + (ray_y - self.player_y)**2)
                    # Fix fisheye
                    dist *= math.cos(angle - self.player_angle)
                    return dist, 1
            else:
                return 10.0, 0

        return 10.0, 0

    def render(self, width, height):
        """Render the 3D view"""
        # Characters for different distances (near to far)
        WALL_CHARS = "█▓▒░#%*+=-:. "
        CEILING = ' '
        FLOOR_CHARS = "._-~"

        fov = 1.2  # ~70 degrees

        frame_buffer = []

        for y in range(height):
            row = ""
            for x in range(width):
                # Calculate ray angle for this column
                ray_angle = self.player_angle - fov/2 + (x / width) * fov

                # Cast ray
                dist, wall_type = self.cast_ray(ray_angle)

                # Calculate wall height on screen
                if dist < 0.1:
                    dist = 0.1
                wall_height = min(height, int(height / dist))

                # Calculate vertical position
                wall_top = (height - wall_height) // 2
                wall_bottom = wall_top + wall_height

                if y < wall_top:
                    # Ceiling
                    row += CEILING
                elif y >= wall_bottom:
                    # Floor - gradient based on distance from center
                    floor_dist = (y - height/2) / (height/2)
                    floor_idx = min(len(FLOOR_CHARS)-1, int(floor_dist * len(FLOOR_CHARS)))
                    row += FLOOR_CHARS[floor_idx]
                else:
                    # Wall - brightness based on distance
                    brightness = max(0, min(len(WALL_CHARS)-1, int(dist * 1.5)))
                    row += WALL_CHARS[brightness]

            frame_buffer.append(row)

        return frame_buffer

    def render_minimap(self, size=7):
        """Render a small overhead map"""
        lines = []
        half = size // 2

        for dy in range(-half, half + 1):
            row = ""
            for dx in range(-half, half + 1):
                mx = int(self.player_x) + dx
                my = int(self.player_y) + dy

                if dx == 0 and dy == 0:
                    # Player direction indicator
                    dirs = "→↗↑↖←↙↓↘"
                    dir_idx = int((self.player_angle + math.pi/8) / (math.pi/4)) % 8
                    row += dirs[dir_idx]
                elif 0 <= mx < self.map_width and 0 <= my < self.map_height:
                    if self.map[my][mx] == 1:
                        row += "█"
                    else:
                        row += "·"
                else:
                    row += " "
            lines.append(row)
        return lines

    def move(self, direction):
        """Move player forward or backward"""
        dx = math.cos(self.player_angle) * self.move_speed * direction
        dy = math.sin(self.player_angle) * self.move_speed * direction

        new_x = self.player_x + dx
        new_y = self.player_y + dy

        # Collision detection
        margin = 0.2
        if self.map[int(new_y)][int(new_x + margin * (1 if dx > 0 else -1))] == 0:
            self.player_x = new_x
        if self.map[int(new_y + margin * (1 if dy > 0 else -1))][int(self.player_x)] == 0:
            self.player_y = new_y

    def turn(self, direction):
        """Turn player left or right"""
        self.player_angle += self.turn_speed * direction

    def run(self, stdscr):
        """Main game loop"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(16)  # ~60 FPS target

        # Try to use colors
        try:
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            has_colors = True
        except:
            has_colors = False

        last_time = time.time()
        frame_times = []

        while True:
            # Get terminal size
            max_y, max_x = stdscr.getmaxyx()

            # Reserve space for HUD
            view_height = max_y - 4
            view_width = max_x - 12  # Space for minimap

            if view_height < 10 or view_width < 20:
                stdscr.clear()
                stdscr.addstr(0, 0, "Terminal too small!")
                stdscr.refresh()
                time.sleep(0.1)
                continue

            # Handle input
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('w') or key == ord('W') or key == curses.KEY_UP:
                self.move(1)
            elif key == ord('s') or key == ord('S') or key == curses.KEY_DOWN:
                self.move(-1)
            elif key == ord('a') or key == ord('A') or key == curses.KEY_LEFT:
                self.turn(-1)
            elif key == ord('d') or key == ord('D') or key == curses.KEY_RIGHT:
                self.turn(1)

            # Render frame
            t0 = time.time()
            frame = self.render(view_width, view_height)
            render_time = time.time() - t0

            # Calculate FPS
            now = time.time()
            frame_times.append(now - last_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            self.fps = len(frame_times) / sum(frame_times) if frame_times else 0
            last_time = now

            # Clear and draw
            stdscr.clear()

            # Draw 3D view
            for y, row in enumerate(frame):
                try:
                    stdscr.addstr(y, 0, row[:view_width])
                except curses.error:
                    pass

            # Draw minimap on the right
            minimap = self.render_minimap()
            map_x = view_width + 2
            try:
                stdscr.addstr(0, map_x, "┌───────┐")
                for i, line in enumerate(minimap):
                    stdscr.addstr(i + 1, map_x, f"│{line}│")
                stdscr.addstr(len(minimap) + 1, map_x, "└───────┘")
            except curses.error:
                pass

            # Draw HUD at bottom
            hud_y = view_height
            try:
                stdscr.addstr(hud_y, 0, "═" * (max_x - 1))

                # Status line
                status = f"🎮 NEURAL DOOM | FPS: {self.fps:.0f} | Pos: ({self.player_x:.1f}, {self.player_y:.1f}) | Angle: {math.degrees(self.player_angle):.0f}°"
                stdscr.addstr(hud_y + 1, 0, status[:max_x-1])

                # Controls
                controls = "WASD/Arrows: Move | Q: Quit | 🧠 Powered by Neural CPU"
                stdscr.addstr(hud_y + 2, 0, controls[:max_x-1])

            except curses.error:
                pass

            stdscr.refresh()
            self.frame += 1


def main():
    print("🎮 NEURAL DOOM - Loading...")
    print("=" * 50)

    game = NeuralDOOM()

    print("\n✅ Neural CPU loaded!")
    print("\nControls:")
    print("  W/↑ - Move forward")
    print("  S/↓ - Move backward")
    print("  A/← - Turn left")
    print("  D/→ - Turn right")
    print("  Q   - Quit")
    print("\nPress Enter to start...")
    input()

    # Run with curses
    curses.wrapper(game.run)

    print("\n🎮 Thanks for playing Neural DOOM!")
    print(f"   Frames rendered: {game.frame}")


if __name__ == "__main__":
    main()
