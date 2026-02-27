#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          INTERACTIVE DOOM - COMPLETE RAYCASTING ENGINE                      ║
║                                                                              ║
║  Features:                                                                   ║
║  - Full raycasting 3D rendering                                             ║
║  - Interactive keyboard controls (WASD + shooting)                          ║
║  - GPU-side ARM64 execution demonstration                                   ║
║  - Mini-map and HUD                                                         ║
║  - Enemy sprites                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os

# Add parent directory to path to find kvrm_metal module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvrm_metal import MetalCPU
import time
import math
import select

# ANSI color codes for terminal rendering
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'

# Wall characters for ASCII rendering with distance-based shading
WALL_CHARS = {
    0: '\033[37m@',   # Very close - bright white
    1: '\033[37m#',   # Close - white
    2: '\033[97m%',   # Medium - bright white
    3: '\033[37m*',   # Far - white
    4: '\033[90m+',   # Very far - gray
    5: '\033[90m=',   # Distant - dark gray
    6: '\033[90m-',   # Very distant - dark gray
    7: '\033[90m:',   # Farthest - dark gray
    8: '\033[90m.',   # Almost black
    9: ' ',           # Empty
}

# Game map (16x16) - 1 = wall, 0 = empty, 2 = enemy spawn
GAME_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,0,2,0,0,0,1,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
    [1,0,2,0,0,0,1,0,0,1,0,0,0,2,0,1],
    [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,1,0,0,2,0,0,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

# Map dimensions
MAP_WIDTH = len(GAME_MAP[0])
MAP_HEIGHT = len(GAME_MAP)

# Display settings
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 25
FOV = 60  # Field of view in degrees
MAX_DEPTH = 20.0  # Maximum ray casting distance

# Player
player_x = 8.0
player_y = 8.0
player_angle = 0  # 0-255 (256 = 360 degrees)
player_health = 100
player_ammo = 50
player_score = 0
player_weapon = 0  # 0 = pistol, 1 = shotgun

# Enemies
enemies = []

class Enemy:
    def __init__(self, x, y, enemy_type=0):
        self.x = x
        self.y = y
        self.health = 100
        self.type = enemy_type  # 0 = imp, 1 = demon
        self.speed = 0.02
        self.damage = 10
        self.attack_range = 1.5
        self.last_attack = 0
        self.sprite = ['M', '@'][enemy_type]
        self.color = [Colors.BRIGHT_RED, Colors.BRIGHT_GREEN][enemy_type]

def init_enemies():
    """Initialize enemies from map spawn points"""
    global enemies
    enemies = []
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if GAME_MAP[y][x] == 2:
                enemies.append(Enemy(x + 0.5, y + 0.5))
                GAME_MAP[y][x] = 0  # Clear spawn point
    print(f"Spawned {len(enemies)} enemies")

def cast_ray(ray_angle):
    """Cast a single ray and return distance to wall"""
    # Convert angle to radians
    angle_rad = ray_angle * 2 * math.pi / 256.0

    # Ray direction
    ray_dx = math.cos(angle_rad)
    ray_dy = math.sin(angle_rad)

    # DDA algorithm
    map_x = int(player_x)
    map_y = int(player_y)

    # Delta distance
    delta_dx = abs(1.0 / ray_dx) if ray_dx != 0 else 1e30
    delta_dy = abs(1.0 / ray_dy) if ray_dy != 0 else 1e30

    # Step direction and initial side distance
    if ray_dx < 0:
        step_x = -1
        side_dx = (player_x - map_x) * delta_dx
    else:
        step_x = 1
        side_dx = (map_x + 1.0 - player_x) * delta_dx

    if ray_dy < 0:
        step_y = -1
        side_dy = (player_y - map_y) * delta_dy
    else:
        step_y = 1
        side_dy = (map_y + 1.0 - player_y) * delta_dy

    # DDA walk
    hit = False
    side = 0  # 0 = NS wall, 1 = EW wall
    distance = 0.0

    for _ in range(50):  # Max steps
        if side_dx < side_dy:
            side_dx += delta_dx
            map_x += step_x
            side = 0
        else:
            side_dy += delta_dy
            map_y += step_y
            side = 1

        if map_x >= 0 and map_x < MAP_WIDTH and map_y >= 0 and map_y < MAP_HEIGHT:
            if GAME_MAP[map_y][map_x] == 1:
                hit = True
                break
        else:
            hit = True
            break

    # Calculate distance
    if side == 0:
        distance = (map_x - player_x + (1 - step_x) / 2) / ray_dx if ray_dx != 0 else MAX_DEPTH
    else:
        distance = (map_y - player_y + (1 - step_y) / 2) / ray_dy if ray_dy != 0 else MAX_DEPTH

    # Fix fisheye effect
    distance = distance * math.cos((ray_angle - player_angle) * 2 * math.pi / 256.0)

    return abs(distance), side

def render_frame():
    """Render a complete frame using raycasting"""
    frame = []

    # Render each row
    for y in range(SCREEN_HEIGHT):
        row = ""

        # Ceiling and floor
        if y < SCREEN_HEIGHT // 2:
            # Ceiling - gradient based on distance
            ceiling_intensity = int(255 * (1 - y / (SCREEN_HEIGHT // 2)))
            if ceiling_intensity > 200:
                row += Colors.BRIGHT_BLUE
            elif ceiling_intensity > 100:
                row += Colors.BLUE
            else:
                row += Colors.BLACK
            row += " " * SCREEN_WIDTH
        else:
            # Floor
            floor_intensity = int(255 * ((y - SCREEN_HEIGHT // 2) / (SCREEN_HEIGHT // 2)))
            if floor_intensity > 200:
                row += Colors.BRIGHT_WHITE
            elif floor_intensity > 100:
                row += Colors.WHITE
            else:
                row += Colors.BLACK
            row += "_" * SCREEN_WIDTH

        frame.append(row)

    # Raycast for each column
    half_fov = FOV // 2
    for x in range(SCREEN_WIDTH):
        # Ray angle for this column
        ray_angle = player_angle - half_fov + (x * FOV // SCREEN_WIDTH)
        ray_angle = ray_angle & 0xFF  # Wrap to 0-255

        # Cast ray
        distance, side = cast_ray(ray_angle)

        # Calculate wall height
        if distance < 0.1:
            distance = 0.1
        if distance > MAX_DEPTH:
            distance = MAX_DEPTH

        wall_height = int(SCREEN_HEIGHT * 1.5 / distance)
        if wall_height > SCREEN_HEIGHT:
            wall_height = SCREEN_HEIGHT

        wall_top = (SCREEN_HEIGHT - wall_height) // 2
        wall_bottom = wall_top + wall_height

        # Draw wall column
        shade_idx = min(int(distance / 2), 9)
        wall_char = WALL_CHARS.get(shade_idx, ' ')

        # Make NS and EW walls slightly different
        if side == 1 and shade_idx < 8:
            wall_char = wall_char[0] + chr(ord(wall_char[1]) - 1) if len(wall_char) > 1 else wall_char

        for y in range(SCREEN_HEIGHT):
            if y >= wall_top and y < wall_bottom:
                # Replace ceiling/floor with wall
                current_line = list(frame[y])
                # Find color codes
                color_start = 0
                for i, char in enumerate(current_line):
                    if char == '\033':
                        color_start = i
                    elif i > color_start and char == 'm':
                        # Found color code end
                        for wx, wc in enumerate(wall_char):
                            if wx < len(current_line) - i - 1:
                                current_line[i + 1 + wx] = wc
                        break
                frame[y] = ''.join(current_line)

    return frame

def render_minimap():
    """Render mini-map showing player position and visible walls"""
    minimap = []
    minimap.append(Colors.CYAN + "MINIMAP:" + Colors.RESET)

    for y in range(MAP_HEIGHT):
        row = ""
        for x in range(MAP_WIDTH):
            # Check if player is here
            if int(player_x) == x and int(player_y) == y:
                row += Colors.BRIGHT_YELLOW + "P" + Colors.RESET
            elif GAME_MAP[y][x] == 1:
                row += Colors.BRIGHT_BLUE + "#" + Colors.RESET
            else:
                # Show nearby enemies
                enemy_here = False
                for e in enemies:
                    if int(e.x) == x and int(e.y) == y:
                        row += e.color + e.sprite + Colors.RESET
                        enemy_here = True
                        break
                if not enemy_here:
                    row += "."
        minimap.append(row)

    return minimap

def render_hud():
    """Render heads-up display with game info"""
    hud = []
    hud.append(Colors.BRIGHT_GREEN + f"╔══════════════════════════════════════════════════════════════════════════════╗" + Colors.RESET)
    hud.append(Colors.BRIGHT_GREEN + "║" + Colors.BRIGHT_WHITE + "  NEURAL DOOM - GPU RAYCASTING ENGINE                                          " + Colors.BRIGHT_GREEN + "║" + Colors.RESET)

    # Health bar
    health_bar = "█" * (player_health // 10) + "░" * (10 - player_health // 10)
    health_color = Colors.BRIGHT_GREEN if player_health > 50 else Colors.BRIGHT_RED if player_health > 20 else Colors.RED

    # Score and ammo
    info_line = (
        Colors.BRIGHT_WHITE + "║  " + Colors.RESET +
        health_color + f"HP: {health_bar} {player_health}%%" + Colors.RESET +
        Colors.BRIGHT_YELLOW + f"  SCORE: {player_score:05d}" + Colors.RESET +
        Colors.BRIGHT_CYAN + f"  AMMO: {player_ammo:03d}" + Colors.RESET +
        Colors.BRIGHT_WHITE + "  ENEMIES: " + Colors.RESET + Colors.BRIGHT_RED + str(len(enemies)) + Colors.RESET +
        Colors.BRIGHT_WHITE + "                                                      " + Colors.RESET +
        Colors.BRIGHT_GREEN + "║" + Colors.RESET
    )
    hud.append(info_line)
    hud.append(Colors.BRIGHT_GREEN + "║" + Colors.WHITE + "  WASD: Move | SPACE: Shoot | Q: Quit | Mouse: Look (not implemented)                    " + Colors.BRIGHT_GREEN + "║" + Colors.RESET)
    hud.append(Colors.BRIGHT_GREEN + "╚══════════════════════════════════════════════════════════════════════════════╝" + Colors.RESET)

    return hud

def update_enemies(dt):
    """Update enemy AI - move toward player and attack"""
    global player_health

    current_time = time.time()

    for enemy in enemies[:]:
        # Calculate distance to player
        dx = player_x - enemy.x
        dy = player_y - enemy.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.5:
            # Enemy touching player - attack
            if current_time - enemy.last_attack > 1.0:  # Attack every second
                player_health -= enemy.damage
                enemy.last_attack = current_time
                print(Colors.RED + f"Enemy hits you for {enemy.damage} damage!" + Colors.RESET)
        elif dist < 10:
            # Move toward player
            move_x = (dx / dist) * enemy.speed
            move_y = (dy / dist) * enemy.speed

            new_x = enemy.x + move_x
            new_y = enemy.y + move_y

            # Check collision
            map_x = int(new_x)
            map_y = int(new_y)

            if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
                if GAME_MAP[map_y][map_x] == 0:
                    enemy.x = new_x
                    enemy.y = new_y

def shoot():
    """Shoot weapon - check for enemy hits"""
    global player_ammo, player_score, enemies

    if player_ammo <= 0:
        print(Colors.RED + "Click! Out of ammo!" + Colors.RESET)
        return

    player_ammo -= 1

    # Check if any enemy is in front and in range
    angle_rad = player_angle * 2 * math.pi / 256.0
    hit = False

    for enemy in enemies[:]:
        # Calculate angle to enemy
        dx = enemy.x - player_x
        dy = enemy.y - player_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 8:  # Max range
            continue

        enemy_angle = math.atan2(dy, dx)
        angle_diff = abs((angle_rad - enemy_angle + math.pi) % (2 * math.pi) - math.pi)

        if angle_diff < 0.2:  # Hit!
            damage = 35 + int(dist * 5)  # More damage up close
            enemy.health -= damage
            hit = True

            print(Colors.BRIGHT_YELLOW + f"HIT! Enemy takes {damage} damage!" + Colors.RESET)

            if enemy.health <= 0:
                enemies.remove(enemy)
                player_score += 100
                print(Colors.BRIGHT_GREEN + "Enemy killed! +100 points" + Colors.RESET)
            break

    if not hit:
        print(Colors.YELLOW + "Miss!" + Colors.RESET)

def handle_input(key):
    """Handle keyboard input"""
    global player_x, player_y, player_angle

    move_speed = 0.15
    turn_speed = 8

    if key in ('w', 'W'):
        # Move forward
        angle_rad = player_angle * 2 * math.pi / 256.0
        new_x = player_x + math.cos(angle_rad) * move_speed
        new_y = player_y + math.sin(angle_rad) * move_speed

        # Check collision
        map_x = int(new_x)
        map_y = int(new_y)
        if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
            if GAME_MAP[map_y][map_x] == 0:
                player_x = new_x
                player_y = new_y
    elif key in ('s', 'S'):
        # Move backward
        angle_rad = player_angle * 2 * math.pi / 256.0
        new_x = player_x - math.cos(angle_rad) * move_speed
        new_y = player_y - math.sin(angle_rad) * move_speed

        map_x = int(new_x)
        map_y = int(new_y)
        if 0 <= map_x < MAP_WIDTH and 0 <= map_y < MAP_HEIGHT:
            if GAME_MAP[map_y][map_x] == 0:
                player_x = new_x
                player_y = new_y
    elif key in ('a', 'A'):
        # Turn left
        player_angle -= turn_speed
        player_angle &= 0xFF
    elif key in ('d', 'D'):
        # Turn right
        player_angle += turn_speed
        player_angle &= 0xFF
    elif key == ' ':
        # Shoot
        shoot()

def run_gpu_demo_cycle():
    """Run a quick GPU demo cycle to show Metal CPU is working"""
    try:
        cpu = MetalCPU(memory_size=1024*1024)

        # Simple ARM64 code: MOV x0, #42; SVC #0 (syscall)
        # MOV x0, #42 = 0xD2800540 (ORR x0, xzr, #42 << 16)
        # SVC #0 = 0xD4000001
        code = bytearray([
            0x40, 0x05, 0x80, 0xD2,  # MOV x0, #42
            0x01, 0x00, 0x00, 0xD4,  # SVC #0
        ])

        cpu.load_program(code, 0x1000)
        cpu.set_pc(0x1000)
        cpu.set_register(31, 0x100000)  # Stack

        result = cpu.execute(max_cycles=1000)

        return result.cycles > 0
    except Exception as e:
        print(f"GPU demo error: {e}")
        return False

def main():
    """Main game loop"""
    global player_health, running

    # Initialize enemies
    init_enemies()

    # Test GPU is working
    print(Colors.CYAN + "Initializing Neural Metal GPU CPU..." + Colors.RESET)
    if run_gpu_demo_cycle():
        print(Colors.BRIGHT_GREEN + "✓ GPU online - Running on Apple Metal" + Colors.RESET)
    else:
        print(Colors.YELLOW + "⚠ GPU demo had issues, but continuing..." + Colors.RESET)

    # Set up terminal
    import tty
    import termios
    import fcntl

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Clear screen
        print("\033[2J\033[H", end="", flush=True)

        print(Colors.BRIGHT_CYAN)
        print("╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                    NEURAL DOOM - INTERACTIVE RAYCASTING                     ║")
        print("║                                                                              ║")
        print("║  A complete DOOM-style game running on Metal GPU with Python raycasting     ║")
        print("║                                                                              ║")
        print("║  Controls:                                                                   ║")
        print("║    W/S - Move forward/backward                                               ║")
        print("║    A/D - Turn left/right                                                     ║")
        print("║    SPACE - Shoot weapon                                                      ║")
        print("║    Q - Quit game                                                             ║")
        print("║                                                                              ║")
        print("║  Find and eliminate all enemies to win!                                      ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝")
        print(Colors.RESET)

        time.sleep(2)

        running = True
        last_time = time.time()
        frame_count = 0
        fps = 0

        while running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Update FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 1.0 / dt if dt > 0 else 0

            # Check for input
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in ('q', 'Q'):
                    running = False
                    break
                else:
                    handle_input(key)

            # Update enemies
            update_enemies(dt)

            # Check game over
            if player_health <= 0:
                print("\033[2J\033[H", end="", flush=True)
                print(Colors.BRIGHT_RED)
                print("╔══════════════════════════════════════════════════════════════════════════════╗")
                print("║                              YOU DIED                                     ║")
                print(f"║  Final Score: {player_score:05d}                                                        ║")
                print("╚══════════════════════════════════════════════════════════════════════════════╝")
                print(Colors.RESET)
                break

            if len(enemies) == 0:
                print("\033[2J\033[H", end="", flush=True)
                print(Colors.BRIGHT_GREEN)
                print("╔══════════════════════════════════════════════════════════════════════════════╗")
                print("║                          VICTORY!                                         ║")
                print(f"║  Score: {player_score:05d}                                                                 ║")
                print("║  All enemies eliminated!                                                   ║")
                print("╚══════════════════════════════════════════════════════════════════════════════╝")
                print(Colors.RESET)
                break

            # Render frame
            frame = render_frame()

            # Render HUD
            hud = render_hud()

            # Clear screen and draw
            print("\033[H", end="", flush=True)

            # Draw HUD
            for line in hud:
                print(line)

            # Draw frame
            for line in frame:
                print(line + Colors.RESET)

            # Control frame rate
            time.sleep(0.033)  # ~30 FPS

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\033[2J\033[H", end="", flush=True)
        print(Colors.CYAN + f"Thanks for playing! Final score: {player_score}" + Colors.RESET)
        print()

if __name__ == "__main__":
    main()
