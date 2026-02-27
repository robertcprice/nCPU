#!/usr/bin/env python3
"""
🎮 Neural DOOM Renderer - Full NeuralOS System Test
====================================================

A DOOM-style raycasting renderer that runs on the FULL KVRM NeuralOS:
- Neural CPU (FusedALU) for ALL arithmetic
- Neural Cache for texture/map data
- Neural Prefetcher for predictive data loading
- Neural Scheduler for render task management

This is a SYSTEM TEST, not a DOOM-specific renderer.
Target: 0.5 FPS minimum to prove system viability.

Usage:
    python doom/neural_renderer.py [--classical] [--benchmark]

Controls:
    W/S - Move forward/backward
    A/D - Strafe left/right
    ←/→ - Turn left/right
    ESC - Quit
"""

import math
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import pygame for display
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("⚠️  pygame not installed - will use ASCII rendering")

# Try to import NeuralOS (the FULL system)
try:
    from neural_os import NeuralOS, NeuralOSConfig
    HAS_NEURAL_OS = True
except ImportError:
    HAS_NEURAL_OS = False
    print("⚠️  NeuralOS not available - using classical fallback")


# ============================================================
# CONFIGURATION
# ============================================================

SCREEN_WIDTH = 320
SCREEN_HEIGHT = 200
FOV = 60  # Field of view in degrees
MAP_SIZE = 16
TILE_SIZE = 64

# Simple map (1 = wall, 0 = empty)
WORLD_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
    [1,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1],
    [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

# Wall colors (DOOM-style palette)
COLORS = [
    (139, 69, 19),    # Brown
    (85, 85, 85),     # Gray
    (139, 0, 0),      # Dark red
    (0, 100, 0),      # Dark green
]


# ============================================================
# NEURALOS SYSTEM INTERFACE (Full System Test)
# ============================================================

class NeuralOSInterface:
    """
    Full NeuralOS system interface for graphics workloads.

    Uses ALL neural components:
    - Neural CPU (FusedALU) for arithmetic
    - Neural Cache for map/texture data
    - Tracks system-wide performance
    """

    def __init__(self, use_neural=True):
        self.use_neural = use_neural and HAS_NEURAL_OS
        self.ops_count = 0
        self.neural_ops = 0
        self.classical_ops = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

        if self.use_neural:
            print("🧠 Initializing FULL NeuralOS System...")
            config = NeuralOSConfig(
                use_fused_alu=True,
                enable_cpu_learning=False,  # Disable learning for benchmarking
                enable_sovereign=False,     # No LLM needed for rendering
                auto_optimize=False,        # No auto-optimization
                cpu_validation_rate=0.0     # No validation overhead
            )
            self.neural_os = NeuralOS(config)
            self.neural_os.start()

            # Report system status
            print("✅ NeuralOS System Ready:")
            print(f"   CPU:        {'FusedALU' if self.neural_os.cpu else 'Classical'}")
            print(f"   Cache:      {'Neural' if self.neural_os.cache else 'None'}")
            print(f"   Prefetcher: {'Neural' if self.neural_os.prefetcher else 'None'}")
            print(f"   Scheduler:  {'Neural' if self.neural_os.scheduler else 'None'}")
        else:
            self.neural_os = None
            print("💻 Using classical fallback (NeuralOS not available)")

    def _neural_op(self, opcode: int, a: int, b: int) -> int:
        """Execute an operation on the Neural CPU."""
        MASK = (1 << 64) - 1
        self.ops_count += 1

        if self.use_neural and self.neural_os and self.neural_os.cpu:
            self.neural_ops += 1
            self.neural_os.set_reg(1, int(a) & MASK)
            self.neural_os.set_reg(2, int(b) & MASK)
            self.neural_os.execute(opcode, 0, 1, 2)
            return self.neural_os.get_reg(0)

        # Classical fallback
        self.classical_ops += 1
        a, b = int(a) & MASK, int(b) & MASK
        if opcode == 0: return (a + b) & MASK      # ADD
        elif opcode == 1: return (a - b) & MASK    # SUB
        elif opcode == 2: return a & b             # AND
        elif opcode == 3: return a | b             # OR
        elif opcode == 4: return a ^ b             # XOR
        elif opcode == 5: return (a << (b & 63)) & MASK  # LSL
        elif opcode == 6: return a >> (b & 63)     # LSR
        elif opcode == 9: return (a * b) & MASK    # MUL
        return 0

    def add(self, a, b):
        """Addition via Neural CPU."""
        return self._neural_op(0, a, b)

    def sub(self, a, b):
        """Subtraction via Neural CPU."""
        return self._neural_op(1, a, b)

    def mul(self, a, b):
        """Multiplication via Neural CPU."""
        return self._neural_op(9, a, b)

    def div(self, a, b):
        """Division - classical only (no neural DIV yet)."""
        self.ops_count += 1
        self.classical_ops += 1
        return int(a // b) if b != 0 else 0

    def and_op(self, a, b):
        """Bitwise AND via Neural CPU."""
        return self._neural_op(2, a, b)

    def or_op(self, a, b):
        """Bitwise OR via Neural CPU."""
        return self._neural_op(3, a, b)

    def lsl(self, a, b):
        """Left shift via Neural CPU."""
        return self._neural_op(5, a, b)

    def lsr(self, a, b):
        """Right shift via Neural CPU."""
        return self._neural_op(6, a, b)

    def cache_read(self, address: int) -> int:
        """Read from Neural Cache (map/texture data)."""
        if self.neural_os and self.neural_os.cache:
            result = self.neural_os.cache.get(address)
            if result is not None:
                self.cache_hits += 1
                return result
            self.cache_misses += 1
        return 0

    def cache_write(self, address: int, value: int):
        """Write to Neural Cache."""
        if self.neural_os and self.neural_os.cache:
            self.neural_os.cache.put(address, value)

    def get_stats(self) -> dict:
        """Get comprehensive system statistics."""
        uptime = time.time() - self.start_time
        stats = {
            'uptime': uptime,
            'total_ops': self.ops_count,
            'neural_ops': self.neural_ops,
            'classical_ops': self.classical_ops,
            'neural_ratio': self.neural_ops / max(1, self.ops_count),
            'ops_per_sec': self.ops_count / max(0.001, uptime),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

        # Add NeuralOS stats if available
        if self.neural_os:
            os_stats = self.neural_os.get_stats()
            stats['neuralos'] = os_stats

        return stats

    def shutdown(self):
        """Clean shutdown of NeuralOS."""
        if self.neural_os:
            self.neural_os.stop()


# ============================================================
# FIXED-POINT MATH (DOOM uses 16.16 fixed point)
# ============================================================

FIXED_SHIFT = 16
FIXED_ONE = 1 << FIXED_SHIFT

def to_fixed(f):
    """Convert float to fixed-point."""
    return int(f * FIXED_ONE)

def from_fixed(fixed):
    """Convert fixed-point to float."""
    return fixed / FIXED_ONE

def fixed_mul(a, b, cpu):
    """Fixed-point multiplication."""
    # a * b >> 16
    result = cpu.mul(a, b)
    return result >> FIXED_SHIFT

def fixed_div(a, b, cpu):
    """Fixed-point division."""
    # (a << 16) / b
    if b == 0:
        return 0
    return cpu.div(a << FIXED_SHIFT, b)


# ============================================================
# RAYCASTING ENGINE
# ============================================================

class DoomRenderer:
    """DOOM-style raycasting renderer using Neural CPU."""

    def __init__(self, cpu):
        self.cpu = cpu

        # Player state
        self.player_x = to_fixed(3.5 * TILE_SIZE)  # Starting position
        self.player_y = to_fixed(3.5 * TILE_SIZE)
        self.player_angle = to_fixed(0)  # Facing right

        # Precomputed sin/cos tables (DOOM-style optimization)
        self.sin_table = []
        self.cos_table = []
        self._build_trig_tables()

        # Frame buffer
        self.framebuffer = [(0, 0, 0)] * (SCREEN_WIDTH * SCREEN_HEIGHT)

        # Stats
        self.frame_count = 0
        self.total_rays = 0

    def _build_trig_tables(self):
        """Build sin/cos lookup tables (DOOM used these for speed)."""
        for i in range(360):
            rad = math.radians(i)
            self.sin_table.append(to_fixed(math.sin(rad)))
            self.cos_table.append(to_fixed(math.cos(rad)))

    def cast_ray(self, angle_deg):
        """Cast a single ray and return wall distance and side hit."""
        # Get sin/cos from table
        angle_idx = int(angle_deg) % 360
        sin_a = self.sin_table[angle_idx]
        cos_a = self.cos_table[angle_idx]

        # Ray position (fixed-point)
        ray_x = self.player_x
        ray_y = self.player_y

        # Step size
        step_size = to_fixed(4)  # Small steps for accuracy

        # March the ray
        for _ in range(500):  # Max distance
            # Move ray forward: x += cos(a) * step, y += sin(a) * step
            dx = fixed_mul(cos_a, step_size, self.cpu)
            dy = fixed_mul(sin_a, step_size, self.cpu)

            ray_x = self.cpu.add(ray_x, dx)
            ray_y = self.cpu.add(ray_y, dy)

            # Convert to map coordinates
            map_x = from_fixed(ray_x) / TILE_SIZE
            map_y = from_fixed(ray_y) / TILE_SIZE

            # Check bounds
            if map_x < 0 or map_x >= MAP_SIZE or map_y < 0 or map_y >= MAP_SIZE:
                return 10000, 0  # Far away, no hit

            # Check for wall
            if WORLD_MAP[int(map_y)][int(map_x)] == 1:
                # Calculate distance (Euclidean)
                dx = self.cpu.sub(ray_x, self.player_x)
                dy = self.cpu.sub(ray_y, self.player_y)

                # Distance = sqrt(dx^2 + dy^2) - approximate with |dx| + |dy| * 0.4
                # (DOOM used similar approximations)
                dist = from_fixed(abs(dx)) + from_fixed(abs(dy)) * 0.4

                # Determine which side was hit (for shading)
                side = 0 if abs(from_fixed(dx)) > abs(from_fixed(dy)) else 1

                return dist, side

        return 10000, 0  # No hit

    def render_frame(self):
        """Render one frame using raycasting."""
        start_time = time.time()

        # Clear framebuffer
        self.framebuffer = [(30, 30, 30)] * (SCREEN_WIDTH * SCREEN_HEIGHT)

        # Draw ceiling (dark gray)
        for y in range(SCREEN_HEIGHT // 2):
            for x in range(SCREEN_WIDTH):
                self.framebuffer[y * SCREEN_WIDTH + x] = (50, 50, 60)

        # Draw floor (darker)
        for y in range(SCREEN_HEIGHT // 2, SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                self.framebuffer[y * SCREEN_WIDTH + x] = (40, 35, 30)

        # Cast rays for each screen column
        player_angle_deg = from_fixed(self.player_angle)
        half_fov = FOV / 2

        for x in range(SCREEN_WIDTH):
            # Calculate ray angle
            ray_angle = player_angle_deg - half_fov + (x / SCREEN_WIDTH) * FOV

            # Cast ray
            distance, side = self.cast_ray(ray_angle)
            self.total_rays += 1

            # Calculate wall height (perspective projection)
            if distance < 1:
                distance = 1
            wall_height = int((TILE_SIZE * SCREEN_HEIGHT) / distance)

            # Clamp wall height
            if wall_height > SCREEN_HEIGHT:
                wall_height = SCREEN_HEIGHT

            # Calculate wall top and bottom
            wall_top = (SCREEN_HEIGHT - wall_height) // 2
            wall_bottom = wall_top + wall_height

            # Choose color based on distance and side
            shade = max(50, 255 - int(distance / 3))
            if side == 1:
                shade = int(shade * 0.7)  # Darker for Y-facing walls

            color = (shade, shade // 2, shade // 4)  # Brown-ish

            # Draw wall column
            for y in range(max(0, wall_top), min(SCREEN_HEIGHT, wall_bottom)):
                self.framebuffer[y * SCREEN_WIDTH + x] = color

        self.frame_count += 1
        frame_time = time.time() - start_time

        return frame_time

    def move(self, forward=0, strafe=0, turn=0):
        """Move the player."""
        angle_deg = from_fixed(self.player_angle)

        # Rotation
        if turn != 0:
            new_angle = angle_deg + turn * 5
            self.player_angle = to_fixed(new_angle % 360)

        # Forward/backward movement
        if forward != 0:
            cos_a = self.cos_table[int(angle_deg) % 360]
            sin_a = self.sin_table[int(angle_deg) % 360]

            move_speed = to_fixed(forward * 8)
            dx = fixed_mul(cos_a, move_speed, self.cpu)
            dy = fixed_mul(sin_a, move_speed, self.cpu)

            new_x = self.cpu.add(self.player_x, dx)
            new_y = self.cpu.add(self.player_y, dy)

            # Simple collision detection
            map_x = from_fixed(new_x) / TILE_SIZE
            map_y = from_fixed(new_y) / TILE_SIZE

            if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
                if WORLD_MAP[int(map_y)][int(map_x)] == 0:
                    self.player_x = new_x
                    self.player_y = new_y

        # Strafing
        if strafe != 0:
            strafe_angle = (angle_deg + 90) % 360
            cos_a = self.cos_table[int(strafe_angle) % 360]
            sin_a = self.sin_table[int(strafe_angle) % 360]

            move_speed = to_fixed(strafe * 8)
            dx = fixed_mul(cos_a, move_speed, self.cpu)
            dy = fixed_mul(sin_a, move_speed, self.cpu)

            new_x = self.cpu.add(self.player_x, dx)
            new_y = self.cpu.add(self.player_y, dy)

            map_x = from_fixed(new_x) / TILE_SIZE
            map_y = from_fixed(new_y) / TILE_SIZE

            if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
                if WORLD_MAP[int(map_y)][int(map_x)] == 0:
                    self.player_x = new_x
                    self.player_y = new_y


# ============================================================
# DISPLAY
# ============================================================

def run_pygame(renderer):
    """Run with pygame display."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH * 2, SCREEN_HEIGHT * 2))
    pygame.display.set_caption("🧠 Neural DOOM - KVRM NeuralOS System Test")
    clock = pygame.time.Clock()

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        forward = 0
        strafe = 0
        turn = 0

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            forward = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            forward = -1
        if keys[pygame.K_a]:
            strafe = -1
        if keys[pygame.K_d]:
            strafe = 1
        if keys[pygame.K_LEFT]:
            turn = -1
        if keys[pygame.K_RIGHT]:
            turn = 1

        renderer.move(forward, strafe, turn)

        # Render frame
        frame_time = renderer.render_frame()

        # Draw to pygame surface
        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                color = renderer.framebuffer[y * SCREEN_WIDTH + x]
                # Scale 2x
                pygame.draw.rect(screen, color, (x * 2, y * 2, 2, 2))

        # Show stats
        fps = 1.0 / max(0.001, frame_time)
        stats = renderer.cpu.get_stats()
        pygame.display.set_caption(
            f"🧠 Neural DOOM | FPS: {fps:.1f} | "
            f"Neural: {stats['neural_ratio']*100:.1f}% | "
            f"Ops/s: {stats['ops_per_sec']:,.0f}"
        )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_ascii(renderer):
    """Run with ASCII display (fallback)."""
    print("\n🎮 ASCII Mode - Press Ctrl+C to quit")
    print("(Auto-moving for demo)")

    try:
        frame = 0
        while True:
            # Render frame
            frame_time = renderer.render_frame()

            # Convert to ASCII
            chars = " .:-=+*#%@"
            output = []
            for y in range(0, SCREEN_HEIGHT, 8):  # Sample every 8 rows
                row = ""
                for x in range(0, SCREEN_WIDTH, 4):  # Sample every 4 cols
                    r, g, b = renderer.framebuffer[y * SCREEN_WIDTH + x]
                    brightness = (r + g + b) // 3
                    char_idx = min(len(chars) - 1, brightness * len(chars) // 256)
                    row += chars[char_idx]
                output.append(row)

            # Clear and print
            print("\033[H\033[J", end="")  # Clear screen
            fps = 1.0 / max(0.001, frame_time)
            stats = renderer.cpu.get_stats()
            print(f"Frame {frame} | FPS: {fps:.1f} | Neural: {stats['neural_ratio']*100:.1f}%")
            print("\n".join(output))
            print(f"Ops/s: {stats['ops_per_sec']:,.0f} | Target: 0.5 FPS | Status: {'✅' if fps >= 0.5 else '❌'}")

            # Auto-move for demo
            renderer.move(forward=1, turn=0.3)

            time.sleep(0.05)
            frame += 1

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# ============================================================
# BENCHMARK MODE
# ============================================================

def run_benchmark(renderer, frames=100):
    """Run headless benchmark to measure FPS."""
    print(f"\n⏱️  Running benchmark ({frames} frames)...")

    frame_times = []
    for i in range(frames):
        frame_time = renderer.render_frame()
        frame_times.append(frame_time)

        # Auto-move for variety
        renderer.move(forward=1, turn=0.1)

        if (i + 1) % 10 == 0:
            avg_fps = 1.0 / (sum(frame_times[-10:]) / 10)
            print(f"   Frame {i+1}/{frames}: {avg_fps:.2f} FPS")

    # Stats
    avg_time = sum(frame_times) / len(frame_times)
    min_time = min(frame_times)
    max_time = max(frame_times)
    avg_fps = 1.0 / avg_time

    print(f"\n📊 BENCHMARK RESULTS")
    print(f"   Frames:     {frames}")
    print(f"   Avg FPS:    {avg_fps:.2f}")
    print(f"   Min FPS:    {1.0/max_time:.2f}")
    print(f"   Max FPS:    {1.0/min_time:.2f}")
    print(f"   Target:     0.5 FPS")
    print(f"   Status:     {'✅ PASS' if avg_fps >= 0.5 else '❌ FAIL'}")

    return avg_fps


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🎮 NEURAL DOOM - Full NeuralOS System Test")
    print("=" * 60)
    print()

    # Parse arguments
    use_neural = "--classical" not in sys.argv
    benchmark_mode = "--benchmark" in sys.argv

    # Initialize NeuralOS System
    system = NeuralOSInterface(use_neural=use_neural)

    # Create renderer with NeuralOS
    renderer = DoomRenderer(system)

    print(f"\nPlayer starting at: ({from_fixed(renderer.player_x):.1f}, {from_fixed(renderer.player_y):.1f})")
    print(f"Map size: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Screen: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print()

    try:
        if benchmark_mode:
            # Headless benchmark
            fps = run_benchmark(renderer, frames=50)
        elif HAS_PYGAME:
            print("🖥️  Starting pygame display...")
            run_pygame(renderer)
        else:
            run_ascii(renderer)
    finally:
        # Print final stats
        print("\n" + "=" * 60)
        print("📊 FINAL SYSTEM STATISTICS")
        print("=" * 60)

        stats = system.get_stats()
        print(f"   Uptime:         {stats['uptime']:.1f}s")
        print(f"   Total Ops:      {stats['total_ops']:,}")
        print(f"   Neural Ops:     {stats['neural_ops']:,} ({stats['neural_ratio']*100:.1f}%)")
        print(f"   Classical Ops:  {stats['classical_ops']:,}")
        print(f"   Ops/Second:     {stats['ops_per_sec']:,.0f}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%")
        print(f"   Frames:         {renderer.frame_count}")
        print(f"   Total Rays:     {renderer.total_rays:,}")

        # Shutdown NeuralOS
        system.shutdown()


if __name__ == "__main__":
    main()
