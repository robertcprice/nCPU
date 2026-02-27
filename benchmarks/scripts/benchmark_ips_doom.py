#!/usr/bin/env python3
"""
🧪 COMPLETE PERFORMANCE BENCHMARK
==================================

Tests:
1. Instructions Per Second (IPS) - raw neural CPU speed
2. DOOM FPS + IPS - real game performance with instruction counting
"""

import time
import math
from neural_cpu_hybrid import FastHybridALU

print("=" * 70)
print("🧪 NEURAL CPU PERFORMANCE BENCHMARK")
print("=" * 70)
print()

# Initialize FastHybridALU (our working model)
print("Initializing FastHybridALU...")
alu = FastHybridALU()
print()

# ============================================================
# TEST 1: RAW INSTRUCTION SPEED (IPS)
# ============================================================

print("=" * 70)
print("📊 TEST 1: INSTRUCTIONS PER SECOND (IPS)")
print("=" * 70)
print()

def benchmark_instructions(num_ops=10000):
    """Benchmark raw instruction execution speed"""
    operations = [
        ('ADD', 12345, 67890),
        ('SUB', 1000, 42),
        ('AND', 0xFFFF, 0xFF00),
        ('OR', 0xF0, 0x0F),
        ('XOR', 0xAA, 0x55),
    ]

    print(f"Executing {num_ops} mixed operations...")

    start = time.time()
    ops_executed = 0

    for i in range(num_ops):
        op, a, b = operations[i % len(operations)]
        result = alu.execute(op, a, b)
        ops_executed += 1

        if (i + 1) % 1000 == 0:
            pass  # Progress every 1000

    elapsed = time.time() - start
    ips = ops_executed / elapsed

    print(f"   Operations: {ops_executed}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   IPS: {ips:.0f} instructions/second")
    print(f"   Avg per op: {elapsed/ops_executed*1000:.3f}ms")

    return ips, elapsed

# Run IPS benchmark
ips, elapsed = benchmark_instructions(10000)
print()

# ============================================================
# TEST 2: DOOM WITH INSTRUCTION COUNTING
# ============================================================

print("=" * 70)
print("🎮 TEST 2: DOOM FPS + INSTRUCTION COUNTING")
print("=" * 70)
print()

class NeuralDOOMWithCounter:
    """DOOM that counts neural instructions executed"""

    def __init__(self, alu):
        self.alu = alu
        self.instruction_count = 0

        # Simple map
        self.map_width = 16
        self.map_height = 16
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

        # Player state
        self.player_x = 2.5
        self.player_y = 2.5
        self.player_angle = 0.0

    def neural_add(self, a, b):
        """Count and execute ADD instruction"""
        self.instruction_count += 1
        return self.alu.execute('ADD', a, b)

    def cast_ray(self, angle):
        """Cast a ray using neural CPU"""
        ray_x = self.player_x
        ray_y = self.player_y

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        step = 0.05

        for i in range(200):
            ray_x += cos_a * step
            ray_y += sin_a * step

            map_x = int(ray_x)
            map_y = int(ray_y)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                if self.map[map_y][map_x] == 1:
                    # Calculate distance using neural ADD
                    dx = abs(int((ray_x - self.player_x) * 1000))
                    dy = abs(int((ray_y - self.player_y) * 1000))
                    result = self.neural_add(dx, dy)
                    distance = math.sqrt(result) / 100
                    return distance
        return 10.0

    def render_frame(self, width=40):  # Reduced width for faster testing
        """Render one frame"""
        fov = 1.2

        for x in range(width):
            ray_angle = self.player_angle - fov/2 + (x / width) * fov
            self.cast_ray(ray_angle)

# Run DOOM benchmark
doom = NeuralDOOMWithCounter(alu)

print(f"Rendering {50} DOOM frames (40 rays each)...")
print()

frame_times = []
total_instructions = 0

for frame in range(50):
    doom.instruction_count = 0

    start = time.time()
    doom.render_frame(width=40)
    elapsed = time.time() - start

    total_instructions += doom.instruction_count
    frame_times.append(elapsed)

    if (frame + 1) % 10 == 0:
        print(f"   Frame {frame+1}/50: {doom.instruction_count} instructions, {elapsed*1000:.1f}ms")

# Calculate statistics
avg_frame_time = sum(frame_times) / len(frame_times)
fps = 1.0 / avg_frame_time
avg_instructions_per_frame = total_instructions / 50
ips_doom = avg_instructions_per_frame * fps

print()
print("=" * 70)
print("📊 DOOM PERFORMANCE RESULTS")
print("=" * 70)
print()
print(f"Resolution: 40 rays per frame")
print(f"Frames rendered: 50")
print()
print(f"FPS: {fps:.2f}")
print(f"Avg frame time: {avg_frame_time*1000:.1f}ms")
print(f"Avg instructions per frame: {avg_instructions_per_frame:.0f}")
print(f"Instructions Per Second (DOOM): {ips_doom:.0f} IPS")
print()

# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("📈 COMPLETE PERFORMANCE SUMMARY")
print("=" * 70)
print()
print(f"Raw Neural CPU Speed: {ips:.0f} IPS")
print(f"DOOM Real-world:      {ips_doom:.0f} IPS @ {fps:.1f} FPS")
print()
print("💡 Insights:")
print(f"   • Each DOOM frame uses ~{avg_instructions_per_frame:.0f} neural instructions")
print(f"   • At {fps:.1f} FPS, that's {ips_doom:.0f} instructions/second")
print(f"   • Raw CPU is {ips/ips_doom:.1f}x faster (DOOM has overhead)")
print()

# Comparison with classical CPU
print("🖥️  Comparison:")
print(f"   • Modern CPU: ~100 Billion IPS (3GHz)")
print(f"   • This Neural CPU: ~{ips:.0f} IPS")
print(f"   • Speed ratio: ~{100_000_000_000/ips:.0f}x slower")
print()

if fps >= 30:
    print("🎉 DOOM is PLAYABLE (>30 FPS)!")
elif fps >= 15:
    print("✅ DOOM is playable (15-30 FPS)")
elif fps >= 5:
    print("⚠️  DOOM runs but slowly (5-15 FPS)")
else:
    print("❌ DOOM is too slow (<5 FPS)")
