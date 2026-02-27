#!/usr/bin/env python3
"""
🎮 DOOM Benchmark with Batched Neural CPU
==========================================

Tests DOOM performance with batched neural operations.
"""

import time
import math
from neural_cpu_batched import BatchedNeuralALU

print("=" * 70)
print("🎮 DOOM BENCHMARK - Batched Neural CPU")
print("=" * 70)
print()

# Initialize BatchedNeuralALU
print("Initializing BatchedNeuralALU...")
alu = BatchedNeuralALU()
print()


class BatchedNeuralDOOM:
    """DOOM using batched neural operations for maximum performance"""

    def __init__(self, alu):
        self.alu = alu

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

    def cast_rays_batched(self, angles):
        """
        Cast multiple rays using batched neural operations.

        Args:
            angles: List of ray angles

        Returns:
            List of distances
        """
        num_rays = len(angles)
        cos_values = [math.cos(a) for a in angles]
        sin_values = [math.sin(a) for a in angles]

        step = 0.05
        max_steps = 200

        # Process all rays together
        ray_positions_x = [self.player_x] * num_rays
        ray_positions_y = [self.player_y] * num_rays

        distances = [10.0] * num_rays
        active = [True] * num_rays

        for step_num in range(max_steps):
            if not any(active):
                break

            # Update active ray positions
            for i in range(num_rays):
                if active[i]:
                    ray_positions_x[i] += cos_values[i] * step
                    ray_positions_y[i] += sin_values[i] * step

                    map_x = int(ray_positions_x[i])
                    map_y = int(ray_positions_y[i])

                    if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                        if self.map[map_y][map_x] == 1:
                            # Calculate distance using neural ADD
                            dx = abs(int((ray_positions_x[i] - self.player_x) * 1000))
                            dy = abs(int((ray_positions_y[i] - self.player_y) * 1000))

                            # Store for batch processing
                            distances[i] = (dx, dy)
                            active[i] = False

        # Batch process all distance calculations
        distance_operands = [(d[0], d[1]) if isinstance(d, tuple) else (0, 0) for d in distances]
        batch_results = self.alu.execute_batch('ADD', distance_operands)

        # Convert results to distances
        final_distances = []
        for i, (dx, dy) in enumerate(distance_operands):
            result = batch_results[i]
            final_distances.append(math.sqrt(result) / 100 if result > 0 else distances[i] if isinstance(distances[i], float) else 10.0)

        return final_distances

    def render_frame(self, width=40):
        """Render one frame using batched ray casting"""
        fov = 1.2
        ray_angles = [
            self.player_angle - fov/2 + (x / width) * fov
            for x in range(width)
        ]

        distances = self.cast_rays_batched(ray_angles)
        return distances


# Run benchmark
doom = BatchedNeuralDOOM(alu)

print("=" * 70)
print("📊 BATCHED DOOM PERFORMANCE TEST")
print("=" * 70)
print()

# Warmup
print("Warming up...")
for _ in range(5):
    doom.render_frame(width=40)
print()

# Benchmark
num_frames = 50
width = 40
print(f"Benchmarking: {num_frames} frames at {width} rays/frame")
print()

frame_times = []
start = time.time()

for frame in range(num_frames):
    frame_start = time.time()
    distances = doom.render_frame(width=width)
    frame_time = time.time() - frame_start
    frame_times.append(frame_time)

    if (frame + 1) % 10 == 0:
        fps = 1.0 / frame_time
        print(f"   Frame {frame+1}/{num_frames}: {fps:.1f} FPS ({frame_time*1000:.1f}ms)")

elapsed = time.time() - start

# Calculate statistics
avg_frame_time = sum(frame_times) / len(frame_times)
min_frame_time = min(frame_times)
max_frame_time = max(frame_times)
fps = 1.0 / avg_frame_time

print()
print("=" * 70)
print("📊 BATCHED DOOM RESULTS")
print("=" * 70)
print()
print(f"Resolution: {width} rays per frame")
print(f"Frames rendered: {num_frames}")
print(f"Total time: {elapsed:.2f}s")
print()
print(f"FPS: {fps:.2f}")
print(f"Avg frame time: {avg_frame_time*1000:.1f}ms")
print(f"Min frame time: {min_frame_time*1000:.1f}ms")
print(f"Max frame time: {max_frame_time*1000:.1f}ms")
print()

# Calculate instruction throughput
rays_per_frame = width
neural_ops_per_frame = rays_per_frame  # 1 ADD per ray
total_neural_ops = num_frames * neural_ops_per_frame
ops_per_sec = total_neural_ops / elapsed

print(f"Neural operations per frame: {neural_ops_per_frame}")
print(f"Total neural operations: {total_neural_ops}")
print(f"Instructions Per Second (DOOM): {ops_per_sec:.0f} IPS")
print()

# Comparison with sequential
print("=" * 70)
print("📈 COMPARISON: Sequential vs Batched")
print("=" * 70)
print()

# From previous benchmark: sequential DOOM ran at ~2 FPS with 79 IPS
sequential_fps = 2.0
sequential_ips = 79.0

speedup = fps / sequential_fps if sequential_fps > 0 else 0
ips_speedup = ops_per_sec / sequential_ips if sequential_ips > 0 else 0

print(f"Sequential: {sequential_fps:.1f} FPS, {sequential_ips:.0f} IPS")
print(f"Batched:    {fps:.1f} FPS, {ops_per_sec:.0f} IPS")
print()
print(f"🚀 FPS Speedup: {speedup:.1f}x")
print(f"🚀 IPS Speedup: {ips_speedup:.1f}x")
print()

# Calculate practical FPS estimates
print("=" * 70)
print("🎯 PRACTICAL FPS ESTIMATES")
print("=" * 70)
print()

batch_40_rays = fps
batch_20_rays = fps * 2  # Half the rays = 2x faster
batch_10_rays = fps * 4  # Quarter the rays = 4x faster

print(f"Current ({width} rays): {batch_40_rays:.1f} FPS")
print(f"Half resolution (20 rays): {batch_20_rays:.1f} FPS")
print(f"Quarter resolution (10 rays): {batch_10_rays:.1f} FPS")
print()

if batch_40_rays >= 30:
    print("🎉 DOOM is PLAYABLE at current resolution (>30 FPS)!")
elif batch_20_rays >= 30:
    print("✅ DOOM is PLAYABLE at half resolution (>30 FPS)")
elif batch_10_rays >= 30:
    print("⚠️  DOOM is playable at quarter resolution (>30 FPS)")
else:
    print(f"💡 Recommendations for 30+ FPS:")
    print(f"   • Use {int(40 * 30 / fps)} rays for 30 FPS")
    print(f"   • Combine with torch.compile() for 2-3x additional speedup")
    print(f"   • Use smaller models (distillation) for 2-3x speedup")

print()
print("=" * 70)
print("📊 NEURAL ALU STATISTICS")
print("=" * 70)
stats = alu.get_stats()
for k, v in stats.items():
    print(f"   {k}: {v}")
