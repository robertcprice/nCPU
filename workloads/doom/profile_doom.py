#!/usr/bin/env python3
"""
🔬 DOOM PROFILER - Find the Bottlenecks
==========================================

Profiles every component of DOOM rendering to identify what's slow:
- Ray casting steps
- Neural ADD operations per ray
- Bit conversions
- Rendering calculations
"""

import time
import math
from neural_cpu_hybrid import FastHybridALU

print("=" * 70)
print("🎮 DOOM PROFILER - Finding Bottlenecks")
print("=" * 70)
print()

# Initialize neural CPU
print("Initializing FastHybridALU...")
alu = FastHybridALU()
print()

class DOOMProfiler:
    """DOOM with detailed component profiling"""

    def __init__(self, alu):
        self.alu = alu
        self.stats = {
            'total_ray_time': [],
            'neural_add_time': [],
            'math_calc_time': [],
            'ray_steps': [],
            'instructions_per_frame': [],
        }

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
            [1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1],
            [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ]

        self.player_x = 2.5
        self.player_y = 2.5
        self.player_angle = 0.0

    def cast_ray_profiled(self, angle):
        """Cast a ray and profile each component"""
        ray_start = time.time()

        ray_x = self.player_x
        ray_y = self.player_y

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        step = 0.05
        steps = 0

        for i in range(200):
            ray_x += cos_a * step
            ray_y += sin_a * step
            steps += 1

            map_x = int(ray_x)
            map_y = int(ray_y)

            if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                if self.map[map_y][map_x] == 1:
                    # Profile distance calculation

                    # Math only (no neural)
                    math_start = time.time()
                    dx = abs(ray_x - self.player_x)
                    dy = abs(ray_y - self.player_y)
                    # Scale to integers for neural CPU
                    dx_scaled = int(dx * 1000)
                    dy_scaled = int(dy * 1000)
                    math_time = time.time() - math_start

                    # Neural ADD operation
                    add_start = time.time()
                    result = self.alu.execute('ADD', dx_scaled, dy_scaled)
                    add_time = time.time() - add_start

                    # Convert back
                    distance = math.sqrt(result) / 100

                    ray_time = time.time() - ray_start

                    self.stats['neural_add_time'].append(add_time)
                    self.stats['math_calc_time'].append(math_time)
                    self.stats['ray_steps'].append(steps)
                    self.stats['total_ray_time'].append(ray_time)

                    return distance, steps
        return 10.0, 200

    def render_frame_profiled(self, width=20):
        """Render one frame with profiling"""
        frame_start = time.time()
        total_rays = 0
        total_steps = 0
        neural_adds = 0

        fov = 1.2

        for x in range(width):
            ray_angle = self.player_angle - fov/2 + (x / width) * fov
            dist, steps = self.cast_ray_profiled(ray_angle)
            total_rays += 1
            total_steps += steps
            neural_adds += 1  # One neural ADD per ray for distance

        frame_time = time.time() - frame_start
        fps = 1.0 / frame_time

        return {
            'frame_time': frame_time,
            'fps': fps,
            'rays': total_rays,
            'total_steps': total_steps,
            'neural_adds': neural_adds
        }

    def print_results(self):
        """Print profiling analysis"""
        print("\n" + "=" * 70)
        print("📊 DOOM PROFILING RESULTS")
        print("=" * 70)
        print()

        if not self.stats['total_ray_time']:
            print("No profiling data collected!")
            return

        # Calculate averages
        avg_ray_time = sum(self.stats['total_ray_time']) / len(self.stats['total_ray_time'])
        avg_add_time = sum(self.stats['neural_add_time']) / len(self.stats['neural_add_time'])
        avg_math_time = sum(self.stats['math_calc_time']) / len(self.stats['math_calc_time'])
        avg_steps = sum(self.stats['ray_steps']) / len(self.stats['ray_steps'])

        print("🔍 PER-RAY BREAKDOWN:")
        print(f"   Total time per ray: {avg_ray_time*1000:.3f}ms")
        print(f"   ├─ Neural ADD: {avg_add_time*1000:.3f}ms ({avg_add_time/avg_ray_time*100:.1f}%)")
        print(f"   ├─ Math (trig): {avg_math_time*1000:.3f}ms ({avg_math_time/avg_ray_time*100:.1f}%)")
        print(f"   └─ Ray marching: {avg_steps:.0f} steps")
        print()

        print("🎯 BOTTLENECK ANALYSIS:")
        print()

        # What's taking the most time?
        total_add_time = sum(self.stats['neural_add_time'])
        total_math_time = sum(self.stats['math_calc_time'])
        total_ray_time = sum(self.stats['total_ray_time'])
        total_other = total_ray_time - total_add_time - total_math_time

        components = [
            ("Neural ADD operations", total_add_time, avg_add_time),
            ("Math calculations (trig)", total_math_time, avg_math_time),
            ("Other (ray marching loop)", total_other, avg_ray_time - avg_add_time - avg_math_time),
        ]

        components.sort(key=lambda x: x[1], reverse=True)

        for name, total, avg in components:
            pct = total / total_ray_time * 100
            print(f"   {name:30s}: {total*1000:.1f}ms ({pct:5.1f}%)  avg: {avg*1000:.3f}ms")

        print()
        print("💡 KEY FINDINGS:")
        print()

        if avg_add_time > avg_math_time * 2:
            print(f"   ⚠️  NEURAL ADD IS THE BOTTLENECK ({avg_add_time/avg_math_time:.1f}x slower than math)")
            print(f"   → Each ray casting does 1 neural ADD operation")
            print(f"   → With 40 rays/frame, that's 40 neural ops/frame")
            print(f"   → At {avg_add_time*1000:.1f}ms per ADD, that's {avg_add_time*1000*40:.0f}ms just for neural math")
            print()
            print("   🚀 SOLUTIONS:")
            print("      1. Batch process rays: Do all 40 distance calculations in one batch")
            print("         - Could reduce 40 API calls → 1 batch call")
            print("         - Estimated 5-10x speedup")
            print()
            print("      2. Cache common distances: Many rays hit walls at similar distances")
            print("         - Pre-compute distance table")
            print("         - Lookup table instead of neural calculation")
            print()
            print("      3. Use classical math for distance: sqrt(dx² + dy²) is fast")
            print("         - Only use neural for actual game logic operations")
            print("         - Distance calc doesn't need to be neural")
        else:
            print(f"   Math (trig) is taking {avg_math_time/avg_add_time:.1f}x more time than neural ops")
            print("   → Consider caching trig values or using lookup tables")

        print()

# Run the profiler
profiler = DOOMProfiler(alu)

print("Rendering 10 frames (20 rays each)...")
print()

for frame in range(10):
    result = profiler.render_frame_profiled(width=20)
    if frame == 0 or frame == 9:
        print(f"Frame {frame+1}: {result['fps']:.1f} FPS, {result['frame_time']*1000:.0f}ms, {result['neural_adds']} neural ADDs")

profiler.print_results()

print()
print("=" * 70)
print("📈 PROJECTION WITH OPTIMIZATIONS:")
print("=" * 70)
print()
print(f"Current performance: ~{result['fps']:.1f} FPS at 20 rays")
print()
print("With batch processing (5x speedup):")
print(f"   → {result['fps']*5:.1f} FPS at 20 rays")
print(f"   → {result['fps']*5*2:.1f} FPS at 10 rays (half resolution)")
print()
print("With classical distance math (2x faster):")
print(f"   → {result['fps']*2:.1f} FPS at 20 rays")
print(f"   → {result['fps']*2*4:.1f} FPS at 10 rays")
print()
print("COMBINED (batch + classical):")
print(f"   → {result['fps']*10:.1f} FPS at 20 rays")
print(f"   → {result['fps']*10*4:.1f} FPS at 10 rays ({result['fps']*10*4:.0f} FPS achievable!)")
print()
print("💡 To reach 60 FPS: Use 5-10 rays resolution + batching + classical math")
