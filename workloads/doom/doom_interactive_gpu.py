#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║          INTERACTIVE DOOM - VECTORIZED GPU RAYCASTER                             ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  OPTIMIZED: All 80 rays cast in PARALLEL using tensor operations!               ║
║                                                                                  ║
║  CONTROLS:                                                                       ║
║    W/↑  - Move forward        A/←  - Turn left                                  ║
║    S/↓  - Move backward       D/→  - Turn right                                 ║
║    Q    - Quit                                                                  ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ════════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ════════════════════════════════════════════════════════════════════════════════

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL KEYBOARD IO (GPU-BASED)
# ════════════════════════════════════════════════════════════════════════════════

class NeuralKeyboardIO(nn.Module):
    """Neural keyboard input handler - ALL on GPU!"""

    def __init__(self, num_keys=256, action_dim=8):
        super().__init__()
        self.num_keys = num_keys
        self.action_dim = action_dim

        # Key embedding table
        self.key_embeddings = nn.Embedding(num_keys, action_dim)

        # Input buffer as GPU tensor
        self.input_buffer = torch.zeros(16, dtype=torch.long, device=device)
        self.buffer_head = 0
        self.buffer_tail = 0

        self._init_key_mappings()

    def _init_key_mappings(self):
        """Initialize key-to-action mappings."""
        with torch.no_grad():
            self.key_embeddings.weight.zero_()
            # W/w = forward
            self.key_embeddings.weight[ord('w'), 0] = 1.0
            self.key_embeddings.weight[ord('W'), 0] = 1.0
            # S/s = backward
            self.key_embeddings.weight[ord('s'), 1] = 1.0
            self.key_embeddings.weight[ord('S'), 1] = 1.0
            # A/a = turn left
            self.key_embeddings.weight[ord('a'), 2] = 1.0
            self.key_embeddings.weight[ord('A'), 2] = 1.0
            # D/d = turn right
            self.key_embeddings.weight[ord('d'), 3] = 1.0
            self.key_embeddings.weight[ord('D'), 3] = 1.0

    @torch.no_grad()
    def push_key(self, key_code: int):
        """Push a key into the GPU input buffer."""
        self.input_buffer[self.buffer_head % 16] = key_code
        self.buffer_head += 1

    @torch.no_grad()
    def process_input(self) -> torch.Tensor:
        """Process all buffered input - returns action tensor on GPU."""
        if self.buffer_head == self.buffer_tail:
            return torch.zeros(self.action_dim, device=device)

        actions = torch.zeros(self.action_dim, device=device)
        for i in range(self.buffer_tail, self.buffer_head):
            key = self.input_buffer[i % 16]
            key_actions = self.key_embeddings(key)
            actions = torch.maximum(actions, key_actions)

        self.buffer_tail = self.buffer_head
        return actions


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL SYSTEM MODELS
# ════════════════════════════════════════════════════════════════════════════════

class TrulyNeuralTimer(nn.Module):
    """Neural timer for game timing."""
    def __init__(self):
        super().__init__()
        self.counter = torch.tensor(0.0, device=device)

    @torch.no_grad()
    def tick(self, delta: float = 1.0) -> torch.Tensor:
        self.counter += delta
        return self.counter


class TrulyNeuralGIC(nn.Module):
    """Neural interrupt controller."""
    def __init__(self, max_irqs=16):
        super().__init__()
        self.irq_pending = torch.zeros(max_irqs, device=device)

    @torch.no_grad()
    def raise_irq(self, irq_num: int):
        self.irq_pending[irq_num] = 1.0


# ════════════════════════════════════════════════════════════════════════════════
# VECTORIZED GPU DOOM RAYCASTER - ALL 80 RAYS IN PARALLEL!
# ════════════════════════════════════════════════════════════════════════════════

class VectorizedDOOMRaycaster(nn.Module):
    """
    FULLY VECTORIZED DOOM Raycaster - ALL rays cast in parallel!

    Key optimization: Instead of looping through 80 columns,
    we compute ALL ray directions and step ALL rays simultaneously
    using tensor broadcasting.
    """

    FB_WIDTH = 80
    FB_HEIGHT = 25
    MAP_SIZE = 16

    WALL_CHARS = b'#%*+=-. '

    def __init__(self):
        super().__init__()

        # Player state as GPU tensors
        self.player_x = torch.tensor(8.0, device=device)
        self.player_y = torch.tensor(8.0, device=device)
        self.player_angle = torch.tensor(0.0, device=device)

        # Movement parameters
        self.move_speed = 0.2
        self.turn_speed = 0.15

        # Map (16x16) - 1 = wall, 0 = empty
        self.map = torch.zeros(self.MAP_SIZE, self.MAP_SIZE, dtype=torch.float32, device=device)
        self._init_map()

        # Pre-compute ray angle offsets (FOV = 60 degrees)
        fov = math.pi / 3
        self.ray_offsets = torch.linspace(fov/2, -fov/2, self.FB_WIDTH, device=device)

        # Framebuffer
        self.framebuffer = torch.full(
            (self.FB_HEIGHT, self.FB_WIDTH),
            ord(' '),
            dtype=torch.uint8,
            device=device
        )

        # Wall character lookup
        self.wall_chars = torch.tensor(list(self.WALL_CHARS), dtype=torch.uint8, device=device)

    def _init_map(self):
        """Initialize map with walls."""
        m = self.map
        # Border walls
        m[0, :] = 1
        m[self.MAP_SIZE-1, :] = 1
        m[:, 0] = 1
        m[:, self.MAP_SIZE-1] = 1
        # Interior walls
        m[3, 3:8] = 1
        m[3:8, 7] = 1
        m[10, 3:12] = 1
        m[5:10, 12] = 1
        m[8, 2:6] = 1
        m[6, 10:14] = 1

    @torch.no_grad()
    def process_input(self, actions: torch.Tensor):
        """Process movement - ALL ON GPU."""
        cos_a = torch.cos(self.player_angle)
        sin_a = torch.sin(self.player_angle)

        # Forward/backward
        move = float(actions[0].item() - actions[1].item()) * self.move_speed
        new_x = self.player_x + cos_a * move
        new_y = self.player_y + sin_a * move

        # Simple collision check
        map_x = int(new_x.clamp(0, self.MAP_SIZE-1).item())
        map_y = int(new_y.clamp(0, self.MAP_SIZE-1).item())

        if self.map[map_y, map_x].item() < 0.5:
            self.player_x = new_x.clamp(1.0, self.MAP_SIZE - 2.0)
            self.player_y = new_y.clamp(1.0, self.MAP_SIZE - 2.0)

        # Turn (A=left=negative angle, D=right=positive angle)
        turn = float(actions[2].item() - actions[3].item()) * self.turn_speed
        self.player_angle = (self.player_angle + turn) % (2 * math.pi)

    @torch.no_grad()
    def cast_all_rays_vectorized(self) -> torch.Tensor:
        """
        Cast ALL 80 rays in PARALLEL using tensor operations!

        This is the key optimization - no Python loops for ray casting.
        """
        # Compute all ray angles at once [80]
        ray_angles = self.player_angle + self.ray_offsets

        # Ray directions [80]
        ray_dx = torch.cos(ray_angles)
        ray_dy = torch.sin(ray_angles)

        # Starting positions (broadcast to all rays) [80]
        ray_x = self.player_x.expand(self.FB_WIDTH).clone()
        ray_y = self.player_y.expand(self.FB_WIDTH).clone()

        # Track which rays have hit
        distances = torch.full((self.FB_WIDTH,), 20.0, device=device)
        hit = torch.zeros(self.FB_WIDTH, dtype=torch.bool, device=device)

        # DDA stepping - ALL rays step together
        step_size = 0.1
        max_steps = 200

        for _ in range(max_steps):
            # Step all rays
            ray_x = ray_x + ray_dx * step_size
            ray_y = ray_y + ray_dy * step_size

            # Check bounds and map hits for all rays
            map_x = ray_x.long().clamp(0, self.MAP_SIZE - 1)
            map_y = ray_y.long().clamp(0, self.MAP_SIZE - 1)

            # Check for wall hits (vectorized lookup)
            wall_hit = self.map[map_y, map_x] > 0.5

            # Calculate distances for newly hit rays
            new_hits = wall_hit & (~hit)
            if new_hits.any():
                dx = ray_x - self.player_x
                dy = ray_y - self.player_y
                new_dist = torch.sqrt(dx * dx + dy * dy)
                distances = torch.where(new_hits, new_dist, distances)
                hit = hit | wall_hit

            # Early exit if all rays hit
            if hit.all():
                break

        return distances

    @torch.no_grad()
    def render(self) -> torch.Tensor:
        """Render scene - VECTORIZED column rendering."""
        # Cast all rays in parallel
        distances = self.cast_all_rays_vectorized()

        # Clear framebuffer
        self.framebuffer.fill_(ord(' '))

        # Compute wall heights for all columns [80]
        wall_heights = (self.FB_HEIGHT / distances.clamp(min=0.1)).clamp(max=self.FB_HEIGHT).long()

        # Compute wall top/bottom for all columns
        wall_tops = ((self.FB_HEIGHT - wall_heights) // 2).clamp(min=0)
        wall_bottoms = (wall_tops + wall_heights).clamp(max=self.FB_HEIGHT)

        # Character indices based on distance
        char_indices = (distances / 2.5).clamp(0, len(self.WALL_CHARS) - 1).long()

        # Render each column (still needs a loop for row-wise filling, but columns are vectorized)
        for col in range(self.FB_WIDTH):
            wt = wall_tops[col].item()
            wb = wall_bottoms[col].item()
            ch = self.wall_chars[char_indices[col]]

            # Draw wall
            self.framebuffer[wt:wb, col] = ch

            # Draw floor
            for row in range(wb, self.FB_HEIGHT):
                if row > self.FB_HEIGHT * 0.7:
                    self.framebuffer[row, col] = ord('.')
                else:
                    self.framebuffer[row, col] = ord(',')

        return self.framebuffer

    def get_framebuffer_str(self) -> str:
        """Convert GPU framebuffer to string."""
        fb_cpu = self.framebuffer.cpu().numpy()
        lines = []
        for row in range(self.FB_HEIGHT):
            line = ''.join(chr(c) for c in fb_cpu[row])
            lines.append(line)
        return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DOOM RUNNER
# ════════════════════════════════════════════════════════════════════════════════

class InteractiveDOOM:
    """Interactive DOOM with proper terminal handling."""

    def __init__(self):
        print()
        print("╔" + "═" * 70 + "╗")
        print("║" + " INTERACTIVE DOOM - VECTORIZED GPU RAYCASTER ".center(70) + "║")
        print("╚" + "═" * 70 + "╝")
        print(f"\n   Device: {device}")

        self.keyboard = NeuralKeyboardIO().to(device)
        self.timer = TrulyNeuralTimer().to(device)
        self.gic = TrulyNeuralGIC().to(device)
        self.raycaster = VectorizedDOOMRaycaster().to(device)

        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.running = True
        self.old_settings = None

        print("   ✅ Vectorized raycaster ready (80 parallel rays)")
        print("   ✅ Neural keyboard IO ready")
        print("\n   Controls: WASD to move/turn, Q to quit\n")

    def _setup_terminal(self):
        """Set up terminal for raw input with BLOCKING writes."""
        import termios
        import tty

        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())  # Use setcbreak instead of setraw

    def _restore_terminal(self):
        """Restore terminal settings."""
        if self.old_settings:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def _read_input(self) -> str:
        """Read input without blocking."""
        import select
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                ch = sys.stdin.read(1)
                # Handle escape sequences for arrow keys
                if ch == '\x1b':
                    ready2, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if ready2:
                        ch2 = sys.stdin.read(1)
                        if ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            if ch3 == 'A': return 'w'  # Up
                            if ch3 == 'B': return 's'  # Down
                            if ch3 == 'C': return 'd'  # Right
                            if ch3 == 'D': return 'a'  # Left
                return ch
        except:
            pass
        return ''

    def render_frame(self):
        """Render one frame."""
        # Read and process input
        key = self._read_input()
        if key:
            if key.lower() == 'q':
                self.running = False
                return
            self.keyboard.push_key(ord(key))

        # Get actions from neural keyboard
        actions = self.keyboard.process_input()

        # Process movement
        self.raycaster.process_input(actions)

        # Render (vectorized!)
        self.raycaster.render()

        # Get framebuffer
        fb = self.raycaster.get_framebuffer_str()

        # Stats
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Display (buffered output to prevent blocking issues)
        output = []
        output.append("\033[H")  # Move cursor to top-left
        output.append("╔" + "═" * 80 + "╗\n")
        for line in fb.split('\n'):
            output.append(f"║{line}║\n")
        output.append("╚" + "═" * 80 + "╝\n")
        output.append(f" WASD: Move/Turn | Q: Quit | FPS: {fps:.1f} | Frame: {self.frame_count}\n")
        output.append(f" Pos: ({self.raycaster.player_x.item():.1f}, {self.raycaster.player_y.item():.1f}) | Angle: {math.degrees(self.raycaster.player_angle.item()):.0f}°\n")

        # Write all at once
        sys.stdout.write(''.join(output))
        sys.stdout.flush()

    def run(self):
        """Main game loop."""
        try:
            self._setup_terminal()
            print("\033[?25l", end='')  # Hide cursor
            print("\033[2J", end='')    # Clear screen

            target_fps = 30
            frame_time = 1.0 / target_fps

            while self.running:
                frame_start = time.perf_counter()

                self.render_frame()

                # Frame rate limiting
                frame_elapsed = time.perf_counter() - frame_start
                if frame_elapsed < frame_time:
                    time.sleep(frame_time - frame_elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            print("\033[?25h", end='')  # Show cursor
            self._restore_terminal()
            print("\n\nDOOM exited.")
            elapsed = time.perf_counter() - self.start_time
            print(f"Total frames: {self.frame_count}")
            if elapsed > 0:
                print(f"Average FPS: {self.frame_count / elapsed:.1f}")


# ════════════════════════════════════════════════════════════════════════════════
# DEMO MODE
# ════════════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run non-interactive demo."""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + " DOOM GPU DEMO - VECTORIZED RAYCASTER ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print(f"\nDevice: {device}\n")

    # Initialize
    keyboard = NeuralKeyboardIO().to(device)
    raycaster = VectorizedDOOMRaycaster().to(device)
    timer = TrulyNeuralTimer().to(device)

    print("✅ Vectorized raycaster initialized (80 parallel rays)")
    print()

    # Benchmark rendering
    print("Benchmarking vectorized rendering...")
    frames = 30

    # Warm up
    for _ in range(5):
        actions = torch.zeros(8, device=device)
        actions[0] = 0.5
        actions[3] = 0.1
        raycaster.process_input(actions)
        raycaster.render()

    # Timed run
    start = time.perf_counter()
    for i in range(frames):
        actions = torch.zeros(8, device=device)
        actions[0] = 0.3  # Forward
        actions[3] = 0.08  # Slight right turn
        raycaster.process_input(actions)
        raycaster.render()
        timer.tick()

    elapsed = time.perf_counter() - start
    fps = frames / elapsed if elapsed > 0 else 0

    print(f"\nRendered {frames} frames in {elapsed:.3f}s")
    print(f"FPS: {fps:.1f}")
    print()

    # Display final frame
    print("Final frame:")
    print("╔" + "═" * 80 + "╗")
    for line in raycaster.get_framebuffer_str().split('\n'):
        print(f"║{line}║")
    print("╚" + "═" * 80 + "╝")
    print()

    print(f"Player: ({raycaster.player_x.item():.2f}, {raycaster.player_y.item():.2f})")
    print(f"Angle: {math.degrees(raycaster.player_angle.item()):.1f}°")
    print()

    # Verify GPU tensors
    print("GPU Tensor Verification:")
    print(f"  • Player X: {raycaster.player_x.device}")
    print(f"  • Map: {raycaster.map.device}")
    print(f"  • Framebuffer: {raycaster.framebuffer.device}")
    print(f"  • Ray offsets: {raycaster.ray_offsets.device}")
    print()

    if fps >= 10:
        print("✅ VECTORIZED RAYCASTER: 10+ FPS achieved!")
    else:
        print(f"⚠️  FPS below target (got {fps:.1f})")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "play":
        game = InteractiveDOOM()
        game.run()
    else:
        print("\nUsage:")
        print("  python doom_interactive_gpu.py demo  - Run benchmark demo")
        print("  python doom_interactive_gpu.py play  - Run interactive game")
        print()
        run_demo()
