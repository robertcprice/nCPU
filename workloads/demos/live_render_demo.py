#!/usr/bin/env python3
"""
🎮 KVRM Neural Renderer - LIVE ANIMATED DEMO
Run: python3 live_render_demo.py

Watch the neural network render in real-time!
Press Ctrl+C to exit.
"""

import torch
import torch.nn as nn
import time
import os
import sys
import math

# Use MPS on Mac, CUDA on GPU, else CPU
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ============================================================
# Architecture (must match trained model)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1)
        )
        self.norm = nn.GroupNorm(8, c)
    def forward(self, x):
        return self.norm(x + self.net(x))

class CNNRenderer(nn.Module):
    def __init__(self, c=64, blocks=8):
        super().__init__()
        self.enc = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(c) for _ in range(blocks)])
        self.out = nn.Conv2d(c, 3, 3, padding=1)
    def forward(self, x):
        x = self.enc(x)
        for b in self.blocks:
            x = b(x)
        return torch.sigmoid(self.out(x))

# ============================================================
# Load model
# ============================================================

model = CNNRenderer()
try:
    ckpt = torch.load('models/renderer_fast_best.pt', map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    print("✅ Loaded neural renderer!")
except Exception as e:
    print(f"⚠️ Using untrained model: {e}")

model = model.to(device)
model.eval()

# ============================================================
# Live rendering
# ============================================================

# ASCII characters from dark to bright
CHARS = " .:-=+*#%@"
# Or use block characters for denser look:
# CHARS = " ░▒▓█"

# Resolution
WIDTH = 80
HEIGHT = 30

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def hide_cursor():
    sys.stdout.write('\033[?25l')
    sys.stdout.flush()

def show_cursor():
    sys.stdout.write('\033[?25h')
    sys.stdout.flush()

def move_cursor_home():
    sys.stdout.write('\033[H')
    sys.stdout.flush()

def render_frame(t):
    """Generate input pattern and render through neural network"""
    with torch.no_grad():
        # Create animated input pattern
        x = torch.zeros(1, 3, HEIGHT, WIDTH, device=device)

        for i in range(HEIGHT):
            for j in range(WIDTH):
                # Animated wave pattern
                cx, cy = j / WIDTH - 0.5, i / HEIGHT - 0.5
                dist = math.sqrt(cx*cx + cy*cy)

                # R: circular wave
                x[0, 0, i, j] = 0.5 + 0.5 * math.sin(dist * 10 - t * 3)
                # G: horizontal wave
                x[0, 1, i, j] = 0.5 + 0.5 * math.sin(j / WIDTH * 6 + t * 2)
                # B: time-based pulse
                x[0, 2, i, j] = 0.5 + 0.5 * math.sin(t * 4)

        # Run through neural renderer
        out = model(x)[0].cpu()

        # Convert RGB to grayscale
        gray = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2]

        return gray

def to_ascii(gray):
    """Convert grayscale tensor to ASCII string"""
    lines = []
    for y in range(gray.shape[0]):
        row = ""
        for x in range(gray.shape[1]):
            idx = int(gray[y, x].clamp(0, 0.999) * len(CHARS))
            row += CHARS[idx]
        lines.append(row)
    return lines

# ============================================================
# Main loop
# ============================================================

print(f"\n🎮 KVRM NEURAL RENDERER - LIVE DEMO")
print(f"   Resolution: {WIDTH}x{HEIGHT}")
print(f"   Device: {device.upper()}")
print(f"   Press Ctrl+C to exit\n")
input("Press Enter to start...")

clear_screen()
hide_cursor()

try:
    frame = 0
    start_time = time.time()

    while True:
        t = time.time() - start_time

        # Render frame
        frame_start = time.time()
        gray = render_frame(t)
        lines = to_ascii(gray)
        frame_time = time.time() - frame_start

        # Display
        move_cursor_home()
        print(f"🎮 KVRM Neural Renderer | Frame: {frame} | FPS: {1/frame_time:.1f} | Time: {t:.1f}s")
        print("=" * WIDTH)
        for line in lines:
            print(line)
        print("=" * WIDTH)
        print("Press Ctrl+C to exit")

        frame += 1

        # Cap at ~15 FPS to be readable
        elapsed = time.time() - frame_start
        if elapsed < 0.066:
            time.sleep(0.066 - elapsed)

except KeyboardInterrupt:
    pass
finally:
    show_cursor()
    clear_screen()
    print(f"\n✅ Rendered {frame} frames!")
    print(f"   Average FPS: {frame / (time.time() - start_time):.1f}")
