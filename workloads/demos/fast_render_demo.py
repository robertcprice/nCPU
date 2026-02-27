#!/usr/bin/env python3
"""
🎮 FAST Neural Renderer Demo
Uses vectorized operations for 10x+ speedup!
"""

import torch
import torch.nn as nn
import time
import os
import sys
import math

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))
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

# Load model
model = CNNRenderer()
try:
    ckpt = torch.load('models/renderer_fast_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
except:
    pass
model = model.to(device)
model.eval()

# Pre-compute coordinate grids (FAST!)
W, H = 80, 35
y_grid = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
x_grid = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)
cx_grid = x_grid - 0.5
cy_grid = y_grid - 0.5
dist_grid = torch.sqrt(cx_grid**2 + cy_grid**2)

CHARS = " .-:=+*#%@"

def render_fast(t):
    """Vectorized rendering - no Python loops!"""
    with torch.no_grad():
        # Create animated input - ALL AT ONCE, no loops
        r = 0.5 + 0.5 * torch.sin(dist_grid * 12 - t * 4)
        g = 0.5 + 0.5 * torch.sin(x_grid * 8 + t * 3)
        b = 0.5 + 0.5 * torch.sin(t * 5)

        inp = torch.stack([r, g, torch.full_like(r, b)], dim=0).unsqueeze(0)

        out = model(inp)[0].cpu()
        gray = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2]

        return gray

def to_ascii_fast(gray):
    """Fast ASCII conversion"""
    idx = (gray.clamp(0, 0.999) * len(CHARS)).int()
    lines = []
    for y in range(H):
        lines.append(''.join(CHARS[i] for i in idx[y].tolist()))
    return lines

def clear():
    os.system('clear' if os.name == 'posix' else 'cls')

def cursor_home():
    sys.stdout.write('\033[H')
    sys.stdout.flush()

print(f"🎮 FAST Neural Renderer | Device: {device.upper()}")
print(f"   Resolution: {W}x{H}")
print("Press Ctrl+C to stop")
input("\nPress Enter to start...")

clear()
sys.stdout.write('\033[?25l')  # Hide cursor

try:
    frame = 0
    start = time.time()
    frame_times = []

    while True:
        t = time.time() - start

        t0 = time.time()
        gray = render_fast(t)
        lines = to_ascii_fast(gray)
        frame_time = time.time() - t0
        frame_times.append(frame_time)

        # Rolling average FPS
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / sum(frame_times)

        cursor_home()
        print(f"🎮 KVRM Neural Renderer | Frame: {frame:4d} | FPS: {avg_fps:5.1f} | t={t:.1f}s")
        print("=" * W)
        print('\n'.join(lines))
        print("=" * W)

        frame += 1

        # Target 30 FPS
        elapsed = time.time() - t0
        if elapsed < 0.033:
            time.sleep(0.033 - elapsed)

except KeyboardInterrupt:
    pass
finally:
    sys.stdout.write('\033[?25h')  # Show cursor
    print(f"\n\n✅ Rendered {frame} frames at avg {frame/(time.time()-start):.1f} FPS")
