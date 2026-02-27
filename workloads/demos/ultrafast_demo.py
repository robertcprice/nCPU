#!/usr/bin/env python3
"""🚀 ULTRA-FAST Neural Renderer - optimized for real-time!"""

import torch
import torch.nn as nn
import time, os, sys

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1))
        self.norm = nn.GroupNorm(8, c)
    def forward(self, x): return self.norm(x + self.net(x))

class CNNRenderer(nn.Module):
    def __init__(self, c=64, blocks=8):
        super().__init__()
        self.enc = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(c) for _ in range(blocks)])
        self.out = nn.Conv2d(c, 3, 3, padding=1)
    def forward(self, x):
        x = self.enc(x)
        for b in self.blocks: x = b(x)
        return torch.sigmoid(self.out(x))

model = CNNRenderer().to(device)
try:
    ckpt = torch.load('models/renderer_fast_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
except: pass
model.eval()

# Pre-compute EVERYTHING
W, H = 100, 40
CHARS = " .-:=+*#%@"
CHAR_ARRAY = list(CHARS)

# Pre-compute grids on GPU
y = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
x = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)
dist = torch.sqrt((x-0.5)**2 + (y-0.5)**2)

# Pre-allocate output buffer
output_buffer = [[' '] * W for _ in range(H)]

@torch.no_grad()
def render(t):
    # Create input (vectorized, on GPU)
    r = 0.5 + 0.5 * torch.sin(dist * 15 - t * 5)
    g = 0.5 + 0.5 * torch.sin(x * 10 + t * 4)
    b_val = 0.5 + 0.5 * torch.sin(torch.tensor(t * 6, device=device))
    inp = torch.stack([r, g, torch.full_like(r, b_val)], 0).unsqueeze(0)

    # Neural render
    out = model(inp)[0]
    gray = (0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2])

    # Sync before transfer
    if device == 'mps':
        torch.mps.synchronize()

    # Convert to indices (still on GPU)
    idx = (gray.clamp(0, 0.999) * len(CHARS)).int().cpu().numpy()

    return idx

def to_string(idx):
    """Ultra-fast string building"""
    lines = []
    for row in idx:
        lines.append(''.join(CHAR_ARRAY[i] for i in row))
    return '\n'.join(lines)

print(f"🚀 ULTRA-FAST Neural Renderer | {device.upper()} | {W}x{H}")
input("Press Enter...")

os.system('clear')
sys.stdout.write('\033[?25l')

try:
    frame = 0
    start = time.time()

    while True:
        t = time.time() - start

        t0 = time.time()
        idx = render(t)
        output = to_string(idx)
        ft = time.time() - t0
        fps = 1/ft if ft > 0 else 999

        sys.stdout.write('\033[H')
        sys.stdout.write(f"🎮 Frame {frame:5d} | FPS: {fps:6.1f} | t={t:.1f}s\n")
        sys.stdout.write("=" * W + "\n")
        sys.stdout.write(output)
        sys.stdout.write("\n" + "=" * W + "\n")
        sys.stdout.flush()

        frame += 1

        # Slight delay to not overwhelm terminal
        if ft < 0.02:
            time.sleep(0.02 - ft)

except KeyboardInterrupt:
    pass
finally:
    sys.stdout.write('\033[?25h')
    elapsed = time.time() - start
    print(f"\n\n✅ {frame} frames in {elapsed:.1f}s = {frame/elapsed:.1f} FPS avg")
