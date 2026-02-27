#!/usr/bin/env python3
"""Run REAL Alpine busybox on Neural CPU - NO FAKES."""

import sys
import time
sys.path.insert(0, '.')

print("=" * 60)
print(" RUNNING REAL ALPINE BUSYBOX - 1.1MB BINARY")
print("=" * 60)

from neural_kernel import NeuralARM64Kernel

kernel = NeuralARM64Kernel()

# Load the REAL busybox - 1,116,408 bytes
with open('binaries/busybox-static', 'rb') as f:
    busybox = f.read()

print(f"\nLoaded REAL busybox: {len(busybox):,} bytes")
print("This is Alpine Linux's actual busybox binary!")
print("\nExecuting: busybox echo 'Hello from REAL Alpine!'")
print("-" * 60)

start = time.time()
result = kernel.run_elf(busybox, ['busybox', 'echo', 'Hello from REAL Alpine!'])
elapsed = time.time() - start

print("-" * 60)
print(f"Exit: {result[0]}, Time: {elapsed:.2f}s")
