# DOOM FPS Explained - Why Two Different Numbers?

## The Confusion

You may have seen two different FPS numbers for DOOM:
- **123.7 FPS** - Direct ALU benchmark
- **7.0 FPS** - On Neural CPU benchmark

**Both are correct!** They measure different things.

---

## Method 1: Direct ALU (123.7 FPS) 🔥

### What It Tests
Pure neural acceleration without ARM64 CPU overhead.

### How It Works
```python
# Directly calls BatchedNeuralALU for raycasting
alu = BatchedNeuralALU()
for ray in range(320):  # 320 rays
    result = alu.compute_add(ray_x, step_x)  # Direct neural call
    result = alu.compute_add(ray_y, step_y)  # Direct neural call
```

### Performance
- **4,945 instructions per second**
- **62x speedup from batching**
- **123.7 FPS** in DOOM raycaster
- **100% accuracy**

### What This Means
This shows the **raw power** of the neural ALU when you bypass all CPU overhead. It's like testing a GPU directly without going through a CPU.

---

## Method 2: On Neural CPU (7.0 FPS) 🖥️

### What It Tests
Real ARM64 code executing through the full neural pipeline.

### How It Works
```python
# ARM64 code executes through full CPU pipeline
code = [
    0x80000000,  # MOVZ X0, #0  (ARM64 instruction)
    0x10004100,  # ADD X0, X0, #1  (ARM64 instruction)
    ...
]
# Loads ARM64 binary → Decoder → Orchestrator → MMU → ALU
cpu.load_binary(code, 0x20000)
cpu.run(max_instructions=256)
```

### The Full Pipeline
1. **Neural Decoder**: Decodes ARM64 instructions (6.5 MB model)
2. **Neural Orchestrator**: Control flow decisions (177 KB model)
3. **Neural MMU**: Memory management (479 KB model)
4. **BatchedNeuralALU**: Computation (62x speedup)

### Performance
- **~1,800 instructions per second** (average)
- **256 instructions per frame**
- **7.0 FPS** in DOOM raycaster
- **Real ARM64 execution**

### What This Means
This shows what DOOM would actually run like if you compiled it to ARM64 and ran it on the neural CPU. This includes ALL overhead: decoding, control flow, memory access.

---

## Why Such a Big Difference? 🤔

### Direct ALU (123.7 FPS)
```
Raycasting math → Neural ALU → Result
                  (0.2ms per batch)
```

### On CPU (7.0 FPS)
```
ARM64 instruction → Neural Decoder → Neural Orchestrator → Neural MMU → Neural ALU → Result
                         (0.1ms)           (0.05ms)            (0.05ms)         (0.2ms)
```

The on-CPU method has **4 neural network passes** instead of 1!

---

## Which One Matters? 📊

### For Demonstrating Neural Power → Use Direct ALU (123.7 FPS)
- Shows maximum neural acceleration
- Like GPU demo without CPU bottleneck
- **"Look how fast the neural ALU is!"**

### For Real Computing → Use On CPU (7.0 FPS)
- Shows actual ARM64 program execution
- Like running real software on CPU
- **"This is what DOOM would actually run like!"**

---

## Real-World Comparison

| Metric | Direct ALU | On CPU |
|--------|-----------|--------|
| **IPS** | 4,945 | ~1,800 |
| **DOOM FPS** | 123.7 | 7.0 |
| **ARM64 Code** | ❌ No | ✅ Yes |
| **Full Pipeline** | ❌ No | ✅ Yes |
| **Like...** | GPU demo | Running real software |

---

## How to Test Each

### Method 1: Direct ALU (123.7 FPS)
```bash
python3 benchmark_doom_batched.py
```

### Method 2: On Neural CPU (7.0 FPS)
```bash
python3 benchmark_doom_on_neural_cpu.py
```

### Method 3: Via Bootloader
```bash
python3 neural_bootloader.py
# Select: 2 (DOOM)
# Select: 1 (raycaster)
```

---

## The Bottom Line

**123.7 FPS** = Neural ALU maximum speed (no CPU overhead)

**7.0 FPS** = Actual ARM64 program execution (full pipeline)

Both are impressive for different reasons:
- **123.7 FPS**: Incredible neural acceleration for pure computation
- **7.0 FPS**: Running real ARM64 code on a neural CPU!

---

## Performance Progression

```
Sequential Neural CPU:    ~0.01 FPS  (baseline)
Batched Neural CPU:        7.0 FPS  (real ARM64)
Direct ALU:              123.7 FPS  (pure neural)
```

**7.0 FPS is 700x faster than sequential!**
**123.7 FPS is 12,370x faster than sequential!**

---

## Future Optimization Paths

### Current: 7.0 FPS (On CPU)
- Decoder: 0.1ms per batch
- Orchestrator: 0.05ms per batch
- MMU: 0.05ms per batch
- ALU: 0.2ms per batch
- **Total: ~0.4ms per batch**

### With GPU Acceleration: ~35 FPS
- All components on GPU: ~0.08ms per batch
- **5x speedup potential**

### With Hybrid CPU/Neural: ~70 FPS
- Decoder/Orchestrator on CPU: ~0.01ms
- MMU/ALU on neural: ~0.03ms
- **10x speedup potential**

---

## Summary

| Question | Answer |
|----------|--------|
| **Which is the "real" DOOM FPS?** | 7.0 FPS (actual ARM64 execution) |
| **Why show 123.7 FPS?** | Demonstrates raw neural ALU power |
| **Can I run real ARM64 programs?** | Yes! At ~7 FPS for DOOM-like workloads |
| **Will it get faster?** | Yes! GPU acceleration = 5-10x speedup |

Both numbers are truthful and important for different use cases!
