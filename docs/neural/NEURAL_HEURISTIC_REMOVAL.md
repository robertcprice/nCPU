# Heuristic Removal Plan

## Current Heuristics to Remove:

### 1. MOVZ Category Forcing (lines 688-697)
```python
# REMOVE THIS:
if op in [0xD2, 0x52, 0xF2, 0x72, 0x12, 0x32]:
    category = 12
    rd = inst & 0x1F
    rn = 31
    rm = 31
```
**REPLACE WITH:** Trust decoder's category output

### 2. ADD/SUB Immediate Override (lines 704-716)
```python
# REMOVE THIS:
if op in [0x91, 0x51, 0x11, 0xb1, 0xd1, 0x71, 0x31, 0xf1]:
    rm = 31
    rd = (inst >> 0) & 0x1F
    rn = (inst >> 5) & 0x1F
    if op in [0x51, 0xf1, 0x11, 0xb1]:
        sets_flags = True
    # Force category...
```
**REPLACE WITH:** Use decoder's rd, rn, rm, sets_flags outputs

### 3. Branch Condition Override (lines 718-724)
```python
# REMOVE THIS:
if op == 0x54:
    category = 10
    cond = inst & 0xF
    # ...
```
**REPLACE WITH:** Trust decoder's branch classification and condition extraction

### 4. Flag Setting in ADD/SUB (lines 773-776, 805-808, 818-822)
```python
# REMOVE THIS:
if sets_flags:
    self.z = (result == 0)
    self.n = (result >> 63) & 1
    self.c = ...
```
**REPLACE WITH:** Use decoder's sets_flags output

## Changes Needed:

### In `_neural_execute` (step function):

1. **Extract decoder outputs:**
```python
sets_flags = decoded['sets_flags'][0, 0].item() > 0
# Add this line to extract sets_flags from decoder
```

2. **Remove all heuristic overrides**

3. **Use decoder's outputs directly**

### In category execution:

1. **Remove manual flag-setting code** - use `sets_flags` variable from decoder

2. **Trust decoder's predictions** - no manual extraction of bits

### Model Loading:

```python
# Load pure neural model
decoder_path = Path('models/final/decoder_pure_neural.pt')
if decoder_path.exists():
    checkpoint = torch.load(decoder_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Pure neural decoder loaded (no heuristics)")
    print(f"   Accuracy: {checkpoint['accuracies']}")
else:
    print(f"⚠️ Pure neural model not found, using fallback")
```

## Testing After Training:

1. Test decoder on MOVZ instructions → should classify as MOVE (cat 12)
2. Test decoder on ADDS/SUBS → should have sets_flags=True
3. Test decoder on B.cond → should classify as BRANCH (cat 10)
4. Test loop execution → should work purely neurally

## Success Criteria:

- [ ] MOVZ instructions classified correctly (neural, no heuristic)
- [ ] ADDS/SUBS set flags correctly (neural, no manual flag setting)
- [ ] Branches execute correctly (neural condition check)
- [ ] DOOM loop runs without heuristics
- [ ] Speedup measured with pure neural optimization
