; clamp.asm - Clamp a value between min and max bounds
; Expected result: R3 = 10 (clamped from -5 to [10,200]),
;                  R6 = 200 (clamped from 300 to [10,200]),
;                  R7 = 50 (50 is already within [10,200])
;
; This program demonstrates:
;   - Range clamping pattern
;   - Multiple comparison branches
;   - Common graphics/DSP operation
;
; Algorithm:
;   clamp(value, lo, hi):
;       if value < lo: return lo
;       if value > hi: return hi
;       return value

    MOV R0, 10      ; lo = 10
    MOV R1, 200     ; hi = 200

    ; Test 1: value = -5 (below range)
    MOV R2, -5
    CMP R2, R0      ; value < lo?
    JS clamp_lo1
    CMP R2, R1      ; value > hi?
    JS in_range1
    JZ in_range1
    MOV R3, R1      ; clamp to hi
    JMP test2
clamp_lo1:
    MOV R3, R0      ; clamp to lo
    JMP test2
in_range1:
    MOV R3, R2      ; in range

test2:
    ; Test 2: value = 300 (above range)
    MOV R4, 300
    CMP R4, R0      ; value < lo?
    JS clamp_lo2
    CMP R4, R1      ; value > hi?
    JS in_range2
    JZ in_range2
    MOV R6, R1      ; clamp to hi
    JMP test3
clamp_lo2:
    MOV R6, R0
    JMP test3
in_range2:
    MOV R6, R4

test3:
    ; Test 3: value = 50 (in range)
    MOV R5, 50
    CMP R5, R0
    JS clamp_lo3
    CMP R5, R1
    JS in_range3
    JZ in_range3
    MOV R7, R1
    JMP done
clamp_lo3:
    MOV R7, R0
    JMP done
in_range3:
    MOV R7, R5

done:
    HALT            ; done - R3=10, R6=200, R7=50
