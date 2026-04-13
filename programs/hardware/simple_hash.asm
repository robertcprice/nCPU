; simple_hash.asm - Simple hash function: hash = ((val * 31) + 7) & 0xFF
; Expected result: R2 result for input 42: (42 * 31 + 7) & 255 = (1302 + 7) & 255 = 1309 & 255 = 29
;
; This program demonstrates:
;   - Simple multiplicative hash function
;   - MUL, ADD, AND in sequence
;   - Bit masking for modular arithmetic
;
; Algorithm:
;   hash(x) = (x * 31 + 7) mod 256
;   Using AND 0xFF for mod 256

    MOV R0, 42      ; input value
    MOV R1, 31      ; hash multiplier
    MOV R3, 7       ; hash addend
    MOV R4, 0xFF    ; mask for mod 256

    MUL R2, R0, R1  ; R2 = 42 * 31 = 1302
    ADD R2, R2, R3  ; R2 = 1302 + 7 = 1309
    AND R2, R2, R4  ; R2 = 1309 & 0xFF = 29

    HALT            ; done - R2 = 29
