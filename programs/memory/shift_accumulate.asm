; shift_accumulate.asm - Build a value by shifting and accumulating bits
; Expected result: R0 = 0xAB = 171 (built bit-group by bit-group)
;
; This program demonstrates:
;   - Building values via shift-left and OR
;   - Nibble-by-nibble construction
;   - Accumulator pattern with shifts
;
; Algorithm:
;   Build 0xAB = 0b10101011 nibble by nibble:
;   result = 0
;   result = (result << 4) | 0xA   ; high nibble
;   result = (result << 4) | 0xB   ; low nibble

    MOV R0, 0       ; result = 0
    MOV R1, 0x0A    ; high nibble = 0xA
    MOV R2, 0x0B    ; low nibble = 0xB

    ; Shift and accumulate high nibble
    SHL R0, R0, 4   ; result <<= 4 (still 0)
    OR  R0, R0, R1  ; result |= 0xA -> result = 0x0A

    ; Shift and accumulate low nibble
    SHL R0, R0, 4   ; result <<= 4 -> 0xA0
    OR  R0, R0, R2  ; result |= 0xB -> result = 0xAB = 171

    HALT            ; done - R0 = 171 (0xAB)
