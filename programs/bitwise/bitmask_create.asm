; bitmask_create.asm - Create bitmasks of various widths
; Expected result: R2 = 0x1F (5-bit mask), R4 = 0xFF (8-bit mask)
;
; This program demonstrates:
;   - Creating bitmask of width N: (1 << N) - 1
;   - Shift and subtract pattern
;   - Multiple mask creation
;
; Algorithm:
;   mask_5 = (1 << 5) - 1 = 31 (0x1F)
;   mask_8 = (1 << 8) - 1 = 255 (0xFF)

    MOV R0, 1       ; base value
    MOV R1, 1       ; constant 1

    ; Create 5-bit mask: 0b11111 = 31
    SHL R2, R0, 5   ; 1 << 5 = 32
    SUB R2, R2, R1  ; 32 - 1 = 31 (0x1F)

    ; Create 8-bit mask: 0b11111111 = 255
    SHL R4, R0, 8   ; 1 << 8 = 256
    SUB R4, R4, R1  ; 256 - 1 = 255 (0xFF)

    ; Create 3-bit mask: 0b111 = 7
    SHL R5, R0, 3   ; 1 << 3 = 8
    SUB R5, R5, R1  ; 8 - 1 = 7

    HALT            ; done - R2 = 31, R4 = 255, R5 = 7
