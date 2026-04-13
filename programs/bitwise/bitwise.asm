; bitwise.asm — Demonstrates AND, OR, XOR, SHL, SHR
;
; All operations use trained neural networks in --mode neural:
;   AND/OR/XOR → NeuralLogical truth tables (bit-by-bit)
;   SHL/SHR    → NeuralShiftNet (shift_decoder + index_net + validity_net)
;
; Expected results:
;   R2 = 0x0F (AND)
;   R3 = 0xFF (OR)
;   R4 = 0xF0 (XOR)
;   R5 = 8    (SHL: 1 << 3)
;   R6 = 4    (SHR: 16 >> 2)

    MOV R0, 0xFF       ; 11111111
    MOV R1, 0x0F       ; 00001111

    ; Bitwise operations
    AND R2, R0, R1     ; R2 = 0xFF & 0x0F = 0x0F
    OR  R3, R0, R1     ; R3 = 0xFF | 0x0F = 0xFF
    XOR R4, R0, R1     ; R4 = 0xFF ^ 0x0F = 0xF0

    ; Shift operations
    MOV R0, 1
    SHL R5, R0, 3      ; R5 = 1 << 3 = 8

    MOV R0, 16
    SHR R6, R0, 2      ; R6 = 16 >> 2 = 4

    HALT
