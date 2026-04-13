; twos_complement.asm - Compute two's complement negation
; Expected result: R2 = -42 (two's complement of 42)
;
; This program demonstrates:
;   - Two's complement: -x = ~x + 1 = (x XOR 0xFFFFFFFF) + 1
;   - XOR with all-ones for bitwise NOT
;   - Classic negation technique
;
; Algorithm:
;   x = 42
;   For 8-bit demo: NOT via XOR with 0xFF, then add 1
;   Using SUB from 0 for the actual negation

    MOV R0, 42      ; x = 42
    MOV R1, 0       ; zero

    ; Method 1: SUB from zero
    SUB R2, R1, R0  ; -x = 0 - x = -42

    ; Method 2: XOR with all-ones + 1 (8-bit demo)
    MOV R3, 0xFF    ; all-ones mask (8-bit)
    MOV R4, 1       ; constant 1
    XOR R5, R0, R3  ; ~x (8-bit) = 0xFF ^ 42 = 213
    ADD R5, R5, R4  ; ~x + 1 = 214 (two's complement in 8 bits)

    HALT            ; done - R2 = -42, R5 = 214 (8-bit two's complement)
