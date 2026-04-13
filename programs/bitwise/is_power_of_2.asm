; is_power_of_2.asm - Check if a number is a power of 2
; Expected result: R2 = 1 (64 is a power of 2), R5 = 0 (65 is not)
;
; This program demonstrates:
;   - Classic n & (n-1) == 0 trick for power-of-2 detection
;   - Bitwise AND for bit pattern testing
;   - SUB to compute n-1
;
; Algorithm:
;   if n > 0 and (n & (n-1)) == 0: result = 1 (is power of 2)
;   else: result = 0

    ; Test 1: n = 64 (is power of 2)
    MOV R0, 64      ; n = 64
    MOV R1, 1       ; constant 1
    MOV R6, 0       ; zero constant

    SUB R3, R0, R1  ; n - 1 = 63
    AND R3, R0, R3  ; n & (n-1)
    CMP R3, R6      ; compare to 0
    JNZ not_pow2_1

    MOV R2, 1       ; result = 1 (is power of 2)
    JMP test2

not_pow2_1:
    MOV R2, 0       ; result = 0

test2:
    ; Test 2: n = 65 (not power of 2)
    MOV R0, 65      ; n = 65
    SUB R4, R0, R1  ; n - 1 = 64
    AND R4, R0, R4  ; n & (n-1)
    CMP R4, R6      ; compare to 0
    JNZ not_pow2_2

    MOV R5, 1       ; result = 1
    JMP done

not_pow2_2:
    MOV R5, 0       ; result = 0

done:
    HALT            ; done - R2 = 1 (64 is pow2), R5 = 0 (65 is not)
