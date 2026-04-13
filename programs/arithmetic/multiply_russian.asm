; multiply_russian.asm - Russian peasant multiplication (multiply via doubling/halving)
; Expected result: R2 = 154 (11 * 14 via Russian peasant method)
;
; This program demonstrates:
;   - Ancient multiplication algorithm
;   - Shift-based doubling and halving
;   - Conditional accumulation based on parity
;
; Algorithm:
;   a = 11, b = 14, result = 0
;   while a > 0:
;       if a is odd: result += b
;       a >>= 1  (halve)
;       b <<= 1  (double)

    MOV R0, 11      ; a = 11
    MOV R1, 14      ; b = 14
    MOV R2, 0       ; result = 0
    MOV R3, 1       ; mask for odd test
    MOV R4, 0       ; zero for comparison

loop:
    CMP R0, R4      ; a == 0?
    JZ done

    AND R5, R0, R3  ; test if a is odd
    CMP R5, R4      ; is it zero (even)?
    JZ skip_add

    ADD R2, R2, R1  ; result += b

skip_add:
    SHR R0, R0, 1   ; a >>= 1 (halve)
    SHL R1, R1, 1   ; b <<= 1 (double)
    JMP loop

done:
    HALT            ; done - R2 = 154 (11 * 14)
