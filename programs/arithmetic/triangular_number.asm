; triangular_number.asm - Compute the Nth triangular number using formula
; Expected result: R2 = 28 (T(7) = 7 * 8 / 2 = 28)
;
; This program demonstrates:
;   - Direct formula: T(n) = n * (n + 1) / 2
;   - MUL and SHR combination
;   - Avoiding loops via closed-form computation
;
; Algorithm:
;   n = 7
;   T(n) = n * (n + 1) / 2

    MOV R0, 7       ; n = 7
    MOV R1, 1       ; constant 1

    ADD R1, R0, R1  ; n + 1 = 8
    MUL R2, R0, R1  ; n * (n + 1) = 56
    SHR R2, R2, 1   ; / 2 = 28

    HALT            ; done - R2 = 28
