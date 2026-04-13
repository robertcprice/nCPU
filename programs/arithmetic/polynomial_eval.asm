; polynomial_eval.asm - Evaluate polynomial using Horner's method
; Expected result: R0 = 123 for p(x) = 2x^2 + 5x + 3 at x=7
; Verification: 2*49 + 5*7 + 3 = 98 + 35 + 3 = 136... wait
; Horner: ((2)*7 + 5)*7 + 3 = (14+5)*7 + 3 = 19*7 + 3 = 133 + 3 = 136
; Expected result: R0 = 136
;
; This program demonstrates:
;   - Horner's method for polynomial evaluation
;   - Shift-and-accumulate pattern
;   - MUL and ADD chain
;
; Algorithm:
;   p(x) = 2x^2 + 5x + 3, x = 7
;   Horner: result = ((a2 * x) + a1) * x + a0
;   result = ((2 * 7) + 5) * 7 + 3 = 136

    MOV R1, 7       ; x = 7
    MOV R2, 2       ; a2 (coefficient of x^2)
    MOV R3, 5       ; a1 (coefficient of x^1)
    MOV R4, 3       ; a0 (constant term)

    ; Horner's method: ((a2 * x + a1) * x + a0)
    MUL R0, R2, R1  ; R0 = a2 * x = 14
    ADD R0, R0, R3  ; R0 = 14 + a1 = 19
    MUL R0, R0, R1  ; R0 = 19 * x = 133
    ADD R0, R0, R4  ; R0 = 133 + a0 = 136

    HALT            ; done - R0 = 136
