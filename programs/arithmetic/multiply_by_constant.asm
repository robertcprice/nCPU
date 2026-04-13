; multiply_by_constant.asm - Multiply by 10 using shifts and adds
; Expected result: R2 = 170 (17 * 10 via shift-add decomposition)
;
; This program demonstrates:
;   - Strength reduction: multiply via shift and add
;   - x * 10 = x * 8 + x * 2 = (x << 3) + (x << 1)
;   - Compiler optimization technique
;
; Algorithm:
;   x = 17
;   result = (x << 3) + (x << 1)   ; x*8 + x*2 = x*10

    MOV R0, 17      ; x = 17

    SHL R1, R0, 3   ; R1 = x << 3 = x * 8 = 136
    SHL R2, R0, 1   ; R2 = x << 1 = x * 2 = 34
    ADD R2, R1, R2  ; result = 136 + 34 = 170

    HALT            ; done - R2 = 170 (17 * 10)
