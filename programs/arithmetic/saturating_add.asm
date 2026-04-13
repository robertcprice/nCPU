; saturating_add.asm - Add two numbers with saturation at max value
; Expected result: R3 = 255 (saturated), R6 = 150 (not saturated)
;
; This program demonstrates:
;   - Saturating arithmetic (clamp to max)
;   - Overflow detection via comparison
;   - Conditional clamping pattern
;
; Algorithm:
;   Saturating add to 255 (8-bit max):
;   result = a + b
;   if result > 255: result = 255

    ; Test 1: overflow case (200 + 100 = 300, saturates to 255)
    MOV R0, 200     ; a = 200
    MOV R1, 100     ; b = 100
    MOV R2, 255     ; max value

    ADD R3, R0, R1  ; result = 200 + 100 = 300
    CMP R3, R2      ; result > 255?
    JS no_sat1      ; if result < max, no saturation needed
    JZ no_sat1      ; if result == max, also fine

    MOV R3, R2      ; saturate: result = 255

no_sat1:
    ; Test 2: no overflow case (100 + 50 = 150)
    MOV R4, 100     ; a = 100
    MOV R5, 50      ; b = 50

    ADD R6, R4, R5  ; result = 100 + 50 = 150
    CMP R6, R2      ; result > 255?
    JS no_sat2
    JZ no_sat2

    MOV R6, R2      ; saturate (not reached)

no_sat2:
    HALT            ; done - R3 = 255 (saturated), R6 = 150 (not saturated)
