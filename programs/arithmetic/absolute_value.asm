; absolute_value.asm - Compute absolute value of a negative number
; Expected result: R1 = 25 (absolute value of -25)
;
; This program demonstrates:
;   - Sign detection using CMP
;   - Two's complement negation via subtraction from zero
;   - Conditional branching
;
; Algorithm:
;   x = -25
;   if x >= 0: abs = x
;   else: abs = 0 - x

    MOV R0, -25     ; x = -25
    MOV R1, 0       ; zero constant

    CMP R0, R1      ; compare x to 0
    JNS positive    ; if x >= 0, it is already positive

    ; x < 0: negate it
    SUB R1, R1, R0  ; abs = 0 - x
    JMP done

positive:
    MOV R1, R0      ; abs = x (already positive)

done:
    HALT            ; done - R1 = 25
