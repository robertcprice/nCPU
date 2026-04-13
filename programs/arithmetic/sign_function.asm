; sign_function.asm - Compute sign(x): returns -1, 0, or 1
; Expected result: R1 = -1 (sign of -42)
;
; This program demonstrates:
;   - Three-way branching pattern
;   - Sign detection and zero detection
;
; Algorithm:
;   x = -42
;   if x == 0: sign = 0
;   elif x < 0: sign = -1
;   else: sign = 1

    MOV R0, -42     ; x = -42
    MOV R2, 0       ; zero constant

    CMP R0, R2      ; compare x to 0
    JZ is_zero      ; if x == 0
    JS is_negative  ; if x < 0

    ; x > 0
    MOV R1, 1       ; sign = 1
    JMP done

is_negative:
    MOV R1, -1      ; sign = -1
    JMP done

is_zero:
    MOV R1, 0       ; sign = 0

done:
    HALT            ; done - R1 = -1
