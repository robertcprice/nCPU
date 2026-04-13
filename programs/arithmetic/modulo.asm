; modulo.asm - Compute modulo (remainder) of two numbers
; Expected result: R2 = 3 (17 mod 7)
;
; This program demonstrates:
;   - Modulo via repeated subtraction
;   - Simple loop with sign-based termination
;
; Algorithm:
;   a = 17, b = 7
;   while a >= b:
;       a -= b
;   result = a

    MOV R0, 17      ; a = 17
    MOV R1, 7       ; b = 7
    MOV R2, R0      ; result = a (working copy)

loop:
    CMP R2, R1      ; compare result to b
    JS done         ; if result < b, done

    SUB R2, R2, R1  ; result -= b
    JMP loop

done:
    HALT            ; done - R2 = 3 (17 mod 7)
