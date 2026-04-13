; gcd.asm - Greatest Common Divisor using Euclidean algorithm
; Expected result: R0 = 12 (GCD of 48 and 36)
;
; This program demonstrates:
;   - Euclidean GCD algorithm
;   - Subtraction-based approach (no modulo)
;   - Two conditional branches in a loop
;
; Algorithm (subtraction-based):
;   a = 48, b = 36
;   while a != b:
;       if a > b: a = a - b
;       else: b = b - a
;   gcd = a

    MOV R0, 48      ; a = 48
    MOV R1, 36      ; b = 36

loop:
    CMP R0, R1      ; compare a and b
    JZ done         ; if a == b, we found GCD
    JS b_greater    ; if a < b (sign flag set), b is greater

    ; a > b case
    SUB R0, R0, R1  ; a = a - b
    JMP loop

b_greater:
    ; b > a case
    SUB R1, R1, R0  ; b = b - a
    JMP loop

done:
    HALT            ; done - R0 = GCD(48, 36) = 12
