; divmod.asm - Division with remainder (quotient and modulo)
; Expected result: R2 = 5 (quotient), R3 = 2 (remainder) for 47 / 9
;
; This program demonstrates:
;   - Repeated subtraction division
;   - Computing both quotient and remainder
;   - Combined comparison and subtraction loop
;
; Algorithm:
;   dividend = 47, divisor = 9
;   quotient = 0
;   remainder = dividend
;   while remainder >= divisor:
;       remainder -= divisor
;       quotient++

    MOV R0, 47      ; dividend
    MOV R1, 9       ; divisor
    MOV R2, 0       ; quotient = 0
    MOV R3, R0      ; remainder = dividend

loop:
    CMP R3, R1      ; compare remainder to divisor
    JS done         ; if remainder < divisor, done

    SUB R3, R3, R1  ; remainder -= divisor
    INC R2          ; quotient++
    JMP loop

done:
    HALT            ; done - R2 = 5 (quotient), R3 = 2 (remainder)
