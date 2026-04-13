; binary_to_bcd.asm - Convert small binary number to BCD (tens and ones)
; Expected result: R2 = 4 (tens digit), R3 = 7 (ones digit) for input 47
;
; This program demonstrates:
;   - Division by 10 via repeated subtraction
;   - Extracting decimal digits
;   - BCD encoding concept
;
; Algorithm:
;   value = 47
;   tens = 0
;   while value >= 10:
;       value -= 10
;       tens++
;   ones = value

    MOV R0, 47      ; value = 47
    MOV R1, 10      ; constant 10
    MOV R2, 0       ; tens digit = 0

tens_loop:
    CMP R0, R1      ; value >= 10?
    JS done_tens    ; if value < 10, done extracting tens

    SUB R0, R0, R1  ; value -= 10
    INC R2          ; tens++
    JMP tens_loop

done_tens:
    MOV R3, R0      ; ones = remaining value

    HALT            ; done - R2 = 4 (tens), R3 = 7 (ones)
