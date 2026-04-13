; multiply.asm - Multiply 7 * 6 using repeated addition
; Expected result: R0 = 42
;
; This program demonstrates:
;   - Multiplication via repeated addition
;   - Countdown loop pattern
;   - Zero-flag based termination
;
; Algorithm:
;   result = 0
;   multiplicand = 7
;   multiplier = 6
;   while multiplier != 0:
;       result += multiplicand
;       multiplier--
;   halt

    MOV R0, 0       ; result = 0
    MOV R1, 7       ; multiplicand
    MOV R2, 6       ; multiplier (countdown)
    MOV R3, 1       ; decrement constant
    MOV R4, 0       ; zero for comparison

loop:
    ADD R0, R0, R1  ; result += multiplicand
    SUB R2, R2, R3  ; multiplier--
    CMP R2, R4      ; compare multiplier to zero
    JNZ loop        ; continue if multiplier != 0

    HALT            ; done - R0 should equal 42
