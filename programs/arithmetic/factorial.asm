; factorial.asm - Calculate 7! (factorial of 7)
; Expected result: R0 = 5040
;
; This program demonstrates:
;   - Nested multiplication using MUL instruction
;   - Countdown loop pattern
;   - Accumulator with multiply
;
; Algorithm:
;   result = 1
;   n = 7
;   while n > 1:
;       result *= n
;       n--
;   halt

    MOV R0, 1       ; result = 1
    MOV R1, 7       ; n = 7
    MOV R2, 1       ; constant 1 for comparison
    MOV R3, 1       ; constant 1 for decrement

loop:
    MUL R0, R0, R1  ; result *= n
    SUB R1, R1, R3  ; n--
    CMP R1, R2      ; compare n to 1
    JNZ loop        ; continue if n != 1

    HALT            ; done - R0 should equal 5040
