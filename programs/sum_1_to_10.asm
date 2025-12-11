; sum_1_to_10.asm - Calculate sum of integers from 1 to 10
; Expected result: R0 = 55
;
; This program demonstrates:
;   - Loop control with conditional jumps
;   - Register-based iteration
;   - Accumulator pattern
;
; Algorithm:
;   sum = 0
;   counter = 1
;   while counter < 11:
;       sum += counter
;       counter++
;   halt

    MOV R0, 0       ; sum = 0 (accumulator)
    MOV R1, 1       ; counter = 1
    MOV R2, 11      ; limit (exclusive)
    MOV R3, 1       ; increment constant

loop:
    ADD R0, R0, R1  ; sum += counter
    ADD R1, R1, R3  ; counter++
    CMP R1, R2      ; compare counter to limit
    JNZ loop        ; continue if counter != limit

    HALT            ; done - R0 should equal 55
