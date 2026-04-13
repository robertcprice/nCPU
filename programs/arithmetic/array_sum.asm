; array_sum.asm - Sum of "array" elements simulated with registers
; Expected result: R7 = 150 (sum of 10 + 20 + 30 + 40 + 50)
;
; This program demonstrates:
;   - Simulating an array using registers
;   - Sequential accumulation
;   - Multiple ADD instructions in sequence
;
; Algorithm:
;   array = [10, 20, 30, 40, 50] stored in R0-R4
;   sum = R0 + R1 + R2 + R3 + R4

    MOV R0, 10      ; array[0] = 10
    MOV R1, 20      ; array[1] = 20
    MOV R2, 30      ; array[2] = 30
    MOV R3, 40      ; array[3] = 40
    MOV R4, 50      ; array[4] = 50

    ADD R7, R0, R1  ; sum = 10 + 20
    ADD R7, R7, R2  ; sum += 30
    ADD R7, R7, R3  ; sum += 40
    ADD R7, R7, R4  ; sum += 50

    HALT            ; done - R7 = 150
