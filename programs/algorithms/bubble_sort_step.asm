; bubble_sort_step.asm - One pass of bubble sort on 4 register values
; Expected result: R0=3, R1=5, R2=7, R3=9 (sorted from 5,3,9,7)
;
; This program demonstrates:
;   - Compare and conditional swap pattern
;   - Multiple comparison-swap sequences
;   - Simulated bubble sort pass using registers
;
; Algorithm:
;   values = [5, 3, 9, 7] in R0-R3
;   Pass 1: compare adjacent pairs and swap if out of order
;   Pass 2: repeat to ensure sorted (4 elements need at most 3 passes)

    MOV R0, 5       ; [0] = 5
    MOV R1, 3       ; [1] = 3
    MOV R2, 9       ; [2] = 9
    MOV R3, 7       ; [3] = 7
    MOV R4, 3       ; pass counter (need at most n-1 passes)

pass:
    CMP R4, R7      ; pass counter == 0? (R7 starts at 0)
    JZ done

    ; Compare R0, R1
    CMP R0, R1      ; R0 > R1?
    JS skip01       ; if R0 < R1, no swap
    JZ skip01       ; if R0 == R1, no swap
    ; Swap R0, R1 via XOR
    XOR R0, R0, R1
    XOR R1, R0, R1
    XOR R0, R0, R1

skip01:
    ; Compare R1, R2
    CMP R1, R2
    JS skip12
    JZ skip12
    XOR R1, R1, R2
    XOR R2, R1, R2
    XOR R1, R1, R2

skip12:
    ; Compare R2, R3
    CMP R2, R3
    JS skip23
    JZ skip23
    XOR R2, R2, R3
    XOR R3, R2, R3
    XOR R2, R2, R3

skip23:
    DEC R4          ; pass counter--
    JMP pass

done:
    HALT            ; done - R0=3, R1=5, R2=7, R3=9
