; arithmetic_series.asm - Sum of arithmetic progression: a, a+d, a+2d, ..., a+nd
; Expected result: R2 = 75 (sum of 5, 10, 15, 20, 25 with a=5, d=5, n=5)
;
; This program demonstrates:
;   - Arithmetic progression generation
;   - Step-based iteration
;   - ADD with constant stride
;
; Algorithm:
;   a = 5 (first term), d = 5 (common difference), terms = 5
;   sum = 0, current = a
;   for i = 0 to terms-1:
;       sum += current
;       current += d

    MOV R0, 5       ; a = first term
    MOV R1, 5       ; d = common difference
    MOV R2, 0       ; sum = 0
    MOV R3, 5       ; number of terms
    MOV R4, 0       ; counter
    MOV R5, R0      ; current = a

loop:
    CMP R4, R3      ; counter == terms?
    JZ done

    ADD R2, R2, R5  ; sum += current
    ADD R5, R5, R1  ; current += d
    INC R4          ; counter++
    JMP loop

done:
    HALT            ; done - R2 = 75 (5+10+15+20+25)
