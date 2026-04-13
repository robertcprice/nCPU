; sum_of_squares.asm - Compute 1^2 + 2^2 + 3^2 + ... + 6^2
; Expected result: R0 = 91 (1+4+9+16+25+36 = 91)
;
; This program demonstrates:
;   - Squaring via MUL
;   - Accumulator with multiplication in loop
;   - Classic mathematical series
;
; Algorithm:
;   sum = 0
;   for i = 1 to 6:
;       sum += i * i

    MOV R0, 0       ; sum = 0
    MOV R1, 1       ; i = 1
    MOV R2, 7       ; limit (exclusive)

loop:
    CMP R1, R2      ; i == 7?
    JZ done

    MUL R3, R1, R1  ; sq = i * i
    ADD R0, R0, R3  ; sum += sq
    INC R1          ; i++
    JMP loop

done:
    HALT            ; done - R0 = 91
