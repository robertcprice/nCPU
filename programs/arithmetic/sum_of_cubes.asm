; sum_of_cubes.asm - Compute 1^3 + 2^3 + 3^3 + 4^3
; Expected result: R0 = 100 (1+8+27+64 = 100)
; Note: sum of first n cubes = (n*(n+1)/2)^2, for n=4: (10)^2 = 100
;
; This program demonstrates:
;   - Cubing via two multiplications
;   - Mathematical identity verification
;   - Loop with MUL chain
;
; Algorithm:
;   sum = 0
;   for i = 1 to 4:
;       sum += i * i * i

    MOV R0, 0       ; sum = 0
    MOV R1, 1       ; i = 1
    MOV R2, 5       ; limit (exclusive)

loop:
    CMP R1, R2      ; i == 5?
    JZ done

    MUL R3, R1, R1  ; sq = i * i
    MUL R3, R3, R1  ; cube = sq * i
    ADD R0, R0, R3  ; sum += cube
    INC R1          ; i++
    JMP loop

done:
    HALT            ; done - R0 = 100
