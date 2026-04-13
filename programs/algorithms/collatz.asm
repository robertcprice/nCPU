; collatz.asm - Count Collatz sequence steps until reaching 1
; Expected result: R1 = 8 (Collatz sequence for 6: 6,3,10,5,16,8,4,2,1 = 8 steps)
;
; This program demonstrates:
;   - Collatz conjecture (3n+1 problem)
;   - Even/odd branching with AND
;   - Mixed arithmetic operations
;
; Algorithm:
;   n = 6
;   steps = 0
;   while n != 1:
;       if n is even: n = n / 2 (SHR by 1)
;       else: n = 3*n + 1
;       steps++

    MOV R0, 6       ; n = 6
    MOV R1, 0       ; steps = 0
    MOV R2, 1       ; constant 1
    MOV R3, 3       ; constant 3
    MOV R4, 0       ; zero for AND test

loop:
    CMP R0, R2      ; n == 1?
    JZ done

    AND R5, R0, R2  ; test if n is odd (n & 1)
    CMP R5, R4      ; is it zero (even)?
    JNZ odd         ; if not zero, n is odd

    ; even case: n = n / 2
    SHR R0, R0, 1   ; n >>= 1
    INC R1          ; steps++
    JMP loop

odd:
    ; odd case: n = 3*n + 1
    MUL R0, R0, R3  ; n = 3 * n
    ADD R0, R0, R2  ; n = n + 1
    INC R1          ; steps++
    JMP loop

done:
    HALT            ; done - R1 = 8 steps
