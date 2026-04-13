; isqrt.asm - Integer square root approximation
; Expected result: R1 = 10 (floor(sqrt(100)) = 10)
;
; This program demonstrates:
;   - Trial multiplication approach
;   - Incremental search with comparison
;   - Finding floor of square root
;
; Algorithm:
;   n = 100
;   guess = 0
;   while (guess+1)*(guess+1) <= n:
;       guess++
;   result = guess

    MOV R0, 100     ; n = 100
    MOV R1, 0       ; guess = 0
    MOV R2, 1       ; constant 1

loop:
    ADD R3, R1, R2  ; next = guess + 1
    MUL R4, R3, R3  ; next_sq = next * next
    CMP R4, R0      ; compare next_sq to n
    JS ok           ; if next_sq < n, keep going
    JZ ok           ; if next_sq == n, include this value

    JMP done        ; next_sq > n, stop

ok:
    MOV R1, R3      ; guess = next
    JMP loop

done:
    HALT            ; done - R1 = 10
