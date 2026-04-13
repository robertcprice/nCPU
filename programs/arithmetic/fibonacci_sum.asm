; fibonacci_sum.asm - Compute Fibonacci numbers and their running sum
; Expected result: R2 = 88 (sum of first 10 Fibonacci: 1+1+2+3+5+8+13+21+34 = 88... wait)
; Actually: F(1)=1,F(2)=1,F(3)=2,F(4)=3,F(5)=5,F(6)=8,F(7)=13,F(8)=21,F(9)=34 = 88
; Expected: R2 = 88 (sum of F(1) through F(9))
;
; This program demonstrates:
;   - Fibonacci generation with simultaneous accumulation
;   - Dual-purpose loop (generate + sum)
;   - Multi-register coordination
;
; Algorithm:
;   prev = 0, curr = 1, sum = 0
;   for 9 iterations:
;       sum += curr
;       next = prev + curr
;       prev = curr
;       curr = next

    MOV R0, 0       ; prev = 0
    MOV R1, 1       ; curr = 1
    MOV R2, 0       ; sum = 0
    MOV R3, 9       ; iterations
    MOV R4, 0       ; counter

loop:
    CMP R4, R3      ; counter == 9?
    JZ done

    ADD R2, R2, R1  ; sum += curr
    ADD R5, R0, R1  ; next = prev + curr
    MOV R0, R1      ; prev = curr
    MOV R1, R5      ; curr = next
    INC R4          ; counter++
    JMP loop

done:
    HALT            ; done - R2 = 88
