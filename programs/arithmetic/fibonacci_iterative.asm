; fibonacci_iterative.asm - Fibonacci with early termination when exceeding limit
; Expected result: R1 = 55 (largest Fibonacci number <= 60)
;
; This program demonstrates:
;   - Fibonacci with threshold check
;   - Early termination pattern
;   - Comparison-based loop exit
;
; Algorithm:
;   prev = 0, curr = 1, limit = 60
;   while next <= limit:
;       next = prev + curr
;       prev = curr
;       curr = next
;   result = prev (last value that fit under limit)

    MOV R0, 0       ; prev = 0
    MOV R1, 1       ; curr = 1
    MOV R2, 60      ; limit

loop:
    ADD R3, R0, R1  ; next = prev + curr
    CMP R3, R2      ; next > limit?
    JS still_ok     ; if next < limit, continue
    JZ still_ok     ; if next == limit, also ok

    ; next > limit, roll back
    JMP done

still_ok:
    MOV R0, R1      ; prev = curr
    MOV R1, R3      ; curr = next
    JMP loop

done:
    HALT            ; done - R1 = 55 (largest Fib <= 60)
