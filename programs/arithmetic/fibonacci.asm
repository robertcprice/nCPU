; fibonacci.asm - Calculate Nth Fibonacci number
; Expected result: R1 = 89 (F(11) after 10 iterations)
;
; This program demonstrates:
;   - Register swapping pattern
;   - Iterative algorithm
;   - Multiple register coordination
;
; Algorithm:
;   fib_prev = 0  (R0) = F(0)
;   fib_curr = 1  (R1) = F(1)
;   for i in range(10):
;       temp = fib_curr
;       fib_curr = fib_prev + fib_curr
;       fib_prev = temp
;   halt
;
; Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...
; After 10 iterations starting from F(0)=0, F(1)=1: F(11) = 89

    MOV R0, 0       ; fib(0) - previous value
    MOV R1, 1       ; fib(1) - current value
    MOV R2, 10      ; N iterations
    MOV R3, 0       ; counter
    MOV R4, 1       ; constant 1 for incrementing

loop:
    MOV R5, R1      ; temp = fib_curr
    ADD R1, R0, R1  ; fib_curr = fib_prev + fib_curr
    MOV R0, R5      ; fib_prev = temp
    ADD R3, R3, R4  ; counter++
    CMP R3, R2      ; compare counter to N
    JNZ loop        ; continue if counter != N

    HALT            ; done - R1 should equal 89
