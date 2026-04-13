; Countdown Demo - Demonstrates DEC and JNS operations
; Counts down from 10 to 0, stores sum in R1
; Expected result: R0 = -1 (went negative to exit), R1 = 55 (sum of 10+9+...+1+0)
;
; This program showcases:
;   - DEC: Decrement register by 1
;   - JNS: Jump if sign flag not set (non-negative)
;
; Algorithm:
;   R0 = 10 (counter, counts down)
;   R1 = 0  (sum accumulator)
;   while R0 >= 0:
;       R1 += R0
;       R0--
;   HALT (R0 = -1, R1 = 55)

    MOV R0, 10      ; counter = 10
    MOV R1, 0       ; sum = 0

loop:
    ADD R1, R1, R0  ; sum += counter
    DEC R0          ; counter-- (sets SF if result < 0)
    JNS loop        ; continue if counter >= 0

    HALT            ; R0 = -1, R1 = 55
