; Count Up Demo - Demonstrates INC and JS operations
; Counts up from -5, demonstrates sign flag transitions
; Expected result: R0 = 5, R1 = 10 (total increments)
;
; This program showcases:
;   - INC: Increment register by 1
;   - JS: Jump if sign flag set (result is negative)
;
; Algorithm:
;   R0 = -5 (counter, starts negative)
;   R1 = 0 (increment count)
;   while R0 < 5:  (loop 10 times: -5,-4,-3,-2,-1,0,1,2,3,4)
;       R0++ (and count increments)

    MOV R0, -5      ; Start at -5 (negative)
    MOV R1, 0       ; increment count
    MOV R2, 5       ; limit

loop:
    INC R0          ; counter++ (sets flags)
    INC R1          ; count the increment
    CMP R0, R2      ; compare to limit
    JNZ loop        ; continue if not equal to 5

    HALT            ; R0 = 5, R1 = 10
