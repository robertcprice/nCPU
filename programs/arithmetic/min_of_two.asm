; min_of_two.asm - Find the minimum of two numbers
; Expected result: R2 = 17 (min of 17 and 42)
;
; This program demonstrates:
;   - Comparison and conditional jump
;   - Inverted branch logic from max
;
; Algorithm:
;   a = 17, b = 42
;   if a <= b: min = a
;   else: min = b

    MOV R0, 17      ; a = 17
    MOV R1, 42      ; b = 42

    CMP R0, R1      ; compare a - b
    JS a_wins       ; if a < b (negative), a is smaller
    JZ a_wins       ; if a == b, either works

    ; a > b
    MOV R2, R1      ; min = b
    JMP done

a_wins:
    MOV R2, R0      ; min = a

done:
    HALT            ; done - R2 = 17
