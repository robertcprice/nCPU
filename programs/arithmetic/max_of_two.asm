; max_of_two.asm - Find the maximum of two numbers
; Expected result: R2 = 42 (max of 17 and 42)
;
; This program demonstrates:
;   - Comparison and conditional jump
;   - Branch-based selection pattern
;
; Algorithm:
;   a = 17, b = 42
;   if a >= b: max = a
;   else: max = b

    MOV R0, 17      ; a = 17
    MOV R1, 42      ; b = 42

    CMP R0, R1      ; compare a - b
    JS b_wins       ; if a < b (negative), b is larger

    ; a >= b
    MOV R2, R0      ; max = a
    JMP done

b_wins:
    MOV R2, R1      ; max = b

done:
    HALT            ; done - R2 = 42
