; count_down_by_three.asm - Count down from 30 by 3, accumulating count
; Expected result: R1 = 10 (30/3 = 10 steps), R0 = 0
;
; This program demonstrates:
;   - Decrement by arbitrary step
;   - SUB in countdown loop
;   - Division via repeated subtraction counting
;
; Algorithm:
;   value = 30, step = 3, count = 0
;   while value > 0:
;       value -= step
;       count++

    MOV R0, 30      ; value = 30
    MOV R1, 0       ; count = 0
    MOV R2, 3       ; step = 3
    MOV R3, 0       ; zero for comparison

loop:
    CMP R0, R3      ; value == 0?
    JZ done
    JS done         ; value < 0 (shouldn't happen with exact division)

    SUB R0, R0, R2  ; value -= 3
    INC R1          ; count++
    JMP loop

done:
    HALT            ; done - R0 = 0, R1 = 10
