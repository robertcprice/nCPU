; timer_countdown.asm - Nested loop timer: outer * inner countdown
; Expected result: R2 = 20 (total inner loop iterations: 4 outer * 5 inner)
;
; This program demonstrates:
;   - Nested loop structure
;   - Loop counter reset pattern
;   - Multiplication by iteration counting
;
; Algorithm:
;   outer = 4, inner = 5
;   total = 0
;   for i = 0 to outer-1:
;       for j = 0 to inner-1:
;           total++

    MOV R0, 4       ; outer loop count
    MOV R1, 5       ; inner loop count
    MOV R2, 0       ; total counter
    MOV R3, 0       ; outer counter
    MOV R5, 0       ; zero constant

outer_loop:
    CMP R3, R0      ; outer done?
    JZ done

    MOV R4, 0       ; reset inner counter

inner_loop:
    CMP R4, R1      ; inner done?
    JZ inner_done

    INC R2          ; total++
    INC R4          ; inner counter++
    JMP inner_loop

inner_done:
    INC R3          ; outer counter++
    JMP outer_loop

done:
    HALT            ; done - R2 = 20
