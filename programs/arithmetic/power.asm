; power.asm - Compute a^b (power function) via repeated multiplication
; Expected result: R2 = 243 (3^5 = 243)
;
; This program demonstrates:
;   - Exponentiation by repeated multiplication
;   - Counter-based loop with MUL
;   - Building up a result iteratively
;
; Algorithm:
;   base = 3, exp = 5
;   result = 1
;   for i = 0 to exp-1:
;       result *= base

    MOV R0, 3       ; base = 3
    MOV R1, 5       ; exponent = 5
    MOV R2, 1       ; result = 1
    MOV R3, 0       ; counter = 0

loop:
    CMP R3, R1      ; counter == exponent?
    JZ done

    MUL R2, R2, R0  ; result *= base
    INC R3          ; counter++
    JMP loop

done:
    HALT            ; done - R2 = 243 (3^5)
