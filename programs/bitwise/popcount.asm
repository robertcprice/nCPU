; popcount.asm - Population count (Hamming weight) via Kernighan's method
; Expected result: R1 = 4 (popcount of 170 = 0b10101010 = 4 bits set)
;
; This program demonstrates:
;   - Kernighan's bit counting: n &= (n-1) clears lowest set bit
;   - More efficient than shifting through all bits
;   - Loop count equals number of set bits
;
; Algorithm:
;   n = 170 (0b10101010)
;   count = 0
;   while n != 0:
;       n = n & (n - 1)   ; clear lowest set bit
;       count++

    MOV R0, 170     ; n = 0b10101010 (4 bits set)
    MOV R1, 0       ; count = 0
    MOV R2, 1       ; constant 1
    MOV R3, 0       ; zero for comparison

loop:
    CMP R0, R3      ; n == 0?
    JZ done

    SUB R4, R0, R2  ; n - 1
    AND R0, R0, R4  ; n = n & (n - 1), clears lowest set bit
    INC R1          ; count++
    JMP loop

done:
    HALT            ; done - R1 = 4
