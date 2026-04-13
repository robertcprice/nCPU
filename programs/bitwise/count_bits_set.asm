; count_bits_set.asm - Count the number of 1-bits in a value (popcount)
; Expected result: R1 = 5 (number of 1-bits in 0b11010110 = 214)
;
; This program demonstrates:
;   - Bit testing via AND with mask
;   - Right shift to walk through bits
;   - Conditional increment pattern
;
; Algorithm:
;   value = 214 (0b11010110, has 5 bits set)
;   count = 0
;   while value != 0:
;       if value & 1: count++
;       value >>= 1

    MOV R0, 214     ; value = 0b11010110 (5 bits set)
    MOV R1, 0       ; count = 0
    MOV R2, 1       ; mask for lowest bit
    MOV R3, 0       ; zero for comparison

loop:
    CMP R0, R3      ; is value == 0?
    JZ done         ; if so, we are done

    AND R4, R0, R2  ; test lowest bit
    CMP R4, R3      ; is lowest bit 0?
    JZ skip_inc     ; if bit is 0, skip increment

    INC R1          ; count++

skip_inc:
    SHR R0, R0, 1   ; value >>= 1
    JMP loop

done:
    HALT            ; done - R1 = 5
