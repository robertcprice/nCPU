; count_leading_zeros.asm - Count leading zeros in a byte (8-bit)
; Expected result: R1 = 2 (leading zeros of 0b00110101 = 53)
;
; This program demonstrates:
;   - Bit scanning from MSB
;   - Left shift to check high bit
;   - Fixed-width (8-bit) analysis
;
; Algorithm:
;   value = 53 (0b00110101)
;   clz = 0
;   mask = 128 (0b10000000)
;   while (value & mask) == 0 and clz < 8:
;       clz++
;       mask >>= 1

    MOV R0, 53      ; value = 0b00110101
    MOV R1, 0       ; clz count = 0
    MOV R2, 128     ; mask = 0b10000000 (MSB of byte)
    MOV R3, 8       ; max bits
    MOV R4, 0       ; zero constant

loop:
    CMP R1, R3      ; clz == 8? (all zeros)
    JZ done

    AND R5, R0, R2  ; test bit under mask
    CMP R5, R4      ; is it zero?
    JNZ done        ; found a 1-bit, stop

    INC R1          ; clz++
    SHR R2, R2, 1   ; mask >>= 1
    JMP loop

done:
    HALT            ; done - R1 = 2 (two leading zeros)
