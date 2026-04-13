; reverse_byte.asm - Reverse the bits of a byte (8-bit value)
; Expected result: R1 = 0b10110100 = 180 (reverse of 0b00101101 = 45)
;
; This program demonstrates:
;   - Bit extraction via AND and shift
;   - Bit insertion via shift and OR
;   - Fixed iteration count loop
;
; Algorithm:
;   input = 45 (0b00101101)
;   output = 0
;   for i in range(8):
;       output <<= 1
;       output |= (input & 1)
;       input >>= 1

    MOV R0, 45      ; input = 0b00101101
    MOV R1, 0       ; output = 0
    MOV R2, 8       ; 8 bits to reverse
    MOV R3, 0       ; counter
    MOV R4, 1       ; mask / increment constant

loop:
    CMP R3, R2      ; counter == 8?
    JZ done

    SHL R1, R1, 1   ; output <<= 1
    AND R5, R0, R4  ; extract lowest bit of input
    OR  R1, R1, R5  ; output |= bit
    SHR R0, R0, 1   ; input >>= 1
    INC R3          ; counter++
    JMP loop

done:
    HALT            ; done - R1 = 180 (0b10110100)
