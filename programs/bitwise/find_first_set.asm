; find_first_set.asm - Find position of lowest set bit (1-indexed)
; Expected result: R1 = 3 (lowest set bit in 0b11010100 = 212 is bit 2, 1-indexed = 3)
;
; This program demonstrates:
;   - Bit scanning from LSB
;   - Shift and test loop
;   - Position tracking
;
; Algorithm:
;   value = 212 (0b11010100, lowest set bit at position 2)
;   pos = 0
;   while (value & 1) == 0 and value != 0:
;       value >>= 1
;       pos++
;   result = pos + 1  (1-indexed)

    MOV R0, 212     ; value = 0b11010100
    MOV R1, 0       ; position counter
    MOV R2, 1       ; mask for lowest bit
    MOV R3, 0       ; zero for comparison

loop:
    CMP R0, R3      ; is value == 0?
    JZ done         ; no bits set at all

    AND R4, R0, R2  ; test lowest bit
    CMP R4, R3      ; is it zero?
    JNZ found       ; found the first set bit

    SHR R0, R0, 1   ; value >>= 1
    INC R1          ; pos++
    JMP loop

found:
    INC R1          ; convert to 1-indexed

done:
    HALT            ; done - R1 = 3 (bit position 2, 1-indexed)
