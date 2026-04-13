; bitfield_extract.asm - Extract a bitfield from a value
; Expected result: R3 = 5 (bits [6:4] of 0b01010011 = 83, which is 0b101 = 5)
;
; This program demonstrates:
;   - Bitfield extraction: (value >> offset) & mask
;   - Right shift to align field
;   - AND with mask to isolate field
;
; Algorithm:
;   value = 83 (0b01010011)
;   Extract bits 4-6 (3-bit field at offset 4)
;   result = (value >> 4) & 0b111

    MOV R0, 83      ; value = 0b01010011
    MOV R1, 7       ; mask = 0b111 (3-bit)

    SHR R2, R0, 4   ; shift right by 4 to align field
    AND R3, R2, R1  ; mask off upper bits

    HALT            ; done - R3 = 5 (0b101)
