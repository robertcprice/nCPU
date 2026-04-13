; toggle_bits.asm - Toggle specific bits using XOR mask
; Expected result: R2 = 0b10100101 = 165 (toggle bits 0,2,4,6 of 0xFF)
;
; This program demonstrates:
;   - XOR for bit toggling
;   - Selective bit manipulation with masks
;   - Common embedded systems pattern
;
; Algorithm:
;   value = 0xFF (all bits set)
;   mask = 0b01011010 = 0x5A (toggle bits 1,3,4,6)
;   result = value XOR mask

    MOV R0, 0xFF    ; value = 11111111
    MOV R1, 0x5A    ; mask  = 01011010

    XOR R2, R0, R1  ; result = 11111111 ^ 01011010 = 10100101 = 0xA5 = 165

    ; Toggle again to restore
    XOR R3, R2, R1  ; 10100101 ^ 01011010 = 11111111 = 0xFF = 255

    ; Toggle specific bit (bit 3)
    MOV R4, 8       ; mask for bit 3 only
    MOV R5, 0
    XOR R5, R5, R4  ; 0 ^ 8 = 8 (bit 3 set)
    XOR R5, R5, R4  ; 8 ^ 8 = 0 (bit 3 cleared)

    HALT            ; done - R2 = 165 (0xA5), R3 = 255 (restored)
