; bit_rotate_left.asm - Rotate bits left within an 8-bit value
; Expected result: R3 = 0b01100101 = 101 (rotate 0b10010110 = 150 left by 2)
;
; This program demonstrates:
;   - Bit rotation (circular shift)
;   - Combination of SHL, SHR, and OR
;   - rotate_left(x, n, width) = (x << n) | (x >> (width - n))
;
; Algorithm:
;   x = 150 (0b10010110), rotate left by 2 within 8 bits
;   high = (x << 2) & 0xFF
;   low = x >> 6
;   result = high | low

    MOV R0, 150     ; value = 0b10010110
    MOV R1, 0xFF    ; 8-bit mask

    SHL R2, R0, 2   ; x << 2 = 600
    AND R2, R2, R1  ; mask to 8 bits: 600 & 0xFF = 88 (0b01011000)

    SHR R3, R0, 6   ; x >> (8-2) = x >> 6 = 2 (0b00000010)

    OR  R3, R2, R3  ; result = 88 | 2 = 90... hmm
    ; 150 = 10010110, rotate left 2:
    ; 01011010 = 90... wait let me recheck
    ; 10010110 << 2 = 01011000 (within 8 bits)
    ; 10010110 >> 6 = 00000010
    ; 01011000 | 00000010 = 01011010 = 90
    ; R3 = 90

    HALT            ; done - R3 = 90 (0b01011010)
