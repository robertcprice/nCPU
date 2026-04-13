; parity.asm - Compute parity (XOR of all bits) of a byte
; Expected result: R1 = 0 (even parity of 0b10101010 = 170, which has 4 set bits)
;
; This program demonstrates:
;   - XOR folding to compute parity
;   - Successive halving of bit width
;   - Parity via XOR reduction
;
; Algorithm:
;   Fold byte in half repeatedly with XOR:
;   x ^= (x >> 4)
;   x ^= (x >> 2)
;   x ^= (x >> 1)
;   parity = x & 1

    MOV R0, 170     ; value = 0b10101010 (even parity: 4 bits set)
    MOV R2, 1       ; mask for lowest bit

    SHR R1, R0, 4   ; R1 = x >> 4
    XOR R0, R0, R1  ; x ^= (x >> 4)

    SHR R1, R0, 2   ; R1 = x >> 2
    XOR R0, R0, R1  ; x ^= (x >> 2)

    SHR R1, R0, 1   ; R1 = x >> 1
    XOR R0, R0, R1  ; x ^= (x >> 1)

    AND R1, R0, R2  ; parity = x & 1

    HALT            ; done - R1 = 0 (even parity)
