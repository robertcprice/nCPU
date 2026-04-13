; is_even_odd.asm - Check if a number is even or odd using AND with 1
; Expected result: R1 = 0 (even), R2 = 1 (odd check of 7)
;
; This program demonstrates:
;   - Bitwise AND to test lowest bit
;   - Even/odd determination
;   - Two separate checks
;
; Algorithm:
;   Test 42: AND with 1 -> 0 (even)
;   Test 7:  AND with 1 -> 1 (odd)

    MOV R0, 42      ; test number (even)
    MOV R3, 1       ; mask for lowest bit

    AND R1, R0, R3  ; R1 = 42 & 1 = 0 (even)

    MOV R0, 7       ; test number (odd)
    AND R2, R0, R3  ; R2 = 7 & 1 = 1 (odd)

    HALT            ; done - R1 = 0 (even), R2 = 1 (odd)
