; xor_swap.asm - Swap two values using XOR (no temporary register)
; Expected result: R0 = 99, R1 = 42 (swapped from R0=42, R1=99)
;
; This program demonstrates:
;   - Classic XOR swap algorithm
;   - XOR instruction usage
;   - In-place value exchange
;
; Algorithm:
;   a = 42, b = 99
;   a ^= b
;   b ^= a
;   a ^= b
;   now a = 99, b = 42

    MOV R0, 42      ; a = 42
    MOV R1, 99      ; b = 99

    XOR R0, R0, R1  ; a = a ^ b
    XOR R1, R0, R1  ; b = a ^ b (= original a)
    XOR R0, R0, R1  ; a = a ^ b (= original b)

    HALT            ; done - R0 = 99, R1 = 42
