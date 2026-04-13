; power_of_two.asm — Computes 2^N using SHL in a loop
;
; Uses the neural shift network to compute powers of two.
; SHL R0, R0, 1 doubles the value each iteration.
;
; Computes 2^8 = 256
;
; Expected results:
;   R0 = 256 (2^8)
;   R1 = 8   (loop counter, should equal N at end)

    MOV R0, 1          ; Start with 2^0 = 1
    MOV R1, 0          ; Loop counter
    MOV R2, 8          ; Target exponent N=8
    MOV R3, 1          ; Increment constant

loop:
    SHL R0, R0, 1      ; R0 = R0 << 1 (double it)
    ADD R1, R1, R3     ; R1++ (count iterations)
    CMP R1, R2         ; Have we shifted N times?
    JNZ loop           ; If not, keep going

    HALT
; R0 = 2^8 = 256
