; average.asm - Compute average of 4 numbers via sum and shift
; Expected result: R5 = 25 (average of 10, 20, 30, 40)
;
; This program demonstrates:
;   - Summing multiple values
;   - Integer division by power of 2 via SHR
;   - Averaging pattern
;
; Algorithm:
;   sum = 10 + 20 + 30 + 40 = 100
;   average = sum / 4 = sum >> 2 = 25

    MOV R0, 10      ; a = 10
    MOV R1, 20      ; b = 20
    MOV R2, 30      ; c = 30
    MOV R3, 40      ; d = 40

    ADD R4, R0, R1  ; sum = 10 + 20 = 30
    ADD R4, R4, R2  ; sum = 30 + 30 = 60
    ADD R4, R4, R3  ; sum = 60 + 40 = 100

    SHR R5, R4, 2   ; average = 100 / 4 = 25

    HALT            ; done - R5 = 25
