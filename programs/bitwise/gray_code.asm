; gray_code.asm - Convert binary to Gray code
; Expected result: R2 = 13 (Gray code of 9: 9 XOR (9 >> 1) = 9 XOR 4 = 13)
;
; This program demonstrates:
;   - Gray code conversion: gray = n XOR (n >> 1)
;   - Single-line formula via shift and XOR
;   - Classic encoding technique
;
; Algorithm:
;   Binary to Gray: gray = n ^ (n >> 1)
;   Verify: 9 = 0b1001, Gray = 0b1101 = 13

    MOV R0, 9       ; binary value = 9 (0b1001)

    SHR R1, R0, 1   ; R1 = n >> 1 = 4 (0b0100)
    XOR R2, R0, R1  ; gray = n ^ (n >> 1) = 13 (0b1101)

    ; Also compute for 0..7 to exercise more
    MOV R0, 0
    SHR R1, R0, 1
    XOR R3, R0, R1  ; gray(0) = 0

    MOV R0, 5       ; binary 5 = 0b101
    SHR R1, R0, 1   ; 2 = 0b010
    XOR R4, R0, R1  ; gray(5) = 7 (0b111)

    HALT            ; done - R2 = 13, R3 = 0, R4 = 7
