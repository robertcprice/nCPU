; matrix_element.asm - Compute 2D matrix element address (row-major)
; Expected result: R4 = 7 (element at row=1, col=2 in a 3-column matrix)
; Matrix layout (3x3): [1,2,3 / 4,5,6 / 7,8,9], element[1][2] = 6
; Address = base + (row * num_cols + col) ... but we simulate with register math
; Expected result: R4 = 5 (offset = 1*3 + 2 = 5, 0-indexed element index)
;
; This program demonstrates:
;   - Row-major matrix indexing
;   - MUL and ADD for address calculation
;   - 2D to 1D index conversion
;
; Algorithm:
;   offset = row * num_cols + col

    MOV R0, 1       ; row = 1
    MOV R1, 2       ; col = 2
    MOV R2, 3       ; num_cols = 3

    MUL R3, R0, R2  ; row * num_cols = 3
    ADD R4, R3, R1  ; offset = 3 + 2 = 5

    ; Verify: build the matrix value at that position
    ; Matrix: [1,2,3,4,5,6,7,8,9], element[5] = 6
    ; Store base offset + 1 (since values start at 1)
    MOV R5, 1
    ADD R6, R4, R5  ; value = offset + 1 = 6

    HALT            ; done - R4 = 5 (offset), R6 = 6 (value)
