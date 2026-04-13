; linear_search.asm - Search for a value in register "array"
; Expected result: R6 = 3 (found at index 3, 0-indexed), R7 = 1 (found flag)
;
; This program demonstrates:
;   - Sequential comparison pattern
;   - Early exit on match
;   - Index tracking
;
; Algorithm:
;   array = [10, 25, 33, 42, 50] in R0-R4
;   target = 42
;   Search sequentially, return index or -1

    MOV R0, 10      ; array[0]
    MOV R1, 25      ; array[1]
    MOV R2, 33      ; array[2]
    MOV R3, 42      ; array[3]
    MOV R4, 50      ; array[4]
    MOV R5, 42      ; target value
    MOV R6, -1      ; index = -1 (not found)
    MOV R7, 0       ; found flag = 0

    ; Check array[0]
    CMP R0, R5
    JZ found0
    ; Check array[1]
    CMP R1, R5
    JZ found1
    ; Check array[2]
    CMP R2, R5
    JZ found2
    ; Check array[3]
    CMP R3, R5
    JZ found3
    ; Check array[4]
    CMP R4, R5
    JZ found4
    JMP done        ; not found

found0:
    MOV R6, 0
    JMP found
found1:
    MOV R6, 1
    JMP found
found2:
    MOV R6, 2
    JMP found
found3:
    MOV R6, 3
    JMP found
found4:
    MOV R6, 4

found:
    MOV R7, 1       ; found flag = 1

done:
    HALT            ; done - R6 = 3 (index), R7 = 1 (found)
