; memory_copy.asm - Simulated memory copy (register to register)
; Expected result: R4=10, R5=20, R6=30, R7=40 (copy of R0-R3)
;
; This program demonstrates:
;   - Register-to-register MOV operations
;   - Simulated memcpy pattern
;   - Sequential data transfer
;
; Algorithm:
;   Copy "source" registers R0-R3 to "destination" registers R4-R7

    ; Source data
    MOV R0, 10      ; src[0]
    MOV R1, 20      ; src[1]
    MOV R2, 30      ; src[2]
    MOV R3, 40      ; src[3]

    ; Copy source to destination
    MOV R4, R0      ; dst[0] = src[0]
    MOV R5, R1      ; dst[1] = src[1]
    MOV R6, R2      ; dst[2] = src[2]
    MOV R7, R3      ; dst[3] = src[3]

    HALT            ; done - R4=10, R5=20, R6=30, R7=40
