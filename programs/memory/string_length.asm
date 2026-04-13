; string_length.asm - Simulated string length (count non-zero registers)
; Expected result: R7 = 5 (five non-zero "characters" before null terminator)
;
; This program demonstrates:
;   - Simulated string as sequence of register values
;   - Null terminator detection (value == 0)
;   - Sequential scanning
;
; Algorithm:
;   "string" = [72, 101, 108, 108, 111, 0] = "Hello\0" in registers
;   Count non-zero values until we hit 0

    MOV R0, 72      ; 'H'
    MOV R1, 101     ; 'e'
    MOV R2, 108     ; 'l'
    MOV R3, 108     ; 'l'
    MOV R4, 111     ; 'o'
    MOV R5, 0       ; null terminator
    MOV R6, 0       ; zero for comparison
    MOV R7, 0       ; length counter

    ; Check each "character"
    CMP R0, R6
    JZ done
    INC R7

    CMP R1, R6
    JZ done
    INC R7

    CMP R2, R6
    JZ done
    INC R7

    CMP R3, R6
    JZ done
    INC R7

    CMP R4, R6
    JZ done
    INC R7

    CMP R5, R6
    JZ done
    INC R7

done:
    HALT            ; done - R7 = 5 (length of "Hello")
