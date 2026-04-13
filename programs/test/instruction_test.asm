; instruction_test.asm - Exercise every instruction in the ISA
; Expected result: Multiple registers set, demonstrates all instruction types
;
; This program demonstrates:
;   - Complete ISA coverage
;   - Every opcode exercised at least once
;   - Comprehensive instruction test
;
; Instructions exercised:
;   MOV (imm), MOV (reg), ADD, SUB, MUL, DIV
;   AND, OR, XOR, SHL, SHR, INC, DEC
;   CMP, JMP, JZ, JNZ, JS, JNS, NOP, HALT

    ; MOV immediate
    MOV R0, 100     ; MOV Rd, imm
    MOV R1, 7       ; MOV Rd, imm

    ; MOV register
    MOV R2, R0      ; MOV Rd, Rs

    ; Arithmetic
    ADD R3, R0, R1  ; ADD: 100 + 7 = 107
    SUB R4, R0, R1  ; SUB: 100 - 7 = 93
    MUL R5, R1, R1  ; MUL: 7 * 7 = 49
    DIV R6, R0, R1  ; DIV: 100 / 7 = 14

    ; Logic
    MOV R0, 0xFF
    MOV R1, 0x0F
    AND R2, R0, R1  ; AND: 0xFF & 0x0F = 0x0F
    OR  R3, R0, R1  ; OR:  0xFF | 0x0F = 0xFF
    XOR R4, R0, R1  ; XOR: 0xFF ^ 0x0F = 0xF0

    ; Shifts
    MOV R0, 1
    SHL R5, R0, 4   ; SHL: 1 << 4 = 16
    MOV R0, 64
    SHR R6, R0, 3   ; SHR: 64 >> 3 = 8

    ; INC / DEC
    MOV R0, 10
    INC R0          ; INC: 10 + 1 = 11
    DEC R0          ; DEC: 11 - 1 = 10

    ; NOP
    NOP             ; No operation

    ; CMP and conditional jumps
    MOV R0, 5
    MOV R1, 5
    CMP R0, R1      ; compare equal
    JZ equal        ; JZ: should jump (zero flag set)
    JMP skip_eq     ; should not reach here

equal:
    MOV R2, 1       ; flag: equal detected

skip_eq:
    MOV R0, 3
    MOV R1, 5
    CMP R0, R1      ; compare less
    JNZ not_zero    ; JNZ: should jump (not zero)

not_zero:
    JS is_neg       ; JS: should jump (3-5 is negative)
    JMP skip_neg

is_neg:
    MOV R3, 1       ; flag: negative detected

skip_neg:
    MOV R0, 7
    MOV R1, 3
    CMP R0, R1      ; compare greater
    JNS not_neg     ; JNS: should jump (7-3 is positive)

not_neg:
    MOV R4, 1       ; flag: positive detected

    ; Unconditional jump
    JMP done        ; JMP: unconditional

    MOV R7, 99      ; should never execute

done:
    HALT            ; HALT: stop execution
