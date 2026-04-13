; compare_and_swap.asm - Compare-and-swap (CAS) primitive
; Expected result: R0 = 99 (swap happened), R3 = 1 (success flag)
;
; This program demonstrates:
;   - CAS pattern: if (location == expected) then location = new_value
;   - Atomic-style operation building block
;   - Conditional write pattern
;
; Algorithm:
;   location = 42, expected = 42, new_value = 99
;   if location == expected:
;       location = new_value
;       success = 1
;   else:
;       success = 0

    MOV R0, 42      ; location (simulated memory cell)
    MOV R1, 42      ; expected value
    MOV R2, 99      ; new value
    MOV R3, 0       ; success flag = 0 (default: fail)

    CMP R0, R1      ; location == expected?
    JNZ cas_fail    ; if not equal, CAS fails

    ; CAS success: write new value
    MOV R0, R2      ; location = new_value
    MOV R3, 1       ; success = 1
    JMP done

cas_fail:
    NOP             ; CAS failed, location unchanged

done:
    HALT            ; done - R0 = 99 (swapped), R3 = 1 (success)
