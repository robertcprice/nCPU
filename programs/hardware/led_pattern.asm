; led_pattern.asm - Generate Knight Rider LED chase pattern (8-bit)
; Expected result: R0 = final LED pattern after 14 shifts (full sweep)
;
; This program demonstrates:
;   - Bouncing LED pattern generation
;   - Direction reversal at boundaries
;   - SHL/SHR alternation
;
; Algorithm:
;   led = 1 (start at bit 0)
;   Shift left 7 times (to bit 7), then right 7 times (back to bit 0)
;   Store each pattern in R0

    MOV R0, 1       ; LED = bit 0 on
    MOV R1, 7       ; shift count for each direction
    MOV R2, 0       ; counter
    MOV R3, 128     ; boundary check (bit 7)
    MOV R4, 1       ; boundary check (bit 0)

shift_left:
    CMP R2, R1      ; shifted 7 times?
    JZ start_right

    SHL R0, R0, 1   ; LED shifts left
    INC R2          ; counter++
    JMP shift_left

start_right:
    MOV R2, 0       ; reset counter

shift_right:
    CMP R2, R1      ; shifted 7 times?
    JZ done

    SHR R0, R0, 1   ; LED shifts right
    INC R2          ; counter++
    JMP shift_right

done:
    HALT            ; done - R0 = 1 (back to start position)
