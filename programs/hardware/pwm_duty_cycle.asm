; pwm_duty_cycle.asm - Generate PWM pattern for 75% duty cycle
; Expected result: R2 = bit pattern representing PWM output over 8 cycles
;
; This program demonstrates:
;   - PWM (Pulse Width Modulation) concept
;   - Comparison-based threshold for on/off
;   - Shift and OR to build output pattern
;
; Algorithm:
;   period = 4, duty = 3 (75%)
;   For 8 cycles, output 1 if counter < duty, else 0
;   Build 8-bit pattern: 11101110 = 0xEE = 238

    MOV R0, 4       ; period
    MOV R1, 3       ; duty (on-time within period)
    MOV R2, 0       ; output pattern
    MOV R3, 0       ; cycle counter (0..7)
    MOV R4, 8       ; total cycles
    MOV R5, 0       ; phase counter (0..period-1)

cycle:
    CMP R3, R4      ; 8 cycles done?
    JZ done

    SHL R2, R2, 1   ; make room for next bit

    ; Is phase < duty? (output high)
    CMP R5, R1      ; phase < duty?
    JS output_high  ; if phase < duty, output 1

    ; Output low (0) - already 0 from shift
    JMP next_cycle

output_high:
    MOV R6, 1
    OR  R2, R2, R6  ; set lowest bit

next_cycle:
    INC R5          ; phase++
    INC R3          ; cycle++

    ; Reset phase if >= period
    CMP R5, R0      ; phase >= period?
    JS no_reset     ; if phase < period, skip
    MOV R5, 0       ; reset phase

no_reset:
    JMP cycle

done:
    HALT            ; done - R2 = 238 (0b11101110)
