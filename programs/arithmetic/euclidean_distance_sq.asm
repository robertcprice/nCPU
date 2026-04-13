; euclidean_distance_sq.asm - Compute squared Euclidean distance
; Expected result: R4 = 25 (distance^2 between (1,2) and (4,6))
; Verification: (4-1)^2 + (6-2)^2 = 9 + 16 = 25
;
; This program demonstrates:
;   - Coordinate difference computation
;   - Squaring via MUL
;   - Distance metric calculation
;
; Algorithm:
;   dx = x2 - x1, dy = y2 - y1
;   dist_sq = dx*dx + dy*dy

    MOV R0, 1       ; x1 = 1
    MOV R1, 2       ; y1 = 2
    MOV R2, 4       ; x2 = 4
    MOV R3, 6       ; y2 = 6

    SUB R5, R2, R0  ; dx = x2 - x1 = 3
    SUB R6, R3, R1  ; dy = y2 - y1 = 4
    MUL R5, R5, R5  ; dx^2 = 9
    MUL R6, R6, R6  ; dy^2 = 16
    ADD R4, R5, R6  ; dist_sq = 9 + 16 = 25

    HALT            ; done - R4 = 25
