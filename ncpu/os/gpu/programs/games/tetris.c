/*
 * tetris.c -- Tetris for freestanding ARM64 on Metal GPU compute shader.
 *
 * 10-wide x 20-tall board, 7 standard tetrominoes (I, O, T, S, Z, J, L),
 * rotation, line clearing, scoring, levels, game-over detection.
 *
 * Controls: w=rotate  a=left  d=right  s=soft-drop  q=quit
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *       -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *       -I demos -e _start demos/arm64_start.S demos/games/tetris.c \
 *       -o /tmp/tetris.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONSTANTS                                                                */
/* ======================================================================== */

#define BOARD_W   10
#define BOARD_H   20
#define NUM_TYPES 7
#define NUM_ROTS  4
#define CELLS     4      /* each tetromino has 4 cells */

/* Scoring: lines cleared -> points (NES-style, multiplied by level+1) */
static const int LINE_SCORES[5] = { 0, 40, 100, 300, 1200 };

/* Lines per level */
#define LINES_PER_LEVEL 10

/* Initial drop interval in ms; decreases with level */
#define BASE_DROP_MS 800
#define MIN_DROP_MS  100
#define DROP_ACCEL   50   /* ms reduction per level */

/* ======================================================================== */
/* TETROMINO DATA                                                           */
/*                                                                          */
/* Each piece is stored as 4 (row, col) offsets for each of 4 rotations.    */
/* Origin is at (0,0). Rotations are precomputed -- no matrix math needed.  */
/* ======================================================================== */

/* Piece indices: 0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L */

/* Row offsets: pieces[type][rotation][cell] */
static const int PIECE_R[NUM_TYPES][NUM_ROTS][CELLS] = {
    /* I */
    { {0,0,0,0}, {0,1,2,3}, {0,0,0,0}, {0,1,2,3} },
    /* O */
    { {0,0,1,1}, {0,0,1,1}, {0,0,1,1}, {0,0,1,1} },
    /* T */
    { {0,0,0,1}, {0,1,1,2}, {1,1,1,0}, {0,1,1,2} },
    /* S */
    { {0,0,1,1}, {0,1,1,2}, {0,0,1,1}, {0,1,1,2} },
    /* Z */
    { {0,0,1,1}, {0,1,1,2}, {0,0,1,1}, {0,1,1,2} },
    /* J */
    { {0,0,0,1}, {0,0,1,2}, {0,1,1,1}, {0,1,2,2} },
    /* L */
    { {0,0,0,1}, {0,1,2,2}, {0,1,1,1}, {0,0,1,2} },
};

static const int PIECE_C[NUM_TYPES][NUM_ROTS][CELLS] = {
    /* I */
    { {0,1,2,3}, {0,0,0,0}, {0,1,2,3}, {1,1,1,1} },
    /* O */
    { {0,1,0,1}, {0,1,0,1}, {0,1,0,1}, {0,1,0,1} },
    /* T */
    { {0,1,2,1}, {1,0,1,1}, {0,1,2,1}, {0,0,1,0} },
    /* S */
    { {1,2,0,1}, {0,0,1,1}, {1,2,0,1}, {0,0,1,1} },
    /* Z */
    { {0,1,1,2}, {1,0,1,0}, {0,1,1,2}, {1,0,1,0} },
    /* J */
    { {0,1,2,0}, {0,1,0,0}, {0,0,1,2}, {1,1,1,0} },
    /* L */
    { {0,1,2,2}, {0,0,0,1}, {0,0,1,2}, {0,1,1,1} },
};

/* Display characters for each piece type (1-indexed on the board) */
static const char PIECE_CHAR[NUM_TYPES] = { '#', '@', '%', '&', '*', '+', '=' };

/* ======================================================================== */
/* GAME STATE                                                               */
/* ======================================================================== */

/* The board stores 0 for empty, or (type+1) for a locked cell */
static int board[BOARD_H][BOARD_W];

/* Current piece state */
static int cur_type;
static int cur_rot;
static int cur_row;    /* top-left origin row on board */
static int cur_col;    /* top-left origin col on board */

/* Next piece */
static int next_type;

/* Score, level, lines */
static int score;
static int level;
static int total_lines;

/* Game state */
static int game_over;

/* Timing */
static long last_drop_time;

/* ======================================================================== */
/* BOARD OPERATIONS                                                         */
/* ======================================================================== */

static void board_clear(void) {
    for (int r = 0; r < BOARD_H; r++)
        for (int c = 0; c < BOARD_W; c++)
            board[r][c] = 0;
}

/*
 * Check if a piece at (row, col) with given rotation fits on the board.
 * Returns 1 if the position is valid, 0 if collision.
 */
static int piece_fits(int type, int rot, int row, int col) {
    for (int i = 0; i < CELLS; i++) {
        int r = row + PIECE_R[type][rot][i];
        int c = col + PIECE_C[type][rot][i];
        if (r < 0 || r >= BOARD_H) return 0;
        if (c < 0 || c >= BOARD_W) return 0;
        if (board[r][c] != 0) return 0;
    }
    return 1;
}

/*
 * Lock the current piece into the board.
 */
static void lock_piece(void) {
    for (int i = 0; i < CELLS; i++) {
        int r = cur_row + PIECE_R[cur_type][cur_rot][i];
        int c = cur_col + PIECE_C[cur_type][cur_rot][i];
        if (r >= 0 && r < BOARD_H && c >= 0 && c < BOARD_W) {
            board[r][c] = cur_type + 1;
        }
    }
}

/*
 * Clear completed lines. Returns the number of lines cleared.
 */
static int clear_lines(void) {
    int cleared = 0;

    for (int r = BOARD_H - 1; r >= 0; r--) {
        int full = 1;
        for (int c = 0; c < BOARD_W; c++) {
            if (board[r][c] == 0) {
                full = 0;
                break;
            }
        }

        if (full) {
            cleared++;
            /* Shift everything above down by one row */
            for (int rr = r; rr > 0; rr--) {
                for (int c = 0; c < BOARD_W; c++) {
                    board[rr][c] = board[rr - 1][c];
                }
            }
            /* Clear top row */
            for (int c = 0; c < BOARD_W; c++) {
                board[0][c] = 0;
            }
            /* Re-check this row since a new row shifted into it */
            r++;
        }
    }

    return cleared;
}

/* ======================================================================== */
/* PIECE SPAWNING                                                           */
/* ======================================================================== */

static int random_piece(void) {
    return rand() % NUM_TYPES;
}

/*
 * Spawn the next piece at the top of the board.
 * Returns 0 on success, 1 if game over (piece doesn't fit).
 */
static int spawn_piece(void) {
    cur_type = next_type;
    cur_rot  = 0;
    cur_row  = 0;
    cur_col  = (BOARD_W / 2) - 1;

    next_type = random_piece();

    if (!piece_fits(cur_type, cur_rot, cur_row, cur_col)) {
        return 1;  /* game over */
    }
    return 0;
}

/* ======================================================================== */
/* INPUT HANDLING                                                           */
/* ======================================================================== */

static void handle_input(void) {
    int ch = getchar();
    if (ch < 0) return;  /* no input available */

    switch (ch) {
        case 'a':  /* move left */
            if (piece_fits(cur_type, cur_rot, cur_row, cur_col - 1)) {
                cur_col--;
            }
            break;

        case 'd':  /* move right */
            if (piece_fits(cur_type, cur_rot, cur_row, cur_col + 1)) {
                cur_col++;
            }
            break;

        case 'w': {  /* rotate clockwise */
            int new_rot = (cur_rot + 1) % NUM_ROTS;
            if (piece_fits(cur_type, new_rot, cur_row, cur_col)) {
                cur_rot = new_rot;
            }
            /* Wall kick: try shifting left or right by 1 */
            else if (piece_fits(cur_type, new_rot, cur_row, cur_col - 1)) {
                cur_rot = new_rot;
                cur_col--;
            }
            else if (piece_fits(cur_type, new_rot, cur_row, cur_col + 1)) {
                cur_rot = new_rot;
                cur_col++;
            }
            /* Wall kick: try shifting by 2 (for I-piece) */
            else if (piece_fits(cur_type, new_rot, cur_row, cur_col - 2)) {
                cur_rot = new_rot;
                cur_col -= 2;
            }
            else if (piece_fits(cur_type, new_rot, cur_row, cur_col + 2)) {
                cur_rot = new_rot;
                cur_col += 2;
            }
            break;
        }

        case 's':  /* soft drop: move down immediately */
            if (piece_fits(cur_type, cur_rot, cur_row + 1, cur_col)) {
                cur_row++;
                score += 1;  /* bonus point for soft drop */
            }
            break;

        case ' ':  /* hard drop: fall all the way */
            while (piece_fits(cur_type, cur_rot, cur_row + 1, cur_col)) {
                cur_row++;
                score += 2;  /* bonus for hard drop */
            }
            break;

        case 'q':  /* quit */
            game_over = 1;
            break;
    }
}

/* ======================================================================== */
/* GRAVITY / DROP                                                           */
/* ======================================================================== */

/*
 * Get the current drop interval based on level.
 */
static int get_drop_interval(void) {
    int interval = BASE_DROP_MS - (level * DROP_ACCEL);
    if (interval < MIN_DROP_MS) interval = MIN_DROP_MS;
    return interval;
}

/*
 * Process a gravity tick: move the piece down or lock it.
 * Returns 1 if a new piece was spawned (and possibly game over).
 */
static int gravity_tick(void) {
    if (piece_fits(cur_type, cur_rot, cur_row + 1, cur_col)) {
        cur_row++;
        return 0;
    }

    /* Piece can't move down -- lock it */
    lock_piece();

    /* Clear completed lines */
    int cleared = clear_lines();
    if (cleared > 0) {
        total_lines += cleared;
        score += LINE_SCORES[cleared] * (level + 1);
        level = total_lines / LINES_PER_LEVEL;
    }

    /* Spawn next piece */
    if (spawn_piece()) {
        game_over = 1;
    }

    return 1;
}

/* ======================================================================== */
/* GHOST PIECE (drop preview)                                               */
/* ======================================================================== */

static int ghost_row(void) {
    int gr = cur_row;
    while (piece_fits(cur_type, cur_rot, gr + 1, cur_col)) {
        gr++;
    }
    return gr;
}

/* ======================================================================== */
/* RENDERING                                                                */
/* ======================================================================== */

/*
 * Build a display buffer combining the locked board, the ghost piece,
 * and the current piece. Then render with borders and HUD.
 *
 * Display uses cursor positioning to avoid full-screen flicker.
 */

/* Display cell values:
 *   0       = empty
 *   1..7    = locked piece type
 *   8       = ghost piece
 *   9..15   = current (active) piece type
 */
static int display[BOARD_H][BOARD_W];

static void build_display(void) {
    /* Copy locked board */
    for (int r = 0; r < BOARD_H; r++)
        for (int c = 0; c < BOARD_W; c++)
            display[r][c] = board[r][c];

    /* Draw ghost piece */
    int gr = ghost_row();
    if (gr != cur_row) {
        for (int i = 0; i < CELLS; i++) {
            int r = gr + PIECE_R[cur_type][cur_rot][i];
            int c = cur_col + PIECE_C[cur_type][cur_rot][i];
            if (r >= 0 && r < BOARD_H && c >= 0 && c < BOARD_W) {
                if (display[r][c] == 0) {
                    display[r][c] = 8;  /* ghost marker */
                }
            }
        }
    }

    /* Draw current piece (overwrites ghost if overlapping) */
    for (int i = 0; i < CELLS; i++) {
        int r = cur_row + PIECE_R[cur_type][cur_rot][i];
        int c = cur_col + PIECE_C[cur_type][cur_rot][i];
        if (r >= 0 && r < BOARD_H && c >= 0 && c < BOARD_W) {
            display[r][c] = 9 + cur_type;
        }
    }
}

static void render(void) {
    /* Home cursor (avoid full clear to reduce flicker) */
    printf("\033[H");

    /* Title and score bar */
    printf("  nCPU TETRIS  |  Score: %-8d  Level: %-3d  Lines: %-4d\n",
           score, level, total_lines);
    printf("\n");

    /* Next piece preview */
    printf("  Next: ");
    /* Render the next piece in a small 2x4 box */
    for (int pr = 0; pr < 2; pr++) {
        if (pr > 0) printf("        ");
        for (int pc = 0; pc < 4; pc++) {
            int found = 0;
            for (int i = 0; i < CELLS; i++) {
                if (PIECE_R[next_type][0][i] == pr &&
                    PIECE_C[next_type][0][i] == pc) {
                    found = 1;
                    break;
                }
            }
            if (found)
                printf("%c", PIECE_CHAR[next_type]);
            else
                printf(" ");
        }
        printf("\n");
    }
    printf("\n");

    /* Top border */
    printf("  +");
    for (int c = 0; c < BOARD_W; c++) printf("--");
    printf("-+\n");

    /* Board rows */
    build_display();
    for (int r = 0; r < BOARD_H; r++) {
        printf("  |");
        for (int c = 0; c < BOARD_W; c++) {
            int v = display[r][c];
            if (v == 0) {
                printf(" .");
            } else if (v >= 1 && v <= 7) {
                /* Locked piece */
                printf(" %c", PIECE_CHAR[v - 1]);
            } else if (v == 8) {
                /* Ghost */
                printf(" :");
            } else {
                /* Active piece (9..15 -> type 0..6) */
                printf(" %c", PIECE_CHAR[v - 9]);
            }
        }
        printf(" |\n");
    }

    /* Bottom border */
    printf("  +");
    for (int c = 0; c < BOARD_W; c++) printf("--");
    printf("-+\n");

    /* Controls */
    printf("\n");
    printf("  [W] Rotate  [A] Left  [D] Right  [S] Drop  [SPACE] Hard Drop  [Q] Quit\n");
}

/* ======================================================================== */
/* GAME OVER SCREEN                                                         */
/* ======================================================================== */

static void render_game_over(void) {
    printf("\033[2J\033[H");
    printf("\n\n");
    printf("  =============================================\n");
    printf("                  GAME OVER\n");
    printf("  =============================================\n");
    printf("\n");
    printf("      Final Score : %d\n", score);
    printf("      Level       : %d\n", level);
    printf("      Lines       : %d\n", total_lines);
    printf("\n");
    printf("  =============================================\n");
    printf("\n");
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    /* Seed RNG from clock */
    srand((unsigned long)clock_ms());

    /* Initialize */
    board_clear();
    score = 0;
    level = 0;
    total_lines = 0;
    game_over = 0;

    /* Generate first "next" piece, then spawn */
    next_type = random_piece();
    if (spawn_piece()) {
        game_over = 1;
    }

    /* Clear screen once at start */
    printf("\033[2J");

    last_drop_time = clock_ms();

    /* ---- Main game loop ---- */
    while (!game_over) {
        /* Process input */
        handle_input();

        /* Check gravity timer */
        long now = clock_ms();
        int interval = get_drop_interval();
        if (now - last_drop_time >= interval) {
            gravity_tick();
            last_drop_time = now;
        }

        /* Render */
        render();

        /* Frame pacing */
        sleep_ms(50);
    }

    /* Show final state then game over screen */
    render_game_over();

    return 0;
}
