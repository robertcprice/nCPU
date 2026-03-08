/*
 * snake.c -- Snake game for freestanding ARM64 on Metal GPU.
 *
 * 40x20 grid. WASD controls. Food spawning, growing tail, score tracking.
 * Border collision and self-collision end the game.
 *
 * Freestanding: no stdlib. All I/O via arm64_libc.h SVC syscalls.
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *       -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *       -I demos -e _start demos/arm64_start.S demos/games/snake.c \
 *       -o /tmp/snake.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONSTANTS                                                                */
/* ======================================================================== */

#define WIDTH       40
#define HEIGHT      20
#define MAX_SNAKE   (WIDTH * HEIGHT)
#define TICK_MS     150      /* ms per game tick                            */
#define INITIAL_LEN 3        /* starting body length (including head)       */

/* Direction encoding */
#define DIR_UP    0
#define DIR_DOWN  1
#define DIR_LEFT  2
#define DIR_RIGHT 3

/* ======================================================================== */
/* DATA STRUCTURES                                                          */
/* ======================================================================== */

/* A position on the grid. */
typedef struct {
    int x;
    int y;
} pos_t;

/* Snake state: ring buffer of body segments. */
static pos_t body[MAX_SNAKE];
static int   head_idx;          /* index into body[] for the head           */
static int   tail_idx;          /* index into body[] for the tail           */
static int   length;            /* current snake length                     */
static int   direction;         /* current movement direction               */
static int   next_direction;    /* buffered direction from input            */

/* Food position. */
static pos_t food;

/* Score. */
static int score;

/* Game-over flag. */
static int game_over;

/* Grid occupancy map: 0=empty, 1=snake, 2=food.
 * Used for O(1) collision checks and fast food placement. */
static char grid[HEIGHT][WIDTH];

/* ======================================================================== */
/* RING BUFFER HELPERS                                                      */
/* ======================================================================== */

static int ring_next(int idx) {
    return (idx + 1) % MAX_SNAKE;
}

static int ring_prev(int idx) {
    return (idx - 1 + MAX_SNAKE) % MAX_SNAKE;
}

/* ======================================================================== */
/* FOOD PLACEMENT                                                           */
/* ======================================================================== */

/* Place food on a random empty cell. If no empty cell exists (full board),
 * the player has effectively won, so we just end the game. */
static void place_food(void) {
    /* Count empty cells. */
    int empty_count = 0;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (grid[y][x] == 0) {
                empty_count++;
            }
        }
    }

    if (empty_count == 0) {
        /* Board full -- player wins. */
        game_over = 1;
        return;
    }

    /* Pick the Nth empty cell. */
    int target = rand() % empty_count;
    int count = 0;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            if (grid[y][x] == 0) {
                if (count == target) {
                    food.x = x;
                    food.y = y;
                    grid[y][x] = 2;
                    return;
                }
                count++;
            }
        }
    }
}

/* ======================================================================== */
/* INITIALIZATION                                                           */
/* ======================================================================== */

static void init_game(void) {
    /* Clear grid. */
    memset(grid, 0, sizeof(grid));

    /* Reset state. */
    score      = 0;
    game_over  = 0;
    direction  = DIR_RIGHT;
    next_direction = DIR_RIGHT;
    length     = INITIAL_LEN;
    head_idx   = INITIAL_LEN - 1;
    tail_idx   = 0;

    /* Place snake in the middle of the grid, extending leftward. */
    int start_x = WIDTH / 2 - INITIAL_LEN / 2;
    int start_y = HEIGHT / 2;

    for (int i = 0; i < INITIAL_LEN; i++) {
        body[i].x = start_x + i;
        body[i].y = start_y;
        grid[start_y][start_x + i] = 1;
    }

    /* Seed RNG with clock and place first food. */
    srand((unsigned long)clock_ms());
    place_food();
}

/* ======================================================================== */
/* INPUT HANDLING                                                           */
/* ======================================================================== */

/* Read all pending input characters. Only the last valid direction change
 * is kept, which prevents the snake from reversing into itself when
 * multiple keys arrive between ticks. */
static void process_input(void) {
    int ch;
    while ((ch = getchar()) != -1) {
        switch (ch) {
            case 'w': case 'W':
                if (direction != DIR_DOWN)
                    next_direction = DIR_UP;
                break;
            case 's': case 'S':
                if (direction != DIR_UP)
                    next_direction = DIR_DOWN;
                break;
            case 'a': case 'A':
                if (direction != DIR_RIGHT)
                    next_direction = DIR_LEFT;
                break;
            case 'd': case 'D':
                if (direction != DIR_LEFT)
                    next_direction = DIR_RIGHT;
                break;
            case 'q': case 'Q':
                game_over = 1;
                return;
        }
    }
}

/* ======================================================================== */
/* GAME LOGIC                                                               */
/* ======================================================================== */

static void update(void) {
    direction = next_direction;

    /* Compute new head position. */
    pos_t head = body[head_idx];
    pos_t new_head;

    switch (direction) {
        case DIR_UP:    new_head.x = head.x;     new_head.y = head.y - 1; break;
        case DIR_DOWN:  new_head.x = head.x;     new_head.y = head.y + 1; break;
        case DIR_LEFT:  new_head.x = head.x - 1; new_head.y = head.y;     break;
        case DIR_RIGHT: new_head.x = head.x + 1; new_head.y = head.y;     break;
        default:        new_head = head; break;
    }

    /* Border collision check. */
    if (new_head.x < 0 || new_head.x >= WIDTH ||
        new_head.y < 0 || new_head.y >= HEIGHT) {
        game_over = 1;
        return;
    }

    /* Check what occupies the target cell. */
    int ate_food = 0;
    if (grid[new_head.y][new_head.x] == 2) {
        ate_food = 1;
    } else if (grid[new_head.y][new_head.x] == 1) {
        /* Self-collision. The tail will retract, so check if this is
         * exactly the current tail position -- that cell will be vacated
         * this tick, making the move legal. */
        pos_t tail = body[tail_idx];
        if (new_head.x != tail.x || new_head.y != tail.y) {
            game_over = 1;
            return;
        }
    }

    /* Advance head in the ring buffer. */
    head_idx = ring_next(head_idx);
    body[head_idx] = new_head;
    grid[new_head.y][new_head.x] = 1;

    if (ate_food) {
        /* Grow: do not retract the tail. */
        score += 10;
        length++;
        place_food();
    } else {
        /* Retract tail. */
        pos_t tail = body[tail_idx];
        grid[tail.y][tail.x] = 0;
        tail_idx = ring_next(tail_idx);
    }
}

/* ======================================================================== */
/* RENDERING                                                                */
/* ======================================================================== */

/* Build the entire frame into a buffer, then write it in one syscall
 * to minimize I/O round-trips through the SVC interface. */

/* Frame buffer: worst case is roughly (WIDTH+3)*HEIGHT + header + footer.
 * Generous allocation to avoid any overflow. */
#define FRAME_BUF_SIZE 2048
static char frame[FRAME_BUF_SIZE];

static void render(void) {
    int pos = 0;

    /* ANSI: clear screen and move cursor to top-left. */
    const char *cls = "\033[2J\033[H";
    int cls_len = strlen(cls);
    memcpy(frame + pos, cls, cls_len);
    pos += cls_len;

    /* Title bar. */
    int n = snprintf(frame + pos, FRAME_BUF_SIZE - pos,
                     "  SNAKE  |  Score: %d  |  WASD to move  |  Q to quit\n", score);
    pos += n;

    /* Top border: +--...--+ */
    frame[pos++] = '+';
    for (int x = 0; x < WIDTH; x++) frame[pos++] = '-';
    frame[pos++] = '+';
    frame[pos++] = '\n';

    /* Grid rows. */
    for (int y = 0; y < HEIGHT; y++) {
        frame[pos++] = '|';
        for (int x = 0; x < WIDTH; x++) {
            char cell = grid[y][x];
            if (cell == 2) {
                /* Food. */
                frame[pos++] = '*';
            } else if (cell == 1) {
                /* Snake segment. Check if this is the head. */
                pos_t head = body[head_idx];
                if (x == head.x && y == head.y) {
                    frame[pos++] = 'O';
                } else {
                    frame[pos++] = 'o';
                }
            } else {
                frame[pos++] = ' ';
            }
        }
        frame[pos++] = '|';
        frame[pos++] = '\n';
    }

    /* Bottom border. */
    frame[pos++] = '+';
    for (int x = 0; x < WIDTH; x++) frame[pos++] = '-';
    frame[pos++] = '+';
    frame[pos++] = '\n';

    /* Flush frame. */
    frame[pos] = '\0';
    sys_write(1, frame, pos);
}

/* ======================================================================== */
/* GAME OVER SCREEN                                                         */
/* ======================================================================== */

static void render_game_over(void) {
    printf("\n");
    printf("  =============================\n");
    printf("         G A M E  O V E R\n");
    printf("  =============================\n");
    printf("       Final Score: %d\n", score);
    printf("       Snake Length: %d\n", length);
    printf("  =============================\n");
    printf("\n");
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    init_game();

    long last_tick = clock_ms();

    while (!game_over) {
        /* Process any pending input. */
        process_input();
        if (game_over) break;

        /* Tick-based update: advance the game at a fixed rate. */
        long now = clock_ms();
        if (now - last_tick >= TICK_MS) {
            last_tick = now;
            update();
            if (game_over) break;
            render();
        }

        /* Yield CPU to avoid busy-spinning the GPU dispatch loop. */
        sleep_ms(10);
    }

    /* Final render showing the state at death, then game-over banner. */
    render();
    render_game_over();

    return 0;
}
