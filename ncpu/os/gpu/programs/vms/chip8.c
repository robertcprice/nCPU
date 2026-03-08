/*
 * chip8.c -- CHIP-8 emulator for freestanding ARM64 on Metal GPU.
 *
 * Full CHIP-8 specification: 4K memory, 16 registers, 64x32 display,
 * 16-level stack, delay/sound timers, 16-key hex keypad.
 * All 35 original opcodes implemented.
 *
 * Display rendered as ASCII art via ANSI escape codes.
 * Built-in demo ROMs: maze (default), countdown.
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *     -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *     -I demos -e _start demos/arm64_start.S demos/vms/chip8.c \
 *     -o /tmp/chip8.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CHIP-8 CONSTANTS                                                         */
/* ======================================================================== */

#define MEM_SIZE    4096
#define NUM_REGS    16
#define STACK_SIZE  16
#define DISPLAY_W   64
#define DISPLAY_H   32
#define NUM_KEYS    16
#define ROM_START   0x200
#define FONT_START  0x000
#define FONT_SIZE   80
#define CYCLES_PER_FRAME  10
#define FRAME_MS          16  /* ~60 Hz */

/* ======================================================================== */
/* CHIP-8 STATE                                                             */
/* ======================================================================== */

static unsigned char mem[MEM_SIZE];
static unsigned char V[NUM_REGS];
static unsigned int  I;
static unsigned int  pc;
static unsigned char delay_timer;
static unsigned char sound_timer;
static unsigned int  stack[STACK_SIZE];
static unsigned char sp;
static unsigned char display[DISPLAY_W * DISPLAY_H];
static unsigned char keys[NUM_KEYS];
static int draw_flag;
static int running;
static int wait_key_reg;  /* register for FX0A, -1 if not waiting */

/* ======================================================================== */
/* FONT SPRITES (0-F, 5 bytes each at 0x000-0x04F)                         */
/* ======================================================================== */

static const unsigned char fontset[FONT_SIZE] = {
    0xF0, 0x90, 0x90, 0x90, 0xF0,  /* 0 */
    0x20, 0x60, 0x20, 0x20, 0x70,  /* 1 */
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  /* 2 */
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  /* 3 */
    0x90, 0x90, 0xF0, 0x10, 0x10,  /* 4 */
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  /* 5 */
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  /* 6 */
    0xF0, 0x10, 0x20, 0x40, 0x40,  /* 7 */
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  /* 8 */
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  /* 9 */
    0xF0, 0x90, 0xF0, 0x90, 0x90,  /* A */
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  /* B */
    0xF0, 0x80, 0x80, 0x80, 0xF0,  /* C */
    0xE0, 0x90, 0x90, 0x90, 0xE0,  /* D */
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  /* E */
    0xF0, 0x80, 0xF0, 0x80, 0x80   /* F */
};

/* ======================================================================== */
/* INITIALIZATION                                                           */
/* ======================================================================== */

static void chip8_init(void) {
    memset(mem, 0, MEM_SIZE);
    memset(V, 0, NUM_REGS);
    memset(stack, 0, sizeof(stack));
    memset(display, 0, DISPLAY_W * DISPLAY_H);
    memset(keys, 0, NUM_KEYS);
    I = 0;  pc = ROM_START;  sp = 0;
    delay_timer = 0;  sound_timer = 0;
    draw_flag = 1;  running = 1;  wait_key_reg = -1;
    memcpy(mem + FONT_START, fontset, FONT_SIZE);
}

/* ======================================================================== */
/* ROM LOADING                                                              */
/* ======================================================================== */

static int chip8_load_rom(const unsigned char *rom, int size) {
    if (size <= 0 || size > (MEM_SIZE - ROM_START)) return -1;
    memcpy(mem + ROM_START, rom, size);
    return 0;
}

static int chip8_load_file(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int total = 0;
    unsigned char buf[512];
    ssize_t n;
    while (total < (MEM_SIZE - ROM_START) && (n = read(fd, buf, sizeof(buf))) > 0) {
        int space = (MEM_SIZE - ROM_START) - total;
        int copy = (int)n < space ? (int)n : space;
        memcpy(mem + ROM_START + total, buf, copy);
        total += copy;
    }
    close(fd);
    return total > 0 ? total : -1;
}

/* ======================================================================== */
/* DISPLAY RENDERING                                                        */
/* ======================================================================== */

static void chip8_render(void) {
    printf("\033[H");  /* home cursor */
    putchar('+');
    for (int x = 0; x < DISPLAY_W; x++) putchar('-');
    putchar('+');  putchar('\n');
    for (int y = 0; y < DISPLAY_H; y++) {
        putchar('|');
        for (int x = 0; x < DISPLAY_W; x++)
            putchar(display[y * DISPLAY_W + x] ? '#' : ' ');
        putchar('|');  putchar('\n');
    }
    putchar('+');
    for (int x = 0; x < DISPLAY_W; x++) putchar('-');
    putchar('+');  putchar('\n');
}

/* ======================================================================== */
/* KEYPAD INPUT                                                             */
/* ======================================================================== */

/*
 * Keyboard layout -> CHIP-8 hex keys:
 *   1234 -> 1 2 3 C    qwer -> 4 5 6 D
 *   asdf -> 7 8 9 E    zxcv -> A 0 B F
 * Also accepts 0-9 and a-f as direct hex input.
 */
static int key_map(int ch) {
    switch (ch) {
        case '1': return 0x1;  case '2': return 0x2;
        case '3': return 0x3;  case '4': return 0xC;
        case 'q': return 0x4;  case 'w': return 0x5;
        case 'e': return 0x6;  case 'r': return 0xD;
        case 'a': return 0x7;  case 's': return 0x8;
        case 'd': return 0x9;  case 'f': return 0xE;
        case 'z': return 0xA;  case 'x': return 0x0;
        case 'c': return 0xB;  case 'v': return 0xF;
        case '0': return 0x0;  case '5': return 0x5;
        case '6': return 0x6;  case '7': return 0x7;
        case '8': return 0x8;  case '9': return 0x9;
        case 'b': return 0xB;
        default:  return -1;
    }
}

static void chip8_poll_keys(void) {
    memset(keys, 0, NUM_KEYS);
    int ch = getchar();
    if (ch == -1) return;
    if (ch == 27 || ch == 'Q') { running = 0; return; }
    int k = key_map(ch);
    if (k >= 0) {
        keys[k] = 1;
        if (wait_key_reg >= 0) {
            V[wait_key_reg] = (unsigned char)k;
            wait_key_reg = -1;
        }
    }
}

/* ======================================================================== */
/* OPCODE EXECUTION (all 35 standard CHIP-8 opcodes)                        */
/* ======================================================================== */

static void chip8_cycle(void) {
    if (wait_key_reg >= 0) return;

    unsigned int opcode = ((unsigned int)mem[pc] << 8) | mem[pc + 1];
    pc += 2;

    unsigned int nnn = opcode & 0x0FFF;
    unsigned int nn  = opcode & 0x00FF;
    unsigned int n   = opcode & 0x000F;
    unsigned int x   = (opcode >> 8) & 0x0F;
    unsigned int y   = (opcode >> 4) & 0x0F;

    switch (opcode & 0xF000) {

    case 0x0000:
        if (opcode == 0x00E0) {
            memset(display, 0, DISPLAY_W * DISPLAY_H);
            draw_flag = 1;
        } else if (opcode == 0x00EE) {
            if (sp > 0) { sp--; pc = stack[sp]; }
        }
        /* 0NNN (RCA 1802 call) ignored */
        break;

    case 0x1000: pc = nnn; break;                          /* 1NNN: JP   */

    case 0x2000:                                            /* 2NNN: CALL */
        if (sp < STACK_SIZE) { stack[sp] = pc; sp++; }
        pc = nnn;
        break;

    case 0x3000: if (V[x] == nn) pc += 2; break;           /* 3XNN: SE   */
    case 0x4000: if (V[x] != nn) pc += 2; break;           /* 4XNN: SNE  */
    case 0x5000: if (V[x] == V[y]) pc += 2; break;         /* 5XY0: SE   */
    case 0x6000: V[x] = (unsigned char)nn; break;           /* 6XNN: LD   */
    case 0x7000: V[x] = (unsigned char)(V[x]+nn); break;    /* 7XNN: ADD  */

    case 0x8000:
        switch (n) {
        case 0x0: V[x] = V[y]; break;
        case 0x1: V[x] |= V[y]; V[0xF] = 0; break;
        case 0x2: V[x] &= V[y]; V[0xF] = 0; break;
        case 0x3: V[x] ^= V[y]; V[0xF] = 0; break;
        case 0x4: {
            unsigned int s = V[x] + V[y];
            V[x] = (unsigned char)(s & 0xFF);
            V[0xF] = (s > 0xFF) ? 1 : 0;
            break;
        }
        case 0x5: {
            unsigned char vf = (V[x] >= V[y]) ? 1 : 0;
            V[x] = (unsigned char)(V[x] - V[y]);
            V[0xF] = vf;
            break;
        }
        case 0x6: {
            unsigned char lsb = V[x] & 1;
            V[x] >>= 1;  V[0xF] = lsb;
            break;
        }
        case 0x7: {
            unsigned char vf = (V[y] >= V[x]) ? 1 : 0;
            V[x] = (unsigned char)(V[y] - V[x]);
            V[0xF] = vf;
            break;
        }
        case 0xE: {
            unsigned char msb = (V[x] >> 7) & 1;
            V[x] <<= 1;  V[0xF] = msb;
            break;
        }
        default: break;
        }
        break;

    case 0x9000: if (V[x] != V[y]) pc += 2; break;         /* 9XY0: SNE */
    case 0xA000: I = nnn; break;                             /* ANNN: LD I */
    case 0xB000: pc = V[0] + nnn; break;                    /* BNNN: JP V0 */
    case 0xC000: V[x] = (unsigned char)(rand() & nn); break; /* CXNN: RND */

    case 0xD000: {
        /* DXYN: Draw sprite. VF=1 on collision (XOR erase). */
        unsigned int xpos = V[x] % DISPLAY_W;
        unsigned int ypos = V[y] % DISPLAY_H;
        V[0xF] = 0;
        for (unsigned int row = 0; row < n; row++) {
            unsigned int py = (ypos + row) % DISPLAY_H;
            unsigned char sb = mem[I + row];
            for (unsigned int col = 0; col < 8; col++) {
                unsigned int px = (xpos + col) % DISPLAY_W;
                if ((sb >> (7 - col)) & 1) {
                    int idx = py * DISPLAY_W + px;
                    if (display[idx]) V[0xF] = 1;
                    display[idx] ^= 1;
                }
            }
        }
        draw_flag = 1;
        break;
    }

    case 0xE000:
        if (nn == 0x9E) {
            if (V[x] < NUM_KEYS && keys[V[x]]) pc += 2;
        } else if (nn == 0xA1) {
            if (V[x] >= NUM_KEYS || !keys[V[x]]) pc += 2;
        }
        break;

    case 0xF000:
        switch (nn) {
        case 0x07: V[x] = delay_timer; break;
        case 0x0A: wait_key_reg = (int)x; break;
        case 0x15: delay_timer = V[x]; break;
        case 0x18: sound_timer = V[x]; break;
        case 0x1E: I += V[x]; if (I > 0xFFF) I &= 0xFFF; break;
        case 0x29: I = FONT_START + (V[x] & 0x0F) * 5; break;
        case 0x33:
            if (I + 2 < MEM_SIZE) {
                mem[I] = V[x] / 100;
                mem[I+1] = (V[x] / 10) % 10;
                mem[I+2] = V[x] % 10;
            }
            break;
        case 0x55:
            for (unsigned int i = 0; i <= x; i++)
                if (I + i < MEM_SIZE) mem[I+i] = V[i];
            I += x + 1;
            break;
        case 0x65:
            for (unsigned int i = 0; i <= x; i++)
                if (I + i < MEM_SIZE) V[i] = mem[I+i];
            I += x + 1;
            break;
        default: break;
        }
        break;

    default: break;
    }
}

/* ======================================================================== */
/* TIMERS                                                                   */
/* ======================================================================== */

static void chip8_tick_timers(void) {
    if (delay_timer > 0) delay_timer--;
    if (sound_timer > 0) {
        if (sound_timer == 1) printf("\a");
        sound_timer--;
    }
}

/* ======================================================================== */
/* BUILT-IN ROM: Maze Generator                                             */
/* ======================================================================== */

/*
 * Classic CHIP-8 maze. Randomly draws / or \ tiles across the 64x32 display.
 *
 *   0x200: V0=0, V1=0                     ; init X, Y
 *   0x204: V2=random&1                    ; pick direction
 *   0x206: skip if V2==0 -> I=backslash   ; select sprite
 *   0x20A: skip if V2==1 -> I=slash
 *   0x20E: DRW V0,V1,4                    ; draw 4x4 tile
 *   0x210: V0+=4                           ; next column
 *   0x212: skip if V0==64 -> JP 0x204     ; end of row?
 *   0x216: V0=0, V1+=4                    ; next row
 *   0x21A: skip if V1==32 -> JP 0x204     ; end of screen?
 *   0x21E: JP 0x21E                        ; halt
 *   0x220: sprite data
 */
static const unsigned char maze_rom[] = {
    0x60, 0x00,  0x61, 0x00,  0xC2, 0x01,  0x32, 0x00,
    0xA2, 0x20,  0x32, 0x01,  0xA2, 0x25,  0xD0, 0x14,
    0x70, 0x04,  0x30, 0x40,  0x12, 0x04,  0x60, 0x00,
    0x71, 0x04,  0x31, 0x20,  0x12, 0x04,  0x12, 0x1E,
    /* backslash sprite */  0x80, 0x40, 0x20, 0x10,
    /* padding */           0x00,
    /* slash sprite */      0x10, 0x20, 0x40, 0x80
};

/* ======================================================================== */
/* BUILT-IN ROM: Countdown (3-2-1)                                          */
/* ======================================================================== */

/*
 * Displays digits 3, 2, 1 centered on screen using built-in font sprites,
 * with 1-second delay timer pauses between each. Then clears and halts.
 */
static const unsigned char countdown_rom[] = {
    0x00, 0xE0,          /* CLS               */
    /* -- digit 3 -- */
    0x60, 0x03,          /* LD V0, 3          */
    0xF0, 0x29,          /* LD I, font[V0]    */
    0x61, 0x1C,          /* LD V1, 28 (x)     */
    0x62, 0x0D,          /* LD V2, 13 (y)     */
    0xD1, 0x25,          /* DRW V1, V2, 5     */
    0x63, 0x3C,          /* LD V3, 60         */
    0xF3, 0x15,          /* DT = V3           */
    0xF3, 0x07,          /* V3 = DT           */
    0x33, 0x00,          /* SE V3, 0          */
    0x12, 0x10,          /* JP wait loop      */
    /* -- digit 2 -- */
    0x00, 0xE0,          /* CLS               */
    0x60, 0x02,          /* LD V0, 2          */
    0xF0, 0x29,          /* LD I, font[V0]    */
    0xD1, 0x25,          /* DRW V1, V2, 5     */
    0x63, 0x3C,          /* LD V3, 60         */
    0xF3, 0x15,          /* DT = V3           */
    0xF3, 0x07,          /* V3 = DT           */
    0x33, 0x00,          /* SE V3, 0          */
    0x12, 0x20,          /* JP wait loop      */
    /* -- digit 1 -- */
    0x00, 0xE0,          /* CLS               */
    0x60, 0x01,          /* LD V0, 1          */
    0xF0, 0x29,          /* LD I, font[V0]    */
    0xD1, 0x25,          /* DRW V1, V2, 5     */
    0x63, 0x3C,          /* LD V3, 60         */
    0xF3, 0x15,          /* DT = V3           */
    0xF3, 0x07,          /* V3 = DT           */
    0x33, 0x00,          /* SE V3, 0          */
    0x12, 0x30,          /* JP wait loop      */
    /* -- done -- */
    0x00, 0xE0,          /* CLS               */
    0x12, 0x38,          /* JP halt           */
};

/* ======================================================================== */
/* ROM TABLE                                                                */
/* ======================================================================== */

struct demo_entry {
    const char          *name;
    const unsigned char *data;
    int                  size;
};

static const struct demo_entry demos[] = {
    { "maze",      maze_rom,      sizeof(maze_rom)      },
    { "countdown", countdown_rom, sizeof(countdown_rom) },
};
#define NUM_DEMOS  (int)(sizeof(demos) / sizeof(demos[0]))

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    chip8_init();
    srand((unsigned long)clock_ms());

    int rom_loaded = 0;
    const char *rom_name = "maze";

    /* Try to read a ROM path or demo name from stdin */
    char path_buf[256];
    int path_len = 0;
    int ch = getchar();
    if (ch != -1 && ch != '\n') {
        path_buf[path_len++] = (char)ch;
        while (path_len < 255) {
            ch = getchar();
            if (ch == -1 || ch == '\n') break;
            path_buf[path_len++] = (char)ch;
        }
        path_buf[path_len] = '\0';

        /* Check built-in demo names */
        for (int i = 0; i < NUM_DEMOS; i++) {
            if (strcmp(path_buf, demos[i].name) == 0) {
                chip8_load_rom(demos[i].data, demos[i].size);
                rom_name = demos[i].name;
                rom_loaded = 1;
                break;
            }
        }
        /* Try loading as file path */
        if (!rom_loaded) {
            int loaded = chip8_load_file(path_buf);
            if (loaded > 0) {
                rom_loaded = 1;
                rom_name = path_buf;
                printf("Loaded ROM: %s (%d bytes)\n", path_buf, loaded);
            } else {
                printf("Failed to load '%s', using built-in maze.\n", path_buf);
            }
        }
    }

    if (!rom_loaded)
        chip8_load_rom(maze_rom, sizeof(maze_rom));

    /* Clear screen, print header */
    printf("\033[2J\033[H");
    printf("nCPU CHIP-8 Emulator | ROM: %s\n", rom_name);
    printf("Keys: 1234/qwer/asdf/zxcv | Q or ESC to quit\n\n");

    /* Emulation loop */
    long last_timer_ms = clock_ms();
    long total_cycles = 0;
    int max_iterations = 50000;  /* safety limit for GPU execution */

    while (running && max_iterations > 0) {
        chip8_poll_keys();
        if (!running) break;

        for (int i = 0; i < CYCLES_PER_FRAME && running; i++) {
            if (pc >= MEM_SIZE - 1) {
                printf("\nPC out of bounds (0x%03x). Halted.\n", pc);
                running = 0;
                break;
            }
            /* Detect halt (JP to self) */
            unsigned int op = ((unsigned int)mem[pc] << 8) | mem[pc + 1];
            if ((op & 0xF000) == 0x1000 && (op & 0x0FFF) == pc) {
                chip8_cycle();
                total_cycles++;
                if (draw_flag) { chip8_render(); draw_flag = 0; }
                printf("PC:%03x I:%03x DT:%02x ST:%02x SP:%d  cyc:%ld\n",
                       pc, I, delay_timer, sound_timer, sp, total_cycles);
                printf("Program halted (JP self at 0x%03x).\n", pc);
                running = 0;
                break;
            }
            chip8_cycle();
            total_cycles++;
        }

        /* Timer tick at ~60 Hz */
        long now = clock_ms();
        if (now - last_timer_ms >= FRAME_MS) {
            chip8_tick_timers();
            last_timer_ms = now;
        }

        /* Render if display changed */
        if (draw_flag) {
            chip8_render();
            draw_flag = 0;
            printf("PC:%03x I:%03x DT:%02x ST:%02x SP:%d  cyc:%ld  [Q=quit]\n",
                   pc, I, delay_timer, sound_timer, sp, total_cycles);
        }

        sleep_ms(1);
        max_iterations--;
    }

    if (max_iterations <= 0)
        printf("\nIteration limit reached (%ld cycles).\n", total_cycles);
    printf("\nCHIP-8 session ended. %ld cycles executed.\n", total_cycles);
    return 0;
}
