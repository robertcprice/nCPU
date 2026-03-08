/*
 * brainfuck.c -- Brainfuck interpreter for freestanding ARM64 on Metal GPU.
 *
 * Interprets the 8 BF commands: > < + - . , [ ]
 * 30,000-cell tape, bracket matching via scan.
 *
 * If no program is provided on stdin, runs a built-in Hello World demo.
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *     -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *     -I demos -e _start demos/arm64_start.S demos/vms/brainfuck.c \
 *     -o /tmp/brainfuck.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONSTANTS                                                                */
/* ======================================================================== */

#define TAPE_SIZE   30000
#define MAX_PROG    8192
#define MAX_DEPTH   256

/* ======================================================================== */
/* BRAINFUCK INTERPRETER                                                    */
/* ======================================================================== */

static unsigned char tape[TAPE_SIZE];

static void bf_run(const char *prog, int prog_len) {
    int pc = 0;        /* program counter */
    int dp = 0;        /* data pointer     */
    int depth;

    /* Zero the tape */
    memset(tape, 0, TAPE_SIZE);

    while (pc < prog_len) {
        char cmd = prog[pc];

        switch (cmd) {
            case '>':
                dp++;
                if (dp >= TAPE_SIZE) dp = 0;  /* wrap */
                break;

            case '<':
                dp--;
                if (dp < 0) dp = TAPE_SIZE - 1;  /* wrap */
                break;

            case '+':
                tape[dp]++;
                break;

            case '-':
                tape[dp]--;
                break;

            case '.':
                putchar((char)tape[dp]);
                break;

            case ',': {
                int ch = getchar();
                tape[dp] = (ch == -1) ? 0 : (unsigned char)ch;
                break;
            }

            case '[':
                if (tape[dp] == 0) {
                    /* Jump forward past matching ] */
                    depth = 1;
                    while (depth > 0 && pc < prog_len) {
                        pc++;
                        if (prog[pc] == '[') depth++;
                        else if (prog[pc] == ']') depth--;
                    }
                }
                break;

            case ']':
                if (tape[dp] != 0) {
                    /* Jump backward to matching [ */
                    depth = 1;
                    while (depth > 0 && pc > 0) {
                        pc--;
                        if (prog[pc] == ']') depth++;
                        else if (prog[pc] == '[') depth--;
                    }
                }
                break;

            default:
                /* Ignore all non-BF characters (comments) */
                break;
        }

        pc++;
    }
}

/* ======================================================================== */
/* BRACKET VALIDATION                                                       */
/* ======================================================================== */

static int bf_validate(const char *prog, int len) {
    int depth = 0;
    for (int i = 0; i < len; i++) {
        if (prog[i] == '[') depth++;
        else if (prog[i] == ']') depth--;
        if (depth < 0) return -1;  /* unmatched ] */
    }
    return depth;  /* 0 = balanced, >0 = unmatched [ */
}

/* ======================================================================== */
/* BUILT-IN DEMO: Hello World                                               */
/* ======================================================================== */

static const char hello_bf[] =
    "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]"
    ">>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    char prog[MAX_PROG];
    int total = 0;

    /* Read program from stdin until EOF or buffer full */
    while (total < MAX_PROG - 1) {
        int ch = getchar();
        if (ch == -1) break;
        prog[total++] = (char)ch;
    }

    /* If nothing was read, run the built-in Hello World */
    if (total == 0) {
        printf("nCPU Brainfuck Interpreter\n");
        printf("No program on stdin -- running built-in Hello World demo:\n\n");

        int demo_len = strlen(hello_bf);
        int check = bf_validate(hello_bf, demo_len);
        if (check != 0) {
            printf("Error: unmatched brackets in demo (%d)\n", check);
            return 1;
        }
        bf_run(hello_bf, demo_len);
        printf("\n");
        return 0;
    }

    /* Validate bracket matching */
    int check = bf_validate(prog, total);
    if (check != 0) {
        if (check < 0)
            printf("Error: unmatched ']' in program\n");
        else
            printf("Error: %d unmatched '[' in program\n", check);
        return 1;
    }

    /* Execute */
    bf_run(prog, total);

    return 0;
}
