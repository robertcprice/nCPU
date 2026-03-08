/*
 * forth.c -- Interactive Forth REPL for freestanding ARM64 on Metal GPU.
 *
 * A compact, standards-inspired Forth interpreter with:
 *   - 64-cell data stack and 64-cell return stack
 *   - Colon definitions, variables, constants
 *   - Control flow: IF/ELSE/THEN, DO/LOOP, BEGIN/UNTIL
 *   - String output via ."
 *   - Integer-only arithmetic (no floating point)
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *     -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *     -I demos -e _start demos/arm64_start.S demos/vms/forth.c \
 *     -o /tmp/forth.elf
 *
 * All I/O via SVC syscalls routed through the Python GPU OS layer.
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* =========================================================================
 * CONFIGURATION
 * ========================================================================= */

#define STACK_SIZE     64
#define RSTACK_SIZE    64
#define MAX_WORDS      256
#define MAX_WORD_LEN   32
#define MAX_BODY       256    /* max compiled cells per word */
#define LINE_BUF_SIZE  256
#define PAD_SIZE       128
#define MAX_VARS       64

/* =========================================================================
 * CELL TYPE
 * ========================================================================= */

typedef long cell_t;

/* =========================================================================
 * DATA STACK
 * ========================================================================= */

static cell_t dstack[STACK_SIZE];
static int    dsp = 0;   /* points to next free slot */

static int push(cell_t v) {
    if (dsp >= STACK_SIZE) {
        printf("? stack overflow\n");
        return -1;
    }
    dstack[dsp++] = v;
    return 0;
}

static int pop(cell_t *v) {
    if (dsp <= 0) {
        printf("? stack underflow\n");
        return -1;
    }
    *v = dstack[--dsp];
    return 0;
}

static int peek(int depth, cell_t *v) {
    int idx = dsp - 1 - depth;
    if (idx < 0 || idx >= dsp) {
        printf("? stack underflow\n");
        return -1;
    }
    *v = dstack[idx];
    return 0;
}

/* =========================================================================
 * RETURN STACK
 * ========================================================================= */

static cell_t rstack[RSTACK_SIZE];
static int    rsp = 0;

static int rpush(cell_t v) {
    if (rsp >= RSTACK_SIZE) {
        printf("? return stack overflow\n");
        return -1;
    }
    rstack[rsp++] = v;
    return 0;
}

static int rpop(cell_t *v) {
    if (rsp <= 0) {
        printf("? return stack underflow\n");
        return -1;
    }
    *v = rstack[--rsp];
    return 0;
}

/* =========================================================================
 * COMPILED WORD OPCODES
 *
 * The body[] of a colon definition is an array of cell_t opcodes.
 * Positive values = index into dictionary (call that word).
 * Negative values = encoded operations:
 * ========================================================================= */

#define OP_LIT          (-1)   /* next cell is a literal */
#define OP_BRANCH       (-2)   /* unconditional branch; next cell = offset */
#define OP_BRANCH0      (-3)   /* branch if TOS == 0;  next cell = offset */
#define OP_DO           (-4)   /* DO: move limit and index to return stack */
#define OP_LOOP         (-5)   /* LOOP: increment, branch if not done */
#define OP_I            (-6)   /* push loop index */
#define OP_DOTQUOTE     (-7)   /* inline string: next cell = length, then chars packed */
#define OP_EXIT         (-8)   /* return from colon def */
#define OP_STORE        (-9)   /* ! */
#define OP_FETCH        (-10)  /* @ */

/* =========================================================================
 * DICTIONARY ENTRY
 * ========================================================================= */

#define FLAG_IMMEDIATE  1
#define FLAG_HIDDEN     2

struct word_entry {
    char     name[MAX_WORD_LEN];
    int      flags;
    int      is_primitive;     /* 1 = built-in, 0 = colon def */

    /* For primitives: function pointer index (we use a switch) */
    int      prim_id;

    /* For colon defs: compiled body */
    cell_t   body[MAX_BODY];
    int      body_len;

    /* For variables: storage address (index into var_store) */
    int      var_index;        /* -1 if not a variable */
};

static struct word_entry dict[MAX_WORDS];
static int dict_count = 0;

/* Variable storage */
static cell_t var_store[MAX_VARS];
static int    var_count = 0;

/* =========================================================================
 * PRIMITIVE IDS
 * ========================================================================= */

enum {
    PRIM_PLUS = 1, PRIM_MINUS, PRIM_STAR, PRIM_SLASH, PRIM_MOD,
    PRIM_DOT, PRIM_DOTS, PRIM_CR, PRIM_EMIT,
    PRIM_DUP, PRIM_DROP, PRIM_SWAP, PRIM_OVER, PRIM_ROT,
    PRIM_NIP, PRIM_TUCK, PRIM_2DUP, PRIM_2DROP,
    PRIM_EQ, PRIM_LT, PRIM_GT, PRIM_NE, PRIM_LE, PRIM_GE,
    PRIM_AND, PRIM_OR, PRIM_XOR, PRIM_INVERT,
    PRIM_NEGATE, PRIM_ABS, PRIM_MIN, PRIM_MAX,
    PRIM_STORE, PRIM_FETCH,
    PRIM_DEPTH, PRIM_PICK, PRIM_ROLL,
    PRIM_RTOF, PRIM_FTOR, PRIM_RFETCH,
    PRIM_HERE, PRIM_CELLS,
    PRIM_SPACES, PRIM_SPACE,
    PRIM_BL,
    PRIM_KEY,
};

/* =========================================================================
 * DICTIONARY: ADD PRIMITIVE
 * ========================================================================= */

static int add_primitive(const char *name, int prim_id, int flags) {
    if (dict_count >= MAX_WORDS) return -1;
    struct word_entry *w = &dict[dict_count];
    strncpy(w->name, name, MAX_WORD_LEN - 1);
    w->name[MAX_WORD_LEN - 1] = '\0';
    w->flags = flags;
    w->is_primitive = 1;
    w->prim_id = prim_id;
    w->body_len = 0;
    w->var_index = -1;
    dict_count++;
    return dict_count - 1;
}

/* =========================================================================
 * DICTIONARY LOOKUP (search from end for most recent definition)
 * ========================================================================= */

static int find_word(const char *name) {
    for (int i = dict_count - 1; i >= 0; i--) {
        if (!(dict[i].flags & FLAG_HIDDEN) && strcmp(dict[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/* =========================================================================
 * EXECUTE A PRIMITIVE
 * ========================================================================= */

static int exec_primitive(int prim_id) {
    cell_t a, b, c;

    switch (prim_id) {
    case PRIM_PLUS:
        if (pop(&b) || pop(&a)) return -1;
        return push(a + b);
    case PRIM_MINUS:
        if (pop(&b) || pop(&a)) return -1;
        return push(a - b);
    case PRIM_STAR:
        if (pop(&b) || pop(&a)) return -1;
        return push(a * b);
    case PRIM_SLASH:
        if (pop(&b) || pop(&a)) return -1;
        if (b == 0) { printf("? division by zero\n"); return -1; }
        return push(a / b);
    case PRIM_MOD:
        if (pop(&b) || pop(&a)) return -1;
        if (b == 0) { printf("? division by zero\n"); return -1; }
        return push(a % b);
    case PRIM_DOT:
        if (pop(&a)) return -1;
        printf("%ld ", a);
        return 0;
    case PRIM_DOTS: {
        printf("<%d> ", dsp);
        for (int i = 0; i < dsp; i++) {
            printf("%ld ", dstack[i]);
        }
        return 0;
    }
    case PRIM_CR:
        putchar('\n');
        return 0;
    case PRIM_EMIT:
        if (pop(&a)) return -1;
        putchar((char)a);
        return 0;
    case PRIM_DUP:
        if (peek(0, &a)) return -1;
        return push(a);
    case PRIM_DROP:
        return pop(&a);
    case PRIM_SWAP:
        if (dsp < 2) { printf("? stack underflow\n"); return -1; }
        a = dstack[dsp - 1];
        dstack[dsp - 1] = dstack[dsp - 2];
        dstack[dsp - 2] = a;
        return 0;
    case PRIM_OVER:
        if (peek(1, &a)) return -1;
        return push(a);
    case PRIM_ROT:
        if (dsp < 3) { printf("? stack underflow\n"); return -1; }
        a = dstack[dsp - 3];
        dstack[dsp - 3] = dstack[dsp - 2];
        dstack[dsp - 2] = dstack[dsp - 1];
        dstack[dsp - 1] = a;
        return 0;
    case PRIM_NIP:
        if (pop(&a) || pop(&b)) return -1;
        return push(a);
    case PRIM_TUCK:
        if (dsp < 2) { printf("? stack underflow\n"); return -1; }
        a = dstack[dsp - 1]; /* TOS */
        b = dstack[dsp - 2]; /* NOS */
        dstack[dsp - 2] = a;
        dstack[dsp - 1] = b;
        return push(a);
    case PRIM_2DUP:
        if (peek(1, &a) || peek(0, &b)) return -1;
        if (push(a) || push(b)) return -1;
        return 0;
    case PRIM_2DROP:
        if (pop(&a) || pop(&b)) return -1;
        return 0;
    case PRIM_EQ:
        if (pop(&b) || pop(&a)) return -1;
        return push(a == b ? -1 : 0);
    case PRIM_LT:
        if (pop(&b) || pop(&a)) return -1;
        return push(a < b ? -1 : 0);
    case PRIM_GT:
        if (pop(&b) || pop(&a)) return -1;
        return push(a > b ? -1 : 0);
    case PRIM_NE:
        if (pop(&b) || pop(&a)) return -1;
        return push(a != b ? -1 : 0);
    case PRIM_LE:
        if (pop(&b) || pop(&a)) return -1;
        return push(a <= b ? -1 : 0);
    case PRIM_GE:
        if (pop(&b) || pop(&a)) return -1;
        return push(a >= b ? -1 : 0);
    case PRIM_AND:
        if (pop(&b) || pop(&a)) return -1;
        return push(a & b);
    case PRIM_OR:
        if (pop(&b) || pop(&a)) return -1;
        return push(a | b);
    case PRIM_XOR:
        if (pop(&b) || pop(&a)) return -1;
        return push(a ^ b);
    case PRIM_INVERT:
        if (pop(&a)) return -1;
        return push(~a);
    case PRIM_NEGATE:
        if (pop(&a)) return -1;
        return push(-a);
    case PRIM_ABS:
        if (pop(&a)) return -1;
        return push(a < 0 ? -a : a);
    case PRIM_MIN:
        if (pop(&b) || pop(&a)) return -1;
        return push(a < b ? a : b);
    case PRIM_MAX:
        if (pop(&b) || pop(&a)) return -1;
        return push(a > b ? a : b);
    case PRIM_STORE:
        if (pop(&a) || pop(&b)) return -1;
        if (a < 0 || a >= var_count) {
            printf("? invalid address %ld\n", a);
            return -1;
        }
        var_store[a] = b;
        return 0;
    case PRIM_FETCH:
        if (pop(&a)) return -1;
        if (a < 0 || a >= var_count) {
            printf("? invalid address %ld\n", a);
            return -1;
        }
        return push(var_store[a]);
    case PRIM_DEPTH:
        return push(dsp);
    case PRIM_PICK:
        if (pop(&a)) return -1;
        if (peek((int)a, &b)) return -1;
        return push(b);
    case PRIM_ROLL: {
        if (pop(&a)) return -1;
        int n = (int)a;
        if (n < 0 || n >= dsp) {
            printf("? invalid roll\n");
            return -1;
        }
        if (n == 0) return 0;
        int idx = dsp - 1 - n;
        cell_t val = dstack[idx];
        for (int i = idx; i < dsp - 1; i++) {
            dstack[i] = dstack[i + 1];
        }
        dstack[dsp - 1] = val;
        return 0;
    }
    case PRIM_RTOF:    /* r> */
        if (rpop(&a)) return -1;
        return push(a);
    case PRIM_FTOR:    /* >r */
        if (pop(&a)) return -1;
        return rpush(a);
    case PRIM_RFETCH:  /* r@ */
        if (rsp <= 0) { printf("? return stack underflow\n"); return -1; }
        return push(rstack[rsp - 1]);
    case PRIM_CELLS:
        if (pop(&a)) return -1;
        return push(a);  /* cells are 1-indexed in our var_store */
    case PRIM_SPACES:
        if (pop(&a)) return -1;
        for (cell_t i = 0; i < a; i++) putchar(' ');
        return 0;
    case PRIM_SPACE:
        putchar(' ');
        return 0;
    case PRIM_BL:
        return push(32);
    case PRIM_KEY:
        a = (cell_t)sys_read(0, (char *)&c, 1);
        if (a <= 0) return push(-1);
        return push(c & 0xFF);
    default:
        printf("? unknown primitive %d\n", prim_id);
        return -1;
    }
}

/* =========================================================================
 * EXECUTE A COMPILED (COLON) DEFINITION
 * ========================================================================= */

static int execute_word(int word_idx);  /* forward declaration */

static int run_body(cell_t *body, int body_len) {
    int ip = 0;
    cell_t a, b;

    while (ip < body_len) {
        cell_t op = body[ip];

        if (op == OP_LIT) {
            ip++;
            if (ip >= body_len) { printf("? bad literal\n"); return -1; }
            if (push(body[ip])) return -1;
            ip++;
        }
        else if (op == OP_BRANCH) {
            ip++;
            if (ip >= body_len) { printf("? bad branch\n"); return -1; }
            ip = (int)body[ip];
        }
        else if (op == OP_BRANCH0) {
            ip++;
            if (ip >= body_len) { printf("? bad branch0\n"); return -1; }
            if (pop(&a)) return -1;
            if (a == 0)
                ip = (int)body[ip];
            else
                ip++;
        }
        else if (op == OP_DO) {
            ip++;
            /* Stack: ( limit index -- ) */
            if (pop(&a) || pop(&b)) return -1;
            /* Push limit then index onto return stack */
            if (rpush(b) || rpush(a)) return -1;
        }
        else if (op == OP_LOOP) {
            ip++;
            if (ip >= body_len) { printf("? bad loop\n"); return -1; }
            /* Increment index on return stack */
            if (rsp < 2) { printf("? return stack underflow in loop\n"); return -1; }
            rstack[rsp - 1] += 1;
            /* Check: if index < limit, branch back */
            cell_t index = rstack[rsp - 1];
            cell_t limit = rstack[rsp - 2];
            if (index < limit) {
                ip = (int)body[ip];
            } else {
                /* Loop done: clean up return stack */
                rsp -= 2;
                ip++;
            }
        }
        else if (op == OP_I) {
            ip++;
            if (rsp < 1) { printf("? no loop index\n"); return -1; }
            if (push(rstack[rsp - 1])) return -1;
        }
        else if (op == OP_DOTQUOTE) {
            ip++;
            if (ip >= body_len) { printf("? bad .\" \n"); return -1; }
            int slen = (int)body[ip];
            ip++;
            /* Characters are packed one per cell */
            for (int i = 0; i < slen && ip < body_len; i++, ip++) {
                putchar((char)body[ip]);
            }
        }
        else if (op == OP_EXIT) {
            return 0;
        }
        else if (op == OP_STORE) {
            ip++;
            if (pop(&a) || pop(&b)) return -1;
            if (a < 0 || a >= var_count) {
                printf("? invalid address %ld\n", a);
                return -1;
            }
            var_store[a] = b;
        }
        else if (op == OP_FETCH) {
            ip++;
            if (pop(&a)) return -1;
            if (a < 0 || a >= var_count) {
                printf("? invalid address %ld\n", a);
                return -1;
            }
            if (push(var_store[a])) return -1;
        }
        else if (op >= 0 && op < dict_count) {
            /* Call another word */
            if (execute_word((int)op)) return -1;
            ip++;
        }
        else {
            printf("? bad opcode %ld at ip=%d\n", op, ip);
            return -1;
        }
    }
    return 0;
}

static int execute_word(int word_idx) {
    struct word_entry *w = &dict[word_idx];
    if (w->is_primitive) {
        /* Variable: just push its address */
        if (w->var_index >= 0) {
            return push((cell_t)w->var_index);
        }
        return exec_primitive(w->prim_id);
    } else {
        return run_body(w->body, w->body_len);
    }
}

/* =========================================================================
 * TOKENIZER
 *
 * Operates on a source line. Advances a cursor through the string.
 * ========================================================================= */

static const char *src_line = 0;
static int         src_pos  = 0;
static int         src_len  = 0;

static void set_source(const char *line, int len) {
    src_line = line;
    src_pos  = 0;
    src_len  = len;
}

/* Skip whitespace */
static void skip_spaces(void) {
    while (src_pos < src_len &&
           (src_line[src_pos] == ' '  ||
            src_line[src_pos] == '\t' ||
            src_line[src_pos] == '\r' ||
            src_line[src_pos] == '\n')) {
        src_pos++;
    }
}

/* Parse the next whitespace-delimited token.
 * Returns token length, or 0 if end of line.
 * Stores token in provided buffer. */
static int next_token(char *buf, int bufsz) {
    skip_spaces();
    if (src_pos >= src_len) return 0;

    int start = src_pos;
    while (src_pos < src_len &&
           src_line[src_pos] != ' '  &&
           src_line[src_pos] != '\t' &&
           src_line[src_pos] != '\r' &&
           src_line[src_pos] != '\n') {
        src_pos++;
    }
    int len = src_pos - start;
    if (len >= bufsz) len = bufsz - 1;
    memcpy(buf, src_line + start, len);
    buf[len] = '\0';
    return len;
}

/* Parse up to a delimiter character (used for ." strings).
 * Returns length of parsed content. Delimiter is consumed. */
static int parse_to(char delim, char *buf, int bufsz) {
    /* skip one leading space if present */
    if (src_pos < src_len && src_line[src_pos] == ' ')
        src_pos++;

    int out = 0;
    while (src_pos < src_len && src_line[src_pos] != delim) {
        if (out < bufsz - 1)
            buf[out++] = src_line[src_pos];
        src_pos++;
    }
    if (src_pos < src_len && src_line[src_pos] == delim)
        src_pos++;  /* consume delimiter */
    buf[out] = '\0';
    return out;
}

/* =========================================================================
 * TRY PARSING A NUMBER (decimal, hex with 0x prefix, or negative)
 * ========================================================================= */

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

static int is_hex_digit(char c) {
    return is_digit(c) ||
           (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return 0;
}

static int try_number(const char *token, cell_t *val) {
    const char *p = token;
    int neg = 0;

    if (*p == '-') {
        neg = 1;
        p++;
        if (*p == '\0') return 0;  /* lone minus sign */
    }

    /* Hex literal: 0x... */
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
        if (!is_hex_digit(*p)) return 0;
        cell_t n = 0;
        while (is_hex_digit(*p)) {
            n = n * 16 + hex_val(*p);
            p++;
        }
        if (*p != '\0') return 0;
        *val = neg ? -n : n;
        return 1;
    }

    /* Decimal */
    if (!is_digit(*p)) return 0;
    cell_t n = 0;
    while (is_digit(*p)) {
        n = n * 10 + (*p - '0');
        p++;
    }
    if (*p != '\0') return 0;
    *val = neg ? -n : n;
    return 1;
}

/* =========================================================================
 * STATE: INTERPRETING vs COMPILING
 * ========================================================================= */

static int state = 0;  /* 0 = interpreting, 1 = compiling */

/* Compilation target */
static int comp_idx = -1;  /* index in dict[] of word being compiled */

/* Control flow stack for compilation (tracks IF/ELSE/DO/BEGIN offsets) */
#define CF_STACK_SIZE 32
static int cf_stack[CF_STACK_SIZE];
static int cf_sp = 0;

static int cf_push(int v) {
    if (cf_sp >= CF_STACK_SIZE) {
        printf("? control flow stack overflow\n");
        return -1;
    }
    cf_stack[cf_sp++] = v;
    return 0;
}

static int cf_pop(int *v) {
    if (cf_sp <= 0) {
        printf("? control flow stack underflow\n");
        return -1;
    }
    *v = cf_stack[--cf_sp];
    return 0;
}

/* Append a cell to the word being compiled */
static int compile_cell(cell_t v) {
    if (comp_idx < 0 || comp_idx >= dict_count) {
        printf("? not compiling\n");
        return -1;
    }
    struct word_entry *w = &dict[comp_idx];
    if (w->body_len >= MAX_BODY) {
        printf("? definition too long\n");
        return -1;
    }
    w->body[w->body_len++] = v;
    return 0;
}

/* =========================================================================
 * OUTER INTERPRETER
 *
 * Processes one token at a time. In interpretation mode, words execute
 * immediately and numbers push to the stack. In compilation mode, words
 * are compiled into the current definition (unless IMMEDIATE).
 * ========================================================================= */

static int interpret_token(const char *token) {
    cell_t num;

    /* ----- COLON: begin definition ----- */
    if (strcmp(token, ":") == 0) {
        if (state != 0) {
            printf("? nested colon definitions not allowed\n");
            return -1;
        }
        char name[MAX_WORD_LEN];
        if (next_token(name, sizeof(name)) == 0) {
            printf("? missing word name\n");
            return -1;
        }
        if (dict_count >= MAX_WORDS) {
            printf("? dictionary full\n");
            return -1;
        }
        comp_idx = dict_count;
        struct word_entry *w = &dict[comp_idx];
        strncpy(w->name, name, MAX_WORD_LEN - 1);
        w->name[MAX_WORD_LEN - 1] = '\0';
        w->flags = FLAG_HIDDEN;   /* hide until ; completes */
        w->is_primitive = 0;
        w->prim_id = 0;
        w->body_len = 0;
        w->var_index = -1;
        dict_count++;
        state = 1;
        cf_sp = 0;
        return 0;
    }

    /* ----- SEMICOLON: end definition ----- */
    if (strcmp(token, ";") == 0) {
        if (state != 1) {
            printf("? ; outside definition\n");
            return -1;
        }
        if (cf_sp != 0) {
            printf("? unmatched control flow\n");
            return -1;
        }
        compile_cell(OP_EXIT);
        dict[comp_idx].flags &= ~FLAG_HIDDEN;  /* reveal */
        state = 0;
        comp_idx = -1;
        return 0;
    }

    /* ----- VARIABLE ----- */
    if (strcmp(token, "variable") == 0) {
        if (state != 0) {
            printf("? variable inside definition\n");
            return -1;
        }
        char name[MAX_WORD_LEN];
        if (next_token(name, sizeof(name)) == 0) {
            printf("? missing variable name\n");
            return -1;
        }
        if (dict_count >= MAX_WORDS || var_count >= MAX_VARS) {
            printf("? out of space\n");
            return -1;
        }
        int vi = var_count++;
        var_store[vi] = 0;
        struct word_entry *w = &dict[dict_count];
        strncpy(w->name, name, MAX_WORD_LEN - 1);
        w->name[MAX_WORD_LEN - 1] = '\0';
        w->flags = 0;
        w->is_primitive = 1;
        w->prim_id = 0;
        w->body_len = 0;
        w->var_index = vi;
        dict_count++;
        return 0;
    }

    /* ----- CONSTANT ----- */
    if (strcmp(token, "constant") == 0) {
        if (state != 0) {
            printf("? constant inside definition\n");
            return -1;
        }
        char name[MAX_WORD_LEN];
        if (next_token(name, sizeof(name)) == 0) {
            printf("? missing constant name\n");
            return -1;
        }
        cell_t val;
        if (pop(&val)) return -1;

        if (dict_count >= MAX_WORDS || var_count >= MAX_VARS) {
            printf("? out of space\n");
            return -1;
        }
        /* Store constant value, create word that pushes it */
        int vi = var_count++;
        var_store[vi] = val;

        /* Create as a colon def that pushes the literal */
        struct word_entry *w = &dict[dict_count];
        strncpy(w->name, name, MAX_WORD_LEN - 1);
        w->name[MAX_WORD_LEN - 1] = '\0';
        w->flags = 0;
        w->is_primitive = 0;
        w->prim_id = 0;
        w->var_index = -1;
        w->body[0] = OP_LIT;
        w->body[1] = val;
        w->body[2] = OP_EXIT;
        w->body_len = 3;
        dict_count++;
        return 0;
    }

    /* ----- CONTROL FLOW (compilation only) ----- */

    if (state == 1) {
        /* IF: compile conditional branch with placeholder */
        if (strcmp(token, "if") == 0) {
            compile_cell(OP_BRANCH0);
            int hole = dict[comp_idx].body_len;
            compile_cell(0);  /* placeholder */
            cf_push(hole);
            return 0;
        }
        /* ELSE: patch IF's branch, compile new unconditional branch */
        if (strcmp(token, "else") == 0) {
            int if_hole;
            if (cf_pop(&if_hole)) return -1;
            compile_cell(OP_BRANCH);
            int else_hole = dict[comp_idx].body_len;
            compile_cell(0);  /* placeholder */
            /* Patch the IF branch to jump here (after ELSE branch) */
            dict[comp_idx].body[if_hole] = dict[comp_idx].body_len;
            cf_push(else_hole);
            return 0;
        }
        /* THEN: patch the most recent branch hole */
        if (strcmp(token, "then") == 0) {
            int hole;
            if (cf_pop(&hole)) return -1;
            dict[comp_idx].body[hole] = dict[comp_idx].body_len;
            return 0;
        }
        /* DO: compile DO opcode */
        if (strcmp(token, "do") == 0) {
            compile_cell(OP_DO);
            cf_push(dict[comp_idx].body_len);  /* loop start address */
            return 0;
        }
        /* LOOP: compile LOOP opcode with back-branch */
        if (strcmp(token, "loop") == 0) {
            int loop_start;
            if (cf_pop(&loop_start)) return -1;
            compile_cell(OP_LOOP);
            compile_cell(loop_start);
            return 0;
        }
        /* I: push loop index */
        if (strcmp(token, "i") == 0) {
            compile_cell(OP_I);
            return 0;
        }
        /* BEGIN: mark loop start */
        if (strcmp(token, "begin") == 0) {
            cf_push(dict[comp_idx].body_len);
            return 0;
        }
        /* UNTIL: branch back to BEGIN if TOS == 0 */
        if (strcmp(token, "until") == 0) {
            int begin_addr;
            if (cf_pop(&begin_addr)) return -1;
            compile_cell(OP_BRANCH0);
            compile_cell(begin_addr);
            return 0;
        }
        /* AGAIN: unconditional branch back to BEGIN */
        if (strcmp(token, "again") == 0) {
            int begin_addr;
            if (cf_pop(&begin_addr)) return -1;
            compile_cell(OP_BRANCH);
            compile_cell(begin_addr);
            return 0;
        }
        /* WHILE: within BEGIN...WHILE...REPEAT */
        if (strcmp(token, "while") == 0) {
            compile_cell(OP_BRANCH0);
            int hole = dict[comp_idx].body_len;
            compile_cell(0);
            /* Push hole on top, but keep BEGIN address below */
            cf_push(hole);
            return 0;
        }
        /* REPEAT: branch back to BEGIN, patch WHILE */
        if (strcmp(token, "repeat") == 0) {
            int while_hole, begin_addr;
            if (cf_pop(&while_hole)) return -1;
            if (cf_pop(&begin_addr)) return -1;
            compile_cell(OP_BRANCH);
            compile_cell(begin_addr);
            dict[comp_idx].body[while_hole] = dict[comp_idx].body_len;
            return 0;
        }

        /* ." in compilation mode: compile inline string */
        if (strcmp(token, ".\"") == 0) {
            char strbuf[PAD_SIZE];
            int slen = parse_to('"', strbuf, sizeof(strbuf));
            compile_cell(OP_DOTQUOTE);
            compile_cell(slen);
            for (int i = 0; i < slen; i++) {
                compile_cell((cell_t)strbuf[i]);
            }
            return 0;
        }
    }

    /* ----- WORD LOOKUP ----- */
    int widx = find_word(token);
    if (widx >= 0) {
        struct word_entry *w = &dict[widx];

        if (state == 1 && !(w->flags & FLAG_IMMEDIATE)) {
            /* Compile a call to this word */
            if (w->is_primitive && w->var_index >= 0) {
                /* Variable: compile literal push of its address */
                compile_cell(OP_LIT);
                compile_cell((cell_t)w->var_index);
            } else if (w->is_primitive) {
                /* Inline the primitive: store as a negative prim opcode.
                 * We use the word index so run_body can dispatch it. */
                compile_cell((cell_t)widx);
            } else {
                compile_cell((cell_t)widx);
            }
            return 0;
        } else {
            /* Execute immediately */
            return execute_word(widx);
        }
    }

    /* ----- ." in interpret mode ----- */
    if (strcmp(token, ".\"") == 0) {
        char strbuf[PAD_SIZE];
        parse_to('"', strbuf, sizeof(strbuf));
        printf("%s", strbuf);
        return 0;
    }

    /* ----- ( comment ) ----- */
    if (strcmp(token, "(") == 0) {
        char discard[PAD_SIZE];
        parse_to(')', discard, sizeof(discard));
        return 0;
    }

    /* ----- \ line comment ----- */
    if (strcmp(token, "\\") == 0) {
        src_pos = src_len;  /* skip rest of line */
        return 0;
    }

    /* ----- NUMBER ----- */
    if (try_number(token, &num)) {
        if (state == 1) {
            compile_cell(OP_LIT);
            compile_cell(num);
        } else {
            if (push(num)) return -1;
        }
        return 0;
    }

    /* ----- UNKNOWN ----- */
    printf("? undefined word: %s\n", token);
    return -1;
}

/* =========================================================================
 * INTERPRET A FULL LINE
 * ========================================================================= */

static int interpret_line(const char *line, int len) {
    set_source(line, len);
    char token[MAX_WORD_LEN];
    int err = 0;

    while (next_token(token, sizeof(token)) > 0) {
        if (interpret_token(token) != 0) {
            err = 1;
            /* On error during compilation, abort the definition */
            if (state == 1) {
                if (comp_idx >= 0 && comp_idx < dict_count) {
                    dict_count--;  /* remove partial definition */
                }
                state = 0;
                comp_idx = -1;
                cf_sp = 0;
            }
            /* Clear data stack on error */
            dsp = 0;
            break;
        }
    }
    return err;
}

/* =========================================================================
 * BUILT-IN WORD DEFINITIONS (written in Forth)
 * ========================================================================= */

static void define_builtins(void) {
    /* Some useful definitions loaded at startup */
    static const char *builtins[] = {
        ": 1+ 1 + ;",
        ": 1- 1 - ;",
        ": 2+ 2 + ;",
        ": 2- 2 - ;",
        ": 2* 2 * ;",
        ": 2/ 2 / ;",
        ": 0= 0 = ;",
        ": 0< 0 < ;",
        ": 0> 0 > ;",
        ": not 0= ;",
        ": <> = not ;",
        ": true -1 ;",
        ": false 0 ;",
        ": ?dup dup if dup then ;",
        ": square dup * ;",
        ": cube dup dup * * ;",
        ": */ rot rot * swap / ;",
        0,
    };

    for (int i = 0; builtins[i]; i++) {
        const char *b = builtins[i];
        interpret_line(b, strlen(b));
    }
}

/* =========================================================================
 * WORDS: list all defined words
 * ========================================================================= */

static void list_words(void) {
    int col = 0;
    for (int i = 0; i < dict_count; i++) {
        if (dict[i].flags & FLAG_HIDDEN) continue;
        int wlen = strlen(dict[i].name);
        if (col + wlen + 1 > 72) {
            putchar('\n');
            col = 0;
        }
        printf("%s ", dict[i].name);
        col += wlen + 1;
    }
    if (col > 0) putchar('\n');
}

/* =========================================================================
 * SEE: decompile a word
 * ========================================================================= */

static void see_word(const char *name) {
    int widx = find_word(name);
    if (widx < 0) {
        printf("? undefined word: %s\n", name);
        return;
    }
    struct word_entry *w = &dict[widx];
    if (w->is_primitive) {
        if (w->var_index >= 0) {
            printf("%s is a variable (addr=%d, val=%ld)\n",
                   name, w->var_index, var_store[w->var_index]);
        } else {
            printf("%s is a primitive\n", name);
        }
        return;
    }

    printf(": %s ", name);
    int ip = 0;
    while (ip < w->body_len) {
        cell_t op = w->body[ip];
        if (op == OP_LIT) {
            ip++;
            printf("%ld ", w->body[ip]);
            ip++;
        } else if (op == OP_BRANCH) {
            ip++;
            printf("BRANCH->%ld ", w->body[ip]);
            ip++;
        } else if (op == OP_BRANCH0) {
            ip++;
            printf("BRANCH0->%ld ", w->body[ip]);
            ip++;
        } else if (op == OP_DO) {
            printf("DO ");
            ip++;
        } else if (op == OP_LOOP) {
            ip++;
            printf("LOOP->%ld ", w->body[ip]);
            ip++;
        } else if (op == OP_I) {
            printf("I ");
            ip++;
        } else if (op == OP_DOTQUOTE) {
            ip++;
            int slen = (int)w->body[ip];
            ip++;
            printf(".\" ");
            for (int i = 0; i < slen && ip < w->body_len; i++, ip++) {
                putchar((char)w->body[ip]);
            }
            printf("\" ");
        } else if (op == OP_EXIT) {
            printf("; ");
            ip++;
        } else if (op == OP_STORE) {
            printf("! ");
            ip++;
        } else if (op == OP_FETCH) {
            printf("@ ");
            ip++;
        } else if (op >= 0 && op < dict_count) {
            printf("%s ", dict[(int)op].name);
            ip++;
        } else {
            printf("?%ld? ", op);
            ip++;
        }
    }
    putchar('\n');
}

/* =========================================================================
 * HANDLE META COMMANDS (words, see, bye)
 * ========================================================================= */

static int handle_meta(const char *token) {
    if (strcmp(token, "words") == 0) {
        list_words();
        return 1;
    }
    if (strcmp(token, "see") == 0) {
        char name[MAX_WORD_LEN];
        if (next_token(name, sizeof(name)) > 0) {
            see_word(name);
        } else {
            printf("? see what?\n");
        }
        return 1;
    }
    if (strcmp(token, "bye") == 0) {
        printf("Goodbye.\n");
        sys_exit(0);
        return 1;  /* unreachable */
    }
    if (strcmp(token, "forget") == 0) {
        char name[MAX_WORD_LEN];
        if (next_token(name, sizeof(name)) > 0) {
            int widx = find_word(name);
            if (widx >= 0) {
                /* Truncate dictionary back to this word */
                dict_count = widx;
                printf("forgot %s and everything after\n", name);
            } else {
                printf("? undefined word: %s\n", name);
            }
        }
        return 1;
    }
    return 0;
}

/* =========================================================================
 * INTERPRET TOKEN (with meta command check)
 * ========================================================================= */

static int interpret_token_full(const char *token) {
    /* Check meta commands first (only in interpret mode) */
    if (state == 0) {
        /* Save and restore source position since handle_meta
         * may consume additional tokens */
        int saved_pos = src_pos;
        if (handle_meta(token)) {
            return 0;
        }
        /* Restore if meta didn't match (but it returns 0 for no match,
         * so src_pos was not modified) */
    }
    return interpret_token(token);
}

/* =========================================================================
 * FULL LINE INTERPRETER (with meta support)
 * ========================================================================= */

static int interpret_line_full(const char *line, int len) {
    set_source(line, len);
    char token[MAX_WORD_LEN];
    int err = 0;

    while (next_token(token, sizeof(token)) > 0) {
        /* Meta commands handled only in interpret mode */
        if (state == 0 && handle_meta(token)) {
            continue;
        }

        if (interpret_token(token) != 0) {
            err = 1;
            if (state == 1) {
                if (comp_idx >= 0 && comp_idx < dict_count) {
                    dict_count--;
                }
                state = 0;
                comp_idx = -1;
                cf_sp = 0;
            }
            dsp = 0;
            break;
        }
    }
    return err;
}

/* =========================================================================
 * READ A LINE FROM STDIN
 * ========================================================================= */

static int read_line(char *buf, int bufsz) {
    int n = sys_read(0, buf, bufsz - 1);
    if (n <= 0) return -1;

    /* Strip trailing newline/carriage return */
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        n--;
    buf[n] = '\0';
    return n;
}

/* =========================================================================
 * INITIALIZE DICTIONARY WITH PRIMITIVES
 * ========================================================================= */

static void init_dictionary(void) {
    /* Arithmetic */
    add_primitive("+",       PRIM_PLUS,    0);
    add_primitive("-",       PRIM_MINUS,   0);
    add_primitive("*",       PRIM_STAR,    0);
    add_primitive("/",       PRIM_SLASH,   0);
    add_primitive("mod",     PRIM_MOD,     0);

    /* Output */
    add_primitive(".",       PRIM_DOT,     0);
    add_primitive(".s",      PRIM_DOTS,    0);
    add_primitive("cr",      PRIM_CR,      0);
    add_primitive("emit",    PRIM_EMIT,    0);

    /* Stack manipulation */
    add_primitive("dup",     PRIM_DUP,     0);
    add_primitive("drop",    PRIM_DROP,    0);
    add_primitive("swap",    PRIM_SWAP,    0);
    add_primitive("over",    PRIM_OVER,    0);
    add_primitive("rot",     PRIM_ROT,     0);
    add_primitive("nip",     PRIM_NIP,     0);
    add_primitive("tuck",    PRIM_TUCK,    0);
    add_primitive("2dup",    PRIM_2DUP,    0);
    add_primitive("2drop",   PRIM_2DROP,   0);

    /* Comparison */
    add_primitive("=",       PRIM_EQ,      0);
    add_primitive("<",       PRIM_LT,      0);
    add_primitive(">",       PRIM_GT,      0);
    add_primitive("<>",      PRIM_NE,      0);
    add_primitive("<=",      PRIM_LE,      0);
    add_primitive(">=",      PRIM_GE,      0);

    /* Bitwise */
    add_primitive("and",     PRIM_AND,     0);
    add_primitive("or",      PRIM_OR,      0);
    add_primitive("xor",     PRIM_XOR,     0);
    add_primitive("invert",  PRIM_INVERT,  0);

    /* Arithmetic extras */
    add_primitive("negate",  PRIM_NEGATE,  0);
    add_primitive("abs",     PRIM_ABS,     0);
    add_primitive("min",     PRIM_MIN,     0);
    add_primitive("max",     PRIM_MAX,     0);

    /* Memory */
    add_primitive("!",       PRIM_STORE,   0);
    add_primitive("@",       PRIM_FETCH,   0);

    /* Stack inspection */
    add_primitive("depth",   PRIM_DEPTH,   0);
    add_primitive("pick",    PRIM_PICK,    0);
    add_primitive("roll",    PRIM_ROLL,    0);

    /* Return stack */
    add_primitive(">r",      PRIM_FTOR,    0);
    add_primitive("r>",      PRIM_RTOF,    0);
    add_primitive("r@",      PRIM_RFETCH,  0);

    /* Misc */
    add_primitive("cells",   PRIM_CELLS,   0);
    add_primitive("spaces",  PRIM_SPACES,  0);
    add_primitive("space",   PRIM_SPACE,   0);
    add_primitive("bl",      PRIM_BL,      0);
    add_primitive("key",     PRIM_KEY,     0);
}

/* =========================================================================
 * BANNER
 * ========================================================================= */

static void print_banner(void) {
    printf("\n");
    printf("  nCPU Forth v1.0 -- Metal GPU Freestanding\n");
    printf("  64-cell stack | Colon defs | DO/LOOP | IF/ELSE/THEN\n");
    printf("  Type 'words' for vocabulary, 'bye' to exit.\n");
    printf("\n");
}

/* =========================================================================
 * MAIN
 * ========================================================================= */

int main(void) {
    char line[LINE_BUF_SIZE];

    /* Initialize the Forth system */
    init_dictionary();
    define_builtins();

    print_banner();

    /* REPL loop */
    for (;;) {
        /* Prompt */
        if (state == 0) {
            printf("> ");
        } else {
            printf("| ");
        }

        /* Read a line */
        int len = read_line(line, sizeof(line));
        if (len < 0) break;  /* EOF */
        if (len == 0) continue;

        /* Interpret */
        int err = interpret_line_full(line, len);
        if (!err && state == 0) {
            printf(" ok\n");
        }
    }

    printf("\n");
    return 0;
}
