/*
 * cc.c — Self-hosting C compiler for ARM64 Metal GPU.
 *
 * A complete C compiler written in freestanding C that runs entirely on
 * the Metal GPU compute shader. Compiles a subset of C into raw ARM64
 * machine code that can be directly executed via SYS_EXEC.
 *
 * Supported:
 *   - Types: int, long, char, void, pointers, arrays
 *   - Functions: declaration, definition, calls, return
 *   - Control flow: if/else, while, for, do-while, break, continue
 *   - Expressions: full precedence, unary, binary, ternary, sizeof
 *   - String/char literals, global/local variables
 *   - #define (constant macros), #include "file" / #include <file>
 *   - Casts, struct (basic)
 *
 * Compile (on host):
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *     -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *     -I demos -e _start demos/arm64_start.S demos/tools/cc.c
 *
 * Usage (on GPU):
 *   cc /tmp/hello.c          -> writes /bin/hello
 *   run /bin/hello            -> executes the compiled program
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONFIGURATION                                                            */
/* ======================================================================== */

#define MAX_TOKENS     32768
#define MAX_SOURCE     262144
#define MAX_OUTPUT     131072
#define MAX_SYMBOLS    1024
#define MAX_STRINGS    256
#define MAX_LOCALS     128
#define MAX_BREAKS     64
#define MAX_CONTS      64
#define MAX_STRUCT_FIELDS 32
#define MAX_STRUCTS    32
#define MAX_DEFINES    128
#define MAX_INCLUDE_BUF 65536

/* ARM64 code generation target addresses */
#define CODE_BASE      0x10000
#define DATA_BASE      0x50000
#define STACK_TOP      0xF00000

/* ======================================================================== */
/* TOKEN TYPES                                                              */
/* ======================================================================== */

enum {
    T_EOF = 0,
    T_NUM, T_STR, T_CHAR_LIT, T_IDENT,
    /* Keywords */
    T_INT, T_LONG, T_CHAR, T_VOID, T_UNSIGNED, T_SIGNED, T_STATIC,
    T_IF, T_ELSE, T_WHILE, T_FOR, T_DO, T_RETURN,
    T_BREAK, T_CONTINUE, T_SIZEOF, T_STRUCT, T_CONST,
    T_ENUM, T_TYPEDEF, T_SWITCH, T_CASE, T_DEFAULT, T_UNION,
    /* Operators */
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_AMP, T_PIPE, T_CARET, T_TILDE, T_BANG,
    T_LT, T_GT, T_LE, T_GE, T_EQ, T_NE,
    T_AND, T_OR,
    T_SHL, T_SHR,
    T_ASSIGN, T_PLUS_EQ, T_MINUS_EQ, T_STAR_EQ, T_SLASH_EQ,
    T_PERCENT_EQ, T_AMP_EQ, T_PIPE_EQ, T_CARET_EQ, T_SHL_EQ, T_SHR_EQ,
    T_INC, T_DEC,
    T_ARROW, T_DOT,
    T_QUESTION, T_COLON, T_SEMICOLON, T_COMMA,
    T_LPAREN, T_RPAREN, T_LBRACE, T_RBRACE, T_LBRACKET, T_RBRACKET,
    T_HASH, T_ELLIPSIS,
};

/* ======================================================================== */
/* TYPE SYSTEM                                                              */
/* ======================================================================== */

enum {
    TY_VOID = 0, TY_CHAR, TY_INT, TY_LONG, TY_PTR, TY_ARRAY, TY_STRUCT,
};

struct Type {
    int kind;
    int size;             /* bytes */
    int array_len;        /* for TY_ARRAY */
    int ptr_to;           /* index into types[] for TY_PTR/TY_ARRAY */
    int struct_id;        /* index into structs[] for TY_STRUCT */
    int is_unsigned;
};

#define MAX_TYPES 256
static struct Type types[MAX_TYPES];
static int n_types;

static int ty_void;
static int ty_char;
static int ty_int;
static int ty_long;

static int type_new(int kind, int size) {
    int id = n_types++;
    types[id].kind = kind;
    types[id].size = size;
    types[id].array_len = 0;
    types[id].ptr_to = 0;
    types[id].struct_id = -1;
    types[id].is_unsigned = 0;
    return id;
}

static int type_ptr(int base) {
    int id = type_new(TY_PTR, 8);
    types[id].ptr_to = base;
    return id;
}

static int type_array(int base, int len) {
    int id = type_new(TY_ARRAY, types[base].size * len);
    types[id].ptr_to = base;
    types[id].array_len = len;
    return id;
}

static int type_size(int t) {
    return types[t].size;
}

static int type_is_ptr(int t) {
    return types[t].kind == TY_PTR || types[t].kind == TY_ARRAY;
}

/* ======================================================================== */
/* STRUCT TABLE                                                             */
/* ======================================================================== */

struct StructField {
    char name[32];
    int type;
    int offset;
};

struct StructDef {
    char name[32];
    struct StructField fields[MAX_STRUCT_FIELDS];
    int n_fields;
    int size;
    int defined;
};

static struct StructDef structs[MAX_STRUCTS];
static int n_structs;

/* ======================================================================== */
/* TOKEN                                                                    */
/* ======================================================================== */

struct Token {
    long type;             /* token type (widened to avoid padding) */
    long val;              /* numeric value */
    long line;             /* source line number */
    char name[128];        /* identifier name or string content */
};
/* sizeof = 8+8+8+128 = 152 */

static struct Token tokens[MAX_TOKENS];
static int n_tokens;
static int tok_pos;

/* ======================================================================== */
/* SYMBOL TABLE                                                             */
/* ======================================================================== */

enum { SYM_GLOBAL = 0, SYM_LOCAL, SYM_PARAM, SYM_FUNC, SYM_ENUM, SYM_TYPEDEF };

struct Symbol {
    char name[64];
    int type;              /* type index */
    int kind;              /* SYM_GLOBAL, SYM_LOCAL, SYM_PARAM, SYM_FUNC */
    int offset;            /* stack offset (local), or data offset (global) */
    int n_params;          /* for functions */
    int defined;           /* function has body */
    int scope;             /* scope depth (0 = global) */
};

static struct Symbol symbols[MAX_SYMBOLS];
static int n_symbols;
static int scope_depth;

/* ======================================================================== */
/* STRING LITERALS                                                          */
/* ======================================================================== */

struct StringLit {
    char data[256];
    int len;
    int data_offset;       /* offset in data section */
};

static struct StringLit strings[MAX_STRINGS];
static int n_strings;

/* ======================================================================== */
/* PREPROCESSOR DEFINES                                                     */
/* ======================================================================== */

struct Define {
    char name[64];
    char value[128];
};

static struct Define defines[MAX_DEFINES];
static int n_defines;

/* Conditional compilation state */
static int ifdef_stack[16];  /* 1=active, 0=skipping */
static int ifdef_depth;
static int ifdef_active;     /* current state: 1=emitting tokens, 0=skipping */

/* Global variable initializer data (written directly into data section) */
#define MAX_INIT_DATA 8192
static unsigned char init_data[MAX_INIT_DATA];
static int init_data_used;  /* tracks whether any init data exists */

/* ======================================================================== */
/* CODE GENERATION STATE                                                    */
/* ======================================================================== */

static unsigned char output[MAX_OUTPUT];
static int code_pos;       /* current position in output[] */
static int data_pos;       /* current offset in data section */

static int stack_offset;   /* current function's stack usage */
static int label_counter;

/* Break/continue stack for loops */
static int break_labels[MAX_BREAKS];
static int break_depth;
static int cont_labels[MAX_CONTS];
static int cont_depth;

/* Forward reference fixups */
struct Fixup {
    int code_offset;       /* where in code[] the branch instruction lives */
    int label;             /* which label to resolve to */
    int type;              /* 0=B, 1=B.cond, 2=BL */
};

#define MAX_FIXUPS 8192
static struct Fixup fixups[MAX_FIXUPS];
static int n_fixups;

/* Label positions (code offset for each label) */
#define MAX_LABELS 8192
static int label_pos[MAX_LABELS];

/* Source buffer */
static char source[MAX_SOURCE];
static int src_len;

/* Error state */
static int had_error;
static int error_line;

/* ======================================================================== */
/* ERROR HANDLING                                                           */
/* ======================================================================== */

static void error(const char *msg) {
    if (had_error) return;
    had_error = 1;
    error_line = (tok_pos < n_tokens) ? tokens[tok_pos].line : 0;
    printf("error: line %d: %s\n", error_line, msg);
    /* Debug: print current and previous token type numbers */
    if (tok_pos < n_tokens) {
        printf("  cur_tok type=%ld pos=%d\n", tokens[tok_pos].type, tok_pos);
    }
    if (tok_pos > 0) {
        printf("  prev_tok type=%ld pos=%d\n", tokens[tok_pos - 1].type, tok_pos - 1);
    }
}

static void error_at(int line, const char *msg) {
    if (had_error) return;
    had_error = 1;
    error_line = line;
    printf("error: line %d: %s\n", line, msg);
}

/* ======================================================================== */
/* LEXER                                                                    */
/* ======================================================================== */

static int is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int is_digit(char c) {
    return c >= '0' && c <= '9';
}

static int is_alnum(char c) {
    return is_alpha(c) || is_digit(c);
}

static int is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int keyword(const char *s) {
    if (!strcmp(s, "int"))      return T_INT;
    if (!strcmp(s, "long"))     return T_LONG;
    if (!strcmp(s, "char"))     return T_CHAR;
    if (!strcmp(s, "void"))     return T_VOID;
    if (!strcmp(s, "unsigned")) return T_UNSIGNED;
    if (!strcmp(s, "signed"))   return T_SIGNED;
    if (!strcmp(s, "static"))   return T_STATIC;
    if (!strcmp(s, "if"))       return T_IF;
    if (!strcmp(s, "else"))     return T_ELSE;
    if (!strcmp(s, "while"))    return T_WHILE;
    if (!strcmp(s, "for"))      return T_FOR;
    if (!strcmp(s, "do"))       return T_DO;
    if (!strcmp(s, "return"))   return T_RETURN;
    if (!strcmp(s, "break"))    return T_BREAK;
    if (!strcmp(s, "continue")) return T_CONTINUE;
    if (!strcmp(s, "sizeof"))   return T_SIZEOF;
    if (!strcmp(s, "struct"))   return T_STRUCT;
    if (!strcmp(s, "const"))    return T_CONST;
    if (!strcmp(s, "enum"))     return T_ENUM;
    if (!strcmp(s, "typedef"))  return T_TYPEDEF;
    if (!strcmp(s, "switch"))   return T_SWITCH;
    if (!strcmp(s, "case"))     return T_CASE;
    if (!strcmp(s, "default"))  return T_DEFAULT;
    if (!strcmp(s, "union"))    return T_UNION;
    return 0;
}

/* Check if identifier matches a #define */
static int find_define(const char *name) {
    for (int i = 0; i < n_defines; i++) {
        if (!strcmp(defines[i].name, name)) return i;
    }
    return -1;
}

static void lex(void) {
    int i = 0;
    int line = 1;
    n_tokens = 0;

    while (i < src_len && n_tokens < MAX_TOKENS - 1) {
        char c = source[i];

        /* Newline */
        if (c == '\n') { line++; i++; continue; }

        /* Whitespace */
        if (is_space(c)) { i++; continue; }

        /* Line comment */
        if (c == '/' && i + 1 < src_len && source[i+1] == '/') {
            while (i < src_len && source[i] != '\n') i++;
            continue;
        }

        /* Block comment */
        if (c == '/' && i + 1 < src_len && source[i+1] == '*') {
            i += 2;
            while (i + 1 < src_len && !(source[i] == '*' && source[i+1] == '/')) {
                if (source[i] == '\n') line++;
                i++;
            }
            i += 2;
            continue;
        }

        /* Preprocessor */
        if (c == '#') {
            i++;
            while (i < src_len && source[i] == ' ') i++;

            /* #ifdef */
            if (i + 5 <= src_len && !strncmp(source + i, "ifdef", 5) && !is_alnum(source[i+5])) {
                i += 5;
                while (i < src_len && source[i] == ' ') i++;
                char mname[64]; int ns = 0;
                while (i < src_len && is_alnum(source[i]) && ns < 63) mname[ns++] = source[i++];
                mname[ns] = '\0';
                while (i < src_len && source[i] != '\n') i++;
                if (ifdef_depth < 15) {
                    ifdef_stack[ifdef_depth++] = ifdef_active;
                    if (ifdef_active) ifdef_active = (find_define(mname) >= 0) ? 1 : 0;
                }
                continue;
            }
            /* #ifndef */
            if (i + 6 <= src_len && !strncmp(source + i, "ifndef", 6) && !is_alnum(source[i+6])) {
                i += 6;
                while (i < src_len && source[i] == ' ') i++;
                char mname[64]; int ns = 0;
                while (i < src_len && is_alnum(source[i]) && ns < 63) mname[ns++] = source[i++];
                mname[ns] = '\0';
                while (i < src_len && source[i] != '\n') i++;
                if (ifdef_depth < 15) {
                    ifdef_stack[ifdef_depth++] = ifdef_active;
                    if (ifdef_active) ifdef_active = (find_define(mname) >= 0) ? 0 : 1;
                }
                continue;
            }
            /* #endif */
            if (i + 5 <= src_len && !strncmp(source + i, "endif", 5) && !is_alnum(source[i+5])) {
                i += 5;
                while (i < src_len && source[i] != '\n') i++;
                if (ifdef_depth > 0) ifdef_active = ifdef_stack[--ifdef_depth];
                continue;
            }
            /* #else */
            if (i + 4 <= src_len && !strncmp(source + i, "else", 4) && !is_alnum(source[i+4])) {
                i += 4;
                while (i < src_len && source[i] != '\n') i++;
                /* Only toggle if parent was active */
                if (ifdef_depth > 0 && ifdef_stack[ifdef_depth - 1]) {
                    ifdef_active = ifdef_active ? 0 : 1;
                }
                continue;
            }
            /* #undef */
            if (i + 5 <= src_len && !strncmp(source + i, "undef", 5) && !is_alnum(source[i+5])) {
                i += 5;
                while (i < src_len && source[i] == ' ') i++;
                char mname[64]; int ns = 0;
                while (i < src_len && is_alnum(source[i]) && ns < 63) mname[ns++] = source[i++];
                mname[ns] = '\0';
                while (i < src_len && source[i] != '\n') i++;
                if (ifdef_active) {
                    int di = find_define(mname);
                    if (di >= 0) {
                        /* Remove by shifting */
                        for (int j = di; j < n_defines - 1; j++) defines[j] = defines[j+1];
                        n_defines--;
                    }
                }
                continue;
            }

            /* If in skipped block, skip all non-ifdef directives */
            if (!ifdef_active) {
                while (i < src_len && source[i] != '\n') i++;
                continue;
            }

            /* #define */
            if (i + 6 <= src_len && !strncmp(source + i, "define", 6) && !is_alnum(source[i+6])) {
                i += 6;
                while (i < src_len && source[i] == ' ') i++;

                /* macro name */
                int ns = 0;
                char dname[64];
                while (i < src_len && is_alnum(source[i]) && ns < 63) {
                    dname[ns++] = source[i++];
                }
                dname[ns] = '\0';

                while (i < src_len && source[i] == ' ') i++;

                /* macro value (rest of line) */
                int vs = 0;
                char dval[128];
                while (i < src_len && source[i] != '\n' && vs < 127) {
                    dval[vs++] = source[i++];
                }
                /* trim trailing whitespace */
                while (vs > 0 && (dval[vs-1] == ' ' || dval[vs-1] == '\t')) vs--;
                dval[vs] = '\0';

                if (n_defines < MAX_DEFINES) {
                    strcpy(defines[n_defines].name, dname);
                    strcpy(defines[n_defines].value, dval);
                    n_defines++;
                }
                continue;
            }

            /* #include "file" or #include <file> */
            if (i + 7 <= src_len && !strncmp(source + i, "include", 7) && !is_alnum(source[i+7])) {
                i += 7;
                while (i < src_len && source[i] == ' ') i++;

                char inc_name[128];
                int fn = 0;
                char delim = 0;
                if (i < src_len && source[i] == '"') { delim = '"'; i++; }
                else if (i < src_len && source[i] == '<') { delim = '>'; i++; }

                if (delim) {
                    while (i < src_len && source[i] != delim && fn < 127)
                        inc_name[fn++] = source[i++];
                    inc_name[fn] = '\0';
                    if (i < src_len && source[i] == delim) i++;
                }
                while (i < src_len && source[i] != '\n') i++;
                if (i < src_len) i++; /* skip newline */

                if (!ifdef_active || fn == 0) continue;

                /* Try to open the included file */
                char inc_path[256];
                int inc_fd = -1;

                /* Try: /usr/include/filename */
                strcpy(inc_path, "/usr/include/");
                strcat(inc_path, inc_name);
                inc_fd = open(inc_path, O_RDONLY);

                if (inc_fd < 0) {
                    /* Try: filename as-is (absolute path) */
                    strcpy(inc_path, inc_name);
                    inc_fd = open(inc_path, O_RDONLY);
                }

                if (inc_fd >= 0) {
                    /* Read included file into temp buffer */
                    char inc_buf[MAX_INCLUDE_BUF];
                    int inc_len = (int)read(inc_fd, inc_buf, MAX_INCLUDE_BUF - 1);
                    close(inc_fd);

                    if (inc_len > 0 && (src_len + inc_len) < MAX_SOURCE) {
                        /* Shift remaining source right to make room */
                        for (int j = src_len - 1; j >= i; j--)
                            source[j + inc_len] = source[j];
                        /* Copy included content into the gap */
                        for (int j = 0; j < inc_len; j++)
                            source[i + j] = inc_buf[j];
                        src_len += inc_len;
                        source[src_len] = '\0';
                        /* Don't advance i — continue lexing from included content */
                    }
                }
                continue;
            }

            /* Unknown preprocessor directive — skip */
            while (i < src_len && source[i] != '\n') i++;
            continue;
        }

        /* If in a skipped #ifdef block, skip all non-preprocessor tokens */
        if (!ifdef_active) {
            if (c == '\n') { line++; }
            i++;
            continue;
        }

        struct Token *t = tokens + n_tokens;
        t->line = line;
        t->val = 0;
        t->name[0] = '\0';

        /* Number */
        if (is_digit(c)) {
            long val = 0;
            if (c == '0' && i + 1 < src_len && (source[i+1] == 'x' || source[i+1] == 'X')) {
                i += 2;
                while (i < src_len) {
                    char h = source[i];
                    if (h >= '0' && h <= '9') val = val * 16 + (h - '0');
                    else if (h >= 'a' && h <= 'f') val = val * 16 + (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') val = val * 16 + (h - 'A' + 10);
                    else break;
                    i++;
                }
            } else {
                while (i < src_len && is_digit(source[i])) {
                    val = val * 10 + (source[i] - '0');
                    i++;
                }
            }
            /* Skip suffixes: UL, ULL, L, LL, U */
            while (i < src_len && (source[i] == 'u' || source[i] == 'U' ||
                                    source[i] == 'l' || source[i] == 'L')) i++;
            t->type = T_NUM;
            t->val = val;
            n_tokens++;
            continue;
        }

        /* Character literal */
        if (c == '\'') {
            i++;
            long ch = 0;
            if (i < src_len && source[i] == '\\') {
                i++;
                switch (source[i]) {
                    case 'n':  ch = '\n'; break;
                    case 't':  ch = '\t'; break;
                    case 'r':  ch = '\r'; break;
                    case '0':  ch = '\0'; break;
                    case '\\': ch = '\\'; break;
                    case '\'': ch = '\''; break;
                    default:   ch = source[i]; break;
                }
                i++;
            } else {
                ch = source[i++];
            }
            if (i < src_len && source[i] == '\'') i++;
            t->type = T_CHAR_LIT;
            t->val = ch;
            n_tokens++;
            continue;
        }

        /* String literal */
        if (c == '"') {
            i++;
            int ns = 0;
            while (i < src_len && source[i] != '"' && ns < 127) {
                if (source[i] == '\\') {
                    i++;
                    switch (source[i]) {
                        case 'n':  t->name[ns++] = '\n'; break;
                        case 't':  t->name[ns++] = '\t'; break;
                        case 'r':  t->name[ns++] = '\r'; break;
                        case '0':  t->name[ns++] = '\0'; break;
                        case '\\': t->name[ns++] = '\\'; break;
                        case '"':  t->name[ns++] = '"';  break;
                        default:   t->name[ns++] = source[i]; break;
                    }
                } else {
                    t->name[ns++] = source[i];
                }
                i++;
            }
            t->name[ns] = '\0';
            t->val = ns;  /* string length */
            if (i < src_len) i++;  /* skip closing " */
            t->type = T_STR;
            n_tokens++;
            continue;
        }

        /* Identifier or keyword */
        if (is_alpha(c)) {
            int ns = 0;
            while (i < src_len && is_alnum(source[i]) && ns < 63) {
                t->name[ns++] = source[i++];
            }
            t->name[ns] = '\0';

            /* Check for #define expansion */
            int di = find_define(t->name);
            if (di >= 0) {
                /* Parse the define value as a number if possible */
                const char *dv = defines[di].value;
                /* Skip parens if present */
                while (*dv == '(' || *dv == ' ') dv++;
                int neg = 0;
                if (*dv == '-') { neg = 1; dv++; }
                if (is_digit(*dv) || (*dv == '0' && (dv[1] == 'x' || dv[1] == 'X'))) {
                    long val = strtol(dv, NULL, 0);
                    if (neg) val = -val;
                    t->type = T_NUM;
                    t->val = val;
                    n_tokens++;
                    continue;
                }
                /* Otherwise check if it's another define/keyword */
                int kw = keyword(defines[di].value);
                if (kw) { t->type = kw; n_tokens++; continue; }
                /* Treat as identifier with define name */
                strcpy(t->name, defines[di].value);
                t->type = T_IDENT;
                n_tokens++;
                continue;
            }

            int kw = keyword(t->name);
            if (kw) { t->type = kw; }
            else { t->type = T_IDENT; }
            n_tokens++;
            continue;
        }

        /* Operators and punctuation */
        i++;
        switch (c) {
            case '+':
                if (source[i] == '+') { t->type = T_INC; i++; }
                else if (source[i] == '=') { t->type = T_PLUS_EQ; i++; }
                else t->type = T_PLUS;
                break;
            case '-':
                if (source[i] == '-') { t->type = T_DEC; i++; }
                else if (source[i] == '=') { t->type = T_MINUS_EQ; i++; }
                else if (source[i] == '>') { t->type = T_ARROW; i++; }
                else t->type = T_MINUS;
                break;
            case '*':
                if (source[i] == '=') { t->type = T_STAR_EQ; i++; }
                else t->type = T_STAR;
                break;
            case '/':
                if (source[i] == '=') { t->type = T_SLASH_EQ; i++; }
                else t->type = T_SLASH;
                break;
            case '%':
                if (source[i] == '=') { t->type = T_PERCENT_EQ; i++; }
                else t->type = T_PERCENT;
                break;
            case '&':
                if (source[i] == '&') { t->type = T_AND; i++; }
                else if (source[i] == '=') { t->type = T_AMP_EQ; i++; }
                else t->type = T_AMP;
                break;
            case '|':
                if (source[i] == '|') { t->type = T_OR; i++; }
                else if (source[i] == '=') { t->type = T_PIPE_EQ; i++; }
                else t->type = T_PIPE;
                break;
            case '^':
                if (source[i] == '=') { t->type = T_CARET_EQ; i++; }
                else t->type = T_CARET;
                break;
            case '~': t->type = T_TILDE; break;
            case '!':
                if (source[i] == '=') { t->type = T_NE; i++; }
                else t->type = T_BANG;
                break;
            case '<':
                if (source[i] == '<') {
                    i++;
                    if (source[i] == '=') { t->type = T_SHL_EQ; i++; }
                    else t->type = T_SHL;
                } else if (source[i] == '=') { t->type = T_LE; i++; }
                else t->type = T_LT;
                break;
            case '>':
                if (source[i] == '>') {
                    i++;
                    if (source[i] == '=') { t->type = T_SHR_EQ; i++; }
                    else t->type = T_SHR;
                } else if (source[i] == '=') { t->type = T_GE; i++; }
                else t->type = T_GT;
                break;
            case '=':
                if (source[i] == '=') { t->type = T_EQ; i++; }
                else t->type = T_ASSIGN;
                break;
            case '?': t->type = T_QUESTION; break;
            case ':': t->type = T_COLON; break;
            case ';': t->type = T_SEMICOLON; break;
            case ',': t->type = T_COMMA; break;
            case '(': t->type = T_LPAREN; break;
            case ')': t->type = T_RPAREN; break;
            case '{': t->type = T_LBRACE; break;
            case '}': t->type = T_RBRACE; break;
            case '[': t->type = T_LBRACKET; break;
            case ']': t->type = T_RBRACKET; break;
            case '.':
                if (source[i] == '.' && source[i+1] == '.') {
                    t->type = T_ELLIPSIS; i += 2;
                } else {
                    t->type = T_DOT;
                }
                break;
            default:
                /* Skip unknown character */
                continue;
        }
        n_tokens++;
    }

    /* EOF token */
    tokens[n_tokens].type = T_EOF;
    tokens[n_tokens].line = line;
    tokens[n_tokens].name[0] = '\0';
}

/* ======================================================================== */
/* TOKEN HELPERS                                                            */
/* ======================================================================== */

static struct Token *peek(void) {
    return tokens + tok_pos;
}

static struct Token *advance(void) {
    if (tok_pos < n_tokens) {
        struct Token *p = tokens + tok_pos;
        tok_pos++;
        return p;
    }
    return tokens + n_tokens; /* EOF */
}

static int check(int type) {
    return tokens[tok_pos].type == type;
}

static int match(int type) {
    if (tokens[tok_pos].type == type) { tok_pos++; return 1; }
    return 0;
}

static void expect(int type) {
    if (tokens[tok_pos].type == type) { tok_pos++; return; }
    if      (type == T_SEMICOLON) error("expected ';'");
    else if (type == T_RPAREN)    error("expected ')'");
    else if (type == T_RBRACE)    error("expected '}'");
    else if (type == T_RBRACKET)  error("expected ']'");
    else if (type == T_LBRACE)    error("expected '{'");
    else                          error("unexpected token");
}

/* ======================================================================== */
/* SYMBOL TABLE OPERATIONS                                                  */
/* ======================================================================== */

static struct Symbol *find_symbol(const char *name) {
    /* Search from most recent (innermost scope) */
    for (int i = n_symbols - 1; i >= 0; i--) {
        if (!strcmp(symbols[i].name, name)) return symbols + i;
    }
    return NULL;
}

static struct Symbol *add_symbol(const char *name, int type, int kind) {
    if (n_symbols >= MAX_SYMBOLS) { error("too many symbols"); return NULL; }
    struct Symbol *s = symbols + n_symbols;
    n_symbols++;
    strncpy(s->name, name, 63);
    s->name[63] = '\0';
    s->type = type;
    s->kind = kind;
    s->offset = 0;
    s->n_params = 0;
    s->defined = 0;
    s->scope = scope_depth;
    return s;
}

static void enter_scope(void) { scope_depth++; }

static void leave_scope(void) {
    while (n_symbols > 0 && symbols[n_symbols - 1].scope == scope_depth) {
        n_symbols--;
    }
    scope_depth--;
}

/* ======================================================================== */
/* ARM64 INSTRUCTION ENCODING                                               */
/* ======================================================================== */

/* Emit a 32-bit little-endian instruction */
static void emit(unsigned int inst) {
    if (code_pos + 4 > MAX_OUTPUT) { error("output too large"); return; }
    output[code_pos++] = inst & 0xFF;
    output[code_pos++] = (inst >> 8) & 0xFF;
    output[code_pos++] = (inst >> 16) & 0xFF;
    output[code_pos++] = (inst >> 24) & 0xFF;
}

/* Register encoding: x0-x28 for general purpose, x29=FP, x30=LR, x31=SP/XZR */
#define XZR 31
#define SP  31
#define FP  29
#define LR  30

/* MOV Xd, Xm */
static void emit_mov(int rd, int rm) {
    /* SP (reg 31) needs ADD instead of ORR — ORR treats reg 31 as XZR */
    if (rd == SP || rm == SP) {
        /* ADD Xd, Xn, #0 — this treats reg 31 as SP */
        emit(0x91000000 | (rm << 5) | rd);
    } else {
        /* ORR Xd, XZR, Xm */
        emit(0xAA0003E0 | (rm << 16) | rd);
    }
}

/* MOV Xd, #imm16 (MOVZ) */
static void emit_movz(int rd, int imm16, int shift) {
    unsigned int hw = shift / 16;
    emit(0xD2800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd);
}

/* MOVK Xd, #imm16, LSL #shift */
static void emit_movk(int rd, int imm16, int shift) {
    unsigned int hw = shift / 16;
    emit(0xF2800000 | (hw << 21) | ((imm16 & 0xFFFF) << 5) | rd);
}

/* Load 64-bit immediate into register */
static void emit_li(int rd, long imm) {
    if (imm >= 0 && imm < 0x10000) {
        emit_movz(rd, (int)(imm & 0xFFFF), 0);
    } else if (imm >= 0 && imm < 0x100000000L) {
        emit_movz(rd, (int)(imm & 0xFFFF), 0);
        if (imm >> 16) emit_movk(rd, (int)((imm >> 16) & 0xFFFF), 16);
    } else {
        emit_movz(rd, (int)(imm & 0xFFFF), 0);
        if ((imm >> 16) & 0xFFFF) emit_movk(rd, (int)((imm >> 16) & 0xFFFF), 16);
        if ((imm >> 32) & 0xFFFF) emit_movk(rd, (int)((imm >> 32) & 0xFFFF), 32);
        if ((imm >> 48) & 0xFFFF) emit_movk(rd, (int)((imm >> 48) & 0xFFFF), 48);
    }
}

/* ADD Xd, Xn, Xm */
static void emit_add(int rd, int rn, int rm) {
    emit(0x8B000000 | (rm << 16) | (rn << 5) | rd);
}

/* ADD Xd, Xn, #imm12 */
static void emit_add_imm(int rd, int rn, int imm) {
    emit(0x91000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd);
}

/* SUB Xd, Xn, Xm */
static void emit_sub(int rd, int rn, int rm) {
    emit(0xCB000000 | (rm << 16) | (rn << 5) | rd);
}

/* SUB Xd, Xn, #imm12 */
static void emit_sub_imm(int rd, int rn, int imm) {
    emit(0xD1000000 | ((imm & 0xFFF) << 10) | (rn << 5) | rd);
}

/* SUBS Xd, Xn, Xm (sets flags) */
static void emit_subs(int rd, int rn, int rm) {
    emit(0xEB000000 | (rm << 16) | (rn << 5) | rd);
}

/* CMP Xn, Xm (SUBS XZR, Xn, Xm) */
static void emit_cmp(int rn, int rm) {
    emit_subs(XZR, rn, rm);
}

/* CMP Xn, #imm12 */
static void emit_cmp_imm(int rn, int imm) {
    emit(0xF100001F | ((imm & 0xFFF) << 10) | (rn << 5));
}

/* MUL Xd, Xn, Xm (MADD Xd, Xn, Xm, XZR) */
static void emit_mul(int rd, int rn, int rm) {
    emit(0x9B007C00 | (rm << 16) | (rn << 5) | rd);
}

/* SDIV Xd, Xn, Xm */
static void emit_sdiv(int rd, int rn, int rm) {
    emit(0x9AC00C00 | (rm << 16) | (rn << 5) | rd);
}

/* UDIV Xd, Xn, Xm */
static void emit_udiv(int rd, int rn, int rm) {
    emit(0x9AC00800 | (rm << 16) | (rn << 5) | rd);
}

/* MSUB Xd, Xn, Xm, Xa  (Xa - Xn*Xm) */
static void emit_msub(int rd, int rn, int rm, int ra) {
    emit(0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd);
}

/* AND Xd, Xn, Xm */
static void emit_and(int rd, int rn, int rm) {
    emit(0x8A000000 | (rm << 16) | (rn << 5) | rd);
}

/* ORR Xd, Xn, Xm */
static void emit_orr(int rd, int rn, int rm) {
    emit(0xAA000000 | (rm << 16) | (rn << 5) | rd);
}

/* EOR Xd, Xn, Xm */
static void emit_eor(int rd, int rn, int rm) {
    emit(0xCA000000 | (rm << 16) | (rn << 5) | rd);
}

/* LSL Xd, Xn, Xm (LSLV) */
static void emit_lsl(int rd, int rn, int rm) {
    emit(0x9AC02000 | (rm << 16) | (rn << 5) | rd);
}

/* LSR Xd, Xn, Xm (LSRV) */
static void emit_lsr(int rd, int rn, int rm) {
    emit(0x9AC02400 | (rm << 16) | (rn << 5) | rd);
}

/* ASR Xd, Xn, Xm (ASRV) */
static void emit_asr(int rd, int rn, int rm) {
    emit(0x9AC02800 | (rm << 16) | (rn << 5) | rd);
}

/* MVN Xd, Xm (ORN Xd, XZR, Xm) */
static void emit_mvn(int rd, int rm) {
    emit(0xAA200000 | (rm << 16) | (XZR << 5) | rd);
}

/* NEG Xd, Xm (SUB Xd, XZR, Xm) */
static void emit_neg(int rd, int rm) {
    emit_sub(rd, XZR, rm);
}

/* CSET Xd, cond  (CSINC Xd, XZR, XZR, !cond) */
static void emit_cset(int rd, int cond) {
    /* CSINC Xd, XZR, XZR, invert(cond) */
    int inv = cond ^ 1;
    emit(0x9A9F07E0 | (inv << 12) | rd);
}

/* Condition codes */
#define COND_EQ 0
#define COND_NE 1
#define COND_LT 11  /* B.LT */
#define COND_GE 10  /* B.GE */
#define COND_LE 13  /* B.LE */
#define COND_GT 12  /* B.GT */
#define COND_LS 9   /* B.LS (unsigned <=) */
#define COND_HI 8   /* B.HI (unsigned >) */
#define COND_CC 3   /* B.CC (unsigned <) */
#define COND_CS 2   /* B.CS (unsigned >=) */

/* STR Xd, [Xn, #imm] (unsigned offset, scaled by 8) */
static void emit_str(int rt, int rn, int offset) {
    int imm12 = (offset / 8) & 0xFFF;
    emit(0xF9000000 | (imm12 << 10) | (rn << 5) | rt);
}

/* LDR Xd, [Xn, #imm] (unsigned offset, scaled by 8) */
static void emit_ldr(int rt, int rn, int offset) {
    int imm12 = (offset / 8) & 0xFFF;
    emit(0xF9400000 | (imm12 << 10) | (rn << 5) | rt);
}

/* STRB Wd, [Xn, #imm] (byte store) */
static void emit_strb(int rt, int rn, int offset) {
    emit(0x39000000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt);
}

/* LDRB Wd, [Xn, #imm] (byte load, zero-extend) */
static void emit_ldrb(int rt, int rn, int offset) {
    emit(0x39400000 | ((offset & 0xFFF) << 10) | (rn << 5) | rt);
}

/* STR Wd, [Xn, #imm] (32-bit store, scaled by 4) */
static void emit_str32(int rt, int rn, int offset) {
    int imm12 = (offset / 4) & 0xFFF;
    emit(0xB9000000 | (imm12 << 10) | (rn << 5) | rt);
}

/* LDR Wd, [Xn, #imm] (32-bit load, scaled by 4) */
static void emit_ldr32(int rt, int rn, int offset) {
    int imm12 = (offset / 4) & 0xFFF;
    emit(0xB9400000 | (imm12 << 10) | (rn << 5) | rt);
}

/* LDRSW Xd, [Xn, #imm] (32-bit load, sign-extend to 64-bit) */
static void emit_ldrsw(int rt, int rn, int offset) {
    int imm12 = (offset / 4) & 0xFFF;
    emit(0xB9800000 | (imm12 << 10) | (rn << 5) | rt);
}

/* STP Xt1, Xt2, [Xn, #imm]! (pre-index) */
static void emit_stp_pre(int rt1, int rt2, int rn, int imm) {
    int imm7 = (imm / 8) & 0x7F;
    emit(0xA9800000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1);
}

/* LDP Xt1, Xt2, [Xn], #imm (post-index) */
static void emit_ldp_post(int rt1, int rt2, int rn, int imm) {
    int imm7 = (imm / 8) & 0x7F;
    emit(0xA8C00000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1);
}

/* SVC #0 */
static void emit_svc(void) {
    emit(0xD4000001);
}

/* NOP */
static void emit_nop(void) {
    emit(0xD503201F);
}

/* RET */
static void emit_ret(void) {
    emit(0xD65F03C0);
}

/* B #offset (unconditional branch, offset in bytes from current PC) */
static int emit_b(int label) {
    int pos = code_pos;
    /* Placeholder — will be fixed up */
    emit(0x14000000);
    if (n_fixups < MAX_FIXUPS) {
        fixups[n_fixups].code_offset = pos;
        fixups[n_fixups].label = label;
        fixups[n_fixups].type = 0;
        n_fixups++;
    } else {
        error("fixup table overflow (B)");
    }
    return pos;
}

/* B.cond #offset */
static int emit_bcond(int cond, int label) {
    int pos = code_pos;
    emit(0x54000000 | cond);
    if (n_fixups < MAX_FIXUPS) {
        fixups[n_fixups].code_offset = pos;
        fixups[n_fixups].label = label;
        fixups[n_fixups].type = 1;
        n_fixups++;
    } else {
        error("fixup table overflow (B.cond)");
    }
    return pos;
}

/* BL #offset (branch and link) */
static int emit_bl(int label) {
    int pos = code_pos;
    emit(0x94000000);
    if (n_fixups < MAX_FIXUPS) {
        fixups[n_fixups].code_offset = pos;
        fixups[n_fixups].label = label;
        fixups[n_fixups].type = 2;
        n_fixups++;
    } else {
        error("fixup table overflow (BL)");
    }
    return pos;
}

/* BLR Xn (branch to register and link) */
static void emit_blr(int rn) {
    emit(0xD63F0000 | (rn << 5));
}

/* BR Xn (branch to register) */
static void emit_br(int rn) {
    emit(0xD61F0000 | (rn << 5));
}

/* Allocate a new label */
static int new_label(void) {
    int l = label_counter++;
    if (l >= MAX_LABELS) {
        error("label table overflow");
        return 0;
    }
    label_pos[l] = -1;
    return l;
}

/* Mark label position */
static void mark_label(int label) {
    if (label < MAX_LABELS) label_pos[label] = code_pos;
}

/* Resolve all fixups */
static void resolve_fixups(void) {
    for (int i = 0; i < n_fixups; i++) {
        struct Fixup *f = fixups + i;
        if (f->label >= MAX_LABELS || label_pos[f->label] < 0) {
            error("unresolved label");
            continue;
        }
        int target = label_pos[f->label];
        int offset = (target - f->code_offset) / 4;

        /* Read existing instruction */
        unsigned int inst =
            output[f->code_offset] |
            (output[f->code_offset + 1] << 8) |
            (output[f->code_offset + 2] << 16) |
            (output[f->code_offset + 3] << 24);

        if (f->type == 3) {
            /* Absolute address: MOVZ+MOVK pair at code_offset */
            /* Compute absolute address of the label */
            long addr = CODE_BASE + target;
            int lo16 = (int)(addr & 0xFFFF);
            int hi16 = (int)((addr >> 16) & 0xFFFF);
            /* Patch MOVZ at code_offset */
            unsigned int movz = 0xD2800000 | ((lo16 & 0xFFFF) << 5) | (inst & 0x1F);
            output[f->code_offset]     = movz & 0xFF;
            output[f->code_offset + 1] = (movz >> 8) & 0xFF;
            output[f->code_offset + 2] = (movz >> 16) & 0xFF;
            output[f->code_offset + 3] = (movz >> 24) & 0xFF;
            /* Patch MOVK at code_offset+4 */
            int rd = inst & 0x1F;
            unsigned int movk = 0xF2A00000 | ((hi16 & 0xFFFF) << 5) | rd;
            output[f->code_offset + 4] = movk & 0xFF;
            output[f->code_offset + 5] = (movk >> 8) & 0xFF;
            output[f->code_offset + 6] = (movk >> 16) & 0xFF;
            output[f->code_offset + 7] = (movk >> 24) & 0xFF;
            continue;
        }

        if (f->type == 0) {
            /* B: imm26 */
            inst = (inst & 0xFC000000) | (offset & 0x03FFFFFF);
        } else if (f->type == 1) {
            /* B.cond: imm19 at bits [23:5] */
            inst = (inst & 0xFF00001F) | ((offset & 0x7FFFF) << 5);
        } else if (f->type == 2) {
            /* BL: imm26 */
            inst = (inst & 0xFC000000) | (offset & 0x03FFFFFF);
        }

        output[f->code_offset]     = inst & 0xFF;
        output[f->code_offset + 1] = (inst >> 8) & 0xFF;
        output[f->code_offset + 2] = (inst >> 16) & 0xFF;
        output[f->code_offset + 3] = (inst >> 24) & 0xFF;
    }
}

/* ======================================================================== */
/* REGISTER ALLOCATOR (simple: x0-x18 scratch, x19-x28 callee-saved)       */
/* ======================================================================== */

#define REG_COUNT 19  /* x0-x18 as scratch registers */
static int reg_used[REG_COUNT];

static int alloc_reg(void) {
    /* Prefer x9-x18 first (not parameter regs) */
    for (int i = 9; i < REG_COUNT; i++) {
        if (!reg_used[i]) { reg_used[i] = 1; return i; }
    }
    /* Fall back to x0-x8 */
    for (int i = 0; i < 9; i++) {
        if (!reg_used[i]) { reg_used[i] = 1; return i; }
    }
    error("out of registers");
    return 0;
}

static void free_reg(int r) {
    if (r >= 0 && r < REG_COUNT) reg_used[r] = 0;
}

static void reset_regs(void) {
    for (int i = 0; i < REG_COUNT; i++) reg_used[i] = 0;
}

/* ======================================================================== */
/* FUNCTION PROLOGUE/EPILOGUE                                               */
/* ======================================================================== */

static int func_ret_label;
static int func_stack_size;

static void emit_prologue(int local_size) {
    /* Align to 16 bytes */
    local_size = (local_size + 15) & ~15;
    func_stack_size = local_size + 16;  /* +16 for FP/LR */

    /* SUB SP, SP, #frame (placeholder — will be fixed up with actual size) */
    emit_sub_imm(SP, SP, func_stack_size);
    /* STP x29, x30, [SP, #0] (signed-offset, zero displacement) */
    emit(0xA9000000 | (LR << 10) | (SP << 5) | FP);
    /* MOV x29, SP */
    emit_mov(FP, SP);

    stack_offset = 16;
}

static void emit_epilogue(void) {
    mark_label(func_ret_label);
    /* MOV SP, x29 */
    emit_mov(SP, FP);
    /* LDP x29, x30, [SP, #0] (signed-offset, zero displacement) */
    emit(0xA9400000 | (LR << 10) | (SP << 5) | FP);
    /* ADD SP, SP, #frame (placeholder — will be fixed up with actual size) */
    emit_add_imm(SP, SP, func_stack_size);
    emit_ret();
}

/* ======================================================================== */
/* LOCAL VARIABLE ACCESS                                                    */
/* ======================================================================== */

/* Locals are stored at [FP, #-offset] where offset increases from 16 */
static int alloc_local(int size) {
    size = (size + 7) & ~7;  /* align to 8 */
    stack_offset += size;
    return -stack_offset;  /* negative offset from FP */
}

/* Load local variable into register */
static void emit_load_local(int rd, int offset, int type) {
    if (types[type].kind == TY_CHAR) {
        /* LDURB (unscaled offset for negative) */
        emit(0x38400000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rd);
    } else if (types[type].size == 4) {
        /* LDURSW (sign-extend 32 to 64) */
        emit(0xB8800000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rd);
    } else {
        /* LDUR X (64-bit) */
        emit(0xF8400000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rd);
    }
}

/* Store register to local variable */
static void emit_store_local(int rs, int offset, int type) {
    if (types[type].kind == TY_CHAR) {
        /* STURB */
        emit(0x38000000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rs);
    } else if (types[type].size == 4) {
        /* STUR W */
        emit(0xB8000000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rs);
    } else {
        /* STUR X */
        emit(0xF8000000 | (((unsigned int)offset & 0x1FF) << 12) | (FP << 5) | rs);
    }
}

/* Load from address in register */
static void emit_load_indirect(int rd, int raddr, int type) {
    if (types[type].kind == TY_CHAR) {
        emit_ldrb(rd, raddr, 0);
    } else if (types[type].size == 4) {
        emit_ldrsw(rd, raddr, 0);
    } else {
        emit_ldr(rd, raddr, 0);
    }
}

/* Store to address in register */
static void emit_store_indirect(int rs, int raddr, int type) {
    if (types[type].kind == TY_CHAR) {
        emit_strb(rs, raddr, 0);
    } else if (types[type].size == 4) {
        emit_str32(rs, raddr, 0);
    } else {
        emit_str(rs, raddr, 0);
    }
}

/* ======================================================================== */
/* GLOBAL VARIABLE ACCESS                                                   */
/* ======================================================================== */

static void emit_load_global_addr(int rd, int data_off) {
    long addr = DATA_BASE + data_off;
    emit_li(rd, addr);
}

/* ======================================================================== */
/* STRING LITERAL HANDLING                                                  */
/* ======================================================================== */

static int add_string(const char *s, int len) {
    /* Check for duplicate */
    for (int i = 0; i < n_strings; i++) {
        if (strings[i].len == len && !memcmp(strings[i].data, s, len)) {
            return i;
        }
    }
    if (n_strings >= MAX_STRINGS) { error("too many strings"); return 0; }
    int idx = n_strings++;
    memcpy(strings[idx].data, s, len);
    strings[idx].data[len] = '\0';
    strings[idx].len = len;
    strings[idx].data_offset = data_pos;
    data_pos += (len + 1 + 7) & ~7;  /* align to 8 */
    return idx;
}

/* ======================================================================== */
/* FORWARD DECLARATIONS FOR RECURSIVE DESCENT                               */
/* ======================================================================== */

static int parse_expr(void);
static int parse_assign(void);
static void parse_stmt(void);
static void parse_block(void);
static int parse_type(void);
static int parse_unary(void);

/* Check if current token starts a type specifier */
static int is_type_start(void) {
    if (check(T_INT) || check(T_LONG) || check(T_CHAR) || check(T_VOID) ||
        check(T_UNSIGNED) || check(T_SIGNED) || check(T_STRUCT) || check(T_CONST) ||
        check(T_STATIC) || check(T_ENUM) || check(T_UNION)) return 1;
    /* Check for typedef names */
    if (check(T_IDENT)) {
        struct Symbol *sym = find_symbol(peek()->name);
        if (sym && sym->kind == SYM_TYPEDEF) return 1;
    }
    return 0;
}

/* Expression result: register number and type */
static int expr_type;  /* type of last expression result */

/* Lvalue tracking for post/pre increment/decrement store-back */
static int last_lval_kind;   /* -1 = none, SYM_LOCAL/SYM_PARAM/SYM_GLOBAL */
static int last_lval_offset; /* stack offset (local/param) or data offset (global) */
static int last_lval_type;   /* variable type */

/* ======================================================================== */
/* TYPE PARSER                                                              */
/* ======================================================================== */

static int parse_type(void) {
    int t = ty_int;  /* default */
    int is_unsigned = 0;
    int is_const = 0;

    if (match(T_CONST)) is_const = 1;
    if (match(T_STATIC)) { /* ignore static for now */ }
    if (match(T_CONST)) is_const = 1;

    if (match(T_UNSIGNED)) {
        is_unsigned = 1;
        if (match(T_LONG)) t = ty_long;
        else if (match(T_INT)) t = ty_int;
        else if (match(T_CHAR)) t = ty_char;
        else t = ty_int;
    } else if (match(T_SIGNED)) {
        if (match(T_LONG)) t = ty_long;
        else if (match(T_INT)) t = ty_int;
        else if (match(T_CHAR)) t = ty_char;
        else t = ty_int;
    } else if (match(T_VOID)) {
        t = ty_void;
    } else if (match(T_CHAR)) {
        t = ty_char;
    } else if (match(T_INT)) {
        t = ty_int;
    } else if (match(T_LONG)) {
        t = ty_long;
        if (match(T_LONG)) { /* long long = long */ }
    } else if (match(T_ENUM)) {
        /* Parse enum { A=0, B, C } or enum Name { ... } */
        /* Optional name */
        if (check(T_IDENT) && tokens[tok_pos + 1].type == T_LBRACE) {
            advance(); /* skip enum tag name */
        } else if (check(T_IDENT)) {
            advance(); /* named enum without body — treat as int */
            return ty_int;
        }
        if (match(T_LBRACE)) {
            int val = 0;
            while (!check(T_RBRACE) && !check(T_EOF)) {
                if (!check(T_IDENT)) { error("expected enum constant"); break; }
                char ename[64];
                strcpy(ename, peek()->name);
                advance();
                if (match(T_ASSIGN)) {
                    /* Explicit value */
                    int neg = 0;
                    if (match(T_MINUS)) neg = 1;
                    if (check(T_NUM)) { val = (int)peek()->val; advance(); }
                    if (neg) val = -val;
                }
                /* Add as SYM_ENUM with value in offset */
                struct Symbol *es = add_symbol(ename, ty_int, SYM_ENUM);
                if (es) { es->offset = val; es->scope = 0; }
                val++;
                if (!match(T_COMMA)) break;
            }
            expect(T_RBRACE);
        }
        return ty_int; /* enums are ints in C */
    } else if (match(T_UNION)) {
        /* Union: like struct but all fields at offset 0 */
        if (!check(T_IDENT) && !check(T_LBRACE)) { error("expected union name or body"); return ty_int; }
        char uname[64];
        uname[0] = '\0';
        if (check(T_IDENT)) {
            strcpy(uname, peek()->name);
            advance();
        }
        /* Find or create struct entry (reuse struct table for unions) */
        int sid = -1;
        if (uname[0]) {
            for (int i = 0; i < n_structs; i++) {
                if (!strcmp(structs[i].name, uname)) { sid = i; break; }
            }
        }
        if (sid < 0) {
            sid = n_structs++;
            if (uname[0]) strcpy(structs[sid].name, uname);
            else strcpy(structs[sid].name, "__anon_union");
            structs[sid].n_fields = 0;
            structs[sid].size = 0;
            structs[sid].defined = 0;
        }
        if (match(T_LBRACE)) {
            int max_size = 0;
            while (!check(T_RBRACE) && !check(T_EOF)) {
                int ft = parse_type();
                while (match(T_STAR)) ft = type_ptr(ft);
                if (!check(T_IDENT)) { error("expected field name"); break; }
                char fname[32];
                strncpy(fname, peek()->name, 31);
                fname[31] = '\0';
                advance();
                int arr_len = 0;
                if (match(T_LBRACKET)) {
                    if (check(T_NUM)) { arr_len = (int)peek()->val; advance(); }
                    expect(T_RBRACKET);
                    if (arr_len > 0) ft = type_array(ft, arr_len);
                }
                if (structs[sid].n_fields < MAX_STRUCT_FIELDS) {
                    int fi = structs[sid].n_fields;
                    structs[sid].n_fields++;
                    strncpy(structs[sid].fields[fi].name, fname, 31);
                    structs[sid].fields[fi].name[31] = '\0';
                    structs[sid].fields[fi].type = ft;
                    structs[sid].fields[fi].offset = 0; /* all fields at offset 0 */
                    if (type_size(ft) > max_size) max_size = type_size(ft);
                }
                expect(T_SEMICOLON);
            }
            expect(T_RBRACE);
            structs[sid].size = (max_size + 7) & ~7;
            structs[sid].defined = 1;
        }
        t = type_new(TY_STRUCT, structs[sid].size);
        types[t].struct_id = sid;
        return t;
    } else if (match(T_STRUCT)) {
        /* struct name (optional for anonymous structs) */
        char sname[64];
        sname[0] = '\0';
        if (check(T_IDENT)) {
            strcpy(sname, peek()->name);
            advance();
        } else if (!check(T_LBRACE)) {
            error("expected struct name or '{'");
            return ty_int;
        }

        /* Find or create struct */
        int sid = -1;
        if (sname[0]) {
            for (int i = 0; i < n_structs; i++) {
                if (!strcmp(structs[i].name, sname)) { sid = i; break; }
            }
        }
        if (sid < 0) {
            sid = n_structs++;
            if (sname[0]) strcpy(structs[sid].name, sname);
            else strcpy(structs[sid].name, "__anon_struct");
            structs[sid].n_fields = 0;
            structs[sid].size = 0;
            structs[sid].defined = 0;
        }

        /* Struct definition? */
        if (match(T_LBRACE)) {
            int off = 0;
            while (!check(T_RBRACE) && !check(T_EOF)) {
                int ft = parse_type();
                while (match(T_STAR)) ft = type_ptr(ft);
                if (!check(T_IDENT)) { error("expected field name"); break; }
                char fname[32];
                strncpy(fname, peek()->name, 31);
                fname[31] = '\0';
                advance();

                int arr_len = 0;
                if (match(T_LBRACKET)) {
                    if (check(T_NUM)) { arr_len = (int)peek()->val; advance(); }
                    expect(T_RBRACKET);
                    if (arr_len > 0) ft = type_array(ft, arr_len);
                }

                /* Add field */
                if (structs[sid].n_fields < MAX_STRUCT_FIELDS) {
                    int align = type_size(ft);
                    if (align > 8) align = 8;
                    if (align > 0) off = (off + align - 1) & ~(align - 1);

                    int fi = structs[sid].n_fields;
                    structs[sid].n_fields++;
                    strncpy(structs[sid].fields[fi].name, fname, 31);
                    structs[sid].fields[fi].name[31] = '\0';
                    structs[sid].fields[fi].type = ft;
                    structs[sid].fields[fi].offset = off;
                    off += type_size(ft);
                }
                expect(T_SEMICOLON);
            }
            expect(T_RBRACE);
            structs[sid].size = (off + 7) & ~7;
            structs[sid].defined = 1;
        }

        t = type_new(TY_STRUCT, structs[sid].size);
        types[t].struct_id = sid;
        return t;
    } else {
        /* Could be a typedef, struct name, or unknown */
        if (check(T_IDENT)) {
            /* Check for typedef */
            struct Symbol *tds = find_symbol(peek()->name);
            if (tds && tds->kind == SYM_TYPEDEF) {
                advance();
                return tds->type;
            }
            /* Check if it's a struct name */
            for (int i = 0; i < n_structs; i++) {
                if (!strcmp(structs[i].name, peek()->name)) {
                    advance();
                    t = type_new(TY_STRUCT, structs[i].size);
                    types[t].struct_id = i;
                    return t;
                }
            }
            error("unknown type");
            return ty_int;
        }
        error("expected type");
        return ty_int;
    }

    if (match(T_LONG)) { t = ty_long; }  /* e.g., unsigned long */

    (void)is_const;
    if (is_unsigned) {
        int ut = type_new(types[t].kind, types[t].size);
        types[ut].is_unsigned = 1;
        return ut;
    }
    return t;
}

/* ======================================================================== */
/* EXPRESSION PARSER — Pratt parser (operator precedence climbing)          */
/* ======================================================================== */

/* Parse primary expression, return register holding result */
static int parse_primary(void) {
    if (had_error) return 0;
    last_lval_kind = -1;  /* Reset lvalue tracking */

    /* Number literal */
    if (check(T_NUM)) {
        long val = peek()->val;
        advance();
        int r = alloc_reg();
        emit_li(r, val);
        expr_type = ty_long;
        return r;
    }

    /* Character literal */
    if (check(T_CHAR_LIT)) {
        long val = peek()->val;
        advance();
        int r = alloc_reg();
        emit_li(r, val);
        expr_type = ty_char;
        return r;
    }

    /* String literal */
    if (check(T_STR)) {
        int sid = add_string(peek()->name, (int)peek()->val);
        advance();

        /* Concatenate adjacent string literals */
        while (check(T_STR)) {
            /* Append to existing string */
            int slen = strings[sid].len;
            int addlen = (int)peek()->val;
            if (slen + addlen < 255) {
                memcpy(strings[sid].data + slen, peek()->name, addlen);
                strings[sid].data[slen + addlen] = '\0';
                strings[sid].len = slen + addlen;
                /* Update data_pos if needed */
                int new_size = (strings[sid].len + 1 + 7) & ~7;
                int old_size = (slen + 1 + 7) & ~7;
                data_pos += new_size - old_size;
            }
            advance();
        }

        int r = alloc_reg();
        emit_load_global_addr(r, strings[sid].data_offset);
        expr_type = type_ptr(ty_char);
        return r;
    }

    /* sizeof */
    if (match(T_SIZEOF)) {
        expect(T_LPAREN);
        int t;
        /* Check if it's a type */
        if (is_type_start()) {
            t = parse_type();
            while (match(T_STAR)) t = type_ptr(t);
        } else {
            /* sizeof(expr) — evaluate type but discard */
            int r = parse_assign();
            t = expr_type;
            free_reg(r);
        }
        expect(T_RPAREN);
        int r = alloc_reg();
        emit_li(r, type_size(t));
        expr_type = ty_long;
        return r;
    }

    /* Parenthesized expression or cast */
    if (match(T_LPAREN)) {
        /* Check for cast: (type)expr */
        if (is_type_start()) {
            int t = parse_type();
            while (match(T_STAR)) t = type_ptr(t);
            expect(T_RPAREN);
            int r = parse_unary();  /* cast operand: unary (includes postfix) */
            expr_type = t;
            return r;
        }
        int r = parse_assign();
        expect(T_RPAREN);
        return r;
    }

    /* Identifier (variable or function call) */
    if (check(T_IDENT)) {
        char name[64];
        strcpy(name, peek()->name);
        advance();

        /* Function call? */
        if (match(T_LPAREN)) {
            /* Parse arguments */
            int args[8];
            int n_args = 0;

            if (!check(T_RPAREN)) {
                args[n_args] = parse_assign();
                n_args++;
                while (match(T_COMMA) && n_args < 8) {
                    args[n_args] = parse_assign();
                    n_args++;
                }
            }
            expect(T_RPAREN);

            /* __syscall(nr, a0, a1, a2, a3, a4) intrinsic */
            if (strcmp(name, "__syscall") == 0) {
                /* Move syscall number (arg0) to X8 */
                if (n_args > 0) emit_mov(8, args[0]);
                /* Move remaining args: arg1→X0, arg2→X1, etc. */
                for (int j = 1; j < n_args; j++) {
                    if (args[j] != j - 1) emit_mov(j - 1, args[j]);
                }
                for (int j = 0; j < n_args; j++) free_reg(args[j]);
                emit_svc();
                /* Result in X0 */
                int r = alloc_reg();
                if (r != 0) emit_mov(r, 0);
                expr_type = ty_long;
                return r;
            }

            /* Save caller-saved registers (x9-x18) that are in use */
            int n_save = 0;
            int save_reg[19];
            int save_off[19];
            for (int i = 9; i < REG_COUNT; i++) {
                if (reg_used[i]) {
                    save_off[n_save] = alloc_local(8);
                    emit_store_local(i, save_off[n_save], ty_long);
                    save_reg[n_save] = i;
                    n_save++;
                }
            }

            /* Find function */
            struct Symbol *fn = find_symbol(name);
            int is_funcptr = 0;
            if (!fn) {
                /* Implicit declaration */
                fn = add_symbol(name, ty_int, SYM_FUNC);
                fn->scope = 0;
            } else if (fn->kind != SYM_FUNC) {
                /* Calling through a function pointer variable */
                is_funcptr = 1;
            }

            if (is_funcptr) {
                /* Load the function pointer BEFORE moving args to x0-x7 */
                int fpr = alloc_reg();
                if (fn->kind == SYM_LOCAL || fn->kind == SYM_PARAM) {
                    emit_load_local(fpr, fn->offset, fn->type);
                } else if (fn->kind == SYM_GLOBAL) {
                    int ta = alloc_reg();
                    emit_load_global_addr(ta, fn->offset);
                    emit_load_indirect(fpr, ta, fn->type);
                    free_reg(ta);
                }
                /* Save funcptr to stack so it survives arg moves */
                int fpr_off = alloc_local(8);
                emit_store_local(fpr, fpr_off, ty_long);
                free_reg(fpr);

                /* Now move args to x0-x7 */
                for (int j = 0; j < n_args; j++) {
                    if (args[j] != j) emit_mov(j, args[j]);
                }
                for (int j = 0; j < n_args; j++) free_reg(args[j]);

                /* Reload function pointer into x18 and BLR */
                emit_load_local(18, fpr_off, ty_long);
                emit_blr(18);
            } else {
                /* Move args to x0-x7 */
                for (int j = 0; j < n_args; j++) {
                    if (args[j] != j) emit_mov(j, args[j]);
                }
                for (int j = 0; j < n_args; j++) free_reg(args[j]);

                /* BL to function */
                if (fn->kind == SYM_FUNC && fn->offset >= 0) {
                    emit_bl(fn->offset);
                } else {
                    int lbl = new_label();
                    fn->offset = lbl;
                    emit_bl(lbl);
                }
            }

            /* Result in x0 */
            int r = alloc_reg();
            if (r != 0) emit_mov(r, 0);

            /* Restore caller-saved registers */
            for (int i = 0; i < n_save; i++) {
                if (save_reg[i] != r) {
                    emit_load_local(save_reg[i], save_off[i], ty_long);
                }
            }

            expr_type = fn->type;
            return r;
        }

        /* Variable access */
        struct Symbol *sym = find_symbol(name);
        if (!sym) {
            error("undefined variable");
            int r = alloc_reg();
            emit_li(r, 0);
            expr_type = ty_int;
            return r;
        }

        /* Enum constant — load immediate value */
        if (sym->kind == SYM_ENUM) {
            int r = alloc_reg();
            emit_li(r, (long)sym->offset);
            expr_type = ty_int;
            return r;
        }

        /* Function pointer: function name without () — load address */
        if (sym->kind == SYM_FUNC && !check(T_LPAREN)) {
            int r = alloc_reg();
            int fn_lbl = sym->offset;
            /* Emit MOVZ+MOVK pair (will be fixed up to absolute address) */
            int movz_pos = code_pos;
            emit_movz(r, 0, 0);
            emit_movk(r, 0, 16);
            /* Add special fixup: type 3 = absolute address (MOVZ+MOVK pair) */
            if (n_fixups < MAX_FIXUPS) {
                fixups[n_fixups].code_offset = movz_pos;
                fixups[n_fixups].label = fn_lbl;
                fixups[n_fixups].type = 3; /* absolute address */
                n_fixups++;
            } else {
                error("fixup table overflow (funcptr)");
            }
            expr_type = type_ptr(ty_void); /* function pointer type */
            return r;
        }

        int r = alloc_reg();
        if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
            if (types[sym->type].kind == TY_ARRAY) {
                /* Array decays to pointer — load address */
                emit_add_imm(r, FP, 0);
                int off = sym->offset;
                if (off < 0) {
                    emit_li(r, (long)off);
                    emit_add(r, FP, r);
                } else {
                    emit_add_imm(r, FP, off);
                }
                expr_type = type_ptr(types[sym->type].ptr_to);
            } else if (types[sym->type].kind == TY_STRUCT) {
                /* Struct — load address for member access */
                int off = sym->offset;
                if (off < 0) {
                    emit_li(r, (long)off);
                    emit_add(r, FP, r);
                } else {
                    emit_add_imm(r, FP, off);
                }
                expr_type = sym->type;
            } else {
                emit_load_local(r, sym->offset, sym->type);
                expr_type = sym->type;
                last_lval_kind = sym->kind;
                last_lval_offset = sym->offset;
                last_lval_type = sym->type;
            }
        } else if (sym->kind == SYM_GLOBAL) {
            if (types[sym->type].kind == TY_ARRAY) {
                emit_load_global_addr(r, sym->offset);
                expr_type = type_ptr(types[sym->type].ptr_to);
            } else if (types[sym->type].kind == TY_STRUCT) {
                /* Struct global — load address */
                emit_load_global_addr(r, sym->offset);
                expr_type = sym->type;
            } else {
                int tmp = alloc_reg();
                emit_load_global_addr(tmp, sym->offset);
                emit_load_indirect(r, tmp, sym->type);
                free_reg(tmp);
                expr_type = sym->type;
                last_lval_kind = sym->kind;
                last_lval_offset = sym->offset;
                last_lval_type = sym->type;
            }
        }
        return r;
    }

    error("expected expression");
    return alloc_reg();
}

/* Parse postfix expressions: [], ., ->, ++, --, function calls */
static int parse_postfix(void) {
    int r = parse_primary();
    if (had_error) return r;

    for (;;) {
        /* Array subscript */
        if (match(T_LBRACKET)) {
            int saved_expr_type = expr_type;
            int idx = parse_assign();
            expr_type = saved_expr_type;
            expect(T_RBRACKET);

            /* Calculate address: base + idx * elem_size */
            int elem_type = ty_long;
            int elem_size = 8;
            if (type_is_ptr(expr_type)) {
                elem_type = types[expr_type].ptr_to;
                elem_size = type_size(elem_type);
            }

            if (elem_size != 1) {
                int tmp = alloc_reg();
                emit_li(tmp, elem_size);
                emit_mul(idx, idx, tmp);
                free_reg(tmp);
            }
            emit_add(r, r, idx);
            free_reg(idx);

            if (types[elem_type].kind == TY_STRUCT) {
                /* For struct elements, keep address for . or -> access */
                expr_type = elem_type;
            } else {
                /* Load scalar value */
                int rd = alloc_reg();
                emit_load_indirect(rd, r, elem_type);
                free_reg(r);
                r = rd;
                expr_type = elem_type;
            }
            continue;
        }

        /* Struct member access: . */
        if (match(T_DOT)) {
            if (!check(T_IDENT)) { error("expected field name"); break; }
            char fname[32];
            strncpy(fname, peek()->name, 31);
            fname[31] = '\0';
            advance();

            if (types[expr_type].kind != TY_STRUCT) {
                error("not a struct");
                break;
            }
            int sid = types[expr_type].struct_id;
            /* Find field */
            int found = 0;
            for (int i = 0; i < structs[sid].n_fields; i++) {
                if (!strcmp(structs[sid].fields[i].name, fname)) {
                    int off = structs[sid].fields[i].offset;
                    if (off > 0) emit_add_imm(r, r, off);
                    int rd = alloc_reg();
                    emit_load_indirect(rd, r, structs[sid].fields[i].type);
                    free_reg(r);
                    r = rd;
                    expr_type = structs[sid].fields[i].type;
                    found = 1;
                    break;
                }
            }
            if (!found) error("unknown field");
            continue;
        }

        /* Struct pointer member: -> */
        if (match(T_ARROW)) {
            if (!check(T_IDENT)) { error("expected field name"); break; }
            char fname[32];
            strncpy(fname, peek()->name, 31);
            fname[31] = '\0';
            advance();

            int base_type = expr_type;
            if (types[base_type].kind == TY_PTR) {
                base_type = types[base_type].ptr_to;
            }
            if (types[base_type].kind != TY_STRUCT) {
                error("not a struct pointer");
                break;
            }
            int sid = types[base_type].struct_id;
            int found = 0;
            for (int i = 0; i < structs[sid].n_fields; i++) {
                if (!strcmp(structs[sid].fields[i].name, fname)) {
                    int off = structs[sid].fields[i].offset;
                    if (off > 0) emit_add_imm(r, r, off);
                    int rd = alloc_reg();
                    emit_load_indirect(rd, r, structs[sid].fields[i].type);
                    free_reg(r);
                    r = rd;
                    expr_type = structs[sid].fields[i].type;
                    found = 1;
                    break;
                }
            }
            if (!found) error("unknown field");
            continue;
        }

        /* Post-increment */
        if (match(T_INC)) {
            /* Result is original value, but variable incremented */
            int rd = alloc_reg();
            emit_mov(rd, r);
            emit_add_imm(r, r, 1);
            /* Store incremented value back to variable */
            if (last_lval_kind == SYM_LOCAL || last_lval_kind == SYM_PARAM) {
                emit_store_local(r, last_lval_offset, last_lval_type);
            } else if (last_lval_kind == SYM_GLOBAL) {
                int ta = alloc_reg();
                emit_load_global_addr(ta, last_lval_offset);
                emit_store_indirect(r, ta, last_lval_type);
                free_reg(ta);
            }
            last_lval_kind = -1;
            free_reg(r);
            r = rd;
            continue;
        }

        /* Post-decrement */
        if (match(T_DEC)) {
            int rd = alloc_reg();
            emit_mov(rd, r);
            emit_sub_imm(r, r, 1);
            /* Store decremented value back to variable */
            if (last_lval_kind == SYM_LOCAL || last_lval_kind == SYM_PARAM) {
                emit_store_local(r, last_lval_offset, last_lval_type);
            } else if (last_lval_kind == SYM_GLOBAL) {
                int ta = alloc_reg();
                emit_load_global_addr(ta, last_lval_offset);
                emit_store_indirect(r, ta, last_lval_type);
                free_reg(ta);
            }
            last_lval_kind = -1;
            free_reg(r);
            r = rd;
            continue;
        }

        break;
    }

    return r;
}

/* Parse unary: -, ~, !, &, *, ++, --, (cast) */
static int parse_unary(void) {
    if (had_error) return 0;

    if (match(T_MINUS)) {
        int r = parse_unary();
        emit_neg(r, r);
        return r;
    }
    if (match(T_TILDE)) {
        int r = parse_unary();
        emit_mvn(r, r);
        return r;
    }
    if (match(T_BANG)) {
        int r = parse_unary();
        emit_cmp_imm(r, 0);
        emit_cset(r, COND_EQ);
        expr_type = ty_int;
        return r;
    }
    if (match(T_AMP)) {
        /* Address-of: &var */
        if (!check(T_IDENT)) { error("expected variable"); return alloc_reg(); }
        char name[64];
        strcpy(name, peek()->name);
        advance();

        struct Symbol *sym = find_symbol(name);
        if (!sym) { error("undefined variable"); return alloc_reg(); }

        int r = alloc_reg();
        if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
            int off = sym->offset;
            emit_li(r, (long)off);
            emit_add(r, FP, r);
        } else if (sym->kind == SYM_GLOBAL) {
            emit_load_global_addr(r, sym->offset);
        }
        expr_type = type_ptr(sym->type);
        return r;
    }
    if (match(T_STAR)) {
        /* Dereference: *expr */
        int r = parse_unary();
        int base_type = expr_type;
        int elem_type = ty_long;
        if (types[base_type].kind == TY_PTR) {
            elem_type = types[base_type].ptr_to;
        }
        int rd = alloc_reg();
        emit_load_indirect(rd, r, elem_type);
        free_reg(r);
        expr_type = elem_type;
        return rd;
    }
    if (match(T_INC)) {
        /* Pre-increment: ++var — increment then return new value */
        int r = parse_unary();
        emit_add_imm(r, r, 1);
        /* Store back to variable */
        if (last_lval_kind == SYM_LOCAL || last_lval_kind == SYM_PARAM) {
            emit_store_local(r, last_lval_offset, last_lval_type);
        } else if (last_lval_kind == SYM_GLOBAL) {
            int ta = alloc_reg();
            emit_load_global_addr(ta, last_lval_offset);
            emit_store_indirect(r, ta, last_lval_type);
            free_reg(ta);
        }
        last_lval_kind = -1;
        return r;
    }
    if (match(T_DEC)) {
        /* Pre-decrement: --var — decrement then return new value */
        int r = parse_unary();
        emit_sub_imm(r, r, 1);
        /* Store back to variable */
        if (last_lval_kind == SYM_LOCAL || last_lval_kind == SYM_PARAM) {
            emit_store_local(r, last_lval_offset, last_lval_type);
        } else if (last_lval_kind == SYM_GLOBAL) {
            int ta = alloc_reg();
            emit_load_global_addr(ta, last_lval_offset);
            emit_store_indirect(r, ta, last_lval_type);
            free_reg(ta);
        }
        last_lval_kind = -1;
        return r;
    }

    return parse_postfix();
}

/* Binary operations with precedence climbing */
static int parse_binop(int min_prec) {
    int left = parse_unary();
    if (had_error) return left;

    for (;;) {
        int op = peek()->type;
        int prec = 0;

        /* Operator precedence */
        switch (op) {
            case T_OR:      prec = 1; break;
            case T_AND:     prec = 2; break;
            case T_PIPE:    prec = 3; break;
            case T_CARET:   prec = 4; break;
            case T_AMP:     prec = 5; break;
            case T_EQ: case T_NE: prec = 6; break;
            case T_LT: case T_GT: case T_LE: case T_GE: prec = 7; break;
            case T_SHL: case T_SHR: prec = 8; break;
            case T_PLUS: case T_MINUS: prec = 9; break;
            case T_STAR: case T_SLASH: case T_PERCENT: prec = 10; break;
            default: return left;
        }

        if (prec < min_prec) return left;
        advance();

        int left_type = expr_type;

        /* Short-circuit for && and || */
        if (op == T_AND) {
            int lbl_false = new_label();
            int lbl_end = new_label();
            emit_cmp_imm(left, 0);
            emit_bcond(COND_EQ, lbl_false);
            free_reg(left);

            int right = parse_binop(prec + 1);
            emit_cmp_imm(right, 0);
            emit_cset(right, COND_NE);
            emit_b(lbl_end);

            mark_label(lbl_false);
            emit_li(right, 0);

            mark_label(lbl_end);
            left = right;
            expr_type = ty_int;
            continue;
        }

        if (op == T_OR) {
            int lbl_true = new_label();
            int lbl_end = new_label();
            emit_cmp_imm(left, 0);
            emit_bcond(COND_NE, lbl_true);
            free_reg(left);

            int right = parse_binop(prec + 1);
            emit_cmp_imm(right, 0);
            emit_cset(right, COND_NE);
            emit_b(lbl_end);

            mark_label(lbl_true);
            emit_li(right, 1);

            mark_label(lbl_end);
            left = right;
            expr_type = ty_int;
            continue;
        }

        int right = parse_binop(prec + 1);

        /* Pointer arithmetic scaling */
        if (op == T_PLUS && type_is_ptr(left_type) && !type_is_ptr(expr_type)) {
            int elem_size = type_size(types[left_type].ptr_to);
            if (elem_size > 1) {
                int tmp = alloc_reg();
                emit_li(tmp, elem_size);
                emit_mul(right, right, tmp);
                free_reg(tmp);
            }
            expr_type = left_type;
        } else if (op == T_MINUS && type_is_ptr(left_type) && !type_is_ptr(expr_type)) {
            int elem_size = type_size(types[left_type].ptr_to);
            if (elem_size > 1) {
                int tmp = alloc_reg();
                emit_li(tmp, elem_size);
                emit_mul(right, right, tmp);
                free_reg(tmp);
            }
            expr_type = left_type;
        }

        switch (op) {
            case T_PLUS:  emit_add(left, left, right); break;
            case T_MINUS: emit_sub(left, left, right); break;
            case T_STAR:  emit_mul(left, left, right); break;
            case T_SLASH: emit_sdiv(left, left, right); break;
            case T_PERCENT: {
                /* a % b = a - (a/b)*b */
                int tmp = alloc_reg();
                emit_sdiv(tmp, left, right);
                emit_msub(left, tmp, right, left);
                free_reg(tmp);
                break;
            }
            case T_AMP:   emit_and(left, left, right); break;
            case T_PIPE:  emit_orr(left, left, right); break;
            case T_CARET: emit_eor(left, left, right); break;
            case T_SHL:   emit_lsl(left, left, right); break;
            case T_SHR:   emit_asr(left, left, right); break;
            case T_EQ:
                emit_cmp(left, right);
                emit_cset(left, COND_EQ);
                expr_type = ty_int;
                break;
            case T_NE:
                emit_cmp(left, right);
                emit_cset(left, COND_NE);
                expr_type = ty_int;
                break;
            case T_LT:
                emit_cmp(left, right);
                emit_cset(left, COND_LT);
                expr_type = ty_int;
                break;
            case T_GT:
                emit_cmp(left, right);
                emit_cset(left, COND_GT);
                expr_type = ty_int;
                break;
            case T_LE:
                emit_cmp(left, right);
                emit_cset(left, COND_LE);
                expr_type = ty_int;
                break;
            case T_GE:
                emit_cmp(left, right);
                emit_cset(left, COND_GE);
                expr_type = ty_int;
                break;
            default: break;
        }
        free_reg(right);
    }
}

/* Ternary: expr ? expr : expr */
static int parse_ternary(void) {
    int r = parse_binop(1);
    if (had_error) return r;

    if (match(T_QUESTION)) {
        int lbl_false = new_label();
        int lbl_end = new_label();

        emit_cmp_imm(r, 0);
        emit_bcond(COND_EQ, lbl_false);
        free_reg(r);

        int rtrue = parse_assign();
        int result = alloc_reg();
        emit_mov(result, rtrue);
        free_reg(rtrue);
        emit_b(lbl_end);

        expect(T_COLON);
        mark_label(lbl_false);
        int rfalse = parse_assign();
        emit_mov(result, rfalse);
        free_reg(rfalse);

        mark_label(lbl_end);
        return result;
    }

    return r;
}

/* Assignment: lvalue = rvalue */
static int parse_assign(void) {
    /* Save position for potential lvalue reparse */
    int save_pos = tok_pos;
    int r = parse_ternary();
    if (had_error) return r;

    int op = peek()->type;
    if (op == T_ASSIGN || op == T_PLUS_EQ || op == T_MINUS_EQ ||
        op == T_STAR_EQ || op == T_SLASH_EQ || op == T_PERCENT_EQ ||
        op == T_AMP_EQ || op == T_PIPE_EQ || op == T_CARET_EQ ||
        op == T_SHL_EQ || op == T_SHR_EQ) {

        advance();
        free_reg(r);

        /* Reparse as lvalue */
        int old_pos = tok_pos;
        tok_pos = save_pos;

        /* Check what kind of lvalue it is */
        if (check(T_IDENT)) {
            char name[64];
            strcpy(name, peek()->name);
            advance();

            struct Symbol *sym = find_symbol(name);
            if (!sym) { error("undefined variable"); tok_pos = old_pos; return alloc_reg(); }

            /* Check for array subscript */
            if (match(T_LBRACKET)) {
                int idx_r = parse_assign();
                expect(T_RBRACKET);
                tok_pos = old_pos;

                /* Get rvalue */
                int rval = parse_assign();

                /* Calculate address */
                int addr = alloc_reg();
                if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
                    emit_li(addr, (long)sym->offset);
                    emit_add(addr, FP, addr);
                } else {
                    emit_load_global_addr(addr, sym->offset);
                }

                int elem_type = types[sym->type].ptr_to;
                int elem_size = type_size(elem_type);
                if (elem_size > 1) {
                    int tmp = alloc_reg();
                    emit_li(tmp, elem_size);
                    emit_mul(idx_r, idx_r, tmp);
                    free_reg(tmp);
                }
                emit_add(addr, addr, idx_r);
                free_reg(idx_r);

                emit_store_indirect(rval, addr, elem_type);
                free_reg(addr);
                return rval;
            }

            /* Check for struct member assignment: p.x = val or pp->field = val */
            {
                int is_dot = match(T_DOT);
                int is_arrow = !is_dot && match(T_ARROW);
                if (is_dot || is_arrow) {
                    if (!check(T_IDENT)) { error("expected field name"); tok_pos = old_pos; return alloc_reg(); }
                    char fname[32];
                    strncpy(fname, peek()->name, 31);
                    fname[31] = '\0';
                    advance();

                    /* Compute base address of struct */
                    int addr = alloc_reg();
                    if (is_arrow) {
                        /* Arrow: load pointer value, then use as address */
                        if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
                            emit_load_local(addr, sym->offset, sym->type);
                        } else {
                            int ta = alloc_reg();
                            emit_load_global_addr(ta, sym->offset);
                            emit_load_indirect(addr, ta, sym->type);
                            free_reg(ta);
                        }
                    } else {
                        /* Dot: load struct address (FP + offset) */
                        if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
                            int off = sym->offset;
                            if (off < 0) {
                                emit_li(addr, (long)off);
                                emit_add(addr, FP, addr);
                            } else {
                                emit_add_imm(addr, FP, off);
                            }
                        } else {
                            emit_load_global_addr(addr, sym->offset);
                        }
                    }

                    /* Find struct type */
                    int struct_type = sym->type;
                    if (is_arrow && types[struct_type].kind == TY_PTR) {
                        struct_type = types[struct_type].ptr_to;
                    }
                    if (types[struct_type].kind != TY_STRUCT) {
                        error("not a struct");
                        tok_pos = old_pos;
                        free_reg(addr);
                        return alloc_reg();
                    }
                    int sid = types[struct_type].struct_id;

                    /* Find field */
                    int field_offset = -1;
                    int field_type = ty_int;
                    for (int i = 0; i < structs[sid].n_fields; i++) {
                        if (!strcmp(structs[sid].fields[i].name, fname)) {
                            field_offset = structs[sid].fields[i].offset;
                            field_type = structs[sid].fields[i].type;
                            break;
                        }
                    }
                    if (field_offset < 0) {
                        error("unknown field");
                        tok_pos = old_pos;
                        free_reg(addr);
                        return alloc_reg();
                    }

                    /* Add field offset */
                    if (field_offset > 0) {
                        emit_add_imm(addr, addr, field_offset);
                    }

                    /* Parse rvalue */
                    tok_pos = old_pos;
                    int rval = parse_assign();

                    /* Store to struct member address */
                    emit_store_indirect(rval, addr, field_type);
                    free_reg(addr);
                    return rval;
                }
            }

            /* Check for pointer deref on left side: *p = ... handled below */
            tok_pos = old_pos;

            /* Simple variable assignment */
            int rval = parse_assign();

            if (op == T_PLUS_EQ || op == T_MINUS_EQ || op == T_STAR_EQ ||
                op == T_SLASH_EQ || op == T_PERCENT_EQ) {
                int cur = alloc_reg();
                if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
                    emit_load_local(cur, sym->offset, sym->type);
                } else {
                    int ta = alloc_reg();
                    emit_load_global_addr(ta, sym->offset);
                    emit_load_indirect(cur, ta, sym->type);
                    free_reg(ta);
                }
                if (op == T_PLUS_EQ)    emit_add(rval, cur, rval);
                else if (op == T_MINUS_EQ) emit_sub(rval, cur, rval);
                else if (op == T_STAR_EQ)  emit_mul(rval, cur, rval);
                else if (op == T_SLASH_EQ) emit_sdiv(rval, cur, rval);
                else if (op == T_PERCENT_EQ) {
                    int tmp = alloc_reg();
                    emit_sdiv(tmp, cur, rval);
                    emit_msub(rval, tmp, rval, cur);
                    free_reg(tmp);
                }
                free_reg(cur);
            }

            if (sym->kind == SYM_LOCAL || sym->kind == SYM_PARAM) {
                emit_store_local(rval, sym->offset, sym->type);
            } else if (sym->kind == SYM_GLOBAL) {
                int ta = alloc_reg();
                emit_load_global_addr(ta, sym->offset);
                emit_store_indirect(rval, ta, sym->type);
                free_reg(ta);
            }
            return rval;
        }

        /* Pointer dereference assignment: *p = val */
        if (match(T_STAR)) {
            int addr_r = parse_unary();
            int deref_type = expr_type;
            if (types[deref_type].kind == TY_PTR) {
                deref_type = types[deref_type].ptr_to;
            }
            tok_pos = old_pos;
            int rval = parse_assign();
            emit_store_indirect(rval, addr_r, deref_type);
            free_reg(addr_r);
            return rval;
        }

        tok_pos = old_pos;
        return parse_assign();
    }

    return r;
}

/* Top-level expression */
static int parse_expr(void) {
    int r = parse_assign();
    while (match(T_COMMA)) {
        free_reg(r);
        r = parse_assign();
    }
    return r;
}

/* ======================================================================== */
/* STATEMENT PARSER                                                         */
/* ======================================================================== */

static void parse_stmt(void) {
    if (had_error) return;

    /* Empty statement */
    if (match(T_SEMICOLON)) return;

    /* Block */
    if (check(T_LBRACE)) {
        parse_block();
        return;
    }

    /* Return */
    if (match(T_RETURN)) {
        if (!check(T_SEMICOLON)) {
            int r = parse_expr();
            if (r != 0) emit_mov(0, r);
            free_reg(r);
        }
        emit_b(func_ret_label);
        expect(T_SEMICOLON);
        reset_regs();
        return;
    }

    /* If/else */
    if (match(T_IF)) {
        expect(T_LPAREN);
        int r = parse_expr();
        expect(T_RPAREN);

        int lbl_else = new_label();
        int lbl_end = new_label();

        emit_cmp_imm(r, 0);
        emit_bcond(COND_EQ, lbl_else);
        free_reg(r);
        reset_regs();

        parse_stmt();

        if (match(T_ELSE)) {
            emit_b(lbl_end);
            mark_label(lbl_else);
            parse_stmt();
            mark_label(lbl_end);
        } else {
            mark_label(lbl_else);
        }
        return;
    }

    /* While */
    if (match(T_WHILE)) {
        int lbl_top = new_label();
        int lbl_end = new_label();

        /* Push break/continue */
        break_labels[break_depth++] = lbl_end;
        cont_labels[cont_depth++] = lbl_top;

        mark_label(lbl_top);
        expect(T_LPAREN);
        int r = parse_expr();
        expect(T_RPAREN);

        emit_cmp_imm(r, 0);
        emit_bcond(COND_EQ, lbl_end);
        free_reg(r);
        reset_regs();

        parse_stmt();
        emit_b(lbl_top);
        mark_label(lbl_end);

        break_depth--;
        cont_depth--;
        return;
    }

    /* For */
    if (match(T_FOR)) {
        expect(T_LPAREN);

        enter_scope();

        /* Init */
        if (!check(T_SEMICOLON)) {
            /* Check for variable declaration */
            if (is_type_start()) {
                /* Variable declaration in for-init */
                int vt = parse_type();
                while (match(T_STAR)) vt = type_ptr(vt);
                if (check(T_IDENT)) {
                    char vname[64];
                    strcpy(vname, peek()->name);
                    advance();
                    int off = alloc_local(type_size(vt));
                    struct Symbol *vs = add_symbol(vname, vt, SYM_LOCAL);
                    vs->offset = off;
                    if (match(T_ASSIGN)) {
                        int rv = parse_assign();
                        emit_store_local(rv, off, vt);
                        free_reg(rv);
                    }
                }
                reset_regs();
            } else {
                int r = parse_expr();
                free_reg(r);
                reset_regs();
            }
        }
        expect(T_SEMICOLON);

        int lbl_top = new_label();
        int lbl_inc = new_label();
        int lbl_end = new_label();

        break_labels[break_depth++] = lbl_end;
        cont_labels[cont_depth++] = lbl_inc;

        mark_label(lbl_top);

        /* Condition */
        if (!check(T_SEMICOLON)) {
            int r = parse_expr();
            emit_cmp_imm(r, 0);
            emit_bcond(COND_EQ, lbl_end);
            free_reg(r);
            reset_regs();
        }
        expect(T_SEMICOLON);

        /* Skip increment for now (save position) */
        int inc_start = tok_pos;
        /* Skip to body by counting parens */
        int depth = 1;
        while (tok_pos < n_tokens && depth > 0) {
            if (check(T_RPAREN)) depth--;
            else if (check(T_LPAREN)) depth++;
            if (depth > 0) tok_pos++;
        }
        if (check(T_RPAREN)) tok_pos++;

        /* Body */
        parse_stmt();

        /* Increment */
        mark_label(lbl_inc);
        int inc_end = tok_pos;
        tok_pos = inc_start;
        if (!check(T_RPAREN)) {
            int r = parse_expr();
            free_reg(r);
            reset_regs();
        }
        tok_pos = inc_end;

        emit_b(lbl_top);
        mark_label(lbl_end);

        break_depth--;
        cont_depth--;
        leave_scope();
        return;
    }

    /* Do-while */
    if (match(T_DO)) {
        int lbl_top = new_label();
        int lbl_end = new_label();
        int lbl_cont = new_label();

        break_labels[break_depth++] = lbl_end;
        cont_labels[cont_depth++] = lbl_cont;

        mark_label(lbl_top);
        parse_stmt();

        mark_label(lbl_cont);
        expect(T_WHILE);
        expect(T_LPAREN);
        int r = parse_expr();
        expect(T_RPAREN);
        expect(T_SEMICOLON);

        emit_cmp_imm(r, 0);
        emit_bcond(COND_NE, lbl_top);
        free_reg(r);
        reset_regs();

        mark_label(lbl_end);
        break_depth--;
        cont_depth--;
        return;
    }

    /* Switch */
    if (match(T_SWITCH)) {
        expect(T_LPAREN);
        int sw_r = parse_expr();
        expect(T_RPAREN);

        /* Save switch value to a local variable */
        int sw_off = alloc_local(8);
        emit_store_local(sw_r, sw_off, ty_long);
        free_reg(sw_r);
        reset_regs();

        int lbl_end = new_label();
        break_labels[break_depth++] = lbl_end;

        expect(T_LBRACE);
        enter_scope();

        int next_case_lbl = -1; /* label for next case comparison */

        while (!check(T_RBRACE) && !check(T_EOF) && !had_error) {
            if (match(T_CASE)) {
                /* Emit label for this case's comparison */
                if (next_case_lbl >= 0) mark_label(next_case_lbl);
                next_case_lbl = new_label();

                /* Parse case value (constant) */
                int neg = 0;
                if (match(T_MINUS)) neg = 1;
                long case_val = 0;
                if (check(T_NUM)) { case_val = peek()->val; advance(); }
                else if (check(T_CHAR_LIT)) { case_val = peek()->val; advance(); }
                else if (check(T_IDENT)) {
                    /* Might be an enum constant */
                    struct Symbol *es = find_symbol(peek()->name);
                    if (es && es->kind == SYM_ENUM) case_val = es->offset;
                    advance();
                }
                if (neg) case_val = -case_val;
                expect(T_COLON);

                /* Compare switch value with case value */
                int tmp = alloc_reg();
                emit_load_local(tmp, sw_off, ty_long);
                int cv = alloc_reg();
                emit_li(cv, case_val);
                emit_cmp(tmp, cv);
                free_reg(tmp);
                free_reg(cv);
                emit_bcond(COND_NE, next_case_lbl);
                reset_regs();
                continue;
            }

            if (match(T_DEFAULT)) {
                if (next_case_lbl >= 0) mark_label(next_case_lbl);
                next_case_lbl = -1;
                expect(T_COLON);
                continue;
            }

            parse_stmt();
        }

        if (next_case_lbl >= 0) mark_label(next_case_lbl);
        expect(T_RBRACE);
        leave_scope();
        mark_label(lbl_end);
        break_depth--;
        return;
    }

    /* Break */
    if (match(T_BREAK)) {
        if (break_depth > 0) {
            emit_b(break_labels[break_depth - 1]);
        } else {
            error("break outside loop/switch");
        }
        expect(T_SEMICOLON);
        return;
    }

    /* Continue */
    if (match(T_CONTINUE)) {
        if (cont_depth > 0) {
            emit_b(cont_labels[cont_depth - 1]);
        } else {
            error("continue outside loop");
        }
        expect(T_SEMICOLON);
        return;
    }

    /* Variable declaration */
    if (is_type_start()) {
        int vt = parse_type();
        while (match(T_STAR)) vt = type_ptr(vt);

        /* Multiple declarations: int a, b, *c; */
        do {
            int cur_type = vt;
            while (match(T_STAR)) cur_type = type_ptr(cur_type);

            if (!check(T_IDENT)) { error("expected variable name"); return; }
            char vname[64];
            strcpy(vname, peek()->name);
            advance();

            /* Array? */
            if (match(T_LBRACKET)) {
                int arr_len = 0;
                if (check(T_NUM)) { arr_len = (int)peek()->val; advance(); }
                expect(T_RBRACKET);
                if (arr_len > 0) cur_type = type_array(cur_type, arr_len);
            }

            int off = alloc_local(type_size(cur_type));
            struct Symbol *vs = add_symbol(vname, cur_type, SYM_LOCAL);
            if (vs) vs->offset = off;

            /* Initializer */
            if (match(T_ASSIGN)) {
                /* Array initializer: int arr[] = {1,2,3} */
                if (check(T_LBRACE) && types[cur_type].kind == TY_ARRAY) {
                    advance();
                    int elem_type = types[cur_type].ptr_to;
                    int elem_size = type_size(elem_type);
                    int idx = 0;
                    while (!check(T_RBRACE) && !check(T_EOF)) {
                        int rv = parse_assign();
                        /* Store at base + idx * elem_size */
                        int addr = alloc_reg();
                        emit_li(addr, (long)off + idx * elem_size);
                        emit_add(addr, FP, addr);
                        emit_store_indirect(rv, addr, elem_type);
                        free_reg(addr);
                        free_reg(rv);
                        idx++;
                        if (!match(T_COMMA)) break;
                    }
                    expect(T_RBRACE);
                } else {
                    int rv = parse_assign();
                    emit_store_local(rv, off, cur_type);
                    free_reg(rv);
                }
            }
            reset_regs();
        } while (match(T_COMMA));

        expect(T_SEMICOLON);
        return;
    }

    /* Expression statement */
    {
        int r = parse_expr();
        free_reg(r);
        reset_regs();
        expect(T_SEMICOLON);
    }
}

static void parse_block(void) {
    expect(T_LBRACE);
    enter_scope();
    while (!check(T_RBRACE) && !check(T_EOF) && !had_error) {
        parse_stmt();
    }
    expect(T_RBRACE);
    leave_scope();
}

/* ======================================================================== */
/* TOP-LEVEL DECLARATIONS                                                   */
/* ======================================================================== */

static void parse_global_decl(void) {
    if (had_error) return;

    /* Skip stray semicolons */
    if (match(T_SEMICOLON)) return;

    /* typedef: typedef <type> <name>; */
    if (match(T_TYPEDEF)) {
        int bt = parse_type();
        while (match(T_STAR)) bt = type_ptr(bt);
        if (check(T_IDENT)) {
            char tname[64];
            strcpy(tname, peek()->name);
            advance();
            struct Symbol *ts = add_symbol(tname, bt, SYM_TYPEDEF);
            if (ts) ts->scope = 0;
        }
        expect(T_SEMICOLON);
        return;
    }

    /* Parse type and name */
    int rt = parse_type();
    while (match(T_STAR)) rt = type_ptr(rt);

    /* Standalone struct/enum definition: struct Foo { ... }; */
    if (match(T_SEMICOLON)) return;

    if (!check(T_IDENT)) { error("expected identifier"); return; }
    char name[64];
    strcpy(name, peek()->name);
    advance();

    /* Function definition/declaration */
    if (match(T_LPAREN)) {
        struct Symbol *fn = find_symbol(name);
        int lbl;

        if (fn && fn->kind == SYM_FUNC) {
            lbl = fn->offset;
            fn->type = rt;  /* Update return type (may have been implicit ty_int) */
        } else {
            lbl = new_label();
            fn = add_symbol(name, rt, SYM_FUNC);
            fn->offset = lbl;
            fn->scope = 0;
        }

        /* Parse parameters */
        enter_scope();
        int n_params = 0;
        int param_types[8];
        char param_names[512]; /* 8 * 64, use param_names + i*64 */

        if (!check(T_RPAREN)) {
            /* Check for void parameter */
            if (check(T_VOID) && tokens[tok_pos + 1].type == T_RPAREN) {
                advance();
            } else {
                do {
                    /* Handle ... (variadic) */
                    if (match(T_ELLIPSIS)) break;

                    int pt = parse_type();
                    while (match(T_STAR)) pt = type_ptr(pt);
                    param_types[n_params] = pt;

                    if (check(T_IDENT)) {
                        strcpy(param_names + n_params * 64, peek()->name);
                        advance();
                    } else {
                        *(param_names + n_params * 64) = '\0';
                    }

                    /* Array parameter decay: int arr[] → int *arr */
                    if (match(T_LBRACKET)) {
                        while (!check(T_RBRACKET) && !check(T_EOF)) advance();
                        expect(T_RBRACKET);
                        param_types[n_params] = type_ptr(pt);
                    }

                    n_params++;
                } while (match(T_COMMA) && n_params < 8);
            }
        }
        expect(T_RPAREN);

        fn->n_params = n_params;

        /* Declaration only? */
        if (match(T_SEMICOLON)) {
            leave_scope();
            return;
        }

        /* Function body */
        fn->defined = 1;
        mark_label(lbl);

        /* Stack frame calculation */
        stack_offset = 16;
        func_ret_label = new_label();

        /* Emit prologue (estimate 256 bytes for locals) */
        int prologue_pos = code_pos;
        emit_prologue(256);  /* placeholder — we'll fix up later */

        /* Add parameters as locals */
        for (int i = 0; i < n_params; i++) {
            if (*(param_names + i * 64)) {
                int off = alloc_local(8);
                struct Symbol *ps = add_symbol(param_names + i * 64, param_types[i], SYM_PARAM);
                if (ps) ps->offset = off;
                /* Store parameter register to stack */
                emit_store_local(i, off, param_types[i]);
            }
        }

        /* Parse function body */
        reset_regs();
        expect(T_LBRACE);
        while (!check(T_RBRACE) && !check(T_EOF) && !had_error) {
            parse_stmt();
        }
        expect(T_RBRACE);

        /* Emit epilogue */
        /* Default return 0 */
        emit_li(0, 0);
        emit_epilogue();

        /* Fix up prologue with actual stack size */
        int actual_stack = (stack_offset + 15) & ~15;
        func_stack_size = actual_stack;

        /* Rewrite prologue SUB SP, SP, #frame instruction */
        unsigned int sub_inst = 0xD1000000 | ((func_stack_size & 0xFFF) << 10) | (SP << 5) | SP;
        output[prologue_pos]     = sub_inst & 0xFF;
        output[prologue_pos + 1] = (sub_inst >> 8) & 0xFF;
        output[prologue_pos + 2] = (sub_inst >> 16) & 0xFF;
        output[prologue_pos + 3] = (sub_inst >> 24) & 0xFF;

        /* Also fix ADD SP, SP, #frame in epilogue */
        int ep_pos = label_pos[func_ret_label];
        if (ep_pos >= 0) {
            /* MOV SP, FP at ep_pos, LDP at ep_pos+4, ADD SP at ep_pos+8 */
            unsigned int add_inst = 0x91000000 | ((func_stack_size & 0xFFF) << 10) | (SP << 5) | SP;
            output[ep_pos + 8] = add_inst & 0xFF;
            output[ep_pos + 9] = (add_inst >> 8) & 0xFF;
            output[ep_pos + 10] = (add_inst >> 16) & 0xFF;
            output[ep_pos + 11] = (add_inst >> 24) & 0xFF;
        }

        leave_scope();
        reset_regs();
        return;
    }

    /* Global variable */
    if (match(T_LBRACKET)) {
        int arr_len = 0;
        if (check(T_NUM)) { arr_len = (int)peek()->val; advance(); }
        expect(T_RBRACKET);
        if (arr_len > 0) rt = type_array(rt, arr_len);
    }

    int off = data_pos;
    data_pos += (type_size(rt) + 7) & ~7;

    struct Symbol *gs = add_symbol(name, rt, SYM_GLOBAL);
    if (gs) gs->offset = off;

    /* Global initializer — write directly into init_data buffer */
    if (match(T_ASSIGN)) {
        if (check(T_LBRACE)) {
            /* Array initializer: { val, val, ... } */
            advance();
            int elem_type = types[rt].kind == TY_ARRAY ? types[rt].ptr_to : ty_int;
            int elem_size = type_size(elem_type);
            int idx = 0;
            while (!check(T_RBRACE) && !check(T_EOF)) {
                int neg = 0;
                if (match(T_MINUS)) neg = 1;
                long val = 0;
                if (check(T_NUM)) { val = peek()->val; advance(); }
                else if (check(T_CHAR_LIT)) { val = peek()->val; advance(); }
                if (neg) val = -val;
                /* Write value at off + idx*elem_size into init_data */
                int pos = off + idx * elem_size;
                if (pos + elem_size <= MAX_INIT_DATA) {
                    if (elem_size == 1) {
                        init_data[pos] = (unsigned char)(val & 0xFF);
                    } else if (elem_size == 4) {
                        init_data[pos]   = (unsigned char)(val & 0xFF);
                        init_data[pos+1] = (unsigned char)((val >> 8) & 0xFF);
                        init_data[pos+2] = (unsigned char)((val >> 16) & 0xFF);
                        init_data[pos+3] = (unsigned char)((val >> 24) & 0xFF);
                    } else {
                        for (int b = 0; b < 8; b++)
                            init_data[pos+b] = (unsigned char)((val >> (b*8)) & 0xFF);
                    }
                    init_data_used = 1;
                }
                idx++;
                if (!match(T_COMMA)) break;
            }
            expect(T_RBRACE);
        } else if (check(T_STR)) {
            /* String initializer: char *p = "hello" */
            int sid = add_string(peek()->name, (int)peek()->val);
            advance();
            /* Store pointer to string in data section */
            long str_addr = DATA_BASE + strings[sid].data_offset;
            if (off + 8 <= MAX_INIT_DATA) {
                for (int b = 0; b < 8; b++)
                    init_data[off+b] = (unsigned char)((str_addr >> (b*8)) & 0xFF);
                init_data_used = 1;
            }
        } else {
            /* Simple constant: int x = 42; or int x = -1; */
            int neg = 0;
            if (match(T_MINUS)) neg = 1;
            long val = 0;
            if (check(T_NUM)) { val = peek()->val; advance(); }
            else if (check(T_CHAR_LIT)) { val = peek()->val; advance(); }
            else if (check(T_IDENT)) {
                /* Might be an enum constant */
                struct Symbol *es = find_symbol(peek()->name);
                if (es && es->kind == SYM_ENUM) val = es->offset;
                advance();
            }
            if (neg) val = -val;
            int sz = type_size(rt);
            if (off + sz <= MAX_INIT_DATA) {
                if (sz == 1) {
                    init_data[off] = (unsigned char)(val & 0xFF);
                } else if (sz == 4) {
                    init_data[off]   = (unsigned char)(val & 0xFF);
                    init_data[off+1] = (unsigned char)((val >> 8) & 0xFF);
                    init_data[off+2] = (unsigned char)((val >> 16) & 0xFF);
                    init_data[off+3] = (unsigned char)((val >> 24) & 0xFF);
                } else {
                    for (int b = 0; b < 8; b++)
                        init_data[off+b] = (unsigned char)((val >> (b*8)) & 0xFF);
                }
                init_data_used = 1;
            }
        }
    }

    expect(T_SEMICOLON);
}

/* ======================================================================== */
/* STARTUP CODE GENERATION                                                  */
/* ======================================================================== */

static void emit_startup(void) {
    /* _start: set up stack, call main, exit */

    /* Set SP = STACK_TOP.
     * Cannot use emit_li(SP, ...) because MOVZ/MOVK treat reg 31 as XZR.
     * Load into X0 first, then MOV SP, X0 (which uses ADD SP, X0, #0). */
    emit_li(0, STACK_TOP);
    emit_mov(SP, 0);

    /* BL main */
    struct Symbol *main_fn = find_symbol("main");
    if (main_fn && main_fn->kind == SYM_FUNC) {
        emit_bl(main_fn->offset);
    } else {
        error("no main function defined");
        return;
    }

    /* SYS_EXIT: mov x8, #93; svc #0 */
    emit_li(8, 93);
    emit_svc();
}

/* ======================================================================== */
/* DATA SECTION EMISSION                                                    */
/* ======================================================================== */

static void emit_data_section(unsigned char *out, int max_size) {
    /* Write initialized global data first */
    if (init_data_used) {
        int copy_len = data_pos < max_size ? data_pos : max_size;
        if (copy_len > MAX_INIT_DATA) copy_len = MAX_INIT_DATA;
        memcpy(out, init_data, copy_len);
    }
    /* Write string literals to data section (overwrites zero-init areas) */
    for (int i = 0; i < n_strings; i++) {
        int off = strings[i].data_offset;
        if (off + strings[i].len + 1 <= max_size) {
            memcpy(out + off, strings[i].data, strings[i].len);
            out[off + strings[i].len] = '\0';
        }
    }
}

/* ======================================================================== */
/* MAIN: COMPILE AND OUTPUT                                                 */
/* ======================================================================== */

static void init_types(void) {
    n_types = 0;
    ty_void = type_new(TY_VOID, 0);
    ty_char = type_new(TY_CHAR, 1);
    ty_int  = type_new(TY_INT, 4);
    ty_long = type_new(TY_LONG, 8);
}

static void compile(void) {
    /* Initialize */
    init_types();
    n_symbols = 0;
    n_strings = 0;
    n_structs = 0;
    n_defines = 0;
    /* Pre-define __CCGPU__ so compiled programs can detect this compiler */
    strcpy(defines[n_defines].name, "__CCGPU__");
    strcpy(defines[n_defines].value, "1");
    n_defines++;
    scope_depth = 0;
    code_pos = 0;
    data_pos = 0;
    label_counter = 0;
    n_fixups = 0;
    had_error = 0;
    break_depth = 0;
    last_lval_kind = -1;
    cont_depth = 0;
    ifdef_depth = 0;
    ifdef_active = 1;
    init_data_used = 0;
    memset(init_data, 0, MAX_INIT_DATA);
    memset(output, 0, MAX_OUTPUT);
    memset(label_pos, 0xFF, sizeof(label_pos));  /* -1 = undefined */

    /* Lex */
    lex();
    tok_pos = 0;

    printf("[cc] Lexed %d tokens\n", n_tokens);

    /* Emit startup code first (BL main will be a forward fixup) */
    int main_label = new_label();

    /* MOV SP, #STACK_TOP using ADD (SP needs ADD encoding, not ORR) */
    emit_li(0, STACK_TOP);   /* load stack top into x0 */
    emit_mov(SP, 0);         /* MOV SP, x0 via ADD */

    /* BL main (forward reference — resolved later) */
    emit_bl(main_label);

    /* SYS_EXIT: mov x8, #93; svc #0 */
    emit_li(8, 93);
    emit_svc();

    /* Parse all top-level declarations */
    while (!check(T_EOF) && !had_error) {
        parse_global_decl();
    }

    if (had_error) return;

    /* If main was found, its label should already be marked */
    {
        struct Symbol *main_fn = find_symbol("main");
        if (main_fn && main_fn->kind == SYM_FUNC) {
            /* Copy the actual label position to main_label */
            if (main_fn->offset < MAX_LABELS && label_pos[main_fn->offset] >= 0) {
                label_pos[main_label] = label_pos[main_fn->offset];
            } else {
                error("main function not defined");
            }
        } else {
            error("no main function defined");
        }
    }

    /* Resolve all branch fixups */
    resolve_fixups();

    printf("[cc] Generated %d bytes of code\n", code_pos);
    printf("[cc] Data section: %d bytes (%d strings)\n", data_pos, n_strings);
    printf("[cc] Symbols: %d, Labels: %d, Fixups: %d\n", n_symbols, label_counter, n_fixups);
}

/*
 * Read compile arguments from /tmp/.cc_args file:
 *   Line 1: source file path
 *   Line 2: output file path (optional, derived from source if absent)
 *
 * Falls back to /tmp/input.c -> /bin/a.out if args file doesn't exist.
 */
static char arg_src_buf[128];
static char arg_out_buf[128];

static void read_args(void) {
    int fd = open("/tmp/.cc_args", O_RDONLY);
    if (fd < 0) {
        strcpy(arg_src_buf, "/tmp/input.c");
        strcpy(arg_out_buf, "/bin/a.out");
        return;
    }
    char buf[256];
    int n = (int)read(fd, buf, 255);
    close(fd);
    if (n <= 0) {
        strcpy(arg_src_buf, "/tmp/input.c");
        strcpy(arg_out_buf, "/bin/a.out");
        return;
    }
    buf[n] = '\0';

    /* Parse line 1: source path */
    int i = 0, j = 0;
    while (i < n && buf[i] != '\n' && buf[i] != '\0' && j < 127)
        arg_src_buf[j++] = buf[i++];
    arg_src_buf[j] = '\0';
    if (i < n && buf[i] == '\n') i++;

    /* Parse line 2: output path */
    j = 0;
    while (i < n && buf[i] != '\n' && buf[i] != '\0' && j < 127)
        arg_out_buf[j++] = buf[i++];
    arg_out_buf[j] = '\0';

    /* Derive output path if not given */
    if (arg_out_buf[0] == '\0') {
        const char *base = strrchr(arg_src_buf, '/');
        base = base ? base + 1 : arg_src_buf;
        strcpy(arg_out_buf, "/bin/");
        j = 5;
        while (*base && *base != '.' && j < 126)
            arg_out_buf[j++] = *base++;
        arg_out_buf[j] = '\0';
    }
}

int main(void) {
    printf("nCPU C Compiler v1.0 (self-hosting, ARM64 Metal GPU)\n");
    printf("====================================================\n");

    read_args();

    const char *src_path = arg_src_buf;
    const char *out_path = arg_out_buf;

    printf("[cc] Source: %s\n", src_path);
    printf("[cc] Output: %s\n", out_path);

    /* Read source file */
    int fd = open(src_path, O_RDONLY);
    if (fd < 0) {
        printf("[cc] ERROR: Cannot open source file: %s\n", src_path);
        return 1;
    }

    src_len = (int)read(fd, source, MAX_SOURCE - 1);
    close(fd);

    if (src_len <= 0) {
        printf("[cc] ERROR: Empty source file\n");
        return 1;
    }
    source[src_len] = '\0';
    printf("[cc] Read %d bytes of source\n", src_len);

    /* Compile */
    compile();

    if (had_error) {
        printf("[cc] COMPILATION FAILED (line %d)\n", error_line);
        return 1;
    }

    /* Build output binary */
    /* Layout: code at offset 0, data section follows */
    /* The code_pos bytes go to .text (loaded at CODE_BASE) */
    /* The data section goes to .data (loaded at DATA_BASE) */

    /* For simplicity, write code section as the binary */
    /* The data section is written separately or appended */

    /* Write code section */
    int out_fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC);
    if (out_fd < 0) {
        printf("[cc] ERROR: Cannot create output file: %s\n", out_path);
        return 1;
    }

    /* Write code */
    write(out_fd, output, code_pos);

    /* Write data section immediately after code (compact binary) */
    /* The runner loads code at CODE_BASE and data at DATA_BASE separately */
    if (data_pos > 0) {
        /* Write 8-byte header: magic 'NCCD' + data_pos */
        unsigned char hdr[8];
        hdr[0] = 'N'; hdr[1] = 'C'; hdr[2] = 'C'; hdr[3] = 'D';
        hdr[4] = (data_pos >>  0) & 0xFF;
        hdr[5] = (data_pos >>  8) & 0xFF;
        hdr[6] = (data_pos >> 16) & 0xFF;
        hdr[7] = (data_pos >> 24) & 0xFF;
        write(out_fd, hdr, 8);

        /* Build data section in unused high memory (0x800000 = 8MB mark,
         * well above BSS and below stack at 0xF00000), then write all
         * at once. This avoids hundreds of small write SVCs — each SVC
         * round-trips 16MB through Metal, so one big write is much faster
         * than hundreds of 4KB writes. */
        {
            unsigned char *dsec = (unsigned char *)0x800000;
            int dsz = (data_pos + 7) & ~7;
            memset(dsec, 0, dsz);
            emit_data_section(dsec, dsz);
            write(out_fd, dsec, dsz);
        }
    }

    close(out_fd);

    int total_size = code_pos;
    if (data_pos > 0) total_size = code_pos + 8 + data_pos;

    printf("[cc] SUCCESS: %s (%d bytes code + %d bytes data)\n",
           out_path, code_pos, data_pos);
    printf("[cc] Binary ready for execution\n");

    return 0;
}
