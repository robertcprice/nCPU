/*
 * NEURAL RTOS - ARM64 Real-Time Operating System
 * ===============================================
 *
 * This ENTIRE OS runs as ARM64 machine code on the Neural CPU.
 * Every instruction is executed through neural network forward passes.
 *
 * Memory Map:
 *   0x10000 - 0x1FFFF: RTOS Kernel Code
 *   0x20000 - 0x2FFFF: Kernel Data
 *   0x30000 - 0x3FFFF: File System
 *   0x40000 - 0x4FFFF: Framebuffer (80x25 terminal)
 *   0x50000 - 0x500FF: Keyboard Input Buffer
 *   0x60000 - 0x6FFFF: User Programs
 *   0x70000 - 0x7FFFF: Stack
 */

/* ============================================================
 * MEMORY MAPPED I/O ADDRESSES
 * ============================================================ */

#define FB_BASE         ((volatile char*)0x40000)
#define FB_WIDTH        80
#define FB_HEIGHT       25

#define KEY_BUFFER      ((volatile char*)0x50000)
#define KEY_STATUS      ((volatile int*)0x50004)

#define FS_BASE         ((volatile char*)0x30000)
#define FS_SIZE         0x10000

/* ============================================================
 * TERMINAL OUTPUT
 * ============================================================ */

static int cursor_x = 0;
static int cursor_y = 0;

void fb_clear(void) {
    for (int i = 0; i < FB_WIDTH * FB_HEIGHT; i++) {
        FB_BASE[i] = ' ';
    }
    cursor_x = 0;
    cursor_y = 0;
}

void fb_scroll(void) {
    // Move all lines up
    for (int y = 0; y < FB_HEIGHT - 1; y++) {
        for (int x = 0; x < FB_WIDTH; x++) {
            FB_BASE[y * FB_WIDTH + x] = FB_BASE[(y + 1) * FB_WIDTH + x];
        }
    }
    // Clear last line
    for (int x = 0; x < FB_WIDTH; x++) {
        FB_BASE[(FB_HEIGHT - 1) * FB_WIDTH + x] = ' ';
    }
    cursor_y = FB_HEIGHT - 1;
}

void fb_putchar(char c) {
    if (c == '\n') {
        cursor_x = 0;
        cursor_y++;
        if (cursor_y >= FB_HEIGHT) {
            fb_scroll();
        }
        return;
    }

    if (c == '\r') {
        cursor_x = 0;
        return;
    }

    if (cursor_x >= FB_WIDTH) {
        cursor_x = 0;
        cursor_y++;
        if (cursor_y >= FB_HEIGHT) {
            fb_scroll();
        }
    }

    FB_BASE[cursor_y * FB_WIDTH + cursor_x] = c;
    cursor_x++;
}

void fb_print(const char* str) {
    while (*str) {
        fb_putchar(*str++);
    }
}

void fb_print_int(int n) {
    if (n < 0) {
        fb_putchar('-');
        n = -n;
    }
    if (n == 0) {
        fb_putchar('0');
        return;
    }

    char buf[12];
    int i = 0;
    while (n > 0) {
        buf[i++] = '0' + (n % 10);
        n /= 10;
    }
    while (i > 0) {
        fb_putchar(buf[--i]);
    }
}

void fb_print_hex(unsigned int n) {
    fb_print("0x");
    char hex[] = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4) {
        fb_putchar(hex[(n >> i) & 0xF]);
    }
}

/* ============================================================
 * KEYBOARD INPUT
 * ============================================================ */

char kb_getchar(void) {
    char c = *KEY_BUFFER;
    *KEY_BUFFER = 0;  // Clear after read
    return c;
}

int kb_readline(char* buf, int max_len) {
    int len = 0;
    fb_putchar('>');
    fb_putchar(' ');

    while (len < max_len - 1) {
        char c = kb_getchar();
        if (c == 0) continue;  // No key pressed

        if (c == '\n' || c == '\r') {
            buf[len] = 0;
            fb_putchar('\n');
            return len;
        }

        if (c == 127 || c == 8) {  // Backspace
            if (len > 0) {
                len--;
                cursor_x--;
                fb_putchar(' ');
                cursor_x--;
            }
            continue;
        }

        buf[len++] = c;
        fb_putchar(c);
    }
    buf[len] = 0;
    return len;
}

/* ============================================================
 * STRING UTILITIES
 * ============================================================ */

int str_len(const char* s) {
    int len = 0;
    while (*s++) len++;
    return len;
}

int str_cmp(const char* a, const char* b) {
    while (*a && *b && *a == *b) {
        a++; b++;
    }
    return *a - *b;
}

int str_ncmp(const char* a, const char* b, int n) {
    while (n > 0 && *a && *b && *a == *b) {
        a++; b++; n--;
    }
    return n == 0 ? 0 : *a - *b;
}

void str_cpy(char* dst, const char* src) {
    while (*src) *dst++ = *src++;
    *dst = 0;
}

/* ============================================================
 * SIMPLE FILE SYSTEM (In Neural Memory)
 * ============================================================ */

#define MAX_FILES 16
#define MAX_FILENAME 16
#define MAX_FILESIZE 1024

typedef struct {
    char name[MAX_FILENAME];
    int size;
    int offset;  // Offset into FS_BASE
    int used;
} FileEntry;

static FileEntry files[MAX_FILES];
static int fs_next_offset = 0;

void fs_init(void) {
    for (int i = 0; i < MAX_FILES; i++) {
        files[i].used = 0;
    }
    fs_next_offset = 0;

    // Create default welcome file
    const char* welcome = "Welcome to Neural RTOS!\n\nThis OS runs entirely on the Neural CPU.\nEvery instruction is a neural network forward pass.\n\nCommands: help, ls, cat, echo, calc, mem, regs\n";

    str_cpy(files[0].name, "welcome.txt");
    files[0].size = str_len(welcome);
    files[0].offset = 0;
    files[0].used = 1;

    for (int i = 0; welcome[i]; i++) {
        FS_BASE[i] = welcome[i];
    }
    fs_next_offset = files[0].size + 1;
}

int fs_find(const char* name) {
    for (int i = 0; i < MAX_FILES; i++) {
        if (files[i].used && str_cmp(files[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

int fs_create(const char* name, const char* content) {
    int idx = -1;
    for (int i = 0; i < MAX_FILES; i++) {
        if (!files[i].used) {
            idx = i;
            break;
        }
    }
    if (idx < 0) return -1;

    int len = str_len(content);
    if (fs_next_offset + len >= FS_SIZE) return -1;

    str_cpy(files[idx].name, name);
    files[idx].size = len;
    files[idx].offset = fs_next_offset;
    files[idx].used = 1;

    for (int i = 0; i < len; i++) {
        FS_BASE[fs_next_offset + i] = content[i];
    }
    fs_next_offset += len + 1;

    return idx;
}

void fs_cat(const char* name) {
    int idx = fs_find(name);
    if (idx < 0) {
        fb_print("File not found: ");
        fb_print(name);
        fb_putchar('\n');
        return;
    }

    for (int i = 0; i < files[idx].size; i++) {
        fb_putchar(FS_BASE[files[idx].offset + i]);
    }
}

void fs_list(void) {
    fb_print("Files:\n");
    for (int i = 0; i < MAX_FILES; i++) {
        if (files[i].used) {
            fb_print("  ");
            fb_print(files[i].name);
            fb_print(" (");
            fb_print_int(files[i].size);
            fb_print(" bytes)\n");
        }
    }
}

/* ============================================================
 * SHELL COMMANDS
 * ============================================================ */

// Neural ALU operations counter
static int neural_ops = 0;
static int total_instructions = 0;

void cmd_help(void) {
    fb_print("\n");
    fb_print("=== NEURAL RTOS SHELL ===\n");
    fb_print("Commands:\n");
    fb_print("  help      - Show this help\n");
    fb_print("  ls        - List files\n");
    fb_print("  cat <f>   - Show file contents\n");
    fb_print("  echo <t>  - Print text\n");
    fb_print("  calc      - Neural calculator\n");
    fb_print("  mem       - Memory info\n");
    fb_print("  regs      - Register state\n");
    fb_print("  clear     - Clear screen\n");
    fb_print("  info      - System info\n");
    fb_print("\n");
}

void cmd_calc(int a, char op, int b) {
    int result = 0;

    // These operations go through the neural ALU!
    switch (op) {
        case '+': result = a + b; neural_ops++; break;
        case '-': result = a - b; neural_ops++; break;
        case '*': result = a * b; neural_ops++; break;
        case '&': result = a & b; neural_ops++; break;
        case '|': result = a | b; neural_ops++; break;
        case '^': result = a ^ b; neural_ops++; break;
        default:
            fb_print("Unknown operator\n");
            return;
    }

    fb_print_int(a);
    fb_putchar(' ');
    fb_putchar(op);
    fb_putchar(' ');
    fb_print_int(b);
    fb_print(" = ");
    fb_print_int(result);
    fb_print(" (neural)\n");
}

void cmd_mem(void) {
    fb_print("\nMemory Map:\n");
    fb_print("  Kernel:  0x10000 - 0x1FFFF\n");
    fb_print("  Data:    0x20000 - 0x2FFFF\n");
    fb_print("  FS:      0x30000 - 0x3FFFF\n");
    fb_print("  FB:      0x40000 - 0x4FFFF\n");
    fb_print("  Kbd:     0x50000 - 0x500FF\n");
    fb_print("  User:    0x60000 - 0x6FFFF\n");
    fb_print("  Stack:   0x70000 - 0x7FFFF\n");
    fb_print("\n");
}

void cmd_info(void) {
    fb_print("\n=== NEURAL RTOS ===\n");
    fb_print("Version: 1.0.0 Synapse\n");
    fb_print("Arch: ARM64 (AArch64)\n");
    fb_print("CPU: Neural Transformer\n");
    fb_print("Neural Ops: ");
    fb_print_int(neural_ops);
    fb_putchar('\n');
    fb_print("Every instruction is neural!\n");
    fb_print("\n");
}

/* ============================================================
 * SHELL PARSER
 * ============================================================ */

int parse_int(const char* s) {
    int n = 0;
    int neg = 0;

    if (*s == '-') {
        neg = 1;
        s++;
    }

    // Check for hex
    if (s[0] == '0' && s[1] == 'x') {
        s += 2;
        while (*s) {
            n *= 16;
            if (*s >= '0' && *s <= '9') n += *s - '0';
            else if (*s >= 'a' && *s <= 'f') n += *s - 'a' + 10;
            else if (*s >= 'A' && *s <= 'F') n += *s - 'A' + 10;
            s++;
        }
    } else {
        while (*s >= '0' && *s <= '9') {
            n = n * 10 + (*s - '0');
            s++;
        }
    }

    return neg ? -n : n;
}

void shell_exec(char* cmd) {
    // Skip leading spaces
    while (*cmd == ' ') cmd++;

    if (str_ncmp(cmd, "help", 4) == 0) {
        cmd_help();
    }
    else if (str_ncmp(cmd, "ls", 2) == 0) {
        fs_list();
    }
    else if (str_ncmp(cmd, "cat ", 4) == 0) {
        fs_cat(cmd + 4);
    }
    else if (str_ncmp(cmd, "echo ", 5) == 0) {
        fb_print(cmd + 5);
        fb_putchar('\n');
    }
    else if (str_ncmp(cmd, "calc ", 5) == 0) {
        // Parse: calc 10 + 20
        char* p = cmd + 5;
        while (*p == ' ') p++;
        int a = parse_int(p);
        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;
        char op = *p++;
        while (*p == ' ') p++;
        int b = parse_int(p);
        cmd_calc(a, op, b);
    }
    else if (str_ncmp(cmd, "mem", 3) == 0) {
        cmd_mem();
    }
    else if (str_ncmp(cmd, "info", 4) == 0) {
        cmd_info();
    }
    else if (str_ncmp(cmd, "clear", 5) == 0) {
        fb_clear();
    }
    else if (cmd[0] != 0) {
        fb_print("Unknown: ");
        fb_print(cmd);
        fb_putchar('\n');
    }
}

/* ============================================================
 * BOOT & MAIN
 * ============================================================ */

void show_banner(void) {
    fb_clear();
    fb_print("\n");
    fb_print("  _   _ _____ _   _ ____      _    _       ____  _____ ___  ____\n");
    fb_print(" | \\ | | ____| | | |  _ \\    / \\  | |     |  _ \\|_   _/ _ \\/ ___|\n");
    fb_print(" |  \\| |  _| | | | | |_) |  / _ \\ | |     | |_) | | || | | \\___ \\\n");
    fb_print(" | |\\  | |___| |_| |  _ <  / ___ \\| |___  |  _ <  | || |_| |___) |\n");
    fb_print(" |_| \\_|_____|\\___/|_| \\_\\/_/   \\_\\_____| |_| \\_\\ |_| \\___/|____/\n");
    fb_print("\n");
    fb_print("        ARM64 Real-Time Operating System on Neural CPU\n");
    fb_print("        ================================================\n");
    fb_print("\n");
    fb_print("  Every instruction executed through neural networks.\n");
    fb_print("  Type 'help' for commands.\n");
    fb_print("\n");
}

void _start(void) {
    // Initialize subsystems
    fb_clear();
    fs_init();

    // Show boot banner
    show_banner();

    // Command buffer
    char cmd[64];

    // Main shell loop - runs entirely on neural CPU!
    while (1) {
        // Read command
        kb_readline(cmd, 64);

        // Execute
        shell_exec(cmd);

        // Increment instruction counter (neural CPU tracks this)
        total_instructions++;
    }
}
