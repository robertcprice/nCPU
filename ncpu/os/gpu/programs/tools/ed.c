/*
 * ed -- Line editor for freestanding ARM64 on Metal GPU.
 *
 * A minimal ed(1)-style line editor running entirely on a Metal compute
 * shader via the ARM64 GPU kernel. No stdlib, no heap, no floating point.
 * All I/O routes through SVC syscalls to the Python syscall handler.
 *
 * Supported commands:
 *   a           Append text after current line (terminate with ".")
 *   i           Insert text before current line (terminate with ".")
 *   d           Delete current line
 *   p           Print current line
 *   n           Print current line with line number
 *   ,p          Print all lines
 *   ,n          Print all lines with line numbers
 *   NUMBER      Set current line to NUMBER and print it
 *   NUMBERcmd   Address line NUMBER then execute cmd (e.g. 3d, 5p)
 *   w [file]    Write buffer to file, print byte count
 *   e [file]    Load file into buffer (replaces contents)
 *   s/old/new/  Substitute first occurrence on current line
 *   j           Join current line with next line
 *   =           Print current line number
 *   $           Go to last line and print it
 *   +/-         Move to next/previous line and print it
 *   q           Quit (warns once if unsaved changes)
 *   Q           Quit unconditionally
 *   h           Print command summary
 *
 * Buffer: 1000 lines x 256 chars, statically allocated.
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/tools/ed.c
 *          -o /tmp/ed.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

#define MAX_LINES 1000
#define LINE_LEN  256
#define CMD_LEN   512
#define FNAME_LEN 256

static char buf[MAX_LINES][LINE_LEN];  /* text buffer                      */
static int  nlines;                    /* number of lines in buffer         */
static int  cur;                       /* current line number (1-indexed)   */
static int  dirty;                     /* buffer modified since last write  */
static char filename[FNAME_LEN];      /* remembered filename for w/e       */
static char cmd[CMD_LEN];             /* command input scratch             */
static char iline[LINE_LEN];          /* text input scratch                */

/* ---- helpers --------------------------------------------------------- */

/* Read one line from stdin. Strip trailing CR/LF. Return length or -1. */
static int read_line(char *dst, int max) {
    ssize_t n = sys_read(0, dst, max - 1);
    if (n <= 0) return -1;
    dst[n] = '\0';
    while (n > 0 && (dst[n-1] == '\n' || dst[n-1] == '\r'))
        dst[--n] = '\0';
    return (int)n;
}

static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* Parse an unsigned decimal integer from s starting at *p. */
static int parse_int(const char *s, int *p) {
    int start = *p, val = 0;
    while (is_digit(s[*p]))
        val = val * 10 + (s[(*p)++] - '0');
    return (*p == start) ? -1 : val;
}

/* Insert a blank line at position pos (0-indexed), shifting others down. */
static int shift_down(int pos) {
    if (nlines >= MAX_LINES) return -1;
    for (int i = nlines; i > pos; i--)
        memcpy(buf[i], buf[i-1], LINE_LEN);
    buf[pos][0] = '\0';
    nlines++;
    return 0;
}

/* Remove line at position pos (0-indexed), shifting others up. */
static void shift_up(int pos) {
    for (int i = pos; i < nlines - 1; i++)
        memcpy(buf[i], buf[i+1], LINE_LEN);
    nlines--;
}

/* ---- file I/O -------------------------------------------------------- */

/* Load a file into the buffer, replacing existing content. */
static int load_file(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    /* Read the whole file into a flat stack buffer (32 KB max). */
    char flat[32768];
    int total = 0;
    ssize_t n;
    while (total < (int)sizeof(flat) - 1 &&
           (n = read(fd, flat + total, sizeof(flat) - 1 - total)) > 0)
        total += (int)n;
    flat[total] = '\0';
    close(fd);
    nlines = 0;
    int pos = 0;
    while (pos < total && nlines < MAX_LINES) {
        int col = 0;
        while (pos < total && flat[pos] != '\n' && col < LINE_LEN - 1)
            buf[nlines][col++] = flat[pos++];
        buf[nlines][col] = '\0';
        if (pos < total && flat[pos] == '\n') pos++;
        nlines++;
    }
    cur = nlines > 0 ? nlines : 0;
    dirty = 0;
    return total;
}

static int write_file(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC);
    if (fd < 0) return -1;
    int total = 0;
    for (int i = 0; i < nlines; i++) {
        int len = strlen(buf[i]);
        if (len > 0) { write(fd, buf[i], len); total += len; }
        write(fd, "\n", 1);
        total++;
    }
    close(fd);
    dirty = 0;
    return total;
}

/* ---- input mode ------------------------------------------------------ */

/* Read lines, insert starting at pos (0-indexed). End with ".". */
static void mode_input(int pos) {
    while (1) {
        int n = read_line(iline, LINE_LEN);
        if (n < 0) break;
        if (n == 1 && iline[0] == '.') break;
        if (shift_down(pos) < 0) { printf("?\n"); break; }
        strncpy(buf[pos], iline, LINE_LEN - 1);
        buf[pos][LINE_LEN - 1] = '\0';
        pos++;
        cur = pos;   /* current line follows the last inserted line */
        dirty = 1;
    }
}

/* ---- substitute ------------------------------------------------------ */

static void do_subst(const char *args) {
    if (cur < 1 || cur > nlines || !args[0]) { printf("?\n"); return; }
    char delim = args[0];
    const char *os = args + 1;                      /* old-string start     */
    const char *oe = strchr(os, delim);              /* old-string end       */
    if (!oe || oe == os) { printf("?\n"); return; }

    const char *ns = oe + 1;                         /* new-string start     */
    const char *ne = strchr(ns, delim);              /* new-string end (opt) */
    int olen = (int)(oe - os);
    int nlen = ne ? (int)(ne - ns) : (int)strlen(ns);
    char old_pat[LINE_LEN], new_pat[LINE_LEN];
    if (olen >= LINE_LEN || nlen >= LINE_LEN) { printf("?\n"); return; }
    memcpy(old_pat, os, olen); old_pat[olen] = '\0';
    memcpy(new_pat, ns, nlen); new_pat[nlen] = '\0';

    char *line = buf[cur - 1];
    char *m = strstr(line, old_pat);
    if (!m) { printf("?\n"); return; }
    int pre = (int)(m - line);
    int suf = strlen(line) - pre - olen;
    if (pre + nlen + suf >= LINE_LEN) { printf("?\n"); return; }
    char tmp[LINE_LEN];
    memcpy(tmp, line, pre);
    memcpy(tmp + pre, new_pat, nlen);
    memcpy(tmp + pre + nlen, m + olen, suf);
    tmp[pre + nlen + suf] = '\0';
    strcpy(line, tmp);
    dirty = 1;
}

/* ---- join ------------------------------------------------------------ */

static void do_join(void) {
    if (cur < 1 || cur >= nlines) { printf("?\n"); return; }
    int len_a = strlen(buf[cur - 1]);
    int len_b = strlen(buf[cur]);
    if (len_a + len_b >= LINE_LEN) { printf("?\n"); return; }
    memcpy(buf[cur - 1] + len_a, buf[cur], len_b + 1);
    shift_up(cur);
    dirty = 1;
}

/* ---- write / edit ---------------------------------------------------- */

static void do_write(const char *arg) {
    while (*arg == ' ') arg++;
    const char *path = *arg ? arg : filename;
    if (!path[0]) { printf("?\n"); return; }
    if (*arg) {
        strncpy(filename, arg, FNAME_LEN - 1);
        filename[FNAME_LEN - 1] = '\0';
    }
    int nb = write_file(path);
    if (nb < 0) printf("?\n"); else printf("%d\n", nb);
}

static void do_edit(const char *arg) {
    while (*arg == ' ') arg++;
    if (*arg) {
        strncpy(filename, arg, FNAME_LEN - 1);
        filename[FNAME_LEN - 1] = '\0';
    }
    if (!filename[0]) { printf("?\n"); return; }
    int nb = load_file(filename);
    if (nb < 0) {
        printf("%s: No such file (new file)\n", filename);
        nlines = 0; cur = 0; dirty = 0;
    } else {
        printf("%d\n", nb);
    }
}

/* ---- help ------------------------------------------------------------ */

static void do_help(void) {
    printf("Commands:\n");
    printf("  a          Append after current line (end with \".\")\n");
    printf("  i          Insert before current line (end with \".\")\n");
    printf("  d          Delete current line\n");
    printf("  p          Print current line\n");
    printf("  n          Print current line with number\n");
    printf("  ,p         Print all lines\n");
    printf("  ,n         Print all lines with numbers\n");
    printf("  NUMBER     Set current line and print\n");
    printf("  w [file]   Write to file (print byte count)\n");
    printf("  e [file]   Load file (replaces buffer)\n");
    printf("  s/old/new/ Substitute on current line\n");
    printf("  j          Join current line with next\n");
    printf("  =          Print current line number\n");
    printf("  $          Go to last line\n");
    printf("  +/-        Next/previous line\n");
    printf("  q          Quit (warns if unsaved)\n");
    printf("  Q          Quit without saving\n");
    printf("  h          This help\n");
}

/* ---- command dispatch ------------------------------------------------ */

/* Returns 1 to quit, 0 to continue. */
static int dispatch(const char *c) {
    int pos = 0;

    /* Empty line: advance and print. */
    if (!c[0]) {
        if (cur < nlines) { cur++; printf("%s\n", buf[cur-1]); }
        else printf("?\n");
        return 0;
    }

    /* ,p / ,n -- print all lines. */
    if (c[0] == ',' && c[2] == '\0') {
        if (c[1] == 'p') {
            for (int i = 0; i < nlines; i++) printf("%s\n", buf[i]);
            if (nlines) cur = nlines;
        } else if (c[1] == 'n') {
            for (int i = 0; i < nlines; i++) printf("%d\t%s\n", i+1, buf[i]);
            if (nlines) cur = nlines;
        } else {
            printf("?\n");
        }
        return 0;
    }

    /* $ -- last line. */
    if (c[0] == '$' && !c[1]) {
        if (nlines) { cur = nlines; printf("%s\n", buf[cur-1]); }
        else printf("?\n");
        return 0;
    }

    /* = -- print current line number. */
    if (c[0] == '=' && !c[1]) { printf("%d\n", cur); return 0; }

    /* Optional line-number address prefix (e.g. "3d", "7p"). */
    if (is_digit(c[0])) {
        int addr = parse_int(c, &pos);
        while (c[pos] == ' ') pos++;
        if (!c[pos]) {
            /* Bare number: go to that line and print. */
            if (addr < 1 || addr > nlines) printf("?\n");
            else { cur = addr; printf("%s\n", buf[cur-1]); }
            return 0;
        }
        /* Set current line, then fall through to command byte. */
        if (addr < 1 || addr > nlines) { printf("?\n"); return 0; }
        cur = addr;
    }

    const char *r = c + pos;

    switch (r[0]) {
    case 'a':
        if (!r[1]) mode_input(cur);
        else printf("?\n");
        return 0;
    case 'i':
        if (!r[1]) mode_input(cur > 0 ? cur - 1 : 0);
        else printf("?\n");
        return 0;
    case 'd':
        if (r[1]) { printf("?\n"); return 0; }
        if (!nlines || cur < 1 || cur > nlines) { printf("?\n"); return 0; }
        shift_up(cur - 1);
        dirty = 1;
        if (cur > nlines) cur = nlines;
        if (cur < 1 && nlines > 0) cur = 1;
        return 0;
    case 'p':
        if (!r[1]) {
            if (cur >= 1 && cur <= nlines) printf("%s\n", buf[cur-1]);
            else printf("?\n");
        } else printf("?\n");
        return 0;
    case 'n':
        if (!r[1]) {
            if (cur >= 1 && cur <= nlines) printf("%d\t%s\n", cur, buf[cur-1]);
            else printf("?\n");
        } else printf("?\n");
        return 0;
    case 's':
        do_subst(r + 1);
        return 0;
    case 'j':
        if (!r[1]) do_join();
        else printf("?\n");
        return 0;
    case 'w':
        do_write(r + 1);
        return 0;
    case 'e':
        do_edit(r + 1);
        return 0;
    case 'q':
        if (r[1]) { printf("?\n"); return 0; }
        if (dirty) { printf("?\n"); dirty = 0; return 0; }
        return 1;
    case 'Q':
        if (r[1]) { printf("?\n"); return 0; }
        return 1;
    case 'h':
        if (!r[1]) do_help();
        else printf("?\n");
        return 0;
    case '+':
        if (!r[1]) {
            if (cur < nlines) { cur++; printf("%s\n", buf[cur-1]); }
            else printf("?\n");
        } else printf("?\n");
        return 0;
    case '-':
        if (!r[1]) {
            if (cur > 1) { cur--; printf("%s\n", buf[cur-1]); }
            else printf("?\n");
        } else printf("?\n");
        return 0;
    default:
        printf("?\n");
        return 0;
    }
}

/* ---- entry point ----------------------------------------------------- */

int main(int argc, char **argv) {
    nlines = 0;
    cur = 0;
    dirty = 0;
    filename[0] = '\0';

    /* Load file from command line argument, if provided. */
    if (argc > 1 && argv[1] && argv[1][0]) {
        strncpy(filename, argv[1], FNAME_LEN - 1);
        filename[FNAME_LEN - 1] = '\0';
        int nb = load_file(filename);
        if (nb >= 0) printf("%d\n", nb);
        else printf("%s: No such file (new file)\n", filename);
    }

    /* Main command loop. */
    while (1) {
        int n = read_line(cmd, CMD_LEN);
        if (n < 0) break;
        if (dispatch(cmd)) break;
    }

    return 0;
}
