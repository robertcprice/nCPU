/*
 * arm64_selfhost.h — Self-hosting C runtime for cc.c on ARM64 Metal GPU.
 *
 * This header provides the same API as arm64_libc.h but without inline
 * assembly, so that cc.c can compile itself. Syscalls use the __syscall()
 * compiler intrinsic which cc.c emits as MOV X8,nr; SVC #0.
 *
 * Only included when __CCGPU__ is defined (i.e., compiled by cc.c itself).
 */

#ifndef ARM64_SELFHOST_H
#define ARM64_SELFHOST_H

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TYPES                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef long ssize_t;
typedef unsigned long size_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

#ifndef NULL
#define NULL 0
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE I/O CONSTANTS                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define O_RDONLY  0
#define O_WRONLY  1
#define O_RDWR   2
#define O_CREAT   64
#define O_TRUNC  512
#define O_APPEND 1024

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

#define AT_FDCWD -100

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SYSCALL NUMBERS                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Standard Linux/ARM64 */
#define SYS_GETCWD     17
#define SYS_MKDIRAT    34
#define SYS_UNLINKAT   35
#define SYS_CHDIR      49
#define SYS_OPENAT     56
#define SYS_CLOSE      57
#define SYS_PIPE2      59
#define SYS_LSEEK      62
#define SYS_READ       63
#define SYS_WRITE      64
#define SYS_EXIT       93
#define SYS_GETPID    172
#define SYS_GETPPID   173
#define SYS_BRK       214
#define SYS_FORK      220
#define SYS_WAIT4     260
#define SYS_DUP3       24

/* Custom GPU syscalls */
#define SYS_GETCHAR   302
#define SYS_CLOCK     303
#define SYS_SLEEP     304
#define SYS_SOCKET    305
#define SYS_BIND      306
#define SYS_LISTEN    307
#define SYS_ACCEPT    308
#define SYS_CONNECT   309
#define SYS_SEND      310
#define SYS_RECV      311
#define SYS_FLUSH_FB  313
#define SYS_KILL      314

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SYSCALL INTRINSIC (emitted by cc.c as MOV X8,nr; SVC #0)                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

long __syscall(long nr, long a0, long a1, long a2, long a3, long a4);

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE I/O WRAPPERS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

int open(const char *path, int flags) {
    return (int)__syscall(SYS_OPENAT, AT_FDCWD, (long)path, (long)flags, 0644, 0);
}

int close(int fd) {
    return (int)__syscall(SYS_CLOSE, (long)fd, 0, 0, 0, 0);
}

long read(int fd, void *buf, long len) {
    return __syscall(SYS_READ, (long)fd, (long)buf, len, 0, 0);
}

long write(int fd, const void *buf, long len) {
    return __syscall(SYS_WRITE, (long)fd, (long)buf, len, 0, 0);
}

long lseek(int fd, long offset, int whence) {
    return __syscall(SYS_LSEEK, (long)fd, offset, (long)whence, 0, 0);
}

void exit(int code) {
    __syscall(SYS_EXIT, (long)code, 0, 0, 0, 0);
}

int mkdir(const char *path) {
    return (int)__syscall(SYS_MKDIRAT, AT_FDCWD, (long)path, 0755, 0, 0);
}

int unlink(const char *path) {
    return (int)__syscall(SYS_UNLINKAT, AT_FDCWD, (long)path, 0, 0, 0);
}

int getcwd(char *buf, long size) {
    return (int)__syscall(SYS_GETCWD, (long)buf, size, 0, 0, 0);
}

int chdir(const char *path) {
    return (int)__syscall(SYS_CHDIR, (long)path, 0, 0, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INTERACTIVE I/O                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

int getchar(void) {
    return (int)__syscall(SYS_GETCHAR, 0, 0, 0, 0, 0);
}

long clock_ms(void) {
    return __syscall(SYS_CLOCK, 0, 0, 0, 0, 0);
}

void sleep_ms(long ms) {
    __syscall(SYS_SLEEP, ms, 0, 0, 0, 0);
}

void sys_flush_fb(int width, int height, void *buffer) {
    __syscall(SYS_FLUSH_FB, (long)width, (long)height, (long)buffer, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PROCESS MANAGEMENT                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

int fork(void) {
    return (int)__syscall(SYS_FORK, 0, 0, 0, 0, 0);
}

int waitpid(int pid, int *status, int options) {
    return (int)__syscall(SYS_WAIT4, (long)pid, (long)status, (long)options, 0, 0);
}

int wait(int *status) {
    return waitpid(-1, status, 0);
}

int pipe(int *pipefd) {
    return (int)__syscall(SYS_PIPE2, (long)pipefd, 0, 0, 0, 0);
}

int dup2(int oldfd, int newfd) {
    return (int)__syscall(SYS_DUP3, (long)oldfd, (long)newfd, 0, 0, 0);
}

int getpid(void) {
    return (int)__syscall(SYS_GETPID, 0, 0, 0, 0, 0);
}

int kill(int pid, int sig) {
    return (int)__syscall(SYS_KILL, (long)pid, (long)sig, 0, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* NETWORKING                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

int socket(int domain, int type, int protocol) {
    return (int)__syscall(SYS_SOCKET, (long)domain, (long)type, (long)protocol, 0, 0);
}

int bind(int fd, long addr, int port) {
    return (int)__syscall(SYS_BIND, (long)fd, addr, (long)port, 0, 0);
}

int listen(int fd, int backlog) {
    return (int)__syscall(SYS_LISTEN, (long)fd, (long)backlog, 0, 0, 0);
}

int accept(int fd) {
    return (int)__syscall(SYS_ACCEPT, (long)fd, 0, 0, 0, 0);
}

int connect(int fd, long addr, int port) {
    return (int)__syscall(SYS_CONNECT, (long)fd, addr, (long)port, 0, 0);
}

long send(int fd, const void *buf, long len) {
    return __syscall(SYS_SEND, (long)fd, (long)buf, len, 0, 0);
}

long recv(int fd, void *buf, long len) {
    return __syscall(SYS_RECV, (long)fd, (long)buf, len, 0, 0);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STRING OPERATIONS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

int strlen(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

char *strcpy(char *dst, const char *src) {
    char *d = dst;
    while (*src) {
        *d = *src;
        d++;
        src++;
    }
    *d = 0;
    return dst;
}

char *strncpy(char *dst, const char *src, int n) {
    int i;
    for (i = 0; i < n && src[i]; i++)
        dst[i] = src[i];
    for (; i < n; i++)
        dst[i] = 0;
    return dst;
}

char *strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;
    while (*src) {
        *d = *src;
        d++;
        src++;
    }
    *d = 0;
    return dst;
}

char *strncat(char *dst, const char *src, int n) {
    char *d = dst;
    while (*d) d++;
    int i;
    for (i = 0; i < n && src[i]; i++) {
        d[i] = src[i];
    }
    d[i] = 0;
    return dst;
}

int strcmp(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return *a - *b;
}

int strncmp(const char *a, const char *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (a[i] != b[i]) return a[i] - b[i];
        if (a[i] == 0) return 0;
    }
    return 0;
}

char *strchr(const char *s, int c) {
    while (*s) {
        if (*s == (char)c) return (char *)s;
        s++;
    }
    return NULL;
}

char *strrchr(const char *s, int c) {
    char *last = NULL;
    while (*s) {
        if (*s == (char)c) last = (char *)s;
        s++;
    }
    if (c == 0) return (char *)s;
    return last;
}

char *strstr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    while (*haystack) {
        const char *h = haystack;
        const char *n = needle;
        while (*h && *n && *h == *n) { h++; n++; }
        if (!*n) return (char *)haystack;
        haystack++;
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MEMORY OPERATIONS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

void *memcpy(void *dst, const void *src, int n) {
    char *d = (char *)dst;
    char *s = (char *)src;
    int i;
    for (i = 0; i < n; i++) d[i] = s[i];
    return dst;
}

void *memmove(void *dst, const void *src, int n) {
    char *d = (char *)dst;
    char *s = (char *)src;
    if (d < s) {
        int i;
        for (i = 0; i < n; i++) d[i] = s[i];
    } else {
        int i;
        for (i = n - 1; i >= 0; i--) d[i] = s[i];
    }
    return dst;
}

void *memset(void *s, int c, int n) {
    char *p = (char *)s;
    int i;
    for (i = 0; i < n; i++) p[i] = (char)c;
    return s;
}

int memcmp(const void *a, const void *b, int n) {
    const char *pa = (const char *)a;
    const char *pb = (const char *)b;
    int i;
    for (i = 0; i < n; i++) {
        if (pa[i] != pb[i]) return pa[i] - pb[i];
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CONVERSION                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

long strtol(const char *s, char *endp, int base) {
    long val = 0;
    int neg = 0;
    while (*s == ' ') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;

    if (base == 0) {
        if (*s == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (*s == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && *s == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }

    while (*s) {
        int d = -1;
        if (*s >= '0' && *s <= '9') d = *s - '0';
        else if (*s >= 'a' && *s <= 'f') d = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') d = *s - 'A' + 10;
        if (d < 0 || d >= base) break;
        val = val * base + d;
        s++;
    }
    return neg ? -val : val;
}

unsigned long strtoul(const char *s, char *endp, int base) {
    return (unsigned long)strtol(s, endp, base);
}

int atoi(const char *s) {
    return (int)strtol(s, NULL, 10);
}

long atol(const char *s) {
    return strtol(s, NULL, 10);
}

char *itoa(long val, char *buf, int base) {
    char *p = buf;
    char *first = buf;
    unsigned long uval;

    if (val < 0 && base == 10) {
        *p = '-';
        p++;
        first = p;
        uval = (unsigned long)(-val);
    } else {
        uval = (unsigned long)val;
    }

    /* Generate digits in reverse */
    char *start = p;
    do {
        int digit = (int)(uval % base);
        if (digit < 10) *p = '0' + digit;
        else *p = 'a' + digit - 10;
        p++;
        uval = uval / base;
    } while (uval > 0);
    *p = 0;

    /* Reverse the digits */
    char *end = p - 1;
    while (start < end) {
        char tmp = *start;
        *start = *end;
        *end = tmp;
        start++;
        end--;
    }
    return buf;
}

int abs(int x) {
    return x < 0 ? -x : x;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* RANDOM NUMBER GENERATOR (LCG)                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

long __rand_state = 12345;

void srand(long seed) {
    __rand_state = seed;
}

int rand(void) {
    __rand_state = __rand_state * 6364136223846793005 + 1442695040888963407;
    return (int)((__rand_state >> 33) & 0x7FFFFFFF);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PRINT HELPERS                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */

void print(const char *s) {
    write(1, s, strlen(s));
}

void putchar(char c) {
    write(1, &c, 1);
}

void puts(const char *s) {
    print(s);
    putchar('\n');
}

void print_int(long n) {
    char buf[21];
    int i = 0;
    int neg = 0;
    if (n < 0) { neg = 1; n = -n; }
    if (n == 0) buf[i++] = '0';
    else {
        while (n > 0) {
            buf[i++] = '0' + (int)(n % 10);
            n = n / 10;
        }
    }
    if (neg) buf[i++] = '-';
    /* Reverse into output */
    char out[21];
    int j;
    for (j = 0; j < i; j++) out[j] = buf[i - 1 - j];
    write(1, out, i);
}

static void print_unsigned(unsigned long n) {
    char buf[21];
    int i = 0;
    if (n == 0) buf[i++] = '0';
    else {
        while (n > 0) {
            buf[i++] = '0' + (int)(n % 10);
            n = n / 10;
        }
    }
    char out[21];
    int j;
    for (j = 0; j < i; j++) out[j] = buf[i - 1 - j];
    write(1, out, i);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PRINTF — Simplified (handles %s, %d, %ld, %u, %x, %c, %%, %p)           */
/*                                                                           */
/* Declared with explicit params since cc.c does not support variadic (...). */
/* The ARM64 calling convention puts args in X0-X7 in order, so callers      */
/* passing fewer args is safe — unused params hold stale register values.    */
/* ═══════════════════════════════════════════════════════════════════════════ */

int printf(const char *fmt, long a0, long a1, long a2, long a3, long a4) {
    long args[6];
    args[0] = a0;
    args[1] = a1;
    args[2] = a2;
    args[3] = a3;
    args[4] = a4;
    args[5] = 0;
    int ai = 0;
    int total = 0;
    int fi = 0;

    while (fmt[fi]) {
        if (fmt[fi] == '%' && fmt[fi + 1]) {
            fi++;

            /* Parse width (optional, e.g. %08x) */
            int zero_pad = 0;
            int width = 0;
            if (fmt[fi] == '0') { zero_pad = 1; fi++; }
            while (fmt[fi] >= '0' && fmt[fi] <= '9') {
                width = width * 10 + (fmt[fi] - '0');
                fi++;
            }

            /* Skip 'l' modifier */
            if (fmt[fi] == 'l') fi++;
            if (fmt[fi] == 'd') {
                if (width > 0 && zero_pad) {
                    /* Zero-padded decimal */
                    char tmp[21];
                    int ti = 0;
                    long v = args[ai];
                    int is_neg = 0;
                    if (v < 0) { is_neg = 1; v = -v; }
                    if (v == 0) tmp[ti++] = '0';
                    else { while (v > 0) { tmp[ti++] = '0' + (int)(v % 10); v = v / 10; } }
                    char out[21];
                    int oi = 0;
                    if (is_neg) out[oi++] = '-';
                    int pad = width - ti - is_neg;
                    while (pad > 0) { out[oi++] = '0'; pad--; }
                    int k;
                    for (k = ti - 1; k >= 0; k--) out[oi++] = tmp[k];
                    write(1, out, oi);
                } else {
                    print_int(args[ai]);
                }
                ai++;
            } else if (fmt[fi] == 'u') {
                print_unsigned((unsigned long)args[ai]);
                ai++;
            } else if (fmt[fi] == 's') {
                char *s = (char *)args[ai];
                if (s) print(s);
                ai++;
            } else if (fmt[fi] == 'x') {
                /* Hex output */
                long v = args[ai];
                ai++;
                char hex[17];
                int hi = 0;
                if (v == 0) { hex[hi++] = '0'; }
                else {
                    char tmp[17];
                    int ti = 0;
                    unsigned long uv = (unsigned long)v;
                    while (uv > 0) {
                        int d = (int)(uv & 15);
                        if (d < 10) tmp[ti++] = '0' + d;
                        else tmp[ti++] = 'a' + d - 10;
                        uv = uv >> 4;
                    }
                    /* Pad with zeros if width specified */
                    if (width > 0 && zero_pad) {
                        while (ti + hi < width) { hex[hi++] = '0'; width--; }
                    }
                    while (ti > 0) hex[hi++] = tmp[--ti];
                }
                /* Pad remaining if needed */
                if (width > 0 && zero_pad && hi < width) {
                    /* Shift digits right, pad left with '0' */
                    char padded[17];
                    int pi = 0;
                    while (pi + hi < width) { padded[pi++] = '0'; }
                    int k;
                    for (k = 0; k < hi; k++) padded[pi++] = hex[k];
                    write(1, padded, pi);
                } else {
                    write(1, hex, hi);
                }
            } else if (fmt[fi] == 'p') {
                /* Pointer: 0x... */
                putchar('0');
                putchar('x');
                unsigned long v = (unsigned long)args[ai];
                ai++;
                char hex[17];
                int hi = 0;
                if (v == 0) { hex[hi++] = '0'; }
                else {
                    char tmp[17];
                    int ti = 0;
                    while (v > 0) {
                        int d = (int)(v & 15);
                        if (d < 10) tmp[ti++] = '0' + d;
                        else tmp[ti++] = 'a' + d - 10;
                        v = v >> 4;
                    }
                    while (ti > 0) hex[hi++] = tmp[--ti];
                }
                write(1, hex, hi);
            } else if (fmt[fi] == 'c') {
                char ch = (char)args[ai];
                ai++;
                putchar(ch);
            } else if (fmt[fi] == '%') {
                putchar('%');
            } else {
                /* Unknown format: print raw */
                putchar('%');
                putchar(fmt[fi]);
            }
            fi++;
        } else {
            putchar(fmt[fi]);
            fi++;
        }
        total++;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SNPRINTF / SPRINTF — writes to buffer instead of stdout                   */
/*                                                                           */
/* Uses explicit params (same as printf).                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void _snprintf_int(char *buf, int *pos, int max, long n) {
    char tmp[21];
    int i = 0;
    int neg = 0;
    if (n < 0) { neg = 1; n = -n; }
    if (n == 0) tmp[i++] = '0';
    else {
        while (n > 0) {
            tmp[i++] = '0' + (int)(n % 10);
            n = n / 10;
        }
    }
    if (neg && *pos < max - 1) buf[(*pos)++] = '-';
    int j;
    for (j = i - 1; j >= 0; j--) {
        if (*pos < max - 1) buf[(*pos)++] = tmp[j];
    }
}

static void _snprintf_hex(char *buf, int *pos, int max, unsigned long v) {
    char tmp[17];
    int ti = 0;
    if (v == 0) { tmp[ti++] = '0'; }
    else {
        while (v > 0) {
            int d = (int)(v & 15);
            if (d < 10) tmp[ti++] = '0' + d;
            else tmp[ti++] = 'a' + d - 10;
            v = v >> 4;
        }
    }
    int j;
    for (j = ti - 1; j >= 0; j--) {
        if (*pos < max - 1) buf[(*pos)++] = tmp[j];
    }
}

int snprintf(char *buf, long n, const char *fmt, long a0, long a1, long a2, long a3, long a4) {
    long args[6];
    args[0] = a0;
    args[1] = a1;
    args[2] = a2;
    args[3] = a3;
    args[4] = a4;
    args[5] = 0;
    int ai = 0;
    int pos = 0;
    int max = (int)n;
    int fi = 0;

    while (fmt[fi] && pos < max - 1) {
        if (fmt[fi] == '%' && fmt[fi + 1]) {
            fi++;
            /* Skip 'l' modifier */
            if (fmt[fi] == 'l') fi++;
            if (fmt[fi] == 'd') {
                _snprintf_int(buf, &pos, max, args[ai]);
                ai++;
            } else if (fmt[fi] == 'u') {
                /* Unsigned - treat as positive */
                _snprintf_int(buf, &pos, max, args[ai]);
                ai++;
            } else if (fmt[fi] == 's') {
                char *s = (char *)args[ai];
                ai++;
                if (s) {
                    while (*s && pos < max - 1) buf[pos++] = *s++;
                }
            } else if (fmt[fi] == 'x') {
                _snprintf_hex(buf, &pos, max, (unsigned long)args[ai]);
                ai++;
            } else if (fmt[fi] == 'c') {
                if (pos < max - 1) buf[pos++] = (char)args[ai];
                ai++;
            } else if (fmt[fi] == '%') {
                if (pos < max - 1) buf[pos++] = '%';
            } else {
                if (pos < max - 1) buf[pos++] = '%';
                if (pos < max - 1) buf[pos++] = fmt[fi];
            }
            fi++;
        } else {
            buf[pos++] = fmt[fi];
            fi++;
        }
    }
    if (max > 0) buf[pos] = 0;
    return pos;
}

int sprintf(char *buf, const char *fmt, long a0, long a1, long a2, long a3, long a4) {
    return snprintf(buf, 4096, fmt, a0, a1, a2, a3, a4);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MEMORY ALLOCATION (bump allocator + free list via SYS_BRK)               */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Block header for allocated memory.
 * Layout: [magic:8][size:8][free:4][pad:4] = 24 bytes, but we align to 16.
 * Simplified: use a 16-byte header with size (includes free flag in high bit).
 */

long __heap_start = 0;
long __heap_end = 0;

static long _sbrk(long incr) {
    long current = __syscall(SYS_BRK, 0, 0, 0, 0, 0);
    if (incr == 0) return current;
    long new_brk = current + incr;
    long result = __syscall(SYS_BRK, new_brk, 0, 0, 0, 0);
    if (result != new_brk) return -1;
    return current;
}

void *malloc(long size) {
    if (__heap_start == 0) {
        __heap_start = _sbrk(0);
        __heap_end = __heap_start;
    }

    if (size == 0) size = 1;
    /* Align to 8 bytes */
    size = (size + 7) & ~7;

    /* Scan free list: header is [size:8][free:8] = 16 bytes */
    long h = __heap_start;
    while (h < __heap_end) {
        long *hdr = (long *)h;
        long bsize = hdr[0];
        long bfree = hdr[1];
        if (bsize == 0) break;
        if (bfree && bsize >= size) {
            hdr[1] = 0;
            return (void *)(h + 16);
        }
        h = h + 16 + bsize;
    }

    /* Extend heap */
    long block = _sbrk(16 + size);
    if (block == -1) return NULL;
    long *hdr = (long *)block;
    hdr[0] = size;
    hdr[1] = 0;
    __heap_end = block + 16 + size;
    return (void *)(block + 16);
}

void free(void *ptr) {
    if (!ptr) return;
    long *hdr = (long *)((char *)ptr - 16);
    hdr[1] = 1;
}

void *calloc(long n, long size) {
    long total = n * size;
    void *ptr = malloc(total);
    if (ptr) memset(ptr, 0, (int)total);
    return ptr;
}

void *realloc(void *ptr, long size) {
    if (!ptr) return malloc(size);
    if (size == 0) { free(ptr); return NULL; }
    long *hdr = (long *)((char *)ptr - 16);
    long old_size = hdr[0];
    if (old_size >= size) return ptr;
    void *new_ptr = malloc(size);
    if (!new_ptr) return NULL;
    memcpy(new_ptr, ptr, (int)old_size);
    free(ptr);
    return new_ptr;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SORTING (insertion sort — safe for small GPU stack)                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

void qsort(void *base, int nmemb, int size, void *compar) {
    char *arr = (char *)base;
    char tmp[256];
    if (size > 256) return;
    int i;
    for (i = 1; i < nmemb; i++) {
        memcpy(tmp, arr + i * size, size);
        int j = i;
        while (j > 0) {
            int cmp = compar((void *)(arr + (j - 1) * size), (void *)tmp);
            if (cmp <= 0) break;
            memcpy(arr + j * size, arr + (j - 1) * size, size);
            j--;
        }
        memcpy(arr + j * size, tmp, size);
    }
}

#endif /* ARM64_SELFHOST_H */
