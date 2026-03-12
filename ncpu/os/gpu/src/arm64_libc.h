/*
 * arm64_libc.h — Freestanding C runtime library for ARM64 Metal GPU kernel.
 *
 * Header-only: all functions are static inline, no .c file needed.
 * Compiles directly into user binaries via aarch64-elf-gcc.
 *
 * Provides: malloc/free, printf, string ops, file I/O, utility functions.
 * All I/O routes through SVC syscalls handled by the Python syscall layer.
 */

#ifndef ARM64_LIBC_H
#define ARM64_LIBC_H

#include "arm64_syscalls.h"

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TYPES                                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef int            int32_t;
typedef unsigned int   uint32_t;
typedef long           int64_t;
typedef unsigned long  uint64_t;

#ifndef NULL
#define NULL ((void*)0)
#endif

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
/* STRING OPERATIONS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Note: strlen is already defined in arm64_syscalls.h */

static inline char *strcpy(char *dst, const char *src) {
    char *d = dst;
    while ((*d++ = *src++))
        ;
    return dst;
}

static inline char *strncpy(char *dst, const char *src, size_t n) {
    size_t i;
    for (i = 0; i < n && src[i]; i++)
        dst[i] = src[i];
    for (; i < n; i++)
        dst[i] = '\0';
    return dst;
}

static inline char *strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d) d++;
    while ((*d++ = *src++))
        ;
    return dst;
}

static inline char *strncat(char *dst, const char *src, size_t n) {
    char *d = dst;
    while (*d) d++;
    size_t i;
    for (i = 0; i < n && src[i]; i++)
        d[i] = src[i];
    d[i] = '\0';
    return dst;
}

static inline int strcmp(const char *a, const char *b) {
    while (*a && *b && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}

static inline int strncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return (unsigned char)a[i] - (unsigned char)b[i];
        if (a[i] == '\0') return 0;
    }
    return 0;
}

static inline char *strchr(const char *s, int c) {
    while (*s) {
        if (*s == (char)c) return (char *)s;
        s++;
    }
    return (c == '\0') ? (char *)s : NULL;
}

static inline char *strrchr(const char *s, int c) {
    const char *last = NULL;
    while (*s) {
        if (*s == (char)c) last = s;
        s++;
    }
    if (c == '\0') return (char *)s;
    return (char *)last;
}

static inline char *strstr(const char *haystack, const char *needle) {
    if (!*needle) return (char *)haystack;
    for (; *haystack; haystack++) {
        const char *h = haystack, *n = needle;
        while (*h && *n && *h == *n) { h++; n++; }
        if (!*n) return (char *)haystack;
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MEMORY OPERATIONS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * memcpy/memset/memmove/memcmp: NOT static inline.
 * GCC emits external calls to these for large struct/array operations
 * even with -ffreestanding, so they must be linkable symbols.
 */
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *s, int c, size_t n) {
    unsigned char *p = (unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

void *memmove(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else {
        d += n; s += n;
        while (n--) *--d = *--s;
    }
    return dst;
}

int memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *a = (const unsigned char *)s1;
    const unsigned char *b = (const unsigned char *)s2;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return a[i] - b[i];
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* CONVERSION                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline long atoi_libc(const char *s) {
    long n = 0;
    int neg = 0;
    while (*s == ' ') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    while (*s >= '0' && *s <= '9') {
        n = n * 10 + (*s - '0');
        s++;
    }
    return neg ? -n : n;
}

static inline long atol(const char *s) {
    return atoi_libc(s);
}

static inline char *itoa(long val, char *buf, int base) {
    char *p = buf;
    char *first = buf;
    unsigned long uval;

    if (val < 0 && base == 10) {
        *p++ = '-';
        first = p;
        uval = (unsigned long)(-val);
    } else {
        uval = (unsigned long)val;
    }

    /* Generate digits in reverse */
    char *start = p;
    do {
        int digit = uval % base;
        *p++ = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
        uval /= base;
    } while (uval);
    *p = '\0';

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

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MEMORY ALLOCATION (bump allocator + free list via SYS_BRK)               */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Block header for allocated memory — magic guard detects corruption */
#define _ALLOC_MAGIC 0xA110CA7E

struct _alloc_header {
    unsigned long magic;  /* corruption guard */
    size_t size;          /* size of data area (not including header) */
    int    free;          /* 1 if freed */
};

#define _ALLOC_HEADER_SIZE sizeof(struct _alloc_header)
#define _ALLOC_ALIGN 8
#define _ALLOC_MAX_SIZE (1024 * 1024)  /* 1 MB max single allocation */

static void *_heap_start = NULL;
static void *_heap_end = NULL;

static inline void *_sbrk(long incr) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_BRK;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    long current = x0;

    if (incr == 0) return (void *)current;

    long new_brk = current + incr;
    register long a0 __asm__("x0") = new_brk;
    register long a8 __asm__("x8") = SYS_BRK;
    __asm__ volatile("svc #0" : "+r"(a0) : "r"(a8) : "memory");

    if (a0 != new_brk) return (void *)-1;
    return (void *)current;
}

static inline void _alloc_init(void) {
    if (!_heap_start) {
        _heap_start = _sbrk(0);
        _heap_end = _heap_start;
    }
}

static inline void *malloc(size_t size) {
    _alloc_init();

    if (size == 0) size = 1;
    if (size > _ALLOC_MAX_SIZE) return NULL;

    /* Align size to 8 bytes */
    size = (size + _ALLOC_ALIGN - 1) & ~(_ALLOC_ALIGN - 1);

    /* First-fit: scan existing freed blocks with corruption detection */
    struct _alloc_header *h = (struct _alloc_header *)_heap_start;
    int max_blocks = 10000;  /* prevent infinite loop on corruption */
    while ((void *)h < _heap_end && max_blocks-- > 0) {
        if (h->magic != _ALLOC_MAGIC) break;  /* heap corrupted — stop scanning */
        if (h->size == 0) break;               /* zero-size block — corruption */
        if (h->free && h->size >= size) {
            h->free = 0;
            return (void *)((char *)h + _ALLOC_HEADER_SIZE);
        }
        h = (struct _alloc_header *)((char *)h + _ALLOC_HEADER_SIZE + h->size);
    }

    /* No free block found — extend heap */
    void *block = _sbrk(_ALLOC_HEADER_SIZE + size);
    if (block == (void *)-1) return NULL;

    h = (struct _alloc_header *)block;
    h->magic = _ALLOC_MAGIC;
    h->size = size;
    h->free = 0;
    _heap_end = (char *)h + _ALLOC_HEADER_SIZE + size;

    return (void *)((char *)h + _ALLOC_HEADER_SIZE);
}

static inline void free(void *ptr) {
    if (!ptr) return;
    struct _alloc_header *h = (struct _alloc_header *)((char *)ptr - _ALLOC_HEADER_SIZE);
    if (h->magic != _ALLOC_MAGIC) return;  /* corruption guard */
    h->free = 1;
}

static inline void *calloc(size_t n, size_t size) {
    size_t total = n * size;
    void *ptr = malloc(total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

static inline void *realloc(void *ptr, size_t size) {
    if (!ptr) return malloc(size);
    if (size == 0) { free(ptr); return NULL; }

    struct _alloc_header *h = (struct _alloc_header *)((char *)ptr - _ALLOC_HEADER_SIZE);
    if (h->magic != _ALLOC_MAGIC) return NULL;  /* corruption guard */
    if (h->size >= size) return ptr;

    void *new_ptr = malloc(size);
    if (!new_ptr) return NULL;
    memcpy(new_ptr, ptr, h->size);
    free(ptr);
    return new_ptr;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FILE I/O (wrapping syscalls)                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline int open(const char *path, int flags) {
    register long x0 __asm__("x0") = AT_FDCWD;
    register long x1 __asm__("x1") = (long)path;
    register long x2 __asm__("x2") = flags;
    register long x3 __asm__("x3") = 0644;  /* mode */
    register long x8 __asm__("x8") = SYS_OPENAT;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x3), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int close(int fd) {
    register long x0 __asm__("x0") = fd;
    register long x8 __asm__("x8") = SYS_CLOSE;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return (int)x0;
}

static inline ssize_t read(int fd, void *buf, size_t count) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)count;
    register long x8 __asm__("x8") = SYS_READ;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (ssize_t)x0;
}

static inline ssize_t write(int fd, const void *buf, size_t count) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)count;
    register long x8 __asm__("x8") = SYS_WRITE;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (ssize_t)x0;
}

static inline long lseek(int fd, long offset, int whence) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = offset;
    register long x2 __asm__("x2") = whence;
    register long x8 __asm__("x8") = SYS_LSEEK;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return x0;
}

static inline int mkdir(const char *path) {
    register long x0 __asm__("x0") = AT_FDCWD;
    register long x1 __asm__("x1") = (long)path;
    register long x2 __asm__("x2") = 0755;
    register long x8 __asm__("x8") = SYS_MKDIRAT;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int unlink(const char *path) {
    register long x0 __asm__("x0") = AT_FDCWD;
    register long x1 __asm__("x1") = (long)path;
    register long x2 __asm__("x2") = 0;  /* flags */
    register long x8 __asm__("x8") = SYS_UNLINKAT;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int rmdir(const char *path) {
    register long x0 __asm__("x0") = AT_FDCWD;
    register long x1 __asm__("x1") = (long)path;
    register long x2 __asm__("x2") = 0x200;  /* AT_REMOVEDIR */
    register long x8 __asm__("x8") = SYS_UNLINKAT;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int getcwd(char *buf, size_t size) {
    register long x0 __asm__("x0") = (long)buf;
    register long x1 __asm__("x1") = (long)size;
    register long x8 __asm__("x8") = SYS_GETCWD;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int chdir(const char *path) {
    register long x0 __asm__("x0") = (long)path;
    register long x8 __asm__("x8") = SYS_CHDIR;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return (int)x0;
}

/* getdents64: fill buffer with directory entries, return bytes read */
static inline int getdents64(int fd, void *buf, size_t bufsize) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)bufsize;
    register long x8 __asm__("x8") = SYS_GETDENTS64;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

/* Custom syscalls */
static inline int sys_compile(const char *src_path, const char *bin_path) {
    register long x0 __asm__("x0") = (long)src_path;
    register long x1 __asm__("x1") = (long)bin_path;
    register long x8 __asm__("x8") = SYS_COMPILE;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int sys_exec(const char *bin_path) {
    register long x0 __asm__("x0") = (long)bin_path;
    register long x8 __asm__("x8") = SYS_EXEC;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return (int)x0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INTERACTIVE I/O                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline int getchar(void) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_GETCHAR;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return (int)x0;
}

static inline long clock_ms(void) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_CLOCK;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return x0;
}

static inline void sleep_ms(long ms) {
    register long x0 __asm__("x0") = ms;
    register long x8 __asm__("x8") = SYS_SLEEP;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* NETWORKING                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline int socket(int domain, int type, int protocol) {
    register long x0 __asm__("x0") = domain;
    register long x1 __asm__("x1") = type;
    register long x2 __asm__("x2") = protocol;
    register long x8 __asm__("x8") = SYS_SOCKET;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int bind(int fd, long addr, int port) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = addr;
    register long x2 __asm__("x2") = port;
    register long x8 __asm__("x8") = SYS_BIND;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int listen(int fd, int backlog) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = backlog;
    register long x8 __asm__("x8") = SYS_LISTEN;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int accept(int fd) {
    register long x0 __asm__("x0") = fd;
    register long x8 __asm__("x8") = SYS_ACCEPT;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
    return (int)x0;
}

static inline int connect(int fd, long addr, int port) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = addr;
    register long x2 __asm__("x2") = port;
    register long x8 __asm__("x8") = SYS_CONNECT;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline ssize_t send(int fd, const void *buf, size_t len) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)len;
    register long x8 __asm__("x8") = SYS_SEND;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (ssize_t)x0;
}

static inline ssize_t recv(int fd, void *buf, size_t len) {
    register long x0 __asm__("x0") = fd;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)len;
    register long x8 __asm__("x8") = SYS_RECV;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (ssize_t)x0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PROCESS MANAGEMENT                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline int fork(void) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_FORK;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory", "cc");
    return (int)x0;
}

static inline int waitpid(int pid, int *status, int options) {
    register long x0 __asm__("x0") = pid;
    register long x1 __asm__("x1") = (long)status;
    register long x2 __asm__("x2") = options;
    register long x8 __asm__("x8") = SYS_WAIT4;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int wait(int *status) {
    return waitpid(-1, status, 0);
}

static inline int pipe(int pipefd[2]) {
    register long x0 __asm__("x0") = (long)pipefd;
    register long x8 __asm__("x8") = SYS_PIPE2;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int dup2(int oldfd, int newfd) {
    register long x0 __asm__("x0") = oldfd;
    register long x1 __asm__("x1") = newfd;
    register long x8 __asm__("x8") = SYS_DUP3;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int getpid(void) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_GETPID;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory", "cc");
    return (int)x0;
}

static inline int getppid(void) {
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_GETPPID;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory", "cc");
    return (int)x0;
}

static inline int kill(int pid, int sig) {
    register long x0 __asm__("x0") = pid;
    register long x1 __asm__("x1") = sig;
    register long x8 __asm__("x8") = SYS_KILL;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int sys_getenv(const char *key, char *buf, size_t bufsz) {
    register long x0 __asm__("x0") = (long)key;
    register long x1 __asm__("x1") = (long)buf;
    register long x2 __asm__("x2") = (long)bufsz;
    register long x8 __asm__("x8") = SYS_GETENV;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    return (int)x0;
}

static inline int sys_setenv(const char *key, const char *value) {
    register long x0 __asm__("x0") = (long)key;
    register long x1 __asm__("x1") = (long)value;
    register long x8 __asm__("x8") = SYS_SETENV;
    __asm__ volatile("svc #0"
        : "+r"(x0)
        : "r"(x1), "r"(x8)
        : "memory");
    return (int)x0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* FORMATTED OUTPUT (printf)                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static inline void puts(const char *s) {
    sys_write(1, s, strlen(s));
    sys_write(1, "\n", 1);
}

/* Internal: write unsigned integer to buffer, return chars written */
static inline int _utoa_buf(char *buf, unsigned long val, int base) {
    char tmp[24];
    int i = 0;
    if (val == 0) { tmp[i++] = '0'; }
    else {
        while (val > 0) {
            int d = val % base;
            tmp[i++] = (d < 10) ? ('0' + d) : ('a' + d - 10);
            val /= base;
        }
    }
    /* Reverse into buf */
    for (int j = 0; j < i; j++) buf[j] = tmp[i - 1 - j];
    return i;
}

static inline int vsnprintf(char *buf, size_t n, const char *fmt, __builtin_va_list ap) {
    size_t pos = 0;

    #define _OUT(c) do { if (pos < n - 1) buf[pos] = (c); pos++; } while(0)

    while (*fmt) {
        if (*fmt != '%') {
            _OUT(*fmt);
            fmt++;
            continue;
        }
        fmt++;  /* skip '%' */

        /* Flags */
        int zero_pad = 0;
        int left_align = 0;
        while (*fmt == '0' || *fmt == '-') {
            if (*fmt == '0') zero_pad = 1;
            if (*fmt == '-') left_align = 1;
            fmt++;
        }

        /* Width */
        int width = 0;
        while (*fmt >= '0' && *fmt <= '9') {
            width = width * 10 + (*fmt - '0');
            fmt++;
        }

        /* Length modifier */
        int is_long = 0;
        if (*fmt == 'l') { is_long = 1; fmt++; }

        /* Conversion */
        char tmp[24];
        int len = 0;
        int is_neg = 0;

        switch (*fmt) {
            case 'd':
            case 'i': {
                long val = is_long ? __builtin_va_arg(ap, long) : (long)__builtin_va_arg(ap, int);
                if (val < 0) { is_neg = 1; val = -val; }
                len = _utoa_buf(tmp, (unsigned long)val, 10);
                break;
            }
            case 'u': {
                unsigned long val = is_long ? __builtin_va_arg(ap, unsigned long) : (unsigned long)__builtin_va_arg(ap, unsigned int);
                len = _utoa_buf(tmp, val, 10);
                break;
            }
            case 'x': {
                unsigned long val = is_long ? __builtin_va_arg(ap, unsigned long) : (unsigned long)__builtin_va_arg(ap, unsigned int);
                len = _utoa_buf(tmp, val, 16);
                break;
            }
            case 'p': {
                unsigned long val = (unsigned long)__builtin_va_arg(ap, void *);
                _OUT('0'); _OUT('x');
                len = _utoa_buf(tmp, val, 16);
                break;
            }
            case 's': {
                const char *s = __builtin_va_arg(ap, const char *);
                if (!s) s = "(null)";
                while (*s) { _OUT(*s); s++; }
                fmt++;
                continue;
            }
            case 'c': {
                char c = (char)__builtin_va_arg(ap, int);
                _OUT(c);
                fmt++;
                continue;
            }
            case '%':
                _OUT('%');
                fmt++;
                continue;
            default:
                _OUT('%');
                _OUT(*fmt);
                fmt++;
                continue;
        }

        /* Padding and output */
        int total = len + is_neg;
        int pad = (width > total) ? (width - total) : 0;

        if (!left_align && pad > 0) {
            char pc = zero_pad ? '0' : ' ';
            if (is_neg && zero_pad) { _OUT('-'); is_neg = 0; }
            for (int i = 0; i < pad; i++) _OUT(pc);
        }
        if (is_neg) _OUT('-');
        for (int i = 0; i < len; i++) _OUT(tmp[i]);
        if (left_align && pad > 0) {
            for (int i = 0; i < pad; i++) _OUT(' ');
        }

        fmt++;
    }

    if (n > 0) buf[(pos < n) ? pos : n - 1] = '\0';

    #undef _OUT
    return (int)pos;
}

static inline int snprintf(char *buf, size_t n, const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int ret = vsnprintf(buf, n, fmt, ap);
    __builtin_va_end(ap);
    return ret;
}

static inline int sprintf(char *buf, const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int ret = vsnprintf(buf, 4096, fmt, ap);  /* no bounds check — caller's responsibility */
    __builtin_va_end(ap);
    return ret;
}

static inline int printf(const char *fmt, ...) {
    char buf[1024];
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int ret = vsnprintf(buf, sizeof(buf), fmt, ap);
    __builtin_va_end(ap);
    if (ret > 0) {
        int out = ret < (int)sizeof(buf) ? ret : (int)sizeof(buf) - 1;
        sys_write(1, buf, out);
    }
    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* UTILITY                                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

static unsigned long _rand_state = 12345;

static inline void srand(unsigned long seed) {
    _rand_state = seed;
}

static inline int rand(void) {
    _rand_state = _rand_state * 6364136223846793005UL + 1442695040888963407UL;
    return (int)((_rand_state >> 33) & 0x7FFFFFFF);
}

static inline void abort(void) {
    sys_exit(134);
    __builtin_unreachable();
}

static inline void exit(int code) {
    sys_exit(code);
    __builtin_unreachable();
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SORTING & PARSING                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Insertion sort — no recursion, safe for small GPU stack */
static inline void qsort(void *base, size_t nmemb, size_t size,
                          int (*compar)(const void *, const void *)) {
    char *arr = (char *)base;
    char tmp[256];  /* max element size */
    if (size > sizeof(tmp)) return;
    for (size_t i = 1; i < nmemb; i++) {
        memcpy(tmp, arr + i * size, size);
        size_t j = i;
        while (j > 0 && compar(arr + (j - 1) * size, tmp) > 0) {
            memcpy(arr + j * size, arr + (j - 1) * size, size);
            j--;
        }
        memcpy(arr + j * size, tmp, size);
    }
}

static inline long strtol(const char *s, char **endptr, int base) {
    long n = 0;
    int neg = 0;
    while (*s == ' ') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    /* Auto-detect base */
    if (base == 0) {
        if (*s == '0' && (s[1] == 'x' || s[1] == 'X')) { base = 16; s += 2; }
        else if (*s == '0') { base = 8; s++; }
        else base = 10;
    } else if (base == 16 && *s == '0' && (s[1] == 'x' || s[1] == 'X')) {
        s += 2;
    }
    while (*s) {
        int d;
        if (*s >= '0' && *s <= '9') d = *s - '0';
        else if (*s >= 'a' && *s <= 'f') d = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') d = *s - 'A' + 10;
        else break;
        if (d >= base) break;
        n = n * base + d;
        s++;
    }
    if (endptr) *endptr = (char *)s;
    return neg ? -n : n;
}

static inline unsigned long strtoul(const char *s, char **endptr, int base) {
    return (unsigned long)strtol(s, endptr, base);
}

/* Minimal sscanf supporting %d, %s, %c */
static inline int sscanf(const char *str, const char *fmt, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    int matched = 0;
    while (*fmt && *str) {
        if (*fmt == '%') {
            fmt++;
            if (*fmt == 'd') {
                int *p = __builtin_va_arg(ap, int *);
                long val = 0;
                int neg = 0;
                while (*str == ' ') str++;
                if (*str == '-') { neg = 1; str++; }
                if (*str < '0' || *str > '9') break;
                while (*str >= '0' && *str <= '9') {
                    val = val * 10 + (*str - '0');
                    str++;
                }
                *p = (int)(neg ? -val : val);
                matched++;
                fmt++;
            } else if (*fmt == 's') {
                char *p = __builtin_va_arg(ap, char *);
                while (*str == ' ') str++;
                while (*str && *str != ' ' && *str != '\n') *p++ = *str++;
                *p = '\0';
                matched++;
                fmt++;
            } else if (*fmt == 'c') {
                char *p = __builtin_va_arg(ap, char *);
                *p = *str++;
                matched++;
                fmt++;
            } else {
                break;
            }
        } else if (*fmt == ' ') {
            while (*str == ' ') str++;
            fmt++;
        } else {
            if (*fmt != *str) break;
            fmt++; str++;
        }
    }
    __builtin_va_end(ap);
    return matched;
}

#endif /* ARM64_LIBC_H */
