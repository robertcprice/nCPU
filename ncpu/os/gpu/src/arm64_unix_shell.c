/*
 * GPU-Native UNIX Shell — Multi-process shell running on ARM64 Metal GPU.
 *
 * Commands: ls, cd, pwd, cat, echo, mkdir, rm, rmdir, touch, wc, cp,
 *           head, cc, run, env, export, grep, sort, uniq, tee, ps, help, exit
 *
 * Features: cwd-aware prompt, output redirection (> and >>), pipes (|),
 *           background jobs (&), command chaining (;, &&, ||),
 *           fork/wait multi-process execution
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/arm64_unix_shell.c
 *          -o /tmp/unix_shell.elf
 */

#include "arm64_libc.h"

/* ═══════════════════════════════════════════════════════════════════════════ */
/* DIRECTORY ENTRY PARSING (matches getdents64 format from Python handler)   */
/* ═══════════════════════════════════════════════════════════════════════════ */

struct dirent64 {
    unsigned short d_reclen;   /* entry length */
    unsigned char  d_type;     /* 0=file, 1=dir */
    char           d_name[];   /* null-terminated */
};

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ENVIRONMENT VARIABLES                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_ENV 32
#define ENV_KEY_SIZE 64
#define ENV_VAL_SIZE 128

static char env_keys[MAX_ENV][ENV_KEY_SIZE];
static char env_vals[MAX_ENV][ENV_VAL_SIZE];
static int env_count = 0;

static void env_init(void) {
    /* Default environment */
    strcpy(env_keys[0], "PATH"); strcpy(env_vals[0], "/bin");
    strcpy(env_keys[1], "HOME"); strcpy(env_vals[1], "/home");
    strcpy(env_keys[2], "SHELL"); strcpy(env_vals[2], "/bin/sh");
    strcpy(env_keys[3], "USER"); strcpy(env_vals[3], "root");
    env_count = 4;
}

static const char *env_get(const char *key) {
    for (int i = 0; i < env_count; i++) {
        if (strcmp(env_keys[i], key) == 0) return env_vals[i];
    }
    return NULL;
}

static void env_set(const char *key, const char *val) {
    for (int i = 0; i < env_count; i++) {
        if (strcmp(env_keys[i], key) == 0) {
            strncpy(env_vals[i], val, ENV_VAL_SIZE - 1);
            env_vals[i][ENV_VAL_SIZE - 1] = '\0';
            return;
        }
    }
    if (env_count < MAX_ENV) {
        strncpy(env_keys[env_count], key, ENV_KEY_SIZE - 1);
        env_keys[env_count][ENV_KEY_SIZE - 1] = '\0';
        strncpy(env_vals[env_count], val, ENV_VAL_SIZE - 1);
        env_vals[env_count][ENV_VAL_SIZE - 1] = '\0';
        env_count++;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STRING UTILITIES                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int starts_with(const char *str, const char *prefix) {
    while (*prefix) {
        if (*str != *prefix) return 0;
        str++; prefix++;
    }
    return 1;
}

/* Skip to the argument portion of a command string */
static const char *skip_to_arg(const char *cmd) {
    while (*cmd && *cmd != ' ' && *cmd != '\t') cmd++;
    while (*cmd == ' ' || *cmd == '\t') cmd++;
    return cmd;
}

/* Trim trailing whitespace in-place */
static void trim_trailing(char *s) {
    int len = strlen(s);
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t' || s[len-1] == '\n' || s[len-1] == '\r')) {
        s[--len] = '\0';
    }
}

/* Check if string contains a '>' for output redirection, split around it.
 * Returns: 0=no redirect, 1=truncate (>), 2=append (>>) */
static int split_redirect(char *cmd, char **out_file) {
    /* Skip pipe chars — don't mistake pipe for redirect */
    char *gt = NULL;
    for (char *p = cmd; *p; p++) {
        if (*p == '|') continue;  /* skip pipes */
        if (*p == '>') { gt = p; break; }
    }
    if (!gt) {
        *out_file = NULL;
        return 0;
    }
    int mode = 1;  /* truncate */
    *gt = '\0';
    gt++;
    if (*gt == '>') { mode = 2; gt++; }  /* >> = append */
    trim_trailing(cmd);
    while (*gt == ' ' || *gt == '\t') gt++;
    *out_file = gt;
    return mode;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* COMMANDS                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void cmd_help(void) {
    printf("Commands:\n");
    printf("  ls [path]        List directory\n");
    printf("  cd <path>        Change directory\n");
    printf("  pwd              Print working directory\n");
    printf("  cat <file>       Display file contents\n");
    printf("  echo [text] [>f] Print text or redirect to file\n");
    printf("  mkdir <dir>      Create directory\n");
    printf("  rm <file>        Remove file\n");
    printf("  rmdir <dir>      Remove empty directory\n");
    printf("  touch <file>     Create empty file\n");
    printf("  wc <file>        Count lines/words/chars\n");
    printf("  cp <src> <dst>   Copy file\n");
    printf("  head <file> [n]  First n lines (default 10)\n");
    printf("  cc <file.c>      Compile C file\n");
    printf("  run <binary>     Execute compiled binary\n");
    printf("  env              Show environment variables\n");
    printf("  export K=V       Set environment variable\n");
    printf("  sha256 <file>    SHA-256 hash of file\n");
    printf("  grep <pat> [f]   Search for pattern in file or stdin\n");
    printf("  sort [file]      Sort lines alphabetically\n");
    printf("  uniq [file]      Remove adjacent duplicates\n");
    printf("  tee <file>       Copy stdin to file and stdout\n");
    printf("  ps               List running processes\n");
    printf("  kill [-9] <pid>  Send signal to process\n");
    printf("  ed [file]        Line editor\n");
    printf("  help             Show this help\n");
    printf("  exit             Exit shell\n");
    printf("\nOperators: cmd | cmd  cmd &  cmd ; cmd  cmd && cmd  cmd || cmd\n");
    printf("Redirect:  cmd > file  cmd >> file\n");
}

static void cmd_pwd(void) {
    char buf[256];
    if (getcwd(buf, sizeof(buf)) == 0) {
        printf("%s\n", buf);
    } else {
        printf("Error: cannot get cwd\n");
    }
}

static void cmd_ls(const char *path) {
    char dirent_buf[4096];
    int dirfd;

    /* Open the directory — for our simple fs, open the path as a file */
    if (*path) {
        dirfd = open(path, O_RDONLY);
    } else {
        dirfd = open(".", O_RDONLY);
    }

    if (dirfd < 0) {
        /* If open fails, try current dir */
        dirfd = open(".", O_RDONLY);
    }

    int n = getdents64(dirfd, dirent_buf, sizeof(dirent_buf));
    if (n <= 0) {
        printf("(empty)\n");
        if (dirfd >= 3) close(dirfd);
        return;
    }

    /* Parse entries */
    int offset = 0;
    while (offset < n) {
        struct dirent64 *de = (struct dirent64 *)(dirent_buf + offset);
        if (de->d_reclen == 0) break;
        if (de->d_type == 1) {
            printf("\033[1;34m%s/\033[0m  ", de->d_name);
        } else {
            printf("%s  ", de->d_name);
        }
        offset += de->d_reclen;
    }
    printf("\n");

    if (dirfd >= 3) close(dirfd);
}

static void cmd_cd(const char *path) {
    if (!*path) path = "/home";
    if (chdir(path) != 0) {
        printf("cd: no such directory: %s\n", path);
    }
}

static void cmd_cat(const char *path) {
    if (!*path) {
        printf("cat: missing operand\n");
        return;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        printf("cat: %s: No such file\n", path);
        return;
    }
    char buf[512];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        write(1, buf, n);
    }
    close(fd);
}

static void cmd_echo(const char *text, const char *redirect_file, int append) {
    if (redirect_file && *redirect_file) {
        int flags = O_WRONLY | O_CREAT;
        flags |= append ? O_APPEND : O_TRUNC;
        int fd = open(redirect_file, flags);
        if (fd < 0) {
            printf("echo: cannot open %s\n", redirect_file);
            return;
        }
        write(fd, text, strlen(text));
        write(fd, "\n", 1);
        close(fd);
    } else {
        printf("%s\n", text);
    }
}

static void cmd_mkdir_sh(const char *path) {
    if (!*path) {
        printf("mkdir: missing operand\n");
        return;
    }
    if (mkdir(path) != 0) {
        printf("mkdir: cannot create directory '%s'\n", path);
    }
}

static void cmd_rm(const char *path) {
    if (!*path) {
        printf("rm: missing operand\n");
        return;
    }
    if (unlink(path) != 0) {
        printf("rm: cannot remove '%s': No such file\n", path);
    }
}

static void cmd_rmdir_sh(const char *path) {
    if (!*path) {
        printf("rmdir: missing operand\n");
        return;
    }
    if (rmdir(path) != 0) {
        printf("rmdir: failed to remove '%s'\n", path);
    }
}

static void cmd_touch(const char *path) {
    if (!*path) {
        printf("touch: missing operand\n");
        return;
    }
    int fd = open(path, O_WRONLY | O_CREAT);
    if (fd >= 0) {
        close(fd);
    } else {
        printf("touch: cannot create '%s'\n", path);
    }
}

static void cmd_wc(const char *path) {
    if (!*path) {
        printf("wc: missing operand\n");
        return;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        printf("wc: %s: No such file\n", path);
        return;
    }

    volatile int lines = 0, words = 0, chars = 0;
    int in_word = 0;
    char buf[512];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t i = 0; i < n; i++) {
            chars++;
            if (buf[i] == '\n') lines++;
            if (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\t') {
                in_word = 0;
            } else {
                if (!in_word) words++;
                in_word = 1;
            }
        }
    }
    close(fd);
    printf("  %d  %d  %d %s\n", (int)lines, (int)words, (int)chars, path);
}

static void cmd_cp(const char *args) {
    /* Parse src and dst from "src dst" */
    char src[256], dst[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && args[i] != '\t' && i < 255) {
        src[i] = args[i]; i++;
    }
    src[i] = '\0';
    while (args[i] == ' ' || args[i] == '\t') i++;
    int j = 0;
    while (args[i] && args[i] != ' ' && args[i] != '\t' && j < 255) {
        dst[j++] = args[i++];
    }
    dst[j] = '\0';

    if (!src[0] || !dst[0]) {
        printf("cp: missing operand\n");
        return;
    }

    int sfd = open(src, O_RDONLY);
    if (sfd < 0) {
        printf("cp: %s: No such file\n", src);
        return;
    }
    int dfd = open(dst, O_WRONLY | O_CREAT | O_TRUNC);
    if (dfd < 0) {
        printf("cp: cannot create '%s'\n", dst);
        close(sfd);
        return;
    }

    char buf[512];
    ssize_t n;
    while ((n = read(sfd, buf, sizeof(buf))) > 0) {
        write(dfd, buf, n);
    }
    close(sfd);
    close(dfd);
}

static void cmd_head(const char *args) {
    char path[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && args[i] != '\t' && i < 255) {
        path[i] = args[i]; i++;
    }
    path[i] = '\0';
    while (args[i] == ' ' || args[i] == '\t') i++;

    int n_lines = 10;
    if (args[i]) {
        n_lines = (int)atoi_libc(args + i);
        if (n_lines <= 0) n_lines = 10;
    }

    if (!path[0]) {
        printf("head: missing operand\n");
        return;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        printf("head: %s: No such file\n", path);
        return;
    }

    int lines_printed = 0;
    char buf[512];
    ssize_t n;
    while (lines_printed < n_lines && (n = read(fd, buf, sizeof(buf))) > 0) {
        for (int j = 0; j < (int)n && lines_printed < n_lines; j++) {
            putchar(buf[j]);
            if (buf[j] == '\n') lines_printed++;
        }
    }
    close(fd);
}

static void cmd_cc(const char *path) {
    if (!*path) {
        printf("cc: missing operand\n");
        return;
    }

    /* Generate output path: foo.c → /bin/foo */
    char out_path[256] = "/bin/";
    const char *base = path;
    /* Find last / */
    const char *p = path;
    while (*p) { if (*p == '/') base = p + 1; p++; }
    /* Copy base name without .c extension */
    int i = 5;
    while (*base && *base != '.' && i < 254) {
        out_path[i++] = *base++;
    }
    out_path[i] = '\0';

    printf("Compiling %s -> %s\n", path, out_path);
    int result = sys_compile(path, out_path);
    if (result == 0) {
        printf("OK\n");
    } else {
        printf("cc: compilation failed\n");
    }
}

static void cmd_run(const char *path) {
    if (!*path) {
        printf("run: missing operand\n");
        return;
    }
    printf("Executing %s...\n", path);
    int result = sys_exec(path);
    if (result != 0) {
        printf("run: cannot execute '%s'\n", path);
    }
    /* If exec succeeds, control transfers to the new binary */
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INLINE SHA-256 (for sha256 command)                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define SHA_ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define SHA_CH(x,y,z)  (((x)&(y))^((~(x))&(z)))
#define SHA_MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define SHA_EP0(x) (SHA_ROTR(x,2)^SHA_ROTR(x,13)^SHA_ROTR(x,22))
#define SHA_EP1(x) (SHA_ROTR(x,6)^SHA_ROTR(x,11)^SHA_ROTR(x,25))
#define SHA_SIG0(x) (SHA_ROTR(x,7)^SHA_ROTR(x,18)^((x)>>3))
#define SHA_SIG1(x) (SHA_ROTR(x,17)^SHA_ROTR(x,19)^((x)>>10))

static const unsigned int sha_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha_transform(unsigned int state[8], const unsigned char block[64]) {
    unsigned int w[64], a, b, c, d, e, f, g, h, t1, t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((unsigned int)block[i*4]<<24)|((unsigned int)block[i*4+1]<<16)|
               ((unsigned int)block[i*4+2]<<8)|((unsigned int)block[i*4+3]);
    for (i = 16; i < 64; i++)
        w[i] = SHA_SIG1(w[i-2]) + w[i-7] + SHA_SIG0(w[i-15]) + w[i-16];
    a=state[0]; b=state[1]; c=state[2]; d=state[3];
    e=state[4]; f=state[5]; g=state[6]; h=state[7];
    for (i = 0; i < 64; i++) {
        t1 = h + SHA_EP1(e) + SHA_CH(e,f,g) + sha_k[i] + w[i];
        t2 = SHA_EP0(a) + SHA_MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

static void cmd_sha256(const char *path) {
    if (!*path) {
        printf("sha256: missing operand\n");
        return;
    }
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        printf("sha256: %s: No such file\n", path);
        return;
    }
    unsigned int state[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                             0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    unsigned char buf[512];
    unsigned char block[64];
    unsigned long total_len = 0;
    int block_pos = 0;
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t i = 0; i < n; i++) {
            block[block_pos++] = buf[i];
            total_len++;
            if (block_pos == 64) {
                sha_transform(state, block);
                block_pos = 0;
            }
        }
    }
    close(fd);
    /* Padding */
    block[block_pos] = 0x80;
    if (block_pos >= 56) {
        memset(block + block_pos + 1, 0, 63 - block_pos);
        sha_transform(state, block);
        memset(block, 0, 56);
    } else {
        memset(block + block_pos + 1, 0, 55 - block_pos);
    }
    unsigned long bits = total_len * 8;
    for (int i = 0; i < 8; i++) block[56+i] = (unsigned char)(bits >> (56 - i*8));
    sha_transform(state, block);
    /* Print hex digest */
    for (int i = 0; i < 8; i++) {
        printf("%08x", state[i]);
    }
    printf("  %s\n", path);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* NEW COMMANDS: grep, sort, uniq, tee, ps                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void cmd_grep(const char *args) {
    /* Parse: pattern [file] */
    char pattern[128];
    char filepath[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && args[i] != '\t' && i < 127) {
        pattern[i] = args[i]; i++;
    }
    pattern[i] = '\0';
    while (args[i] == ' ' || args[i] == '\t') i++;
    int j = 0;
    while (args[i] && args[i] != ' ' && j < 255) {
        filepath[j++] = args[i++];
    }
    filepath[j] = '\0';

    if (!pattern[0]) {
        printf("grep: missing pattern\n");
        return;
    }

    int fd = 0;  /* stdin */
    if (filepath[0]) {
        fd = open(filepath, O_RDONLY);
        if (fd < 0) {
            printf("grep: %s: No such file\n", filepath);
            return;
        }
    }

    char line[512];
    int pos = 0;
    char buf[256];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t k = 0; k < n; k++) {
            if (buf[k] == '\n' || pos >= (int)sizeof(line) - 1) {
                line[pos] = '\0';
                if (strstr(line, pattern)) {
                    printf("%s\n", line);
                }
                pos = 0;
            } else {
                line[pos++] = buf[k];
            }
        }
    }
    /* Handle last line without newline */
    if (pos > 0) {
        line[pos] = '\0';
        if (strstr(line, pattern)) {
            printf("%s\n", line);
        }
    }
    if (fd > 2) close(fd);
}

static void cmd_sort(const char *args) {
    char filepath[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && i < 255) filepath[i] = args[i], i++;
    filepath[i] = '\0';

    int fd = 0;
    if (filepath[0]) {
        fd = open(filepath, O_RDONLY);
        if (fd < 0) {
            printf("sort: %s: No such file\n", filepath);
            return;
        }
    }

    /* Read all lines (up to 256 lines x 128 chars) */
    #define SORT_MAX_LINES 256
    #define SORT_LINE_LEN 128
    static char lines[SORT_MAX_LINES][SORT_LINE_LEN];
    int nlines = 0;
    int pos = 0;
    char buf[256];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t k = 0; k < n; k++) {
            if (buf[k] == '\n' || pos >= SORT_LINE_LEN - 1) {
                lines[nlines][pos] = '\0';
                if (nlines < SORT_MAX_LINES - 1) nlines++;
                pos = 0;
            } else {
                lines[nlines][pos++] = buf[k];
            }
        }
    }
    if (pos > 0) {
        lines[nlines][pos] = '\0';
        nlines++;
    }
    if (fd > 2) close(fd);

    /* Insertion sort */
    for (int a = 1; a < nlines; a++) {
        char tmp[SORT_LINE_LEN];
        memcpy(tmp, lines[a], SORT_LINE_LEN);
        int b = a;
        while (b > 0 && strcmp(lines[b-1], tmp) > 0) {
            memcpy(lines[b], lines[b-1], SORT_LINE_LEN);
            b--;
        }
        memcpy(lines[b], tmp, SORT_LINE_LEN);
    }

    for (int a = 0; a < nlines; a++) {
        printf("%s\n", lines[a]);
    }
}

static void cmd_uniq(const char *args) {
    char filepath[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && i < 255) filepath[i] = args[i], i++;
    filepath[i] = '\0';

    int fd = 0;
    if (filepath[0]) {
        fd = open(filepath, O_RDONLY);
        if (fd < 0) {
            printf("uniq: %s: No such file\n", filepath);
            return;
        }
    }

    char prev[512] = {0};
    char line[512];
    int pos = 0;
    char buf[256];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (ssize_t k = 0; k < n; k++) {
            if (buf[k] == '\n' || pos >= (int)sizeof(line) - 1) {
                line[pos] = '\0';
                if (strcmp(line, prev) != 0) {
                    printf("%s\n", line);
                    strcpy(prev, line);
                }
                pos = 0;
            } else {
                line[pos++] = buf[k];
            }
        }
    }
    if (pos > 0) {
        line[pos] = '\0';
        if (strcmp(line, prev) != 0) printf("%s\n", line);
    }
    if (fd > 2) close(fd);
}

static void cmd_tee(const char *args) {
    char filepath[256];
    int i = 0;
    while (args[i] && args[i] != ' ' && i < 255) filepath[i] = args[i], i++;
    filepath[i] = '\0';

    if (!filepath[0]) {
        printf("tee: missing operand\n");
        return;
    }

    int outfd = open(filepath, O_WRONLY | O_CREAT | O_TRUNC);
    if (outfd < 0) {
        printf("tee: cannot open %s\n", filepath);
        return;
    }

    char buf[256];
    ssize_t n;
    while ((n = read(0, buf, sizeof(buf))) > 0) {
        write(1, buf, n);     /* stdout */
        write(outfd, buf, n); /* file */
    }
    close(outfd);
}

static void cmd_ps(void) {
    /* SYS_PS with buf_addr=0 tells Python to write to stdout */
    register long x0 __asm__("x0") = 0;
    register long x8 __asm__("x8") = SYS_PS;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8) : "memory");
}

static void cmd_kill(const char *args) {
    if (!*args) {
        printf("kill: usage: kill [-9] <pid>\n");
        return;
    }
    int sig = SIGTERM;
    const char *p = args;
    if (*p == '-') {
        p++;
        sig = (int)strtol(p, NULL, 10);
        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;
    }
    if (!*p) {
        printf("kill: usage: kill [-9] <pid>\n");
        return;
    }
    int pid = (int)strtol(p, NULL, 10);
    if (pid <= 0) {
        printf("kill: invalid PID\n");
        return;
    }
    int result = kill(pid, sig);
    if (result < 0) {
        printf("kill: no such process: %d\n", pid);
    }
}

static void cmd_env(void) {
    for (int i = 0; i < env_count; i++) {
        printf("%s=%s\n", env_keys[i], env_vals[i]);
    }
}

static void cmd_export(const char *arg) {
    if (!*arg) {
        printf("export: usage: export KEY=VALUE\n");
        return;
    }
    /* Find = */
    const char *eq = strchr(arg, '=');
    if (!eq) {
        printf("export: usage: export KEY=VALUE\n");
        return;
    }
    char key[64];
    int klen = eq - arg;
    if (klen >= 64) klen = 63;
    memcpy(key, arg, klen);
    key[klen] = '\0';
    env_set(key, eq + 1);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* COMMAND DISPATCH                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Execute a single built-in command. Returns 0 on success, -1 on not found,
 * 1 for "exit" signal. */
static int exec_builtin(char *cmd_str) {
    trim_trailing(cmd_str);
    while (*cmd_str == ' ') cmd_str++;
    if (*cmd_str == '\0') return 0;

    /* Handle output redirection */
    char *redirect_file = NULL;
    int redir_mode = split_redirect(cmd_str, &redirect_file);
    int saved_stdout = -1;

    if (redir_mode > 0 && redirect_file && *redirect_file) {
        int flags = O_WRONLY | O_CREAT;
        flags |= (redir_mode == 2) ? O_APPEND : O_TRUNC;
        int rfd = open(redirect_file, flags);
        if (rfd >= 0) {
            saved_stdout = dup2(1, 99);  /* Save stdout to fd 99 */
            dup2(rfd, 1);                /* Redirect stdout to file */
            close(rfd);
        }
    }

    const char *args = skip_to_arg(cmd_str);
    int ret = 0;

    if (strcmp(cmd_str, "exit") == 0) {
        ret = 1;
    } else if (strcmp(cmd_str, "help") == 0 || starts_with(cmd_str, "help ")) {
        cmd_help();
    } else if (strcmp(cmd_str, "pwd") == 0) {
        cmd_pwd();
    } else if (strcmp(cmd_str, "env") == 0) {
        cmd_env();
    } else if (strcmp(cmd_str, "ps") == 0) {
        cmd_ps();
    } else if (starts_with(cmd_str, "kill")) {
        cmd_kill(args);
    } else if (starts_with(cmd_str, "cd")) {
        cmd_cd(args);
    } else if (starts_with(cmd_str, "ls")) {
        cmd_ls(args);
    } else if (starts_with(cmd_str, "cat ") || strcmp(cmd_str, "cat") == 0) {
        cmd_cat(args);
    } else if (starts_with(cmd_str, "echo")) {
        /* Echo with redirect already handled above — pass text directly */
        cmd_echo(args, NULL, 0);
    } else if (starts_with(cmd_str, "mkdir ")) {
        cmd_mkdir_sh(args);
    } else if (starts_with(cmd_str, "rm ") && !starts_with(cmd_str, "rmdir")) {
        cmd_rm(args);
    } else if (starts_with(cmd_str, "rmdir ")) {
        cmd_rmdir_sh(args);
    } else if (starts_with(cmd_str, "touch ")) {
        cmd_touch(args);
    } else if (starts_with(cmd_str, "wc ")) {
        cmd_wc(args);
    } else if (starts_with(cmd_str, "cp ")) {
        cmd_cp(args);
    } else if (starts_with(cmd_str, "head ")) {
        cmd_head(args);
    } else if (starts_with(cmd_str, "cc ")) {
        cmd_cc(args);
    } else if (starts_with(cmd_str, "run ")) {
        cmd_run(args);
    } else if (starts_with(cmd_str, "export ")) {
        cmd_export(args);
    } else if (starts_with(cmd_str, "sha256 ")) {
        cmd_sha256(args);
    } else if (starts_with(cmd_str, "grep")) {
        cmd_grep(args);
    } else if (starts_with(cmd_str, "sort")) {
        cmd_sort(args);
    } else if (starts_with(cmd_str, "uniq")) {
        cmd_uniq(args);
    } else if (starts_with(cmd_str, "tee ")) {
        cmd_tee(args);
    } else if (starts_with(cmd_str, "ed")) {
        printf("ed: use 'run /bin/ed' (compile with 'cc /tools/ed.c' first)\n");
    } else {
        printf("%s: command not found\n", cmd_str);
        ret = -1;
    }

    /* Restore stdout if redirected */
    if (saved_stdout >= 0) {
        dup2(saved_stdout, 1);
        close(saved_stdout);
    }

    return ret;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* PIPELINE EXECUTION                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_PIPELINE 8

/* Split command line on '|', return number of stages */
static int parse_pipeline(char *cmdline, char *cmds[], int max) {
    int count = 0;
    cmds[count++] = cmdline;
    for (char *p = cmdline; *p; p++) {
        if (*p == '|') {
            *p = '\0';
            trim_trailing(cmds[count - 1]);
            p++;
            while (*p == ' ') p++;
            if (count < max) cmds[count++] = p;
        }
    }
    return count;
}

/* Execute a pipeline using fork/pipe/dup2 */
static int exec_pipeline(char *cmds[], int n) {
    if (n == 1) {
        return exec_builtin(cmds[0]);
    }

    int prev_rd = -1;
    int pids[MAX_PIPELINE];
    int npids = 0;

    for (int i = 0; i < n; i++) {
        int pipefd[2] = {-1, -1};
        if (i < n - 1) {
            pipe(pipefd);
        }

        int pid = fork();
        if (pid == 0) {
            /* Child process */
            if (prev_rd != -1) {
                dup2(prev_rd, 0);  /* stdin = previous pipe read end */
                close(prev_rd);
            }
            if (i < n - 1) {
                dup2(pipefd[1], 1);  /* stdout = this pipe write end */
                close(pipefd[0]);
                close(pipefd[1]);
            }
            exec_builtin(cmds[i]);
            exit(0);
        }

        /* Parent */
        if (pid > 0 && npids < MAX_PIPELINE) pids[npids++] = pid;
        if (prev_rd != -1) close(prev_rd);
        if (i < n - 1) {
            close(pipefd[1]);
            prev_rd = pipefd[0];
        }
    }

    /* Wait for all children */
    for (int i = 0; i < npids; i++) {
        waitpid(pids[i], NULL, 0);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* COMMAND CHAINING (;, &&, ||)                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int exec_chained(char *line) {
    /* Process ; first (unconditional sequencing) */
    char *cmds[16];
    int ncmds = 0;

    /* Operator types: 0=none, 1=;, 2=&&, 3=|| */
    int ops[16];
    ops[0] = 0;

    char *p = line;
    cmds[ncmds++] = p;

    while (*p) {
        if (*p == ';') {
            *p = '\0';
            p++;
            while (*p == ' ') p++;
            if (ncmds < 16) {
                ops[ncmds] = 1;  /* ; */
                cmds[ncmds++] = p;
            }
        } else if (*p == '&' && *(p+1) == '&') {
            *p = '\0';
            p += 2;
            while (*p == ' ') p++;
            if (ncmds < 16) {
                ops[ncmds] = 2;  /* && */
                cmds[ncmds++] = p;
            }
        } else if (*p == '|' && *(p+1) == '|') {
            *p = '\0';
            p += 2;
            while (*p == ' ') p++;
            if (ncmds < 16) {
                ops[ncmds] = 3;  /* || */
                cmds[ncmds++] = p;
            }
        } else {
            p++;
        }
    }

    int last_ret = 0;
    for (int i = 0; i < ncmds; i++) {
        /* Check operator condition */
        if (i > 0) {
            if (ops[i] == 2 && last_ret != 0) continue;   /* && but prev failed */
            if (ops[i] == 3 && last_ret == 0) continue;    /* || but prev succeeded */
        }

        trim_trailing(cmds[i]);
        if (cmds[i][0] == '\0') continue;

        /* Check for pipeline within this command */
        char *pipe_cmds[MAX_PIPELINE];
        int nstages = parse_pipeline(cmds[i], pipe_cmds, MAX_PIPELINE);
        last_ret = exec_pipeline(pipe_cmds, nstages);
    }

    return last_ret;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MAIN LOOP                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

static char input_buf[512];

int main(void) {
    env_init();

    /* Print MOTD */
    cmd_cat("/etc/motd");
    printf("\nType 'help' for commands.\n\n");

    while (1) {
        /* CWD-aware prompt */
        char cwd[256];
        if (getcwd(cwd, sizeof(cwd)) != 0) {
            strcpy(cwd, "?");
        }
        printf("gpu:%s$ ", cwd);

        /* Read a line */
        ssize_t n = sys_read(0, input_buf, sizeof(input_buf) - 1);
        if (n <= 0) break;

        input_buf[n] = '\0';
        trim_trailing(input_buf);
        if (input_buf[0] == '\0') continue;

        /* Check for "exit" first */
        if (strcmp(input_buf, "exit") == 0) {
            printf("Goodbye!\n");
            break;
        }

        /* Check for background job (&) */
        int len = strlen(input_buf);
        int is_bg = 0;
        if (len > 0 && input_buf[len - 1] == '&') {
            /* Make sure it's not && */
            if (len < 2 || input_buf[len - 2] != '&') {
                is_bg = 1;
                input_buf[len - 1] = '\0';
                trim_trailing(input_buf);
            }
        }

        if (is_bg) {
            int pid = fork();
            if (pid == 0) {
                /* Child: execute command and exit */
                exec_chained(input_buf);
                exit(0);
            } else if (pid > 0) {
                printf("[%d] started\n", pid);
            } else {
                printf("fork failed\n");
                exec_chained(input_buf);
            }
        } else {
            /* Foreground: execute with chaining support */
            int ret = exec_chained(input_buf);
            if (ret == 1) {
                printf("Goodbye!\n");
                break;
            }
        }
    }

    return 0;
}
