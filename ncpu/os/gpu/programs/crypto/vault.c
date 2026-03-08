/*
 * Password Vault — Freestanding C for ARM64 Metal GPU kernel.
 *
 * SHA-256 key derivation + XOR stream cipher for password storage.
 * Reads/writes encrypted vault entries to the GPU filesystem.
 *
 * Features:
 *   - SHA-256 key derivation from master password
 *   - XOR encryption/decryption of stored passwords
 *   - Up to 32 vault entries, each with a site name and encrypted password
 *   - File persistence: saves vault to /etc/vault.dat, loads on init
 *   - Demo mode: seeds 3 entries, lists them, retrieves and verifies
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/crypto/vault.c
 *          -o /tmp/vault.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* =========================================================================== */
/* SHA-256 CONSTANTS                                                           */
/* =========================================================================== */

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* =========================================================================== */
/* SHA-256 IMPLEMENTATION                                                      */
/* =========================================================================== */

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x,  2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x,  6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x,  7) ^ ROTR(x, 18) ^ ((x) >>  3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

struct sha256_state {
    uint32_t h[8];
    unsigned char buf[64];
    uint32_t buflen;
    uint64_t total;
};

static void sha256_init(struct sha256_state *s) {
    s->h[0] = 0x6a09e667;
    s->h[1] = 0xbb67ae85;
    s->h[2] = 0x3c6ef372;
    s->h[3] = 0xa54ff53a;
    s->h[4] = 0x510e527f;
    s->h[5] = 0x9b05688c;
    s->h[6] = 0x1f83d9ab;
    s->h[7] = 0x5be0cd19;
    s->buflen = 0;
    s->total = 0;
}

static void sha256_transform(struct sha256_state *s, const unsigned char *block) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    /* Prepare message schedule: big-endian decode for first 16 words */
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24)
             | ((uint32_t)block[i * 4 + 1] << 16)
             | ((uint32_t)block[i * 4 + 2] <<  8)
             | ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    a = s->h[0]; b = s->h[1]; c = s->h[2]; d = s->h[3];
    e = s->h[4]; f = s->h[5]; g = s->h[6]; h = s->h[7];

    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + W[i];
        uint32_t T2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void sha256_update(struct sha256_state *s, const unsigned char *data,
                          uint32_t len) {
    s->total += len;
    while (len > 0) {
        uint32_t space = 64 - s->buflen;
        uint32_t copy = (len < space) ? len : space;
        memcpy(s->buf + s->buflen, data, copy);
        s->buflen += copy;
        data += copy;
        len -= copy;
        if (s->buflen == 64) {
            sha256_transform(s, s->buf);
            s->buflen = 0;
        }
    }
}

static void sha256_final(struct sha256_state *s, unsigned char digest[32]) {
    /* Pad: append 0x80, then zeros, then 64-bit big-endian bit length */
    uint64_t bits = s->total * 8;
    unsigned char pad = 0x80;
    sha256_update(s, &pad, 1);

    /* Zero-pad until buffer is 56 bytes (room for 8-byte length) */
    unsigned char zero = 0x00;
    while (s->buflen != 56) {
        sha256_update(s, &zero, 1);
    }

    /* Append 64-bit bit count in big-endian */
    unsigned char len_bytes[8];
    for (int i = 7; i >= 0; i--) {
        len_bytes[i] = (unsigned char)(bits & 0xFF);
        bits >>= 8;
    }
    sha256_update(s, len_bytes, 8);

    /* Write digest in big-endian */
    for (int i = 0; i < 8; i++) {
        digest[i * 4 + 0] = (unsigned char)(s->h[i] >> 24);
        digest[i * 4 + 1] = (unsigned char)(s->h[i] >> 16);
        digest[i * 4 + 2] = (unsigned char)(s->h[i] >>  8);
        digest[i * 4 + 3] = (unsigned char)(s->h[i]);
    }
}

/*
 * sha256 — convenience wrapper: hash arbitrary data into a 32-byte digest.
 */
static void sha256(const unsigned char *data, uint32_t len,
                   unsigned char digest[32]) {
    struct sha256_state ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, digest);
}

/* =========================================================================== */
/* HEX FORMATTING                                                             */
/* =========================================================================== */

static const char hex_chars[] = "0123456789abcdef";

/*
 * Format a byte array as a hex string. Caller must provide a buffer
 * of at least (len * 2 + 1) bytes.
 */
static void hex_encode(const unsigned char *data, int len, char *out) {
    for (int i = 0; i < len; i++) {
        out[i * 2]     = hex_chars[(data[i] >> 4) & 0x0F];
        out[i * 2 + 1] = hex_chars[data[i] & 0x0F];
    }
    out[len * 2] = '\0';
}

/* =========================================================================== */
/* VAULT DATA STRUCTURES                                                       */
/* =========================================================================== */

#define VAULT_MAX_ENTRIES  32
#define VAULT_NAME_LEN     48
#define VAULT_PASS_LEN     64
#define VAULT_MAGIC        0x56415554  /* "VAUT" */

struct vault_entry {
    char  name[VAULT_NAME_LEN];             /* plaintext site/service name  */
    unsigned char encrypted[VAULT_PASS_LEN]; /* XOR-encrypted password       */
    uint32_t pass_len;                       /* original plaintext length    */
};

struct vault {
    uint32_t magic;
    uint32_t count;
    unsigned char key[32];                   /* SHA-256 derived key          */
    struct vault_entry entries[VAULT_MAX_ENTRIES];
};

static struct vault the_vault;

/* =========================================================================== */
/* XOR STREAM CIPHER                                                           */
/* =========================================================================== */

/*
 * XOR encrypt/decrypt using a repeating key stream derived from the
 * master key concatenated with the entry name. This makes each entry's
 * cipher stream unique even under the same master password.
 *
 * key_stream[i] = SHA256(master_key || name || counter_block)[i % 32]
 *
 * For simplicity, we derive a per-entry key by hashing (master_key + name)
 * and XOR the password against that 32-byte per-entry key, repeating as needed.
 */
static void derive_entry_key(const unsigned char master_key[32],
                             const char *name,
                             unsigned char entry_key[32]) {
    struct sha256_state ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, master_key, 32);
    sha256_update(&ctx, (const unsigned char *)name, strlen(name));
    sha256_final(&ctx, entry_key);
}

static void xor_crypt(const unsigned char *input, unsigned char *output,
                      uint32_t len, const unsigned char key[32]) {
    for (uint32_t i = 0; i < len; i++) {
        output[i] = input[i] ^ key[i % 32];
    }
}

/* =========================================================================== */
/* VAULT OPERATIONS                                                            */
/* =========================================================================== */

/*
 * vault_init — derive the 256-bit encryption key from the master password.
 */
static void vault_init(const char *master_password) {
    memset(&the_vault, 0, sizeof(the_vault));
    the_vault.magic = VAULT_MAGIC;
    the_vault.count = 0;
    sha256((const unsigned char *)master_password, strlen(master_password),
           the_vault.key);
}

/*
 * vault_add — encrypt and store a password under the given name.
 * Returns 0 on success, -1 if vault is full, -2 if password too long.
 */
static int vault_add(const char *name, const char *password) {
    if (the_vault.count >= VAULT_MAX_ENTRIES) {
        printf("[vault] error: vault full (%d/%d entries)\n",
               the_vault.count, VAULT_MAX_ENTRIES);
        return -1;
    }

    uint32_t plen = strlen(password);
    if (plen >= VAULT_PASS_LEN) {
        printf("[vault] error: password too long (max %d chars)\n",
               VAULT_PASS_LEN - 1);
        return -2;
    }

    struct vault_entry *e = &the_vault.entries[the_vault.count];

    /* Store site name */
    strncpy(e->name, name, VAULT_NAME_LEN - 1);
    e->name[VAULT_NAME_LEN - 1] = '\0';

    /* Derive per-entry key and encrypt */
    unsigned char entry_key[32];
    derive_entry_key(the_vault.key, e->name, entry_key);

    memset(e->encrypted, 0, VAULT_PASS_LEN);
    e->pass_len = plen;
    xor_crypt((const unsigned char *)password, e->encrypted, plen, entry_key);

    the_vault.count++;
    return 0;
}

/*
 * vault_get — decrypt and copy a stored password into buf.
 * Returns 0 on success, -1 if entry not found.
 */
static int vault_get(const char *name, char *buf, uint32_t bufsize) {
    for (uint32_t i = 0; i < the_vault.count; i++) {
        if (strcmp(the_vault.entries[i].name, name) == 0) {
            struct vault_entry *e = &the_vault.entries[i];

            unsigned char entry_key[32];
            derive_entry_key(the_vault.key, e->name, entry_key);

            uint32_t copy_len = e->pass_len;
            if (copy_len >= bufsize) copy_len = bufsize - 1;

            xor_crypt(e->encrypted, (unsigned char *)buf, copy_len, entry_key);
            buf[copy_len] = '\0';
            return 0;
        }
    }
    return -1;
}

/*
 * vault_list — print all stored entry names to stdout.
 */
static void vault_list(void) {
    if (the_vault.count == 0) {
        printf("  (vault is empty)\n");
        return;
    }
    for (uint32_t i = 0; i < the_vault.count; i++) {
        printf("  [%d] %s (%d chars)\n", i, the_vault.entries[i].name,
               the_vault.entries[i].pass_len);
    }
}

/*
 * vault_delete — remove an entry by name, compacting the array.
 * Returns 0 on success, -1 if not found.
 */
static int vault_delete(const char *name) {
    for (uint32_t i = 0; i < the_vault.count; i++) {
        if (strcmp(the_vault.entries[i].name, name) == 0) {
            /* Shift remaining entries down */
            for (uint32_t j = i; j < the_vault.count - 1; j++) {
                memcpy(&the_vault.entries[j], &the_vault.entries[j + 1],
                       sizeof(struct vault_entry));
            }
            /* Clear the last slot */
            memset(&the_vault.entries[the_vault.count - 1], 0,
                   sizeof(struct vault_entry));
            the_vault.count--;
            return 0;
        }
    }
    return -1;
}

/* =========================================================================== */
/* FILESYSTEM PERSISTENCE                                                      */
/* =========================================================================== */

#define VAULT_FILE "/etc/vault.dat"

/*
 * vault_save — write the vault to the GPU filesystem.
 * Format: [magic:4][count:4][entries...] (key is NOT saved to disk).
 */
static int vault_save(void) {
    int fd = open(VAULT_FILE, O_WRONLY | O_CREAT | O_TRUNC);
    if (fd < 0) {
        printf("[vault] error: cannot open %s for writing\n", VAULT_FILE);
        return -1;
    }

    /* Write header: magic + count */
    write(fd, &the_vault.magic, sizeof(uint32_t));
    write(fd, &the_vault.count, sizeof(uint32_t));

    /* Write each entry */
    for (uint32_t i = 0; i < the_vault.count; i++) {
        write(fd, &the_vault.entries[i], sizeof(struct vault_entry));
    }

    close(fd);
    return 0;
}

/*
 * vault_load — read the vault from the GPU filesystem.
 * The master key must already be set (via vault_init) before calling.
 * Returns 0 on success, -1 on error, -2 on bad magic.
 */
static int vault_load(void) {
    int fd = open(VAULT_FILE, O_RDONLY);
    if (fd < 0) return -1;  /* file does not exist — not an error for first run */

    uint32_t magic = 0, count = 0;
    ssize_t n;

    n = read(fd, &magic, sizeof(uint32_t));
    if (n != sizeof(uint32_t) || magic != VAULT_MAGIC) {
        close(fd);
        return -2;
    }

    n = read(fd, &count, sizeof(uint32_t));
    if (n != sizeof(uint32_t) || count > VAULT_MAX_ENTRIES) {
        close(fd);
        return -2;
    }

    the_vault.count = count;
    for (uint32_t i = 0; i < count; i++) {
        n = read(fd, &the_vault.entries[i], sizeof(struct vault_entry));
        if (n != sizeof(struct vault_entry)) {
            close(fd);
            return -2;
        }
    }

    close(fd);
    return 0;
}

/* =========================================================================== */
/* VERIFICATION HELPERS                                                        */
/* =========================================================================== */

/*
 * Print a SHA-256 digest as hex, for visual verification.
 */
static void print_hash(const char *label, const unsigned char digest[32]) {
    char hex[65];
    hex_encode(digest, 32, hex);
    printf("  %s: %s\n", label, hex);
}

/*
 * Verify that a retrieved password matches the expected value.
 */
static int verify_password(const char *name, const char *expected) {
    char retrieved[VAULT_PASS_LEN];
    int rc = vault_get(name, retrieved, sizeof(retrieved));
    if (rc != 0) {
        printf("  FAIL: '%s' not found in vault\n", name);
        return 0;
    }
    if (strcmp(retrieved, expected) == 0) {
        printf("  PASS: '%s' -> '%s'\n", name, retrieved);
        return 1;
    } else {
        printf("  FAIL: '%s' -> got '%s', expected '%s'\n",
               name, retrieved, expected);
        return 0;
    }
}

/* =========================================================================== */
/* MAIN — DEMO MODE                                                            */
/* =========================================================================== */

int main(void) {
    printf("========================================\n");
    printf("  nCPU Password Vault (SHA-256 + XOR)\n");
    printf("  Running on ARM64 Metal GPU Kernel\n");
    printf("========================================\n\n");

    /* ---- Phase 1: Key Derivation ---------------------------------------- */
    const char *master_password = "correct-horse-battery-staple";

    printf("[1] Key Derivation\n");
    printf("  Master password: \"%s\"\n", master_password);

    vault_init(master_password);

    print_hash("Derived key", the_vault.key);

    /* Also show that different passwords produce different keys */
    unsigned char alt_hash[32];
    sha256((const unsigned char *)"wrong-password", 14, alt_hash);
    print_hash("Alt key    ", alt_hash);
    printf("\n");

    /* ---- Phase 2: Store Entries ----------------------------------------- */
    printf("[2] Storing Entries\n");

    vault_add("github.com",    "gh-t0k3n-Sup3rS3cr3t!");
    printf("  Added: github.com\n");

    vault_add("bank.example",  "My$ecur3B@nkP@ss99");
    printf("  Added: bank.example\n");

    vault_add("email.service", "Em@il_2024_xKz!pQ");
    printf("  Added: email.service\n");

    printf("  Total entries: %d\n\n", the_vault.count);

    /* ---- Phase 3: List Entries ------------------------------------------ */
    printf("[3] Vault Contents\n");
    vault_list();
    printf("\n");

    /* ---- Phase 4: Retrieve & Verify ------------------------------------- */
    printf("[4] Retrieve & Verify\n");

    int pass_count = 0;
    pass_count += verify_password("github.com",    "gh-t0k3n-Sup3rS3cr3t!");
    pass_count += verify_password("bank.example",  "My$ecur3B@nkP@ss99");
    pass_count += verify_password("email.service", "Em@il_2024_xKz!pQ");

    /* Test retrieval of non-existent entry */
    char buf[VAULT_PASS_LEN];
    int rc = vault_get("nonexistent.site", buf, sizeof(buf));
    if (rc == -1) {
        printf("  PASS: 'nonexistent.site' correctly returned not-found\n");
        pass_count++;
    } else {
        printf("  FAIL: 'nonexistent.site' should not exist\n");
    }
    printf("\n");

    /* ---- Phase 5: Persistence ------------------------------------------- */
    printf("[5] Persistence (Save/Load)\n");

    /* Create /etc directory */
    mkdir("/etc");

    /* Save vault to filesystem */
    rc = vault_save();
    if (rc == 0) {
        printf("  Saved vault to %s (%d entries)\n", VAULT_FILE, the_vault.count);
    } else {
        printf("  WARNING: could not save vault\n");
    }

    /* Clear in-memory vault, re-derive key, reload from disk */
    unsigned char saved_key[32];
    memcpy(saved_key, the_vault.key, 32);

    memset(&the_vault, 0, sizeof(the_vault));
    the_vault.magic = VAULT_MAGIC;

    /* Re-derive key from master password (simulates new session) */
    sha256((const unsigned char *)master_password, strlen(master_password),
           the_vault.key);

    /* Verify key matches */
    if (memcmp(the_vault.key, saved_key, 32) == 0) {
        printf("  Key re-derivation: consistent\n");
    } else {
        printf("  Key re-derivation: MISMATCH (bug)\n");
    }

    /* Load vault from disk */
    rc = vault_load();
    if (rc == 0) {
        printf("  Loaded vault from %s (%d entries)\n", VAULT_FILE,
               the_vault.count);
    } else if (rc == -1) {
        printf("  No saved vault found (first run)\n");
    } else {
        printf("  WARNING: vault file corrupt\n");
    }

    /* Verify entries survived the round-trip */
    pass_count += verify_password("github.com",    "gh-t0k3n-Sup3rS3cr3t!");
    pass_count += verify_password("bank.example",  "My$ecur3B@nkP@ss99");
    pass_count += verify_password("email.service", "Em@il_2024_xKz!pQ");
    printf("\n");

    /* ---- Phase 6: Delete ------------------------------------------------ */
    printf("[6] Delete Entry\n");

    rc = vault_delete("bank.example");
    if (rc == 0) {
        printf("  Deleted: bank.example\n");
    } else {
        printf("  FAIL: could not delete bank.example\n");
    }

    printf("  Remaining entries:\n");
    vault_list();

    /* Verify deleted entry is gone */
    rc = vault_get("bank.example", buf, sizeof(buf));
    if (rc == -1) {
        printf("  PASS: 'bank.example' correctly removed\n");
        pass_count++;
    } else {
        printf("  FAIL: 'bank.example' still accessible after delete\n");
    }

    /* Verify surviving entries still work */
    pass_count += verify_password("github.com",    "gh-t0k3n-Sup3rS3cr3t!");
    pass_count += verify_password("email.service", "Em@il_2024_xKz!pQ");
    printf("\n");

    /* ---- Phase 7: Tamper Detection -------------------------------------- */
    printf("[7] Tamper Detection\n");

    /* Try decrypting with a wrong master password */
    unsigned char wrong_key[32];
    sha256((const unsigned char *)"wrong-password", 14, wrong_key);

    unsigned char wrong_entry_key[32];
    derive_entry_key(wrong_key, "github.com", wrong_entry_key);

    /* Find the github entry and try to decrypt with wrong key */
    for (uint32_t i = 0; i < the_vault.count; i++) {
        if (strcmp(the_vault.entries[i].name, "github.com") == 0) {
            char wrong_result[VAULT_PASS_LEN];
            xor_crypt(the_vault.entries[i].encrypted,
                      (unsigned char *)wrong_result,
                      the_vault.entries[i].pass_len,
                      wrong_entry_key);
            wrong_result[the_vault.entries[i].pass_len] = '\0';

            if (strcmp(wrong_result, "gh-t0k3n-Sup3rS3cr3t!") != 0) {
                printf("  PASS: wrong master password produces garbage: '%s'\n",
                       wrong_result);
                pass_count++;
            } else {
                printf("  FAIL: wrong password decrypted correctly (impossible)\n");
            }
            break;
        }
    }
    printf("\n");

    /* ---- Summary -------------------------------------------------------- */
    printf("========================================\n");
    printf("  Results: %d/10 checks passed\n", pass_count);
    printf("========================================\n");

    return 0;
}
