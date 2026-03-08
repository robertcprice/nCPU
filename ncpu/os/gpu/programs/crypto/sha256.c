/*
 * sha256.c -- Pure C SHA-256 and HMAC-SHA256 for freestanding ARM64 on Metal GPU.
 *
 * All integer arithmetic (rotate, XOR, addition). No floating point.
 * Compiles with:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *       -march=armv8-a -mgeneral-regs-only -I../  sha256.c \
 *       -T ../arm64.ld ../arm64_start.S -o sha256
 *
 * Reference: FIPS 180-4 (Secure Hash Standard)
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/* BIT MANIPULATION PRIMITIVES                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define ROTR(x, n)  (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n)   ((x) >> (n))

/* SHA-256 logical functions (FIPS 180-4 Section 4.1.2) */
#define Ch(x, y, z)   (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define Sigma0(x)      (ROTR(x,  2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define Sigma1(x)      (ROTR(x,  6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x)      (ROTR(x,  7) ^ ROTR(x, 18) ^ SHR(x,  3))
#define sigma1(x)      (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SHA-256 CONSTANTS                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * First 32 bits of the fractional parts of the cube roots
 * of the first 64 primes (2..311).
 */
static const unsigned int K[64] = {
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

/*
 * Initial hash values: first 32 bits of the fractional parts
 * of the square roots of the first 8 primes (2..19).
 */
static const unsigned int H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SHA-256 CONTEXT                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

struct sha256_ctx {
    unsigned int   state[8];       /* hash state H0..H7              */
    unsigned char  block[64];      /* partial block buffer            */
    unsigned int   block_len;      /* bytes in partial block [0..63]  */
    unsigned long  total_len;      /* total message length in bytes   */
};

/* ═══════════════════════════════════════════════════════════════════════════ */
/* INTERNAL: PROCESS A SINGLE 512-BIT BLOCK                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * sha256_transform -- compress one 64-byte block into the running state.
 *
 * This is the core of SHA-256: the 64-round compression function.
 * All operations are 32-bit unsigned integer arithmetic (add, rotate, XOR, AND).
 */
static void sha256_transform(unsigned int state[8], const unsigned char block[64])
{
    unsigned int W[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int T1, T2;
    int t;

    /* Step 1: Prepare the message schedule (FIPS 180-4 Section 6.2.2 Step 1) */
    /* First 16 words: big-endian decode from the 64-byte block */
    for (t = 0; t < 16; t++) {
        W[t] = ((unsigned int)block[t * 4 + 0] << 24)
             | ((unsigned int)block[t * 4 + 1] << 16)
             | ((unsigned int)block[t * 4 + 2] <<  8)
             | ((unsigned int)block[t * 4 + 3]);
    }

    /* Remaining 48 words: expansion via sigma functions */
    for (t = 16; t < 64; t++) {
        W[t] = sigma1(W[t - 2]) + W[t - 7] + sigma0(W[t - 15]) + W[t - 16];
    }

    /* Step 2: Initialize working variables (Section 6.2.2 Step 2) */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    /* Step 3: 64 rounds of compression (Section 6.2.2 Step 3) */
    for (t = 0; t < 64; t++) {
        T1 = h + Sigma1(e) + Ch(e, f, g) + K[t] + W[t];
        T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    /* Step 4: Update state (Section 6.2.2 Step 4) */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* STREAMING INTERFACE: INIT / UPDATE / FINAL                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

static void sha256_init(struct sha256_ctx *ctx)
{
    int i;
    for (i = 0; i < 8; i++)
        ctx->state[i] = H0[i];
    ctx->block_len = 0;
    ctx->total_len = 0;
}

static void sha256_update(struct sha256_ctx *ctx,
                          const unsigned char *data, size_t len)
{
    size_t i;

    ctx->total_len += len;

    for (i = 0; i < len; i++) {
        ctx->block[ctx->block_len++] = data[i];
        if (ctx->block_len == 64) {
            sha256_transform(ctx->state, ctx->block);
            ctx->block_len = 0;
        }
    }
}

/*
 * sha256_final -- finalize the digest.
 *
 * Pads the message per FIPS 180-4 Section 5.1.1:
 *   1. Append bit '1' (0x80 byte)
 *   2. Append zeros until message length === 448 mod 512 (56 mod 64 bytes)
 *   3. Append original message length as 64-bit big-endian
 */
static void sha256_final(struct sha256_ctx *ctx, unsigned char out[32])
{
    unsigned long bits = ctx->total_len * 8;   /* total length in bits */
    int i;

    /* Append the 0x80 byte */
    ctx->block[ctx->block_len++] = 0x80;

    /* If not enough room for the 8-byte length, pad this block and process */
    if (ctx->block_len > 56) {
        while (ctx->block_len < 64)
            ctx->block[ctx->block_len++] = 0x00;
        sha256_transform(ctx->state, ctx->block);
        ctx->block_len = 0;
    }

    /* Pad with zeros up to byte 56 */
    while (ctx->block_len < 56)
        ctx->block[ctx->block_len++] = 0x00;

    /* Append total bit length as 64-bit big-endian */
    ctx->block[56] = (unsigned char)(bits >> 56);
    ctx->block[57] = (unsigned char)(bits >> 48);
    ctx->block[58] = (unsigned char)(bits >> 40);
    ctx->block[59] = (unsigned char)(bits >> 32);
    ctx->block[60] = (unsigned char)(bits >> 24);
    ctx->block[61] = (unsigned char)(bits >> 16);
    ctx->block[62] = (unsigned char)(bits >>  8);
    ctx->block[63] = (unsigned char)(bits);

    sha256_transform(ctx->state, ctx->block);

    /* Produce the final 32-byte digest in big-endian */
    for (i = 0; i < 8; i++) {
        out[i * 4 + 0] = (unsigned char)(ctx->state[i] >> 24);
        out[i * 4 + 1] = (unsigned char)(ctx->state[i] >> 16);
        out[i * 4 + 2] = (unsigned char)(ctx->state[i] >>  8);
        out[i * 4 + 3] = (unsigned char)(ctx->state[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ONE-SHOT SHA-256                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * sha256_hash -- compute the SHA-256 digest of an arbitrary-length message.
 *
 * Parameters:
 *   data  -- pointer to input bytes
 *   len   -- input length in bytes
 *   out   -- 32-byte output buffer for the digest
 */
void sha256_hash(const unsigned char *data, size_t len, unsigned char out[32])
{
    struct sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, out);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HMAC-SHA256 (RFC 2104)                                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */

/*
 * sha256_hmac -- compute HMAC-SHA256.
 *
 * HMAC(K, m) = H((K' ^ opad) || H((K' ^ ipad) || m))
 *
 * Where:
 *   K' = K if len(K) <= 64, else K' = SHA256(K)
 *   ipad = 0x36 repeated 64 times
 *   opad = 0x5c repeated 64 times
 *
 * Parameters:
 *   key   -- HMAC key bytes
 *   klen  -- key length in bytes
 *   data  -- message bytes
 *   dlen  -- message length in bytes
 *   out   -- 32-byte output buffer for the MAC
 */
void sha256_hmac(const unsigned char *key, size_t klen,
                 const unsigned char *data, size_t dlen,
                 unsigned char out[32])
{
    unsigned char k_prime[64];
    unsigned char pad[64];
    unsigned char inner_hash[32];
    struct sha256_ctx ctx;
    int i;

    /* Step 1: Derive K' */
    memset(k_prime, 0, 64);
    if (klen > 64) {
        /* Keys longer than block size are first hashed */
        sha256_hash(key, klen, k_prime);
        /* Remaining bytes of k_prime are already zero */
    } else {
        memcpy(k_prime, key, klen);
        /* Remaining bytes of k_prime are already zero */
    }

    /* Step 2: Inner hash -- H((K' ^ ipad) || message) */
    for (i = 0; i < 64; i++)
        pad[i] = k_prime[i] ^ 0x36;

    sha256_init(&ctx);
    sha256_update(&ctx, pad, 64);
    sha256_update(&ctx, data, dlen);
    sha256_final(&ctx, inner_hash);

    /* Step 3: Outer hash -- H((K' ^ opad) || inner_hash) */
    for (i = 0; i < 64; i++)
        pad[i] = k_prime[i] ^ 0x5c;

    sha256_init(&ctx);
    sha256_update(&ctx, pad, 64);
    sha256_update(&ctx, inner_hash, 32);
    sha256_final(&ctx, out);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* HEX OUTPUT HELPER                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

static const char hex_chars[] = "0123456789abcdef";

/*
 * print_hex -- print a byte array as lowercase hex to stdout.
 */
static void print_hex(const unsigned char *data, size_t len)
{
    size_t i;
    char hex[3];
    hex[2] = '\0';

    for (i = 0; i < len; i++) {
        hex[0] = hex_chars[(data[i] >> 4) & 0x0f];
        hex[1] = hex_chars[data[i] & 0x0f];
        printf("%s", hex);
    }
}

/*
 * hex_to_bytes -- parse a hex string into a byte array.
 * Returns number of bytes written.
 */
static int hex_to_bytes(const char *hex, unsigned char *out, int max_out)
{
    int i = 0;
    while (hex[0] && hex[1] && i < max_out) {
        unsigned char hi, lo;

        if (hex[0] >= '0' && hex[0] <= '9')      hi = hex[0] - '0';
        else if (hex[0] >= 'a' && hex[0] <= 'f')  hi = hex[0] - 'a' + 10;
        else if (hex[0] >= 'A' && hex[0] <= 'F')  hi = hex[0] - 'A' + 10;
        else break;

        if (hex[1] >= '0' && hex[1] <= '9')      lo = hex[1] - '0';
        else if (hex[1] >= 'a' && hex[1] <= 'f')  lo = hex[1] - 'a' + 10;
        else if (hex[1] >= 'A' && hex[1] <= 'F')  lo = hex[1] - 'A' + 10;
        else break;

        out[i++] = (hi << 4) | lo;
        hex += 2;
    }
    return i;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MAIN                                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    unsigned char digest[32];
    unsigned char expected[32];

    /* Known SHA-256("Hello, GPU!") computed offline */
    const char *expected_hex =
        "58a66274bf11742c69199a2b216eab0217995c4d78053df6d169cb4fe86a2a08";

    const char *test_msg = "Hello, GPU!";
    size_t test_len = strlen(test_msg);

    /* ---- SHA-256 Test ---- */
    printf("SHA-256 on Metal GPU Compute Shader\n");
    printf("====================================\n\n");

    printf("Input:    \"%s\" (%lu bytes)\n", test_msg, (unsigned long)test_len);

    sha256_hash((const unsigned char *)test_msg, test_len, digest);

    printf("SHA-256:  ");
    print_hex(digest, 32);
    printf("\n");

    printf("Expected: %s\n", expected_hex);

    /* Verify against known answer */
    hex_to_bytes(expected_hex, expected, 32);

    if (memcmp(digest, expected, 32) == 0) {
        printf("Result:   PASS\n\n");
    } else {
        printf("Result:   FAIL\n\n");
    }

    /* ---- HMAC-SHA256 Test ---- */
    /*
     * RFC 4231 Test Case 2:
     *   Key  = "Jefe"
     *   Data = "what do ya want for nothing?"
     *   HMAC = 5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843
     */
    const char *hmac_key = "Jefe";
    const char *hmac_data = "what do ya want for nothing?";
    const char *hmac_expected_hex =
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";

    unsigned char hmac_result[32];
    unsigned char hmac_expected[32];

    printf("HMAC-SHA256 (RFC 4231 Test Case 2)\n");
    printf("====================================\n\n");

    printf("Key:      \"%s\"\n", hmac_key);
    printf("Data:     \"%s\"\n", hmac_data);

    sha256_hmac((const unsigned char *)hmac_key, strlen(hmac_key),
                (const unsigned char *)hmac_data, strlen(hmac_data),
                hmac_result);

    printf("HMAC:     ");
    print_hex(hmac_result, 32);
    printf("\n");

    printf("Expected: %s\n", hmac_expected_hex);

    hex_to_bytes(hmac_expected_hex, hmac_expected, 32);

    if (memcmp(hmac_result, hmac_expected, 32) == 0) {
        printf("Result:   PASS\n\n");
    } else {
        printf("Result:   FAIL\n\n");
    }

    /* ---- Empty string test ---- */
    /*
     * SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
     */
    const char *empty_expected_hex =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    unsigned char empty_expected[32];

    printf("SHA-256 Empty String Test\n");
    printf("====================================\n\n");

    sha256_hash((const unsigned char *)"", 0, digest);

    printf("SHA-256:  ");
    print_hex(digest, 32);
    printf("\n");

    printf("Expected: %s\n", empty_expected_hex);

    hex_to_bytes(empty_expected_hex, empty_expected, 32);

    if (memcmp(digest, empty_expected, 32) == 0) {
        printf("Result:   PASS\n\n");
    } else {
        printf("Result:   FAIL\n\n");
    }

    /* ---- Multi-block test (56 bytes triggers exact padding boundary) ---- */
    /*
     * SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
     * = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
     */
    const char *multi_msg =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    const char *multi_expected_hex =
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";

    unsigned char multi_expected[32];

    printf("SHA-256 Multi-Block Test (56 bytes)\n");
    printf("====================================\n\n");

    printf("Input:    \"%s\"\n", multi_msg);

    sha256_hash((const unsigned char *)multi_msg, strlen(multi_msg), digest);

    printf("SHA-256:  ");
    print_hex(digest, 32);
    printf("\n");

    printf("Expected: %s\n", multi_expected_hex);

    hex_to_bytes(multi_expected_hex, multi_expected, 32);

    if (memcmp(digest, multi_expected, 32) == 0) {
        printf("Result:   PASS\n\n");
    } else {
        printf("Result:   FAIL\n\n");
    }

    printf("All tests complete.\n");

    return 0;
}
