/*
 * AES-128 -- Pure C implementation for freestanding ARM64 Metal GPU kernel.
 *
 * Implements AES-128 ECB and CBC encrypt/decrypt with precomputed S-boxes.
 * Integer-only, no floating point, no stdlib. All crypto operations use
 * unsigned char byte arrays and unsigned int 32-bit words.
 *
 * State layout: flat 16-byte array in FIPS 197 column-major wire format.
 *   Byte index i maps to state position (row, col) = (i % 4, i / 4).
 *   So bytes 0-3 are column 0, bytes 4-7 are column 1, etc.
 *   ShiftRows operates on rows: row r = {s[r], s[r+4], s[r+8], s[r+12]}.
 *   MixColumns operates on columns: col c = {s[4c], s[4c+1], s[4c+2], s[4c+3]}.
 *
 * This is the natural FIPS 197 layout where input bytes map directly
 * into the state array without any transposition: state[i] = input[i].
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/crypto/aes128.c
 *          -o /tmp/aes128.elf
 *
 * Reference: FIPS 197 (Advanced Encryption Standard)
 *            NIST SP 800-38A (Block Cipher Modes of Operation)
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* =========================================================================
 * CONSTANTS
 * ========================================================================= */

#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    16
#define AES_NUM_ROUNDS  10
#define AES_KEY_WORDS   44  /* Nb * (Nr + 1) = 4 * 11 */

/* =========================================================================
 * S-BOX AND INVERSE S-BOX (FIPS 197 precomputed lookup tables)
 * ========================================================================= */

static const unsigned char sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static const unsigned char rsbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
};

/* Round constants (FIPS 197 Section 5.2) */
static const unsigned char rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

/* =========================================================================
 * GF(2^8) ARITHMETIC
 * ========================================================================= */

/*
 * Multiply by 2 in GF(2^8) with irreducible polynomial
 * x^8 + x^4 + x^3 + x + 1 (0x11b).
 */
static unsigned char xtime(unsigned char x) {
    return (unsigned char)((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

/*
 * General multiplication in GF(2^8) via Russian-peasant (shift-and-add).
 * Used only by InvMixColumns for coefficients {9, 11, 13, 14}.
 */
static unsigned char gf_mul(unsigned char a, unsigned char b) {
    unsigned char p = 0;
    int i;
    for (i = 0; i < 8; i++) {
        if (b & 1)
            p ^= a;
        a = xtime(a);
        b >>= 1;
    }
    return p;
}

/* =========================================================================
 * KEY EXPANSION (FIPS 197 Section 5.2)
 *
 * Expands a 16-byte key into 44 words (176 bytes) of round key material.
 * Round key for round r starts at byte offset r * 16.
 * ========================================================================= */

static void aes_key_expand(const unsigned char *key,
                           unsigned char *rk) {
    unsigned char temp[4];
    int i;

    /* First round key is the cipher key itself (4 words = 16 bytes) */
    for (i = 0; i < AES_KEY_SIZE; i++)
        rk[i] = key[i];

    /* Generate remaining 40 words (word index 4..43) */
    for (i = 4; i < AES_KEY_WORDS; i++) {
        /* Copy previous word w[i-1] */
        temp[0] = rk[(i - 1) * 4 + 0];
        temp[1] = rk[(i - 1) * 4 + 1];
        temp[2] = rk[(i - 1) * 4 + 2];
        temp[3] = rk[(i - 1) * 4 + 3];

        if ((i & 3) == 0) {
            /* RotWord: [a0, a1, a2, a3] -> [a1, a2, a3, a0] */
            unsigned char t = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = t;

            /* SubWord: apply S-box to each byte */
            temp[0] = sbox[temp[0]];
            temp[1] = sbox[temp[1]];
            temp[2] = sbox[temp[2]];
            temp[3] = sbox[temp[3]];

            /* XOR round constant into first byte */
            temp[0] ^= rcon[i >> 2];
        }

        /* w[i] = w[i-4] XOR temp */
        rk[i * 4 + 0] = rk[(i - 4) * 4 + 0] ^ temp[0];
        rk[i * 4 + 1] = rk[(i - 4) * 4 + 1] ^ temp[1];
        rk[i * 4 + 2] = rk[(i - 4) * 4 + 2] ^ temp[2];
        rk[i * 4 + 3] = rk[(i - 4) * 4 + 3] ^ temp[3];
    }
}

/* =========================================================================
 * AES ENCRYPTION ROUND OPERATIONS
 *
 * State is a flat 16-byte array in column-major (FIPS 197 wire) format:
 *   Bytes 0-3 = column 0, bytes 4-7 = column 1, etc.
 *   state[r + 4*c] = element at row r, column c.
 *
 * Row r consists of bytes: {s[r], s[r+4], s[r+8], s[r+12]}
 * Column c consists of bytes: {s[4c], s[4c+1], s[4c+2], s[4c+3]}
 * ========================================================================= */

/*
 * SubBytes: apply S-box to every byte. Order-independent.
 */
static void sub_bytes(unsigned char *s) {
    int i;
    for (i = 0; i < 16; i++)
        s[i] = sbox[s[i]];
}

/*
 * ShiftRows: cyclically shift row r left by r positions.
 *
 * Row r = {s[r], s[r+4], s[r+8], s[r+12]}
 *
 * Row 0: no shift
 * Row 1: shift left by 1: {s[1],s[5],s[9],s[13]} -> {s[5],s[9],s[13],s[1]}
 * Row 2: shift left by 2: {s[2],s[6],s[10],s[14]} -> {s[10],s[14],s[2],s[6]}
 * Row 3: shift left by 3: {s[3],s[7],s[11],s[15]} -> {s[15],s[3],s[7],s[11]}
 */
static void shift_rows(unsigned char *s) {
    unsigned char t;

    /* Row 1: shift left by 1 */
    t    = s[1];
    s[1] = s[5];
    s[5] = s[9];
    s[9] = s[13];
    s[13] = t;

    /* Row 2: shift left by 2 (swap pairs) */
    t     = s[2];
    s[2]  = s[10];
    s[10] = t;
    t     = s[6];
    s[6]  = s[14];
    s[14] = t;

    /* Row 3: shift left by 3 = shift right by 1 */
    t     = s[15];
    s[15] = s[11];
    s[11] = s[7];
    s[7]  = s[3];
    s[3]  = t;
}

/*
 * MixColumns: linear mixing of each column using the matrix:
 *   [2 3 1 1]
 *   [1 2 3 1]
 *   [1 1 2 3]
 *   [3 1 1 2]
 *
 * Column c consists of bytes {s[4c], s[4c+1], s[4c+2], s[4c+3]}.
 * Each output byte: new[i] = 2*a[i] ^ 3*a[(i+1)%4] ^ a[(i+2)%4] ^ a[(i+3)%4]
 * where 3*x = xtime(x) ^ x.
 */
static void mix_columns(unsigned char *s) {
    int c;
    for (c = 0; c < 4; c++) {
        int base = c * 4;
        unsigned char a0 = s[base + 0];
        unsigned char a1 = s[base + 1];
        unsigned char a2 = s[base + 2];
        unsigned char a3 = s[base + 3];

        s[base + 0] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
        s[base + 1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
        s[base + 2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
        s[base + 3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
    }
}

/*
 * AddRoundKey: XOR state with 16 bytes of round key material.
 * Both state and round key are in the same column-major wire format,
 * so a simple bytewise XOR is correct.
 */
static void add_round_key(unsigned char *s, const unsigned char *rk) {
    int i;
    for (i = 0; i < 16; i++)
        s[i] ^= rk[i];
}

/* =========================================================================
 * AES DECRYPTION ROUND OPERATIONS
 * ========================================================================= */

/*
 * InvSubBytes: apply inverse S-box to every byte.
 */
static void inv_sub_bytes(unsigned char *s) {
    int i;
    for (i = 0; i < 16; i++)
        s[i] = rsbox[s[i]];
}

/*
 * InvShiftRows: cyclically shift row r RIGHT by r positions.
 *
 * Row 1: shift right by 1: {s[1],s[5],s[9],s[13]} -> {s[13],s[1],s[5],s[9]}
 * Row 2: shift right by 2: swap pairs (same as forward)
 * Row 3: shift right by 3 = shift left by 1
 */
static void inv_shift_rows(unsigned char *s) {
    unsigned char t;

    /* Row 1: shift right by 1 */
    t     = s[13];
    s[13] = s[9];
    s[9]  = s[5];
    s[5]  = s[1];
    s[1]  = t;

    /* Row 2: shift right by 2 (swap pairs, same as forward) */
    t     = s[2];
    s[2]  = s[10];
    s[10] = t;
    t     = s[6];
    s[6]  = s[14];
    s[14] = t;

    /* Row 3: shift right by 3 = shift left by 1 */
    t    = s[3];
    s[3] = s[7];
    s[7] = s[11];
    s[11] = s[15];
    s[15] = t;
}

/*
 * InvMixColumns: inverse column mixing using the matrix:
 *   [14 11 13  9]
 *   [ 9 14 11 13]
 *   [13  9 14 11]
 *   [11 13  9 14]
 */
static void inv_mix_columns(unsigned char *s) {
    int c;
    for (c = 0; c < 4; c++) {
        int base = c * 4;
        unsigned char a0 = s[base + 0];
        unsigned char a1 = s[base + 1];
        unsigned char a2 = s[base + 2];
        unsigned char a3 = s[base + 3];

        s[base + 0] = gf_mul(a0, 14) ^ gf_mul(a1, 11) ^ gf_mul(a2, 13) ^ gf_mul(a3, 9);
        s[base + 1] = gf_mul(a0, 9)  ^ gf_mul(a1, 14) ^ gf_mul(a2, 11) ^ gf_mul(a3, 13);
        s[base + 2] = gf_mul(a0, 13) ^ gf_mul(a1, 9)  ^ gf_mul(a2, 14) ^ gf_mul(a3, 11);
        s[base + 3] = gf_mul(a0, 11) ^ gf_mul(a1, 13) ^ gf_mul(a2, 9)  ^ gf_mul(a3, 14);
    }
}

/* =========================================================================
 * AES-128 ECB MODE
 * ========================================================================= */

/*
 * Encrypt a single 16-byte block in-place.
 * rk must be 176 bytes (44 words) from aes_key_expand().
 */
static void aes_ecb_encrypt(unsigned char *block,
                            const unsigned char *rk) {
    int round;

    /* Initial round key addition */
    add_round_key(block, rk);

    /* Rounds 1 through Nr-1: SubBytes, ShiftRows, MixColumns, AddRoundKey */
    for (round = 1; round < AES_NUM_ROUNDS; round++) {
        sub_bytes(block);
        shift_rows(block);
        mix_columns(block);
        add_round_key(block, rk + round * 16);
    }

    /* Final round (no MixColumns) */
    sub_bytes(block);
    shift_rows(block);
    add_round_key(block, rk + AES_NUM_ROUNDS * 16);
}

/*
 * Decrypt a single 16-byte block in-place.
 * Uses equivalent inverse cipher (FIPS 197 Section 5.3).
 */
static void aes_ecb_decrypt(unsigned char *block,
                            const unsigned char *rk) {
    int round;

    add_round_key(block, rk + AES_NUM_ROUNDS * 16);

    for (round = AES_NUM_ROUNDS - 1; round > 0; round--) {
        inv_shift_rows(block);
        inv_sub_bytes(block);
        add_round_key(block, rk + round * 16);
        inv_mix_columns(block);
    }

    /* Final round (no InvMixColumns) */
    inv_shift_rows(block);
    inv_sub_bytes(block);
    add_round_key(block, rk);
}

/* =========================================================================
 * AES-128 CBC MODE (NIST SP 800-38A)
 * ========================================================================= */

/*
 * Encrypt data using AES-128 CBC mode.
 * len must be a multiple of 16 (no padding applied).
 * iv is updated in-place to hold the last ciphertext block on exit.
 */
static void aes_cbc_encrypt(unsigned char *data, int len,
                            const unsigned char *rk,
                            unsigned char *iv) {
    int off, i;

    for (off = 0; off < len; off += AES_BLOCK_SIZE) {
        /* XOR plaintext block with IV (or previous ciphertext block) */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            data[off + i] ^= iv[i];

        aes_ecb_encrypt(data + off, rk);

        /* This ciphertext block becomes the next IV */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            iv[i] = data[off + i];
    }
}

/*
 * Decrypt data using AES-128 CBC mode.
 * len must be a multiple of 16.
 * iv is updated in-place.
 */
static void aes_cbc_decrypt(unsigned char *data, int len,
                            const unsigned char *rk,
                            unsigned char *iv) {
    unsigned char prev[AES_BLOCK_SIZE];
    int off, i;

    for (off = 0; off < len; off += AES_BLOCK_SIZE) {
        /* Save ciphertext block before decryption overwrites it */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            prev[i] = data[off + i];

        aes_ecb_decrypt(data + off, rk);

        /* XOR with IV to recover plaintext */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            data[off + i] ^= iv[i];

        /* Update IV to saved ciphertext block */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            iv[i] = prev[i];
    }
}

/* =========================================================================
 * UTILITY
 * ========================================================================= */

static void print_hex(const char *label, const unsigned char *data, int len) {
    int i;
    printf("%s: ", label);
    for (i = 0; i < len; i++)
        printf("%02x", data[i]);
    printf("\n");
}

static int bytes_equal(const unsigned char *a, const unsigned char *b, int len) {
    int i;
    for (i = 0; i < len; i++) {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

/* =========================================================================
 * TEST HARNESS
 *
 * Tests against three authoritative test vectors:
 *   1. FIPS 197 Appendix B (ECB encrypt + decrypt roundtrip)
 *   2. NIST SP 800-38A F.2.1 (CBC encrypt + decrypt roundtrip)
 *   3. Zero-key KAT (ECB encrypt + decrypt roundtrip)
 * ========================================================================= */

int main(void) {
    unsigned char rk[AES_KEY_WORDS * 4];
    int pass_count = 0;
    int fail_count = 0;
    int i;

    printf("=== AES-128 Crypto Test Suite ===\n\n");

    /* FIPS 197 Appendix B test key (shared by ECB and CBC tests) */
    unsigned char key[AES_KEY_SIZE] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    };

    aes_key_expand(key, rk);

    /* ------------------------------------------------------------------
     * TEST 1: FIPS 197 Appendix B -- ECB encrypt
     *
     * Plaintext:  3243f6a8 885a308d 313198a2 e0370734
     * Expected:   3925841d 02dc09fb dc118597 196a0b32
     * ------------------------------------------------------------------ */
    {
        unsigned char pt[AES_BLOCK_SIZE] = {
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
        };
        unsigned char original[AES_BLOCK_SIZE];
        unsigned char expected_ct[AES_BLOCK_SIZE] = {
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32
        };

        for (i = 0; i < AES_BLOCK_SIZE; i++)
            original[i] = pt[i];

        print_hex("ECB plaintext ", pt, AES_BLOCK_SIZE);

        aes_ecb_encrypt(pt, rk);
        print_hex("ECB ciphertext", pt, AES_BLOCK_SIZE);

        if (bytes_equal(pt, expected_ct, AES_BLOCK_SIZE)) {
            printf("ECB encrypt vs FIPS 197: PASS\n");
            pass_count++;
        } else {
            printf("ECB encrypt vs FIPS 197: FAIL\n");
            print_hex("  expected", expected_ct, AES_BLOCK_SIZE);
            fail_count++;
        }

        aes_ecb_decrypt(pt, rk);
        print_hex("ECB decrypted ", pt, AES_BLOCK_SIZE);

        if (bytes_equal(pt, original, AES_BLOCK_SIZE)) {
            printf("ECB roundtrip: PASS\n");
            pass_count++;
        } else {
            printf("ECB roundtrip: FAIL\n");
            fail_count++;
        }
    }

    printf("\n");

    /* ------------------------------------------------------------------
     * TEST 2: NIST SP 800-38A F.2.1 -- CBC encrypt (2 blocks)
     *
     * IV:           000102030405060708090a0b0c0d0e0f
     * Block 1 PT:   6bc1bee22e409f96e93d7e117393172a
     * Block 1 CT:   7649abac8119b246cee98e9b12e9197d
     * Block 2 PT:   ae2d8a571e03ac9c9eb76fac45af8e51
     * Block 2 CT:   5086cb9b507219ee95db113a917678b2
     * ------------------------------------------------------------------ */
    {
        unsigned char pt[32] = {
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
            0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
            0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51
        };
        unsigned char original[32];
        unsigned char iv[AES_BLOCK_SIZE] = {
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
        };
        unsigned char iv_save[AES_BLOCK_SIZE];
        unsigned char expected_ct[32] = {
            0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46,
            0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9, 0x19, 0x7d,
            0x50, 0x86, 0xcb, 0x9b, 0x50, 0x72, 0x19, 0xee,
            0x95, 0xdb, 0x11, 0x3a, 0x91, 0x76, 0x78, 0xb2
        };

        for (i = 0; i < 32; i++)
            original[i] = pt[i];
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            iv_save[i] = iv[i];

        print_hex("CBC plaintext ", pt, 32);
        print_hex("CBC IV        ", iv, AES_BLOCK_SIZE);

        aes_cbc_encrypt(pt, 32, rk, iv);
        print_hex("CBC ciphertext", pt, 32);

        if (bytes_equal(pt, expected_ct, 32)) {
            printf("CBC encrypt vs NIST: PASS\n");
            pass_count++;
        } else {
            printf("CBC encrypt vs NIST: FAIL\n");
            print_hex("  expected", expected_ct, 32);
            fail_count++;
        }

        /* Restore IV and decrypt */
        for (i = 0; i < AES_BLOCK_SIZE; i++)
            iv[i] = iv_save[i];

        aes_cbc_decrypt(pt, 32, rk, iv);
        print_hex("CBC decrypted ", pt, 32);

        if (bytes_equal(pt, original, 32)) {
            printf("CBC roundtrip: PASS\n");
            pass_count++;
        } else {
            printf("CBC roundtrip: FAIL\n");
            fail_count++;
        }
    }

    printf("\n");

    /* ------------------------------------------------------------------
     * TEST 3: Zero-key Known Answer Test
     *
     * Key:       00000000000000000000000000000000
     * Plaintext: 00000000000000000000000000000000
     * Expected:  66e94bd4ef8a2c3b884cfa59ca342b2e
     * ------------------------------------------------------------------ */
    {
        unsigned char zkey[AES_KEY_SIZE];
        unsigned char zrk[AES_KEY_WORDS * 4];
        unsigned char block[AES_BLOCK_SIZE];
        unsigned char original[AES_BLOCK_SIZE];
        unsigned char expected[AES_BLOCK_SIZE] = {
            0x66, 0xe9, 0x4b, 0xd4, 0xef, 0x8a, 0x2c, 0x3b,
            0x88, 0x4c, 0xfa, 0x59, 0xca, 0x34, 0x2b, 0x2e
        };

        for (i = 0; i < AES_KEY_SIZE; i++)
            zkey[i] = 0;
        for (i = 0; i < AES_BLOCK_SIZE; i++) {
            block[i] = 0;
            original[i] = 0;
        }

        aes_key_expand(zkey, zrk);
        aes_ecb_encrypt(block, zrk);

        if (bytes_equal(block, expected, AES_BLOCK_SIZE)) {
            printf("ECB zero-key KAT: PASS\n");
            pass_count++;
        } else {
            printf("ECB zero-key KAT: FAIL\n");
            print_hex("  got     ", block, AES_BLOCK_SIZE);
            print_hex("  expected", expected, AES_BLOCK_SIZE);
            fail_count++;
        }

        aes_ecb_decrypt(block, zrk);

        if (bytes_equal(block, original, AES_BLOCK_SIZE)) {
            printf("ECB zero-key roundtrip: PASS\n");
            pass_count++;
        } else {
            printf("ECB zero-key roundtrip: FAIL\n");
            fail_count++;
        }
    }

    printf("\n");

    /* ------------------------------------------------------------------
     * SUMMARY
     * ------------------------------------------------------------------ */
    printf("=== Results: %d passed, %d failed ===\n", pass_count, fail_count);

    return fail_count;
}
