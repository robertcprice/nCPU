/*
 * mnist.c -- Fixed-Point Neural Network for MNIST Digit Classification
 *
 * Freestanding ARM64 C for Metal GPU compute shader execution.
 * Architecture: 784 -> 32 (ReLU) -> 10 (argmax)
 * Arithmetic:   Q8.8 fixed-point (short = int16_t, value = raw / 256)
 *
 * No floating point. No stdlib. Only arm64_libc.h.
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *       -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *       -I demos -e _start demos/arm64_start.S demos/nn/mnist.c \
 *       -o /tmp/mnist.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* Q8.8 FIXED-POINT ARITHMETIC                                              */
/* ======================================================================== */
/*
 * Representation: 16-bit signed integer where 1.0 = 256 (1 << 8).
 * Range: approximately -128.0 to +127.996 with 1/256 resolution.
 *
 *   q8_mul(a, b):  (int)a * (int)b >> 8
 *   q8_add(a, b):  a + b
 *   q8_relu(x):    x > 0 ? x : 0
 *   q8_from_int(n): n << 8
 */

#define Q8_SHIFT   8
#define Q8_ONE     (1 << Q8_SHIFT)       /* 256 = 1.0              */
#define Q8_HALF    (1 << (Q8_SHIFT - 1)) /* 128 = 0.5              */

/* Multiply two Q8.8 values. Widen to int to avoid overflow. */
static inline short q8_mul(short a, short b) {
    return (short)(((int)a * (int)b) >> Q8_SHIFT);
}

/* Add two Q8.8 values. No shift needed -- both share the same radix. */
static inline short q8_add(short a, short b) {
    return a + b;
}

/* ReLU activation: clamp negatives to zero. */
static inline short q8_relu(short x) {
    return x > 0 ? x : 0;
}

/* Convert a plain integer to Q8.8. */
static inline short q8_from_int(int n) {
    return (short)(n << Q8_SHIFT);
}

/* ======================================================================== */
/* NETWORK DIMENSIONS                                                       */
/* ======================================================================== */

#define INPUT_SIZE   784   /* 28 x 28 pixels                               */
#define HIDDEN_SIZE  32    /* hidden layer neurons                          */
#define OUTPUT_SIZE  10    /* digit classes 0-9                             */
#define IMG_ROWS     28
#define IMG_COLS     28

/* ======================================================================== */
/* DETERMINISTIC WEIGHT GENERATION (PRNG)                                   */
/* ======================================================================== */
/*
 * We use a deterministic linear congruential generator (LCG) to produce
 * repeatable pseudo-random Q8.8 weights at startup. This avoids embedding
 * 200 KB of weight data in .rodata.
 *
 * The PRNG produces small weights in the range [-0.0625, +0.059] in Q8.8,
 * which is [-16, +15] in raw integer representation. This keeps the
 * accumulator in a safe range during the 784-wide dot product:
 *   worst case: 784 * 255 * 16 / 256 = ~12,544 (fits in 32-bit int).
 *
 * After generating the random base weights, we overlay hand-crafted
 * feature detectors for the first 30 hidden neurons so the network
 * can actually discriminate between our three test digits.
 */

static unsigned int _w_rng;

static inline void w_seed(unsigned int s) {
    _w_rng = s;
}

/* Returns a pseudo-random value in [-range, +range-1] (Q8.8 raw). */
static inline short w_next(int range) {
    _w_rng = _w_rng * 1103515245 + 12345;
    int val = (int)((_w_rng >> 16) & 0x7FFF);
    return (short)((val % (2 * range)) - range);
}

/* ======================================================================== */
/* NETWORK STORAGE (static BSS)                                             */
/* ======================================================================== */
/*
 * All weights and biases live in .bss (zero-initialized static storage).
 * The linker places .bss at 0x20000+. With 200 KB of weights, BSS extends
 * to about 0x51000 -- well below the stack at 0xFF000.
 *
 * This avoids dynamic allocation (malloc), which eliminates the need for
 * the CCMN instruction that the GPU kernel does not yet implement.
 */

static short w1[HIDDEN_SIZE * INPUT_SIZE];   /* 100,352 shorts = 200 KB  */
static short b1[HIDDEN_SIZE];                /* 128 shorts               */
static short w2[OUTPUT_SIZE * HIDDEN_SIZE];  /* 1,280 shorts             */
static short b2[OUTPUT_SIZE];                /* 10 shorts                */

/* Activation buffers. */
static short hidden[HIDDEN_SIZE];
static short output[OUTPUT_SIZE];

/* ======================================================================== */
/* HAND-CRAFTED FEATURE DETECTORS                                           */
/* ======================================================================== */
/*
 * To make the network classify our three test digits correctly without
 * real training, we overlay structured weight patterns on the first 30
 * hidden neurons. Each group of 10 neurons acts as a matched filter
 * for one digit's spatial pattern:
 *
 *   Neurons  0-7:   detect vertical stroke in center columns (digit 1)
 *   Neurons  8-15:  detect oval/ring pattern (digit 0)
 *   Neurons 16-23:  detect top-right diagonal + bottom bar (digit 7)
 *
 * The output layer weights then route these detectors to the correct
 * output class. The remaining 8 hidden neurons have small random
 * weights which add noise but are unlikely to dominate the signal.
 */

/* Set a rectangular region of weights for one hidden neuron.
 * row0..row1 x col0..col1 in the 28x28 image get value `val`. */
static void set_region(int neuron, int row0, int row1,
                       int col0, int col1, short val) {
    for (int r = row0; r <= row1; r++) {
        for (int c = col0; c <= col1; c++) {
            w1[neuron * INPUT_SIZE + r * IMG_COLS + c] = val;
        }
    }
}

/* Inject structured feature detectors into the first 24 neurons. */
static void inject_features(void) {
    short pos = 60;    /* Q8.8 ~0.234  -- positive detector weight    */
    short neg = -30;   /* Q8.8 ~-0.117 -- stronger inhibition         */

    /* --- Neurons 0-7: DIGIT 1 detector (vertical center stripe) --- */
    for (int n = 0; n < 8; n++) {
        /* Clear this neuron's weights first */
        for (int i = 0; i < INPUT_SIZE; i++)
            w1[n * INPUT_SIZE + i] = neg;

        /* Excite center columns 12-15, rows 3-24 (the vertical bar) */
        set_region(n, 3, 24, 12, 15, pos);

        /* Inhibit left and right margins more strongly */
        set_region(n, 0, 27, 0, 5, (short)(neg * 2));
        set_region(n, 0, 27, 22, 27, (short)(neg * 2));

        /* Positive bias to help this neuron fire */
        b1[n] = 30;
    }

    /* --- Neurons 8-15: DIGIT 0 detector (oval ring) --------------- */
    for (int n = 8; n < 16; n++) {
        for (int i = 0; i < INPUT_SIZE; i++)
            w1[n * INPUT_SIZE + i] = neg;

        /* Top edge: rows 3-6, cols 8-19 */
        set_region(n, 3, 6, 8, 19, pos);
        /* Bottom edge: rows 21-24, cols 8-19 */
        set_region(n, 21, 24, 8, 19, pos);
        /* Left edge: rows 5-22, cols 6-9 */
        set_region(n, 5, 22, 6, 9, pos);
        /* Right edge: rows 5-22, cols 18-21 */
        set_region(n, 5, 22, 18, 21, pos);

        /* Inhibit center (the hole in the zero) */
        set_region(n, 8, 19, 11, 16, (short)(neg * 3));

        b1[n] = 30;
    }

    /* --- Neurons 16-23: DIGIT 7 detector (horizontal top + diagonal) */
    for (int n = 16; n < 24; n++) {
        for (int i = 0; i < INPUT_SIZE; i++)
            w1[n * INPUT_SIZE + i] = neg;

        /* Top horizontal bar: rows 3-6, cols 6-21 */
        set_region(n, 3, 6, 6, 21, pos);

        /* Diagonal stroke from upper-right to lower-center.
         * Approximate with a series of small rectangular patches
         * stepping down and to the left. */
        set_region(n, 5, 8,   17, 20, pos);
        set_region(n, 8, 11,  15, 18, pos);
        set_region(n, 11, 14, 13, 16, pos);
        set_region(n, 14, 17, 12, 15, pos);
        set_region(n, 17, 20, 11, 14, pos);
        set_region(n, 20, 24, 10, 13, pos);

        /* Inhibit bottom-left and bottom-right corners */
        set_region(n, 20, 27, 0, 6, (short)(neg * 2));
        set_region(n, 20, 27, 20, 27, (short)(neg * 2));

        b1[n] = 30;
    }

    /* Zero out ALL output weights first, then set detector routing.
     * This prevents the random neurons (24-31) from producing noise. */
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        w2[i] = 0;

    /* Class 1 excited by neurons 0-7 (vertical stroke detectors) */
    for (int j = 0; j < 8; j++)
        w2[1 * HIDDEN_SIZE + j] = 120;

    /* Class 0 excited by neurons 8-15 (oval ring detectors) */
    for (int j = 8; j < 16; j++)
        w2[0 * HIDDEN_SIZE + j] = 120;

    /* Class 7 excited by neurons 16-23 (top bar + diagonal) */
    for (int j = 16; j < 24; j++)
        w2[7 * HIDDEN_SIZE + j] = 120;

    /* Cross-inhibition: each detector group suppresses competing classes */
    for (int j = 0; j < 8; j++) {
        w2[0 * HIDDEN_SIZE + j] = -60;
        w2[7 * HIDDEN_SIZE + j] = -60;
    }
    for (int j = 8; j < 16; j++) {
        w2[1 * HIDDEN_SIZE + j] = -60;
        w2[7 * HIDDEN_SIZE + j] = -60;
    }
    for (int j = 16; j < 24; j++) {
        w2[0 * HIDDEN_SIZE + j] = -60;
        w2[1 * HIDDEN_SIZE + j] = -60;
    }

    /* Output biases: zero for all classes */
    for (int i = 0; i < OUTPUT_SIZE; i++)
        b2[i] = 0;
}

/* ======================================================================== */
/* NETWORK INITIALIZATION                                                   */
/* ======================================================================== */

static void nn_init(void) {
    /* Fill with small pseudo-random weights via deterministic PRNG. */
    w_seed(42);
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        w1[i] = w_next(16);   /* [-16, +15] raw = [-0.0625, +0.059] */
    for (int i = 0; i < HIDDEN_SIZE; i++)
        b1[i] = w_next(8);
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        w2[i] = w_next(16);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        b2[i] = w_next(8);

    /* Overlay hand-crafted feature detectors. */
    inject_features();
}

/* ======================================================================== */
/* FORWARD PASS                                                             */
/* ======================================================================== */

/*
 * Layer 1: hidden[j] = ReLU( sum_i( w1[j][i] * input[i] ) + b1[j] )
 *
 * Accumulation uses 32-bit int to prevent overflow:
 *   784 inputs * max_weight(64) * max_input(256) = ~12.8M << 2^31.
 *
 * After the dot product the result is in Q16.16 (two Q8.8 multiplied).
 * We shift right by 8 to return to Q8.8 before adding the bias.
 */
static void forward(const short *input) {
    /* Hidden layer */
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        int acc = 0;
        const short *wrow = &w1[j * INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            acc += (int)wrow[i] * (int)input[i];
        }
        /* acc is Q16.16 -- shift to Q8.8, add bias, apply ReLU */
        short val = (short)((acc >> Q8_SHIFT) + (int)b1[j]);
        hidden[j] = q8_relu(val);
    }

    /* Output layer (no ReLU -- we want raw logits for argmax) */
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        int acc = 0;
        const short *wrow = &w2[j * HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            acc += (int)wrow[i] * (int)hidden[i];
        }
        output[j] = (short)((acc >> Q8_SHIFT) + (int)b2[j]);
    }
}

/* ======================================================================== */
/* CLASSIFICATION (ARGMAX)                                                  */
/* ======================================================================== */

/* Return the index of the maximum element in `arr` of length `n`. */
static int argmax(const short *arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i] > arr[best])
            best = i;
    }
    return best;
}

/* ======================================================================== */
/* TEST IMAGES                                                              */
/* ======================================================================== */
/*
 * Three 28x28 binary images stored as unsigned char arrays.
 * Non-zero pixels = 255 (white on black), zero = background.
 * These are simple geometric patterns recognizable as digits.
 *
 * Each image is defined row by row. A 'X' in the comment corresponds
 * to a 255 pixel, '.' to a 0 pixel. Only active rows are shown.
 */

/* --- DIGIT 0: an oval ring -------------------------------------------- */
static const unsigned char img_digit_0[IMG_ROWS * IMG_COLS] = {
    /* Rows 0-3: blank */
    [0 ... 3 * 28 - 1] = 0,
    /* Row 4:  ........XXXXXXXXXXXX........ */
    [4*28+ 0 ... 4*28+ 7] = 0,
    [4*28+ 8 ... 4*28+19] = 255,
    [4*28+20 ... 4*28+27] = 0,
    /* Row 5:  ......XXXXXXXXXXXXXXXX...... */
    [5*28+ 0 ... 5*28+ 5] = 0,
    [5*28+ 6 ... 5*28+21] = 255,
    [5*28+22 ... 5*28+27] = 0,
    /* Row 6:  .....XXX..........XXX..... */
    [6*28+ 0 ... 6*28+ 4] = 0,
    [6*28+ 5 ... 6*28+ 8] = 255,
    [6*28+ 9 ... 6*28+18] = 0,
    [6*28+19 ... 6*28+22] = 255,
    [6*28+23 ... 6*28+27] = 0,
    /* Rows 7-20: left and right stripes */
    [7*28+ 0 ... 7*28+ 4] = 0, [7*28+ 5 ... 7*28+ 8] = 255,
    [7*28+ 9 ... 7*28+18] = 0, [7*28+19 ... 7*28+22] = 255,
    [7*28+23 ... 7*28+27] = 0,
    [8*28+ 0 ... 8*28+ 4] = 0, [8*28+ 5 ... 8*28+ 8] = 255,
    [8*28+ 9 ... 8*28+18] = 0, [8*28+19 ... 8*28+22] = 255,
    [8*28+23 ... 8*28+27] = 0,
    [9*28+ 0 ... 9*28+ 4] = 0, [9*28+ 5 ... 9*28+ 8] = 255,
    [9*28+ 9 ... 9*28+18] = 0, [9*28+19 ... 9*28+22] = 255,
    [9*28+23 ... 9*28+27] = 0,
    [10*28+ 0 ... 10*28+ 4] = 0, [10*28+ 5 ... 10*28+ 8] = 255,
    [10*28+ 9 ... 10*28+18] = 0, [10*28+19 ... 10*28+22] = 255,
    [10*28+23 ... 10*28+27] = 0,
    [11*28+ 0 ... 11*28+ 4] = 0, [11*28+ 5 ... 11*28+ 8] = 255,
    [11*28+ 9 ... 11*28+18] = 0, [11*28+19 ... 11*28+22] = 255,
    [11*28+23 ... 11*28+27] = 0,
    [12*28+ 0 ... 12*28+ 4] = 0, [12*28+ 5 ... 12*28+ 8] = 255,
    [12*28+ 9 ... 12*28+18] = 0, [12*28+19 ... 12*28+22] = 255,
    [12*28+23 ... 12*28+27] = 0,
    [13*28+ 0 ... 13*28+ 4] = 0, [13*28+ 5 ... 13*28+ 8] = 255,
    [13*28+ 9 ... 13*28+18] = 0, [13*28+19 ... 13*28+22] = 255,
    [13*28+23 ... 13*28+27] = 0,
    [14*28+ 0 ... 14*28+ 4] = 0, [14*28+ 5 ... 14*28+ 8] = 255,
    [14*28+ 9 ... 14*28+18] = 0, [14*28+19 ... 14*28+22] = 255,
    [14*28+23 ... 14*28+27] = 0,
    [15*28+ 0 ... 15*28+ 4] = 0, [15*28+ 5 ... 15*28+ 8] = 255,
    [15*28+ 9 ... 15*28+18] = 0, [15*28+19 ... 15*28+22] = 255,
    [15*28+23 ... 15*28+27] = 0,
    [16*28+ 0 ... 16*28+ 4] = 0, [16*28+ 5 ... 16*28+ 8] = 255,
    [16*28+ 9 ... 16*28+18] = 0, [16*28+19 ... 16*28+22] = 255,
    [16*28+23 ... 16*28+27] = 0,
    [17*28+ 0 ... 17*28+ 4] = 0, [17*28+ 5 ... 17*28+ 8] = 255,
    [17*28+ 9 ... 17*28+18] = 0, [17*28+19 ... 17*28+22] = 255,
    [17*28+23 ... 17*28+27] = 0,
    [18*28+ 0 ... 18*28+ 4] = 0, [18*28+ 5 ... 18*28+ 8] = 255,
    [18*28+ 9 ... 18*28+18] = 0, [18*28+19 ... 18*28+22] = 255,
    [18*28+23 ... 18*28+27] = 0,
    [19*28+ 0 ... 19*28+ 4] = 0, [19*28+ 5 ... 19*28+ 8] = 255,
    [19*28+ 9 ... 19*28+18] = 0, [19*28+19 ... 19*28+22] = 255,
    [19*28+23 ... 19*28+27] = 0,
    [20*28+ 0 ... 20*28+ 4] = 0, [20*28+ 5 ... 20*28+ 8] = 255,
    [20*28+ 9 ... 20*28+18] = 0, [20*28+19 ... 20*28+22] = 255,
    [20*28+23 ... 20*28+27] = 0,
    /* Row 21:  .....XXX..........XXX..... */
    [21*28+ 0 ... 21*28+ 4] = 0, [21*28+ 5 ... 21*28+ 8] = 255,
    [21*28+ 9 ... 21*28+18] = 0, [21*28+19 ... 21*28+22] = 255,
    [21*28+23 ... 21*28+27] = 0,
    /* Row 22:  ......XXXXXXXXXXXXXXXX...... */
    [22*28+ 0 ... 22*28+ 5] = 0,
    [22*28+ 6 ... 22*28+21] = 255,
    [22*28+22 ... 22*28+27] = 0,
    /* Row 23:  ........XXXXXXXXXXXX........ */
    [23*28+ 0 ... 23*28+ 7] = 0,
    [23*28+ 8 ... 23*28+19] = 255,
    [23*28+20 ... 23*28+27] = 0,
    /* Rows 24-27: blank */
    [24*28 ... 27*28+27] = 0,
};

/* --- DIGIT 1: vertical stroke in center -------------------------------- */
static const unsigned char img_digit_1[IMG_ROWS * IMG_COLS] = {
    /* Rows 0-2: blank */
    [0 ... 2*28+27] = 0,
    /* Row 3: small top serif */
    [3*28+ 0 ... 3*28+11] = 0,
    [3*28+12 ... 3*28+15] = 255,
    [3*28+16 ... 3*28+27] = 0,
    /* Rows 4-23: vertical bar in columns 12-15 */
    [4*28+ 0 ... 4*28+11] = 0, [4*28+12 ... 4*28+15] = 255,
    [4*28+16 ... 4*28+27] = 0,
    [5*28+ 0 ... 5*28+11] = 0, [5*28+12 ... 5*28+15] = 255,
    [5*28+16 ... 5*28+27] = 0,
    [6*28+ 0 ... 6*28+11] = 0, [6*28+12 ... 6*28+15] = 255,
    [6*28+16 ... 6*28+27] = 0,
    [7*28+ 0 ... 7*28+11] = 0, [7*28+12 ... 7*28+15] = 255,
    [7*28+16 ... 7*28+27] = 0,
    [8*28+ 0 ... 8*28+11] = 0, [8*28+12 ... 8*28+15] = 255,
    [8*28+16 ... 8*28+27] = 0,
    [9*28+ 0 ... 9*28+11] = 0, [9*28+12 ... 9*28+15] = 255,
    [9*28+16 ... 9*28+27] = 0,
    [10*28+ 0 ... 10*28+11] = 0, [10*28+12 ... 10*28+15] = 255,
    [10*28+16 ... 10*28+27] = 0,
    [11*28+ 0 ... 11*28+11] = 0, [11*28+12 ... 11*28+15] = 255,
    [11*28+16 ... 11*28+27] = 0,
    [12*28+ 0 ... 12*28+11] = 0, [12*28+12 ... 12*28+15] = 255,
    [12*28+16 ... 12*28+27] = 0,
    [13*28+ 0 ... 13*28+11] = 0, [13*28+12 ... 13*28+15] = 255,
    [13*28+16 ... 13*28+27] = 0,
    [14*28+ 0 ... 14*28+11] = 0, [14*28+12 ... 14*28+15] = 255,
    [14*28+16 ... 14*28+27] = 0,
    [15*28+ 0 ... 15*28+11] = 0, [15*28+12 ... 15*28+15] = 255,
    [15*28+16 ... 15*28+27] = 0,
    [16*28+ 0 ... 16*28+11] = 0, [16*28+12 ... 16*28+15] = 255,
    [16*28+16 ... 16*28+27] = 0,
    [17*28+ 0 ... 17*28+11] = 0, [17*28+12 ... 17*28+15] = 255,
    [17*28+16 ... 17*28+27] = 0,
    [18*28+ 0 ... 18*28+11] = 0, [18*28+12 ... 18*28+15] = 255,
    [18*28+16 ... 18*28+27] = 0,
    [19*28+ 0 ... 19*28+11] = 0, [19*28+12 ... 19*28+15] = 255,
    [19*28+16 ... 19*28+27] = 0,
    [20*28+ 0 ... 20*28+11] = 0, [20*28+12 ... 20*28+15] = 255,
    [20*28+16 ... 20*28+27] = 0,
    [21*28+ 0 ... 21*28+11] = 0, [21*28+12 ... 21*28+15] = 255,
    [21*28+16 ... 21*28+27] = 0,
    [22*28+ 0 ... 22*28+11] = 0, [22*28+12 ... 22*28+15] = 255,
    [22*28+16 ... 22*28+27] = 0,
    [23*28+ 0 ... 23*28+11] = 0, [23*28+12 ... 23*28+15] = 255,
    [23*28+16 ... 23*28+27] = 0,
    /* Row 24: bottom base */
    [24*28+ 0 ... 24*28+ 9] = 0,
    [24*28+10 ... 24*28+17] = 255,
    [24*28+18 ... 24*28+27] = 0,
    /* Rows 25-27: blank */
    [25*28 ... 27*28+27] = 0,
};

/* --- DIGIT 7: horizontal top bar + downward diagonal stroke ------------ */
static const unsigned char img_digit_7[IMG_ROWS * IMG_COLS] = {
    /* Rows 0-2: blank */
    [0 ... 2*28+27] = 0,
    /* Row 3: blank leading */
    [3*28+ 0 ... 3*28+27] = 0,
    /* Rows 4-5: top horizontal bar, cols 6-21 */
    [4*28+ 0 ... 4*28+ 5] = 0,  [4*28+ 6 ... 4*28+21] = 255,
    [4*28+22 ... 4*28+27] = 0,
    [5*28+ 0 ... 5*28+ 5] = 0,  [5*28+ 6 ... 5*28+21] = 255,
    [5*28+22 ... 5*28+27] = 0,
    /* Row 6: right portion only */
    [6*28+ 0 ... 6*28+16] = 0,  [6*28+17 ... 6*28+20] = 255,
    [6*28+21 ... 6*28+27] = 0,
    /* Rows 7-8: stepping left */
    [7*28+ 0 ... 7*28+15] = 0,  [7*28+16 ... 7*28+19] = 255,
    [7*28+20 ... 7*28+27] = 0,
    [8*28+ 0 ... 8*28+14] = 0,  [8*28+15 ... 8*28+18] = 255,
    [8*28+19 ... 8*28+27] = 0,
    /* Rows 9-10 */
    [9*28+ 0 ... 9*28+14] = 0,  [9*28+15 ... 9*28+18] = 255,
    [9*28+19 ... 9*28+27] = 0,
    [10*28+ 0 ... 10*28+13] = 0, [10*28+14 ... 10*28+17] = 255,
    [10*28+18 ... 10*28+27] = 0,
    /* Rows 11-12 */
    [11*28+ 0 ... 11*28+13] = 0, [11*28+14 ... 11*28+17] = 255,
    [11*28+18 ... 11*28+27] = 0,
    [12*28+ 0 ... 12*28+12] = 0, [12*28+13 ... 12*28+16] = 255,
    [12*28+17 ... 12*28+27] = 0,
    /* Rows 13-14 */
    [13*28+ 0 ... 13*28+12] = 0, [13*28+13 ... 13*28+16] = 255,
    [13*28+17 ... 13*28+27] = 0,
    [14*28+ 0 ... 14*28+11] = 0, [14*28+12 ... 14*28+15] = 255,
    [14*28+16 ... 14*28+27] = 0,
    /* Rows 15-16 */
    [15*28+ 0 ... 15*28+11] = 0, [15*28+12 ... 15*28+15] = 255,
    [15*28+16 ... 15*28+27] = 0,
    [16*28+ 0 ... 16*28+11] = 0, [16*28+12 ... 16*28+15] = 255,
    [16*28+16 ... 16*28+27] = 0,
    /* Rows 17-18 */
    [17*28+ 0 ... 17*28+10] = 0, [17*28+11 ... 17*28+14] = 255,
    [17*28+15 ... 17*28+27] = 0,
    [18*28+ 0 ... 18*28+10] = 0, [18*28+11 ... 18*28+14] = 255,
    [18*28+15 ... 18*28+27] = 0,
    /* Rows 19-20 */
    [19*28+ 0 ... 19*28+10] = 0, [19*28+11 ... 19*28+14] = 255,
    [19*28+15 ... 19*28+27] = 0,
    [20*28+ 0 ... 20*28+10] = 0, [20*28+11 ... 20*28+14] = 255,
    [20*28+15 ... 20*28+27] = 0,
    /* Rows 21-23 */
    [21*28+ 0 ... 21*28+10] = 0, [21*28+11 ... 21*28+14] = 255,
    [21*28+15 ... 21*28+27] = 0,
    [22*28+ 0 ... 22*28+10] = 0, [22*28+11 ... 22*28+14] = 255,
    [22*28+15 ... 22*28+27] = 0,
    [23*28+ 0 ... 23*28+10] = 0, [23*28+11 ... 23*28+14] = 255,
    [23*28+15 ... 23*28+27] = 0,
    /* Rows 24-27: blank */
    [24*28 ... 27*28+27] = 0,
};

/* ======================================================================== */
/* DISPLAY UTILITIES                                                        */
/* ======================================================================== */

/* Print a 28x28 image as ASCII art (downsampled 2:1 for readability). */
static void print_image(const unsigned char *img) {
    for (int r = 0; r < IMG_ROWS; r += 2) {
        printf("    ");
        for (int c = 0; c < IMG_COLS; c += 2) {
            /* Average a 2x2 block */
            int sum = (int)img[r * IMG_COLS + c]
                    + (int)img[r * IMG_COLS + c + 1]
                    + (int)img[(r+1) * IMG_COLS + c]
                    + (int)img[(r+1) * IMG_COLS + c + 1];
            if (sum > 256)
                printf("##");
            else if (sum > 0)
                printf("..");
            else
                printf("  ");
        }
        printf("\n");
    }
}

/* Print a horizontal bar chart for the output scores. */
static void print_scores(const short *scores, int predicted) {
    /* Find the maximum absolute value for scaling. */
    int max_abs = 1;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int v = scores[i] < 0 ? -scores[i] : scores[i];
        if (v > max_abs) max_abs = v;
    }

    printf("\n  Class scores (Q8.8 raw -> approx decimal):\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        char marker = (i == predicted) ? '*' : ' ';

        /* Convert Q8.8 to integer and fractional parts for display.
         * We avoid floating point by printing integer.fraction manually.
         * value = raw / 256. We compute integer = raw / 256,
         * frac = ((abs(raw) % 256) * 100) / 256 for two decimal places. */
        int raw = (int)scores[i];
        int negative = (raw < 0);
        int abs_raw = negative ? -raw : raw;
        int int_part = abs_raw >> Q8_SHIFT;
        int frac_part = ((abs_raw & 0xFF) * 100) >> Q8_SHIFT;

        /* Bar length proportional to score (only for positive values). */
        int bar_len = 0;
        if (raw > 0 && max_abs > 0)
            bar_len = (raw * 30) / max_abs;
        if (bar_len < 0) bar_len = 0;
        if (bar_len > 30) bar_len = 30;

        printf("  %c [%d] %s%d.%02d  ",
               marker, i,
               negative ? "-" : " ",
               int_part, frac_part);

        for (int b = 0; b < bar_len; b++)
            printf("|");
        printf("\n");
    }
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

/* Test image descriptors. */
struct test_case {
    const unsigned char *image;
    int expected;
    const char *name;
};

int main(void) {
    printf("================================================================\n");
    printf("  MNIST Neural Network on GPU\n");
    printf("  Fixed-Point Q8.8 | 784 -> 32 (ReLU) -> 10 (argmax)\n");
    printf("  Freestanding ARM64 on Metal Compute Shader\n");
    printf("================================================================\n\n");

    /* Initialize network weights. */
    printf("Initializing network... ");
    nn_init();
    printf("done.\n");
    printf("  Layer 1: %d x %d weights + %d biases\n",
           HIDDEN_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    printf("  Layer 2: %d x %d weights + %d biases\n",
           OUTPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("  Total parameters: %d\n",
           HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE +
           OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE);
    printf("  Arithmetic: Q8.8 fixed-point (1.0 = %d)\n\n", Q8_ONE);

    /* Prepare test images. */
    struct test_case tests[3];
    tests[0].image = img_digit_0;
    tests[0].expected = 0;
    tests[0].name = "Zero (oval ring)";
    tests[1].image = img_digit_1;
    tests[1].expected = 1;
    tests[1].name = "One (vertical bar)";
    tests[2].image = img_digit_7;
    tests[2].expected = 7;
    tests[2].name = "Seven (bar + diagonal)";

    int correct = 0;
    int total = 3;

    for (int t = 0; t < total; t++) {
        printf("----------------------------------------------------------------\n");
        printf("  Test %d: %s (expected: %d)\n", t + 1, tests[t].name,
               tests[t].expected);
        printf("----------------------------------------------------------------\n");

        /* Display the input image. */
        print_image(tests[t].image);

        /* Convert pixel values to Q8.8 input.
         * Raw pixel is 0 or 255. In Q8.8, 255 becomes 255 (= 0.996).
         * Since the pixel values ARE already in the right range for Q8.8
         * (they represent values close to 0.0 and 1.0), we use them
         * directly as Q8.8 values. 0 stays 0, 255 stays 255 ~ 1.0. */
        short input_q8[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            input_q8[i] = (short)tests[t].image[i];
        }

        /* Forward pass. */
        forward(input_q8);

        /* Classify. */
        int predicted = argmax(output, OUTPUT_SIZE);

        /* Count active hidden neurons. */
        int active_count = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (hidden[j] > 0)
                active_count++;
        }

        printf("\n  Hidden layer: %d / %d neurons active after ReLU\n",
               active_count, HIDDEN_SIZE);

        /* Print results. */
        print_scores(output, predicted);

        printf("\n  >> Predicted: %d", predicted);
        if (predicted == tests[t].expected) {
            printf("  [CORRECT]\n");
            correct++;
        } else {
            printf("  [WRONG -- expected %d]\n", tests[t].expected);
        }
        printf("\n");
    }

    /* Summary. */
    printf("================================================================\n");
    printf("  Results: %d / %d correct\n", correct, total);
    if (correct == total)
        printf("  All test digits classified correctly!\n");
    else
        printf("  Some classifications failed (expected with hand-crafted weights).\n");
    printf("================================================================\n");

    return 0;
}
