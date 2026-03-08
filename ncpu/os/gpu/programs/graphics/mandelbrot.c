/*
 * mandelbrot.c -- Fixed-point Mandelbrot set renderer for ARM64 Metal GPU.
 *
 * Renders a 320x240 Mandelbrot set using Q16.16 fixed-point arithmetic
 * (no floating point on this GPU). Writes RGBA pixels to a framebuffer
 * buffer, then calls SYS_FLUSH_FB to display.
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/graphics/mandelbrot.c
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONFIGURATION                                                            */
/* ======================================================================== */

#define WIDTH   320
#define HEIGHT  240
#define MAX_ITER 64

/* Q16.16 fixed-point arithmetic */
#define FP_SHIFT 16
#define FP_ONE   (1 << FP_SHIFT)     /* 65536 = 1.0 */
#define FP_TWO   (2 << FP_SHIFT)     /* 131072 = 2.0 */
#define FP_FOUR  (4 << FP_SHIFT)     /* 262144 = 4.0 */

/* Mandelbrot viewing window (Q16.16) */
#define X_MIN  (-2 * FP_ONE)         /* -2.0 */
#define X_MAX  (FP_ONE)              /*  1.0 */
#define Y_MIN  (-(FP_ONE + FP_ONE/2))/* -1.5 */
#define Y_MAX  (FP_ONE + FP_ONE/2)   /*  1.5 */

/* ======================================================================== */
/* FIXED-POINT MULTIPLY                                                     */
/* ======================================================================== */

/*
 * Multiply two Q16.16 values. Uses 64-bit intermediate to avoid overflow.
 */
static long fp_mul(long a, long b) {
    return (a * b) >> FP_SHIFT;
}

/* ======================================================================== */
/* COLOR PALETTE                                                            */
/* ======================================================================== */

/*
 * Map iteration count to an RGBA color.
 * Uses a smooth gradient: black (in set) → blue → cyan → yellow → red.
 */
static unsigned int iter_to_color(int iter) {
    if (iter >= MAX_ITER) return 0xFF000000; /* Black — in the set */

    /* Normalize to 0-255 range */
    int t = (iter * 255) / MAX_ITER;

    unsigned int r, g, b;

    if (t < 64) {
        /* Black → Blue */
        r = 0; g = 0; b = t * 4;
    } else if (t < 128) {
        /* Blue → Cyan */
        int s = t - 64;
        r = 0; g = s * 4; b = 255;
    } else if (t < 192) {
        /* Cyan → Yellow */
        int s = t - 128;
        r = s * 4; g = 255; b = 255 - s * 4;
    } else {
        /* Yellow → Red */
        int s = t - 192;
        r = 255; g = 255 - s * 4; b = 0;
    }

    /* Clamp */
    if (r > 255) r = 255;
    if (g > 255) g = 255;
    if (b > 255) b = 255;

    /* Pack as RGBA (little-endian: R in lowest byte) */
    return r | (g << 8) | (b << 16) | (0xFF << 24);
}

/* ======================================================================== */
/* MANDELBROT COMPUTATION                                                   */
/* ======================================================================== */

/*
 * Compute the Mandelbrot iteration count for a point (cx, cy) in Q16.16.
 *
 * z = z^2 + c, starting from z = 0.
 * Returns iteration count when |z|^2 > 4, or MAX_ITER if in the set.
 */
static int mandelbrot(long cx, long cy) {
    long zx = 0, zy = 0;
    int iter = 0;

    while (iter < MAX_ITER) {
        long zx2 = fp_mul(zx, zx);
        long zy2 = fp_mul(zy, zy);

        /* Check |z|^2 > 4.0 */
        if (zx2 + zy2 > FP_FOUR) break;

        /* z = z^2 + c */
        long new_zx = zx2 - zy2 + cx;
        long new_zy = 2 * fp_mul(zx, zy) + cy;
        zx = new_zx;
        zy = new_zy;
        iter++;
    }

    return iter;
}

/* ======================================================================== */
/* FRAMEBUFFER                                                              */
/* ======================================================================== */

/* Static framebuffer — 320*240*4 = 307,200 bytes */
static unsigned int framebuffer[WIDTH * HEIGHT];

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    printf("Mandelbrot Set Renderer (GPU Metal)\n");
    printf("Resolution: %dx%d, Max iterations: %d\n", WIDTH, HEIGHT, MAX_ITER);
    printf("Using Q16.16 fixed-point arithmetic\n");
    printf("Rendering...\n");

    /* Step sizes in fixed-point */
    long dx = (X_MAX - X_MIN) / WIDTH;
    long dy = (Y_MAX - Y_MIN) / HEIGHT;

    int pixels = 0;
    long cy = Y_MIN;

    for (int y = 0; y < HEIGHT; y++) {
        long cx = X_MIN;
        for (int x = 0; x < WIDTH; x++) {
            int iter = mandelbrot(cx, cy);
            framebuffer[y * WIDTH + x] = iter_to_color(iter);
            cx += dx;
            pixels++;
        }
        cy += dy;

        /* Progress every 48 rows */
        if ((y % 48) == 47) {
            printf("  Row %d/%d\n", y + 1, HEIGHT);
        }
    }

    printf("Rendered %d pixels\n", pixels);

    /* Flush framebuffer to display */
    sys_flush_fb(WIDTH, HEIGHT, framebuffer);

    printf("Done.\n");
    return 0;
}
