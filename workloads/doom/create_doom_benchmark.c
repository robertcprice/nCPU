/**
 * DOOM-like Rendering Benchmark
 *
 * This program generates ARM64 code that mimics DOOM's rendering loops:
 * - Pixel plotting loops (framebuffer clearing)
 * - Span drawing (horizontal line rendering)
 * - Texture mapping loops
 * - Sprite rendering
 *
 * Compile: gcc -o create_doom_benchmark create_doom_benchmark.c
 * Run: ./create_doom_benchmark > doom_benchmark.s
 * Then: aarch64-linux-gnu-as doom_benchmark.s -o doom_benchmark.o
 *      aarch64-linux-gnu-ld doom_benchmark.o -o doom_benchmark.elf
 */

#include <stdio.h>
#include <stdint.h>

// Helper to print ARM64 instructions
#define INSTR(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

void generate_memset_loop(int label) {
    INSTR("/* Loop %d: Framebuffer clear (memset-like) */", label);
    INSTR("memset_loop_%d:", label);
    INSTR("    mov x0, #0x20000      /* Framebuffer address */");
    INSTR("    mov x1, #0            /* Value to write */");
    INSTR("    mov x2, #0x10000      /* Count (64k pixels) */");
    INSTR("memset_loop_%d_body:", label);
    INSTR("    str w1, [x0], #4      /* Store and increment */");
    INSTR("    subs x2, x2, #1       /* Decrement counter */");
    INSTR("    b.ne memset_loop_%d_body /* Branch back if not zero */", label);
    INSTR("");
}

void generate_span_drawing_loop(int label) {
    INSTR("/* Loop %d: Span drawing (horizontal line) */", label);
    INSTR("span_loop_%d:", label);
    INSTR("    mov x0, #0x30000      /* Destination address */");
    INSTR("    mov x1, #0            /* Texture U coordinate */");
    INSTR("    mov x2, #320          /* Screen width */");
    INSTR("span_loop_%d_body:", label);
    INSTR("    /* Simulated texture lookup */");
    INSTR("    ldr w3, [x0, x1, lsl #2] /* Load pixel */");
    INSTR("    add w3, w3, #1        /* Brighten */");
    INSTR("    str w3, [x0, x1, lsl #2] /* Store pixel */");
    INSTR("    add x1, x1, #1        /* Next pixel */");
    INSTR("    cmp x1, x2            /* Check if done */");
    INSTR("    b.ne span_loop_%d_body", label);
    INSTR("");
}

void generate_texture_mapping_loop(int label) {
    INSTR("/* Loop %d: Texture mapping (vertical column) */", label);
    INSTR("texmap_loop_%d:", label);
    INSTR("    mov x0, #0x40000      /* Framebuffer base */");
    INSTR("    mov x1, #0            /* Y coordinate */");
    INSTR("    mov x2, #200          /* Height */");
    INSTR("    mov x3, #320          /* Width (stride) */");
    INSTR("texmap_loop_%d_body:", label);
    INSTR("    /* Calculate framebuffer address */");
    INSTR("    mov x4, x1");
    INSTR("    mul x4, x4, x3        /* y * stride */");
    INSTR("    add x4, x4, x0        /* + base address */");
    INSTR("    /* Store pixel */");
    INSTR("    mov w5, #0xFF         /* White pixel */");
    INSTR("    str w5, [x4]          /* Write pixel */");
    INSTR("    add x1, x1, #1        /* Increment Y */");
    INSTR("    cmp x1, x2            /* Check if done */");
    INSTR("    b.ne texmap_loop_%d_body", label);
    INSTR("");
}

void generate_sprite_rendering_loop(int label) {
    INSTR("/* Loop %d: Sprite rendering (nested loops) */", label);
    INSTR("sprite_loop_%d:", label);
    INSTR("    mov x0, #0x50000      /* Sprite data */");
    INSTR("    mov x1, #0            /* Row counter */");
    INSTR("    mov x2, #64           /* Sprite height */");
    INSTR("sprite_outer_%d:", label);
    INSTR("    mov x3, #0            /* Column counter */");
    INSTR("    mov x4, #64           /* Sprite width */");
    INSTR("sprite_inner_%d:", label);
    INSTR("    /* Load sprite pixel */");
    INSTR("    ldr w5, [x0], #4      /* Load and increment */");
    INSTR("    /* Store to framebuffer */");
    INSTR("    ldr x6, =0x20000      /* Framebuffer base */");
    INSTR("    str w5, [x6, x3, lsl #2] /* Store with offset */");
    INSTR("    add x3, x3, #1        /* Next column */");
    INSTR("    cmp x3, x4            /* Check if row done */");
    INSTR("    b.ne sprite_inner_%d", label);
    INSTR("    add x1, x1, #1        /* Next row */");
    INSTR("    cmp x1, x2            /* Check if sprite done */");
    INSTR("    b.ne sprite_outer_%d", label);
    INSTR("");
}

void generate_z_buffer_loop(int label) {
    INSTR("/* Loop %d: Z-buffer clearing */", label);
    INSTR("zbuffer_loop_%d:", label);
    INSTR("    mov x0, #0x60000      /* Z-buffer address */");
    INSTR("    mov x1, #0xFFFFFFFF    /* Max Z value */");
    INSTR("    mov x2, #0x8000       /* 16k depth values */");
    INSTR("zbuffer_loop_%d_body:", label);
    INSTR("    str w1, [x0], #4      /* Store Z value */");
    INSTR("    subs x2, x2, #1       /* Decrement */");
    INSTR("    b.ne zbuffer_loop_%d_body", label);
    INSTR("");
}

int main() {
    INSTR("/*");
    INSTR(" * DOOM-like Rendering Benchmark");
    INSTR(" * Simulates the loop-heavy rendering operations");
    INSTR(" */");
    INSTR("");
    INSTR(".section .text");
    INSTR(".globl _start");
    INSTR("");
    INSTR("_start:");
    INSTR("    /* Setup */");
    INSTR("    mov sp, #0x80000");
    INSTR("");
    INSTR("    /* Run multiple frames */");
    INSTR("    mov x19, #100         /* 100 frames */");
    INSTR("frame_loop:");
    INSTR("");

    // Generate different types of rendering loops
    // DOOM does this every frame!
    for (int frame = 0; frame < 5; frame++) {
        generate_memset_loop(frame);
        generate_span_drawing_loop(frame);
        generate_texture_mapping_loop(frame);
        generate_z_buffer_loop(frame);
    }

    INSTR("    /* Decrement frame counter */");
    INSTR("    subs x19, x19, #1");
    INSTR("    b.ne frame_loop");
    INSTR("");
    INSTR("    /* Exit */");
    INSTR("    mov x0, #0");
    INSTR("    mov x8, #93           /* exit syscall */");
    INSTR("    svc #0");
    INSTR("");

    return 0;
}
