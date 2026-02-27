/*
 * DOOM-like Rendering Benchmark for ARM64
 * Mimics the rendering loop structure of DOOM
 */

.section .text
.globl _start

_start:
    /* Setup stack */
    mov sp, #0x80000

    /* Frame counter */
    mov x19, #10          /* 10 frames */

frame_loop:
    /* ============== FRAMEBUFFER CLEAR ============== */
    /* DOOM clears the 320x200 framebuffer every frame */
    mov x0, #0x20000      /* Framebuffer address */
    mov x1, #0            /* Color black */
    mov x2, #16000        /* 320*200 / 4 = 16000 words */

framebuffer_clear_loop:
    str w1, [x0], #4      /* Store and increment */
    subs x2, x2, #1       /* Decrement counter */
    b.ne framebuffer_clear_loop

    /* ============== SPAN DRAWING LOOPS ============== */
    /* DOOM draws horizontal spans for each screen row */
    mov x0, #0x20000      /* Framebuffer */
    mov x1, #0            /* Row counter */
    mov x22, #200         /* 200 rows */

span_outer_loop:
    mov x2, #0            /* Column counter */
    mov x3, #320          /* 320 pixels per row */
    mov x4, x0            /* Current row address */

span_inner_loop:
    /* Simulated texture lookup and pixel write */
    ldr w5, [x4, x2, lsl #2] /* Read pixel */
    add w5, w5, #1        /* Modify */
    str w5, [x4, x2, lsl #2] /* Write pixel */
    add x2, x2, #1        /* Next column */
    cmp x2, x3
    b.ne span_inner_loop

    /* Next row */
    add x0, x0, #1280     /* 320 * 4 = 1280 bytes per row */
    add x1, x1, #1
    cmp x1, x22
    b.ne span_outer_loop

    /* ============== TEXTURE MAPPING LOOPS ============== */
    /* DOOM maps textures to walls using vertical columns */
    mov x0, #0x50000      /* Texture data */
    mov x1, #0x20000      /* Framebuffer */
    mov x2, #0            /* Column counter */
    mov x22, #320         /* 320 columns */

texture_loop:
    /* For each column, draw vertical strip */
    mov x3, #0            /* Row counter */
    mov x4, #200          /* 200 rows */

texture_column_loop:
    /* Calculate texture coordinate */
    mov w5, #0xFF         /* White pixel */
    ldr w6, [x0], #4      /* Read from texture */

    /* Calculate framebuffer position */
    mov x7, #320
    mul x7, x3, x7        /* row * width */
    add x7, x7, x2        /* + column */
    lsl x7, x7, #2        /* * 4 (bytes per pixel) */
    add x7, x7, x1        /* + framebuffer base */

    str w5, [x7]          /* Write pixel */

    add x3, x3, #1
    cmp x3, x4
    b.ne texture_column_loop

    add x2, x2, #1
    cmp x2, x22
    b.ne texture_loop

    /* ============== SPRITE RENDERING ============== */
    /* DOOM renders sprites as 2D bitmaps */
    mov x0, #0x60000      /* Sprite data */
    mov x1, #0x20000      /* Framebuffer */
    mov x2, #0            /* Sprite row counter */
    mov x22, #64          /* 64x64 sprite */

sprite_outer_loop:
    mov x3, #0            /* Sprite column counter */
    mov x4, #64           /* 64 pixels wide */

sprite_inner_loop:
    /* Read sprite pixel */
    ldr w5, [x0], #4

    /* Skip transparent pixels */
    cmp w5, #0
    b.eq sprite_skip_pixel

    /* Write to framebuffer */
    /* Calculate position: (base_y + row) * width + (base_x + col) */
    mov x6, #100          /* base_x */
    mov x7, #50           /* base_y */
    add x6, x6, x3        /* + col */
    add x7, x7, x2        /* + row */

    mov x8, #320
    mul x8, x7, x8        /* row * width */
    add x8, x8, x6        /* + col */
    lsl x8, x8, #2        /* * 4 */
    add x8, x8, x1        /* + framebuffer base */

    str w5, [x8]          /* Write pixel */

sprite_skip_pixel:
    add x3, x3, #1
    cmp x3, x4
    b.ne sprite_inner_loop

    add x2, x2, #1
    cmp x2, x22
    b.ne sprite_outer_loop

    /* ============== Z-BUFFER CLEAR ============== */
    /* DOOM clears Z-buffer for each frame */
    mov x0, #0x70000      /* Z-buffer */
    mov x1, #0xFFFFFFFF    /* Max depth */
    mov x2, #16000        /* Same size as framebuffer */

zbuffer_loop:
    str w1, [x0], #4
    subs x2, x2, #1
    b.ne zbuffer_loop

    /* Next frame */
    subs x19, x19, #1
    b.ne frame_loop

    /* Exit */
    mov x0, #0
    mov x8, #93           /* exit syscall */
    svc #0

    /* Infinite hang if we return */
hang:
    b hang
