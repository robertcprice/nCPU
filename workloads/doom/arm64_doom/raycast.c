/*
 * Minimal ARM64 Raycasting Engine for Neural CPU
 * ================================================
 *
 * This is REAL ARM64 code that:
 * 1. Reads player state from memory
 * 2. Does raycasting with ARM64 instructions
 * 3. Writes to framebuffer in memory
 *
 * Memory Map:
 * 0x10000 - 0x10FFF: Code
 * 0x20000 - 0x2000F: Player state (x, y, angle)
 * 0x30000 - 0x300FF: Map data (16x16)
 * 0x40000 - 0x4FFFF: Framebuffer (80x25 ASCII)
 * 0x50000 - 0x50003: Keyboard input
 */

// Fixed point math (16.16 format)
#define FP_SHIFT 16
#define FP_ONE (1 << FP_SHIFT)
#define TO_FP(x) ((x) << FP_SHIFT)
#define FROM_FP(x) ((x) >> FP_SHIFT)

// Memory addresses
#define PLAYER_X    ((volatile int*)0x20000)
#define PLAYER_Y    ((volatile int*)0x20004)
#define PLAYER_ANG  ((volatile int*)0x20008)
#define MAP_BASE    ((volatile unsigned char*)0x30000)
#define FB_BASE     ((volatile unsigned char*)0x40000)
#define KEY_INPUT   ((volatile int*)0x50000)

// Screen dimensions
#define WIDTH 80
#define HEIGHT 25
#define MAP_SIZE 16

// Precomputed sin/cos table (256 entries, fixed point)
// sin[i] = sin(i * 2*PI / 256) * FP_ONE
static const int sin_table[256] = {
    0, 1608, 3212, 4808, 6393, 7962, 9512, 11039,
    12540, 14010, 15447, 16846, 18205, 19520, 20788, 22006,
    23170, 24279, 25330, 26320, 27246, 28106, 28899, 29622,
    30274, 30853, 31357, 31786, 32138, 32413, 32610, 32729,
    32768, 32729, 32610, 32413, 32138, 31786, 31357, 30853,
    30274, 29622, 28899, 28106, 27246, 26320, 25330, 24279,
    23170, 22006, 20788, 19520, 18205, 16846, 15447, 14010,
    12540, 11039, 9512, 7962, 6393, 4808, 3212, 1608,
    0, -1608, -3212, -4808, -6393, -7962, -9512, -11039,
    -12540, -14010, -15447, -16846, -18205, -19520, -20788, -22006,
    -23170, -24279, -25330, -26320, -27246, -28106, -28899, -29622,
    -30274, -30853, -31357, -31786, -32138, -32413, -32610, -32729,
    -32768, -32729, -32610, -32413, -32138, -31786, -31357, -30853,
    -30274, -29622, -28899, -28106, -27246, -26320, -25330, -24279,
    -23170, -22006, -20788, -19520, -18205, -16846, -15447, -14010,
    -12540, -11039, -9512, -7962, -6393, -4808, -3212, -1608,
    0, 1608, 3212, 4808, 6393, 7962, 9512, 11039,
    12540, 14010, 15447, 16846, 18205, 19520, 20788, 22006,
    23170, 24279, 25330, 26320, 27246, 28106, 28899, 29622,
    30274, 30853, 31357, 31786, 32138, 32413, 32610, 32729,
    32768, 32729, 32610, 32413, 32138, 31786, 31357, 30853,
    30274, 29622, 28899, 28106, 27246, 26320, 25330, 24279,
    23170, 22006, 20788, 19520, 18205, 16846, 15447, 14010,
    12540, 11039, 9512, 7962, 6393, 4808, 3212, 1608,
    0, -1608, -3212, -4808, -6393, -7962, -9512, -11039,
    -12540, -14010, -15447, -16846, -18205, -19520, -20788, -22006,
    -23170, -24279, -25330, -26320, -27246, -28106, -28899, -29622,
    -30274, -30853, -31357, -31786, -32138, -32413, -32610, -32729,
    -32768, -32729, -32610, -32413, -32138, -31786, -31357, -30853,
    -30274, -29622, -28899, -28106, -27246, -26320, -25330, -24279,
    -23170, -22006, -20788, -19520, -18205, -16846, -15447, -14010,
    -12540, -11039, -9512, -7962, -6393, -4808, -3212, -1608
};

// Get sin value (angle in 0-255 range)
static inline int fp_sin(int angle) {
    return sin_table[angle & 255];
}

// Get cos value (cos = sin + 64)
static inline int fp_cos(int angle) {
    return sin_table[(angle + 64) & 255];
}

// Fixed point multiply
static inline int fp_mul(int a, int b) {
    return (int)(((long long)a * b) >> FP_SHIFT);
}

// Check if map cell is wall
static inline int is_wall(int x, int y) {
    if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE) return 1;
    return MAP_BASE[y * MAP_SIZE + x];
}

// Cast single ray, return distance (fixed point)
static int cast_ray(int px, int py, int angle) {
    int cos_a = fp_cos(angle);
    int sin_a = fp_sin(angle);

    int ray_x = px;
    int ray_y = py;

    // Step size (0.05 in fixed point = 3277)
    int step = 3277;

    for (int i = 0; i < 200; i++) {
        ray_x += fp_mul(cos_a, step);
        ray_y += fp_mul(sin_a, step);

        int map_x = FROM_FP(ray_x);
        int map_y = FROM_FP(ray_y);

        if (is_wall(map_x, map_y)) {
            // Calculate distance
            int dx = ray_x - px;
            int dy = ray_y - py;
            // Approximate distance (Manhattan for simplicity)
            int dist = (dx > 0 ? dx : -dx) + (dy > 0 ? dy : -dy);
            return dist;
        }
    }

    return TO_FP(10); // Max distance
}

// Render frame to framebuffer
void render_frame(void) {
    int px = *PLAYER_X;
    int py = *PLAYER_Y;
    int pa = *PLAYER_ANG;

    // FOV: 60 degrees = ~42 angle units (256 = 360 degrees)
    int fov = 42;
    int half_fov = fov / 2;

    // Wall characters based on distance
    static const char wall_chars[] = "@#%*+=-:. ";

    for (int x = 0; x < WIDTH; x++) {
        // Calculate ray angle
        int ray_angle = pa - half_fov + (x * fov / WIDTH);

        // Cast ray
        int dist = cast_ray(px, py, ray_angle);

        // Calculate wall height
        int wall_height = HEIGHT * FP_ONE / (dist + 1);
        if (wall_height > HEIGHT) wall_height = HEIGHT;

        int wall_top = (HEIGHT - wall_height) / 2;
        int wall_bottom = wall_top + wall_height;

        // Choose character based on distance
        int char_idx = FROM_FP(dist) / 2;
        if (char_idx > 9) char_idx = 9;
        char wall_char = wall_chars[char_idx];

        // Draw column
        for (int y = 0; y < HEIGHT; y++) {
            char c;
            if (y < wall_top) {
                c = ' ';  // Ceiling
            } else if (y >= wall_bottom) {
                c = '.';  // Floor
            } else {
                c = wall_char;  // Wall
            }
            FB_BASE[y * WIDTH + x] = c;
        }
    }
}

// Process keyboard input
void process_input(void) {
    int key = *KEY_INPUT;
    *KEY_INPUT = 0;  // Clear input

    int px = *PLAYER_X;
    int py = *PLAYER_Y;
    int pa = *PLAYER_ANG;

    // Movement speed (0.1 in fixed point)
    int move_speed = 6554;
    // Turn speed (5 angle units)
    int turn_speed = 5;

    switch (key) {
        case 'w':  // Forward
            px += fp_mul(fp_cos(pa), move_speed);
            py += fp_mul(fp_sin(pa), move_speed);
            break;
        case 's':  // Backward
            px -= fp_mul(fp_cos(pa), move_speed);
            py -= fp_mul(fp_sin(pa), move_speed);
            break;
        case 'a':  // Turn left
            pa -= turn_speed;
            break;
        case 'd':  // Turn right
            pa += turn_speed;
            break;
    }

    // Collision detection
    int map_x = FROM_FP(px);
    int map_y = FROM_FP(py);
    if (!is_wall(map_x, map_y)) {
        *PLAYER_X = px;
        *PLAYER_Y = py;
    }
    *PLAYER_ANG = pa & 255;
}

// Main entry point
void _start(void) {
    // Initialize player position (5.5, 5.5 in fixed point)
    *PLAYER_X = TO_FP(5) + FP_ONE/2;
    *PLAYER_Y = TO_FP(5) + FP_ONE/2;
    *PLAYER_ANG = 0;

    // Main loop
    while (1) {
        process_input();
        render_frame();
    }
}
