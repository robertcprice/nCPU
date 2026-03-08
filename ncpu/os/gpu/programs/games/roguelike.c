/*
 * Roguelike Dungeon Crawler -- Freestanding C for ARM64 Metal GPU kernel.
 *
 * A turn-based dungeon crawler with procedural BSP dungeon generation,
 * bump-to-attack combat, potions, gold, and descending stairs.
 *
 * Map: 60x20 display. Characters:
 *   @ player   # wall   . floor   + door   > stairs down
 *   M monster  $ gold   ! potion
 *
 * Controls: wasd / hjkl movement, > descend stairs, q quit
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/games/roguelike.c
 *          -o /tmp/roguelike.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ========================================================================== */
/* CONSTANTS                                                                  */
/* ========================================================================== */

#define MAP_W       60
#define MAP_H       20
#define MAP_SIZE    (MAP_W * MAP_H)

#define MAX_ROOMS   8
#define MIN_ROOM_W  5
#define MIN_ROOM_H  3
#define MAX_ROOM_W  14
#define MAX_ROOM_H  7

#define MAX_MONSTERS 20
#define MAX_ITEMS    30

#define PLAYER_HP   20
#define PLAYER_ATK  5

#define TILE_WALL   '#'
#define TILE_FLOOR  '.'
#define TILE_DOOR   '+'
#define TILE_STAIRS '>'
#define TILE_VOID   ' '

/* ========================================================================== */
/* DATA STRUCTURES                                                            */
/* ========================================================================== */

struct rect {
    int x, y, w, h;
};

struct monster {
    int x, y;
    int hp, max_hp;
    int atk;
    int alive;
    char glyph;
    char name[16];
};

struct item {
    int x, y;
    int active;
    char glyph;   /* '$' for gold, '!' for potion */
    int value;     /* gold amount or heal amount */
};

struct player {
    int x, y;
    int hp, max_hp;
    int atk;
    int gold;
    int depth;
};

/* ========================================================================== */
/* GLOBAL STATE                                                               */
/* ========================================================================== */

static char map[MAP_SIZE];

static struct rect rooms[MAX_ROOMS];
static int num_rooms;

static struct monster monsters[MAX_MONSTERS];
static int num_monsters;

static struct item items[MAX_ITEMS];
static int num_items;

static struct player player;

static char msg_buf[128];
static int msg_active;

static int game_running;
static int turns;

/* ========================================================================== */
/* UTILITY                                                                    */
/* ========================================================================== */

static int abs_val(int x) {
    return x < 0 ? -x : x;
}

/* Random number in range [lo, hi] inclusive. */
static int rand_range(int lo, int hi) {
    if (lo >= hi) return lo;
    int range = hi - lo + 1;
    int r = rand() % range;
    if (r < 0) r = -r;
    return lo + r;
}

/* ========================================================================== */
/* MAP ACCESS                                                                 */
/* ========================================================================== */

static char map_get(int x, int y) {
    if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H)
        return TILE_WALL;
    return map[y * MAP_W + x];
}

static void map_set(int x, int y, char tile) {
    if (x >= 0 && x < MAP_W && y >= 0 && y < MAP_H)
        map[y * MAP_W + x] = tile;
}

/* ========================================================================== */
/* MESSAGE SYSTEM                                                             */
/* ========================================================================== */

static void msg(const char *text) {
    int i = 0;
    while (text[i] && i < 126) {
        msg_buf[i] = text[i];
        i++;
    }
    msg_buf[i] = '\0';
    msg_active = 1;
}

static void msg_fmt2(const char *a, const char *b) {
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%s%s", a, b);
    msg(tmp);
}

static void msg_combat(const char *who, const char *target, int dmg) {
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "%s hits %s for %d damage!", who, target, dmg);
    msg(tmp);
}

static void msg_kill(const char *who) {
    char tmp[128];
    snprintf(tmp, sizeof(tmp), "You killed the %s!", who);
    msg(tmp);
}

/* ========================================================================== */
/* DUNGEON GENERATION                                                         */
/* ========================================================================== */

/* Fill the entire map with walls. */
static void map_fill_walls(void) {
    for (int i = 0; i < MAP_SIZE; i++)
        map[i] = TILE_WALL;
}

/* Carve a rectangular room into the map. */
static void carve_room(const struct rect *r) {
    for (int y = r->y; y < r->y + r->h; y++) {
        for (int x = r->x; x < r->x + r->w; x++) {
            map_set(x, y, TILE_FLOOR);
        }
    }
}

/* Check if a room overlaps any existing room (with 1-tile margin). */
static int room_overlaps(const struct rect *r) {
    for (int i = 0; i < num_rooms; i++) {
        struct rect *o = &rooms[i];
        if (r->x - 1 < o->x + o->w &&
            r->x + r->w + 1 > o->x &&
            r->y - 1 < o->y + o->h &&
            r->y + r->h + 1 > o->y) {
            return 1;
        }
    }
    return 0;
}

/* Center of a room. */
static void room_center(const struct rect *r, int *cx, int *cy) {
    *cx = r->x + r->w / 2;
    *cy = r->y + r->h / 2;
}

/* Carve a horizontal corridor. */
static void carve_h_corridor(int x1, int x2, int y) {
    int start = x1 < x2 ? x1 : x2;
    int end   = x1 < x2 ? x2 : x1;
    for (int x = start; x <= end; x++) {
        if (map_get(x, y) == TILE_WALL)
            map_set(x, y, TILE_FLOOR);
    }
}

/* Carve a vertical corridor. */
static void carve_v_corridor(int y1, int y2, int x) {
    int start = y1 < y2 ? y1 : y2;
    int end   = y1 < y2 ? y2 : y1;
    for (int y = start; y <= end; y++) {
        if (map_get(x, y) == TILE_WALL)
            map_set(x, y, TILE_FLOOR);
    }
}

/* Place doors where corridors meet room edges. */
static void place_doors(void) {
    for (int y = 1; y < MAP_H - 1; y++) {
        for (int x = 1; x < MAP_W - 1; x++) {
            if (map_get(x, y) != TILE_FLOOR) continue;

            /* A door candidate: floor tile with walls on two opposite sides
               and floor on the other two. This detects corridor-room junctions. */
            int n = (map_get(x, y - 1) == TILE_FLOOR) ? 1 : 0;
            int s = (map_get(x, y + 1) == TILE_FLOOR) ? 1 : 0;
            int w = (map_get(x - 1, y) == TILE_FLOOR) ? 1 : 0;
            int e = (map_get(x + 1, y) == TILE_FLOOR) ? 1 : 0;

            /* Horizontal passage through walls */
            if (w && e && !n && !s) {
                /* Check if this is a chokepoint (wall above AND below) */
                if (map_get(x, y - 1) == TILE_WALL &&
                    map_get(x, y + 1) == TILE_WALL) {
                    /* Only place door with some probability to avoid too many */
                    if (rand_range(0, 3) == 0)
                        map_set(x, y, TILE_DOOR);
                }
            }
            /* Vertical passage through walls */
            if (n && s && !w && !e) {
                if (map_get(x - 1, y) == TILE_WALL &&
                    map_get(x + 1, y) == TILE_WALL) {
                    if (rand_range(0, 3) == 0)
                        map_set(x, y, TILE_DOOR);
                }
            }
        }
    }
}

/* Generate a full dungeon level. */
static void generate_dungeon(void) {
    map_fill_walls();
    num_rooms = 0;
    num_monsters = 0;
    num_items = 0;

    /* Target 4-6 rooms */
    int target_rooms = rand_range(4, 6);
    int attempts = 0;

    while (num_rooms < target_rooms && attempts < 200) {
        attempts++;

        int rw = rand_range(MIN_ROOM_W, MAX_ROOM_W);
        int rh = rand_range(MIN_ROOM_H, MAX_ROOM_H);
        int rx = rand_range(1, MAP_W - rw - 1);
        int ry = rand_range(1, MAP_H - rh - 1);

        struct rect candidate;
        candidate.x = rx;
        candidate.y = ry;
        candidate.w = rw;
        candidate.h = rh;

        if (room_overlaps(&candidate))
            continue;

        rooms[num_rooms] = candidate;
        carve_room(&candidate);
        num_rooms++;
    }

    /* Connect rooms with L-shaped corridors (horizontal then vertical). */
    for (int i = 1; i < num_rooms; i++) {
        int cx1, cy1, cx2, cy2;
        room_center(&rooms[i - 1], &cx1, &cy1);
        room_center(&rooms[i], &cx2, &cy2);

        /* Alternate corridor direction for variety. */
        if (rand_range(0, 1) == 0) {
            carve_h_corridor(cx1, cx2, cy1);
            carve_v_corridor(cy1, cy2, cx2);
        } else {
            carve_v_corridor(cy1, cy2, cx1);
            carve_h_corridor(cx1, cx2, cy2);
        }
    }

    /* Place doors at corridor-room junctions. */
    place_doors();

    /* Place stairs in the last room. */
    {
        int sx, sy;
        room_center(&rooms[num_rooms - 1], &sx, &sy);
        map_set(sx, sy, TILE_STAIRS);
    }

    /* Place monsters in rooms (skip first room = player spawn). */
    for (int r = 1; r < num_rooms; r++) {
        int count = rand_range(1, 3);
        for (int c = 0; c < count && num_monsters < MAX_MONSTERS; c++) {
            int mx = rand_range(rooms[r].x + 1, rooms[r].x + rooms[r].w - 2);
            int my = rand_range(rooms[r].y + 1, rooms[r].y + rooms[r].h - 2);

            if (map_get(mx, my) != TILE_FLOOR) continue;

            struct monster *m = &monsters[num_monsters];
            m->x = mx;
            m->y = my;
            m->hp = rand_range(5, 10);
            m->max_hp = m->hp;
            m->atk = rand_range(2, 4);
            m->alive = 1;
            m->glyph = 'M';

            /* Give monsters different names based on depth for flavor. */
            int kind = rand_range(0, 4);
            switch (kind) {
                case 0: strcpy(m->name, "Goblin");  break;
                case 1: strcpy(m->name, "Rat");     break;
                case 2: strcpy(m->name, "Bat");     break;
                case 3: strcpy(m->name, "Slime");   break;
                default: strcpy(m->name, "Kobold"); break;
            }

            num_monsters++;
        }
    }

    /* Place gold and potions in rooms (skip first room). */
    for (int r = 1; r < num_rooms; r++) {
        /* Gold */
        int gold_count = rand_range(0, 2);
        for (int g = 0; g < gold_count && num_items < MAX_ITEMS; g++) {
            int ix = rand_range(rooms[r].x + 1, rooms[r].x + rooms[r].w - 2);
            int iy = rand_range(rooms[r].y + 1, rooms[r].y + rooms[r].h - 2);
            if (map_get(ix, iy) != TILE_FLOOR) continue;

            struct item *it = &items[num_items];
            it->x = ix;
            it->y = iy;
            it->active = 1;
            it->glyph = '$';
            it->value = rand_range(5, 25);
            num_items++;
        }

        /* Potions */
        if (rand_range(0, 2) == 0 && num_items < MAX_ITEMS) {
            int ix = rand_range(rooms[r].x + 1, rooms[r].x + rooms[r].w - 2);
            int iy = rand_range(rooms[r].y + 1, rooms[r].y + rooms[r].h - 2);
            if (map_get(ix, iy) == TILE_FLOOR) {
                struct item *it = &items[num_items];
                it->x = ix;
                it->y = iy;
                it->active = 1;
                it->glyph = '!';
                it->value = rand_range(3, 8);
                num_items++;
            }
        }
    }

    /* Place player in the center of the first room. */
    room_center(&rooms[0], &player.x, &player.y);
}

/* ========================================================================== */
/* ENTITY QUERIES                                                             */
/* ========================================================================== */

/* Find a living monster at position (x,y). Returns index or -1. */
static int monster_at(int x, int y) {
    for (int i = 0; i < num_monsters; i++) {
        if (monsters[i].alive && monsters[i].x == x && monsters[i].y == y)
            return i;
    }
    return -1;
}

/* Find an active item at position (x,y). Returns index or -1. */
static int item_at(int x, int y) {
    for (int i = 0; i < num_items; i++) {
        if (items[i].active && items[i].x == x && items[i].y == y)
            return i;
    }
    return -1;
}

/* ========================================================================== */
/* RENDERING                                                                  */
/* ========================================================================== */

/*
 * Render the dungeon into a single output buffer and write it all at once.
 * This avoids per-character syscalls and reduces GPU dispatch overhead.
 *
 * Layout:
 *   Line 0:      message line (if active)
 *   Lines 1-20:  map (60 cols x 20 rows)
 *   Line 21:     status bar
 *   Line 22:     controls hint
 */
static void render(void) {
    /* ANSI: clear screen and move cursor home */
    printf("\033[2J\033[H");

    /* Message line */
    if (msg_active) {
        printf("\033[1;33m%s\033[0m\n", msg_buf);
        msg_active = 0;
    } else {
        printf("\n");
    }

    /* Build each row of the map. */
    char line[MAP_W + 2];  /* +1 newline, +1 null */

    for (int y = 0; y < MAP_H; y++) {
        for (int x = 0; x < MAP_W; x++) {
            char c = map_get(x, y);

            /* Check for player */
            if (x == player.x && y == player.y) {
                c = '@';
            } else {
                /* Check for monster */
                int mi = monster_at(x, y);
                if (mi >= 0) {
                    c = monsters[mi].glyph;
                } else {
                    /* Check for item */
                    int ii = item_at(x, y);
                    if (ii >= 0) {
                        c = items[ii].glyph;
                    }
                }
            }

            line[x] = c;
        }
        line[MAP_W] = '\n';
        line[MAP_W + 1] = '\0';

        /* Color specific characters for visual clarity. */
        /* We print character-by-character to inject ANSI codes. */
        for (int x = 0; x < MAP_W; x++) {
            char ch = line[x];
            switch (ch) {
                case '@':
                    printf("\033[1;32m@\033[0m");
                    break;
                case 'M':
                    printf("\033[1;31mM\033[0m");
                    break;
                case '$':
                    printf("\033[1;33m$\033[0m");
                    break;
                case '!':
                    printf("\033[1;35m!\033[0m");
                    break;
                case '>':
                    printf("\033[1;36m>\033[0m");
                    break;
                case '+':
                    printf("\033[0;33m+\033[0m");
                    break;
                case '#':
                    printf("\033[0;37m#\033[0m");
                    break;
                case '.':
                    printf("\033[0;90m.\033[0m");
                    break;
                default:
                    printf("%c", ch);
                    break;
            }
        }
        printf("\n");
    }

    /* Status bar */
    printf("\033[1;37m");
    printf("HP: %d/%d  Gold: %d  Depth: %d  Turn: %d",
           player.hp, player.max_hp, player.gold, player.depth, turns);
    printf("\033[0m\n");

    /* Controls */
    printf("\033[0;90mwasd/hjkl:move  >:descend  q:quit\033[0m\n");
}

/* ========================================================================== */
/* MONSTER AI                                                                 */
/* ========================================================================== */

/* Simple AI: if a monster is within 5 tiles of the player, move toward them.
   Otherwise stay put. Monsters only move on floor/door tiles. */
static int tile_walkable(int x, int y) {
    char t = map_get(x, y);
    return (t == TILE_FLOOR || t == TILE_DOOR || t == TILE_STAIRS);
}

static void monsters_act(void) {
    for (int i = 0; i < num_monsters; i++) {
        struct monster *m = &monsters[i];
        if (!m->alive) continue;

        int dx = player.x - m->x;
        int dy = player.y - m->y;
        int dist = abs_val(dx) + abs_val(dy);

        /* Only chase if within detection range */
        if (dist > 6) continue;

        /* If adjacent, attack the player */
        if (dist == 1) {
            int dmg = m->atk;
            player.hp -= dmg;
            msg_combat(m->name, "you", dmg);
            if (player.hp <= 0) {
                player.hp = 0;
                msg("You died! Game over.");
                game_running = 0;
            }
            continue;
        }

        /* Move toward player (prefer axis with greater distance). */
        int mx = 0, my = 0;
        if (abs_val(dx) >= abs_val(dy)) {
            mx = (dx > 0) ? 1 : -1;
        } else {
            my = (dy > 0) ? 1 : -1;
        }

        int nx = m->x + mx;
        int ny = m->y + my;

        /* Only move onto walkable, unoccupied tiles. */
        if (tile_walkable(nx, ny) &&
            monster_at(nx, ny) < 0 &&
            !(nx == player.x && ny == player.y)) {
            m->x = nx;
            m->y = ny;
        }
    }
}

/* ========================================================================== */
/* PLAYER ACTIONS                                                             */
/* ========================================================================== */

/* Check if the player can walk onto a tile. */
static int player_can_walk(int x, int y) {
    char t = map_get(x, y);
    return (t == TILE_FLOOR || t == TILE_DOOR || t == TILE_STAIRS);
}

/* Try to move the player. If a monster is at the target, attack it instead. */
static void player_move(int dx, int dy) {
    int nx = player.x + dx;
    int ny = player.y + dy;

    /* Bounds check */
    if (nx < 0 || nx >= MAP_W || ny < 0 || ny >= MAP_H) return;

    /* Check for monster (bump attack) */
    int mi = monster_at(nx, ny);
    if (mi >= 0) {
        struct monster *m = &monsters[mi];
        int dmg = player.atk;
        m->hp -= dmg;
        if (m->hp <= 0) {
            m->alive = 0;
            msg_kill(m->name);
            /* Drop gold on death */
            if (num_items < MAX_ITEMS) {
                struct item *loot = &items[num_items];
                loot->x = m->x;
                loot->y = m->y;
                loot->active = 1;
                loot->glyph = '$';
                loot->value = rand_range(3, 12);
                num_items++;
            }
        } else {
            msg_combat("You", m->name, dmg);
        }
        return;
    }

    /* Check walkability */
    if (!player_can_walk(nx, ny)) return;

    /* Move */
    player.x = nx;
    player.y = ny;

    /* Pick up items */
    int ii = item_at(nx, ny);
    if (ii >= 0) {
        struct item *it = &items[ii];
        if (it->glyph == '$') {
            player.gold += it->value;
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "Picked up %d gold!", it->value);
            msg(tmp);
        } else if (it->glyph == '!') {
            int heal = it->value;
            player.hp += heal;
            if (player.hp > player.max_hp)
                player.hp = player.max_hp;
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "Drank a potion! Healed %d HP.", heal);
            msg(tmp);
        }
        it->active = 0;
    }
}

/* Try to descend stairs. */
static void try_descend(void) {
    if (map_get(player.x, player.y) == TILE_STAIRS) {
        player.depth++;
        char tmp[64];
        snprintf(tmp, sizeof(tmp), "You descend to depth %d...", player.depth);
        msg(tmp);

        /* Scale difficulty slightly: monsters get stronger each level. */
        generate_dungeon();

        /* Boost monsters for deeper levels. */
        for (int i = 0; i < num_monsters; i++) {
            monsters[i].hp += player.depth - 1;
            monsters[i].max_hp += player.depth - 1;
            monsters[i].atk += (player.depth - 1) / 2;
        }
    } else {
        msg("No stairs here.");
    }
}

/* ========================================================================== */
/* GAME INIT                                                                  */
/* ========================================================================== */

static void game_init(void) {
    player.hp = PLAYER_HP;
    player.max_hp = PLAYER_HP;
    player.atk = PLAYER_ATK;
    player.gold = 0;
    player.depth = 1;

    msg_active = 0;
    msg_buf[0] = '\0';
    game_running = 1;
    turns = 0;

    srand(clock_ms());

    generate_dungeon();
    msg("Welcome to the dungeon! Find the stairs (>) to descend.");
}

/* ========================================================================== */
/* TITLE SCREEN                                                               */
/* ========================================================================== */

static void show_title(void) {
    printf("\033[2J\033[H");
    printf("\n");
    printf("\033[1;33m");
    printf("  ######   #######   ######   ##   ##  ########\n");
    printf("  ##   ##  ##   ##  ##    ##  ##   ##  ##\n");
    printf("  ######   ##   ##  ##        ##   ##  ######\n");
    printf("  ##   ##  ##   ##  ##   ###  ##   ##  ##\n");
    printf("  ##   ##  ##   ##  ##    ##  ##   ##  ##\n");
    printf("  ##   ##  #######   ######    #####   ########\n");
    printf("\033[0m\n");
    printf("\033[1;36m");
    printf("     D U N G E O N   C R A W L E R\n");
    printf("\033[0m\n");
    printf("     Running on ARM64 Metal GPU\n");
    printf("     Neural CPU Project (nCPU)\n");
    printf("\n");
    printf("  \033[0;90mControls:\033[0m\n");
    printf("    wasd / hjkl  -- Move\n");
    printf("    >            -- Descend stairs\n");
    printf("    q            -- Quit\n");
    printf("\n");
    printf("  \033[0;90mSymbols:\033[0m\n");
    printf("    \033[1;32m@\033[0m Player   ");
    printf("\033[1;31mM\033[0m Monster  ");
    printf("\033[1;33m$\033[0m Gold\n");
    printf("    \033[1;35m!\033[0m Potion   ");
    printf("\033[1;36m>\033[0m Stairs   ");
    printf("\033[0;33m+\033[0m Door\n");
    printf("\n");
    printf("  Press any key to begin...\n");
}

/* ========================================================================== */
/* DEATH SCREEN                                                               */
/* ========================================================================== */

static void show_death(void) {
    printf("\033[2J\033[H");
    printf("\n");
    printf("\033[1;31m");
    printf("   ____    _    __  __ _____    _____     _______ ____\n");
    printf("  / ___|  / \\  |  \\/  | ____|  / _ \\ \\   / / ____|  _ \\\n");
    printf("  | |  _ / _ \\ | |\\/| |  _|   | | | \\ \\ / /|  _| | |_) |\n");
    printf("  | |_| / ___ \\| |  | | |___  | |_| |\\ V / | |___|  _ <\n");
    printf("   \\____|_/   \\_\\_|  |_|_____|  \\___/  \\_/  |_____|_| \\_\\\n");
    printf("\033[0m\n\n");
    printf("  You perished on depth %d after %d turns.\n", player.depth, turns);
    printf("  Gold collected: %d\n", player.gold);
    printf("\n");
    printf("  Press any key to exit...\n");
}

/* ========================================================================== */
/* MAIN LOOP                                                                  */
/* ========================================================================== */

int main(void) {
    show_title();

    /* Wait for any keypress to start */
    while (1) {
        int ch = getchar();
        if (ch != -1) break;
        sleep_ms(50);
    }

    game_init();

    while (game_running) {
        render();

        /* Wait for input */
        int ch = -1;
        while (ch == -1) {
            ch = getchar();
            if (ch == -1) sleep_ms(50);
        }

        int moved = 0;

        switch (ch) {
            /* Movement: wasd */
            case 'w': player_move( 0, -1); moved = 1; break;
            case 's': player_move( 0,  1); moved = 1; break;
            case 'a': player_move(-1,  0); moved = 1; break;
            case 'd': player_move( 1,  0); moved = 1; break;

            /* Movement: vi keys (hjkl) */
            case 'h': player_move(-1,  0); moved = 1; break;
            case 'j': player_move( 0,  1); moved = 1; break;
            case 'k': player_move( 0, -1); moved = 1; break;
            case 'l': player_move( 1,  0); moved = 1; break;

            /* Descend stairs */
            case '>': try_descend(); moved = 1; break;

            /* Quit */
            case 'q':
                game_running = 0;
                break;

            default:
                break;
        }

        /* Monsters act after each player turn. */
        if (moved && game_running) {
            monsters_act();
            turns++;
        }
    }

    /* Show death screen if player died, otherwise just quit. */
    if (player.hp <= 0) {
        show_death();
        while (1) {
            int ch = getchar();
            if (ch != -1) break;
            sleep_ms(50);
        }
    } else {
        printf("\033[2J\033[H");
        printf("\nYou escaped the dungeon!\n");
        printf("Depth reached: %d\n", player.depth);
        printf("Gold collected: %d\n", player.gold);
        printf("Turns taken: %d\n", turns);
    }

    return 0;
}
