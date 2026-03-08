/*
 * adventure.c -- Text adventure game for freestanding ARM64 on Metal GPU.
 *
 * 8 rooms in a dungeon/castle. 5 collectible items. Two-word command parser.
 * Puzzles: torch for dungeon, key for throne room, sword for dragon.
 * Win condition: bring the gem back to the Entrance Hall.
 *
 * Freestanding: no stdlib. All I/O via arm64_libc.h SVC syscalls.
 *
 * Compile:
 *   aarch64-elf-gcc -nostdlib -ffreestanding -static -O2 \
 *       -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld \
 *       -I demos -e _start demos/arm64_start.S demos/games/adventure.c \
 *       -o /tmp/adventure.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONSTANTS                                                                */
/* ======================================================================== */

#define NUM_ROOMS       8
#define NUM_ITEMS       5
#define MAX_INVENTORY   10
#define INPUT_BUF_SIZE  128
#define MAX_ROOM_ITEMS  5

/* Room indices */
#define ROOM_ENTRANCE       0
#define ROOM_LIBRARY        1
#define ROOM_ARMORY         2
#define ROOM_KITCHEN        3
#define ROOM_DUNGEON        4
#define ROOM_TOWER          5
#define ROOM_THRONE         6
#define ROOM_SECRET         7

/* Item indices */
#define ITEM_KEY        0
#define ITEM_SWORD      1
#define ITEM_TORCH      2
#define ITEM_GEM        3
#define ITEM_SCROLL     4

/* Direction indices for exits */
#define DIR_NORTH   0
#define DIR_SOUTH   1
#define DIR_EAST    2
#define DIR_WEST    3

/* Special sentinel for no exit */
#define NO_EXIT     -1

/* Game state flags */
#define FLAG_DRAGON_DEAD    1
#define FLAG_THRONE_OPEN    2

/* ======================================================================== */
/* DATA STRUCTURES                                                          */
/* ======================================================================== */

typedef struct {
    const char *name;
    const char *description;
    const char *dark_description;       /* shown if room is dark and no torch */
    int exits[4];                       /* n, s, e, w — room index or NO_EXIT */
    int items[MAX_ROOM_ITEMS];          /* item indices present, -1 = empty */
    int is_dark;                        /* requires torch to see */
    int needs_key;                      /* requires key to enter */
    int has_dragon;                     /* dragon blocks gem until defeated */
} room_t;

typedef struct {
    const char *name;
    const char *description;
    const char *use_description;        /* what happens when used */
} item_t;

/* ======================================================================== */
/* GAME DATA                                                                */
/* ======================================================================== */

static item_t items[NUM_ITEMS] = {
    { "key",    "A heavy iron key, cold to the touch.",
                "The key turns in the lock with a grinding click." },
    { "sword",  "A gleaming steel sword with runes along the blade.",
                "You swing the sword in a deadly arc!" },
    { "torch",  "A wooden torch wrapped in oil-soaked cloth, burning brightly.",
                "The torch flares, casting warm light into the darkness." },
    { "gem",    "A brilliant ruby gem that pulses with inner fire.",
                "The gem glows warmly in your hands." },
    { "scroll", "An ancient parchment covered in faded writing.",
                "The scroll reads: 'The gem of power rests beyond the dragon's keep.\n"
                "Only steel can pierce its scales. Return the gem whence you began.'" },
};

static room_t rooms[NUM_ROOMS] = {
    /* ROOM_ENTRANCE (0) */
    {
        "Entrance Hall",
        "You stand in a grand entrance hall. Faded tapestries hang from stone\n"
        "walls, depicting battles long forgotten. A cold draft sweeps through\n"
        "the iron-banded doors behind you. Corridors lead in several directions.",
        NULL,
        { ROOM_LIBRARY, NO_EXIT, ROOM_ARMORY, ROOM_KITCHEN },
        { ITEM_TORCH, -1, -1, -1, -1 },
        0, 0, 0
    },
    /* ROOM_LIBRARY (1) */
    {
        "Library",
        "Towering bookshelves line every wall, their shelves sagging under\n"
        "the weight of countless leather-bound volumes. Dust motes drift\n"
        "through beams of light from a narrow window. A reading desk sits\n"
        "in the center, covered in papers.",
        NULL,
        { NO_EXIT, ROOM_ENTRANCE, ROOM_THRONE, ROOM_SECRET },
        { ITEM_SCROLL, -1, -1, -1, -1 },
        0, 0, 0
    },
    /* ROOM_ARMORY (2) */
    {
        "Armory",
        "Weapon racks line the walls, though most are empty. Rusted suits\n"
        "of armor stand like silent sentinels. The air smells of oil and\n"
        "old iron. A few weapons still gleam amid the decay.",
        NULL,
        { NO_EXIT, NO_EXIT, NO_EXIT, ROOM_ENTRANCE },
        { ITEM_SWORD, -1, -1, -1, -1 },
        0, 0, 0
    },
    /* ROOM_KITCHEN (3) */
    {
        "Kitchen",
        "A cavernous kitchen with a massive stone hearth dominates the room.\n"
        "Copper pots hang from iron hooks overhead. The remains of a meal\n"
        "sit on a scarred wooden table. A trapdoor in the floor leads down.",
        NULL,
        { NO_EXIT, ROOM_DUNGEON, ROOM_ENTRANCE, NO_EXIT },
        { ITEM_KEY, -1, -1, -1, -1 },
        0, 0, 0
    },
    /* ROOM_DUNGEON (4) */
    {
        "Dungeon",
        "The dungeon stretches into shadow. Iron cages line the walls,\n"
        "some still holding ancient bones. Water drips from the ceiling\n"
        "into stagnant pools. The air is thick with the smell of damp stone.\n"
        "A narrow staircase spirals upward.",
        "It is pitch black. You can't see anything. You need a light source\n"
        "to explore this place safely.",
        { ROOM_KITCHEN, NO_EXIT, ROOM_SECRET, NO_EXIT },
        { -1, -1, -1, -1, -1 },
        1, 0, 0
    },
    /* ROOM_TOWER (5) */
    {
        "Tower",
        "You emerge at the top of a crumbling tower. Wind howls through gaps\n"
        "in the stonework. Far below, mist-covered lands stretch to the horizon.\n"
        "A massive dragon coils in the center, its scales like hammered bronze,\n"
        "guarding a pedestal upon which something glitters.",
        NULL,
        { NO_EXIT, ROOM_SECRET, NO_EXIT, NO_EXIT },
        { ITEM_GEM, -1, -1, -1, -1 },
        0, 0, 1
    },
    /* ROOM_THRONE (6) */
    {
        "Throne Room",
        "A vast throne room opens before you. Columns of dark marble support\n"
        "a vaulted ceiling painted with stars. At the far end, an obsidian\n"
        "throne sits upon a raised dais. The silence is absolute and heavy.",
        NULL,
        { NO_EXIT, NO_EXIT, NO_EXIT, ROOM_LIBRARY },
        { -1, -1, -1, -1, -1 },
        0, 1, 0
    },
    /* ROOM_SECRET (7) */
    {
        "Secret Passage",
        "A narrow passage winds between the walls. The stones here are older\n"
        "than the rest of the castle, covered in strange carvings that seem\n"
        "to shift in the torchlight. It connects hidden corners of the keep.",
        NULL,
        { ROOM_TOWER, NO_EXIT, ROOM_LIBRARY, ROOM_DUNGEON },
        { -1, -1, -1, -1, -1 },
        0, 0, 0
    },
};

/* ======================================================================== */
/* PLAYER STATE                                                             */
/* ======================================================================== */

static int player_room;
static int inventory[MAX_INVENTORY];
static int inventory_count;
static int game_flags;
static int game_over;
static int move_count;

/* ======================================================================== */
/* INPUT BUFFER                                                             */
/* ======================================================================== */

static char input_buf[INPUT_BUF_SIZE];

/* ======================================================================== */
/* UTILITY FUNCTIONS                                                        */
/* ======================================================================== */

static void print_line(void) {
    printf("----------------------------------------\n");
}

static int has_item(int item_id) {
    for (int i = 0; i < inventory_count; i++) {
        if (inventory[i] == item_id) return 1;
    }
    return 0;
}

static int room_has_item(int room_id, int item_id) {
    for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
        if (rooms[room_id].items[i] == item_id) return 1;
    }
    return 0;
}

static void add_item_to_inventory(int item_id) {
    if (inventory_count >= MAX_INVENTORY) {
        printf("Your inventory is full.\n");
        return;
    }
    inventory[inventory_count++] = item_id;
}

static void remove_item_from_inventory(int item_id) {
    for (int i = 0; i < inventory_count; i++) {
        if (inventory[i] == item_id) {
            /* Shift remaining items down */
            for (int j = i; j < inventory_count - 1; j++) {
                inventory[j] = inventory[j + 1];
            }
            inventory_count--;
            return;
        }
    }
}

static void add_item_to_room(int room_id, int item_id) {
    for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
        if (rooms[room_id].items[i] == -1) {
            rooms[room_id].items[i] = item_id;
            return;
        }
    }
}

static void remove_item_from_room(int room_id, int item_id) {
    for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
        if (rooms[room_id].items[i] == item_id) {
            rooms[room_id].items[i] = -1;
            return;
        }
    }
}

/* Convert a character to lowercase */
static char to_lower(char c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    return c;
}

/* Convert string to lowercase in-place */
static void str_to_lower(char *s) {
    while (*s) {
        *s = to_lower(*s);
        s++;
    }
}

/* ======================================================================== */
/* DISPLAY                                                                  */
/* ======================================================================== */

static const char *dir_names[4] = { "north", "south", "east", "west" };

static void describe_room(void) {
    room_t *r = &rooms[player_room];

    print_line();
    printf("  %s\n", r->name);
    print_line();

    /* Check if room is dark and player has no torch */
    if (r->is_dark && !has_item(ITEM_TORCH)) {
        printf("%s\n", r->dark_description);
    } else {
        printf("%s\n", r->description);

        /* Special: dragon alive in tower */
        if (r->has_dragon && !(game_flags & FLAG_DRAGON_DEAD)) {
            printf("\nA fearsome DRAGON blocks your path, smoke curling from\n"
                   "its nostrils. It watches you with ancient, calculating eyes.\n");
        }

        /* Special: dragon is dead */
        if (r->has_dragon && (game_flags & FLAG_DRAGON_DEAD)) {
            printf("\nThe dragon lies slain, its bronze scales dulled in death.\n"
                   "The pedestal beyond it is now accessible.\n");
        }

        /* List items on the ground */
        int found_items = 0;
        for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
            int item_id = r->items[i];
            if (item_id >= 0) {
                /* Don't show gem if dragon is alive and blocking */
                if (item_id == ITEM_GEM && r->has_dragon &&
                    !(game_flags & FLAG_DRAGON_DEAD)) {
                    continue;
                }
                if (!found_items) {
                    printf("\nYou see:\n");
                    found_items = 1;
                }
                printf("  - %s\n", items[item_id].name);
            }
        }
    }

    /* Show exits */
    printf("\nExits:");
    int any_exit = 0;
    for (int d = 0; d < 4; d++) {
        if (r->exits[d] != NO_EXIT) {
            if (r->exits[d] == ROOM_THRONE && !(game_flags & FLAG_THRONE_OPEN)) {
                printf(" %s(locked)", dir_names[d]);
            } else {
                printf(" %s", dir_names[d]);
            }
            any_exit = 1;
        }
    }
    if (!any_exit) printf(" none");
    printf("\n");
}

/* ======================================================================== */
/* COMMAND PARSER                                                           */
/* ======================================================================== */

static char verb[32];
static char noun[32];

static void parse_command(const char *input) {
    verb[0] = '\0';
    noun[0] = '\0';

    /* Skip leading whitespace */
    while (*input == ' ' || *input == '\t') input++;
    if (*input == '\0') return;

    /* Extract verb */
    int i = 0;
    while (*input && *input != ' ' && *input != '\t' && i < 30) {
        verb[i++] = *input++;
    }
    verb[i] = '\0';

    /* Skip whitespace between verb and noun */
    while (*input == ' ' || *input == '\t') input++;

    /* Extract noun */
    i = 0;
    while (*input && *input != ' ' && *input != '\t' && *input != '\n' && i < 30) {
        noun[i++] = *input++;
    }
    noun[i] = '\0';

    /* Normalize to lowercase */
    str_to_lower(verb);
    str_to_lower(noun);
}

/* ======================================================================== */
/* COMMAND: GO                                                              */
/* ======================================================================== */

static int direction_from_noun(const char *n) {
    if (strcmp(n, "north") == 0 || strcmp(n, "n") == 0) return DIR_NORTH;
    if (strcmp(n, "south") == 0 || strcmp(n, "s") == 0) return DIR_SOUTH;
    if (strcmp(n, "east")  == 0 || strcmp(n, "e") == 0) return DIR_EAST;
    if (strcmp(n, "west")  == 0 || strcmp(n, "w") == 0) return DIR_WEST;
    return -1;
}

static void cmd_go(void) {
    if (noun[0] == '\0') {
        printf("Go where? Specify a direction: north, south, east, west.\n");
        return;
    }

    int dir = direction_from_noun(noun);
    if (dir < 0) {
        printf("That's not a direction. Try north, south, east, or west.\n");
        return;
    }

    int dest = rooms[player_room].exits[dir];

    if (dest == NO_EXIT) {
        printf("You can't go %s from here. There is no passage.\n", dir_names[dir]);
        return;
    }

    /* Check if destination needs a key (Throne Room) */
    if (rooms[dest].needs_key && !(game_flags & FLAG_THRONE_OPEN)) {
        if (has_item(ITEM_KEY)) {
            printf("You insert the iron key into the lock. %s\n",
                   items[ITEM_KEY].use_description);
            printf("The heavy door swings open with a groan.\n\n");
            game_flags |= FLAG_THRONE_OPEN;
            /* Key is consumed */
            remove_item_from_inventory(ITEM_KEY);
        } else {
            printf("The door to the %s is locked. You need a key.\n",
                   rooms[dest].name);
            return;
        }
    }

    player_room = dest;
    move_count++;
    describe_room();

    /* Win condition check */
    if (player_room == ROOM_ENTRANCE && has_item(ITEM_GEM)) {
        printf("\n");
        print_line();
        printf("  *** VICTORY! ***\n");
        print_line();
        printf("\nAs you step into the Entrance Hall, the gem blazes with\n"
               "brilliant light. The ancient magic of the castle recognizes\n"
               "its rightful bearer. The iron doors swing wide, and golden\n"
               "sunlight floods the hall for the first time in centuries.\n\n"
               "You have escaped the castle with the gem of power!\n\n"
               "Moves: %d\n", move_count);
        print_line();
        game_over = 1;
    }
}

/* ======================================================================== */
/* COMMAND: LOOK                                                            */
/* ======================================================================== */

static void cmd_look(void) {
    if (noun[0] != '\0') {
        /* Look at a specific item */
        /* Check inventory first */
        for (int i = 0; i < inventory_count; i++) {
            if (strcmp(items[inventory[i]].name, noun) == 0) {
                printf("%s\n", items[inventory[i]].description);
                return;
            }
        }
        /* Check room */
        room_t *r = &rooms[player_room];
        if (!(r->is_dark && !has_item(ITEM_TORCH))) {
            for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
                int item_id = r->items[i];
                if (item_id >= 0 && strcmp(items[item_id].name, noun) == 0) {
                    printf("%s\n", items[item_id].description);
                    return;
                }
            }
        }
        printf("You don't see '%s' here.\n", noun);
    } else {
        describe_room();
    }
}

/* ======================================================================== */
/* COMMAND: TAKE / GET                                                      */
/* ======================================================================== */

static void cmd_take(void) {
    if (noun[0] == '\0') {
        printf("Take what?\n");
        return;
    }

    room_t *r = &rooms[player_room];

    /* Can't take items in dark rooms without torch */
    if (r->is_dark && !has_item(ITEM_TORCH)) {
        printf("It's too dark to find anything here.\n");
        return;
    }

    /* Special: gem blocked by dragon */
    if (strcmp(noun, "gem") == 0 && r->has_dragon &&
        !(game_flags & FLAG_DRAGON_DEAD)) {
        printf("The dragon snarls and snaps at you! You can't reach the gem\n"
               "while the beast still lives.\n");
        return;
    }

    /* Find item in room */
    for (int i = 0; i < MAX_ROOM_ITEMS; i++) {
        int item_id = r->items[i];
        if (item_id >= 0 && strcmp(items[item_id].name, noun) == 0) {
            if (inventory_count >= MAX_INVENTORY) {
                printf("Your inventory is full. Drop something first.\n");
                return;
            }
            remove_item_from_room(player_room, item_id);
            add_item_to_inventory(item_id);
            printf("You pick up the %s.\n", items[item_id].name);
            return;
        }
    }

    printf("There is no '%s' here to take.\n", noun);
}

/* ======================================================================== */
/* COMMAND: DROP                                                            */
/* ======================================================================== */

static void cmd_drop(void) {
    if (noun[0] == '\0') {
        printf("Drop what?\n");
        return;
    }

    for (int i = 0; i < inventory_count; i++) {
        if (strcmp(items[inventory[i]].name, noun) == 0) {
            int item_id = inventory[i];
            remove_item_from_inventory(item_id);
            add_item_to_room(player_room, item_id);
            printf("You drop the %s.\n", items[item_id].name);
            return;
        }
    }

    printf("You don't have '%s'.\n", noun);
}

/* ======================================================================== */
/* COMMAND: USE                                                             */
/* ======================================================================== */

static void cmd_use(void) {
    if (noun[0] == '\0') {
        printf("Use what?\n");
        return;
    }

    /* Must have the item */
    if (!has_item(ITEM_TORCH) && strcmp(noun, "torch") == 0) {
        printf("You don't have a torch.\n");
        return;
    }
    if (!has_item(ITEM_SWORD) && strcmp(noun, "sword") == 0) {
        printf("You don't have a sword.\n");
        return;
    }
    if (!has_item(ITEM_KEY) && strcmp(noun, "key") == 0) {
        printf("You don't have a key.\n");
        return;
    }

    /* Torch */
    if (strcmp(noun, "torch") == 0) {
        if (has_item(ITEM_TORCH)) {
            printf("%s\n", items[ITEM_TORCH].use_description);
            if (rooms[player_room].is_dark) {
                printf("The darkness recedes. You can now see your surroundings.\n\n");
                describe_room();
            }
            return;
        }
    }

    /* Sword — only meaningful against the dragon */
    if (strcmp(noun, "sword") == 0) {
        if (has_item(ITEM_SWORD)) {
            if (player_room == ROOM_TOWER && rooms[ROOM_TOWER].has_dragon &&
                !(game_flags & FLAG_DRAGON_DEAD)) {
                printf("%s\n\n", items[ITEM_SWORD].use_description);
                printf("The blade bites deep between the dragon's scales! With a\n"
                       "thunderous roar, the great beast thrashes once and falls\n"
                       "still. Smoke drifts from its nostrils one final time.\n\n"
                       "The dragon is slain!\n\n");
                game_flags |= FLAG_DRAGON_DEAD;
                /* Show what's now accessible */
                printf("On the pedestal behind the fallen dragon, a gem glitters.\n");
            } else {
                printf("You swing the sword, but there is nothing here to fight.\n");
            }
            return;
        }
    }

    /* Key — only meaningful at the Throne Room door */
    if (strcmp(noun, "key") == 0) {
        if (has_item(ITEM_KEY)) {
            printf("There is no lock here to use the key on.\n"
                   "Try going toward a locked door.\n");
            return;
        }
    }

    /* Scroll */
    if (strcmp(noun, "scroll") == 0) {
        if (has_item(ITEM_SCROLL)) {
            printf("%s\n", items[ITEM_SCROLL].use_description);
            return;
        }
        printf("You don't have a scroll.\n");
        return;
    }

    /* Gem */
    if (strcmp(noun, "gem") == 0) {
        if (has_item(ITEM_GEM)) {
            printf("%s\n", items[ITEM_GEM].use_description);
            return;
        }
        printf("You don't have the gem.\n");
        return;
    }

    printf("You can't figure out how to use '%s' here.\n", noun);
}

/* ======================================================================== */
/* COMMAND: INVENTORY                                                       */
/* ======================================================================== */

static void cmd_inventory(void) {
    if (inventory_count == 0) {
        printf("You are carrying nothing.\n");
        return;
    }
    printf("You are carrying:\n");
    for (int i = 0; i < inventory_count; i++) {
        printf("  - %s\n", items[inventory[i]].name);
    }
}

/* ======================================================================== */
/* COMMAND: HELP                                                            */
/* ======================================================================== */

static void cmd_help(void) {
    printf("\nAvailable commands:\n");
    printf("  go <direction>   Move north/south/east/west (or n/s/e/w)\n");
    printf("  look [item]      Examine surroundings or a specific item\n");
    printf("  take <item>      Pick up an item (also: get)\n");
    printf("  drop <item>      Drop an item from inventory\n");
    printf("  use <item>       Use an item\n");
    printf("  inventory        Show what you carry (also: i)\n");
    printf("  help             Show this help\n");
    printf("  quit             End the game\n\n");
    printf("Directions can also be entered alone: north, n, south, s, etc.\n\n");
}

/* ======================================================================== */
/* COMMAND DISPATCH                                                         */
/* ======================================================================== */

static void process_command(void) {
    if (verb[0] == '\0') return;

    /* Movement verbs */
    if (strcmp(verb, "go") == 0 || strcmp(verb, "move") == 0) {
        cmd_go();
        return;
    }

    /* Bare direction as verb */
    if (strcmp(verb, "north") == 0 || strcmp(verb, "n") == 0 ||
        strcmp(verb, "south") == 0 || strcmp(verb, "s") == 0 ||
        strcmp(verb, "east")  == 0 || strcmp(verb, "e") == 0 ||
        strcmp(verb, "west")  == 0 || strcmp(verb, "w") == 0) {
        /* Treat verb as noun for the go command */
        strcpy(noun, verb);
        cmd_go();
        return;
    }

    /* Look */
    if (strcmp(verb, "look") == 0 || strcmp(verb, "l") == 0 ||
        strcmp(verb, "examine") == 0) {
        cmd_look();
        return;
    }

    /* Take / Get */
    if (strcmp(verb, "take") == 0 || strcmp(verb, "get") == 0 ||
        strcmp(verb, "grab") == 0 || strcmp(verb, "pick") == 0) {
        cmd_take();
        return;
    }

    /* Drop */
    if (strcmp(verb, "drop") == 0 || strcmp(verb, "put") == 0) {
        cmd_drop();
        return;
    }

    /* Use */
    if (strcmp(verb, "use") == 0) {
        cmd_use();
        return;
    }

    /* Inventory */
    if (strcmp(verb, "inventory") == 0 || strcmp(verb, "i") == 0 ||
        strcmp(verb, "inv") == 0) {
        cmd_inventory();
        return;
    }

    /* Help */
    if (strcmp(verb, "help") == 0 || strcmp(verb, "?") == 0) {
        cmd_help();
        return;
    }

    /* Quit */
    if (strcmp(verb, "quit") == 0 || strcmp(verb, "exit") == 0 ||
        strcmp(verb, "q") == 0) {
        printf("Thanks for playing! You made %d moves.\n", move_count);
        game_over = 1;
        return;
    }

    printf("I don't understand '%s'. Type 'help' for commands.\n", verb);
}

/* ======================================================================== */
/* GAME INITIALIZATION                                                      */
/* ======================================================================== */

static void game_init(void) {
    player_room = ROOM_ENTRANCE;
    inventory_count = 0;
    game_flags = 0;
    game_over = 0;
    move_count = 0;

    for (int i = 0; i < MAX_INVENTORY; i++) {
        inventory[i] = -1;
    }
}

/* ======================================================================== */
/* TITLE SCREEN                                                             */
/* ======================================================================== */

static void show_title(void) {
    printf("\n");
    print_line();
    printf("       CASTLE OF THE BRONZE DRAGON\n");
    printf("       A Text Adventure on Metal GPU\n");
    print_line();
    printf("\n");
    printf("Long ago, a dragon of bronze scales claimed an ancient castle\n"
           "and the gem of power within it. Many have entered these halls.\n"
           "None have returned.\n\n"
           "You stand before the castle gates, torch-smoke drifting on the\n"
           "wind. Somewhere inside, the gem awaits. Find it, and bring it\n"
           "back to the entrance to claim your victory.\n\n"
           "Type 'help' for a list of commands.\n\n");
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    game_init();
    show_title();
    describe_room();

    while (!game_over) {
        printf("\n> ");

        /* Read input */
        ssize_t n = sys_read(0, input_buf, INPUT_BUF_SIZE - 1);
        if (n <= 0) break;

        /* Null-terminate and strip newline */
        input_buf[n] = '\0';
        if (n > 0 && input_buf[n - 1] == '\n') {
            input_buf[n - 1] = '\0';
            n--;
        }
        if (n > 0 && input_buf[n - 1] == '\r') {
            input_buf[n - 1] = '\0';
            n--;
        }
        if (n == 0) continue;

        parse_command(input_buf);
        process_command();
    }

    return 0;
}
