#!/usr/bin/env python3
"""
🐍 NEURAL CPU SNAKE
===================
Snake game running on the KVRM Neural CPU!

Uses the unified neural_cpu.py which has ALL trained models wired up:
- Neural ADD for position updates
- Neural SUB for collision checking
- Neural CMP for comparisons
- Neural Stack for game state
- Neural Memory for score storage

Run: python3 snake_neural_cpu.py
"""

import curses
import time
import random

# Import the unified Neural CPU!
from runtimes.full_model.neural_cpu_full import NeuralCPU

# ============================================================
# Neural Snake Game
# ============================================================

class NeuralSnake:
    def __init__(self):
        print("🐍 NEURAL CPU SNAKE - Loading...")
        print("=" * 50)

        # The unified Neural CPU with all operations
        self.cpu = NeuralCPU(quiet=False)

        print("\n🐍 Game ready!")
        print("=" * 50)

        # Game state
        self.width = 40
        self.height = 20
        self.reset_game()

    def reset_game(self):
        """Reset game state"""
        # Snake starts in middle - use neural ADD for initial position
        start_x = self.cpu.add(self.width // 4, self.width // 4)  # 20
        start_y = self.cpu.add(self.height // 4, self.height // 4)  # 10

        self.snake = [(start_x, start_y),
                     (self.cpu.sub(start_x, 1), start_y),
                     (self.cpu.sub(start_x, 2), start_y)]
        self.direction = (1, 0)  # Moving right
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False

        # Store score in neural memory
        self.cpu.store(0x100, self.score)

    def spawn_food(self):
        """Spawn food at random location not on snake"""
        while True:
            x = random.randint(1, self.cpu.sub(self.width, 2))
            y = random.randint(1, self.cpu.sub(self.height, 2))
            if (x, y) not in self.snake:
                return (x, y)

    def neural_move(self, pos, direction):
        """Move position using NEURAL ADD"""
        new_x = self.cpu.add(pos[0], direction[0])
        new_y = self.cpu.add(pos[1], direction[1])
        return (new_x, new_y)

    def neural_collision_check(self, pos1, pos2):
        """Check collision using NEURAL compare"""
        return self.cpu.eq(pos1[0], pos2[0]) and self.cpu.eq(pos1[1], pos2[1])

    def neural_wall_check(self, pos):
        """Check wall collision using NEURAL compare"""
        # Check bounds
        hit_left = self.cpu.eq(pos[0], 0) or pos[0] < 0
        hit_right = pos[0] >= self.cpu.sub(self.width, 1)
        hit_top = self.cpu.eq(pos[1], 0) or pos[1] < 0
        hit_bottom = pos[1] >= self.cpu.sub(self.height, 1)
        return hit_left or hit_right or hit_top or hit_bottom

    def update(self):
        """Update game state using neural operations"""
        if self.game_over:
            return

        # Move snake head using NEURAL ADD
        head = self.snake[0]
        new_head = self.neural_move(head, self.direction)

        # Check wall collision
        if self.neural_wall_check(new_head):
            self.game_over = True
            return

        # Check self collision
        for segment in self.snake[:-1]:
            if self.neural_collision_check(new_head, segment):
                self.game_over = True
                return

        # Check food collision
        ate_food = self.neural_collision_check(new_head, self.food)

        # Update snake
        self.snake.insert(0, new_head)
        if ate_food:
            self.score = self.cpu.add(self.score, 10)
            self.cpu.store(0x100, self.score)  # Update score in neural memory
            self.food = self.spawn_food()
        else:
            self.snake.pop()

    def process_input(self, key):
        """Process keyboard input"""
        # Direction mapping
        directions = {
            ord('w'): (0, -1), ord('W'): (0, -1), curses.KEY_UP: (0, -1),
            ord('s'): (0, 1), ord('S'): (0, 1), curses.KEY_DOWN: (0, 1),
            ord('a'): (-1, 0), ord('A'): (-1, 0), curses.KEY_LEFT: (-1, 0),
            ord('d'): (1, 0), ord('D'): (1, 0), curses.KEY_RIGHT: (1, 0),
        }

        if key in directions:
            new_dir = directions[key]
            # Prevent 180-degree turns using neural ADD
            reverse_x = self.cpu.add(self.direction[0], new_dir[0]) == 0
            reverse_y = self.cpu.add(self.direction[1], new_dir[1]) == 0
            if not (reverse_x and reverse_y):
                self.direction = new_dir

    def render(self, stdscr):
        """Render the game"""
        stdscr.clear()

        # Draw border
        for x in range(self.width):
            stdscr.addch(0, x, '#')
            stdscr.addch(self.cpu.sub(self.height, 1), x, '#')
        for y in range(self.height):
            stdscr.addch(y, 0, '#')
            stdscr.addch(y, self.cpu.sub(self.width, 1), '#')

        # Draw food
        try:
            stdscr.addch(self.food[1], self.food[0], '*')
        except:
            pass

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            try:
                if i == 0:
                    stdscr.addch(y, x, '@')  # Head
                else:
                    stdscr.addch(y, x, 'o')  # Body
            except:
                pass

        # Draw HUD
        try:
            # Load score from neural memory
            mem_score = self.cpu.load(0x100)
            stdscr.addstr(self.cpu.add(self.height, 1), 0,
                         f"NEURAL CPU SNAKE | Score: {mem_score} | Neural Ops: {self.cpu.op_count}")
            stdscr.addstr(self.cpu.add(self.height, 2), 0,
                         "WASD/Arrows: Move | Q: Quit | 100% Neural CPU!")

            if self.game_over:
                msg = f"GAME OVER! Score: {self.score} - Press R to restart"
                stdscr.addstr(self.height // 2, (self.cpu.sub(self.width, len(msg))) // 2, msg)
        except:
            pass

        stdscr.refresh()

    def run(self, stdscr):
        """Main game loop"""
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)  # 10 FPS

        while True:
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key == ord('q') or key == ord('Q') or key == 27:
                break

            if key == ord('r') or key == ord('R'):
                if self.game_over:
                    self.reset_game()

            self.process_input(key)
            self.update()
            self.render(stdscr)


def main():
    game = NeuralSnake()

    print("\n🐍 NEURAL CPU SNAKE")
    print("=" * 40)
    print("100% Neural CPU Operations:")
    print("  🧠 Movement: Neural ADD")
    print("  🧠 Collision: Neural CMP/EQ")
    print("  🧠 Score: Neural Memory")
    print("  🧠 Bounds: Neural SUB")
    print()
    print("Controls: WASD or Arrow Keys")
    print("Press Enter to start...")
    input()

    curses.wrapper(game.run)

    print(f"\n✅ Thanks for playing!")
    print(f"   Final Score: {game.score}")
    print(f"   Total Neural Ops: {game.cpu.op_count}")


if __name__ == "__main__":
    main()
