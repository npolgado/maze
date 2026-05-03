"""
Pygame Braid Maze Game
----------------------
GUI version of maze.py using pygame.

Controls:
  - Move: WASD or arrow keys
  - Zoom: Mouse wheel, +/- keys
  - New maze: N
  - Reset player on current maze: R
  - Quit: Q or Esc
"""

from __future__ import annotations

import argparse
from typing import Optional

import pygame

from maze import DIRS, Maze


MOVE_KEYS = {
    pygame.K_w: "N",
    pygame.K_UP: "N",
    pygame.K_s: "S",
    pygame.K_DOWN: "S",
    pygame.K_d: "E",
    pygame.K_RIGHT: "E",
    pygame.K_a: "W",
    pygame.K_LEFT: "W",
}


COLORS = {
    "bg": (16, 18, 24),
    "maze_bg": (28, 32, 42),
    "wall": (235, 240, 255),
    "visited": (66, 88, 120),
    "start": (75, 130, 255),
    "finish": (62, 192, 120),
    "player": (255, 215, 90),
    "hud_text": (235, 240, 255),
    "hud_win": (120, 235, 160),
}


class MazeGame:
    def __init__(
        self,
        rows: int,
        cols: int,
        seed: Optional[int],
        max_rect: int,
        width: int,
        height: int,
    ):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.max_rect = max_rect

        pygame.init()
        pygame.display.set_caption("Braid Maze - Pygame")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.small_font = pygame.font.SysFont("consolas", 16)

        self.cell_size = 8
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.follow_player = True
        self.moves = 0
        self.won = False

        self.maze_surface: Optional[pygame.Surface] = None
        self.maze_surface_cell_size = -1

        self._new_maze(self.seed)
        self.cell_size = self._fit_cell_size()
        self._invalidate_surface()
        self._center_on_player(force=True)

    def _new_maze(self, seed: Optional[int]):
        self.maze = Maze(self.rows, self.cols, seed, max_rect=self.max_rect)
        self.start = self.maze.start
        self.finish = self.maze.finish
        self.player = self.start
        self.visited = {self.start}
        self.moves = 0
        self.won = False
        self._invalidate_surface()

    def _fit_cell_size(self) -> int:
        w, h = self.screen.get_size()
        usable_w = max(200, w - 40)
        usable_h = max(200, h - 120)
        size = int(min(usable_w / self.cols, usable_h / self.rows))
        return max(4, min(24, size))

    def _world_size(self) -> tuple[int, int]:
        return self.cols * self.cell_size + 1, self.rows * self.cell_size + 1

    def _invalidate_surface(self):
        self.maze_surface = None
        self.maze_surface_cell_size = -1

    def _ensure_maze_surface(self):
        if self.maze_surface is not None and self.maze_surface_cell_size == self.cell_size:
            return

        world_w, world_h = self._world_size()
        surface = pygame.Surface((world_w, world_h))
        surface.fill(COLORS["maze_bg"])

        cs = self.cell_size

        # Start and finish markers.
        sx, sy = self.start[1] * cs, self.start[0] * cs
        fx, fy = self.finish[1] * cs, self.finish[0] * cs
        pygame.draw.rect(surface, COLORS["start"], (sx + 1, sy + 1, cs - 1, cs - 1))
        pygame.draw.rect(surface, COLORS["finish"], (fx + 1, fy + 1, cs - 1, cs - 1))

        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cs
                y = r * cs
                passages = self.maze.grid[r][c].passages
                if "N" not in passages:
                    pygame.draw.line(surface, COLORS["wall"], (x, y), (x + cs, y), 1)
                if "W" not in passages:
                    pygame.draw.line(surface, COLORS["wall"], (x, y), (x, y + cs), 1)
                if r == self.rows - 1 and "S" not in passages:
                    pygame.draw.line(surface, COLORS["wall"], (x, y + cs), (x + cs, y + cs), 1)
                if c == self.cols - 1 and "E" not in passages:
                    pygame.draw.line(surface, COLORS["wall"], (x + cs, y), (x + cs, y + cs), 1)

        self.maze_surface = surface
        self.maze_surface_cell_size = self.cell_size

    def _center_on_player(self, force: bool = False):
        if not self.follow_player and not force:
            return

        screen_w, screen_h = self.screen.get_size()
        world_w, world_h = self._world_size()
        px = self.player[1] * self.cell_size + self.cell_size / 2
        py = self.player[0] * self.cell_size + self.cell_size / 2

        self.camera_x = px - screen_w / 2
        self.camera_y = py - screen_h / 2
        self._clamp_camera()

    def _clamp_camera(self):
        world_w, world_h = self._world_size()
        screen_w, screen_h = self.screen.get_size()
        self.camera_x = max(0, min(self.camera_x, max(0, world_w - screen_w)))
        self.camera_y = max(0, min(self.camera_y, max(0, world_h - screen_h)))

    def _try_move(self, d: str):
        if self.won:
            return

        r, c = self.player
        if d not in self.maze.grid[r][c].passages:
            return

        dr, dc = DIRS[d]
        self.player = (r + dr, c + dc)
        self.visited.add(self.player)
        self.moves += 1
        self._center_on_player()

        if self.player == self.finish:
            self.won = True

    def _zoom(self, direction: int):
        old_size = self.cell_size
        self.cell_size = max(3, min(40, self.cell_size + direction))
        if self.cell_size == old_size:
            return

        # Keep player near the same viewport spot after zoom.
        pr, pc = self.player
        old_px = pc * old_size + old_size / 2
        old_py = pr * old_size + old_size / 2
        rel_x = old_px - self.camera_x
        rel_y = old_py - self.camera_y

        new_px = pc * self.cell_size + self.cell_size / 2
        new_py = pr * self.cell_size + self.cell_size / 2
        self.camera_x = new_px - rel_x
        self.camera_y = new_py - rel_y

        self._invalidate_surface()
        self._clamp_camera()

    def _reset_player(self):
        self.player = self.start
        self.visited = {self.start}
        self.moves = 0
        self.won = False
        self._center_on_player(force=True)

    def _draw(self):
        self._ensure_maze_surface()
        assert self.maze_surface is not None

        self.screen.fill(COLORS["bg"])
        offset_x = int(-self.camera_x)
        offset_y = int(-self.camera_y)

        self.screen.blit(self.maze_surface, (offset_x, offset_y))

        # Visited overlay.
        cs = self.cell_size
        for r, c in self.visited:
            x = c * cs + offset_x + 1
            y = r * cs + offset_y + 1
            if x + cs < 0 or y + cs < 0:
                continue
            if x > self.screen.get_width() or y > self.screen.get_height():
                continue
            pygame.draw.rect(self.screen, COLORS["visited"], (x, y, max(1, cs - 1), max(1, cs - 1)))

        # Restore start/finish on top of visited.
        sx, sy = self.start[1] * cs + offset_x, self.start[0] * cs + offset_y
        fx, fy = self.finish[1] * cs + offset_x, self.finish[0] * cs + offset_y
        pygame.draw.rect(self.screen, COLORS["start"], (sx + 1, sy + 1, max(1, cs - 1), max(1, cs - 1)))
        pygame.draw.rect(self.screen, COLORS["finish"], (fx + 1, fy + 1, max(1, cs - 1), max(1, cs - 1)))

        # Player marker.
        pr, pc = self.player
        px = pc * cs + offset_x + cs // 2
        py = pr * cs + offset_y + cs // 2
        pygame.draw.circle(self.screen, COLORS["player"], (px, py), max(3, cs // 3))

        # HUD
        hud_text = (
            f"Moves: {self.moves}   Size: {self.rows}x{self.cols}   "
            f"Zoom: {self.cell_size}px   Controls: WASD/Arrows, Wheel +/- zoom, N new, R reset, Q quit"
        )
        hud = self.small_font.render(hud_text, True, COLORS["hud_text"])
        self.screen.blit(hud, (12, 10))

        if self.won:
            win_text = self.font.render(
                f"You reached the finish in {self.moves} moves!  Press N for a new maze.",
                True,
                COLORS["hud_win"],
            )
            self.screen.blit(win_text, (12, 34))
        else:
            title = self.font.render("BRAID MAZE", True, COLORS["hud_text"])
            self.screen.blit(title, (12, 34))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self._clamp_camera()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        break
                    if event.key in MOVE_KEYS:
                        self._try_move(MOVE_KEYS[event.key])
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self._zoom(+1)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self._zoom(-1)
                    elif event.key == pygame.K_n:
                        self._new_maze(None)
                        self._center_on_player(force=True)
                    elif event.key == pygame.K_r:
                        self._reset_player()
                    elif event.key == pygame.K_f:
                        self.cell_size = self._fit_cell_size()
                        self._invalidate_surface()
                        self._center_on_player(force=True)

                if event.type == pygame.MOUSEWHEEL:
                    self._zoom(event.y)

            self._draw()
            self.clock.tick(60)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Braid Maze GUI (pygame)")
    parser.add_argument("--rows", type=int, default=100, help="Maze height (default 100)")
    parser.add_argument("--cols", type=int, default=100, help="Maze width (default 100)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible mazes")
    parser.add_argument(
        "--max-rect",
        "--max_rect",
        type=int,
        default=3,
        help="Suppress rectangular loops with both dims <= this (default 5)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Window width (default 1280)")
    parser.add_argument("--height", type=int, default=820, help="Window height (default 820)")
    args = parser.parse_args()

    game = MazeGame(
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
        max_rect=args.max_rect,
        width=args.width,
        height=args.height,
    )
    game.run()


if __name__ == "__main__":
    main()
