"""
Braid Maze Game
---------------
Top-down 2D maze game. Navigate from S (start) to F (finish).
Controls: WASD or arrow keys. Q to quit.

Maze generation goals:
  1. No dead ends (every cell has at least two exits).
  2. Multiple loops and alternate routes.
  3. Avoid "thin" rectangular loops (2xN / Nx2).
  4. Prefer longer loops and break very long straight hallways.
"""

from collections import deque
import random
import sys
import os

# ── platform-specific key reading ────────────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def read_key():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):          # special key prefix on Windows
            ch2 = msvcrt.getwch()
            return {
                "H": "up", "P": "down", "K": "left", "M": "right",
            }.get(ch2, "")
        return ch.lower()
else:
    import tty
    import termios

    def read_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(ch3, "")
            return ch.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── maze data structures ──────────────────────────────────────────────────────

# Directions: index → (row_delta, col_delta)
DIRS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


class Cell:
    """One cell in the maze grid."""

    def __init__(self):
        # Set of direction strings this cell has open passages to
        self.passages: set[str] = set()


class Maze:
    def __init__(self, rows: int, cols: int, seed: int | None = None, max_rect: int = 5):
        self.rows = rows
        self.cols = cols
        self.max_rect = max_rect
        self.rng = random.Random(seed)
        self.grid: list[list[Cell]] = [[Cell() for _ in range(cols)] for _ in range(rows)]

        self.start = (0, 0)
        self.finish = (rows - 1, cols - 1)
        self.loop_gain_total = 0
        self.stats: dict[str, int] = {}

        self._generate()
        self.start, self.finish = self._pick_endpoints()
        self.stats = self._collect_generation_stats()

    # ── internal helpers ──────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _neighbors(self, r: int, c: int) -> list[tuple[str, int, int]]:
        """Return (direction, nr, nc) for all in-bounds neighbors."""
        result = []
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                result.append((d, nr, nc))
        return result

    def _open_passage(self, r: int, c: int, d: str) -> bool:
        """Carve a two-way passage between (r,c) and its neighbor in direction d."""
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if d in self.grid[r][c].passages:
            return False
        self.grid[r][c].passages.add(d)
        self.grid[nr][nc].passages.add(OPPOSITE[d])
        return True

    def _close_passage(self, r: int, c: int, d: str) -> bool:
        """Close a two-way passage between (r,c) and its neighbor in direction d."""
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        if d not in self.grid[r][c].passages:
            return False
        self.grid[r][c].passages.remove(d)
        self.grid[nr][nc].passages.remove(OPPOSITE[d])
        return True

    def _has_passage(self, r: int, c: int, d: str) -> bool:
        return d in self.grid[r][c].passages

    def _clear_grid(self):
        for row in self.grid:
            for cell in row:
                cell.passages.clear()

    def _snapshot(self) -> dict[str, object]:
        return {
            "grid": [[set(cell.passages) for cell in row] for row in self.grid],
            "loop_gain_total": self.loop_gain_total,
        }

    def _restore(self, snapshot: dict[str, object]):
        saved = snapshot["grid"]
        assert isinstance(saved, list)
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].passages = set(saved[r][c])
        self.loop_gain_total = int(snapshot["loop_gain_total"])

    # ── generation pipeline ───────────────────────────────────────────────

    def _generate(self):
        area = self.rows * self.cols
        if area <= 2500:
            attempts = 8
        elif area <= 6400:
            attempts = 4
        else:
            attempts = 2
        best_snapshot: dict[str, object] | None = None
        best_score: int | None = None

        for _ in range(attempts):
            self._clear_grid()
            self.loop_gain_total = 0

            self._generate_tree()
            self._eliminate_dead_ends(strict=True)
            if self._dead_ends():
                self._eliminate_dead_ends(strict=False)
            self._add_long_loops()
            self._break_long_hallways(limit=6)
            self._enforce_constraints()

            dead_count = len(self._dead_ends())
            thin_rects = 0 if self._find_one_thin_rectangle() is None else 1
            longest_hall = self._longest_straight_hallway()

            score = (
                dead_count * 10_000_000
                + thin_rects * 1_000_000
                + longest_hall * 100
                - self.loop_gain_total
            )
            if best_score is None or score < best_score:
                best_score = score
                best_snapshot = self._snapshot()

            if dead_count == 0 and thin_rects == 0:
                return

        if best_snapshot is not None:
            self._restore(best_snapshot)
            self._enforce_constraints()

    def _generate_tree(self):
        """Step 1: build a perfect maze via DFS backtracking."""
        visited = [[False] * self.cols for _ in range(self.rows)]
        visited[0][0] = True

        dirs0 = list(DIRS.keys())
        self.rng.shuffle(dirs0)
        stack = [(0, 0, dirs0, 0)]

        while stack:
            r, c, dirs, idx = stack[-1]
            if idx >= len(dirs):
                stack.pop()
                continue

            stack[-1] = (r, c, dirs, idx + 1)
            d = dirs[idx]
            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc

            if self._in_bounds(nr, nc) and not visited[nr][nc]:
                self._open_passage(r, c, d)
                visited[nr][nc] = True
                ndirs = list(DIRS.keys())
                self.rng.shuffle(ndirs)
                stack.append((nr, nc, ndirs, 0))

    def _dead_ends(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if len(self.grid[r][c].passages) == 1
        ]

    def _eliminate_dead_ends(self, strict: bool) -> bool:
        """Step 2: braid until no dead ends remain."""
        max_rounds = max(4, self.rows * self.cols * 4)

        for _ in range(max_rounds):
            dead = self._dead_ends()
            if not dead:
                return True

            self.rng.shuffle(dead)
            opened_any = False

            for r, c in dead:
                if len(self.grid[r][c].passages) != 1:
                    continue

                candidates: list[tuple[float, str, int]] = []
                for d, nr, nc in self._neighbors(r, c):
                    if d in self.grid[r][c].passages:
                        continue

                    forbidden = self._would_create_forbidden_rect_loop(r, c, d)
                    if forbidden and strict:
                        continue

                    cycle_len = self._cycle_length_if_open(r, c, d, cap=140)
                    score = cycle_len * 10 + self._hallway_branch_score(r, c, d)
                    if len(self.grid[nr][nc].passages) == 1:
                        score += 12
                    if forbidden:
                        score -= 80

                    candidates.append((score + self.rng.random(), d, cycle_len))

                if not candidates:
                    continue

                _, best_d, cycle_len = max(candidates, key=lambda x: x[0])
                if self._open_passage(r, c, best_d):
                    self.loop_gain_total += cycle_len
                    opened_any = True

            if not opened_any:
                return False

        return len(self._dead_ends()) == 0

    def _add_long_loops(self):
        """Add a limited number of extra edges that create long cycles."""
        target = max(1, (self.rows * self.cols) // 45)
        for _ in range(target):
            walls = self._wall_edges()
            if not walls:
                break

            sample = self.rng.sample(walls, min(80, len(walls)))
            best: tuple[float, int, int, str, int] | None = None

            for r, c, d in sample:
                if self._would_create_forbidden_rect_loop(r, c, d):
                    continue

                cycle_len = self._cycle_length_if_open(r, c, d, cap=160)
                if cycle_len < 7:
                    continue

                score = cycle_len * 10 + self._hallway_branch_score(r, c, d) + self.rng.random()
                if best is None or score > best[0]:
                    best = (score, r, c, d, cycle_len)

            if best is None:
                break

            _, r, c, d, cycle_len = best
            if self._open_passage(r, c, d):
                self.loop_gain_total += cycle_len

    def _break_long_hallways(self, limit: int = 6):
        """Add branch edges into long straight corridors to avoid monotony."""
        for _ in range(2):
            changed = False
            runs = self._straight_runs()
            self.rng.shuffle(runs)

            for orient, fixed, start, end in runs:
                length = end - start + 1
                if length <= limit:
                    continue

                center = (start + end) // 2
                indices = list(range(start, end + 1))
                indices.sort(key=lambda x: abs(x - center))

                opened = False
                for idx in indices:
                    r, c = (fixed, idx) if orient == "H" else (idx, fixed)
                    branch_dirs = ["N", "S"] if orient == "H" else ["E", "W"]
                    self.rng.shuffle(branch_dirs)

                    for d in branch_dirs:
                        dr, dc = DIRS[d]
                        nr, nc = r + dr, c + dc
                        if not self._in_bounds(nr, nc):
                            continue
                        if d in self.grid[r][c].passages:
                            continue
                        if self._would_create_forbidden_rect_loop(r, c, d):
                            continue

                        cycle_len = self._cycle_length_if_open(r, c, d, cap=160)
                        if cycle_len < 6:
                            continue

                        if self._open_passage(r, c, d):
                            self.loop_gain_total += cycle_len
                            changed = True
                            opened = True
                            break

                    if opened:
                        break

            if not changed:
                return

    def _enforce_constraints(self) -> bool:
        """Repair pass: remove dead ends and thin rectangles until stable."""
        max_rounds = max(8, (self.rows + self.cols) // 2)

        for _ in range(max_rounds):
            dead_before = len(self._dead_ends())
            thin_rect = self._find_one_thin_rectangle()
            if dead_before == 0 and thin_rect is None:
                return True

            progressed = False

            if dead_before > 0:
                self._eliminate_dead_ends(strict=True)
                if self._dead_ends():
                    self._eliminate_dead_ends(strict=False)
                if len(self._dead_ends()) < dead_before:
                    progressed = True

            if thin_rect is not None and self._break_one_thin_rectangle(thin_rect):
                progressed = True

            dead_after = len(self._dead_ends())
            thin_after = self._find_one_thin_rectangle()
            if dead_after == 0 and thin_after is None:
                return True

            if not progressed:
                break

        return len(self._dead_ends()) == 0 and self._find_one_thin_rectangle() is None

    # ── scoring and shape constraints ────────────────────────────────────

    def _shortest_path_len(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cap: int = 200,
        blocked_cell: tuple[int, int] | None = None,
    ) -> int | None:
        if start == goal:
            return 0
        if blocked_cell is not None and (start == blocked_cell or goal == blocked_cell):
            return None

        dq = deque([(start[0], start[1], 0)])
        seen = {start}

        while dq:
            r, c, dist = dq.popleft()
            if dist >= cap:
                continue

            for d in self.grid[r][c].passages:
                dr, dc = DIRS[d]
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nxt in seen or nxt == blocked_cell:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                dq.append((nr, nc, dist + 1))

        return None

    def _cycle_length_if_open(self, r: int, c: int, d: str, cap: int = 200) -> int:
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        dist = self._shortest_path_len((r, c), (nr, nc), cap=cap)
        if dist is None:
            return cap + 1
        return dist + 1

    def _straight_corridor_info(self, r: int, c: int) -> tuple[str | None, int]:
        passages = self.grid[r][c].passages

        if passages == {"E", "W"}:
            span = 1
            cc = c - 1
            while cc >= 0 and self.grid[r][cc].passages == {"E", "W"}:
                span += 1
                cc -= 1
            cc = c + 1
            while cc < self.cols and self.grid[r][cc].passages == {"E", "W"}:
                span += 1
                cc += 1
            return "H", span

        if passages == {"N", "S"}:
            span = 1
            rr = r - 1
            while rr >= 0 and self.grid[rr][c].passages == {"N", "S"}:
                span += 1
                rr -= 1
            rr = r + 1
            while rr < self.rows and self.grid[rr][c].passages == {"N", "S"}:
                span += 1
                rr += 1
            return "V", span

        return None, 1

    def _hallway_branch_score(self, r: int, c: int, d: str) -> int:
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        score = 0

        for rr, cc in ((r, c), (nr, nc)):
            orient, span = self._straight_corridor_info(rr, cc)
            if orient is None or span < 5:
                continue
            if orient == "H" and d in ("N", "S"):
                score += span * 4
            if orient == "V" and d in ("E", "W"):
                score += span * 4

        return score

    def _rect_perimeter_open_except(
        self,
        r1: int,
        c1: int,
        r2: int,
        c2: int,
        skip_r: int,
        skip_c: int,
        skip_d: str,
    ) -> bool:
        """Check whether all edges around rectangle perimeter are open except one edge."""
        for row in (r1, r2):
            for c in range(c1, c2):
                if skip_d == "E" and skip_r == row and skip_c == c:
                    continue
                if not self._has_passage(row, c, "E"):
                    return False
        for col in (c1, c2):
            for r in range(r1, r2):
                if skip_d == "S" and skip_r == r and skip_c == col:
                    continue
                if not self._has_passage(r, col, "S"):
                    return False
        return True

    def _would_create_thin_rect_horizontal(self, r: int, c: int) -> bool:
        """Would opening (r,c)->E create a 2xN rectangle for any N>=2?"""
        for other_row in (r - 1, r + 1):
            if not (0 <= other_row < self.rows):
                continue

            if not self._has_passage(other_row, c, "E"):
                continue

            left = c
            i = c - 1
            while i >= 0 and self._has_passage(r, i, "E") and self._has_passage(other_row, i, "E"):
                left = i
                i -= 1

            right = c + 1
            i = c + 1
            while i < self.cols - 1 and self._has_passage(r, i, "E") and self._has_passage(other_row, i, "E"):
                right = i + 1
                i += 1

            top_row = min(r, other_row)
            left_has = any(self._has_passage(top_row, x, "S") for x in range(left, c + 1))
            if not left_has:
                continue
            right_has = any(self._has_passage(top_row, x, "S") for x in range(c + 1, right + 1))
            if right_has:
                return True

        return False

    def _would_create_thin_rect_vertical(self, r: int, c: int) -> bool:
        """Would opening (r,c)->S create an Nx2 rectangle for any N>=2?"""
        for other_col in (c - 1, c + 1):
            if not (0 <= other_col < self.cols):
                continue

            if not self._has_passage(r, other_col, "S"):
                continue

            top = r
            i = r - 1
            while i >= 0 and self._has_passage(i, c, "S") and self._has_passage(i, other_col, "S"):
                top = i
                i -= 1

            bottom = r + 1
            i = r + 1
            while i < self.rows - 1 and self._has_passage(i, c, "S") and self._has_passage(i, other_col, "S"):
                bottom = i + 1
                i += 1

            left_col = min(c, other_col)
            top_has = any(self._has_passage(x, left_col, "E") for x in range(top, r + 1))
            if not top_has:
                continue
            bottom_has = any(self._has_passage(x, left_col, "E") for x in range(r + 1, bottom + 1))
            if bottom_has:
                return True

        return False

    def _would_create_small_rect_loop(self, r: int, c: int, d: str) -> bool:
        """Avoid tiny rectangles up to max_rect to keep cycles longer."""
        if self.max_rect < 3:
            return False

        if d == "N":
            r, c, d = r - 1, c, "S"
        elif d == "W":
            r, c, d = r, c - 1, "E"

        if d == "E":
            h_values = range(3, self.max_rect + 1)
            w_values = range(3, self.max_rect + 1)
        elif d == "S":
            h_values = range(3, self.max_rect + 1)
            w_values = range(3, self.max_rect + 1)
        else:
            return False

        for h in h_values:
            for w in w_values:
                if d == "E":
                    c1_lo = max(0, c - w + 2)
                    c1_hi = min(c + 1, self.cols - w + 1)
                    for c1 in range(c1_lo, c1_hi):
                        c2 = c1 + w - 1
                        r2 = r + h - 1
                        if r2 < self.rows and self._rect_perimeter_open_except(r, c1, r2, c2, r, c, "E"):
                            return True
                        r1 = r - h + 1
                        if r1 >= 0 and self._rect_perimeter_open_except(r1, c1, r, c2, r, c, "E"):
                            return True
                else:  # d == "S"
                    r1_lo = max(0, r - h + 2)
                    r1_hi = min(r + 1, self.rows - h + 1)
                    for r1 in range(r1_lo, r1_hi):
                        r2 = r1 + h - 1
                        c2 = c + w - 1
                        if c2 < self.cols and self._rect_perimeter_open_except(r1, c, r2, c2, r, c, "S"):
                            return True
                        c1 = c - w + 1
                        if c1 >= 0 and self._rect_perimeter_open_except(r1, c1, r2, c, r, c, "S"):
                            return True
        return False

    def _would_create_forbidden_rect_loop(self, r: int, c: int, d: str) -> bool:
        """Forbid thin rectangles (2xN / Nx2), plus tiny rectangles up to max_rect."""
        if d == "E":
            if self._would_create_thin_rect_horizontal(r, c):
                return True
        elif d == "S":
            if self._would_create_thin_rect_vertical(r, c):
                return True
        elif d == "W":
            if self._would_create_thin_rect_horizontal(r, c - 1):
                return True
        elif d == "N":
            if self._would_create_thin_rect_vertical(r - 1, c):
                return True

        return self._would_create_small_rect_loop(r, c, d)

    def _wall_edges(self) -> list[tuple[int, int, str]]:
        walls: list[tuple[int, int, str]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols and "E" not in self.grid[r][c].passages:
                    walls.append((r, c, "E"))
                if r + 1 < self.rows and "S" not in self.grid[r][c].passages:
                    walls.append((r, c, "S"))
        return walls

    def _straight_runs(self) -> list[tuple[str, int, int, int]]:
        runs: list[tuple[str, int, int, int]] = []

        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if self.grid[r][c].passages == {"E", "W"}:
                    start = c
                    while c + 1 < self.cols and self.grid[r][c + 1].passages == {"E", "W"}:
                        c += 1
                    runs.append(("H", r, start, c))
                c += 1

        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if self.grid[r][c].passages == {"N", "S"}:
                    start = r
                    while r + 1 < self.rows and self.grid[r + 1][c].passages == {"N", "S"}:
                        r += 1
                    runs.append(("V", c, start, r))
                r += 1

        return runs

    def _longest_straight_hallway(self) -> int:
        longest = 0
        for _, _, start, end in self._straight_runs():
            longest = max(longest, end - start + 1)
        return longest

    def _find_one_thin_rectangle(self) -> tuple[int, int, int, int] | None:
        """Find one 2xN or Nx2 rectangular loop, or None if absent."""
        # 2xN: two adjacent rows with long parallel horizontal edges and
        # at least two vertical connectors between them.
        for r1 in range(self.rows - 1):
            c = 0
            while c < self.cols - 1:
                if not (self._has_passage(r1, c, "E") and self._has_passage(r1 + 1, c, "E")):
                    c += 1
                    continue
                start = c
                while c < self.cols - 1 and self._has_passage(r1, c, "E") and self._has_passage(r1 + 1, c, "E"):
                    c += 1
                end = c  # rightmost cell column in this run
                connectors = [x for x in range(start, end + 1) if self._has_passage(r1, x, "S")]
                if len(connectors) >= 2:
                    return (r1, connectors[0], r1 + 1, connectors[-1])

        # Nx2: two adjacent cols with long parallel vertical edges and
        # at least two horizontal connectors between them.
        for c1 in range(self.cols - 1):
            r = 0
            while r < self.rows - 1:
                if not (self._has_passage(r, c1, "S") and self._has_passage(r, c1 + 1, "S")):
                    r += 1
                    continue
                start = r
                while r < self.rows - 1 and self._has_passage(r, c1, "S") and self._has_passage(r, c1 + 1, "S"):
                    r += 1
                end = r  # bottommost cell row in this run
                connectors = [x for x in range(start, end + 1) if self._has_passage(x, c1, "E")]
                if len(connectors) >= 2:
                    return (connectors[0], c1, connectors[-1], c1 + 1)

        return None

    def _rect_perimeter_edges(self, rect: tuple[int, int, int, int]) -> list[tuple[int, int, str]]:
        """Return perimeter edges as canonical (r,c,d) with d in {'E','S'}."""
        r1, c1, r2, c2 = rect
        edges: list[tuple[int, int, str]] = []
        for c in range(c1, c2):
            edges.append((r1, c, "E"))
            edges.append((r2, c, "E"))
        for r in range(r1, r2):
            edges.append((r, c1, "S"))
            edges.append((r, c2, "S"))
        return edges

    def _is_thin_rectangle_open(self, rect: tuple[int, int, int, int]) -> bool:
        r1, c1, r2, c2 = rect
        if not (0 <= r1 < r2 < self.rows and 0 <= c1 < c2 < self.cols):
            return False
        if not (r2 - r1 == 1 or c2 - c1 == 1):
            return False
        for c in range(c1, c2):
            if not self._has_passage(r1, c, "E"):
                return False
            if not self._has_passage(r2, c, "E"):
                return False
        for r in range(r1, r2):
            if not self._has_passage(r, c1, "S"):
                return False
            if not self._has_passage(r, c2, "S"):
                return False
        return True

    def _break_one_thin_rectangle(self, rect: tuple[int, int, int, int] | None = None) -> bool:
        """Try to break one thin rectangle while keeping constraints strong."""
        if rect is None:
            rect = self._find_one_thin_rectangle()
        if rect is None:
            return False

        old_thin = self._count_thin_rectangles()
        edges = self._rect_perimeter_edges(rect)

        # Prefer edges with high endpoint degree so closing them won't create dead ends.
        candidates: list[tuple[int, int, int, int, str]] = []
        for r, c, d in edges:
            if not self._has_passage(r, c, d):
                continue
            dr, dc = DIRS[d]
            nr, nc = r + dr, c + dc
            deg_a = len(self.grid[r][c].passages)
            deg_b = len(self.grid[nr][nc].passages)
            candidates.append((min(deg_a, deg_b), deg_a + deg_b, r, c, d))

        candidates.sort(reverse=True)

        for _, _, r, c, d in candidates:
            snap = self._snapshot()
            if not self._close_passage(r, c, d):
                continue

            if self._dead_ends():
                self._eliminate_dead_ends(strict=True)
            if self._dead_ends():
                self._eliminate_dead_ends(strict=False)

            new_thin = self._count_thin_rectangles()
            rect_removed = not self._is_thin_rectangle_open(rect)
            if len(self._dead_ends()) == 0 and (new_thin < old_thin or rect_removed):
                return True

            self._restore(snap)

        return False

    def _count_thin_rectangles(self) -> int:
        """Count all axis-aligned 2xN / Nx2 rectangular loops."""
        rects: set[tuple[int, int, int, int]] = set()

        def east(r: int, c: int) -> bool:
            return "E" in self.grid[r][c].passages

        def south(r: int, c: int) -> bool:
            return "S" in self.grid[r][c].passages

        # h = 2
        for r1 in range(self.rows - 1):
            r2 = r1 + 1
            for c1 in range(self.cols - 1):
                if not south(r1, c1):
                    continue
                for c2 in range(c1 + 1, self.cols):
                    if not south(r1, c2):
                        continue
                    ok = True
                    for c in range(c1, c2):
                        if not east(r1, c) or not east(r2, c):
                            ok = False
                            break
                    if ok:
                        rects.add((r1, c1, r2, c2))

        # w = 2
        for c1 in range(self.cols - 1):
            c2 = c1 + 1
            for r1 in range(self.rows - 1):
                if not east(r1, c1):
                    continue
                for r2 in range(r1 + 1, self.rows):
                    if not east(r2, c1):
                        continue
                    ok = True
                    for r in range(r1, r2):
                        if not south(r, c1) or not south(r, c2):
                            ok = False
                            break
                    if ok:
                        rects.add((r1, c1, r2, c2))

        return len(rects)

    # ── endpoints and stats ───────────────────────────────────────────────

    def _farthest_from(self, start: tuple[int, int]) -> tuple[tuple[int, int], int]:
        dq = deque([start])
        dist = {start: 0}
        farthest = start

        while dq:
            r, c = dq.popleft()
            current = dist[(r, c)]
            if current > dist[farthest]:
                farthest = (r, c)

            for d in self.grid[r][c].passages:
                dr, dc = DIRS[d]
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nxt in dist:
                    continue
                dist[nxt] = current + 1
                dq.append(nxt)

        return farthest, dist[farthest]

    def _pick_endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        a, _ = self._farthest_from((0, 0))
        b, _ = self._farthest_from(a)
        return a, b

    def _collect_generation_stats(self) -> dict[str, int]:
        return {
            "dead_ends": len(self._dead_ends()),
            "thin_rectangles": self._count_thin_rectangles(),
            "loop_gain_total": self.loop_gain_total,
            "longest_hallway": self._longest_straight_hallway(),
        }

    def cumulative_shortest_loops(self, cap: int = 300) -> tuple[int, int]:
        """For each tile, add the length of its shortest cycle. Returns (sum, missing)."""
        total = 0
        missing = 0

        for r in range(self.rows):
            for c in range(self.cols):
                nbrs: list[tuple[int, int]] = []
                for d in self.grid[r][c].passages:
                    dr, dc = DIRS[d]
                    nbrs.append((r + dr, c + dc))

                best: int | None = None
                for i in range(len(nbrs)):
                    for j in range(i + 1, len(nbrs)):
                        dist = self._shortest_path_len(nbrs[i], nbrs[j], cap=cap, blocked_cell=(r, c))
                        if dist is None:
                            continue
                        cycle_len = dist + 2
                        if best is None or cycle_len < best:
                            best = cycle_len

                if best is None:
                    missing += 1
                else:
                    total += best

        return total, missing

    # ── visualization helpers ─────────────────────────────────────────────

    def to_2d_array(self) -> list[list[str]]:
        """Return an expanded wall/path grid using '#', ' ', 'S', 'F'."""
        h = self.rows * 2 + 1
        w = self.cols * 2 + 1
        arr = [["#" for _ in range(w)] for _ in range(h)]

        for r in range(self.rows):
            for c in range(self.cols):
                rr, cc = 2 * r + 1, 2 * c + 1
                arr[rr][cc] = " "
                for d in self.grid[r][c].passages:
                    dr, dc = DIRS[d]
                    arr[rr + dr][cc + dc] = " "

        sr, sc = self.start
        fr, fc = self.finish
        arr[2 * sr + 1][2 * sc + 1] = "S"
        arr[2 * fr + 1][2 * fc + 1] = "F"
        return arr

    def render_2d_array(self) -> str:
        return "\n".join("".join(row) for row in self.to_2d_array())

    # ── ASCII rendering ───────────────────────────────────────────────────

    def render(
        self,
        player: tuple[int, int],
        start: tuple[int, int],
        finish: tuple[int, int],
        visited: set[tuple[int, int]] | None = None,
    ) -> str:
        if visited is None:
            visited = set()

        lines = []
        # Top border
        top = "+" + "+".join("---" if not self._has_passage(0, c, "N") else "   " for c in range(self.cols)) + "+"
        lines.append(top)

        for r in range(self.rows):
            # Middle row: west wall + cell content + east walls
            mid = ""
            for c in range(self.cols):
                west_wall = "|" if not self._has_passage(r, c, "W") else " "
                mid += west_wall
                if (r, c) == player:
                    mid += " @ "
                elif (r, c) == start:
                    mid += " S "
                elif (r, c) == finish:
                    mid += " F "
                elif (r, c) in visited:
                    mid += " · "
                else:
                    mid += "   "
            mid += "|"  # east border
            lines.append(mid)

            # South walls row
            south = ""
            for c in range(self.cols):
                south += "+"
                south += "   " if self._has_passage(r, c, "S") else "---"
            south += "+"
            lines.append(south)

        return "\n".join(lines)


# ── game loop ─────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if sys.platform == "win32" else "clear")


def play(rows: int = 10, cols: int = 15, seed: int | None = None, max_rect: int = 5):
    maze = Maze(rows, cols, seed, max_rect=max_rect)
    start = maze.start
    finish = maze.finish
    player = start
    visited: set[tuple[int, int]] = {start}
    moves = 0

    while True:
        clear()
        print("  BRAID MAZE  —  reach F from S")
        print("  WASD / arrow keys to move  |  Q to quit\n")
        print(maze.render(player, start, finish, visited))
        print(f"\n  Moves: {moves}")

        if player == finish:
            print("\n  You reached the finish! Well done.\n")
            break

        key = read_key()

        direction = {
            "w": "N", "up": "N",
            "s": "S", "down": "S",
            "d": "E", "right": "E",
            "a": "W", "left": "W",
        }.get(key)

        if key == "q":
            print("\n  Quitting. Bye!\n")
            break

        if direction and direction in maze.grid[player[0]][player[1]].passages:
            dr, dc = DIRS[direction]
            player = (player[0] + dr, player[1] + dc)
            visited.add(player)
            moves += 1


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Braid Maze Game")
    parser.add_argument("--rows", type=int, default=10, help="Maze height (default 10)")
    parser.add_argument("--cols", type=int, default=15, help="Maze width  (default 15)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible mazes")
    parser.add_argument("--max-rect", "--max_rect", type=int, default=5,
                        help="Suppress tiny rectangular loops up to this size (default 5)")
    args = parser.parse_args()

    play(rows=args.rows, cols=args.cols, seed=args.seed, max_rect=args.max_rect)
