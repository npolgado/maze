#!/usr/bin/env python3
"""
High-quality random braid maze generator for a 30 x 30 grid.

A braid maze is a connected maze graph with no dead ends. This generator does
not simply carve a perfect maze and knock out random walls. Instead it creates
many sparse connected candidate graphs, repairs them into braid mazes, scores
them, improves them with simulated annealing, and keeps the best result.

Only the Python standard library is used.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import argparse
import math
import os
import random
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]


DIRS: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass
class ScoreWeights:
    """Tunable weights for the quality function."""

    connected_reward: float = 100000.0
    disconnected_penalty: float = 250000.0
    dead_end_penalty: float = 35000.0
    isolated_penalty: float = 60000.0
    loop_depth_weight: float = 85.0
    short_loop_penalty: float = 420.0
    rectangle_penalty: float = 900.0
    narrow_rectangle_penalty: float = 140.0
    hallway_penalty: float = 180.0
    openness_penalty: float = 14000.0
    degree4_penalty: float = 280.0
    dense_window_penalty: float = 850.0
    intersection_reward: float = 145.0
    sf_distance_weight: float = 42.0
    sf_redundancy_weight: float = 950.0
    cyclomatic_weight: float = 180.0
    regularity_penalty: float = 120.0


@dataclass
class ScoreDetails:
    total: float
    connected: bool
    components: int
    dead_ends: int
    isolated: int
    loop_depth_score: float
    short_loop_count: int
    small_rectangle_penalty: float
    long_hallway_penalty: float
    openness_penalty: float
    branch_reward: float
    start_finish_score: float
    cyclomatic: int
    longest_straight_hallway: int
    small_rectangle_estimate: int
    average_degree: float
    degree_counts: Dict[int, int]
    shortest_start_finish: int
    start_finish_redundancy: int


class MazeGraph:
    """Undirected graph whose nodes are orthogonally adjacent grid cells."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.adj: Dict[Cell, Set[Cell]] = {
            (x, y): set() for y in range(height) for x in range(width)
        }

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    def copy(self) -> "MazeGraph":
        other = MazeGraph(self.width, self.height)
        other.adj = {cell: set(neighbors) for cell, neighbors in self.adj.items()}
        return other

    def cells(self) -> Iterable[Cell]:
        return self.adj.keys()

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def grid_neighbors(self, cell: Cell) -> List[Cell]:
        x, y = cell
        result = []
        for dx, dy in DIRS:
            n = (x + dx, y + dy)
            if self.in_bounds(n):
                result.append(n)
        return result

    def has_edge(self, a: Cell, b: Cell) -> bool:
        return b in self.adj[a]

    def add_edge(self, a: Cell, b: Cell) -> None:
        if a == b or not self.in_bounds(a) or not self.in_bounds(b):
            raise ValueError(f"invalid edge {a!r}-{b!r}")
        ax, ay = a
        bx, by = b
        if abs(ax - bx) + abs(ay - by) != 1:
            raise ValueError(f"edge endpoints must be adjacent: {a!r}-{b!r}")
        self.adj[a].add(b)
        self.adj[b].add(a)

    def remove_edge(self, a: Cell, b: Cell) -> None:
        self.adj[a].remove(b)
        self.adj[b].remove(a)

    def degree(self, cell: Cell) -> int:
        return len(self.adj[cell])

    def edges(self) -> List[Edge]:
        result: List[Edge] = []
        for a, neighbors in self.adj.items():
            for b in neighbors:
                if a < b:
                    result.append((a, b))
        return result

    def closed_edges(self) -> List[Edge]:
        result: List[Edge] = []
        for cell in self.cells():
            for n in self.grid_neighbors(cell):
                if cell < n and not self.has_edge(cell, n):
                    result.append((cell, n))
        return result

    def component_count(self) -> int:
        seen: Set[Cell] = set()
        count = 0
        for start in self.cells():
            if start in seen:
                continue
            count += 1
            queue = deque([start])
            seen.add(start)
            while queue:
                cur = queue.popleft()
                for n in self.adj[cur]:
                    if n not in seen:
                        seen.add(n)
                        queue.append(n)
        return count

    def is_connected(self) -> bool:
        return self.component_count() == 1

    def shortest_path(
        self,
        start: Cell,
        goal: Cell,
        blocked_edge: Optional[Edge] = None,
    ) -> Optional[List[Cell]]:
        blocked = normalized_edge(*blocked_edge) if blocked_edge else None
        queue = deque([start])
        parent: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cur = queue.popleft()
            if cur == goal:
                path: List[Cell] = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]  # type: ignore[assignment]
                path.reverse()
                return path
            for n in self.adj[cur]:
                if blocked and normalized_edge(cur, n) == blocked:
                    continue
                if n not in parent:
                    parent[n] = cur
                    queue.append(n)
        return None

    def distance(
        self,
        start: Cell,
        goal: Cell,
        blocked_edge: Optional[Edge] = None,
    ) -> Optional[int]:
        path = self.shortest_path(start, goal, blocked_edge)
        if path is None:
            return None
        return len(path) - 1


def normalized_edge(a: Cell, b: Cell) -> Edge:
    return (a, b) if a < b else (b, a)


def random_spanning_tree(width: int, height: int, rng: random.Random) -> MazeGraph:
    """Create a randomized DFS spanning tree, used only as a connected seed."""

    maze = MazeGraph(width, height)
    start = (rng.randrange(width), rng.randrange(height))
    stack = [start]
    seen = {start}

    while stack:
        cur = stack[-1]
        choices = [n for n in maze.grid_neighbors(cur) if n not in seen]
        if not choices:
            stack.pop()
            continue
        n = rng.choice(choices)
        maze.add_edge(cur, n)
        seen.add(n)
        stack.append(n)

    return maze


def generate_sparse_connected_candidate(
    width: int,
    height: int,
    rng: random.Random,
    extra_edge_ratio: float,
) -> MazeGraph:
    """
    Start from a connected sparse graph, then add a modest number of biased
    extra edges. This creates loops before repair, but keeps the graph labyrinthine.
    """

    maze = random_spanning_tree(width, height, rng)
    closed = maze.closed_edges()
    rng.shuffle(closed)
    target_extra = int(width * height * extra_edge_ratio)

    for _ in range(target_extra):
        if not closed:
            break
        scored: List[Tuple[float, Edge]] = []
        sample = rng.sample(closed, min(30, len(closed)))
        for edge in sample:
            a, b = edge
            deg_bonus = (maze.degree(a) <= 1) + (maze.degree(b) <= 1)
            # Use Manhattan span as a cheap proxy during seeding. Full graph
            # loop depth is measured later for finalists.
            d = abs(a[0] - b[0]) + abs(a[1] - b[1])
            loop_bonus = min(d, 12) / 12.0
            open_penalty = 0.25 * (maze.degree(a) + maze.degree(b))
            scored.append((2.5 * deg_bonus + loop_bonus - open_penalty + rng.random(), edge))
        _, chosen = max(scored, key=lambda item: item[0])
        a, b = chosen
        maze.add_edge(a, b)
        closed.remove(chosen)

    return maze


def repair_dead_ends(maze: MazeGraph, rng: random.Random) -> None:
    """
    Add carefully chosen edges until every cell has degree >= 2.

    The repair operation prefers edges that touch dead ends and whose existing
    alternative path is long, because that creates a deeper loop instead of a
    tiny square.
    """

    guard = maze.cell_count * 8
    while guard > 0:
        guard -= 1
        dead = [cell for cell in maze.cells() if maze.degree(cell) < 2]
        if not dead:
            return

        rng.shuffle(dead)
        cell = dead[0]
        options = [
            normalized_edge(cell, n)
            for n in maze.grid_neighbors(cell)
            if not maze.has_edge(cell, n)
        ]

        if not options:
            # Extremely rare after a connected seed. If a cell is boxed by all
            # possible grid edges, help the lowest-degree neighbor instead.
            neighbors = sorted(maze.grid_neighbors(cell), key=maze.degree)
            cell = neighbors[0]
            options = [
                normalized_edge(cell, n)
                for n in maze.grid_neighbors(cell)
                if not maze.has_edge(cell, n)
            ]

        best_edge = None
        best_value = -10**9
        for edge in options:
            a, b = edge
            alt = maze.distance(a, b)
            alt_value = 0 if alt is None else min(alt, 40)
            dead_help = (maze.degree(a) < 2) + (maze.degree(b) < 2)
            open_penalty = maze.degree(a) + maze.degree(b)
            tiny_rectangle_penalty = local_rectangle_risk(maze, edge) * 8
            value = 120 * dead_help + 4 * alt_value - 7 * open_penalty
            value -= tiny_rectangle_penalty
            value += rng.random()
            if value > best_value:
                best_value = value
                best_edge = edge

        if best_edge is None:
            raise RuntimeError("could not repair dead end")
        maze.add_edge(*best_edge)

    raise RuntimeError("dead-end repair did not converge")


def local_rectangle_risk(maze: MazeGraph, edge: Edge) -> int:
    """Cheap local estimate of whether adding an edge closes a tiny square."""

    a, b = edge
    ax, ay = a
    bx, by = b
    risk = 0
    if ax == bx:
        for dx in (-1, 1):
            c = (ax + dx, ay)
            d = (bx + dx, by)
            if maze.in_bounds(c) and maze.in_bounds(d):
                if maze.has_edge(a, c) and maze.has_edge(c, d) and maze.has_edge(d, b):
                    risk += 1
    else:
        for dy in (-1, 1):
            c = (ax, ay + dy)
            d = (bx, by + dy)
            if maze.in_bounds(c) and maze.in_bounds(d):
                if maze.has_edge(a, c) and maze.has_edge(c, d) and maze.has_edge(d, b):
                    risk += 1
    return risk


def farthest_pair_by_sampling(
    maze: MazeGraph,
    rng: random.Random,
    samples: int = 28,
) -> Tuple[Cell, Cell, int]:
    """Pick a good start/finish pair by repeated BFS sweeps."""

    cells = list(maze.cells())
    best = (cells[0], cells[-1], -1)
    starts = rng.sample(cells, min(samples, len(cells)))

    for start in starts:
        distances = all_distances(maze, start)
        finish, dist = max(distances.items(), key=lambda item: item[1])
        if dist > best[2]:
            best = (start, finish, dist)
    return best


def all_distances(maze: MazeGraph, start: Cell) -> Dict[Cell, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for n in maze.adj[cur]:
            if n not in distances:
                distances[n] = distances[cur] + 1
                queue.append(n)
    return distances


def loop_depth_metrics(
    maze: MazeGraph,
    rng: random.Random,
    sample_size: int,
) -> Tuple[float, int]:
    """
    For sampled open edges, remove that edge virtually and find the shortest
    alternate path between its endpoints. Longer alternate paths imply larger,
    more interesting loops. Tiny alternate paths imply boring small cycles.
    """

    edges = maze.edges()
    if not edges:
        return 0.0, 0

    sample = rng.sample(edges, min(sample_size, len(edges)))
    total = 0.0
    short_loops = 0
    for edge in sample:
        a, b = edge
        alt = maze.distance(a, b, blocked_edge=edge)
        if alt is None:
            continue
        if alt <= 5:
            short_loops += 1
            total -= (6 - alt) * 2.0
        total += math.sqrt(alt) * 8.0 + min(alt, 40) * 0.35
    return total / len(sample), short_loops


def average_alternate_loop_size(
    maze: MazeGraph,
    rng: random.Random,
    sample_size: int = 240,
) -> float:
    """Average raw alternate path length after removing sampled open edges."""

    edges = maze.edges()
    if not edges:
        return 0.0
    sample = rng.sample(edges, min(sample_size, len(edges)))
    lengths: List[int] = []
    for edge in sample:
        alt = maze.distance(edge[0], edge[1], blocked_edge=edge)
        if alt is not None:
            lengths.append(alt)
    return sum(lengths) / len(lengths) if lengths else 0.0


def quick_loop_depth_proxy(
    maze: MazeGraph,
    rng: random.Random,
    sample_size: int,
) -> Tuple[float, int]:
    """
    Fast loop-quality proxy for the annealing inner loop.

    It rewards sampled edges that are not part of tiny 2x2 closures and that
    touch lower-degree cells. The full alternate-path BFS loop-depth metric is
    still used when a candidate is finalized.
    """

    edges = maze.edges()
    if not edges:
        return 0.0, 0
    sample = rng.sample(edges, min(sample_size, len(edges)))
    total = 0.0
    short = 0
    for edge in sample:
        a, b = edge
        tiny = local_rectangle_risk(maze, edge)
        if tiny:
            short += tiny
        total += 24.0 - tiny * 10.0 - max(0, maze.degree(a) - 3) * 2.0
        total -= max(0, maze.degree(b) - 3) * 2.0
    return total / len(sample), short


def small_rectangle_penalty(maze: MazeGraph) -> Tuple[float, int]:
    """
    Approximate boring rectangular cycles.

    For each small/narrow rectangle, the perimeter is checked. If every
    perimeter edge is open and there is little internal complexity, it looks
    like a plain rectangular racetrack and is penalized. This intentionally
    targets 2x2, 2x3, 2x4, and narrow 2xN loops without needing full cycle
    basis analysis.
    """

    penalty = 0.0
    count = 0
    max_w = min(8, maze.width)
    max_h = min(8, maze.height)

    for rw in range(2, max_w + 1):
        for rh in range(2, max_h + 1):
            narrow = rw == 2 or rh == 2
            small = rw <= 4 and rh <= 4
            if not (narrow or small):
                continue
            for x0 in range(maze.width - rw + 1):
                for y0 in range(maze.height - rh + 1):
                    if rectangle_perimeter_open(maze, x0, y0, rw, rh):
                        internal = rectangle_internal_edges(maze, x0, y0, rw, rh)
                        if internal <= max(1, (rw * rh) // 8):
                            count += 1
                            area = rw * rh
                            if rw == 2 and rh == 2:
                                penalty += 2.8
                            elif rw == 2 or rh == 2:
                                penalty += 1.2 + area * 0.08
                            else:
                                penalty += 0.75 + area * 0.04
    return penalty, count


def quick_small_rectangle_penalty(maze: MazeGraph) -> Tuple[float, int]:
    """Fast inner-loop approximation: count only fully open 2x2 squares."""

    count = 0
    for y in range(maze.height - 1):
        for x in range(maze.width - 1):
            if (
                maze.has_edge((x, y), (x + 1, y))
                and maze.has_edge((x, y + 1), (x + 1, y + 1))
                and maze.has_edge((x, y), (x, y + 1))
                and maze.has_edge((x + 1, y), (x + 1, y + 1))
            ):
                count += 1
    return count * 2.8, count


def narrow_rectangle_loop_counts(maze: MazeGraph, max_long_side: int = 12) -> Dict[str, int]:
    """Count plain 2xN rectangular perimeter loops by outer cell dimensions."""

    counts = {f"2x{n}": 0 for n in range(2, max_long_side + 1)}
    for n in range(2, max_long_side + 1):
        for rw, rh in ((2, n), (n, 2)):
            if rw > maze.width or rh > maze.height:
                continue
            label = f"2x{n}"
            for x0 in range(maze.width - rw + 1):
                for y0 in range(maze.height - rh + 1):
                    if rectangle_perimeter_open(maze, x0, y0, rw, rh):
                        internal = rectangle_internal_edges(maze, x0, y0, rw, rh)
                        if internal <= max(1, (rw * rh) // 8):
                            counts[label] += 1
    return counts


def rectangle_perimeter_open(
    maze: MazeGraph,
    x0: int,
    y0: int,
    rw: int,
    rh: int,
) -> bool:
    x1 = x0 + rw - 1
    y1 = y0 + rh - 1

    for x in range(x0, x1):
        if not maze.has_edge((x, y0), (x + 1, y0)):
            return False
        if not maze.has_edge((x, y1), (x + 1, y1)):
            return False
    for y in range(y0, y1):
        if not maze.has_edge((x0, y), (x0, y + 1)):
            return False
        if not maze.has_edge((x1, y), (x1, y + 1)):
            return False
    return True


def rectangle_internal_edges(
    maze: MazeGraph,
    x0: int,
    y0: int,
    rw: int,
    rh: int,
) -> int:
    x1 = x0 + rw - 1
    y1 = y0 + rh - 1
    total = 0
    for y in range(y0 + 1, y1):
        for x in range(x0, x1):
            if maze.has_edge((x, y), (x + 1, y)):
                total += 1
    for x in range(x0 + 1, x1):
        for y in range(y0, y1):
            if maze.has_edge((x, y), (x, y + 1)):
                total += 1
    return total


def straight_hallway_lengths(maze: MazeGraph) -> List[int]:
    """
    Find maximal straight runs of degree-2 cells with no intersection in the run.
    A bend breaks the run. End cells may be intersections or boundary cells.
    """

    lengths: List[int] = []
    seen_segments: Set[Edge] = set()

    for cell in maze.cells():
        if maze.degree(cell) != 2:
            continue
        neighbors = list(maze.adj[cell])
        horizontal = all(n[1] == cell[1] for n in neighbors)
        vertical = all(n[0] == cell[0] for n in neighbors)
        if not (horizontal or vertical):
            continue

        for n in neighbors:
            seg = normalized_edge(cell, n)
            if seg in seen_segments:
                continue
            dx = 0 if n[0] == cell[0] else (1 if n[0] > cell[0] else -1)
            dy = 0 if n[1] == cell[1] else (1 if n[1] > cell[1] else -1)
            run = [cell]
            prev = cell
            cur = n
            while True:
                seen_segments.add(normalized_edge(prev, cur))
                if maze.degree(cur) != 2:
                    break
                cur_neighbors = list(maze.adj[cur])
                straight = all(
                    (other[0] == cur[0] if dx == 0 else other[1] == cur[1])
                    for other in cur_neighbors
                )
                if not straight:
                    break
                run.append(cur)
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt not in maze.adj[cur]:
                    break
                prev, cur = cur, nxt
            if len(run) > 1:
                lengths.append(len(run))

    return lengths


def long_hallway_penalty(maze: MazeGraph, threshold: int = 6) -> Tuple[float, int]:
    lengths = straight_hallway_lengths(maze)
    penalty = 0.0
    for length in lengths:
        if length > threshold:
            penalty += (length - threshold) ** 1.7
    return penalty, max(lengths, default=0)


def openness_penalty(maze: MazeGraph) -> float:
    degrees = [maze.degree(cell) for cell in maze.cells()]
    avg_degree = sum(degrees) / len(degrees)
    penalty = max(0.0, avg_degree - 2.42) ** 2 * 9.0
    penalty += sum(1 for d in degrees if d == 4) * 0.12

    # Penalize 3x3 windows that are almost completely open.
    dense_windows = 0
    for y0 in range(maze.height - 2):
        for x0 in range(maze.width - 2):
            possible = 12
            open_edges = 0
            for y in range(y0, y0 + 3):
                for x in range(x0, x0 + 2):
                    if maze.has_edge((x, y), (x + 1, y)):
                        open_edges += 1
            for x in range(x0, x0 + 3):
                for y in range(y0, y0 + 2):
                    if maze.has_edge((x, y), (x, y + 1)):
                        open_edges += 1
            if open_edges >= possible - 1:
                dense_windows += 1
    penalty += dense_windows * 0.7
    return penalty


def quick_openness_penalty(
    maze: MazeGraph,
    average_degree: float,
    degree_counts: Dict[int, int],
) -> float:
    """Fast inner-loop openness estimate based on degree distribution only."""

    penalty = max(0.0, average_degree - 2.42) ** 2 * 9.0
    penalty += degree_counts.get(4, 0) * 0.12
    return penalty


def regularity_penalty(maze: MazeGraph) -> float:
    """Penalize over-repeated degree patterns in rows and columns."""

    penalty = 0.0
    for y in range(maze.height):
        run_degree = None
        run_len = 0
        for x in range(maze.width):
            d = maze.degree((x, y))
            if d == run_degree:
                run_len += 1
            else:
                if run_len >= 8:
                    penalty += run_len - 7
                run_degree = d
                run_len = 1
        if run_len >= 8:
            penalty += run_len - 7

    for x in range(maze.width):
        run_degree = None
        run_len = 0
        for y in range(maze.height):
            d = maze.degree((x, y))
            if d == run_degree:
                run_len += 1
            else:
                if run_len >= 8:
                    penalty += run_len - 7
                run_degree = d
                run_len = 1
        if run_len >= 8:
            penalty += run_len - 7
    return penalty


def start_finish_redundancy(maze: MazeGraph, start: Cell, finish: Cell) -> int:
    """
    Estimate route redundancy by removing sampled edges from one shortest path.
    Count how often start and finish remain connected.
    """

    path = maze.shortest_path(start, finish)
    if path is None or len(path) < 2:
        return 0
    path_edges = [normalized_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if len(path_edges) <= 16:
        sample = path_edges
    else:
        step = max(1, len(path_edges) // 16)
        sample = path_edges[::step][:16]
    return sum(1 for edge in sample if maze.distance(start, finish, edge) is not None)


def score_maze(
    maze: MazeGraph,
    rng: random.Random,
    weights: ScoreWeights,
    start: Optional[Cell] = None,
    finish: Optional[Cell] = None,
    loop_sample_size: int = 120,
    expensive: bool = True,
) -> ScoreDetails:
    components = maze.component_count()
    connected = components == 1
    degrees = [maze.degree(cell) for cell in maze.cells()]
    degree_counts = {d: degrees.count(d) for d in range(5)}
    dead_ends = sum(1 for d in degrees if d <= 1)
    isolated = sum(1 for d in degrees if d == 0)
    average_degree = sum(degrees) / len(degrees)

    if not expensive and (start is None or finish is None):
        start = (0, 0)
        finish = (maze.width - 1, maze.height - 1)
        sf_dist = abs(finish[0] - start[0]) + abs(finish[1] - start[1])
    elif start is None or finish is None:
        start, finish, sf_dist = farthest_pair_by_sampling(maze, rng)
    else:
        sf_dist = maze.distance(start, finish) or 0

    if expensive:
        loop_depth, short_loops = loop_depth_metrics(maze, rng, loop_sample_size)
    else:
        loop_depth, short_loops = quick_loop_depth_proxy(maze, rng, loop_sample_size)
    hallway_raw, longest_hallway = long_hallway_penalty(maze)

    if expensive:
        rect_penalty_raw, rect_count = small_rectangle_penalty(maze)
        open_raw = openness_penalty(maze)
        regular_raw = regularity_penalty(maze)
    else:
        # The annealing inner loop calls the scorer thousands of times. These
        # approximations keep optimization fast; the final/best candidates are
        # always rescored with the full rectangular and dense-window scans.
        rect_penalty_raw, rect_count = quick_small_rectangle_penalty(maze)
        open_raw = quick_openness_penalty(maze, average_degree, degree_counts)
        regular_raw = 0.0
    redundancy = start_finish_redundancy(maze, start, finish) if expensive else 0
    edge_count = len(maze.edges())
    cyclomatic = edge_count - maze.cell_count + components

    branch_reward = degree_counts.get(3, 0) + degree_counts.get(4, 0) * 0.35
    sf_score = sf_dist * weights.sf_distance_weight
    sf_score += redundancy * weights.sf_redundancy_weight

    total = 0.0
    total += weights.connected_reward if connected else -weights.disconnected_penalty * components
    total -= weights.dead_end_penalty * dead_ends
    total -= weights.isolated_penalty * isolated
    total += weights.loop_depth_weight * loop_depth
    total -= weights.short_loop_penalty * short_loops
    total -= weights.rectangle_penalty * rect_penalty_raw
    total -= weights.narrow_rectangle_penalty * rect_count
    total -= weights.hallway_penalty * hallway_raw
    total -= weights.openness_penalty * open_raw
    total -= weights.degree4_penalty * degree_counts.get(4, 0)
    total += weights.intersection_reward * branch_reward
    total += sf_score
    total += weights.cyclomatic_weight * min(cyclomatic, maze.cell_count // 3)
    total -= weights.regularity_penalty * regular_raw

    return ScoreDetails(
        total=total,
        connected=connected,
        components=components,
        dead_ends=dead_ends,
        isolated=isolated,
        loop_depth_score=loop_depth,
        short_loop_count=short_loops,
        small_rectangle_penalty=rect_penalty_raw,
        long_hallway_penalty=hallway_raw,
        openness_penalty=open_raw,
        branch_reward=branch_reward,
        start_finish_score=sf_score,
        cyclomatic=cyclomatic,
        longest_straight_hallway=longest_hallway,
        small_rectangle_estimate=rect_count,
        average_degree=average_degree,
        degree_counts=degree_counts,
        shortest_start_finish=sf_dist,
        start_finish_redundancy=redundancy,
    )


def is_valid_braid(maze: MazeGraph) -> bool:
    return maze.is_connected() and all(maze.degree(cell) >= 2 for cell in maze.cells())


def propose_addition(maze: MazeGraph, rng: random.Random) -> Optional[Edge]:
    closed = maze.closed_edges()
    if not closed:
        return None
    sample = rng.sample(closed, min(18, len(closed)))
    best_edge = None
    best_value = -10**9
    for edge in sample:
        a, b = edge
        span = abs(a[0] - b[0]) + abs(a[1] - b[1])
        value = span * 1.8
        value -= (maze.degree(a) + maze.degree(b)) * 2.4
        value -= local_rectangle_risk(maze, edge) * 18
        value += rng.random() * 8
        if value > best_value:
            best_value = value
            best_edge = edge
    return best_edge


def propose_removal(maze: MazeGraph, rng: random.Random) -> Optional[Edge]:
    edges = maze.edges()
    removable = [
        edge for edge in edges
        if maze.degree(edge[0]) > 2 and maze.degree(edge[1]) > 2
    ]
    if not removable:
        return None
    return rng.choice(removable)


def removal_is_safe(maze: MazeGraph, edge: Edge) -> bool:
    a, b = edge
    if maze.degree(a) <= 2 or maze.degree(b) <= 2:
        return False
    # If there is an alternate path between endpoints, removing this edge keeps
    # the whole connected graph connected.
    return maze.distance(a, b, blocked_edge=edge) is not None


def optimize_maze(
    maze: MazeGraph,
    rng: random.Random,
    weights: ScoreWeights,
    iterations: int,
) -> Tuple[MazeGraph, ScoreDetails, Cell, Cell]:
    """Improve a candidate with add/remove edge flips and annealed acceptance."""

    repair_dead_ends(maze, rng)
    start, finish, _ = farthest_pair_by_sampling(maze, rng)
    current_score = score_maze(
        maze,
        rng,
        weights,
        start,
        finish,
        loop_sample_size=32,
        expensive=False,
    )
    best = maze.copy()
    best_start, best_finish = start, finish
    best_score = current_score

    for i in range(iterations):
        temperature = max(0.04, 1.0 - i / max(1, iterations))
        candidate = maze.copy()

        # Additions are more common when the graph is close to the sparse target;
        # removals trim excess openness and overly dense intersections.
        if rng.random() < 0.58:
            edge = propose_addition(candidate, rng)
            if edge is None:
                continue
            candidate.add_edge(*edge)
        else:
            edge = propose_removal(candidate, rng)
            if edge is None or not removal_is_safe(candidate, edge):
                continue
            candidate.remove_edge(*edge)

        if not is_valid_braid(candidate):
            continue

        cand_start, cand_finish = start, finish
        cand_score = score_maze(
            candidate,
            rng,
            weights,
            cand_start,
            cand_finish,
            loop_sample_size=32,
            expensive=False,
        )
        delta = cand_score.total - current_score.total
        accept = delta >= 0
        if not accept:
            scaled = delta / max(1.0, 2200.0 * temperature)
            accept = rng.random() < math.exp(max(-40.0, scaled))

        if accept:
            maze = candidate
            start, finish = cand_start, cand_finish
            current_score = cand_score
            if cand_score.total > best_score.total:
                best = candidate.copy()
                best_start, best_finish = cand_start, cand_finish
                best_score = cand_score

    final_score = score_maze(best, rng, weights, best_start, best_finish, loop_sample_size=240)
    return best, final_score, best_start, best_finish


def generate_braid_maze(
    width: int = 30,
    height: int = 30,
    seed: Optional[int] = None,
    attempts: int = 200,
    improve_iterations: int = 80,
    extra_edge_ratio: float = 0.11,
    crop_outer_perimeter: bool = True,
    progress: bool = True,
) -> Tuple[MazeGraph, Cell, Cell, ScoreDetails]:
    """Generate many candidates, optimize each, and keep the best maze."""

    rng = random.Random(seed)
    weights = ScoreWeights()
    best_maze: Optional[MazeGraph] = None
    best_start: Optional[Cell] = None
    best_finish: Optional[Cell] = None
    best_score: Optional[ScoreDetails] = None

    if progress:
        print_progress_bar(0, attempts)
    for attempt in range(1, attempts + 1):
        candidate = generate_sparse_connected_candidate(
            width,
            height,
            rng,
            extra_edge_ratio=extra_edge_ratio + rng.uniform(-0.025, 0.04),
        )
        maze, details, start, finish = optimize_maze(
            candidate,
            rng,
            weights,
            iterations=improve_iterations,
        )
        if best_score is None or details.total > best_score.total:
            best_maze = maze
            best_start = start
            best_finish = finish
            best_score = details
        if progress:
            print_progress_bar(attempt, attempts)

    assert best_maze is not None
    assert best_start is not None
    assert best_finish is not None
    assert best_score is not None

    if crop_outer_perimeter:
        best_maze = crop_outer_perimeter_maze(best_maze)
        reconnect_components(best_maze, rng)
        best_start, best_finish, _ = farthest_pair_by_sampling(best_maze, rng, samples=40)

    finalize_playable_maze(best_maze, rng)
    best_start, best_finish, _ = farthest_pair_by_sampling(best_maze, rng, samples=40)
    best_score = score_maze(
        best_maze,
        rng,
        weights,
        best_start,
        best_finish,
        loop_sample_size=240,
        expensive=True,
    )

    if not best_maze.is_connected():
        raise RuntimeError("internal error: final maze is disconnected")
    if any(best_maze.degree(cell) < 2 for cell in best_maze.cells()):
        raise RuntimeError("internal error: final maze still has dead ends")
    if count_2x2_squares(best_maze) != 0:
        raise RuntimeError("internal error: final maze still has 2x2 squares")
    return best_maze, best_start, best_finish, best_score


def finalize_playable_maze(maze: MazeGraph, rng: random.Random) -> None:
    """Enforce final display constraints after crop/cleanup side effects."""

    for _ in range(maze.cell_count * 4):
        reconnect_components(maze, rng)
        repair_dead_ends(maze, rng)
        removed = remove_all_2x2_squares(maze, rng)
        if (
            removed == 0
            and maze.is_connected()
            and all(maze.degree(cell) >= 2 for cell in maze.cells())
            and count_2x2_squares(maze) == 0
        ):
            return
    raise RuntimeError("could not finalize maze without dead ends and 2x2 squares")


def count_2x2_squares(maze: MazeGraph) -> int:
    count = 0
    for y in range(maze.height - 1):
        for x in range(maze.width - 1):
            if is_open_2x2_square(maze, x, y):
                count += 1
    return count


def is_open_2x2_square(maze: MazeGraph, x: int, y: int) -> bool:
    return (
        maze.has_edge((x, y), (x + 1, y))
        and maze.has_edge((x, y + 1), (x + 1, y + 1))
        and maze.has_edge((x, y), (x, y + 1))
        and maze.has_edge((x + 1, y), (x + 1, y + 1))
    )


def remove_all_2x2_squares(maze: MazeGraph, rng: random.Random) -> int:
    """
    Break every fully open 2x2 perimeter loop.

    Removing one perimeter edge from a 2x2 square still leaves those endpoints
    connected through the other three sides, so global connectivity is preserved.
    """

    removed = 0
    changed = True
    while changed:
        changed = False
        for y in range(maze.height - 1):
            for x in range(maze.width - 1):
                square_edges = [
                    normalized_edge((x, y), (x + 1, y)),
                    normalized_edge((x, y + 1), (x + 1, y + 1)),
                    normalized_edge((x, y), (x, y + 1)),
                    normalized_edge((x + 1, y), (x + 1, y + 1)),
                ]
                if not all(maze.has_edge(a, b) for a, b in square_edges):
                    continue

                square_cells = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
                for edge in square_edges:
                    if maze.degree(edge[0]) <= 2:
                        add_support_edge(maze, edge[0], square_cells, rng)
                    if maze.degree(edge[1]) <= 2:
                        add_support_edge(maze, edge[1], square_cells, rng)

                edge = max(
                    square_edges,
                    key=lambda e: (
                        maze.degree(e[0]) + maze.degree(e[1]),
                        rng.random(),
                    ),
                )
                maze.remove_edge(*edge)
                removed += 1
                changed = True
    return removed


def add_support_edge(
    maze: MazeGraph,
    cell: Cell,
    square_cells: Set[Cell],
    rng: random.Random,
) -> bool:
    """Add an edge near a square cell so breaking the square will not make a dead end."""

    options = []
    for n in maze.grid_neighbors(cell):
        if maze.has_edge(cell, n):
            continue
        outside_bonus = 1 if n not in square_cells else 0
        square_risk = local_rectangle_risk(maze, normalized_edge(cell, n))
        options.append((outside_bonus, -square_risk, rng.random(), n))

    if not options:
        return False

    _, _, _, n = max(options)
    maze.add_edge(cell, n)
    return True


def print_progress_bar(done: int, total: int, width: int = 34) -> None:
    ratio = 1.0 if total <= 0 else done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if done >= total else ""
    print(f"\rGenerating maze [{bar}] {ratio * 100:6.2f}%", end=end, flush=True)


def crop_outer_perimeter_maze(maze: MazeGraph) -> MazeGraph:
    """
    Remove the outer ring of cells and shift remaining cells to start at (0, 0).

    This intentionally happens after scoring/selecting the best full maze. It
    can create dead ends in the final playable maze, as requested.
    """

    if maze.width <= 2 or maze.height <= 2:
        raise ValueError("maze must be at least 3x3 to crop the outer perimeter")

    cropped = MazeGraph(maze.width - 2, maze.height - 2)
    for (ax, ay), (bx, by) in maze.edges():
        if 0 < ax < maze.width - 1 and 0 < ay < maze.height - 1:
            if 0 < bx < maze.width - 1 and 0 < by < maze.height - 1:
                cropped.add_edge((ax - 1, ay - 1), (bx - 1, by - 1))
    return cropped


def reconnect_components(maze: MazeGraph, rng: random.Random) -> None:
    """
    Reconnect components after perimeter cropping so the GUI remains playable.

    The crop itself may cut away a bridge that used the outer ring. This helper
    opens the smallest number of internal walls needed to restore connectivity.
    """

    while maze.component_count() > 1:
        components = connected_components(maze)
        comp_index: Dict[Cell, int] = {}
        for i, comp in enumerate(components):
            for cell in comp:
                comp_index[cell] = i

        bridges: List[Edge] = []
        for cell in maze.cells():
            for n in maze.grid_neighbors(cell):
                if cell < n and comp_index[cell] != comp_index[n]:
                    bridges.append((cell, n))

        if not bridges:
            raise RuntimeError("could not reconnect cropped maze")

        # Prefer links involving lower-degree cells. A touch of randomness keeps
        # the repaired crop from becoming too regular.
        bridge = min(
            bridges,
            key=lambda edge: maze.degree(edge[0]) + maze.degree(edge[1]) + rng.random(),
        )
        maze.add_edge(*bridge)


def connected_components(maze: MazeGraph) -> List[Set[Cell]]:
    components: List[Set[Cell]] = []
    seen: Set[Cell] = set()
    for start in maze.cells():
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for n in maze.adj[cur]:
                if n not in seen:
                    seen.add(n)
                    comp.add(n)
                    queue.append(n)
        components.append(comp)
    return components


def maze_to_ascii(maze: MazeGraph, start: Cell, finish: Cell) -> str:
    """
    Render as classic ASCII walls.

    Cells are displayed at odd coordinates. A passage is carved between adjacent
    cells. S and F mark start and finish.
    """

    rows = [["#" for _ in range(maze.width * 2 + 1)] for _ in range(maze.height * 2 + 1)]
    for x, y in maze.cells():
        rows[2 * y + 1][2 * x + 1] = " "
    for a, b in maze.edges():
        ax, ay = a
        bx, by = b
        rows[ay + by + 1][ax + bx + 1] = " "

    fx, fy = finish
    rows[2 * fy + 1][2 * fx + 1] = "F"
    return "\n".join("".join(row) for row in rows)


def maze_to_ascii_with_player(
    maze: MazeGraph,
    start: Cell,
    finish: Cell,
    player: Cell,
) -> str:
    """Render the maze with @ as the current player location."""

    rows = [list(row) for row in maze_to_ascii(maze, start, finish).splitlines()]
    px, py = player
    rows[2 * py + 1][2 * px + 1] = "@"
    return "\n".join("".join(row) for row in rows)


def play_maze(maze: MazeGraph, start: Cell, finish: Cell) -> None:
    """
    Simple terminal play mode.

    Use WASD to move through open graph edges. Press Q to quit.
    """

    player = start
    moves = {
        "w": (0, -1),
        "a": (-1, 0),
        "s": (0, 1),
        "d": (1, 0),
    }

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print(maze_to_ascii_with_player(maze, start, finish, player))
        print()
        print("Move with W/A/S/D. Q quits. @ is you, F is the finish.")

        if player == finish:
            print("You reached the finish!")
            return

        choice = input("> ").strip().lower()[:1]
        if choice == "q":
            return
        if choice not in moves:
            continue

        dx, dy = moves[choice]
        nxt = (player[0] + dx, player[1] + dy)
        if nxt in maze.adj[player]:
            player = nxt


class MazeGui:
    """Small Tkinter GUI for playing the generated maze."""

    def __init__(self, maze: MazeGraph, start: Cell, finish: Cell, cell_size: int = 20) -> None:
        self.maze = maze
        self.start = start
        self.finish = finish
        self.player = start
        self.base_cell_size = cell_size
        self.cell_size = float(cell_size)
        self.margin = 18
        self.wall_width = 3
        self.player_item: Optional[int] = None
        self.discovered: Set[Cell] = set()
        self.visible: Set[Cell] = set()

        self.canvas_width = 760
        self.canvas_height = 760
        self.view_x = 0.0
        self.view_y = 0.0

        self.root = tk.Tk()
        self.root.title("Braid Maze")
        self.root.configure(bg="#2f3030")
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)
        self.root.after(700, lambda: self.root.attributes("-topmost", False))

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#333536",
            highlightthickness=0,
        )
        self.canvas.pack(padx=14, pady=(0, 14))

        self.root.bind("<KeyPress>", self.on_key)
        self.update_visibility()
        self.draw_scene()

    def cell_center(self, cell: Cell) -> Tuple[float, float]:
        x, y = cell
        return (
            self.view_x + x * self.cell_size + self.cell_size / 2,
            self.view_y + y * self.cell_size + self.cell_size / 2,
        )

    def cell_box(self, cell: Cell, pad: int = 4) -> Tuple[float, float, float, float]:
        cx, cy = self.cell_center(cell)
        r = max(3.0, self.cell_size / 2 - pad)
        return (cx - r, cy - r, cx + r, cy + r)

    def update_visibility(self) -> None:
        """Reveal the player's current cell plus straight line-of-sight corridors."""

        visible = {self.player}
        for dx, dy in DIRS:
            cur = self.player
            while True:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt not in self.maze.adj[cur]:
                    break
                visible.add(nxt)
                cur = nxt
        self.visible = visible
        self.discovered.update(visible)
        self.update_zoom()

    def update_zoom(self) -> None:
        """Fit the entire discovered area into the frame."""

        fit_size = min(
            (self.canvas_width - self.margin * 2) / self.maze.width,
            (self.canvas_height - self.margin * 2) / self.maze.height,
        )
        if not self.discovered:
            self.cell_size = fit_size
            self.view_x = self.margin
            self.view_y = self.margin
            return

        min_x = min(x for x, _ in self.discovered)
        max_x = max(x for x, _ in self.discovered)
        min_y = min(y for _, y in self.discovered)
        max_y = max(y for _, y in self.discovered)
        known_w = max_x - min_x + 1
        known_h = max_y - min_y + 1

        self.cell_size = min(
            (self.canvas_width - self.margin * 2) / known_w,
            (self.canvas_height - self.margin * 2) / known_h,
        )

        known_px_w = known_w * self.cell_size
        known_px_h = known_h * self.cell_size
        self.view_x = (self.canvas_width - known_px_w) / 2 - min_x * self.cell_size
        self.view_y = (self.canvas_height - known_px_h) / 2 - min_y * self.cell_size

    def draw_scene(self) -> None:
        """Draw remembered cells dimly and current line-of-sight brightly."""

        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0,
            0,
            self.canvas_width,
            self.canvas_height,
            fill="#333536",
            outline="",
        )
        self.draw_floor()
        self.draw_passages()
        self.draw_walls()
        self.draw_markers()
        self.draw_player()

    def draw_player(self) -> None:
        """Create or move only the player marker."""

        box = self.cell_box(self.player, 3)
        self.player_item = self.canvas.create_oval(
            *box,
            fill="#3e4141",
            outline="#c7aa74",
            width=2,
            tags=("player",),
        )

    def draw_floor(self) -> None:
        for x, y in self.discovered:
            x0 = self.view_x + x * self.cell_size
            y0 = self.view_y + y * self.cell_size
            x1 = x0 + self.cell_size
            y1 = y0 + self.cell_size
            if (x, y) in self.visible:
                color = "#b9afa2" if (x + y) % 2 == 0 else "#afa598"
            else:
                color = "#7b756d" if (x + y) % 2 == 0 else "#746e67"
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

    def draw_passages(self) -> None:
        for a, b in self.maze.edges():
            if a not in self.discovered or b not in self.discovered:
                continue
            ax, ay = self.cell_center(a)
            bx, by = self.cell_center(b)
            bright = a in self.visible and b in self.visible
            self.canvas.create_line(
                ax,
                ay,
                bx,
                by,
                fill="#aca293" if bright else "#837b71",
                width=max(4, int(self.cell_size // 2)),
                capstyle=tk.ROUND,
            )

    def draw_walls(self) -> None:
        w = max(2, int(self.wall_width * min(1.6, self.cell_size / self.base_cell_size)))
        s = self.cell_size

        for x, y in self.discovered:
            wall_color = "#4d4944" if (x, y) in self.visible else "#56514b"
            x0 = self.view_x + x * s
            y0 = self.view_y + y * s
            x1 = x0 + s
            y1 = y0 + s
            cell = (x, y)
            if y > 0 and (x, y - 1) in self.discovered and (x, y - 1) not in self.maze.adj[cell]:
                self.canvas.create_line(x0, y0, x1, y0, fill=wall_color, width=w)
            if x < self.maze.width - 1 and (x + 1, y) in self.discovered and (x + 1, y) not in self.maze.adj[cell]:
                self.canvas.create_line(x1, y0, x1, y1, fill=wall_color, width=w)
            if y < self.maze.height - 1 and (x, y + 1) in self.discovered and (x, y + 1) not in self.maze.adj[cell]:
                self.canvas.create_line(x0, y1, x1, y1, fill=wall_color, width=w)
            if x > 0 and (x - 1, y) in self.discovered and (x - 1, y) not in self.maze.adj[cell]:
                self.canvas.create_line(x0, y0, x0, y1, fill=wall_color, width=w)

    def draw_markers(self) -> None:
        if self.finish not in self.discovered:
            return
        self.canvas.create_oval(*self.cell_box(self.finish, 5), fill="#a86558", outline="")
        self.canvas.create_text(
            *self.cell_center(self.finish),
            text="F",
            fill="white",
            font=("Segoe UI", max(8, int(self.cell_size // 2)), "bold"),
        )

    def on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        moves = {
            "w": (0, -1),
            "up": (0, -1),
            "a": (-1, 0),
            "left": (-1, 0),
            "s": (0, 1),
            "down": (0, 1),
            "d": (1, 0),
            "right": (1, 0),
        }
        if key in ("q", "escape"):
            self.root.destroy()
            return
        if key not in moves:
            return

        dx, dy = moves[key]
        nxt = (self.player[0] + dx, self.player[1] + dy)
        if nxt in self.maze.adj[self.player]:
            self.player = nxt
            self.update_visibility()
            self.draw_scene()
            if self.player == self.finish:
                messagebox.showinfo("Braid Maze", "You reached the finish!")

    def run(self) -> None:
        self.root.mainloop()


def validate_and_report(
    maze: MazeGraph,
    start: Cell,
    finish: Cell,
    details: ScoreDetails,
) -> str:
    stats_rng = random.Random(90210)
    avg_loop_size = average_alternate_loop_size(maze, stats_rng)
    rect_counts = narrow_rectangle_loop_counts(maze)
    rect_summary = ", ".join(f"{name}: {count}" for name, count in rect_counts.items())

    lines = [
        f"average loop size: {avg_loop_size:.2f}",
        f"2xN rectangular loops: {rect_summary}",
    ]
    return "\n".join(lines)


def export_graph_data(maze: MazeGraph, start: Cell, finish: Cell) -> Dict[str, object]:
    """Return data useful for game engines such as Unity."""

    return {
        "width": maze.width,
        "height": maze.height,
        "start": start,
        "finish": finish,
        "cells": [
            {
                "x": x,
                "y": y,
                "neighbors": sorted(maze.adj[(x, y)]),
            }
            for y in range(maze.height)
            for x in range(maze.width)
        ],
        "edges": maze.edges(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a scored 30x30 braid maze.")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument(
        "--extra-edge-ratio",
        type=float,
        default=0.11,
        help="Initial extra loop edges per cell before repair/optimization.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the generated maze in the terminal with WASD controls.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a Tkinter window and play with arrow keys or WASD.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open the GUI after generation.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=20,
        help="Pixel size of each maze cell in GUI mode.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not remove the outer perimeter after selecting the best maze.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide candidate-generation progress messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maze, start, finish, details = generate_braid_maze(
        width=args.width,
        height=args.height,
        seed=args.seed,
        attempts=args.attempts,
        improve_iterations=args.iterations,
        extra_edge_ratio=args.extra_edge_ratio,
        crop_outer_perimeter=not args.no_crop,
        progress=not args.quiet,
    )

    print(validate_and_report(maze, start, finish, details))

    if args.play:
        input("\nPress Enter to start play mode...")
        play_maze(maze, start, finish)

    open_gui = args.gui or (not args.no_gui and not args.play)
    if open_gui:
        print("Opening GUI...")
        try:
            MazeGui(maze, start, finish, cell_size=args.cell_size).run()
        except tk.TclError as exc:
            print(f"Could not open GUI: {exc}")


if __name__ == "__main__":
    main()
