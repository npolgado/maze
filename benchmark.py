#!/usr/bin/env python3
"""
Benchmark suite for maze generation quality and performance.

This suite compares `maze.py` and `braid_maze.py` against the goals in
`docs/benchmarking.md`:
  - hard constraints (connectivity, braid validity, tiny-loop suppression),
  - structural interest,
  - navigation/play metrics,
  - weighted quality score,
  - Pareto frontier (quality vs generation time),
  - golden-seed regression checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import csv
import html
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import maze as maze_impl

try:
    import braid_maze as braid_impl
    BRAID_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - only hit on missing tkinter/env issues
    braid_impl = None  # type: ignore[assignment]
    BRAID_IMPORT_ERROR = str(exc)


Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]

DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
DEFAULT_SIZES = [(12, 12), (20, 20), (30, 30)]


@dataclass
class MazeData:
    rows: int
    cols: int
    adj: Dict[Cell, Set[Cell]]
    start: Cell
    finish: Cell
    source_details: Dict[str, Any]


@dataclass
class RunResult:
    generator: str
    rows: int
    cols: int
    seed: int
    elapsed_ms: float
    success: bool
    hard_pass: bool
    quality_score: float
    error: Optional[str]
    metrics: Dict[str, Any]
    quality_components: Dict[str, float]
    source_details: Dict[str, Any]

    def key(self) -> str:
        return f"{self.generator}:{self.rows}x{self.cols}:{self.seed}"


@dataclass
class ReportDigest:
    path: Path
    generated_at: str
    run_count: int
    per_generator: Dict[str, Dict[str, float]]
    runs: List[Dict[str, Any]]


def normalize_edge(a: Cell, b: Cell) -> Edge:
    return (a, b) if a < b else (b, a)


def all_edges(adj: Dict[Cell, Set[Cell]]) -> List[Edge]:
    edges: List[Edge] = []
    for a, neighbors in adj.items():
        for b in neighbors:
            if a < b:
                edges.append((a, b))
    return edges


def has_edge(adj: Dict[Cell, Set[Cell]], a: Cell, b: Cell) -> bool:
    return b in adj.get(a, set())


def bfs_distances(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    blocked_edge: Optional[Edge] = None,
    banned_nodes: Optional[Set[Cell]] = None,
) -> Dict[Cell, int]:
    banned = banned_nodes or set()
    if start in banned:
        return {}

    blocked = normalize_edge(*blocked_edge) if blocked_edge else None
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt in banned:
                continue
            if blocked and normalize_edge(cur, nxt) == blocked:
                continue
            if nxt not in distances:
                distances[nxt] = distances[cur] + 1
                queue.append(nxt)
    return distances


def shortest_path(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    goal: Cell,
    banned_edges: Optional[Set[Edge]] = None,
    banned_nodes: Optional[Set[Cell]] = None,
) -> Optional[List[Cell]]:
    if start == goal:
        return [start]

    banned_e = banned_edges or set()
    banned_n = banned_nodes or set()
    if start in banned_n or goal in banned_n:
        return None

    queue = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while queue:
        cur = queue.popleft()
        for nxt in adj[cur]:
            if nxt in banned_n:
                continue
            if normalize_edge(cur, nxt) in banned_e:
                continue
            if nxt in parent:
                continue
            parent[nxt] = cur
            if nxt == goal:
                path = [goal]
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            queue.append(nxt)
    return None


def shortest_distance(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    goal: Cell,
    blocked_edge: Optional[Edge] = None,
) -> Optional[int]:
    distances = bfs_distances(adj, start, blocked_edge=blocked_edge)
    return distances.get(goal)


def path_edges(path: Sequence[Cell]) -> List[Edge]:
    return [normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]


def component_count(adj: Dict[Cell, Set[Cell]]) -> int:
    seen: Set[Cell] = set()
    count = 0
    for start in adj:
        if start in seen:
            continue
        count += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            cur = queue.popleft()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return count


def degree_entropy(degrees: Dict[Cell, int]) -> Tuple[float, float]:
    if not degrees:
        return 0.0, 0.0
    counts: Dict[int, int] = {}
    for d in degrees.values():
        counts[d] = counts.get(d, 0) + 1
    total = len(degrees)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(5.0)  # possible degrees in orthogonal grid are 0..4
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return entropy, max(0.0, min(1.0, normalized))


def rectangle_perimeter_open(
    adj: Dict[Cell, Set[Cell]],
    r0: int,
    c0: int,
    rh: int,
    rw: int,
) -> bool:
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1

    for c in range(c0, c1):
        if not has_edge(adj, (r0, c), (r0, c + 1)):
            return False
        if not has_edge(adj, (r1, c), (r1, c + 1)):
            return False
    for r in range(r0, r1):
        if not has_edge(adj, (r, c0), (r + 1, c0)):
            return False
        if not has_edge(adj, (r, c1), (r + 1, c1)):
            return False
    return True


def rectangle_internal_edges(
    adj: Dict[Cell, Set[Cell]],
    r0: int,
    c0: int,
    rh: int,
    rw: int,
) -> int:
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    total = 0

    for r in range(r0 + 1, r1):
        for c in range(c0, c1):
            if has_edge(adj, (r, c), (r, c + 1)):
                total += 1
    for c in range(c0 + 1, c1):
        for r in range(r0, r1):
            if has_edge(adj, (r, c), (r + 1, c)):
                total += 1
    return total


def count_2x2_squares(rows: int, cols: int, adj: Dict[Cell, Set[Cell]]) -> int:
    count = 0
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (
                has_edge(adj, (r, c), (r, c + 1))
                and has_edge(adj, (r + 1, c), (r + 1, c + 1))
                and has_edge(adj, (r, c), (r + 1, c))
                and has_edge(adj, (r, c + 1), (r + 1, c + 1))
            ):
                count += 1
    return count


def narrow_rectangle_loop_counts(
    rows: int,
    cols: int,
    adj: Dict[Cell, Set[Cell]],
    max_long_side: int = 12,
) -> Dict[str, int]:
    counts = {f"2x{n}": 0 for n in range(2, max_long_side + 1)}
    for n in range(2, max_long_side + 1):
        for rh, rw in ((2, n), (n, 2)):
            if rh > rows or rw > cols:
                continue
            label = f"2x{n}"
            for r0 in range(rows - rh + 1):
                for c0 in range(cols - rw + 1):
                    if rectangle_perimeter_open(adj, r0, c0, rh, rw):
                        internal = rectangle_internal_edges(adj, r0, c0, rh, rw)
                        if internal <= max(1, (rh * rw) // 8):
                            counts[label] += 1
    return counts


def straight_hallway_lengths(adj: Dict[Cell, Set[Cell]], degrees: Dict[Cell, int]) -> List[int]:
    lengths: List[int] = []
    seen_segments: Set[Edge] = set()

    for cell in adj:
        if degrees[cell] != 2:
            continue
        neighbors = list(adj[cell])
        horizontal = all(n[0] == cell[0] for n in neighbors)
        vertical = all(n[1] == cell[1] for n in neighbors)
        if not (horizontal or vertical):
            continue

        for n in neighbors:
            seg = normalize_edge(cell, n)
            if seg in seen_segments:
                continue

            step_r = 0 if n[0] == cell[0] else (1 if n[0] > cell[0] else -1)
            step_c = 0 if n[1] == cell[1] else (1 if n[1] > cell[1] else -1)
            run = [cell]
            prev = cell
            cur = n
            while True:
                seen_segments.add(normalize_edge(prev, cur))
                if degrees[cur] != 2:
                    break
                cur_neighbors = list(adj[cur])
                straight = all(
                    (other[0] == cur[0] if step_r == 0 else other[1] == cur[1])
                    for other in cur_neighbors
                )
                if not straight:
                    break
                run.append(cur)
                nxt = (cur[0] + step_r, cur[1] + step_c)
                if nxt not in adj[cur]:
                    break
                prev, cur = cur, nxt

            if len(run) > 1:
                lengths.append(len(run))

    return lengths


def junction_corridor_lengths(adj: Dict[Cell, Set[Cell]], degrees: Dict[Cell, int]) -> List[int]:
    junctions = {cell for cell, deg in degrees.items() if deg >= 3}
    if not junctions:
        return []

    seen_segments: Set[Edge] = set()
    lengths: List[int] = []

    for junction in junctions:
        for nxt in adj[junction]:
            segment = normalize_edge(junction, nxt)
            if segment in seen_segments:
                continue

            prev = junction
            cur = nxt
            length = 1

            while True:
                seen_segments.add(normalize_edge(prev, cur))
                if cur in junctions:
                    if cur != junction:
                        lengths.append(length)
                    break
                if degrees[cur] != 2:
                    break
                options = [x for x in adj[cur] if x != prev]
                if not options:
                    break
                prev, cur = cur, options[0]
                length += 1

    return lengths


def loop_depth_distribution(
    adj: Dict[Cell, Set[Cell]],
    rng: random.Random,
    sample_size: int = 240,
) -> Dict[str, Any]:
    edges = all_edges(adj)
    if not edges:
        return {
            "sample_size": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "short_loops_le5": 0,
            "raw_lengths": [],
        }

    sample = rng.sample(edges, min(sample_size, len(edges)))
    lengths: List[int] = []
    for edge in sample:
        alt = shortest_distance(adj, edge[0], edge[1], blocked_edge=edge)
        if alt is not None:
            lengths.append(alt)

    if not lengths:
        return {
            "sample_size": len(sample),
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "short_loops_le5": len(sample),
            "raw_lengths": [],
        }

    sorted_lengths = sorted(lengths)
    short_loops = sum(1 for v in lengths if v <= 5)
    return {
        "sample_size": len(sample),
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "p10": percentile(sorted_lengths, 10),
        "p90": percentile(sorted_lengths, 90),
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "short_loops_le5": short_loops,
        "raw_lengths": lengths,
    }


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(0.0, min(100.0, p)) / 100.0 * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def sample_pair_distances(
    adj: Dict[Cell, Set[Cell]],
    rng: random.Random,
    start_samples: int = 24,
    goals_per_start: int = 12,
) -> List[int]:
    cells = list(adj.keys())
    if len(cells) <= 1:
        return []
    starts = rng.sample(cells, min(start_samples, len(cells)))
    distances: List[int] = []
    for start in starts:
        dmap = bfs_distances(adj, start)
        vals = [d for cell, d in dmap.items() if cell != start]
        if not vals:
            continue
        if len(vals) <= goals_per_start:
            distances.extend(vals)
        else:
            distances.extend(rng.sample(vals, goals_per_start))
    return distances


def k_shortest_paths(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    finish: Cell,
    k: int = 3,
) -> List[List[Cell]]:
    first = shortest_path(adj, start, finish)
    if first is None:
        return []

    shortest_paths: List[List[Cell]] = [first]
    candidates: List[Tuple[int, int, List[Cell]]] = []
    candidate_seen: Set[Tuple[Cell, ...]] = set()
    tie = itertools.count()

    for _ in range(1, k):
        prev_path = shortest_paths[-1]
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            banned_edges: Set[Edge] = set()
            for chosen in shortest_paths:
                if len(chosen) > i and chosen[: i + 1] == root_path:
                    banned_edges.add(normalize_edge(chosen[i], chosen[i + 1]))

            banned_nodes = set(root_path[:-1])
            spur_path = shortest_path(
                adj,
                spur_node,
                finish,
                banned_edges=banned_edges,
                banned_nodes=banned_nodes,
            )
            if spur_path is None:
                continue

            total_path = root_path[:-1] + spur_path
            key = tuple(total_path)
            if key in candidate_seen:
                continue
            candidate_seen.add(key)
            heapq.heappush(candidates, (len(total_path) - 1, next(tie), total_path))

        if not candidates:
            break

        _, _, next_path = heapq.heappop(candidates)
        shortest_paths.append(next_path)

    return shortest_paths


def edge_disjoint_path_count(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    finish: Cell,
    max_paths: int = 4,
) -> int:
    working = {cell: set(neighbors) for cell, neighbors in adj.items()}
    count = 0

    for _ in range(max_paths):
        path = shortest_path(working, start, finish)
        if path is None or len(path) < 2:
            break
        count += 1
        for a, b in path_edges(path):
            if b in working[a]:
                working[a].remove(b)
            if a in working[b]:
                working[b].remove(a)

    return count


def limited_memory_agent_overhead(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    finish: Cell,
    rng: random.Random,
    memory_window: int = 8,
    trials: int = 6,
) -> Dict[str, float]:
    optimal = shortest_distance(adj, start, finish)
    if optimal is None:
        return {
            "optimal_steps": -1.0,
            "mean_steps": -1.0,
            "overhead_ratio": 99.0,
            "failure_rate": 1.0,
        }
    if optimal == 0:
        return {
            "optimal_steps": 0.0,
            "mean_steps": 0.0,
            "overhead_ratio": 1.0,
            "failure_rate": 0.0,
        }

    dist_to_finish = bfs_distances(adj, finish)
    max_steps = max(150, len(adj) * 10)
    steps_taken: List[int] = []
    failures = 0

    for _ in range(trials):
        cur = start
        prev: Optional[Cell] = None
        recent = deque([start], maxlen=memory_window)
        steps = 0

        while steps < max_steps and cur != finish:
            neighbors = list(adj[cur])
            if not neighbors:
                break

            current_dist = dist_to_finish.get(cur, len(adj) * 2)
            scored: List[Tuple[float, Cell]] = []
            for nxt in neighbors:
                nxt_dist = dist_to_finish.get(nxt, len(adj) * 2)
                score = 0.0
                if prev is not None and nxt == prev:
                    score -= 1.8
                if nxt in recent:
                    score -= 0.9
                if nxt_dist < current_dist:
                    score += 2.2
                elif nxt_dist > current_dist:
                    score -= 0.7
                score += rng.random() * 0.6
                scored.append((score, nxt))

            _, nxt = max(scored, key=lambda item: item[0])
            prev, cur = cur, nxt
            recent.append(cur)
            steps += 1

        if cur == finish:
            steps_taken.append(steps)
        else:
            failures += 1
            steps_taken.append(max_steps)

    mean_steps = statistics.fmean(steps_taken) if steps_taken else float("inf")
    return {
        "optimal_steps": float(optimal),
        "mean_steps": float(mean_steps),
        "overhead_ratio": float(mean_steps / optimal) if optimal > 0 else 1.0,
        "failure_rate": failures / max(1, trials),
    }


def local_recoverability_cost(
    adj: Dict[Cell, Set[Cell]],
    degrees: Dict[Cell, int],
    start: Cell,
    finish: Cell,
) -> Dict[str, float]:
    dist_from_start = bfs_distances(adj, start)
    dist_to_finish = bfs_distances(adj, finish)
    sf_dist = dist_from_start.get(finish)
    if sf_dist is None:
        return {"avg_wrong_turn_extra_cost": 99.0, "sample_count": 0.0}

    extras: List[float] = []
    big_penalty = float(sf_dist + 12)

    for cell, deg in degrees.items():
        if deg < 3:
            continue
        ds = dist_from_start.get(cell)
        if ds is None:
            continue
        if ds + dist_to_finish.get(cell, sf_dist + 1) != sf_dist:
            continue

        good_neighbors = [
            nxt
            for nxt in adj[cell]
            if dist_to_finish.get(nxt) is not None
            and ds + 1 + dist_to_finish[nxt] == sf_dist
        ]
        if not good_neighbors:
            continue

        best_good = min(1 + dist_to_finish[nxt] for nxt in good_neighbors)
        wrong_neighbors = [nxt for nxt in adj[cell] if nxt not in good_neighbors]
        for wrong in wrong_neighbors:
            wrong_cost = 1 + dist_to_finish.get(wrong, big_penalty)
            extras.append(max(0.0, float(wrong_cost - best_good)))

    if not extras:
        return {"avg_wrong_turn_extra_cost": 0.0, "sample_count": 0.0}
    return {
        "avg_wrong_turn_extra_cost": float(statistics.fmean(extras)),
        "sample_count": float(len(extras)),
    }


def exploration_coverage_curve(
    adj: Dict[Cell, Set[Cell]],
    start: Cell,
    rng: random.Random,
) -> Dict[str, Any]:
    total_cells = len(adj)
    if total_cells == 0:
        return {
            "steps": 0,
            "coverage_auc": 0.0,
            "coverage_at_25pct": 0.0,
            "coverage_at_50pct": 0.0,
            "coverage_at_100pct": 0.0,
            "coverage_at_200pct": 0.0,
        }

    horizon = max(100, min(total_cells * 4, 4000))
    visited: Set[Cell] = {start}
    recent = deque([start], maxlen=10)
    cur = start
    prev: Optional[Cell] = None
    coverage = [1.0 / total_cells]

    for _ in range(horizon):
        neighbors = list(adj[cur])
        if not neighbors:
            coverage.append(coverage[-1])
            continue

        scored: List[Tuple[float, Cell]] = []
        for nxt in neighbors:
            score = 0.0
            if nxt not in visited:
                score += 2.0
            if prev is not None and nxt == prev:
                score -= 0.8
            if nxt in recent:
                score -= 0.45
            score += rng.random() * 0.4
            scored.append((score, nxt))
        _, nxt = max(scored, key=lambda item: item[0])

        prev, cur = cur, nxt
        visited.add(cur)
        recent.append(cur)
        coverage.append(len(visited) / total_cells)

    auc = 0.0
    for i in range(len(coverage) - 1):
        auc += (coverage[i] + coverage[i + 1]) * 0.5
    auc /= max(1, len(coverage) - 1)

    def at_step(step: int) -> float:
        idx = min(step, len(coverage) - 1)
        return float(coverage[idx])

    return {
        "steps": horizon,
        "coverage_auc": float(auc),
        "coverage_at_25pct": at_step(total_cells // 4),
        "coverage_at_50pct": at_step(total_cells // 2),
        "coverage_at_100pct": at_step(total_cells),
        "coverage_at_200pct": at_step(total_cells * 2),
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_quality_score(metrics: Dict[str, Any], rows: int, cols: int) -> Tuple[float, Dict[str, float]]:
    hard = metrics["hard_constraints"]
    structural = metrics["structural"]
    navigation = metrics["navigation"]
    cell_count = max(1, rows * cols)

    loop_depth = structural["loop_depth"]
    loop_mean = float(loop_depth["mean"])
    loop_spread = float(loop_depth["p90"] - loop_depth["p10"])
    loop_score = 0.7 * clamp01((loop_mean - 4.0) / 16.0) + 0.3 * clamp01((loop_spread - 2.0) / 18.0)

    longest_hallway = float(structural["corridor"]["longest_straight_hallway"])
    over_threshold = float(structural["corridor"]["hallways_over_threshold"])
    corridor_score = 1.0
    corridor_score -= 0.65 * clamp01((longest_hallway - 6.0) / 14.0)
    corridor_score -= 0.35 * clamp01(over_threshold / max(3.0, cell_count / 50.0))
    corridor_score = clamp01(corridor_score)

    decision_density = float(structural["decision"]["junctions_per_100_cells"])
    cadence = structural["decision"]["decision_cadence"]
    density_score = 1.0 - min(1.0, abs(decision_density - 24.0) / 18.0)
    cadence_score = 0.4 if cadence is None else (1.0 - min(1.0, abs(float(cadence) - 5.5) / 4.5))
    decision_score = clamp01(0.55 * density_score + 0.45 * cadence_score)

    degree_score = clamp01(float(structural["degree_entropy"]["normalized"]))

    sf_len = float(navigation["start_finish"]["shortest_length"])
    sf_percentile = float(navigation["start_finish"]["length_percentile"])
    sf_len_norm = sf_len / max(1.0, float(rows + cols - 2))
    sf_score = 0.6 * clamp01((sf_percentile - 50.0) / 45.0) + 0.4 * clamp01((sf_len_norm - 1.2) / 2.5)

    edge_disjoint = float(navigation["redundancy"]["edge_disjoint_paths"])
    near_disjoint = float(navigation["redundancy"]["near_disjoint_count"])
    redundancy_score = 0.7 * clamp01((edge_disjoint - 1.0) / 2.0) + 0.3 * clamp01(near_disjoint / 2.0)

    detour2 = navigation["detour_factor"]["second_best_ratio"]
    detour3 = navigation["detour_factor"]["third_best_ratio"]
    detour_values = []
    if detour2 is not None:
        detour_values.append(clamp01((float(detour2) - 1.05) / 1.4))
    if detour3 is not None:
        detour_values.append(clamp01((float(detour3) - 1.05) / 1.4))
    detour_score = statistics.fmean(detour_values) if detour_values else 0.0

    backtrack = navigation["backtrack"]
    overhead = float(backtrack["overhead_ratio"])
    failure_rate = float(backtrack["failure_rate"])
    backtrack_score = 1.0 - clamp01((overhead - 1.0) / 2.2)
    backtrack_score *= 1.0 - 0.35 * failure_rate

    recover = navigation["recoverability"]
    recover_extra = float(recover["avg_wrong_turn_extra_cost"])
    recover_score = 1.0 - clamp01(recover_extra / 16.0)

    explore_auc = float(navigation["exploration"]["coverage_auc"])
    exploration_score = clamp01(explore_auc)

    tiny_loop_count = float(hard["count_2x2"] * 3 + hard["narrow_2xn_total"])
    tiny_loop_score = 1.0 - clamp01(tiny_loop_count / max(2.0, cell_count / 15.0))

    hard_pass = float(hard["hard_pass"])

    components = {
        "hard_constraints": hard_pass,
        "loop_depth": loop_score,
        "corridor_monotony": corridor_score,
        "decision_structure": decision_score,
        "degree_entropy": degree_score,
        "start_finish": sf_score,
        "redundancy": redundancy_score,
        "detour_factor": detour_score,
        "backtrack_pressure": backtrack_score,
        "local_recoverability": recover_score,
        "exploration_pace": exploration_score,
        "tiny_loop_suppression": tiny_loop_score,
    }

    weights = {
        "hard_constraints": 0.18,
        "loop_depth": 0.12,
        "corridor_monotony": 0.07,
        "decision_structure": 0.08,
        "degree_entropy": 0.06,
        "start_finish": 0.11,
        "redundancy": 0.10,
        "detour_factor": 0.06,
        "backtrack_pressure": 0.08,
        "local_recoverability": 0.05,
        "exploration_pace": 0.06,
        "tiny_loop_suppression": 0.03,
    }

    base = sum(components[name] * weights[name] for name in weights)
    score = base * 100.0

    if not hard["connected"]:
        score -= 35.0
    min_degree = float(hard["min_degree"])
    if min_degree < 2:
        score -= 25.0 * (2.0 - min_degree)
    score -= min(25.0, float(hard["dead_ends"]) * 4.0)
    score -= min(20.0, float(hard["count_2x2"]) * 6.0)
    score -= min(15.0, float(hard["narrow_2xn_total"]) * 1.5)

    return max(0.0, min(100.0, score)), components


def adapt_maze_py(maze_obj: maze_impl.Maze) -> MazeData:
    rows = maze_obj.rows
    cols = maze_obj.cols
    adj: Dict[Cell, Set[Cell]] = {(r, c): set() for r in range(rows) for c in range(cols)}

    for r in range(rows):
        for c in range(cols):
            for d in maze_obj.grid[r][c].passages:
                dr, dc = maze_impl.DIRS[d]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[(r, c)].add((nr, nc))

    return MazeData(
        rows=rows,
        cols=cols,
        adj=adj,
        start=maze_obj.start,
        finish=maze_obj.finish,
        source_details={"maze_py_stats": maze_obj.stats},
    )


def adapt_braid_maze(
    maze_obj: Any,
    start: Tuple[int, int],
    finish: Tuple[int, int],
    details: Any,
) -> MazeData:
    rows = maze_obj.height
    cols = maze_obj.width
    adj: Dict[Cell, Set[Cell]] = {(r, c): set() for r in range(rows) for c in range(cols)}

    for (x, y), neighbors in maze_obj.adj.items():
        here = (y, x)
        for nx, ny in neighbors:
            adj[here].add((ny, nx))

    source_details: Dict[str, Any] = {}
    if details is not None:
        source_details["braid_score_details"] = {
            "total": details.total,
            "connected": details.connected,
            "components": details.components,
            "dead_ends": details.dead_ends,
            "loop_depth_score": details.loop_depth_score,
            "small_rectangle_estimate": details.small_rectangle_estimate,
            "longest_straight_hallway": details.longest_straight_hallway,
            "shortest_start_finish": details.shortest_start_finish,
            "start_finish_redundancy": details.start_finish_redundancy,
        }

    return MazeData(
        rows=rows,
        cols=cols,
        adj=adj,
        start=(start[1], start[0]),
        finish=(finish[1], finish[0]),
        source_details=source_details,
    )


def compute_metrics(maze_data: MazeData, rng: random.Random) -> Dict[str, Any]:
    rows = maze_data.rows
    cols = maze_data.cols
    adj = maze_data.adj
    start = maze_data.start
    finish = maze_data.finish
    cell_count = len(adj)

    components = component_count(adj)
    connected = components == 1
    degrees = {cell: len(neighbors) for cell, neighbors in adj.items()}
    min_degree = min(degrees.values()) if degrees else 0
    dead_ends = sum(1 for d in degrees.values() if d == 1)
    isolated = sum(1 for d in degrees.values() if d == 0)

    count_2x2 = count_2x2_squares(rows, cols, adj)
    narrow_counts = narrow_rectangle_loop_counts(rows, cols, adj, max_long_side=min(12, max(rows, cols)))
    narrow_total = sum(count for name, count in narrow_counts.items() if name != "2x2")

    hard_pass = (
        connected
        and min_degree >= 2
        and dead_ends == 0
        and count_2x2 == 0
        and narrow_total == 0
    )

    loop_depth = loop_depth_distribution(adj, rng, sample_size=240)
    hallway_lengths = straight_hallway_lengths(adj, degrees)
    hall_threshold = 6
    hall_over = sum(1 for length in hallway_lengths if length > hall_threshold)
    junctions = [cell for cell, deg in degrees.items() if deg >= 3]
    junction_density = len(junctions) * 100.0 / max(1, cell_count)
    cadence_lengths = junction_corridor_lengths(adj, degrees)
    cadence = statistics.fmean(cadence_lengths) if cadence_lengths else None
    entropy_raw, entropy_norm = degree_entropy(degrees)

    sf_path = shortest_path(adj, start, finish)
    sf_len = (len(sf_path) - 1) if sf_path else float("inf")
    sampled_distances = sample_pair_distances(adj, rng)
    if sampled_distances and math.isfinite(sf_len):
        sf_percentile = 100.0 * sum(1 for d in sampled_distances if d <= sf_len) / len(sampled_distances)
    else:
        sf_percentile = 0.0

    k_paths = k_shortest_paths(adj, start, finish, k=3)
    base_path_edges = set(path_edges(k_paths[0])) if k_paths else set()
    edge_disjoint = edge_disjoint_path_count(adj, start, finish, max_paths=4)

    near_disjoint_count = 0
    overlap_ratios: List[float] = []
    second_ratio: Optional[float] = None
    third_ratio: Optional[float] = None
    if k_paths:
        shortest_len = max(1, len(k_paths[0]) - 1)
        if len(k_paths) >= 2:
            second_ratio = (len(k_paths[1]) - 1) / shortest_len
        if len(k_paths) >= 3:
            third_ratio = (len(k_paths[2]) - 1) / shortest_len

        for alt in k_paths[1:]:
            alt_edges = set(path_edges(alt))
            overlap = len(base_path_edges & alt_edges) / max(1, len(alt_edges))
            overlap_ratios.append(overlap)
            if overlap <= 0.35:
                near_disjoint_count += 1

    backtrack = limited_memory_agent_overhead(adj, start, finish, rng)
    recoverability = local_recoverability_cost(adj, degrees, start, finish)
    exploration = exploration_coverage_curve(adj, start, rng)

    return {
        "hard_constraints": {
            "connected": connected,
            "components": components,
            "min_degree": min_degree,
            "dead_ends": dead_ends,
            "isolated": isolated,
            "count_2x2": count_2x2,
            "narrow_2xn_total": narrow_total,
            "narrow_2xn_counts": narrow_counts,
            "hard_pass": hard_pass,
        },
        "structural": {
            "loop_depth": loop_depth,
            "corridor": {
                "longest_straight_hallway": max(hallway_lengths, default=0),
                "hallways_over_threshold": hall_over,
                "threshold": hall_threshold,
            },
            "decision": {
                "junction_count": len(junctions),
                "junctions_per_100_cells": junction_density,
                "decision_cadence": cadence,
                "cadence_sample_count": len(cadence_lengths),
            },
            "degree_entropy": {
                "raw": entropy_raw,
                "normalized": entropy_norm,
            },
        },
        "navigation": {
            "start_finish": {
                "shortest_length": sf_len if math.isfinite(sf_len) else -1,
                "length_percentile": sf_percentile,
                "distance_sample_count": len(sampled_distances),
            },
            "redundancy": {
                "edge_disjoint_paths": edge_disjoint,
                "near_disjoint_count": near_disjoint_count,
                "path_count_evaluated": len(k_paths),
                "overlap_ratios_vs_shortest": overlap_ratios,
            },
            "detour_factor": {
                "second_best_ratio": second_ratio,
                "third_best_ratio": third_ratio,
            },
            "backtrack": backtrack,
            "recoverability": recoverability,
            "exploration": exploration,
        },
    }


def run_maze_generator(rows: int, cols: int, seed: int, max_rect: int) -> MazeData:
    maze_obj = maze_impl.Maze(rows=rows, cols=cols, seed=seed, max_rect=max_rect)
    return adapt_maze_py(maze_obj)


def run_braid_generator(
    rows: int,
    cols: int,
    seed: int,
    attempts: int,
    iterations: int,
    extra_edge_ratio: float,
    crop_outer_perimeter: bool,
) -> MazeData:
    if braid_impl is None:
        raise RuntimeError(f"Could not import braid_maze.py: {BRAID_IMPORT_ERROR}")
    maze_obj, start, finish, details = braid_impl.generate_braid_maze(
        width=cols,
        height=rows,
        seed=seed,
        attempts=attempts,
        improve_iterations=iterations,
        extra_edge_ratio=extra_edge_ratio,
        crop_outer_perimeter=crop_outer_perimeter,
        progress=False,
    )
    return adapt_braid_maze(maze_obj, start, finish, details)


def run_one(
    generator: str,
    rows: int,
    cols: int,
    seed: int,
    args: argparse.Namespace,
) -> RunResult:
    start_time = time.perf_counter()
    success = False
    hard_pass = False
    quality = 0.0
    error: Optional[str] = None
    metrics: Dict[str, Any] = {}
    components: Dict[str, float] = {}
    source_details: Dict[str, Any] = {}

    try:
        if generator == "maze":
            maze_data = run_maze_generator(rows, cols, seed, max_rect=args.maze_max_rect)
        elif generator == "braid":
            maze_data = run_braid_generator(
                rows=rows,
                cols=cols,
                seed=seed,
                attempts=args.braid_attempts,
                iterations=args.braid_iterations,
                extra_edge_ratio=args.braid_extra_edge_ratio,
                crop_outer_perimeter=not args.braid_no_crop,
            )
        else:
            raise ValueError(f"unknown generator '{generator}'")

        metric_rng = random.Random((seed << 16) ^ (rows << 8) ^ cols ^ 0xBEE5)
        metrics = compute_metrics(maze_data, metric_rng)
        quality, components = compute_quality_score(metrics, maze_data.rows, maze_data.cols)
        success = True
        hard_pass = bool(metrics["hard_constraints"]["hard_pass"])
        source_details = maze_data.source_details
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return RunResult(
        generator=generator,
        rows=rows,
        cols=cols,
        seed=seed,
        elapsed_ms=elapsed_ms,
        success=success,
        hard_pass=hard_pass,
        quality_score=quality,
        error=error,
        metrics=metrics,
        quality_components=components,
        source_details=source_details,
    )


def parse_sizes(value: str) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for chunk in value.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if "x" in chunk:
            left, right = chunk.split("x", 1)
            rows = int(left.strip())
            cols = int(right.strip())
        else:
            rows = int(chunk)
            cols = rows
        if rows <= 1 or cols <= 1:
            raise ValueError(f"invalid maze size '{chunk}', must be >= 2x2")
        result.append((rows, cols))
    if not result:
        raise ValueError("no valid sizes parsed")
    return result


def parse_seeds(value: str) -> List[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("no valid seeds parsed")
    return seeds


def summarize(results: Sequence[RunResult]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"per_generator": {}, "overall": {}}
    by_gen: Dict[str, List[RunResult]] = {}
    for run in results:
        by_gen.setdefault(run.generator, []).append(run)

    for generator, runs in by_gen.items():
        successes = [r for r in runs if r.success]
        hard_passes = [r for r in runs if r.hard_pass]
        qualities = [r.quality_score for r in successes]
        times = [r.elapsed_ms for r in successes]
        summary["per_generator"][generator] = {
            "runs": len(runs),
            "successes": len(successes),
            "success_rate": len(successes) / max(1, len(runs)),
            "hard_passes": len(hard_passes),
            "hard_pass_rate": len(hard_passes) / max(1, len(runs)),
            "mean_quality": statistics.fmean(qualities) if qualities else 0.0,
            "median_quality": statistics.median(qualities) if qualities else 0.0,
            "mean_time_ms": statistics.fmean(times) if times else 0.0,
            "median_time_ms": statistics.median(times) if times else 0.0,
        }

    successes = [r for r in results if r.success]
    summary["overall"] = {
        "runs": len(results),
        "successes": len(successes),
        "success_rate": len(successes) / max(1, len(results)),
        "hard_passes": sum(1 for r in results if r.hard_pass),
        "hard_pass_rate": sum(1 for r in results if r.hard_pass) / max(1, len(results)),
    }
    return summary


def pareto_frontier_indices(results: Sequence[RunResult]) -> Set[int]:
    points = [(i, r.elapsed_ms, r.quality_score) for i, r in enumerate(results) if r.success]
    points.sort(key=lambda item: item[1])
    frontier: Set[int] = set()
    best_quality = -float("inf")
    for idx, _, quality in points:
        if quality > best_quality:
            frontier.add(idx)
            best_quality = quality
    return frontier


def render_ascii_pareto(results: Sequence[RunResult], frontier: Set[int], width: int = 64, height: int = 18) -> str:
    successful = [(i, r) for i, r in enumerate(results) if r.success]
    if not successful:
        return "No successful runs for Pareto chart."

    times = [r.elapsed_ms for _, r in successful]
    qualities = [r.quality_score for _, r in successful]
    min_t, max_t = min(times), max(times)
    min_q, max_q = min(qualities), max(qualities)
    span_t = max(1e-9, max_t - min_t)
    span_q = max(1e-9, max_q - min_q)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def symbol(generator: str, is_frontier: bool) -> str:
        if generator == "maze":
            return "M" if is_frontier else "m"
        if generator == "braid":
            return "B" if is_frontier else "b"
        return "X" if is_frontier else "x"

    for idx, run in successful:
        x = int((run.elapsed_ms - min_t) / span_t * (width - 1))
        y = int((run.quality_score - min_q) / span_q * (height - 1))
        row = height - 1 - y
        col = x
        is_frontier = idx in frontier
        ch = symbol(run.generator, is_frontier)

        current = grid[row][col]
        if current == " " or current.islower() and ch.isupper():
            grid[row][col] = ch

    lines = ["".join(row) for row in grid]
    lines.append(f"time ms: min={min_t:.1f}, max={max_t:.1f}")
    lines.append(f"quality: min={min_q:.2f}, max={max_q:.2f}")
    lines.append("legend: m=maze, b=braid, uppercase=pareto frontier")
    return "\n".join(lines)


def flatten_run(run: RunResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "generator": run.generator,
        "rows": run.rows,
        "cols": run.cols,
        "seed": run.seed,
        "elapsed_ms": round(run.elapsed_ms, 3),
        "success": run.success,
        "hard_pass": run.hard_pass,
        "quality_score": round(run.quality_score, 4),
        "error": run.error or "",
    }
    if not run.success:
        return row

    hard = run.metrics["hard_constraints"]
    structural = run.metrics["structural"]
    navigation = run.metrics["navigation"]
    row.update(
        {
            "connected": hard["connected"],
            "components": hard["components"],
            "min_degree": hard["min_degree"],
            "dead_ends": hard["dead_ends"],
            "count_2x2": hard["count_2x2"],
            "narrow_2xn_total": hard["narrow_2xn_total"],
            "loop_mean": round(structural["loop_depth"]["mean"], 4),
            "loop_p90": round(structural["loop_depth"]["p90"], 4),
            "longest_hallway": structural["corridor"]["longest_straight_hallway"],
            "hallways_over_threshold": structural["corridor"]["hallways_over_threshold"],
            "junctions_per_100_cells": round(structural["decision"]["junctions_per_100_cells"], 4),
            "decision_cadence": structural["decision"]["decision_cadence"],
            "degree_entropy_norm": round(structural["degree_entropy"]["normalized"], 4),
            "sf_shortest_length": navigation["start_finish"]["shortest_length"],
            "sf_length_percentile": round(navigation["start_finish"]["length_percentile"], 4),
            "edge_disjoint_paths": navigation["redundancy"]["edge_disjoint_paths"],
            "near_disjoint_count": navigation["redundancy"]["near_disjoint_count"],
            "detour_second_ratio": navigation["detour_factor"]["second_best_ratio"],
            "detour_third_ratio": navigation["detour_factor"]["third_best_ratio"],
            "backtrack_overhead": round(navigation["backtrack"]["overhead_ratio"], 4),
            "recoverability_cost": round(navigation["recoverability"]["avg_wrong_turn_extra_cost"], 4),
            "exploration_auc": round(navigation["exploration"]["coverage_auc"], 4),
        }
    )
    return row


def write_csv(path: Path, results: Sequence[RunResult]) -> None:
    rows = [flatten_run(run) for run in results]
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nSummary")
    print("-------")
    for generator, values in summary["per_generator"].items():
        print(
            f"{generator:>5}: runs={values['runs']}, success={values['success_rate']:.1%}, "
            f"hard-pass={values['hard_pass_rate']:.1%}, "
            f"mean quality={values['mean_quality']:.2f}, "
            f"mean time={values['mean_time_ms']:.1f}ms"
        )
    overall = summary["overall"]
    print(
        f"overall: runs={overall['runs']}, success={overall['success_rate']:.1%}, "
        f"hard-pass={overall['hard_pass_rate']:.1%}"
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ascii_bar(value: float, max_value: float, width: int = 16, fill: str = "#", empty: str = "-") -> str:
    if width <= 0:
        return ""
    if max_value <= 0:
        filled = 0
    else:
        ratio = max(0.0, min(1.0, value / max_value))
        filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return fill * filled + empty * (width - filled)


def summarize_external_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_gen: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        generator = str(run.get("generator", "unknown"))
        by_gen.setdefault(generator, []).append(run)

    summary: Dict[str, Dict[str, float]] = {}
    for generator, values in by_gen.items():
        successes = [r for r in values if bool(r.get("success", False))]
        hard_passes = [r for r in values if bool(r.get("hard_pass", False))]
        qualities = [_as_float(r.get("quality_score"), 0.0) for r in successes]
        times = [_as_float(r.get("elapsed_ms"), 0.0) for r in successes]
        summary[generator] = {
            "runs": float(len(values)),
            "success_rate": len(successes) / max(1, len(values)),
            "hard_pass_rate": len(hard_passes) / max(1, len(values)),
            "mean_quality": statistics.fmean(qualities) if qualities else 0.0,
            "mean_time_ms": statistics.fmean(times) if times else 0.0,
        }
    return summary


def load_report_digest(path: Path) -> Optional[ReportDigest]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    runs = payload.get("runs")
    if not isinstance(runs, list):
        return None

    generated_at = str(payload.get("generated_at", "unknown"))
    per_generator: Dict[str, Dict[str, float]] = {}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        raw_pg = summary.get("per_generator")
        if isinstance(raw_pg, dict):
            for generator, values in raw_pg.items():
                if not isinstance(values, dict):
                    continue
                per_generator[str(generator)] = {
                    "runs": _as_float(values.get("runs"), 0.0),
                    "success_rate": _as_float(values.get("success_rate"), 0.0),
                    "hard_pass_rate": _as_float(values.get("hard_pass_rate"), 0.0),
                    "mean_quality": _as_float(values.get("mean_quality"), 0.0),
                    "mean_time_ms": _as_float(values.get("mean_time_ms"), 0.0),
                }

    if not per_generator:
        per_generator = summarize_external_runs(runs)

    return ReportDigest(
        path=path,
        generated_at=generated_at,
        run_count=len(runs),
        per_generator=per_generator,
        runs=runs,
    )


def discover_report_digests(output_dir: Path, pattern: str, limit: int) -> List[ReportDigest]:
    if not output_dir.exists():
        return []

    candidates = [p for p in output_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".json"]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit > 0:
        candidates = candidates[:limit]
    candidates.reverse()

    digests: List[ReportDigest] = []
    for path in candidates:
        digest = load_report_digest(path)
        if digest is not None:
            digests.append(digest)
    return digests


def build_visual_summary_data(digests: Sequence[ReportDigest]) -> Dict[str, Any]:
    all_runs: List[Dict[str, Any]] = []
    for digest in digests:
        for run in digest.runs:
            generator = str(run.get("generator", "unknown"))
            all_runs.append(
                {
                    "report_name": digest.path.name,
                    "generated_at": digest.generated_at,
                    "generator": generator,
                    "rows": int(_as_float(run.get("rows"), 0.0)),
                    "cols": int(_as_float(run.get("cols"), 0.0)),
                    "seed": int(_as_float(run.get("seed"), 0.0)),
                    "success": bool(run.get("success", False)),
                    "hard_pass": bool(run.get("hard_pass", False)),
                    "quality_score": _as_float(run.get("quality_score"), 0.0),
                    "elapsed_ms": _as_float(run.get("elapsed_ms"), 0.0),
                }
            )

    aggregate = summarize_external_runs(all_runs)
    successful = [run for run in all_runs if run["success"]]
    best_runs = sorted(successful, key=lambda run: run["quality_score"], reverse=True)[:12]

    report_rows = []
    for digest in digests:
        report_rows.append(
            {
                "name": digest.path.name,
                "generated_at": digest.generated_at,
                "run_count": digest.run_count,
                "per_generator": digest.per_generator,
            }
        )

    return {
        "report_count": len(digests),
        "reports": report_rows,
        "all_runs": all_runs,
        "successful_runs": successful,
        "aggregate_per_generator": aggregate,
        "best_runs": best_runs,
    }


def pareto_frontier_points(points: Sequence[Dict[str, Any]]) -> Set[int]:
    indexed = [
        (i, _as_float(point["elapsed_ms"]), _as_float(point["quality_score"]))
        for i, point in enumerate(points)
    ]
    indexed.sort(key=lambda item: item[1])
    frontier: Set[int] = set()
    best_quality = -float("inf")
    for idx, _, quality in indexed:
        if quality > best_quality:
            frontier.add(idx)
            best_quality = quality
    return frontier


def render_ascii_pareto_points(
    points: Sequence[Dict[str, Any]],
    width: int = 64,
    height: int = 16,
) -> str:
    if not points:
        return "No successful runs for Pareto chart."

    times = [_as_float(point["elapsed_ms"]) for point in points]
    qualities = [_as_float(point["quality_score"]) for point in points]
    min_t, max_t = min(times), max(times)
    min_q, max_q = min(qualities), max(qualities)
    span_t = max(1e-9, max_t - min_t)
    span_q = max(1e-9, max_q - min_q)

    frontier = pareto_frontier_points(points)
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def symbol(generator: str, is_frontier: bool) -> str:
        if generator == "maze":
            return "M" if is_frontier else "m"
        if generator == "braid":
            return "B" if is_frontier else "b"
        return "X" if is_frontier else "x"

    for idx, point in enumerate(points):
        x = int((_as_float(point["elapsed_ms"]) - min_t) / span_t * (width - 1))
        y = int((_as_float(point["quality_score"]) - min_q) / span_q * (height - 1))
        row = height - 1 - y
        col = x
        ch = symbol(str(point["generator"]), idx in frontier)
        current = grid[row][col]
        if current == " " or current.islower() and ch.isupper():
            grid[row][col] = ch

    lines = ["".join(row) for row in grid]
    lines.append(f"time ms: min={min_t:.1f}, max={max_t:.1f}")
    lines.append(f"quality: min={min_q:.2f}, max={max_q:.2f}")
    lines.append("legend: m=maze, b=braid, uppercase=pareto frontier")
    return "\n".join(lines)


def render_terminal_visual_summary(visual: Dict[str, Any]) -> str:
    reports = visual["reports"]
    aggregate = visual["aggregate_per_generator"]
    successful = visual["successful_runs"]
    best_runs = visual["best_runs"]

    lines = []
    lines.append("\nVisual Summary")
    lines.append("--------------")
    lines.append(
        f"reports loaded={visual['report_count']}, total runs={len(visual['all_runs'])}, "
        f"successful runs={len(successful)}"
    )

    if not aggregate:
        lines.append("No benchmark reports found.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Aggregate By Generator")
    lines.append("----------------------")
    time_values = [values["mean_time_ms"] for values in aggregate.values()]
    max_time = max(time_values) if time_values else 1.0
    min_time = min(time_values) if time_values else 0.0
    speed_span = max(1e-9, max_time - min_time)

    for generator in sorted(aggregate.keys()):
        values = aggregate[generator]
        quality_bar = ascii_bar(values["mean_quality"], 100.0, width=18)
        speed_score = 1.0 if max_time == min_time else (max_time - values["mean_time_ms"]) / speed_span
        speed_bar = ascii_bar(speed_score, 1.0, width=12)
        hard_bar = ascii_bar(values["hard_pass_rate"], 1.0, width=12)
        lines.append(
            f"{generator:>5}  quality {values['mean_quality']:6.2f} [{quality_bar}]  "
            f"time {values['mean_time_ms']:8.1f}ms [{speed_bar}]  "
            f"hard {values['hard_pass_rate']*100:6.1f}% [{hard_bar}]"
        )

    lines.append("")
    lines.append("Per Report (latest)")
    lines.append("-------------------")
    for report in reports[-10:]:
        base = f"{report['name']:<28} runs={report['run_count']:<3}"
        chunks = []
        for generator in sorted(report["per_generator"].keys()):
            values = report["per_generator"][generator]
            chunks.append(
                f"{generator}:q={values['mean_quality']:.1f},t={values['mean_time_ms']:.1f}ms,h={values['hard_pass_rate']*100:.0f}%"
            )
        if not chunks:
            chunks.append("no generator summary")
        lines.append(base + " | " + " | ".join(chunks))

    lines.append("")
    lines.append("Top Runs By Quality")
    lines.append("-------------------")
    for run in best_runs[:8]:
        lines.append(
            f"{run['quality_score']:6.2f}  {run['generator']:>5}  {run['rows']}x{run['cols']}  "
            f"seed={run['seed']:<5}  time={run['elapsed_ms']:.1f}ms  report={run['report_name']}"
        )

    lines.append("")
    lines.append("Pareto Across Loaded Reports")
    lines.append("----------------------------")
    lines.append(render_ascii_pareto_points(successful))
    return "\n".join(lines)


def _generator_color(generator: str) -> str:
    if generator == "maze":
        return "#2d6cdf"
    if generator == "braid":
        return "#d35400"
    return "#444444"


def render_visual_summary_html(visual: Dict[str, Any]) -> str:
    reports = visual["reports"]
    aggregate = visual["aggregate_per_generator"]
    successful = visual["successful_runs"]
    best_runs = visual["best_runs"]
    generators = sorted(aggregate.keys())

    metric_cards = []
    for generator in generators:
        values = aggregate[generator]
        q = values["mean_quality"]
        t = values["mean_time_ms"]
        hp = values["hard_pass_rate"] * 100.0
        metric_cards.append(
            f"""
            <div class="card">
              <h3>{html.escape(generator)}</h3>
              <div class="metric">Mean quality <strong>{q:.2f}</strong></div>
              <div class="bar"><span style="width:{max(0.0, min(100.0, q)):.2f}%"></span></div>
              <div class="metric">Mean time <strong>{t:.1f} ms</strong></div>
              <div class="metric">Hard pass <strong>{hp:.1f}%</strong></div>
              <div class="bar"><span style="width:{max(0.0, min(100.0, hp)):.2f}%"></span></div>
            </div>
            """
        )

    report_headers = "".join(f"<th>{html.escape(generator)} (q/t/h%)</th>" for generator in generators)
    report_rows = []
    for report in reports:
        cells = []
        for generator in generators:
            values = report["per_generator"].get(generator)
            if values is None:
                cells.append("<td>-</td>")
            else:
                cells.append(
                    "<td>"
                    f"{values['mean_quality']:.1f} / {values['mean_time_ms']:.1f} / {values['hard_pass_rate']*100:.0f}"
                    "</td>"
                )
        report_rows.append(
            "<tr>"
            f"<td>{html.escape(report['name'])}</td>"
            f"<td>{html.escape(report['generated_at'])}</td>"
            f"<td>{report['run_count']}</td>"
            f"{''.join(cells)}"
            "</tr>"
        )

    if successful:
        w, h = 960, 340
        pad = 42
        times = [_as_float(point["elapsed_ms"]) for point in successful]
        qualities = [_as_float(point["quality_score"]) for point in successful]
        min_t, max_t = min(times), max(times)
        min_q, max_q = min(qualities), max(qualities)
        span_t = max(1e-9, max_t - min_t)
        span_q = max(1e-9, max_q - min_q)
        circles = []
        for point in successful:
            x = pad + (_as_float(point["elapsed_ms"]) - min_t) / span_t * (w - 2 * pad)
            y = h - pad - (_as_float(point["quality_score"]) - min_q) / span_q * (h - 2 * pad)
            title = (
                f"{point['generator']} {point['rows']}x{point['cols']} seed={point['seed']} "
                f"q={point['quality_score']:.2f} t={point['elapsed_ms']:.1f}ms ({point['report_name']})"
            )
            circles.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{_generator_color(point['generator'])}'>"
                f"<title>{html.escape(title)}</title>"
                "</circle>"
            )
        scatter_svg = (
            f"<svg viewBox='0 0 {w} {h}' class='scatter'>"
            f"<line x1='{pad}' y1='{h-pad}' x2='{w-pad}' y2='{h-pad}' class='axis' />"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{h-pad}' class='axis' />"
            f"{''.join(circles)}"
            f"<text x='{w/2:.1f}' y='{h-8}' text-anchor='middle' class='label'>generation time (ms)</text>"
            f"<text x='14' y='{h/2:.1f}' transform='rotate(-90 14,{h/2:.1f})' text-anchor='middle' class='label'>quality score</text>"
            "</svg>"
        )
    else:
        scatter_svg = "<p>No successful runs available for scatter plot.</p>"

    best_rows = []
    for run in best_runs[:20]:
        best_rows.append(
            "<tr>"
            f"<td>{run['quality_score']:.2f}</td>"
            f"<td>{run['generator']}</td>"
            f"<td>{run['rows']}x{run['cols']}</td>"
            f"<td>{run['seed']}</td>"
            f"<td>{run['elapsed_ms']:.1f}</td>"
            f"<td>{'yes' if run['hard_pass'] else 'no'}</td>"
            f"<td>{html.escape(run['report_name'])}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Visual Summary</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2a37;
      --muted: #5c6b7a;
      --line: #d8dee5;
      --accent: #2d6cdf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #eef3fb 0%, var(--bg) 45%);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 24px auto 48px;
      padding: 0 16px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
    }}
    .card h3 {{ margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.6px; font-size: 13px; color: var(--muted); }}
    .metric {{ margin: 5px 0; }}
    .bar {{
      height: 8px;
      background: #e8edf3;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 6px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #4a88f2, var(--accent));
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f2f6fb; }}
    .scatter {{
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .axis {{ stroke: #7a8795; stroke-width: 1; }}
    .label {{ fill: #5c6b7a; font-size: 12px; }}
    .legend {{
      display: flex;
      gap: 12px;
      margin: 8px 0 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Benchmark Visual Summary</h1>
    <div class="sub">Reports loaded: {visual['report_count']} | Total runs: {len(visual['all_runs'])} | Successful runs: {len(successful)}</div>

    <div class="grid">
      {''.join(metric_cards) if metric_cards else "<div class='card'>No generator data available.</div>"}
    </div>

    <div class="panel">
      <h2>Per Report Overview</h2>
      <table>
        <thead>
          <tr>
            <th>Report</th>
            <th>Generated</th>
            <th>Runs</th>
            {report_headers}
          </tr>
        </thead>
        <tbody>
          {''.join(report_rows) if report_rows else "<tr><td colspan='12'>No reports found.</td></tr>"}
        </tbody>
      </table>
    </div>

    <div class="panel">
      <h2>Quality vs Time (All Successful Runs)</h2>
      <div class="legend">
        <span class="chip"><span class="dot" style="background:#2d6cdf"></span>maze</span>
        <span class="chip"><span class="dot" style="background:#d35400"></span>braid</span>
      </div>
      {scatter_svg}
    </div>

    <div class="panel">
      <h2>Top Runs By Quality</h2>
      <table>
        <thead>
          <tr>
            <th>Quality</th>
            <th>Generator</th>
            <th>Size</th>
            <th>Seed</th>
            <th>Time (ms)</th>
            <th>Hard Pass</th>
            <th>Report</th>
          </tr>
        </thead>
        <tbody>
          {''.join(best_rows) if best_rows else "<tr><td colspan='7'>No successful runs found.</td></tr>"}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def write_visual_summary_file(
    output_dir: Path,
    visual: Dict[str, Any],
    visual_name: Optional[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = visual_name or f"visual_summary_{stamp}"
    if not base_name.lower().endswith(".html"):
        base_name = f"{base_name}.html"
    html_path = output_dir / base_name
    html_path.write_text(render_visual_summary_html(visual), encoding="utf-8")
    return html_path


def open_visual_summary(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if os.uname().sysname == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def generate_visual_summary(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    digests = discover_report_digests(output_dir, args.visual_pattern, args.visual_limit)
    if not digests:
        print("\nVisual Summary")
        print("--------------")
        print(f"No benchmark report JSON files found in {output_dir} matching '{args.visual_pattern}'.")
        return 2

    visual = build_visual_summary_data(digests)
    print(render_terminal_visual_summary(visual))
    html_path = write_visual_summary_file(output_dir, visual, args.visual_name)
    print(f"\nWrote visual summary HTML: {html_path}")

    if args.open_visual:
        try:
            open_visual_summary(html_path)
            print("Opened visual summary in default browser.")
        except Exception as exc:
            print(f"Could not open browser automatically: {exc}")
    return 0


def build_golden_snapshot(
    results: Sequence[RunResult],
    sizes: Sequence[Tuple[int, int]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    if not sizes:
        return {"entries": []}
    target_size = sizes[min(1, len(sizes) - 1)]
    target_seeds = set(seeds[:3])
    entries = []
    for run in results:
        if run.seed not in target_seeds:
            continue
        if (run.rows, run.cols) != target_size:
            continue
        entries.append(
            {
                "key": run.key(),
                "generator": run.generator,
                "rows": run.rows,
                "cols": run.cols,
                "seed": run.seed,
                "success": run.success,
                "hard_pass": run.hard_pass,
                "quality_score": run.quality_score,
                "elapsed_ms": run.elapsed_ms,
            }
        )
    entries.sort(key=lambda item: item["key"])
    return {
        "target_size": {"rows": target_size[0], "cols": target_size[1]},
        "target_seeds": list(sorted(target_seeds)),
        "entries": entries,
    }


def check_golden(
    current: Sequence[RunResult],
    golden_data: Dict[str, Any],
    max_quality_drop: float,
    max_time_regression_ratio: float,
) -> List[str]:
    messages: List[str] = []
    current_map = {run.key(): run for run in current}

    for entry in golden_data.get("entries", []):
        key = entry["key"]
        run = current_map.get(key)
        if run is None:
            messages.append(f"missing run for golden key {key}")
            continue

        if entry.get("success", False) and not run.success:
            messages.append(f"{key}: baseline succeeded, current failed ({run.error})")
            continue
        if not entry.get("success", False) and run.success:
            messages.append(f"{key}: baseline failed but current succeeded (review baseline)")
            continue
        if not run.success:
            continue

        if entry.get("hard_pass", False) and not run.hard_pass:
            messages.append(f"{key}: hard constraints regressed (baseline pass -> current fail)")

        baseline_quality = float(entry.get("quality_score", 0.0))
        if run.quality_score + max_quality_drop < baseline_quality:
            messages.append(
                f"{key}: quality dropped from {baseline_quality:.2f} to {run.quality_score:.2f}"
            )

        baseline_time = float(entry.get("elapsed_ms", 0.0))
        if baseline_time > 0:
            allowed = baseline_time * max_time_regression_ratio
            if run.elapsed_ms > allowed:
                messages.append(
                    f"{key}: time regressed from {baseline_time:.1f}ms to {run.elapsed_ms:.1f}ms"
                )

    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark maze generators.")
    parser.add_argument(
        "--generators",
        default="maze,braid",
        help="Comma-separated generators: maze,braid",
    )
    parser.add_argument(
        "--sizes",
        default=",".join(f"{r}x{c}" for r, c in DEFAULT_SIZES),
        help="Comma-separated sizes such as 12x12,20x20,30x30",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds",
    )
    parser.add_argument("--maze-max-rect", type=int, default=5, help="maze.py max_rect parameter")
    parser.add_argument("--braid-attempts", type=int, default=80, help="braid candidate count")
    parser.add_argument("--braid-iterations", type=int, default=40, help="braid annealing iterations")
    parser.add_argument("--braid-extra-edge-ratio", type=float, default=0.11)
    parser.add_argument(
        "--braid-no-crop",
        action="store_true",
        help="Do not crop outer perimeter in braid generator",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory for JSON/CSV artifacts",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional basename for output files (without extension)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Write a small golden seed snapshot from this run",
    )
    parser.add_argument(
        "--check-golden",
        action="store_true",
        help="Check this run against existing golden snapshot",
    )
    parser.add_argument(
        "--golden-file",
        default="benchmarks/golden_seeds.json",
        help="Golden snapshot path",
    )
    parser.add_argument(
        "--max-quality-drop",
        type=float,
        default=3.0,
        help="Allowed quality drop when checking golden seeds",
    )
    parser.add_argument(
        "--max-time-regression-ratio",
        type=float,
        default=1.35,
        help="Allowed runtime ratio vs golden snapshot",
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Skip benchmark execution and generate visual summary from existing JSON reports.",
    )
    parser.add_argument(
        "--no-visual-summary",
        action="store_true",
        help="Skip visual summary generation after benchmark run.",
    )
    parser.add_argument(
        "--visual-pattern",
        default="*.json",
        help="Glob pattern for report files in output-dir (used by visual summary).",
    )
    parser.add_argument(
        "--visual-limit",
        type=int,
        default=12,
        help="Max recent report JSON files to include in visual summary (0 = no limit).",
    )
    parser.add_argument(
        "--visual-name",
        default=None,
        help="Optional HTML filename for visual summary output.",
    )
    parser.add_argument(
        "--open-visual",
        action="store_true",
        help="Open visual summary HTML in default browser after writing it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.visual_only:
        return generate_visual_summary(args)

    generators = [name.strip().lower() for name in args.generators.split(",") if name.strip()]
    for generator in generators:
        if generator not in {"maze", "braid"}:
            raise ValueError(f"unsupported generator '{generator}' (use maze, braid)")
    sizes = parse_sizes(args.sizes)
    seeds = parse_seeds(args.seeds)

    total_runs = len(generators) * len(sizes) * len(seeds)
    print(
        f"Running benchmark suite: generators={generators}, sizes={sizes}, "
        f"seeds={len(seeds)}, total runs={total_runs}"
    )

    results: List[RunResult] = []
    done = 0
    for generator in generators:
        for rows, cols in sizes:
            for seed in seeds:
                done += 1
                print(f"[{done:03d}/{total_runs}] {generator} {rows}x{cols} seed={seed} ... ", end="")
                run = run_one(generator, rows, cols, seed, args)
                results.append(run)
                if run.success:
                    print(
                        f"{run.elapsed_ms:.1f}ms, quality={run.quality_score:.2f}, "
                        f"hard_pass={run.hard_pass}"
                    )
                else:
                    print(f"FAILED ({run.error})")

    summary = summarize(results)
    print_summary(summary)

    frontier = pareto_frontier_indices(results)
    print("\nPareto (quality vs generation time)")
    print("-----------------------------------")
    print(render_ascii_pareto(results, frontier))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = args.output_name or f"benchmark_{stamp}"
    json_path = output_dir / f"{base_name}.json"

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "generators": generators,
            "sizes": [{"rows": r, "cols": c} for r, c in sizes],
            "seeds": seeds,
            "maze_max_rect": args.maze_max_rect,
            "braid_attempts": args.braid_attempts,
            "braid_iterations": args.braid_iterations,
            "braid_extra_edge_ratio": args.braid_extra_edge_ratio,
            "braid_crop_outer_perimeter": not args.braid_no_crop,
        },
        "summary": summary,
        "pareto_frontier_keys": [results[i].key() for i in sorted(frontier)],
        "runs": [
            {
                "key": run.key(),
                "generator": run.generator,
                "rows": run.rows,
                "cols": run.cols,
                "seed": run.seed,
                "elapsed_ms": run.elapsed_ms,
                "success": run.success,
                "hard_pass": run.hard_pass,
                "quality_score": run.quality_score,
                "error": run.error,
                "metrics": run.metrics,
                "quality_components": run.quality_components,
                "source_details": run.source_details,
            }
            for run in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote JSON report: {json_path}")

    if not args.no_csv:
        csv_path = output_dir / f"{base_name}.csv"
        write_csv(csv_path, results)
        print(f"Wrote CSV report:  {csv_path}")

    golden_path = Path(args.golden_file)
    if args.update_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = build_golden_snapshot(results, sizes, seeds)
        golden_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"Wrote golden snapshot: {golden_path}")

    if args.check_golden:
        if not golden_path.exists():
            print(f"Golden snapshot not found: {golden_path}")
            return 2
        golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
        regressions = check_golden(
            current=results,
            golden_data=golden_data,
            max_quality_drop=args.max_quality_drop,
            max_time_regression_ratio=args.max_time_regression_ratio,
        )
        if regressions:
            print("\nGolden check: FAILED")
            for msg in regressions:
                print(f"- {msg}")
            return 1
        print("\nGolden check: PASSED")

    if not args.no_visual_summary:
        _ = generate_visual_summary(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
