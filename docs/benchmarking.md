# BENCHMARKING 

1) Hard Constraints (must pass)

- Connectivity (1 component).
- Braid validity (min degree >= 2, dead ends = 0).
- Tiny-loop suppression (2x2 count, 2xN / Nx2 count).
- Generation success rate (no retries/failures).

2) Structural Interest Metrics

- Loop depth: alternate-path length per edge (distribution, not just mean).
- Corridor monotony: longest straight hallway + count of hallways over threshold.
- Decision density: junctions per 100 cells (degree >= 3).
- Decision cadence: average distance between junctions along traversable paths.
- Degree entropy: diversity of node degrees (avoids overly uniform layouts).

3) Navigation/Play Metrics (most important for “fun”)

- Start-finish path length percentile (avoid too short).
- Route redundancy: number of edge-disjoint or near-disjoint S→F alternatives.
- Detour factor: how much longer second/third best paths are vs shortest.
- Backtrack pressure proxy: simulated “limited-memory agent” overhead vs optimal.
- Local recoverability: if player chooses wrong at a junction, expected cost to recover.
- Exploration coverage curve: % maze discovered over steps (pace of novelty).

A practical benchmark protocol:

1. Fixed seed suite (e.g., 10 seeds) across 3 sizes.
2. Collect all metrics per maze.
3. Use a weighted score + Pareto chart (quality vs generation time).
4. Keep a small “golden seed set” for regression checks.

Draft a concrete benchmark.py schema and scoring formula tuned to your current maze.py and braid_maze.py implementations.

## TODO

- [x] Define metrics for benchmarking
- [x] Create initial benchmark.py
- [ ] Compare maze.py to braid_maze.py
- [ ] Make test cases