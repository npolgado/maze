# BENCHMARKING

## How to Run

```bash
# Full run: both generators, 10 seeds × 3 sizes (60 runs total)
python benchmark.py

# Single generator
python benchmark.py --generators maze
python benchmark.py --generators braid

# Custom sizes / seeds
python benchmark.py --sizes 10x10,20x20 --seeds 1,2,3

# Regenerate the HTML visual summary from existing JSON results (no re-run)
python benchmark.py --visual-only

# Open the HTML report in a browser automatically
python benchmark.py --open-visual

# Golden seed workflow
python benchmark.py --update-golden   # write regression baseline
python benchmark.py --check-golden    # fail if quality drops >3 pts or time regresses >35%
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--generators` | `maze,braid` | Comma-separated generators to run |
| `--sizes` | `12x12,20x20,30x30` | Maze sizes |
| `--seeds` | 10 fixed seeds | Seeds for reproducibility |
| `--output-dir` | `benchmarks/results/` | Where JSON/CSV land |
| `--output-name` | timestamp | Basename for output files |
| `--no-csv` | off | Skip CSV export |
| `--visual-only` | off | Skip run, re-render HTML from existing JSON |
| `--open-visual` | off | Auto-open HTML in browser after writing it |
| `--update-golden` | off | Snapshot current results as regression baseline |
| `--check-golden` | off | Compare current run against the baseline |
| `--no-visual-summary` | off | Skip HTML generation |

## What It Does

1. **Generate** — For each (generator, size, seed) combination, create a maze and normalize it to a shared adjacency-list format (`MazeData`).

2. **Compute metrics** — Three groups per maze:
   - **Hard constraints** — connectivity (must be 1 component), min degree ≥ 2 (no dead ends), 2×2 square loops, narrow 2×N loops. All must pass for `hard_pass = True`.
   - **Structural interest** — loop depth distribution (alternate-path length per sampled edge), corridor monotony (longest straight hallway + count over threshold), decision density (junctions per 100 cells), decision cadence (avg steps between junctions along corridors), degree entropy.
   - **Navigation / play** — start-to-finish path length percentile vs sampled pairs, route redundancy (edge-disjoint paths + near-disjoint alternatives via Yen's k-shortest), detour factor (how much longer 2nd/3rd paths are), backtrack pressure (simulated limited-memory agent overhead vs optimal), local recoverability (extra cost of a wrong turn at a junction), exploration coverage AUC (pace of novelty for a biased random walker).

3. **Score** — A weighted sum of 12 normalized components (0–100). Hard constraint failures apply direct point deductions on top.

   | Component | Weight |
   |---|---|
   | hard_constraints | 0.18 |
   | loop_depth | 0.12 |
   | start_finish | 0.11 |
   | redundancy | 0.10 |
   | backtrack_pressure | 0.08 |
   | decision_structure | 0.08 |
   | corridor_monotony | 0.07 |
   | exploration_pace | 0.06 |
   | degree_entropy | 0.06 |
   | detour_factor | 0.06 |
   | local_recoverability | 0.05 |
   | tiny_loop_suppression | 0.03 |

4. **Output** — After all runs:
   - `benchmarks/results/benchmark_<timestamp>.json` — full metrics per run.
   - `benchmarks/results/benchmark_<timestamp>.csv` — flat table for spreadsheet analysis.
   - ASCII Pareto chart in terminal (quality vs time; `M`/`m` = maze, `B`/`b` = braid, uppercase = Pareto-frontier).
   - `benchmarks/results/visual_summary_<timestamp>.html` — cards per generator, per-report table, SVG scatter plot, top-runs table, and a **Component Breakdown** panel (see below).

5. **Component Breakdown panel** — The HTML report includes a tabbed panel showing per-generator mean scores for each of the 12 quality components across all successful runs:
   - **Radar** — spider/radar SVG with one polygon per generator.
   - **Bar Chart** — horizontal bars per component, labelled with weight and numeric score.
   - **Table** — sortable table with a Δ column and winner column when two generators are present.

   The panel is driven by `COMPONENT_ORDER`, `_COMPONENT_WEIGHT`, and `_COMPONENT_LABEL` module-level constants in `benchmark.py`.

6. **Golden regression** — `--update-golden` snapshots quality scores and times for the first 3 seeds at the middle size into `benchmarks/golden_seeds.json`. `--check-golden` on a later run exits 1 if quality drops by more than 3 points or generation time grows by more than 35%.

---

## Metric Definitions

### 1) Hard Constraints (must pass)

- Connectivity (1 component).
- Braid validity (min degree >= 2, dead ends = 0).
- Tiny-loop suppression (2x2 count, 2xN / Nx2 count).
- Generation success rate (no retries/failures).

### 2) Structural Interest Metrics

- Loop depth: alternate-path length per edge (distribution, not just mean).
- Corridor monotony: longest straight hallway + count of hallways over threshold.
- Decision density: junctions per 100 cells (degree >= 3).
- Decision cadence: average distance between junctions along traversable paths.
- Degree entropy: diversity of node degrees (avoids overly uniform layouts).

### 3) Navigation/Play Metrics (most important for "fun")

- Start-finish path length percentile (avoid too short).
- Route redundancy: number of edge-disjoint or near-disjoint S→F alternatives.
- Detour factor: how much longer second/third best paths are vs shortest.
- Backtrack pressure proxy: simulated "limited-memory agent" overhead vs optimal.
- Local recoverability: if player chooses wrong at a junction, expected cost to recover.
- Exploration coverage curve: % maze discovered over steps (pace of novelty).

---

## Scoring Notes

### Hard constraint failures double-penalize

When `hard_pass = False`, the score takes two separate hits:

1. **Component weight** — `hard_constraints` = `float(hard_pass)` = `0.0`, so the 18% weight block contributes nothing to the base score (up to −18 points).
2. **Direct deductions** applied on top of the weighted sum:
   - Not connected: −35
   - `min_degree < 2`: −25 × (2 − min_degree)
   - Dead ends: −min(25, dead_ends × 4)
   - 2×2 squares: −min(20, count_2x2 × 6)
   - Narrow 2×N loops: −min(15, narrow_total × 1.5)

A maze failing multiple hard constraints can lose 50+ points from penalties alone, which combined with the correlated degradation of structural metrics (loop depth, tiny-loop suppression, etc.) readily drives the total to 0.

### Why both generators currently score 0 and braid always fails `hard_pass`

`hard_pass` requires `narrow_total == 0` — meaning **no 2×N rectangular perimeter loops for any N from 3 up to `min(12, max(rows, cols))`**. This is a strict standard that neither generator currently meets:

- **braid_maze**: `finalize_playable_maze` removes dead ends and 2×2 squares but makes no attempt to eliminate wider 2×N loops. Braid mazes need many loops everywhere to avoid dead ends, so 2×3, 2×4, etc. loops are unavoidable. `hard_pass` fails on every single run (0%).

- **maze**: `_enforce_constraints` tries to remove thin rectangles, but its detection algorithm differs from the benchmark's `narrow_rectangle_loop_counts`, and later passes (`_add_long_loops`, `_break_long_hallways`) can introduce new narrow rectangles that are not re-checked.

Until the generators are updated to suppress 2×N loops, `hard_pass` rates will remain near 0 and quality scores will be heavily penalized.

---

## TODO

- [x] Define metrics for benchmarking
- [x] Create initial benchmark.py
- [x] Compare maze.py to braid_maze.py
- [ ] Tune weights in `compute_quality_score`
- [ ] Add per-size breakdown to visual summary
- [ ] Make test cases