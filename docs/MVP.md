# TODO

# Minimal Viable Product 
- [x] Write code to generate a braid maze.

## Requirements:
- [-] The maze must have no dead ends.
- [x] Every walkable cell should have at least two walkable neighboring cells, except optional entrance/exit cells if needed.
- [x] The maze should contain loops, so there are multiple possible paths between different areas.
- [x] There should be multiple possible routes from the start to the finish.
- [x] Do not generate a standard “perfect maze” with exactly one solution.
- [x] Generate the maze on a rectangular grid.
- [x] Use walls and passages.
- [x] Include a start and finish point.
- [x] Make the output easy to visualize, either as ASCII or as a 2D array.
- [x] Keep the code simple and readable.

Suggested approach:
Generate a normal perfect maze first using DFS/backtracking, Prim’s algorithm, or Kruskal’s algorithm.
Then “braid” it by removing selected walls next to dead ends until all dead ends are eliminated.
Verify that no dead-end cells remain.
Verify that start and finish are connected and that multiple routes exist.

Please write the full code with comments and a short explanation.