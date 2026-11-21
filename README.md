# Sokoban Solver — make.fun sokoban-solver-01-run-20261121

Push-optimal Sokoban solver in a single `main.ts` (Node 22, no dependencies).
It answers all 5 instances of a set in one run, well inside the 60-second wall
budget, and writes byte-identical output on every run.

## How it works

The competition draws each level as a one-square hallway with 1–4 small rooms
off it (3–4 boxes per room, each room push-optimal on its own, boxes never
leaving their room in an optimal line). The solver exploits that shape:

1. **Split into rooms.** The hallway's dead squares (squares a box can never be
   pushed from onto a goal) are cut out of the floor graph; each leftover part
   holding boxes is one room.
2. **Solve each room push-optimally** with A* over pushes. A state is the boxes
   plus the player's reachable region (named by its lowest square), so only
   pushes branch. The estimate is the minimum-cost assignment of boxes to
   goals over per-goal push distances — admissible, so the first solved state
   found is push-optimal for the room.
3. **Prune dead states:** boxes on dead squares, and frozen 2×2 blocks of walls
   and boxes holding a box off its goal.
4. **Stitch the rooms together**, walking the player between rooms (other rooms'
   boxes stand as walls; finished rooms' goals stand as walls since their boxes
   sit there).

Fallbacks keep every instance answered instead of empty: plain push-BFS per
room, then whole-level push-A*, then whole-level greedy best-first. All budgets
are counted in states expanded, never by the wall clock, so two runs write
identical bytes. No network, no clock reads, no writes outside the output
directory.

## Results

- Practice set (kit): **1,000,000** — all 5 instances solved at exactly P*
  (13 / 16 / 29 / 47 / 60 pushes). House baseline plain-astar 1.0.0 scores
  200,000 on the same set.
- 10 freshly drawn sets from the kit's own generator (unseen seeds):
  **1,000,000 on every set**, each solved at exactly P*, each set in under
  0.5 s.
- Determinism: two runs on the practice set write byte-identical output.
- Grader simulation (user 65534, no network, read-only entry, cwd `/`,
  60 s limit): exit 0, output byte-identical to a normal run.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
