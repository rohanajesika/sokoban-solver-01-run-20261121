// Sokoban push-optimal solver — entry for make.fun sokoban-solver-01-run-20261121.
//
// Strategy (built for how this competition draws its levels: a one-square hallway
// with 1-4 small rooms off it, 3-4 boxes per room, each room push-optimal on its own,
// boxes never leaving their room in an optimal line):
//   1. Split the level into rooms by cutting the hallway's dead squares out of the
//      floor graph; each leftover part holding boxes is one room.
//   2. Solve each room on its own with A* over pushes (a state is the boxes plus the
//      player's reachable region, named by its lowest square). The estimate is the
//      minimum-cost assignment of boxes to goals over per-goal push distances, which
//      is admissible, so the first solved state found is push-optimal for the room.
//   3. Prune dead states: boxes on dead squares (no push path to any goal, ignoring
//      the other boxes) and frozen 2x2 blocks of walls and boxes.
//   4. Stitch the rooms' move strings together, walking the player between rooms.
// Fallbacks (in order) keep an instance answered instead of empty: plain push-BFS
// per room, whole-level push-A*, whole-level greedy best-first. Everything is
// budgeted by work (states expanded), never by the wall clock, so two runs write
// byte-identical output. No network, no clock reads, no writes outside the output
// directory. Node 22, erasable TypeScript only, no dependencies.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work budgets (states expanded), never the clock: the same input always costs the same work. */
const ROOM_ASTAR_BUDGET = 500_000;
const ROOM_BFS_BUDGET = 1_000_000;
const WHOLE_ASTAR_BUDGET = 800_000;
const WHOLE_GREEDY_BUDGET = 800_000;

const DIRS: { dx: number; dy: number; step: string; push: string }[] = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

/** A level in XSB text; short rows pad as walls, the outer frame counts as wall. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goalSet[i] = 1;
        goals.push(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

/**
 * The squares from which a box can be pushed onto some goal, ignoring the other
 * boxes: pull the goals backwards (a box at y can be pushed to x when the player
 * can stand at y - d). A box off this set is dead wherever it stands.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const dir of DIRS) {
      const d = dir.dy * w + dir.dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Per goal, the fewest pushes a lone box needs to reach it from each square
 * (same pull-BFS, one goal at a time). A box's true cost to a goal is never below
 * this, so these distances ground an admissible estimate.
 */
function pushDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (const dir of DIRS) {
        const d = dir.dy * w + dir.dx;
        const y = x - d;
        if (!wall[y] && !wall[y - d] && dist[y] < 0) {
          dist[y] = dist[x] + 1;
          queue.push(y);
        }
      }
    }
    return dist;
  });
}

/** Minimum-cost assignment of boxes to goals (Hungarian, O(n^3)); Infinity when no box-to-goal matching exists. */
function assignmentCost(boxes: number[], distToGoal: Int32Array[]): number {
  const n = boxes.length;
  const BIG = 1_000_000;
  const cost: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    let any = false;
    for (let j = 0; j < n; j++) {
      const d = distToGoal[j][boxes[i]];
      row.push(d < 0 ? BIG : d);
      if (d >= 0) any = true;
    }
    if (!any) return Infinity;
    cost.push(row);
  }
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const total = -v[0];
  return total >= BIG ? Infinity : total;
}

/** Admissible estimate for a box set: assignment cost for small sets, sum of nearest-goal distances otherwise. */
function estimate(boxes: number[], distToGoal: Int32Array[]): number {
  if (boxes.length <= 6 && boxes.length === distToGoal.length) return assignmentCost(boxes, distToGoal);
  let sum = 0;
  for (const b of boxes) {
    let best = Infinity;
    for (let j = 0; j < distToGoal.length; j++) {
      const d = distToGoal[j][b];
      if (d >= 0 && d < best) best = d;
    }
    if (best === Infinity) return Infinity;
    sum += best;
  }
  return sum;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes holding a box off its goal. */
function frozenBlock(level: Level, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goalSet } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
  }
  return false;
}

/** A binary heap on (f, -g, sequence): the smallest f first, deeper on ties, insertion order last. Deterministic. */
class Heap {
  // Plain field (no access modifier): erasable TypeScript only, so the file
  // also runs on Node 22's type stripping without a second thought.
  a: { f: number; g: number; seq: number; s: number }[] = [];
  get size(): number {
    return this.a.length;
  }
  push(e: { f: number; g: number; seq: number; s: number }): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop(): { f: number; g: number; seq: number; s: number } {
    const a = this.a;
    const top = a[0];
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && less(a[c + 1], a[c])) c++;
        if (!less(a[c], a[i])) break;
        [a[i], a[c]] = [a[c], a[i]];
        i = c;
      }
    }
    return top;
  }
}

function less(x: { f: number; g: number; seq: number }, y: { f: number; g: number; seq: number }): boolean {
  return x.f < y.f || (x.f === y.f && (x.g > y.g || (x.g === y.g && x.seq < y.seq)));
}

interface SearchOutcome {
  pushes: number;
  moves: string;
  expansions: number;
}

/**
 * Push-optimal search over one level (a room, or the whole board as a fallback).
 * A state is the boxes and the player's reachable region, named by its lowest
 * square; the player walks for free inside a region, so only pushes branch.
 * A* with the assignment estimate when `astar` is set, plain breadth-first over
 * pushes otherwise; greedy best-first when `greedy` is set. Returns null when
 * the work budget runs out or no solution exists within it.
 */
function searchPushes(level: Level, budget: number, astar: boolean, greedy = false): SearchOutcome | null {
  const { w, wall, goalSet } = level;
  const size = wall.length;
  const n = level.boxes.length;
  if (n === 0) return { pushes: 0, moves: '', expansions: 0 };
  const live = liveSquares(level);
  for (const b of level.boxes) if (!live[b]) return null;
  const distToGoal = astar || greedy ? pushDistances(level) : [];
  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b]);
  if (solved(level.boxes)) return { pushes: 0, moves: '', expansions: 0 };

  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  // A second mark array for the successor regions: regionLow must never
  // re-stamp the cells the current expansion's flood just marked.
  const lowSeen = new Int32Array(size);
  let lowStamp = 0;
  // Flood the player's region from `from` around walls and boxes; returns its lowest square.
  const regionLow = (occupied: Uint8Array, from: number): number => {
    const mark = ++lowStamp;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    lowSeen[from] = mark;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < low) low = p;
      for (const dir of DIRS) {
        const q = p + dir.dy * w + dir.dx;
        if (!wall[q] && !occupied[q] && lowSeen[q] !== mark) {
          lowSeen[q] = mark;
          stack[sp++] = q;
        }
      }
    }
    return low;
  };

  // State key: the region's lowest square folded with the boxes. Numeric while it
  // fits exactly in a double (rooms do); a string for the whole-board fallback.
  const numericKeys = size ** (n + 1) < 2 ** 53;
  const keyOf = (boxes: number[], low: number): number | string => {
    if (!numericKeys) return `${low}|${boxes.join(',')}`;
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const boxesList: number[][] = [level.boxes.slice()];
  const playerList: number[] = [level.player];
  // Each state's region, named by its lowest square (the key's other half).
  const lowList: number[] = [];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const gScore: number[] = [0];
  const occupied = new Uint8Array(size);
  for (const b of level.boxes) occupied[b] = 1;
  const best = new Map<number | string, number>();
  lowList.push(regionLow(occupied, level.player));
  best.set(keyOf(level.boxes, lowList[0]), 0);
  for (const b of level.boxes) occupied[b] = 0;

  const heap = new Heap();
  const h0 = astar || greedy ? estimate(level.boxes, distToGoal) : 0;
  if (h0 === Infinity) return null;
  let seq = 0;
  heap.push({ f: h0, g: 0, seq: seq++, s: 0 });

  let expansions = 0;
  let found = -1;
  while (heap.size > 0) {
    const { g, s } = heap.pop();
    if (gScore[s] !== g) continue;
    // A better path to this key may have been found since this entry was
    // pushed: only the state's key's current best is ever expanded.
    if (best.get(keyOf(boxesList[s], lowList[s])) !== s) continue;
    const boxes = boxesList[s];
    // The goal is checked when a state is popped, not when it is made: with a
    // consistent estimate this is what makes the first solved state optimal.
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expansions >= budget) return null;
    expansions++;
    for (const b of boxes) occupied[b] = 1;
    const regionMark = ++stamp;
    // Flood this state's region once; every push checks reachability against it.
    let sp = 0;
    stack[sp++] = playerList[s];
    seen[playerList[s]] = regionMark;
    while (sp > 0) {
      const p = stack[--sp];
      for (const dir of DIRS) {
        const q = p + dir.dy * w + dir.dx;
        if (!wall[q] && !occupied[q] && seen[q] !== regionMark) {
          seen[q] = regionMark;
          stack[sp++] = q;
        }
      }
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < DIRS.length; di++) {
        const dir = DIRS[di];
        const d = dir.dy * w + dir.dx;
        const t = b + d;
        if (seen[b - d] !== regionMark || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(level, occupied, t);
        let key: number | string = '';
        let next: number[] | null = null;
        let nextLow = -1;
        let h = 0;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, c) => a - c);
          nextLow = regionLow(occupied, b);
          key = keyOf(next, nextLow);
          if (best.get(key) === undefined || g + 1 < gScore[best.get(key)!]) {
            h = astar || greedy ? estimate(next, distToGoal) : 0;
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || h === Infinity) continue;
        const prev = best.get(key);
        if (prev !== undefined && gScore[prev] <= g + 1) continue;
        const t2 = boxesList.length;
        boxesList.push(next);
        playerList.push(b);
        lowList.push(nextLow);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        pushDir.push(di);
        gScore.push(g + 1);
        best.set(key, t2);
        heap.push({ f: greedy ? h : g + 1 + h, g: g + 1, seq: seq++, s: t2 });
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of pushes, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const from = pushFrom[s];
    const to = pushTo[s];
    const dir = DIRS[pushDir[s]];
    const d = dir.dy * w + dir.dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.push;
    player = from;
    boxes = boxesList[s].slice();
  }
  return { pushes: path.length, moves, expansions };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (up first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = dir.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' || ch === 'U' ? -w : ch === 'd' || ch === 'D' ? w : ch === 'l' || ch === 'L' ? -1 : 1;
  return p;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box
 * there reaches no goal) out of the floor, and each part left that holds a box is
 * a room. Null when the level is no set of rooms (a box starts in the hallway, or
 * some part holds a different count of boxes and goals).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const dir of DIRS) {
        const q = squares[i] + dir.dy * w + dir.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Solve one level room by room, each room push-optimal on its own (the other rooms'
 * boxes standing as walls, finished rooms' goals as walls since their boxes sit
 * there). Falls back to a whole-level search when the level is no set of rooms or
 * a room's search gives up; an empty string when nothing solves it.
 */
function solveLevel(level: Level): string {
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const subGoals = room.goals.slice().sort((a, b) => a - b);
      const subGoalSet = new Uint8Array(level.wall.length);
      for (const g of subGoals) subGoalSet[g] = 1;
      const sub: Level = {
        w: level.w,
        h: level.h,
        wall,
        goals: subGoals,
        goalSet: subGoalSet,
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = searchPushes(sub, ROOM_ASTAR_BUDGET, true) ?? searchPushes(sub, ROOM_BFS_BUDGET, false);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    if (ok) return moves;
  }
  const whole = searchPushes(level, WHOLE_ASTAR_BUDGET, true) ?? searchPushes(level, WHOLE_GREEDY_BUDGET, false, true);
  return whole === null ? '' : whole.moves;
}

// Answer every instance: write an empty solution first, so a failure mid-run
// still leaves every instance answered in the stated format, then overwrite
// with the real solution as each level is solved.
const ids = readdirSync(INPUT_DIR).sort();
for (const id of ids) {
  try {
    if (!statSync(join(INPUT_DIR, id)).isDirectory()) continue;
    mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
    writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), '\n');
  } catch {
    // A missing or unreadable input still gets its file attempted below.
  }
}
for (const id of ids) {
  let moves = '';
  try {
    if (!statSync(join(INPUT_DIR, id)).isDirectory()) continue;
    const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
    moves = solveLevel(level);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
