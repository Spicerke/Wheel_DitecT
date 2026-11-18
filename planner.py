"""
planner.py - occupancy-grid map + A* path planner.

This is a stand-in for whatever algorithm the real car will use: everything downstream
only sees a list of (x, y) waypoints in metres (see vehicle.py for the frame).

Map files are plain text, one character per cell:
  #  obstacle        .  free        S  car start (free)        G  default goal (free)
Row 0 is the top of the map (largest y).

The planner is run in a separate process (see init_worker / plan_path) so a long search
can't steal the GIL from the 500 Hz wheel loop.
"""

import heapq
import math
import signal
import time
from collections import deque

SQRT2 = math.sqrt(2.0)
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


class GridMap:
    def __init__(self, rows, cols, res, occ, start=None, goal=None):
        self.rows, self.cols, self.res = rows, cols, res
        self.occ = occ                     # bytearray rows*cols, 1 = blocked
        self.start, self.goal = start, goal  # world (x, y) or None

    # ---------- loading ----------
    @classmethod
    def load(cls, path, res=0.5):
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        rows, cols = len(lines), max(len(ln) for ln in lines)
        occ = bytearray(rows * cols)
        g = cls(rows, cols, res, occ)
        for r, ln in enumerate(lines):
            for c in range(cols):
                ch = ln[c] if c < len(ln) else "#"
                if ch == "#":
                    occ[r * cols + c] = 1
                elif ch == "S":
                    g.start = g.cell_to_world(r, c)
                elif ch == "G":
                    g.goal = g.cell_to_world(r, c)
        return g

    def inflated(self, radius_m):
        """Copy with obstacles grown by radius_m, so the planner can treat the car as a point."""
        rad = int(math.ceil(radius_m / self.res))
        offsets = [(dr, dc) for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1)
                   if dr * dr + dc * dc <= rad * rad]
        out = bytearray(self.occ)
        R, C = self.rows, self.cols
        for i, v in enumerate(self.occ):
            if not v:
                continue
            r, c = divmod(i, C)
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    out[nr * C + nc] = 1
        return GridMap(R, C, self.res, out, self.start, self.goal)

    # ---------- coordinates ----------
    def cell_to_world(self, r, c):
        return ((c + 0.5) * self.res, (self.rows - r - 0.5) * self.res)

    def world_to_cell(self, x, y):
        return (int(self.rows - y / self.res), int(x / self.res))

    def blocked(self, x, y):
        r, c = self.world_to_cell(x, y)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return True
        return bool(self.occ[r * self.cols + c])

    def line_free(self, a, b):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / (self.res * 0.5)))
        for i in range(n + 1):
            t = i / n
            if self.blocked(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t):
                return False
        return True

    def nearest_free(self, r, c, max_cells=40):
        """BFS outward from (r, c) to the closest unblocked cell index, or None."""
        R, C = self.rows, self.cols
        r, c = max(0, min(R - 1, r)), max(0, min(C - 1, c))
        start = r * C + c
        seen = {start}
        q = deque([(start, 0)])
        while q:
            i, d = q.popleft()
            if not self.occ[i]:
                return i
            if d >= max_cells:
                continue
            ir, ic = divmod(i, C)
            for dr, dc, _ in _MOVES[:4]:
                nr, nc = ir + dr, ic + dc
                n = nr * C + nc
                if 0 <= nr < R and 0 <= nc < C and n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    def to_rows(self):
        C = self.cols
        return ["".join("#" if v else "." for v in self.occ[r * C:(r + 1) * C])
                for r in range(self.rows)]


# ---------- search ----------
def astar(grid, start, goal):
    """8-connected A* over cell indices. Returns (list of cell indices, nodes expanded)."""
    R, C, occ = grid.rows, grid.cols, grid.occ
    gr, gc = divmod(goal, C)

    def h(i):
        r, c = divmod(i, C)
        dr, dc = abs(r - gr), abs(c - gc)
        return dr + dc + (SQRT2 - 2.0) * min(dr, dc)

    inf = float("inf")
    g = [inf] * (R * C)
    came = {}
    closed = bytearray(R * C)
    g[start] = 0.0
    heap = [(h(start), 0.0, start)]
    expanded = 0
    while heap:
        _, gs, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1], expanded
        closed[cur] = 1
        expanded += 1
        r, c = divmod(cur, C)
        for dr, dc, cost in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            n = nr * C + nc
            if occ[n] or closed[n]:
                continue
            if dr and dc and (occ[r * C + nc] or occ[nr * C + c]):
                continue                                   # no squeezing past corners
            ng = gs + cost
            if ng < g[n]:
                g[n] = ng
                came[n] = cur
                heapq.heappush(heap, (ng + h(n), ng, n))
    return [], expanded


def shortcut(grid, pts):
    """Greedy string-pulling: skip waypoints while there's a straight free line."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and grid.line_free(pts[i], pts[j + 1]):
            j += 1
        out.append(pts[j])
        i = j
    return out


def resample(pts, spacing):
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / spacing))
        for k in range(1, n + 1):
            out.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    return out


def plan(grid, start_xy, goal_xy, spacing=0.25):
    t0 = time.perf_counter()
    s = grid.nearest_free(*grid.world_to_cell(*start_xy))
    g = grid.nearest_free(*grid.world_to_cell(*goal_xy), max_cells=4)
    if s is None:
        return {"error": "car is not near any free space"}
    if g is None:
        return {"error": "goal is inside an obstacle"}
    cells, expanded = astar(grid, s, g)
    ms = (time.perf_counter() - t0) * 1000
    if not cells:
        return {"error": "no path to goal", "ms": ms, "expanded": expanded}
    pts = [tuple(start_xy)] + [grid.cell_to_world(*divmod(i, grid.cols)) for i in cells]
    pts = resample(shortcut(grid, pts), spacing)
    return {"path": pts, "ms": (time.perf_counter() - t0) * 1000, "expanded": expanded}


# ---------- worker-process entry points ----------
_MAP = None


def init_worker(grid):
    global _MAP
    _MAP = grid
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # parent handles Ctrl+C


def plan_path(start_xy, goal_xy):
    return plan(_MAP, start_xy, goal_xy)
