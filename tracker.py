"""
tracker.py - turn a waypoint path + car pose into a steering angle (pure pursuit).

This is the piece that stays when the A* planner is swapped for the car's own algorithm:
give it a list of (x, y) points and the car's pose, get back a road-wheel angle.
"""

import math


class PurePursuit:
    def __init__(self, wheelbase, lookahead_min=4.0, lookahead_gain=1.0, window=80):
        self.L = wheelbase
        self.lmin, self.k = lookahead_min, lookahead_gain
        self.window = window                 # how far ahead (in points) to search for progress
        self.path, self.cum, self.idx = [], [], 0

    def set_path(self, path, pose):
        self.path = path
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.idx = self._nearest(pose, 0, len(path))

    def _nearest(self, pose, lo, hi):
        best, bi = float("inf"), lo
        for i in range(lo, hi):
            x, y = self.path[i]
            d = (x - pose.x) ** 2 + (y - pose.y) ** 2
            if d < best:
                best, bi = d, i
        return bi

    def update(self, pose, speed):
        """Returns (steer_rad, info). info has target point, cross-track error, remaining m."""
        p = self.path
        self.idx = self._nearest(pose, self.idx, min(len(p), self.idx + self.window))
        ld = max(self.lmin, self.k * abs(speed))
        j = self.idx
        while j < len(p) - 1 and math.hypot(p[j][0] - pose.x, p[j][1] - pose.y) < ld:
            j += 1
        tx, ty = p[j]
        dx, dy = tx - pose.x, ty - pose.y
        alpha = math.atan2(dy, dx) - pose.yaw
        steer = math.atan2(2.0 * self.L * math.sin(alpha), max(math.hypot(dx, dy), 1e-3))

        # signed cross-track error vs the segment we're on (+ = car is left of path)
        i = min(self.idx, len(p) - 2) if len(p) > 1 else 0
        cte = 0.0
        if len(p) > 1:
            (x0, y0), (x1, y1) = p[i], p[i + 1]
            sl = math.hypot(x1 - x0, y1 - y0) or 1e-9
            cte = ((x1 - x0) * (pose.y - y0) - (y1 - y0) * (pose.x - x0)) / sl
        cx, cy = p[self.idx]
        remaining = self.cum[-1] - self.cum[self.idx] + math.hypot(cx - pose.x, cy - pose.y)
        return steer, {"target": (tx, ty), "cte": cte, "remaining": remaining,
                       "lookahead": ld}


class SlewLimiter:
    """Caps how fast the commanded wheel angle can move (deg/s), so the motor never snaps."""

    def __init__(self, rate):
        self.rate, self.value = rate, 0.0

    def reset(self, value):
        self.value = value

    def step(self, target, dt):
        d = max(-self.rate * dt, min(self.rate * dt, target - self.value))
        self.value += d
        return self.value
