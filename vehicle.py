"""
vehicle.py - simulated car (kinematic bicycle model) + steering-wheel <-> road-wheel angles.

All unit/sign conversions live here so nothing else has to think about them:
  World frame : x right, y up, metres. yaw in radians, 0 = +x, counter-clockwise positive.
  Road wheel  : steer angle in radians, positive = turning left (counter-clockwise).
  Steering wh.: degrees, positive = right (same as WheelController.angle()).

Full steering-wheel lock maps to full road-wheel lock, so the steering ratio follows from
the lock-to-lock range passed in (e.g. 720 deg / 2 / 30 deg max steer -> 12:1).
"""

import math
from dataclasses import dataclass


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


class Steering:
    def __init__(self, lock_to_lock_deg, max_steer_deg=30.0):
        self.max_steer = math.radians(max_steer_deg)
        self.wheel_limit = lock_to_lock_deg / 2.0
        self.ratio = self.wheel_limit / max_steer_deg

    def wheel_to_steer(self, wheel_deg):
        s = -math.radians(wheel_deg / self.ratio)
        return max(-self.max_steer, min(self.max_steer, s))

    def steer_to_wheel(self, steer_rad):
        s = max(-self.max_steer, min(self.max_steer, steer_rad))
        return -math.degrees(s) * self.ratio


class Bicycle:
    """Rear-axle-referenced kinematic bicycle. Fine at low speed with no tyre slip."""

    def __init__(self, wheelbase=2.7):
        self.wheelbase = wheelbase

    def step(self, pose, speed, steer, dt):
        yaw = pose.yaw + speed / self.wheelbase * math.tan(steer) * dt
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return Pose(pose.x + speed * math.cos(pose.yaw) * dt,
                    pose.y + speed * math.sin(pose.yaw) * dt,
                    yaw)
