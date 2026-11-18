"""
vehicle.py - simulated car (kinematic bicycle model) + steering-wheel <-> road-wheel angles.

All unit/sign conversions live here so nothing else has to think about them:
  World frame : x right, y up, metres. yaw in radians, 0 = +x, counter-clockwise positive.
  Road wheel  : steer angle in radians, positive = turning left (counter-clockwise).
  Steering wh.: degrees, positive = right (same as WheelController.angle()).

wheel angle = road-wheel angle * ratio. The only limits are physical: the car's road-wheel
lock (max_steer) and how far the wheel base can turn (wheel_limit, the Boxflat range).
"""

import math
from dataclasses import dataclass


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


class Steering:
    def __init__(self, ratio, max_steer_deg=30.0, wheel_limit_deg=450.0):
        self.ratio = ratio
        self.max_steer = math.radians(max_steer_deg)
        self.wheel_limit = wheel_limit_deg

    def wheel_to_steer(self, wheel_deg):
        s = -math.radians(wheel_deg / self.ratio)
        return max(-self.max_steer, min(self.max_steer, s))

    def steer_to_wheel(self, steer_rad):
        s = max(-self.max_steer, min(self.max_steer, steer_rad))
        w = -math.degrees(s) * self.ratio
        return max(-self.wheel_limit, min(self.wheel_limit, w))


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
