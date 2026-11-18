"""
override.py - decide when the human has grabbed the wheel, and when they've let go.

Inputs come from WheelController: human_torque (disturbance-observer estimate, fraction of
full-scale torque; needs moza_model.json from `moza_force.py`) and wheel angle / velocity.

Takeover (while the motor is steering):
  filtered |human_torque| > torque_enter, OR wheel is > angle_enter_deg away from where
  we're commanding it - sustained for enter_s.
Release (while the wheel is free):
  wheel is still (|vel| < still_dps) AND filtered |human_torque| < torque_exit -
  sustained for exit_s.

Known limitation: with the wheel fully free, a human holding it perfectly still looks the same
as nobody touching it, so a long steady hold will hand back control. If they're still holding,
the motor ramps in gently and the torque check immediately flips back to override.
"""


class OverrideDetector:
    def __init__(self, torque_enter=None, torque_exit=None, angle_enter_deg=45.0,
                 enter_s=0.12, exit_s=0.6, still_dps=10.0, filter_alpha=0.2):
        self.torque_enter, self.torque_exit = torque_enter, torque_exit
        self.angle_enter = angle_enter_deg
        self.enter_s, self.exit_s = enter_s, exit_s
        self.still_dps = still_dps
        self.alpha = filter_alpha
        self.filtered = 0.0
        self._t = 0.0

    def reset(self):
        self._t = 0.0

    def _filter(self, human_torque):
        self.filtered += self.alpha * (human_torque - self.filtered)
        return self.filtered

    def takeover(self, human_torque, angle_err_deg, dt):
        f = self._filter(human_torque)
        hit = abs(angle_err_deg) > self.angle_enter or (
            self.torque_enter is not None and abs(f) > self.torque_enter)
        self._t = self._t + dt if hit else 0.0
        return self._t >= self.enter_s

    def released(self, human_torque, velocity_dps, dt):
        f = self._filter(human_torque)
        quiet = abs(velocity_dps) < self.still_dps and (
            self.torque_exit is None or abs(f) < self.torque_exit)
        self._t = self._t + dt if quiet else 0.0
        return self._t >= self.exit_s
