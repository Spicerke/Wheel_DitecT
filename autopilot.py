#!/usr/bin/env python3
"""
autopilot.py - plan a path, steer the MOZA wheel along it, let the human take over.

A simulated car (vehicle.py) drives around a grid map. The car always steers with whatever
angle the PHYSICAL wheel is at - exactly like a steer-by-wire car - so:
  AUTO        the motor turns the wheel to follow the planned path (pure pursuit)
  OVERRIDE    the human grabbed the wheel: motor goes fully free, car follows the human
  REPLANNING  human let go: plan a fresh path from wherever the car is now, then back to AUTO

Loops / threads:
  wheel   500 Hz  WheelController.run (moza_drive.py), own thread
  control  50 Hz  this file: tracking, override detection, car sim
  planner  async  A* in a separate process (planner.py)
  web      10 Hz  debug UI stream (debug_server.py + web/index.html)

Usage (run moza_force.py once first so moza_model.json exists - that's what lets us feel
the human's hand while the motor is driving):
  python3 autopilot.py                       # then open http://localhost:8080, click a goal
  python3 autopilot.py --host 0.0.0.0        # view the UI from another machine on the LAN
  python3 autopilot.py --go                  # immediately drive to the map's G

UI: click = set goal, shift+click = move car (when idle), Space = stop.
    Manual button: drag the wheel picture (or arrow keys) to steer; up/down = speed.

SAFETY: the wheel will turn by itself. Start with a low --max-torque.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from debug_server import DebugHub, serve
from moza_drive import WheelController, find_moza
from override import OverrideDetector
from planner import GridMap, init_worker, plan_path
from tracker import PurePursuit, SlewLimiter
from vehicle import Bicycle, Pose, Steering

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(HERE, "moza_model.json")
DRIVING = ("AUTO", "OVERRIDE", "REPLANNING")
MODE_CODE = {"IDLE": 0, "PLANNING": 1, "AUTO": 2, "OVERRIDE": 3, "REPLANNING": 4,
             "CENTERING": 5, "MANUAL": 6}


class Autopilot:
    def __init__(self, ctl, grid, plan_grid, cfg, hub, pool, detector):
        self.ctl, self.grid, self.plan_grid = ctl, grid, plan_grid
        self.cfg, self.hub, self.pool, self.detector = cfg, hub, pool, detector
        self.full_torque = ctl.max_torque
        lock = min(cfg.wheel_lock, ctl.rotation / 2.0)
        if lock < cfg.wheel_lock:
            print(f"--wheel-lock {cfg.wheel_lock:.0f} exceeds the Boxflat range; using {lock:.0f}")
        self.steering = Steering(2.0 * lock, cfg.max_steer)
        self.car = Bicycle(cfg.wheelbase)
        sx, sy = grid.start or (grid.cols * grid.res / 2, grid.rows * grid.res / 2)
        self.start_pose = Pose(sx, sy, math.radians(cfg.start_yaw))
        self.pose = Pose(**vars(self.start_pose))
        self.tracker = PurePursuit(cfg.wheelbase, cfg.lookahead, cfg.lookahead_gain)
        self.slew = SlewLimiter(cfg.slew)

        self.state = "IDLE"
        self.msg = "click the map to set a goal"
        self.goal = None
        self._future = None
        self._ramp_done = True
        self.info = {}
        self.plan_stats = {}
        self.overrides = 0
        self.collision = False
        self._loop_worst = 0.0
        self._timing = (0.0, 0.0)
        self._timing_t = 0.0
        self.log = None                              # csv.writer, set by main()
        self.manual_deg = 0.0                        # UI-commanded wheel angle in MANUAL
        self.manual_speed = cfg.speed

    # ---------- state changes ----------
    def _set(self, state, msg=None):
        if state != self.state:
            print(f"[{time.strftime('%H:%M:%S')}] {self.state} -> {state}"
                  + (f"  ({msg})" if msg else ""))
        self.state = state
        if msg is not None:
            self.msg = msg

    def stop(self, msg="stopped"):
        self.ctl.release()
        self.ctl.max_torque = self.full_torque
        self._future = None
        self._set("IDLE", msg)

    def set_goal(self, x, y):
        if self.plan_grid.blocked(x, y):
            self.msg = f"goal ({x:.1f}, {y:.1f}) is inside an obstacle"
            return
        self.goal = (x, y)
        if self.state == "OVERRIDE":
            self.msg = "new goal; will plan when you let go"
            return
        self._request_plan()
        if self.state in ("CENTERING", "MANUAL"):
            self.ctl.release()
            self.ctl.max_torque = self.full_torque
            self._set("IDLE")
        if self.state == "IDLE":
            self._set("PLANNING", "planning")

    def _request_plan(self):
        self._future = self.pool.submit(plan_path, (self.pose.x, self.pose.y), self.goal)

    def _start_wheel(self):
        """Hand the wheel to the motor gently: start from where it is, ramp torque up."""
        self.slew.reset(self.ctl.angle())
        self.ctl.max_torque = 0.0
        self._ramp_done = False
        self.ctl.goto(self.slew.value)

    def _drive_wheel(self, desired, angle, dt):
        """Slew toward `desired` wheel deg and send it. Returns the commanded angle."""
        lim = self.steering.wheel_limit
        target = self.slew.step(max(-lim, min(lim, desired)), dt)
        if not self._ramp_done:
            # Motor is still weak: keep the command on a short leash to the wheel so lag
            # can't pile up and get released as a lunge (or a false takeover).
            target = max(angle - 20.0, min(angle + 20.0, target))
            self.slew.reset(target)
            self.ctl.max_torque = min(self.full_torque, self.ctl.max_torque
                                      + self.full_torque * dt / self.cfg.ramp)
            self._ramp_done = self.ctl.max_torque >= self.full_torque
        self.ctl.goto(target)
        return target

    def _center(self):
        """Drive the wheel back to 0 deg, then release it - see step()."""
        self._start_wheel()
        self._centered_t = 0.0
        self._set("CENTERING", "centering wheel")

    def _engage(self):
        self._start_wheel()
        self.detector.reset()
        self._set("AUTO", "following path")

    def _manual(self):
        """UI drives the wheel picture; the motor turns the real wheel to match."""
        self._future = None
        lim = self.steering.wheel_limit
        self.manual_deg = max(-lim, min(lim, self.ctl.angle()))
        self._start_wheel()
        self._set("MANUAL", "manual: drag the wheel / arrow keys to steer")

    def _poll_plan(self):
        f = self._future
        if f is None or not f.done():
            return
        self._future = None
        try:
            res = f.result()
        except Exception as e:                       # worker crashed
            res = {"error": f"planner failed: {e}"}
        self.plan_stats = {"ms": round(res.get("ms", 0), 1), "expanded": res.get("expanded", 0)}
        if "error" in res:
            if self.state == "PLANNING":
                self._set("IDLE", res["error"])
            elif self.state == "REPLANNING":
                self._set("OVERRIDE", res["error"] + "; still yours")   # retried on next release
            else:
                self.msg = res["error"]
            return
        self.tracker.set_path(res["path"], self.pose)
        self.hub.set_path(res["path"])
        if self.state in ("PLANNING", "REPLANNING"):
            self._engage()

    def _handle_commands(self):
        while not self.hub.commands.empty():
            c = self.hub.commands.get_nowait()
            try:
                kind = c.get("cmd")
                if kind == "goal":
                    self.set_goal(float(c["x"]), float(c["y"]))
                elif kind == "stop":
                    self.stop()
                elif kind == "pose" and self.state == "IDLE":
                    self.pose = Pose(float(c["x"]), float(c["y"]),
                                     float(c.get("yaw", self.pose.yaw)))
                elif kind == "manual":
                    if self.state != "MANUAL":
                        self._manual()
                elif kind == "steer" and self.state == "MANUAL":
                    d = float(c["deg"])
                    if math.isfinite(d):
                        self.manual_deg = d
                elif kind == "speed":
                    v = float(c["v"])
                    if math.isfinite(v):
                        self.manual_speed = max(0.0, min(2.0 * self.cfg.speed + 2.0, v))
                elif kind == "reset":
                    self.stop("reset")
                    self.pose = Pose(**vars(self.start_pose))
                    self._center()
            except (KeyError, TypeError, ValueError):
                self.msg = f"bad command: {c!r}"[:120]

    # ---------- one control tick ----------
    def step(self, dt, now):
        self._handle_commands()
        self._poll_plan()

        ctl = self.ctl
        angle, vel, ht = ctl.angle(), ctl.last_velocity, ctl.human_torque
        target = None

        if self.state == "AUTO":
            err = angle - self.slew.value if self._ramp_done else 0.0
            if self.cfg.override and self.detector.takeover(ht, err, dt):
                ctl.release()
                ctl.max_torque = self.full_torque
                self.overrides += 1
                self.detector.reset()
                self._set("OVERRIDE", "human steering: " + self.detector.reason)
            else:
                steer, self.info = self.tracker.update(self.pose, self.cfg.speed)
                if self.info["remaining"] < self.cfg.goal_tol:
                    self.stop("arrived")
                else:
                    target = self._drive_wheel(self.steering.steer_to_wheel(steer), angle, dt)
        elif self.state == "OVERRIDE":
            if self.goal and self._future is None and self.detector.released(ht, vel, dt):
                self.detector.reset()
                self._request_plan()
                self._set("REPLANNING", "you let go; replanning from here")
        elif self.state == "REPLANNING":
            self.detector.released(ht, vel, dt)       # keep the filter warm
        elif self.state == "MANUAL":
            target = self._drive_wheel(self.manual_deg, angle, dt)
        elif self.state == "CENTERING":
            target = self._drive_wheel(0.0, angle, dt)
            settled = abs(angle) < 2.0 and abs(vel) < 10.0
            self._centered_t = self._centered_t + dt if settled else 0.0
            if self._centered_t > 0.3:
                self.stop("wheel centered")

        # simulated car always steers with the physical wheel
        if self.state == "MANUAL":
            speed = self.manual_speed
        else:
            speed = self.cfg.speed if self.state in DRIVING else 0.0
        steer_actual = self.steering.wheel_to_steer(angle)
        self.pose = self.car.step(self.pose, speed, steer_actual, dt)
        self.collision = self.grid.blocked(self.pose.x, self.pose.y)

        if now - self._timing_t >= 1.0:
            self._timing = (ctl.pop_loop_dt_max() * 1000, self._loop_worst * 1000)
            self._loop_worst, self._timing_t = 0.0, now

        d = self.detector
        if self.log:
            self.log.writerow([f"{now:.3f}", self.state, f"{angle:.2f}",
                               "" if target is None else f"{target:.2f}", f"{vel:.1f}",
                               f"{ht:.5f}", f"{d.filtered:.5f}", f"{ctl.last_output:.4f}",
                               f"{ctl.max_torque:.3f}", f"{self.pose.x:.2f}",
                               f"{self.pose.y:.2f}"])
        self.hub.add_sample([round(now, 3), round(angle, 1),
                             None if target is None else round(target, 1),
                             round(d.filtered, 4), MODE_CODE[self.state]])
        tgt = self.info.get("target") if self.state == "AUTO" else None
        self.hub.publish({
            "mode": self.state, "msg": self.msg, "t": round(now, 2),
            "x": round(self.pose.x, 3), "y": round(self.pose.y, 3),
            "yaw": round(self.pose.yaw, 4), "steer": round(steer_actual, 4),
            "speed": speed, "wheel": round(angle, 1),
            "wheel_limit": self.steering.wheel_limit, "manual_speed": self.manual_speed,
            "target": None if target is None else round(target, 1),
            "vel": round(vel, 1), "torque": round(d.filtered, 4),
            "enter": d.torque_enter, "exit": d.torque_exit,
            "lookahead_pt": tgt and [round(tgt[0], 2), round(tgt[1], 2)],
            "cte": round(self.info.get("cte", 0.0), 2) if self.state == "AUTO" else None,
            "remaining": round(self.info.get("remaining", 0.0), 1) if self.goal else None,
            "goal": self.goal, "plan": self.plan_stats, "overrides": self.overrides,
            "collision": self.collision, "torque_scale": round(ctl.max_torque, 3),
            "wheel_dt_ms": round(self._timing[0], 2), "loop_dt_ms": round(self._timing[1], 2),
        })

    def run(self):
        period = 1.0 / self.cfg.rate
        prev = time.monotonic()
        while True:
            now = time.monotonic()
            dt = now - prev
            prev = now
            self._loop_worst = max(self._loop_worst, dt)
            self.step(min(dt, 0.1), now)
            time.sleep(max(0.0, period - (time.monotonic() - now)))


def build_detector(args, model):
    noise = (model or {}).get("noise", 0.0)
    enter = args.takeover_torque
    exit_ = args.release_torque
    if model:
        enter = enter if enter is not None else max(5 * noise, 0.03)
        exit_ = exit_ if exit_ is not None else max(2 * noise, 0.015)
    angle = args.takeover_angle
    if angle is None:
        angle = None if model else 45.0          # with a model, hand torque is the signal
    elif angle <= 0:
        angle = None
    return OverrideDetector(enter, exit_, angle_enter_deg=angle,
                            enter_s=args.takeover_time, exit_s=args.release_time)


def main():
    ap = argparse.ArgumentParser(description="Path-following autopilot for a MOZA wheel.")
    a = ap.add_argument
    a("--map", default=os.path.join(HERE, "maps", "default.txt"))
    a("--resolution", type=float, default=0.5, help="map cell size, m")
    a("--inflate", type=float, default=1.5, help="keep path this far from obstacles, m")
    a("--go", action="store_true", help="drive to the map's G on startup")
    a("--start-yaw", type=float, default=0.0, help="initial heading, deg (0 = +x)")
    # car
    a("--speed", type=float, default=2.0, help="car speed, m/s")
    a("--wheelbase", type=float, default=2.7)
    a("--max-steer", type=float, default=30.0, help="road-wheel lock, deg")
    a("--wheel-lock", type=float, default=360.0,
      help="steering-wheel angle each way at full road-wheel lock, deg (sets the ratio)")
    a("--lookahead", type=float, default=4.0, help="min pure-pursuit lookahead, m")
    a("--lookahead-gain", type=float, default=1.0, help="lookahead per m/s")
    a("--goal-tol", type=float, default=1.0, help="arrival radius, m")
    a("--rate", type=float, default=50.0, help="control loop rate, Hz")
    # wheel
    a("--device")
    a("--rotation", type=float, default=900, help="rotation range set in Boxflat, deg")
    a("--max-torque", type=float, default=0.3)
    a("--kp", type=float, default=0.04)
    a("--kd", type=float, default=0.0015)
    a("--invert", action="store_true")
    a("--slew", type=float, default=540.0,
      help="max commanded wheel speed, deg/s (too low makes tracking oscillate)")
    a("--ramp", type=float, default=0.5, help="seconds to ramp motor torque on engage")
    # override
    a("--override", action="store_true",
      help="enable human-takeover detection (off by default: hands-off driving only)")
    a("--takeover-torque", type=float, help="human torque to take over (default 5x noise)")
    a("--release-torque", type=float, help="human torque counted as hands-off (default 2x noise)")
    a("--takeover-angle", type=float,
      help="wheel-vs-command error to take over, deg (default: off with a wheel model, "
           "45 without; 0 = off)")
    a("--takeover-time", type=float, default=0.12, help="s the takeover signal must persist")
    a("--release-time", type=float, default=0.6, help="s of hands-off before replanning")
    # ui
    a("--host", default="127.0.0.1")
    a("--port", type=int, default=8080)
    a("--log", help="write a 50 Hz CSV (angle, command, torques, mode...) for tuning")
    args = ap.parse_args()

    grid = GridMap.load(args.map, args.resolution)
    plan_grid = grid.inflated(args.inflate)

    # Planner process first, before any threads exist.
    pool = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                               initializer=init_worker, initargs=(plan_grid,))
    pool.submit(int).result()

    model = None
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE) as f:
            model = json.load(f)
        print(f"loaded wheel model ({model.get('calibrated', '?')}), "
              f"noise {model.get('noise', 0) * 100:.2f}% FS")
    else:
        print("WARNING: no moza_model.json - run `python3 moza_force.py` once. Without it "
              "takeover is only detected by angle error (--takeover-angle).")

    dev = find_moza(args.device)
    print(f"Using {dev.name} at {dev.path}")
    ctl = WheelController(dev, args.rotation, args.max_torque, args.kp, args.kd,
                          args.invert, model=model)
    ctl.watchdog_s = 0.25
    if not args.invert:
        print("Detecting force direction in 2 s - hands off...")
        time.sleep(2)
        ctl.calibrate_direction()
    threading.Thread(target=ctl.run, daemon=True).start()

    detector = build_detector(args, model)
    if not args.override:
        print("human takeover: OFF (pass --override to enable)")
    fmt = lambda v: "off" if v is None else f"{v * 100:.1f}% FS"
    print(f"takeover: torque {fmt(detector.torque_enter)}, angle "
          f"{'off' if detector.angle_enter is None else f'{detector.angle_enter:.0f} deg'}; "
          f"release torque {fmt(detector.torque_exit)}")

    hub = DebugHub({"rows": grid.rows, "cols": grid.cols, "res": grid.res,
                    "occ": grid.to_rows(), "inflated": plan_grid.to_rows(),
                    "start": grid.start, "goal": grid.goal})
    srv = serve(hub, args.host, args.port)
    print(f"debug UI: http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}"
          f":{args.port}")

    pilot = Autopilot(ctl, grid, plan_grid, args, hub, pool, detector)
    log_fh = None
    if args.log:
        log_fh = open(args.log, "w", newline="")
        pilot.log = csv.writer(log_fh)
        pilot.log.writerow(["t", "mode", "angle", "command", "vel_dps", "human_torque",
                            "human_torque_filt", "motor_out", "torque_scale", "x", "y"])
    if args.go and grid.goal:
        pilot.set_goal(*grid.goal)
    try:
        pilot.run()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        if log_fh:
            log_fh.close()
        srv.shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
        print("\nReleased.")


if __name__ == "__main__":
    main()
