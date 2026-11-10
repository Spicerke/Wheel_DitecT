#!/usr/bin/env python3
"""
moza_drive.py - make a MOZA wheel base turn itself to positions / torques you command.

Uses the standard Linux force-feedback (evdev) interface exposed by the in-kernel
hid-universal-pidff driver, so no MOZA-specific protocol is needed.

How it works:
  * Reads the wheel's current angle from the steering axis (ABS_X).
  * Runs a PD (spring + damper) control loop at ~500 Hz and sends the result to the
    base as a continuously updated FF_CONSTANT force effect.
  * Commands come from stdin (interactive) or from Python if you import WheelController.

Requirements:
  pip install evdev
  Read/write access to the wheel's /dev/input/event* node (see udev rule in the notes).

Interactive commands:
  <number>        go to that angle in degrees (0 = center, + = right, - = left)
  t <value>       apply raw torque, -1.0 .. 1.0 (scaled by --max-torque)
  free            release the wheel (zero force)
  pos             print current angle
  q               quit

SAFETY: a direct-drive base can spin hard. Keep hands and cables clear, start with a
low --max-torque, and set a sane rotation range/power limit in Boxflat first.
"""

import argparse
import math
import sys
import threading
import time

import evdev
from evdev import ecodes, ff

MOZA_VENDOR_ID = 0x346E


def find_moza(path=None):
    if path:
        return evdev.InputDevice(path)
    for p in evdev.list_devices():
        dev = evdev.InputDevice(p)
        caps = dev.capabilities()
        if dev.info.vendor == MOZA_VENDOR_ID and ecodes.EV_FF in caps:
            return dev
    sys.exit("No MOZA device with force feedback found. Check permissions on "
             "/dev/input/event* or pass --device /dev/input/eventN.")


class WheelController:
    def __init__(self, dev, rotation_deg=900.0, max_torque=0.3,
                 kp=0.04, kd=0.0015, invert=False, rate_hz=500, model=None):
        self.dev = dev
        self.rotation = rotation_deg          # must match the range set in Boxflat
        self.max_torque = max(0.0, min(1.0, max_torque))
        self.kp, self.kd = kp, kd
        self.sign = -1.0 if invert else 1.0
        self.period = 1.0 / rate_hz

        self.absinfo = dev.absinfo(ecodes.ABS_X)
        self._lock = threading.Lock()
        self._mode = "free"                   # "free" | "position" | "torque"
        self._target_deg = 0.0
        self._torque_cmd = 0.0
        self._running = False

        self.last_output = 0.0                # most recent controller output, -1..1
        self.last_velocity = 0.0              # filtered wheel velocity, deg/s

        # Human-torque observer. Torques are in fractions of the device's full scale.
        # Wheel model: J*accel = motor + human - b*vel - fc*sign(vel)
        # J, b, fc come from `moza_force.py calibrate`; all zero = no model.
        self.J = self.b = self.fc = 0.0
        if model:
            self.set_model(model["J"], model["b"], model["fc"])
        self.observer_bw = 30.0               # observer bandwidth, rad/s (higher = faster, noisier)
        self._z = 0.0
        self.human_torque = 0.0               # estimated human torque, + = toward the right
        self._log = None                      # set to a list to record (t, pos, vel, motor)

        # Full device gain, disable built-in autocenter so it doesn't fight us.
        try:
            dev.write(ecodes.EV_FF, ecodes.FF_GAIN, 0xFFFF)
        except OSError:
            pass
        try:
            dev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, 0)
        except OSError:
            pass

        self.effect_id = self._upload_constant(0, effect_id=-1)
        dev.write(ecodes.EV_FF, self.effect_id, 1)   # start playing (infinite length)

    # ---------- low level ----------
    def _upload_constant(self, level, effect_id):
        level = int(max(-32767, min(32767, level)))
        effect = ff.Effect(
            ecodes.FF_CONSTANT, effect_id,
            0x4000,                              # direction: along the wheel axis
            ff.Trigger(0, 0),
            ff.Replay(0, 0),                     # length 0 = play until stopped
            ff.EffectType(ff_constant_effect=ff.Constant(level, ff.Envelope(0, 0, 0, 0))),
        )
        return self.dev.upload_effect(effect)

    def _apply(self, torque_norm):
        """torque_norm in -1..1, scaled by max_torque, sign-corrected."""
        t = max(-1.0, min(1.0, torque_norm)) * self.max_torque * self.sign
        self._upload_constant(t * 32767, self.effect_id)

    def angle(self):
        """Current wheel angle in degrees (0 = center)."""
        info = self.dev.absinfo(ecodes.ABS_X)
        mid = (info.max + info.min) / 2.0
        half = (info.max - info.min) / 2.0
        return (info.value - mid) / half * (self.rotation / 2.0)

    # ---------- public API ----------
    def goto(self, deg):
        lim = self.rotation / 2.0
        with self._lock:
            self._target_deg = max(-lim, min(lim, deg))
            self._mode = "position"

    def torque(self, value):
        with self._lock:
            self._torque_cmd = value
            self._mode = "torque"

    def release(self):
        with self._lock:
            self._mode = "free"

    def set_model(self, J, b, fc):
        self.J, self.b, self.fc = float(J), float(b), float(fc)
        self._z = 0.0

    def _observe(self, vel, motor, dt):
        """Disturbance observer: human = LPF(J*accel + b*vel + friction - motor)."""
        g = self.observer_bw
        friction = self.fc * math.tanh(vel / 1.0)
        u = -motor + self.b * vel + friction - g * self.J * vel
        self._z += min(dt, 0.01) * g * (u - self._z)
        self.human_torque = g * self.J * vel + self._z

    def calibrate_direction(self):
        """Nudge the wheel and flip the sign if positive torque turns it left."""
        start = self.angle()
        self.sign = 1.0
        self._apply(0.25)
        time.sleep(0.25)
        self._apply(0.0)
        moved = self.angle() - start
        if abs(moved) < 0.5:
            print(f"calibrate: wheel barely moved ({moved:.2f} deg); keeping default "
                  "direction. Use --invert if it turns the wrong way.")
            return
        if moved < 0:
            self.sign = -1.0
        print(f"calibrate: moved {moved:+.1f} deg, direction sign = {self.sign:+.0f}")
        time.sleep(0.3)

    def run(self):
        self._running = True
        prev = self.angle()
        prev_t = time.monotonic()
        vel = 0.0
        while self._running:
            now = time.monotonic()
            pos = self.angle()
            dt = max(now - prev_t, 1e-4)
            # low-pass filtered velocity (deg/s) for the damping term
            vel = 0.7 * vel + 0.3 * ((pos - prev) / dt)
            prev, prev_t = pos, now

            with self._lock:
                mode, target, tcmd = self._mode, self._target_deg, self._torque_cmd

            if mode == "position":
                out = self.kp * (target - pos) - self.kd * vel
            elif mode == "torque":
                out = tcmd
            else:
                out = 0.0
            self.last_output = max(-1.0, min(1.0, out))   # -1..1, before max_torque scaling
            self.last_velocity = vel
            motor = self.last_output * self.max_torque   # torque actually sent, full-scale units
            self._observe(vel, motor, dt)
            if self._log is not None:
                self._log.append((now, pos, vel, motor))
            self._apply(out)

            time.sleep(max(0.0, self.period - (time.monotonic() - now)))

    def stop(self):
        self._running = False
        try:
            self._apply(0.0)
            self.dev.write(ecodes.EV_FF, self.effect_id, 0)
            self.dev.erase_effect(self.effect_id)
        except OSError:
            pass


def repl(ctl):
    print("Commands: <deg> | t <-1..1> | free | pos | q")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if not cmd:
            continue
        try:
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "free":
                ctl.release()
            elif cmd == "pos":
                print(f"{ctl.angle():+.1f} deg")
            elif cmd.startswith("t "):
                ctl.torque(float(cmd.split()[1]))
            else:
                ctl.goto(float(cmd))
        except ValueError:
            print("?", cmd)


def main():
    ap = argparse.ArgumentParser(description="Drive a MOZA wheel base from commands.")
    ap.add_argument("--device", help="/dev/input/eventN (auto-detected if omitted)")
    ap.add_argument("--rotation", type=float, default=900,
                    help="total rotation range in degrees, as set in Boxflat (default 900)")
    ap.add_argument("--max-torque", type=float, default=0.3,
                    help="cap on output force, 0..1 of device max (default 0.3)")
    ap.add_argument("--kp", type=float, default=0.04, help="stiffness per degree of error")
    ap.add_argument("--kd", type=float, default=0.0015, help="damping per deg/s")
    ap.add_argument("--invert", action="store_true", help="flip force direction")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the startup nudge that detects force direction")
    args = ap.parse_args()

    dev = find_moza(args.device)
    print(f"Using {dev.name} at {dev.path}")
    ctl = WheelController(dev, args.rotation, args.max_torque, args.kp, args.kd, args.invert)

    if not args.no_calibrate and not args.invert:
        print("Calibrating direction in 2 s - hands off the wheel...")
        time.sleep(2)
        ctl.calibrate_direction()

    loop = threading.Thread(target=ctl.run, daemon=True)
    loop.start()
    try:
        repl(ctl)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        print("Released.")


if __name__ == "__main__":
    main()
