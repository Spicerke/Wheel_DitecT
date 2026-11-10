#!/usr/bin/env python3
"""
moza_force.py - turn the wheel by hand, read the torque you're applying.

MOZA bases have no torque sensor the PC can read, so the force is estimated from how the
wheel moves: torque = inertia * acceleration + damping * velocity + friction
(plus whatever the motor is doing, which we know because we command it).

The inertia/damping/friction of YOUR wheel + rim are measured once, automatically:
the first run spins the wheel back and forth by itself for ~20 s (hands off!) and saves
the result to moza_model.json. After that it goes straight to reading.

Modes:
  default   wheel is free (motor off). You feel nothing; readings come from motion,
            so holding the wheel still and pushing reads ~0.
  --hold    a soft spring holds the wheel where it starts, so pushing against it
            without moving much still gives a reading.

The same estimate is available in code as WheelController.human_torque, so it keeps
working when an algorithm is driving the wheel with ctl.goto().

Examples:
  python3 moza_force.py --peak-nm 5.5            # R5, print Nm
  python3 moza_force.py --hold --csv log.csv     # resist, and log to CSV
  python3 moza_force.py --recalibrate            # redo the model (e.g. new rim)

Requires: pip install evdev numpy, and moza_drive.py in the same folder.
"""

import argparse
import csv
import json
import math
import os
import threading
import time

import numpy as np

from moza_drive import WheelController, find_moza

if not hasattr(WheelController, "set_model"):
    raise SystemExit("moza_drive.py next to this script is an old version. Replace it with the "
                     "latest moza_drive.py (the one that contains 'human_torque').")

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moza_model.json")


def excite(ctl, seconds, amp_deg):
    """Hands-off: drive the wheel through a mix of slow and fast swings."""
    parts = [(0.3, 1.0), (0.7, 0.5), (1.3, 0.25), (2.1, 0.12)]   # (Hz, weight)
    scale = amp_deg / sum(w for _, w in parts)
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        if t >= seconds:
            break
        ctl.goto(scale * sum(w * math.sin(2 * math.pi * f * t + i)
                             for i, (f, w) in enumerate(parts)))
        time.sleep(0.005)
    ctl.goto(0.0)
    time.sleep(1.0)


def fit_model(log):
    """Least-squares fit of motor = J*accel + b*vel + fc*sign(vel) (no human touching)."""
    if len(log) < 500:
        raise SystemExit(f"Calibration recorded only {len(log)} samples. Make sure moza_drive.py "
                         "is the latest version (it must contain 'human_torque').")
    t, pos, _, motor = np.array(log).T
    k = np.ones(11) / 11
    vel = np.convolve(np.gradient(pos, t), k, "same")
    acc = np.convolve(np.gradient(vel, t), k, "same")
    mot = np.convolve(motor, k, "same")
    m = slice(100, -100)                                 # drop filter edge effects
    X = np.column_stack([acc, vel, np.tanh(vel / 1.0)])[m]
    (J, b, fc), *_ = np.linalg.lstsq(X, mot[m], rcond=None)
    pred = X @ np.array([J, b, fc])
    r2 = 1 - np.var(mot[m] - pred) / max(np.var(mot[m]), 1e-12)
    return max(J, 0.0), max(b, 0.0), max(fc, 0.0), r2


def calibrate(ctl, amp_deg):
    print("\nCALIBRATING - hands OFF the wheel, keep the area clear. Starting in 3 s...")
    time.sleep(3)
    ctl._log = []
    excite(ctl, 20, amp_deg)
    log, ctl._log = ctl._log, None
    J, b, fc, r2 = fit_model(log)
    ctl.set_model(J, b, fc)
    print(f"model: inertia={J:.3e} damping={b:.3e} friction={fc:.3e}  (fit R^2={r2:.2f})")
    if r2 < 0.6:
        print("warning: poor fit - readings will be rough. Try a larger --max-torque, or "
              "turn off Boxflat's damper/friction/inertia filters, then --recalibrate.")

    # Hands-off check: whatever the estimator reports now is noise.
    samples = []
    t0 = time.monotonic()
    stop = threading.Event()
    threading.Thread(target=lambda: (excite(ctl, 8, amp_deg), stop.set()), daemon=True).start()
    while not stop.is_set():
        samples.append(ctl.human_torque)
        time.sleep(0.01)
    noise = float(np.sqrt(np.mean(np.square(samples)))) if samples else 0.0
    print(f"noise floor (hands off): {noise * 100:.2f}% of full scale RMS")

    model = {"J": J, "b": b, "fc": fc, "r2": r2, "noise": noise,
             "rotation": ctl.rotation, "calibrated": time.strftime("%Y-%m-%d %H:%M")}
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"saved {MODEL_FILE}\n")
    return model


def main():
    ap = argparse.ArgumentParser(description="Read the torque applied to a MOZA wheel.")
    ap.add_argument("--device", help="/dev/input/eventN (auto-detected if omitted)")
    ap.add_argument("--rotation", type=float, default=900,
                    help="total rotation range in degrees, as set in Boxflat")
    ap.add_argument("--hold", action="store_true",
                    help="softly hold the wheel so static pushes register")
    ap.add_argument("--max-torque", type=float, default=0.3,
                    help="motor force cap for calibration/--hold, 0..1 (default 0.3)")
    ap.add_argument("--peak-nm", type=float, default=None,
                    help="base's peak torque to print Nm "
                         "(R3 3.9, R5 5.5, R9 9, R12 12, R16 16, R21 21)")
    ap.add_argument("--ffb-percent", type=float, default=100,
                    help="force feedback strength %% set in Boxflat (default 100)")
    ap.add_argument("--recalibrate", action="store_true", help="re-measure the wheel model")
    ap.add_argument("--cal-amp", type=float, default=40, help="calibration swing, degrees")
    ap.add_argument("--rate", type=float, default=20, help="print rate in Hz")
    ap.add_argument("--csv", help="also write readings to this CSV file")
    ap.add_argument("--invert", action="store_true", help="flip force direction")
    args = ap.parse_args()

    dev = find_moza(args.device)
    print(f"Using {dev.name} at {dev.path}")
    ctl = WheelController(dev, args.rotation, args.max_torque, kp=0.04, kd=0.0015,
                          invert=args.invert)
    if not args.invert:
        print("Detecting force direction in 2 s - hands off...")
        time.sleep(2)
        ctl.calibrate_direction()
    threading.Thread(target=ctl.run, daemon=True).start()

    model = None
    if os.path.exists(MODEL_FILE) and not args.recalibrate:
        with open(MODEL_FILE) as f:
            model = json.load(f)
        ctl.set_model(model["J"], model["b"], model["fc"])
        print(f"loaded model from {MODEL_FILE} ({model.get('calibrated', '?')})")
    else:
        model = calibrate(ctl, args.cal_amp)

    if args.hold:
        ctl.kp, ctl.kd = 0.01, 0.0005          # soft spring
        ctl.goto(ctl.angle())
    else:
        ctl.release()                          # motor off, wheel free
    time.sleep(0.3)

    # Unit conversion: estimator works in fractions of the base's full-scale torque.
    nm = args.peak_nm * args.ffb_percent / 100.0 if args.peak_nm else None
    unit = "Nm" if nm else "% full scale"
    threshold = 4 * model.get("noise", 0.0)

    writer = fh = None
    if args.csv:
        fh = open(args.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["time_s", "angle_deg", "velocity_dps", "torque_frac", "torque_nm"])

    print(f"Turn the wheel. Positive = turning right. Ctrl+C to stop.")
    print(f"{'angle':>8} {'torque (' + unit + ')':>18}")
    t0 = time.monotonic()
    smoothed = 0.0
    try:
        while True:
            smoothed = 0.7 * smoothed + 0.3 * ctl.human_torque
            shown = 0.0 if abs(smoothed) < threshold else smoothed   # hide noise at rest
            val = f"{shown * nm:+.2f}" if nm else f"{shown * 100:+.1f}"
            print(f"\r{ctl.angle():+8.1f} {val:>18}   ", end="", flush=True)
            if writer:
                writer.writerow([f"{time.monotonic() - t0:.3f}", f"{ctl.angle():.2f}",
                                 f"{ctl.last_velocity:.1f}", f"{smoothed:.5f}",
                                 f"{smoothed * nm:.4f}" if nm else ""])
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.stop()
        if fh:
            fh.close()
        print("\nReleased.")


if __name__ == "__main__":
    main()
