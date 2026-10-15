"""Identify a car's steering response from logged laps.

    python3 scripts/identify_steering.py                       # all completed CARLA runs
    python3 scripts/identify_steering.py outputs/carla/bench_*  # chosen runs

For each sample the effective road-wheel angle is recovered from the measured yaw
rate with the bicycle relation  delta = atan(wheelbase * yaw_rate / v), and compared
with the normalised command actually applied one step earlier. Fits

    delta = u * A / (1 + c * v^2)

and prints A (full_command_angle_rad) and c (speed_coeff_s2pm2) for the vehicle JSON.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*")
    p.add_argument("--wheelbase", type=float, default=None, help="Default: the value stored in each run.")
    p.add_argument("--min-speed", type=float, default=3.0)
    args = p.parse_args()

    runs = args.runs or sorted(glob.glob(str(PROJECT_ROOT / "outputs/carla/bench_*")))
    speeds, slopes_num, slopes_den, rows = [], [], [], []
    for run in runs:
        meta_path, log_path = Path(run) / "meta.json", Path(run) / "run_log.csv"
        if not meta_path.exists() or not log_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("plant") != "carla" or meta.get("status") != "completed":
            continue
        wheelbase = args.wheelbase or float(meta.get("wheelbase_m", 2.51))
        with log_path.open() as f:
            data = list(csv.DictReader(f))
        col = lambda k: np.array([float(r[k]) for r in data])  # noqa: E731
        t, v = col("t_s"), col("v_mps")
        yaw = np.unwrap(col("yaw_rapp_rad"))
        command = -col("steer_norm")                 # logged in CARLA's sign convention
        yaw_rate = np.gradient(yaw, t)
        delta = np.arctan(wheelbase * yaw_rate[2:] / np.maximum(v[2:], 0.1))
        rows.append(np.column_stack((v[2:], delta, command[1:-1])))
    if not rows:
        raise SystemExit("No completed CARLA runs found.")
    v, delta, u = np.vstack(rows).T
    keep = (v > args.min_speed) & (np.abs(u) > 0.08) & (np.abs(u) < 0.9)
    v, delta, u = v[keep], delta[keep], u[keep]

    print(f"{keep.sum()} samples from {len(rows)} runs")
    print("speed bin   angle per unit command (rad)")
    centres, gains = [], []
    for lo in np.arange(args.min_speed, 20.0, 2.0):
        m = (v >= lo) & (v < lo + 2.0)
        if m.sum() < 200:
            continue
        k = float(np.sum(delta[m] * u[m]) / np.sum(u[m] ** 2))
        centres.append(lo + 1.0)
        gains.append(k)
        print(f"  {lo:4.0f}-{lo + 2:4.0f} m/s   {k:.3f}   (n={m.sum()})")
    centres, gains = np.array(centres), np.array(gains)
    # 1/gain = (1 + c v^2) / A is linear in v^2
    slope, intercept = np.polyfit(centres ** 2, 1.0 / gains, 1)
    full_angle, coeff = 1.0 / intercept, slope / intercept
    print(f"\nfull_command_angle_rad = {full_angle:.3f}   ({np.degrees(full_angle):.1f} deg)")
    print(f"speed_coeff_s2pm2      = {coeff:.4f}")
    for vv, g in zip(centres, gains):
        print(f"  check {vv:4.0f} m/s: measured {g:.3f}, model {full_angle / (1 + coeff * vv * vv):.3f}")


if __name__ == "__main__":
    main()
