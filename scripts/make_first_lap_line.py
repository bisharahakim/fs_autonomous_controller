"""First-lap (exploration) reference: the track centreline with a cautious speed profile.

On the first lap there is no optimised racing line yet, so the car follows the middle
of the track at a speed it can hold without knowing what comes next. This writes that
reference in the same CSV format as the racing lines:

    python3 scripts/make_first_lap_line.py                 # defaults: ay 5 m/s^2, v_max 10 m/s
    python3 scripts/make_first_lap_line.py --ay 4 --v-max 8

-> inputs/racelines/hockenheim_fsg_first_lap.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.track_registry import get_track  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="hockenheim_fsg")
    p.add_argument("--name", default="hockenheim_fsg_first_lap")
    p.add_argument("--ay", type=float, default=5.0, help="Lateral acceleration the speed plan allows (m/s^2).")
    p.add_argument("--v-max", type=float, default=10.0, help="Top speed on the first lap (m/s).")
    p.add_argument("--accel", type=float, default=2.5, help="Planned acceleration (m/s^2).")
    p.add_argument("--decel", type=float, default=4.0, help="Planned braking (m/s^2).")
    p.add_argument("--step", type=float, default=0.5, help="Point spacing (m).")
    p.add_argument("--smooth-cm", type=float, default=5.0, help="Spline smoothing of the centreline (cm).")
    args = p.parse_args()

    pts = get_track(args.track).build_centerline()
    xy = np.array([[q.x, q.y] for q in pts], dtype=float)
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    seg = np.hypot(*np.diff(np.vstack((xy, xy[0])), axis=0).T)
    xy = xy[seg > 1e-3]                                # drop repeated points (the spline rejects them)
    seg = np.hypot(*np.diff(np.vstack((xy, xy[0])), axis=0).T)
    s_raw = np.concatenate(([0.0], np.cumsum(seg)))
    grid = np.arange(0.0, s_raw[-1] - 1e-6, 1.0)       # resample every 1 m before fitting
    ring = np.vstack((xy, xy[0]))
    xy = np.column_stack((np.interp(grid, s_raw, ring[:, 0]), np.interp(grid, s_raw, ring[:, 1])))
    closed = np.vstack((xy, xy[0]))
    tck, _ = splprep([closed[:, 0], closed[:, 1]], k=5, s=len(closed) * (args.smooth_cm / 100.0) ** 2, per=1)

    u_fine = np.linspace(0.0, 1.0, 40001)
    fine = np.array(splev(u_fine, tck)).T
    s_fine = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(fine, axis=0).T))))
    total = float(s_fine[-1])
    n = int(round(total / args.step))
    s = np.arange(n) * total / n
    u = np.interp(s, s_fine, u_fine)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    psi = np.arctan2(dy, dx) - np.pi / 2.0            # same heading convention as the racing lines
    psi = (psi + np.pi) % (2.0 * np.pi) - np.pi

    # Speed: corner limit, then acceleration and braking limits (two passes around the closed loop).
    v = np.minimum(args.v_max, np.sqrt(args.ay / np.maximum(np.abs(kappa), 1e-6)))
    ds = total / n
    for _ in range(2):
        for i in range(1, 2 * n):                     # forward: acceleration
            a, b = (i - 1) % n, i % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * args.accel * ds))
        for i in range(2 * n - 2, -1, -1):            # backward: braking
            a, b = (i + 1) % n, i % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2.0 * args.decel * ds))
    ax = (np.roll(v, -1) ** 2 - v ** 2) / (2.0 * ds)

    rows = np.column_stack((s, x, y, psi, kappa, v, ax))
    rows = np.vstack((rows, np.concatenate(([total], rows[0, 1:]))))
    out = PROJECT_ROOT / "inputs" / "racelines" / f"{args.name}.csv"
    with out.open("w") as f:
        f.write(f"# first-lap reference: track centreline, ay {args.ay}, v_max {args.v_max}, "
                f"accel {args.accel}, decel {args.decel}\n# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        for r in rows:
            f.write("; ".join(f"{val:.7f}" for val in r) + "\n")
    lap = float(np.sum(ds / np.maximum(v, 0.1)))
    print(f"Wrote {out}\n  length {total:.1f} m, tightest radius {1 / np.abs(kappa).max():.1f} m, "
          f"speed {v.min():.1f}-{v.max():.1f} m/s, planned lap {lap:.1f} s")


if __name__ == "__main__":
    main()
