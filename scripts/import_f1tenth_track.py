"""Turn an F1TENTH race-track centreline into a cone track for the CARLA benchmark.

Source data: github.com/f1tenth/f1tenth_racetracks (GPL-3.0), scaled-down real circuits
with a fixed 2.2 m width. Only the centreline shape is used; the cone corridor width is
set separately (5 m for the Audi, like the Hockenheim layout).

    git clone --depth 1 https://github.com/f1tenth/f1tenth_racetracks.git ~/f1tenth_racetracks
    python3 scripts/import_f1tenth_track.py ~/f1tenth_racetracks/Montreal/Montreal_centerline.csv \\
        --name montreal --min-radius 5.0

Scaling: by default the shape is scaled so that its tightest corner (centreline radius)
is --min-radius; or give --scale directly. The start is moved to the origin, heading +x,
like the Hockenheim layout, so the car spawns in the same place in the CARLA level.

Writes inputs/tracks/<name>_centreline.csv and prints lap length, size, tightest radius
and the closest approach between different parts of the track (they must not overlap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.csv_track import TRACKS_DIR  # noqa: E402


def smooth_loop(xy: np.ndarray, smooth_m: float, n_out: int):
    seg = np.hypot(*np.diff(np.vstack((xy, xy[:1])), axis=0).T)
    xy = xy[seg > 1e-6]
    closed = np.vstack((xy, xy[:1]))
    tck, _ = splprep(closed.T, s=len(closed) * smooth_m ** 2, per=1, k=3)
    u = np.linspace(0.0, 1.0, n_out, endpoint=False)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    return np.column_stack((x, y)), kappa


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    ring = np.vstack((xy, xy[:1]))
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(ring, axis=0).T))))
    grid = np.arange(0.0, s[-1] - 1e-9, step)
    return np.column_stack((np.interp(grid, s, ring[:, 0]), np.interp(grid, s, ring[:, 1])))


def radius_profile(xy: np.ndarray, window_m: float = 2.0) -> np.ndarray:
    """Curvature radius per point, averaged over a short window (removes single-point spikes)."""
    step = float(np.median(np.hypot(*np.diff(xy, axis=0).T)))
    nxt, prv = np.roll(xy, -1, axis=0), np.roll(xy, 1, axis=0)
    heading = np.unwrap(np.arctan2(nxt[:, 1] - prv[:, 1], nxt[:, 0] - prv[:, 0]))
    dpsi = np.diff(np.r_[heading, heading[0] + (heading[-1] - heading[0]) + (heading[1] - heading[0])])
    n = max(1, int(round(window_m / step / 2)))
    k = np.abs(dpsi) / step
    k = np.convolve(np.r_[k[-n:], k, k[:n]], np.ones(2 * n + 1) / (2 * n + 1), mode="valid")
    return 1.0 / np.maximum(k, 1e-9)


def closest_other_part(xy: np.ndarray, skip_m: float = 30.0) -> float:
    """Smallest distance between two centreline points that are > skip_m apart along the lap."""
    step = float(np.median(np.hypot(*np.diff(xy, axis=0).T)))
    skip = int(skip_m / step)
    tree = cKDTree(xy)
    best = np.inf
    for i, p in enumerate(xy):
        for j in tree.query_ball_point(p, 25.0):
            gap = abs(i - j)
            gap = min(gap, len(xy) - gap)
            if gap > skip:
                best = min(best, float(np.hypot(*(xy[j] - p))))
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="f1tenth_racetracks <Track>_centerline.csv")
    p.add_argument("--name", required=True, help="Track name for the benchmark (e.g. montreal).")
    p.add_argument("--min-radius", type=float, default=5.0,
                   help="Scale the shape so its tightest centreline radius is this (m).")
    p.add_argument("--scale", type=float, default=None, help="Fixed scale factor instead of --min-radius.")
    p.add_argument("--reverse", action="store_true", help="Drive the track the other way round.")
    p.add_argument("--smooth-cm", type=float, default=3.0, help="Smoothing of the source centreline (cm, source units).")
    p.add_argument("--corridor", type=float, default=5.0, help="Cone corridor the track will get (for the overlap check).")
    p.add_argument("--compact", action="store_true",
                   help="Rotate the layout to the angle that makes its bounding square smallest (long tracks "
                        "then lie diagonally) and centre it on --center.")
    p.add_argument("--center", type=float, nargs=2, default=(75.0, 75.0), metavar=("X", "Y"),
                   help="With --compact: where to put the middle of the layout (track frame). The default is the "
                        "middle of the ground of FSD_Empty_Asphalt (examples/probe_ground.py: about x, y -250..400 m).")
    p.add_argument("--keep-origin", action="store_true",
                   help="Leave the start at the origin. By default the layout is shifted onto the patch of "
                        "ground the Hockenheim layout uses, which is known to be drivable in the CARLA level.")
    args = p.parse_args()

    src = np.loadtxt(args.source, delimiter=",", comments="#", usecols=(0, 1))
    if args.reverse:
        src = src[::-1]
    shape, _ = smooth_loop(src, args.smooth_cm / 100.0, 20000)
    shape = resample(shape, 0.05)
    if args.scale:
        scale = args.scale
    else:                                           # radius is measured on the scaled track, so iterate
        scale = 1.0
        for _ in range(6):
            scale *= args.min_radius / radius_profile(resample(shape * scale, 0.5)).min()
    xy = resample(shape * scale, 0.5)

    # Start at the origin heading +x (the Hockenheim layout starts the same way).
    heading = np.arctan2(*(xy[1] - xy[0])[::-1])
    c, s = np.cos(-heading), np.sin(-heading)
    xy = (xy - xy[0]) @ np.array([[c, -s], [s, c]]).T

    if args.compact:
        best = None
        for deg in np.arange(0.0, 180.0, 0.5):
            a = np.radians(deg)
            rot = xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
            side = float(np.max(rot.max(axis=0) - rot.min(axis=0)))
            if best is None or side < best[0]:
                best = (side, rot)
        xy = best[1]
        xy = xy - 0.5 * (xy.max(axis=0) + xy.min(axis=0)) + np.asarray(args.center)
    elif not args.keep_origin:
        # Hockenheim layout footprint in the track frame (x -22..169, y -0.5..196 m).
        foot_lo, foot_hi = np.array([-22.0, -0.5]), np.array([169.0, 196.0])
        size = xy.max(axis=0) - xy.min(axis=0)
        xy = xy - xy.min(axis=0) + foot_lo
        if np.any(size > foot_hi - foot_lo):
            print(f"WARNING: the layout ({size[0]:.0f} x {size[1]:.0f} m) is larger than the Hockenheim footprint "
                  f"({(foot_hi - foot_lo)[0]:.0f} x {(foot_hi - foot_lo)[1]:.0f} m); check the level's ground "
                  "with examples/probe_ground.py.")
    radius = radius_profile(xy)
    length = float(np.sum(np.hypot(*np.diff(np.vstack((xy, xy[:1])), axis=0).T)))
    gap = closest_other_part(xy)
    lo, hi = xy.min(axis=0), xy.max(axis=0)

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    out = TRACKS_DIR / f"{args.name}_centreline.csv"
    with out.open("w") as f:
        f.write(f"# {args.name}: centreline from f1tenth_racetracks ({args.source.name}, GPL-3.0), "
                f"scaled x{scale:.3f}, start heading +x{', reversed' if args.reverse else ''}"
                f"{', rotated to fit and centred' if args.compact else '' if args.keep_origin else ', placed on the Hockenheim footprint'}\n")
        f.write("# x_m, y_m\n")
        for x, y in xy:
            f.write(f"{x:.4f}, {y:.4f}\n")

    print(f"Wrote {out}")
    print(f"  scale x{scale:.3f}  lap {length:.0f} m  size {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} m "
          f"(x {lo[0]:.0f}..{hi[0]:.0f}, y {lo[1]:.0f}..{hi[1]:.0f})")
    print(f"  tightest centreline radius {radius.min():.1f} m")
    clearance = gap - args.corridor
    print(f"  closest approach of two different parts of the lap: {gap:.1f} m "
          f"({'OK' if clearance > 2.0 else 'TOO CLOSE: cones of the two parts would overlap'}, "
          f"{clearance:.1f} m between their cone rows at a {args.corridor:.0f} m corridor)")


if __name__ == "__main__":
    main()
