"""Minimum-curvature raceline for a given corridor width and vehicle.

Uses the same core as the TUM global_racetrajectory_optimization "mincurv" mode
(the `trajectory_planning_helpers` package): minimise curvature inside the
corridor, then compute a speed profile from the vehicle's acceleration limits.

    pip install --no-deps trajectory-planning-helpers && pip install quadprog
    python3 scripts/optimize_raceline.py

Writes inputs/racelines/<name>.csv in the same format the controllers already
read: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2 (last row closes the lap).

The defaults describe the CARLA Audi TT on the 5.0 m cone corridor. Its limits
were measured from benchmark logs (see the comments on each constant).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trajectory_planning_helpers as tph
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.track_registry import get_track  # noqa: E402

# Engine-limited acceleration [speed m/s, m/s^2], measured at the benchmark's 0.75 throttle cap.
AX_MACHINE = np.array([[0.0, 3.4], [6.0, 3.0], [10.0, 2.4], [14.0, 2.0], [18.0, 1.5], [22.0, 1.1], [30.0, 0.8]])


def closed_centerline(track_name: str, spacing_m: float) -> np.ndarray:
    points = get_track(track_name).build_centerline()
    xy = np.array([[p.x, p.y] for p in points], dtype=float)
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    xy_closed = np.vstack((xy, xy[0]))
    samples = np.arange(0.0, s[-1] - 1e-9, spacing_m)
    return np.column_stack((np.interp(samples, s, xy_closed[:, 0]), np.interp(samples, s, xy_closed[:, 1])))


def optimise(args) -> dict:
    centre = closed_centerline(args.track, args.step_opt)
    half = args.track_width / 2.0
    reftrack = np.column_stack((centre, np.full(len(centre), half), np.full(len(centre), half)))

    closed_path = np.vstack((reftrack[:, :2], reftrack[0, :2]))
    coeffs_x, coeffs_y, a_matrix, normvec = tph.calc_splines.calc_splines(path=closed_path)

    # Effective vehicle width = body + (cone radius + safety margin) on both sides.
    w_veh = args.car_width + 2.0 * (args.cone_radius + args.margin)
    if w_veh >= args.track_width:
        raise SystemExit(f"Effective width {w_veh:.2f} m does not fit a {args.track_width:.2f} m corridor.")
    kappa_bound = 1.0 / args.min_radius

    alpha, _ = tph.opt_min_curv.opt_min_curv(
        reftrack=reftrack, normvectors=normvec, A=a_matrix, kappa_bound=kappa_bound, w_veh=w_veh, closed=True,
    )
    raw_line = tph.create_raceline.create_raceline(
        refline=reftrack[:, :2], normvectors=normvec, alpha=alpha, stepsize_interp=args.step_out,
    )[0]

    # The raw solution has curvature ripple in the tightest corner (single points far
    # tighter than their neighbours). A smoothing spline that moves the line by a few
    # centimetres removes it; heading and curvature are then taken from that spline.
    closed = np.vstack((raw_line, raw_line[0]))
    tck, _ = splprep([closed[:, 0], closed[:, 1]], k=5, s=len(closed) * (args.smooth_cm / 100.0) ** 2, per=1)
    u_fine = np.linspace(0.0, 1.0, 40001)
    fine = np.array(splev(u_fine, tck)).T
    s_fine = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(fine, axis=0).T))))
    total = float(s_fine[-1])
    count = int(round(total / args.step_out))
    s_points = np.arange(count) * total / count
    u = np.interp(s_points, s_fine, u_fine)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    raceline = np.column_stack((x, y))
    kappa = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
    psi = np.arctan2(dy, dx) - np.pi / 2.0          # same heading convention as the existing files
    psi = (psi + np.pi) % (2.0 * np.pi) - np.pi
    el_lengths = np.full(count, total / count)

    if np.abs(kappa).max() > kappa_bound * 1.25:
        print(f"WARNING: tightest radius {1.0 / np.abs(kappa).max():.2f} m is well below --min-radius "
              f"{args.min_radius:.2f} m; the corridor does not allow a wider arc there.")

    ggv = np.array([[0.0, args.ax_tyre, args.ay_max], [args.v_max + 10.0, args.ax_tyre, args.ay_max]])
    vx = tph.calc_vel_profile.calc_vel_profile(
        ggv=ggv, ax_max_machines=AX_MACHINE * np.array([1.0, args.accel_scale]), v_max=args.v_max,
        kappa=kappa, el_lengths=el_lengths,
        closed=True, drag_coeff=0.0, m_veh=1500.0, dyn_model_exp=1.0, filt_window=None,
    )
    ax = tph.calc_ax_profile.calc_ax_profile(
        vx_profile=np.append(vx, vx[0]), el_lengths=el_lengths, eq_length_output=False,
    )

    # Clearance check against the real cone positions (not the optimiser's model).
    left, right = get_track(args.track).build_cones()
    cones = []
    for l, r in zip(left, right):
        mid = np.array([0.5 * (l.x + r.x), 0.5 * (l.y + r.y)])
        unit = np.array([l.x, l.y]) - mid
        unit /= np.linalg.norm(unit)
        cones += [mid + unit * half, mid - unit * half]
    cone_gap = cKDTree(raceline).query(np.array(cones))[0] - args.car_width / 2.0 - args.cone_radius

    return {
        "s": s_points, "xy": raceline, "psi": psi, "kappa": kappa, "vx": vx, "ax": ax,
        "el_lengths": el_lengths, "centre": centre, "w_veh": w_veh,
        "offset": cKDTree(closed_centerline(args.track, 0.1)).query(raceline)[0],
        "cone_gap": cone_gap,
        "lap_time": float(np.sum(el_lengths / np.maximum(vx, 0.1))),
    }


def write_csv(result: dict, path: Path, header: str) -> None:
    s, xy, psi, kappa, vx, ax = (result[k] for k in ("s", "xy", "psi", "kappa", "vx", "ax"))
    total = float(s[-1] + result["el_lengths"][-1])
    rows = np.column_stack((s, xy, psi, kappa, vx, ax))
    rows = np.vstack((rows, np.concatenate(([total], rows[0, 1:]))))   # closing row = first point
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# {header}\n# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        for row in rows:
            f.write("; ".join(f"{value:.7f}" for value in row) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="hockenheim_fsg")
    p.add_argument("--name", default="hockenheim_fsg_audi_5m", help="Output file name (without .csv).")
    p.add_argument("--track-width", type=float, default=5.0, help="Cone-to-cone corridor width (m).")
    p.add_argument("--car-width", type=float, default=1.99, help="CARLA bounding box of the Audi TT (m).")
    p.add_argument("--cone-radius", type=float, default=0.20)
    p.add_argument("--margin", type=float, default=0.30, help="Extra clearance to each cone beyond its radius (m).")
    p.add_argument("--min-radius", type=float, default=5.0,
                   help="Tightest radius the line may ask for (m). The Audi measured 4.15 m at full lock.")
    p.add_argument("--ay-max", type=float, default=8.5,
                   help="Cornering limit (m/s^2). The Audi sustained about 9.5 in the benchmark laps.")
    p.add_argument("--ax-tyre", type=float, default=6.0,
                   help="Braking limit (m/s^2). The Audi reached about 7.6 at 0.64 brake.")
    p.add_argument("--v-max", type=float, default=22.0)
    p.add_argument("--accel-scale", type=float, default=1.0,
                   help="Scale the measured acceleration curve. Above 1 plans more speed than the capped "
                        "throttle delivers, so the car runs at full allowed throttle on the straights.")
    p.add_argument("--smooth-cm", type=float, default=3.0,
                   help="Smoothing of the optimised line (cm). Removes curvature ripple in tight corners.")
    p.add_argument("--step-opt", type=float, default=1.5, help="Spacing of optimisation points (m).")
    p.add_argument("--step-out", type=float, default=0.5, help="Spacing of the written raceline (m).")
    args = p.parse_args()

    result = optimise(args)
    out_path = PROJECT_ROOT / "inputs" / "racelines" / f"{args.name}.csv"
    header = (f"min-curvature, track {args.track}, corridor {args.track_width} m, effective car width "
              f"{result['w_veh']:.2f} m, R_min {args.min_radius} m, ay {args.ay_max}, ax_tyre {args.ax_tyre}, v_max {args.v_max}")
    write_csv(result, out_path, header)

    kappa, vx = result["kappa"], result["vx"]
    print(f"Wrote {out_path}")
    print(f"  length {result['s'][-1] + result['el_lengths'][-1]:.1f} m, {len(vx) + 1} rows")
    print(f"  tightest radius {1.0 / np.abs(kappa).max():.2f} m, widest offset from centre {result['offset'].max():.2f} m")
    print(f"  smallest gap between the car body and a cone when exactly on the line: {result['cone_gap'].min():.2f} m")
    print(f"  speed {vx.min():.1f} to {vx.max():.1f} m/s, planned lap {result['lap_time']:.1f} s")


if __name__ == "__main__":
    main()
