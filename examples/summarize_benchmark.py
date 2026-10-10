"""Summarise benchmark runs written by run_carla_benchmark.py.

    python3 examples/summarize_benchmark.py                   # CARLA bench_* runs
    python3 examples/summarize_benchmark.py --include-legacy    # also the September rapp_lap_* logs
    python3 examples/summarize_benchmark.py --include-dry-runs  # also --plant kinematic runs

Runs that were interrupted (Ctrl+C) are listed but left out of the averages.

Every metric is recomputed from the logged positions with one method, so runs
from different scripts are directly comparable. Cross-track error is the
distance to the raceline polyline (segments, not nearest vertex).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.hockenheim_fsg_track import build_fsg_hockenheim_cones  # noqa: E402

RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "carla"

TRACK_WIDTH_M = 5.0      # same as spawn_hockenheim_cones_wide.py --width 5.0
CAR_LENGTH_M = 4.18      # measured from the CARLA bounding box
CAR_WIDTH_M = 1.99
CONE_RADIUS_M = 0.20
IGNORE_BEFORE_S = 2.0


def load_cones(width_m: float) -> np.ndarray:
    left, right = build_fsg_hockenheim_cones()
    pts = []
    for l, r in zip(left, right):
        mid = np.array([0.5 * (l.x + r.x), 0.5 * (l.y + r.y)])
        half = np.array([l.x, l.y]) - mid
        norm = np.linalg.norm(half)
        if norm < 1e-9:
            continue
        unit = half / norm
        pts.append(mid + unit * width_m / 2.0)
        pts.append(mid - unit * width_m / 2.0)
    return np.array(pts)


def cross_track(xy: np.ndarray, raceline_xy: np.ndarray) -> np.ndarray:
    a = raceline_xy[:-1]
    seg = np.roll(a, -1, axis=0) - a
    seg_sq = np.maximum((seg**2).sum(axis=1), 1e-12)
    out = np.empty(len(xy))
    for i, p in enumerate(xy):
        d = p - a
        u = np.clip((d * seg).sum(axis=1) / seg_sq, 0.0, 1.0)
        out[i] = np.sqrt((((a + u[:, None] * seg) - p) ** 2).sum(axis=1).min())
    return out


def cone_clearance(
    xy: np.ndarray,
    yaw: np.ndarray,
    cones: np.ndarray,
    half_length: float,
    half_width: float,
    center_forward: float,
) -> np.ndarray:
    """Clearance (m) from the car body rectangle to each cone surface: [samples, cones].

    center_forward is how far the body centre sits ahead of the logged position.
    """
    dx = cones[None, :, 0] - xy[:, None, 0]
    dy = cones[None, :, 1] - xy[:, None, 1]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    local_x = c * dx + s * dy - center_forward
    local_y = -s * dx + c * dy
    ex = np.maximum(np.abs(local_x) - half_length, 0.0)
    ey = np.maximum(np.abs(local_y) - half_width, 0.0)
    return np.hypot(ex, ey) - CONE_RADIUS_M


def controller_label(meta: dict) -> str:
    name = meta.get("controller", "?")
    params = meta.get("controller_params") or {}
    if name == "pp":
        gain = params.get("lookahead_gain_s", 0.0)
        return f"pp L={params.get('lookahead_m', '?')}" + (f"+{gain}v" if gain else "")
    if name == "rapp":
        return (f"rapp g={params.get('speed_gain', '?')} "
                f"[{params.get('lookahead_min', '?')},{params.get('lookahead_max', '?')}]")
    if name == "pid":
        return f"pid {params.get('kp', '?')}/{params.get('ki', '?')}/{params.get('kd', '?')}"
    return name


def analyse(run_dir: Path, raceline_xy: np.ndarray, cones: np.ndarray, include_dry_runs: bool) -> dict | None:
    log_path = run_dir / "run_log.csv"
    if not log_path.exists():
        return None
    with log_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 10:
        return None
    col = lambda name: np.array([float(r[name]) for r in rows])  # noqa: E731

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        label = controller_label(meta)
    elif run_dir.name.startswith("bench_"):  # stopped before any metadata was written
        meta = {"controller": run_dir.name.split("_")[1], "status": "interrupted", "lap_time_s": None}
        label = meta["controller"] + " (no meta)"
    else:  # September logs: no metadata, RAPP, wall-clock time, lap flag in the log
        completed = int(float(rows[-1].get("lap_complete", 0) or 0)) == 1
        meta = {
            "status": "completed" if completed else "timeout",
            "lap_time_s": float(rows[-1]["t_s"]) if completed else None,
            "speed_scale": 1.0,
        }
        label = "rapp (Sept, legacy)"
    if meta.get("plant") == "kinematic":
        if not include_dry_runs:
            return None
        label += " DRY"

    # Car footprint: the simulator's own bounding box when the run recorded it,
    # otherwise the assumed Audi dimensions.
    extent = meta.get("bbox_half_extent_m")
    offset = meta.get("bbox_center_offset_m")
    if isinstance(extent, list) and isinstance(offset, list):
        half_length, half_width, center_forward, footprint = extent[0], extent[1], offset[0], "sim"
    else:
        half_length, half_width, center_forward, footprint = CAR_LENGTH_M / 2.0, CAR_WIDTH_M / 2.0, 0.0, "assumed"

    t = col("t_s")
    xy = np.column_stack((col("x_m"), col("y_m")))
    yaw = col("yaw_rapp_rad")
    v = col("v_mps")
    xte = cross_track(xy, raceline_xy)
    steer = col("steering_rad")
    dt = np.diff(t)
    steer_rate = np.diff(steer) / np.maximum(dt, 1e-6)

    moving = t >= IGNORE_BEFORE_S
    clearance = cone_clearance(xy[moving], yaw[moving], cones, half_length, half_width, center_forward)
    cones_hit = int((clearance.min(axis=0) <= 0.0).sum())

    return {
        "run": run_dir.name,
        "controller": label,
        "speed_scale": float(meta.get("speed_scale", 1.0)),
        "status": meta["status"],
        "lap_time_s": meta.get("lap_time_s"),
        "xte_mean_m": float(xte.mean()),
        "xte_median_m": float(np.median(xte)),
        "xte_p95_m": float(np.percentile(xte, 95)),
        "xte_max_m": float(xte.max()),
        "steer_rate_rms_radps": float(np.sqrt(np.mean(steer_rate**2))),
        "steer_saturated_pct": float(100.0 * np.mean(np.abs(col("steer_norm")) > 0.99)),
        "v_max_mps": float(v.max()),
        "v_mean_mps": float(v.mean()),
        "min_cone_clearance_m": float(clearance.min()),
        "cones_hit_geometric": cones_hit,
        "cones_hit_sim": meta.get("sim_cones_hit"),
        "footprint": footprint,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-legacy", action="store_true", help="Also include the September rapp_lap_* logs.")
    p.add_argument("--include-dry-runs", action="store_true", help="Also include --plant kinematic runs.")
    p.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args()

    raceline_xy = np.loadtxt(RACELINE_PATH, delimiter=";", comments="#")[:, 1:3]
    cones = load_cones(TRACK_WIDTH_M)

    run_dirs = sorted(args.root.glob("bench_*"))
    if args.include_legacy:
        run_dirs += sorted(args.root.glob("rapp_lap_*"))
    results = [r for r in (analyse(d, raceline_xy, cones, args.include_dry_runs) for d in run_dirs) if r is not None]
    if not results:
        raise SystemExit(f"No runs found under {args.root}")

    print(f"{'run':<32}{'controller':<24}{'scale':>6}{'status':>12}{'lap s':>8}"
          f"{'xte avg':>8}{'p95':>6}{'max':>6}{'steer rate':>11}{'sat %':>6}{'v max':>6}{'clear':>7}{'geo':>4}{'sim':>4}")
    for r in results:
        lap = f"{r['lap_time_s']:.2f}" if r["lap_time_s"] is not None else "-"
        print(f"{r['run']:<32}{r['controller']:<24}{r['speed_scale']:>6.2f}{r['status']:>12}{lap:>8}"
              f"{r['xte_mean_m']:>8.2f}{r['xte_p95_m']:>6.2f}{r['xte_max_m']:>6.2f}"
              f"{r['steer_rate_rms_radps']:>11.3f}{r['steer_saturated_pct']:>6.1f}{r['v_max_mps']:>6.1f}"
              f"{r['min_cone_clearance_m']:>7.2f}{r['cones_hit_geometric']:>4d}"
              f"{'-' if r['cones_hit_sim'] is None else r['cones_hit_sim']:>4}")

    print("\ngeo = cones inside the car footprint, computed from the logged path; "
          "sim = cones the simulator reported touched.")
    footprints = sorted({r["footprint"] for r in results})
    print(f"Footprint used for geo: {', '.join(footprints)} "
          "(sim = the simulator's bounding box, assumed = 4.20 x 1.85 m).")
    print("\nPer controller and speed scale")
    print(f"{'controller':<24}{'scale':>6}{'runs':>5}{'done':>5}{'lap mean':>9}{'lap std':>8}"
          f"{'xte avg':>8}{'xte max':>8}{'steer rate':>11}{'geo hits':>9}{'sim hits':>9}")
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in results:
        if r["status"] != "interrupted":
            groups.setdefault((r["controller"], r["speed_scale"]), []).append(r)
    for (controller, scale), runs in sorted(groups.items()):
        laps = [r["lap_time_s"] for r in runs if r["status"] == "completed"]
        lap_mean = f"{np.mean(laps):.2f}" if laps else "-"
        lap_std = f"{np.std(laps):.2f}" if len(laps) > 1 else "-"
        sim = [r["cones_hit_sim"] for r in runs if r["cones_hit_sim"] is not None]
        sim_text = f"{np.mean(sim):.1f}" if sim else "-"
        print(f"{controller:<24}{scale:>6.2f}{len(runs):>5}{len(laps):>5}{lap_mean:>9}{lap_std:>8}"
              f"{np.mean([r['xte_mean_m'] for r in runs]):>8.2f}{max(r['xte_max_m'] for r in runs):>8.2f}"
              f"{np.mean([r['steer_rate_rms_radps'] for r in runs]):>11.3f}"
              f"{np.mean([r['cones_hit_geometric'] for r in runs]):>9.1f}{sim_text:>9}")

    out_path = args.root / "benchmark_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-run table written to {out_path}")


if __name__ == "__main__":
    main()
