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
CAR_LENGTH_M = 4.18      # measured from the CARLA bounding box of vehicle.ue4.audi.tt
CAR_WIDTH_M = 1.99
CONE_RADIUS_M = 0.20
IGNORE_BEFORE_S = 2.0
CONE_PENALTY_S = 2.0     # FS Rules 2026 D 10.1.7: DOO in Autocross / Trackdrive


def load_cones(width_m: float, track: str = "hockenheim_fsg") -> np.ndarray:
    if track == "hockenheim_fsg":
        left, right = build_fsg_hockenheim_cones()
    else:
        from visualization.track_registry import get_track
        left, right = get_track(track).build_cones()
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


def is_pfpp(meta: dict) -> bool:
    """PFPP runs: --controller pfpp, or the same configuration run before it had a name."""
    if meta.get("controller") == "pfpp":
        return True
    p = meta.get("controller_params") or {}
    return (meta.get("controller") == "rapp" and p.get("lookahead_min") == 3.5 and p.get("lookahead_max") == 6.0
            and p.get("curv_gain", 1.0) == 1.0 and meta.get("reference") == "rear"
            and meta.get("steer_map") == "calibrated" and bool(meta.get("feedforward"))
            and bool(meta.get("predict_delay")))


def controller_label(meta: dict) -> str:
    if is_pfpp(meta):
        gain = (meta.get("controller_params") or {}).get("speed_gain", 0.45)
        return "pfpp" + (f" g={gain}" if gain != 0.45 else "")
    name = meta.get("controller", "?")
    params = meta.get("controller_params") or {}
    if name == "pp":
        gain = params.get("lookahead_gain_s", 0.0)
        return f"pp L={params.get('lookahead_m', '?')}" + (f"+{gain}v" if gain else "")
    if name == "rapp":
        curv = params.get("curv_gain", 1.0)
        return (f"rapp g={params.get('speed_gain', '?')} "
                f"[{params.get('lookahead_min', '?')},{params.get('lookahead_max', '?')}]"
                + ((" no-curv" if curv >= 50 else f" c={curv}") if curv != 1.0 else "")
                + (f" tau={params['delay_comp_s']}" if params.get("delay_comp_s") else ""))
    if name == "explore":
        e = (meta.get("controller_params") or {}).get("perception", {})
        tag = f" noise={e['noise_m']:g}" if e.get("noise_m") else ""
        tag += f" miss={e['miss_prob']:g}" if e.get("miss_prob") else ""
        seed = (meta.get("controller_params") or {}).get("kinorrt", {}).get("seed", 1)
        return "explore kinorrt" + (f" seed={seed}" if seed != 1 else "") + tag
    if name == "pid":
        return f"pid {params.get('kp', '?')}/{params.get('ki', '?')}/{params.get('kd', '?')}"
    return name


_RACELINE_CACHE: dict[str, np.ndarray] = {}
_CONES_CACHE: dict[str, np.ndarray] = {}


def raceline_for(meta: dict, default_xy: np.ndarray) -> tuple[np.ndarray, str]:
    """Each run is scored against the raceline it actually followed."""
    name = meta.get("raceline")
    if not name or name == RACELINE_PATH.name:
        return default_xy, ""
    path = RACELINE_PATH.parent / name
    if not path.exists():
        return default_xy, " line=?"
    if name not in _RACELINE_CACHE:
        _RACELINE_CACHE[name] = np.loadtxt(path, delimiter=";", comments="#")[:, 1:3]
    return _RACELINE_CACHE[name], f" line={Path(name).stem[-12:]}"


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
    raceline_xy, line_label = raceline_for(meta, raceline_xy)
    track = meta.get("track", "hockenheim_fsg")
    if track != "hockenheim_fsg":                  # another layout: its own cones, and say so
        if track not in _CONES_CACHE:
            _CONES_CACHE[track] = load_cones(TRACK_WIDTH_M, track)
        cones = _CONES_CACHE[track]
        label += f" track={track}"
        line_label = ""
    label += line_label
    speed_ref = meta.get("speed_ref", "lookahead")
    if speed_ref != "lookahead":
        label += f" v={speed_ref}"
    car = (meta.get("vehicle_params") or {}).get("name", "audi_tt_carla")
    if car != "audi_tt_carla":
        label += " car=" + car.replace("audi_tt_carla_", "")
    if meta.get("grip_cap_mps2"):
        label += f" +grip{meta['grip_cap_mps2']:g}"
    pfpp = meta_path.exists() and (is_pfpp(meta) or meta.get("controller") == "explore")  # additions are in the name
    if meta.get("reference") == "rear" and not pfpp:
        label += " +rear"
    if meta.get("steer_map") == "calibrated" and not pfpp:
        label += " +cal"
    if meta.get("feedforward") and not pfpp:
        label += " +ff"
    if meta.get("predict_delay") and not pfpp:
        label += " +predict" + {"simple": "(simple)", "fitted": "(v14)", "v14": "(v14)"}.get(
            meta.get("predict_model", "simple"), "")
    setting = label
    if meta.get("delay_s"):
        label += f" d={meta['delay_s']:.2f}s"
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
    if not moving.any():                           # stopped within the first seconds (e.g. a crashed test)
        return None
    clearance = cone_clearance(xy[moving], yaw[moving], cones, half_length, half_width, center_forward)
    cones_hit = int((clearance.min(axis=0) <= 0.0).sum())

    return {
        "run": run_dir.name,
        "controller": label,
        "setting": setting,
        "delay_s": float(meta.get("delay_s") or 0.0),
        "speed_scale": float(meta.get("speed_scale", 1.0)),
        "status": meta["status"],
        "lap_time_s": meta.get("lap_time_s"),
        "fs_time_s": (meta["lap_time_s"] + CONE_PENALTY_S * cones_hit) if meta.get("lap_time_s") else None,
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

    print(f"{'run':<32}{'controller':<40}{'scale':>6}{'status':>12}{'lap s':>8}"
          f"{'cones':>6}{'FS time':>8}{'xte avg':>8}{'p95':>6}{'max':>6}{'steer rate':>11}{'sat %':>6}{'v max':>6}{'clear':>7}")
    for r in results:
        lap = f"{r['lap_time_s']:.2f}" if r["lap_time_s"] is not None else "-"
        fs = f"{r['fs_time_s']:.2f}" if r["fs_time_s"] is not None else "-"
        print(f"{r['run']:<32}{r['controller']:<40}{r['speed_scale']:>6.2f}{r['status']:>12}{lap:>8}"
              f"{r['cones_hit_geometric']:>6d}{fs:>8}"
              f"{r['xte_mean_m']:>8.2f}{r['xte_p95_m']:>6.2f}{r['xte_max_m']:>6.2f}"
              f"{r['steer_rate_rms_radps']:>11.3f}{r['steer_saturated_pct']:>6.1f}{r['v_max_mps']:>6.1f}"
              f"{r['min_cone_clearance_m']:>7.2f}")

    print(f"\ncones = cones inside the car's outline ({CAR_LENGTH_M:.2f} x {CAR_WIDTH_M:.2f} m, CARLA's bounding box), "
          "from the logged path.")
    print(f"FS time = lap time + {CONE_PENALTY_S:.0f} s per cone (FS Rules 2026, D 10.1.7).")

    full_speed = [r for r in results if r["speed_scale"] == 1.0 and r["status"] != "interrupted"]
    delays = sorted({r["delay_s"] for r in full_speed})
    if full_speed:
        print("\nFS time in seconds (cones hit), by actuation delay, at full speed")
        print(f"{'setting':<36}" + "".join(f"{f'delay {d:.2f} s':>18}" for d in delays) + f"{'worst':>10}")
        table: dict[str, dict[float, dict]] = {}
        for r in full_speed:
            table.setdefault(r["setting"], {})[r["delay_s"]] = r   # latest run wins
        def worst(cells: dict[float, dict]) -> float:
            times = [c["fs_time_s"] for c in cells.values()]
            return float("inf") if any(t is None for t in times) or len(cells) < len(delays) else max(times)
        for setting, cells in sorted(table.items(), key=lambda kv: (worst(kv[1]), kv[0])):
            row = f"{setting:<36}"
            for d in delays:
                c = cells.get(d)
                if c is None:
                    row += f"{'-':>18}"
                elif c["fs_time_s"] is None:
                    row += f"{c['status']:>18}"
                else:
                    row += f"{c['fs_time_s']:.1f} ({c['cones_hit_geometric']})".rjust(18)
            w = worst(cells)
            row += f"{'-' if w == float('inf') else format(w, '.1f'):>10}"
            print(row)
        print("worst = the slowest FS time across all delays; '-' where a delay was not run or the lap failed.")

    out_path = args.root / "benchmark_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-run table written to {out_path}")


if __name__ == "__main__":
    main()
