"""Report figures for the first lap on an unknown track (--controller explore).

    python3 scripts/make_exploration_figures.py                      # latest CARLA explore run, Hockenheim
    python3 scripts/make_exploration_figures.py --track brandshatch
    python3 scripts/make_exploration_figures.py --run outputs/carla/bench_explore_...   # a specific run
    python3 scripts/make_exploration_figures.py --dry                # allow kinematic (offline) runs

Writes to outputs/report_figures/:
  <prefix>_firstlap_overview   whole track: the cones, the driven path, the centreline the car did not
                               know (dashed), and for comparison the first lap that followed the known
                               centreline (if that run exists)
  <prefix>_firstlap_hairpin    three moments in the tightest corner: cones seen so far, the KinoRRT
                               tree, the chosen plan, the driven path and the detection range
  <prefix>_firstlap_planning   planning time per replan, and speed along the lap
  <prefix>_firstlap_results    table (.md/.csv): lap, cones, FS time, distance from the centreline,
                               closest cone, planning time, replans
prefix = carla for Hockenheim, carla_<track> otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import summarize_benchmark as sb  # noqa: E402
from fs_controller.exploration import KinoRRT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "report_figures"
RUNS = PROJECT_ROOT / "outputs" / "carla"
BLUE, YELLOW = "#2a78d6", "#eda100"
DRIVEN, PLAN, TREE, CENTRE, COMPARE, HIT = "#2a78d6", "#1baf7a", "#b9b8b3", "#52514e", "#eb6834", "#e24948"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONE_BLUE, CONE_YELLOW = "#1f4e9e", "#e3a600"


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf/.png")


# ----------------------------------------------------------------------------- runs
def noisy(m: dict) -> bool:
    per = (m.get("controller_params") or {}).get("perception", {})
    return bool(per.get("noise_m") or per.get("miss_prob"))


def find_run(track: str, dry: bool, controller: str = "explore", raceline_has: str | None = None,
             with_noise: bool = False):
    """Latest completed run at 0.15 s; for explore, with perfect (or, with_noise, imperfect) detection."""
    best = None
    for d in sorted(RUNS.glob("bench_*")):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        if m.get("controller") != controller or (m.get("track") or "hockenheim_fsg") != track:
            continue
        if m.get("status") != "completed" or (m.get("plant") == "kinematic" and not dry):
            continue
        if raceline_has and raceline_has not in str(m.get("raceline", "")):
            continue
        if controller == "pp" and not m.get("predict_delay"):
            continue
        if round(float(m.get("delay_s") or 0), 2) != 0.15:
            continue
        if controller == "explore" and noisy(m) != with_noise:
            continue
        best = d
    return best


def load(run: Path, cones: np.ndarray, centre_xy: np.ndarray) -> dict:
    m = json.loads((run / "meta.json").read_text())
    log = pd.read_csv(run / "run_log.csv")
    moving = log[log.t_s >= sb.IGNORE_BEFORE_S]
    ext = m.get("bbox_half_extent_m")
    off = m.get("bbox_center_offset_m")
    hl, hw, fwd = (ext[0], ext[1], off[0]) if isinstance(ext, list) and isinstance(off, list) else (2.09, 1.0, 0.0)
    clear = sb.cone_clearance(moving[["x_m", "y_m"]].values, moving.yaw_rapp_rad.values, cones, hl, hw, fwd)
    hits = np.where(clear.min(axis=0) <= 0.0)[0]
    dist = sb.cross_track(log[["x_m", "y_m"]].values, centre_xy)
    out = {"meta": m, "log": log, "hits": hits, "min_gap": float(clear.min()), "centre_dist": dist,
           "body": (hl, hw, fwd)}
    npz = run / "exploration.npz"
    if npz.exists():
        out["exp"] = dict(np.load(npz))
    return out


# ----------------------------------------------------------------------------- figures
def draw_cones(ax, cones, seen=None, size=9):
    col = np.arange(len(cones)) % 2
    for k, colour in ((0, CONE_BLUE), (1, CONE_YELLOW)):
        sel = col == k
        if seen is None:
            ax.scatter(cones[sel, 0], cones[sel, 1], s=size, color=colour, zorder=3, linewidths=0)
        else:
            a, b = sel & seen, sel & ~seen
            ax.scatter(cones[a, 0], cones[a, 1], s=size * 2.2, color=colour, zorder=3, linewidths=0)
            ax.scatter(cones[b, 0], cones[b, 1], s=size * 2.2, facecolors="none", edgecolors=colour,
                       linewidths=0.8, alpha=0.45, zorder=3)


def plot_overview(run: dict, compare: dict | None, cones, centre_xy, name: str, track_label: str) -> None:
    log = run["log"]
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    draw_cones(ax, cones, size=5)
    ring = np.vstack((centre_xy, centre_xy[:1]))
    ax.plot(ring[:, 0], ring[:, 1], color=CENTRE, lw=0.8, ls=(0, (4, 3)), label="Track centreline (not known to the car)")
    if compare is not None:
        cl = compare["log"]
        ax.plot(cl.x_m, cl.y_m, color=COMPARE, lw=1.0, ls=":", label=f"First lap following the known centreline "
                f"({compare['meta']['lap_time_s']:.1f} s, {len(compare['hits'])} cones)")
    ax.plot(log.x_m, log.y_m, color=DRIVEN, lw=1.4, label=f"KinoRRT + PFPP on the unknown track "
            f"({run['meta']['lap_time_s']:.1f} s, {len(run['hits'])} cones)")
    for h in run["hits"]:
        ax.plot(*cones[h], marker="x", color=HIT, ms=7, mew=2, zorder=5)
    ax.plot(log.x_m.iloc[0], log.y_m.iloc[0], marker=">", color=TEXT, ms=8, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"First lap on an unknown track: {track_label}, 0.15 s delay, cones seen up to "
                 f"{run['exp']['perception'][0]:g} m ahead" if "exp" in run else "First lap", fontsize=10, color=TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=8.5, ncol=1)
    save(fig, name)


def tree_segments(rows: np.ndarray, step_m: float) -> list[np.ndarray]:
    out = []
    pts = np.linspace(0.0, step_m, 5)
    for i in range(1, len(rows)):
        par = int(rows[i, 4])
        ax_, ay, _ = KinoRRT.arc(float(rows[par, 0]), float(rows[par, 1]), float(rows[par, 2]), float(rows[i, 3]), pts)
        out.append(np.column_stack((ax_, ay)))
    return out


def seen_until(log: pd.DataFrame, t: float, cones: np.ndarray, rng_m: float, fov_deg: float, fwd: float) -> np.ndarray:
    seen = np.zeros(len(cones), bool)
    part = log[log.t_s <= t].iloc[::4]
    for x, y, yaw in part[["x_m", "y_m", "yaw_rapp_rad"]].values:
        sx, sy = x + fwd * math.cos(yaw), y + fwd * math.sin(yaw)
        d = cones - (sx, sy)
        bearing = (np.arctan2(d[:, 1], d[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        seen |= (np.hypot(d[:, 0], d[:, 1]) <= rng_m) & (np.abs(bearing) <= math.radians(fov_deg) / 2)
    return seen


def car_outline(x, y, yaw, hl, hw, fwd):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
    return np.array([(x + (px + fwd) * c - py * s, y + (px + fwd) * s + py * c) for px, py in pts])


def plot_hairpin(run: dict, cones, centre_xy, centre_k, name: str) -> None:
    exp, log = run["exp"], run["log"]
    rng_m, fov, fwd = (float(v) for v in exp["perception"])
    apex = int(np.argmax(np.abs(centre_k)))
    centre = centre_xy[apex]
    # replans while the car is approaching and inside the tightest corner
    replan_t, plan_t = exp["replan_t"], exp["plan_t"]
    near = []
    for k, t in enumerate(replan_t):
        r = log.iloc[int(np.argmin(np.abs(log.t_s.values - t)))]
        dist = math.hypot(r.x_m - centre[0], r.y_m - centre[1])
        if dist < 14.0:
            near.append((k, t, dist))
    if not near:
        print("  (no replan near the tightest corner; hairpin figure skipped)")
        return
    approach = [n for n in near if n[1] <= near[int(np.argmin([q[2] for q in near]))][1]]
    picks = [approach[0], approach[len(approach) // 2], near[int(np.argmin([q[2] for q in near]))]]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    ts, tree_start = exp["tree_start"], list(exp["tree_start"][1:]) + [len(exp["trees"])]
    ps, path_end = exp["path_start"], list(exp["path_start"][1:]) + [len(exp["paths"])]
    hl, hw, bfwd = run["body"]
    for ax, (k, t, _) in zip(axes, picks):
        r = log.iloc[int(np.argmin(np.abs(log.t_s.values - t)))]
        seen = seen_until(log, t, cones, rng_m, fov, fwd)
        draw_cones(ax, cones, seen)
        rows = exp["trees"][ts[k]:tree_start[k]]
        for seg in tree_segments(rows, float(exp["step_m"])):
            ax.plot(seg[:, 0], seg[:, 1], color=TREE, lw=0.7, zorder=1)
        # the chosen plan of this replan (history only holds successful replans: take the latest at or before t)
        j = int(np.searchsorted(plan_t, t + 1e-6, side="right") - 1)
        if 0 <= j < len(ps):
            q = exp["paths"][ps[j]:path_end[j]]
            ax.plot(q[:, 0], q[:, 1], color=PLAN, lw=2.0, zorder=4)
        past = log[log.t_s <= t]
        ax.plot(past.x_m, past.y_m, color=DRIVEN, lw=1.6, zorder=4)
        rest = log[log.t_s > t]
        ax.plot(rest.x_m, rest.y_m, color=DRIVEN, lw=0.8, ls=":", alpha=0.6, zorder=2)
        body = car_outline(r.x_m, r.y_m, r.yaw_rapp_rad, hl, hw, bfwd)
        ax.fill(body[:, 0], body[:, 1], color=TEXT, alpha=0.85, zorder=5)
        sx, sy = r.x_m + fwd * math.cos(r.yaw_rapp_rad), r.y_m + fwd * math.sin(r.yaw_rapp_rad)
        a = np.linspace(-math.radians(fov) / 2, math.radians(fov) / 2, 40) + r.yaw_rapp_rad
        wedge = np.vstack(([sx, sy], np.column_stack((sx + rng_m * np.cos(a), sy + rng_m * np.sin(a))), [sx, sy]))
        ax.plot(wedge[:, 0], wedge[:, 1], color=MUTED, lw=0.6, ls="--", zorder=1)
        ax.set_xlim(centre[0] - 13, centre[0] + 13)
        ax.set_ylim(centre[1] - 13, centre[1] + 13)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.set_title(f"t = {t:.1f} s, v = {r.v_mps:.1f} m/s, tree {len(rows)} nodes", fontsize=9.5, color=TEXT)
    handles = [
        plt.Line2D([], [], color=TREE, lw=1, label="KinoRRT tree (this replan)"),
        plt.Line2D([], [], color=PLAN, lw=2, label="Chosen plan"),
        plt.Line2D([], [], color=DRIVEN, lw=1.6, label="Driven so far"),
        plt.Line2D([], [], color=MUTED, lw=0.6, ls="--", label=f"Detection range ({rng_m:g} m, {fov:g} deg)"),
        plt.Line2D([], [], marker="o", color=CONE_BLUE, ls="", label="Cone seen"),
        plt.Line2D([], [], marker="o", markerfacecolor="none", color=CONE_BLUE, alpha=0.5, ls="", label="Not seen yet"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Planning the tightest corner from the cones in view", fontsize=11, color=TEXT)
    save(fig, name)


def plot_planning(run: dict, compare: dict | None, name: str) -> None:
    exp, log = run["exp"], run["log"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ms = exp["plan_ms"]
    a1.hist(ms, bins=40, color=PLAN, alpha=0.85)
    for q, lab in ((np.mean(ms), "mean"), (np.percentile(ms, 95), "95%")):
        a1.axvline(q, color=TEXT, lw=0.9, ls="--")
        a1.text(q, a1.get_ylim()[1] * 0.92, f" {lab} {q:.0f} ms", fontsize=8, color=TEXT)
    a1.set_xlabel("Planning time per replan (ms, Python, one CPU core)", fontsize=9, color=TEXT)
    a1.set_ylabel("Replans", fontsize=9, color=TEXT)
    style_axes(a1)
    a2.plot(log.s_m, log.v_mps, color=DRIVEN, lw=1.0, label="KinoRRT + PFPP (track unknown)")
    if compare is not None:
        cl = compare["log"]
        a2.plot(cl.s_m, cl.v_mps, color=COMPARE, lw=1.0, ls=":", label="Following the known centreline")
    a2.set_xlabel("Distance along the lap (m)", fontsize=9, color=TEXT)
    a2.set_ylabel("Speed (m/s)", fontsize=9, color=TEXT)
    a2.legend(frameon=False, fontsize=8.5, loc="lower right")
    style_axes(a2)
    save(fig, name)


def write_table(rows: list[tuple[str, dict]], name: str) -> None:
    header = ["Run", "Lap (s)", "Cones", "FS time (s)", "Mean dist. from centreline (m)",
              "Max dist. (m)", "Closest cone (m)", "Plan mean (ms)", "Plan 95% (ms)", "Plan max (ms)",
              "Replans", "Kept previous plan", "Tight-margin plans"]
    table = []
    for label, r in rows:
        m, st = r["meta"], r["meta"].get("exploration", {})
        lap = m["lap_time_s"]
        cones = len(r["hits"])
        table.append([label, f"{lap:.2f}", cones, f"{lap + sb.CONE_PENALTY_S * cones:.2f}",
                      f"{r['centre_dist'].mean():.2f}", f"{r['centre_dist'].max():.2f}", f"{r['min_gap']:.2f}",
                      st.get("plan_ms_mean", "-"), st.get("plan_ms_p95", "-"), st.get("plan_ms_max", "-"),
                      st.get("replans", "-"), st.get("failed_replans", "-"), st.get("tight_margin_plans", "-")])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{name}.csv").open("w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    with (OUT_DIR / f"{name}.md").open("w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in table:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
    print(f"  {name}.csv/.md")
    for row in table:
        print("   ", " | ".join(str(c) for c in row))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="hockenheim_fsg")
    p.add_argument("--run", type=Path, default=None, help="Explore run folder (default: the latest completed one).")
    p.add_argument("--compare", type=Path, default=None,
                   help="Known-centreline first-lap run to compare with (default: latest, if any).")
    p.add_argument("--dry", action="store_true", help="Also accept offline (kinematic) runs.")
    args = p.parse_args()

    prefix = "carla" if args.track == "hockenheim_fsg" else f"carla_{args.track}"
    run_dir = args.run or find_run(args.track, args.dry)
    if run_dir is None:
        raise SystemExit("No completed explore run at 0.15 s for this track (use --run, or --dry for offline runs).")
    cones = sb.load_cones(sb.TRACK_WIDTH_M, args.track)
    centre = np.loadtxt(PROJECT_ROOT / "inputs" / "racelines" / f"{args.track}_first_lap.csv", delimiter=";", comments="#")
    centre_xy, centre_k = centre[:-1, 1:3], centre[:-1, 4]
    run = load(Path(run_dir), cones, centre_xy)
    cmp_dir = args.compare or find_run(args.track, args.dry, controller="pp", raceline_has="first_lap")
    compare = load(Path(cmp_dir), cones, centre_xy) if cmp_dir else None
    print(f"Explore run: {Path(run_dir).name}" + (f" | compared with {Path(cmp_dir).name}" if cmp_dir else ""))
    label = {"hockenheim_fsg": "FSG Hockenheim"}.get(args.track, args.track.replace("_", " ").title())
    plot_overview(run, compare, cones, centre_xy, f"{prefix}_firstlap_overview", label)
    if "exp" in run:
        plot_hairpin(run, cones, centre_xy, centre_k, f"{prefix}_firstlap_hairpin")
        plot_planning(run, compare, f"{prefix}_firstlap_planning")
    rows = [("KinoRRT + PFPP, track unknown", run)]
    noisy_dir = find_run(args.track, args.dry, with_noise=True)
    if noisy_dir is not None and args.run is None:
        nm = load(Path(noisy_dir), cones, centre_xy)
        per = nm["meta"]["controller_params"]["perception"]
        rows.append((f"... with noisy detection ({100 * per['noise_m']:.0f} cm, {100 * per['miss_prob']:.0f}% missed)", nm))
    if compare is not None:
        rows.append(("PP 4 m with prediction + steering map, known centreline", compare))
    write_table(rows, f"{prefix}_firstlap_results")


if __name__ == "__main__":
    main()
