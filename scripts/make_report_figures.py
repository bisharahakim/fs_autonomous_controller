"""Report figures and the results table from the CARLA benchmark logs.

    python3 scripts/make_report_figures.py            # writes outputs/report_figures/carla_*.{pdf,png} + results tables

Runs are picked by their metadata (meta.json), not by folder name, so the figures
always show exactly the configurations defined in SERIES below. When the same
configuration was run more than once at the same delay, the latest run is used.

Figures
  carla_fs_time_vs_delay     FS time (lap + 2 s per cone) against actuation delay
  carla_error_vs_delay       95th-percentile tracking error against actuation delay
  carla_robustness           FS time at 0.15 s under wrong lookahead / wrong steering model
  carla_error_along_track    |tracking error| along the lap at 0.15 s, ours vs classic PP
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import summarize_benchmark as sb  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "report_figures"
DESIGN_DELAY_S = 0.15

# Categorical colours in fixed order (validated palette, light mode). Each series also
# has its own marker and line style, so identity never depends on colour alone.
SERIES = {
    "ours": dict(label="Ours (PP + feed-forward + delay prediction, speed-scaled L)",
                 short="Ours", color="#2a78d6", marker="o", ls="-"),
    "pp_pkg": dict(label="Pure pursuit 4 m + same additions",
                   short="PP 4 m + additions", color="#eb6834", marker="s", ls="--"),
    "pp_classic": dict(label="Classic pure pursuit 4 m",
                       short="Classic PP 4 m", color="#1baf7a", marker="^", ls="-."),
    "rpp_classic": dict(label="RAPP as published (adaptive lookahead, no additions)",
                        short="RAPP (no additions)", color="#eda100", marker="D", ls=":"),
}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


# ----------------------------------------------------------------------------- selection
def has_package(m: dict) -> bool:
    return (m.get("reference") == "rear" and m.get("steer_map") == "calibrated"
            and bool(m.get("feedforward")) and bool(m.get("predict_delay")))


def no_package(m: dict) -> bool:
    return (m.get("reference") in (None, "centre") and m.get("steer_map") in (None, "legacy")
            and not m.get("feedforward") and not m.get("predict_delay"))


def car(m: dict) -> str:
    return (m.get("vehicle_params") or {}).get("name", "audi_tt_carla")


def rapp_params(m: dict) -> tuple:
    p = m.get("controller_params") or {}
    return (p.get("speed_gain"), p.get("lookahead_min"), p.get("lookahead_max"), p.get("curv_gain", 1.0))


def classify(m: dict) -> tuple[str, str] | None:
    """(series, condition) for a run, or None when it is not part of the report set.

    condition is 'nominal' or one of the robustness variations.
    """
    ctrl = m.get("controller")
    params = m.get("controller_params") or {}
    vehicle = car(m)
    if ctrl == "rapp":
        gain, lmin, lmax, curv = rapp_params(m)
        if has_package(m) and lmin == 3.5 and lmax == 6.0 and curv == 1.0:
            if vehicle != "audi_tt_carla":
                return "ours", "steer " + vehicle.split("steer_")[-1].replace("minus", "-").replace("plus", "+") + "%"
            return "ours", {0.45: "nominal", 0.30: "lookahead short", 0.60: "lookahead long"}.get(gain, f"g={gain}")
        if no_package(m) and (gain, lmin, lmax, curv) == (0.45, 2.0, 6.0, 1.0) and vehicle == "audi_tt_carla":
            return "rpp_classic", "nominal"
        return None
    if ctrl == "pp" and not params.get("lookahead_gain_s"):
        length = params.get("lookahead_m")
        cond = {4.0: "nominal", 2.0: "lookahead short", 6.0: "lookahead long"}.get(length)
        if cond is None:
            return None
        if has_package(m):
            if vehicle != "audi_tt_carla":
                if length != 4.0:
                    return None
                return "pp_pkg", "steer " + vehicle.split("steer_")[-1].replace("minus", "-").replace("plus", "+") + "%"
            return "pp_pkg", cond
        if no_package(m) and vehicle == "audi_tt_carla":
            return "pp_classic", cond
    return None


def collect(root: Path, require_grip: bool) -> dict:
    raceline_xy = np.loadtxt(sb.RACELINE_PATH, delimiter=";", comments="#")[:, 1:3]
    cones = sb.load_cones(sb.TRACK_WIDTH_M)
    table: dict[tuple[str, str, float], dict] = {}
    for run_dir in sorted(root.glob("bench_*")):          # sorted by time: later runs overwrite
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text())
        if m.get("plant") == "kinematic" or m.get("status") == "interrupted":
            continue
        if float(m.get("speed_scale", 1.0)) != 1.0 or m.get("speed_ref") != "envelope":
            continue
        if Path(m.get("raceline", sb.RACELINE_PATH.name)).name != sb.RACELINE_PATH.name:
            continue
        if require_grip and not m.get("grip_cap_mps2"):
            continue
        key = classify(m)
        if key is None:
            continue
        r = sb.analyse(run_dir, raceline_xy, cones, include_dry_runs=False)
        if r is None:
            continue
        r["series"], r["condition"] = key
        r["dir"] = run_dir
        table[(key[0], key[1], round(float(m.get("delay_s") or 0.0), 2))] = r
    return table


# ----------------------------------------------------------------------------- plotting
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


def failed_value(r: dict) -> bool:
    return r["fs_time_s"] is None


def plot_vs_delay(table: dict, metric: str, ylabel: str, name: str, title: str, cap: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    style_axes(ax)
    for sid, s in SERIES.items():
        pts = sorted((d, r) for (ser, cond, d), r in table.items() if ser == sid and cond == "nominal")
        if not pts:
            continue
        xs, ys, notes = [], [], []
        for d, r in pts:
            value = r[metric]
            if value is None:                      # lap not completed
                notes.append((d, r["status"]))
                continue
            xs.append(d * 1000)
            ys.append(min(value, cap) if cap else value)
        ax.plot(xs, ys, color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=7,
                markeredgecolor="white", markeredgewidth=1.0, label=s["label"], clip_on=False)
        if metric == "fs_time_s":
            for (d, r) in pts:
                if r[metric] is not None and r["cones_hit_geometric"]:
                    y = min(r[metric], cap) if cap else r[metric]
                    ax.annotate(f"{r['cones_hit_geometric']}", (d * 1000, y), textcoords="offset points",
                                xytext=(7, 4), fontsize=8, color=MUTED)
        if cap:
            for d, r in pts:
                if r[metric] is not None and r[metric] > cap:
                    ax.annotate(f"{r[metric]:.0f}", (d * 1000, cap), textcoords="offset points",
                                xytext=(-4, 6), fontsize=8, color=MUTED, ha="right")
    ax.axvline(DESIGN_DELAY_S * 1000, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.text(DESIGN_DELAY_S * 1000, ax.get_ylim()[0], " design delay (0.15 s)", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Actuation delay (ms)", color=TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelcolor=TEXT)
    save(fig, name)


def plot_robustness(table: dict) -> None:
    conditions = ["nominal", "lookahead short", "lookahead long", "steer -20%", "steer +20%"]
    labels = {
        "nominal": "Nominal",
        "lookahead short": "Short lookahead\n(PP 2 m, ours 0.30 v)",
        "lookahead long": "Long lookahead\n(PP 6 m, ours 0.60 v)",
        "steer -20%": "Steering model 20% low",
        "steer +20%": "Steering model 20% high",
    }
    series = ["ours", "pp_pkg", "pp_classic"]
    rows = [c for c in conditions if any((s, c, DESIGN_DELAY_S) in table for s in series)]
    if not rows:
        print("  (no robustness runs yet; skipped carla_robustness)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.75 * len(rows) + 1.2))
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    offsets = np.linspace(-0.22, 0.22, len(series))
    for k, sid in enumerate(series):
        s = SERIES[sid]
        for i, cond in enumerate(rows):
            r = table.get((sid, cond, DESIGN_DELAY_S))
            if r is None:
                continue
            y = i + offsets[k]
            if failed_value(r):
                ax.text(ax.get_xlim()[0], y, f"  {s['short']}: {r['status']}", fontsize=8, color=MUTED, va="center")
                continue
            ax.plot([r["fs_time_s"]], [y], marker=s["marker"], color=s["color"], ms=8,
                    markeredgecolor="white", markeredgewidth=1.0, ls="none",
                    label=s["label"] if i == 0 or not ax.get_legend_handles_labels()[1].count(s["label"]) else None)
            ax.annotate(f"{r['fs_time_s']:.1f}" + (f" ({r['cones_hit_geometric']})" if r["cones_hit_geometric"] else ""),
                        (r["fs_time_s"], y), textcoords="offset points", xytext=(8, -3), fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[c] for c in rows], color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("FS time at 0.15 s delay (s) — lap + 2 s per cone; (n) = cones", color=TEXT, fontsize=10)
    ax.set_title("What happens when a setting is wrong", color=TEXT, fontsize=11, loc="left")
    handles, labels_ = ax.get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.45 / len(rows) - 0.08), ncol=1, labelcolor=TEXT)
    save(fig, "carla_robustness")


def plot_error_along_track(table: dict) -> None:
    pair = [sid for sid in ("pp_classic", "ours") if (sid, "nominal", DESIGN_DELAY_S) in table]
    if len(pair) < 2:
        print("  (need classic PP and ours at 0.15 s; skipped carla_error_along_track)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    style_axes(ax)
    for sid in pair:
        r = table[(sid, "nominal", DESIGN_DELAY_S)]
        rows = list(csv.DictReader((r["dir"] / "run_log.csv").open()))
        t_s = np.array([float(x["t_s"]) for x in rows])
        s_m = np.array([float(x["s_m"]) for x in rows])
        xte = np.abs(np.array([float(x["xte_m"]) for x in rows]))
        # One lap only: drop the standing start and anything after s wraps back to 0.
        keep = (t_s >= sb.IGNORE_BEFORE_S) & (s_m >= np.maximum.accumulate(s_m) - 50.0)
        s_m, xte = s_m[keep], xte[keep]
        # Worst error in each 5 m of track, so the two laps can be compared corner by corner.
        edges = np.arange(0.0, s_m.max() + 5.0, 5.0)
        idx = np.digitize(s_m, edges) - 1
        worst = np.array([xte[idx == i].max() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])
        s = SERIES[sid]
        ax.plot(edges[:-1] + 2.5, worst, color=s["color"], lw=1.6, ls=s["ls"], label=s["label"])
    ax.set_xlabel("Distance along the lap (m)", color=TEXT, fontsize=10)
    ax.set_ylabel("Worst |error| per 5 m (m)", color=TEXT, fontsize=10)
    ax.set_title("Tracking error along the lap at 0.15 s delay", color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT)
    save(fig, "carla_error_along_track")


def write_tables(table: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted(table, key=lambda k: (list(SERIES).index(k[0]), k[1] != "nominal", k[1], k[2]))
    fields = ["series", "condition", "delay_s", "status", "lap_time_s", "cones_hit_geometric", "fs_time_s",
              "xte_mean_m", "xte_p95_m", "xte_max_m", "steer_rate_rms_radps", "run"]
    with (OUT_DIR / "carla_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in keys:
            w.writerow(table[k])
    lines = ["| Controller | Condition | Delay (s) | Lap (s) | Cones | FS time (s) | Mean error (m) | 95% error (m) | Max error (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        r = table[k]
        lap = f"{r['lap_time_s']:.2f}" if r["lap_time_s"] else r["status"]
        fs = f"{r['fs_time_s']:.2f}" if r["fs_time_s"] else "-"
        lines.append(f"| {SERIES[k[0]]['short']} | {k[1]} | {k[2]:.2f} | {lap} | {r['cones_hit_geometric']} | {fs} | "
                     f"{r['xte_mean_m']:.2f} | {r['xte_p95_m']:.2f} | {r['xte_max_m']:.2f} |")
    (OUT_DIR / "carla_results.md").write_text("\n".join(lines) + "\n")
    print("  carla_results.csv / carla_results.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=sb.OUTPUT_ROOT)
    p.add_argument("--any-speed-plan", action="store_true",
                   help="Also use runs without the grip cap (by default only grip-capped runs are compared).")
    args = p.parse_args()

    table = collect(args.root, require_grip=not args.any_speed_plan)
    if not table:
        raise SystemExit("No matching runs found.")
    print(f"{len(table)} runs selected:")
    for (sid, cond, d), r in sorted(table.items()):
        print(f"  {SERIES[sid]['short']:<22} {cond:<16} d={d:.2f}  {r['run']}")
    print(f"Writing to {OUT_DIR}")
    plot_vs_delay(table, "fs_time_s", "FS time (s): lap + 2 s per cone", "carla_fs_time_vs_delay",
                  "FS time vs actuation delay (small numbers = cones hit)", cap=140.0)
    plot_vs_delay(table, "xte_p95_m", "95th-percentile tracking error (m)", "carla_error_vs_delay",
                  "Tracking error vs actuation delay")
    plot_robustness(table)
    plot_error_along_track(table)
    write_tables(table)


if __name__ == "__main__":
    main()
