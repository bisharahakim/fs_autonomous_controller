from __future__ import annotations

import math
import os
import sys
import tempfile
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_track_baseline import run_baseline  # noqa: E402
from fs_controller.raceline import load_raceline, wrap_angle  # noqa: E402
from visualization.track_registry import get_track  # noqa: E402


TRACK_NAME = "hockenheim_fsg"
RACELINE_PATH = PROJECT_ROOT / "inputs" / "racelines" / "hockenheim_fsg_benjamin24_safe.csv"
SPEED_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_speed_tracking.png"
LATERAL_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_lateral_accel.png"
INCIDENT_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_incident_locations.png"
TRACKING_ERROR_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_tracking_error.png"
FIRST_5S_PATH = PROJECT_ROOT / "outputs" / "diag_first_5s.csv"
OPTIMIZER_MAX_LAT_ACCEL_MPS2 = 10.30


def main() -> None:
    raceline = load_raceline(RACELINE_PATH)
    align_raceline_psi_to_geometry(raceline)
    result = run_baseline(TRACK_NAME, RACELINE_PATH, speed_cap_mps=None)
    samples = result.samples

    if not samples:
        raise RuntimeError("No samples were produced by the baseline run.")

    nearest_indices = np.array([sample.nearest_index for sample in samples], dtype=int)
    sample_s = raceline.s_m[nearest_indices]
    target_speed_seen = np.array([sample.target_speed_mps for sample in samples])
    actual_speed = np.array([sample.state.speed for sample in samples])

    save_speed_plot(
        raceline_s=raceline.s_m[:-1],
        raceline_speed=raceline.v_target_mps[:-1],
        sample_s=sample_s,
        target_speed_seen=target_speed_seen,
        actual_speed=actual_speed,
    )

    actual_lat_accel = driven_lateral_acceleration(samples)
    planned_lat_accel = np.abs(
        raceline.v_target_mps[nearest_indices] ** 2 * raceline.kappa_radpm[nearest_indices]
    )
    save_lateral_accel_plot(
        time_s=np.array([sample.time_s for sample in samples]),
        actual_lat_accel=actual_lat_accel,
        planned_lat_accel=planned_lat_accel,
    )

    track = get_track(TRACK_NAME)
    centerline = track.build_centerline()
    left_cones, right_cones = track.build_cones()
    save_incident_location_plot(
        centerline=centerline,
        left_cones=left_cones,
        right_cones=right_cones,
        samples=samples,
        events=result.events,
        raceline=raceline,
    )
    tracking_s, tracking_error = raceline_tracking_error(samples, raceline)
    save_tracking_error_plot(tracking_s, tracking_error)
    save_first_5s_csv(samples, [*left_cones, *right_cones], raceline)

    print(f"Speed tracking plot: {SPEED_PLOT_PATH}")
    print(f"Lateral acceleration plot: {LATERAL_PLOT_PATH}")
    print(f"Incident location plot: {INCIDENT_PLOT_PATH}")
    print(f"Tracking error plot: {TRACKING_ERROR_PLOT_PATH}")
    print(f"First 5 seconds CSV: {FIRST_5S_PATH}")
    print(f"Lap completed: {result.completed_lap}")
    print(f"Final/lap time: {samples[-1].time_s:.2f} s")
    print(f"Incidents recorded: {len(result.events)}")
    print_incident_tuples(result.events, samples, raceline)


def align_raceline_psi_to_geometry(raceline) -> None:
    start_index = raceline.closest_index_to(0.0, 0.0)
    psi_error = wrap_angle(float(raceline.psi_rad[start_index]) - raceline.geometric_heading_at(start_index))
    if abs(abs(math.degrees(psi_error)) - 90.0) < 15.0:
        raceline.rotate_psi(-psi_error)


def save_speed_plot(
    raceline_s: np.ndarray,
    raceline_speed: np.ndarray,
    sample_s: np.ndarray,
    target_speed_seen: np.ndarray,
    actual_speed: np.ndarray,
) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    SPEED_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(raceline_s, raceline_speed, color="#111827", linewidth=1.0, alpha=0.75, label="optimizer CSV vx")
    ax.scatter(
        sample_s,
        target_speed_seen,
        color="#2563eb",
        s=8,
        alpha=0.7,
        label="target speed seen by controller",
    )
    ax.scatter(
        sample_s,
        actual_speed,
        color="#dc2626",
        s=8,
        alpha=0.55,
        label="actual vehicle speed",
    )
    ax.set_title("Raceline Speed Tracking Diagnostic")
    ax.set_xlabel("arc length s [m]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(SPEED_PLOT_PATH, dpi=180)
    plt.close(fig)


def save_lateral_accel_plot(
    time_s: np.ndarray,
    actual_lat_accel: np.ndarray,
    planned_lat_accel: np.ndarray,
) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    LATERAL_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(time_s, actual_lat_accel, color="#dc2626", linewidth=1.3, label="actual |v^2 kappa|")
    ax.plot(time_s, planned_lat_accel, color="#2563eb", linewidth=1.2, label="optimizer planned |v_target^2 kappa|")
    ax.axhline(
        OPTIMIZER_MAX_LAT_ACCEL_MPS2,
        color="#111827",
        linewidth=1.0,
        linestyle="--",
        label=f"optimizer max {OPTIMIZER_MAX_LAT_ACCEL_MPS2:.2f} m/s^2",
    )
    ax.set_title("Lateral Acceleration Diagnostic")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral acceleration magnitude [m/s^2]")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(LATERAL_PLOT_PATH, dpi=180)
    plt.close(fig)


def driven_lateral_acceleration(samples) -> np.ndarray:
    if len(samples) < 3:
        return np.zeros(len(samples))

    accelerations = np.zeros(len(samples))
    points = np.array([(sample.state.x, sample.state.y) for sample in samples])
    speeds = np.array([sample.state.speed for sample in samples])
    for i in range(1, len(samples) - 1):
        p_prev = points[i - 1]
        p = points[i]
        p_next = points[i + 1]
        a = np.linalg.norm(p - p_prev)
        b = np.linalg.norm(p_next - p)
        c = np.linalg.norm(p_next - p_prev)
        if a < 1e-9 or b < 1e-9 or c < 1e-9:
            continue
        twice_area = cross_2d(p - p_prev, p_next - p_prev)
        kappa = 2.0 * twice_area / (a * b * c)
        accelerations[i] = abs(speeds[i] * speeds[i] * kappa)

    accelerations[0] = accelerations[1]
    accelerations[-1] = accelerations[-2]
    return accelerations


def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def save_incident_location_plot(
    centerline,
    left_cones,
    right_cones,
    samples,
    events,
    raceline,
) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    cone_hits = [event for event in events if event.kind == "cone_hit"]
    off_tracks = [event for event in events if event.kind == "off_track"]

    INCIDENT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.plot([p.x for p in centerline], [p.y for p in centerline], color="#111827", linewidth=1.2, label="centerline")
    ax.scatter([p.x for p in left_cones], [p.y for p in left_cones], s=8, color="#f59e0b", label="left cones")
    ax.scatter([p.x for p in right_cones], [p.y for p in right_cones], s=8, color="#3b82f6", label="right cones")
    ax.plot(raceline.x_m, raceline.y_m, color="#dc2626", linewidth=1.4, label="planned raceline")
    ax.plot([s.state.x for s in samples], [s.state.y for s in samples], color="#2563eb", linewidth=1.5, label="driven trajectory")
    ax.scatter([samples[0].state.x], [samples[0].state.y], s=70, color="#16a34a", marker="o", label="start", zorder=7)
    if cone_hits:
        ax.scatter([e.x_m for e in cone_hits], [e.y_m for e in cone_hits], s=90, color="#facc15", marker="x", linewidths=2.4, label="cone hit", zorder=8)
    if off_tracks:
        ax.scatter([e.x_m for e in off_tracks], [e.y_m for e in off_tracks], s=80, color="#d946ef", marker="x", linewidths=2.0, label="off track", zorder=8)
    ax.set_title("Hockenheim FSG incident locations")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(INCIDENT_PLOT_PATH, dpi=180)
    plt.close(fig)


def raceline_tracking_error(samples, raceline) -> tuple[np.ndarray, np.ndarray]:
    rx = np.asarray(raceline.x_m[:-1], dtype=float)
    ry = np.asarray(raceline.y_m[:-1], dtype=float)
    rs = np.asarray(raceline.s_m[:-1], dtype=float)
    rx2 = np.roll(rx, -1)
    ry2 = np.roll(ry, -1)
    rs2 = np.roll(rs, -1)
    rs2[-1] = raceline.total_s
    seg_dx = rx2 - rx
    seg_dy = ry2 - ry
    seg_len2 = np.maximum(seg_dx * seg_dx + seg_dy * seg_dy, 1e-12)

    tracking_s = np.zeros(len(samples))
    tracking_error = np.zeros(len(samples))
    point_count = len(rx)
    for sample_index, sample in enumerate(samples):
        nearest = int(sample.nearest_index) % point_count
        best_dist = float("inf")
        best_s = float(rs[nearest])
        for index in [(nearest + offset) % point_count for offset in range(-30, 31)]:
            t = ((sample.state.x - rx[index]) * seg_dx[index] + (sample.state.y - ry[index]) * seg_dy[index]) / seg_len2[index]
            t = min(1.0, max(0.0, float(t)))
            proj_x = rx[index] + t * seg_dx[index]
            proj_y = ry[index] + t * seg_dy[index]
            distance = math.hypot(sample.state.x - proj_x, sample.state.y - proj_y)
            if distance < best_dist:
                if index == point_count - 1:
                    seg_s = (rs2[index] - rs[index]) % raceline.total_s
                    best_s = (rs[index] + t * seg_s) % raceline.total_s
                else:
                    best_s = rs[index] + t * (rs2[index] - rs[index])
                best_dist = distance
        tracking_s[sample_index] = best_s
        tracking_error[sample_index] = best_dist
    return tracking_s, tracking_error


def save_tracking_error_plot(tracking_s: np.ndarray, tracking_error: np.ndarray) -> None:
    setup_matplotlib()
    import matplotlib.pyplot as plt

    TRACKING_ERROR_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(tracking_s, tracking_error, color="#2563eb", linewidth=1.6, label="distance to planned raceline")
    ax.axhline(0.95, color="#dc2626", linewidth=1.4, linestyle="--", label="0.95 m cone-clearance threshold")
    ax.set_title("Hockenheim FSG raceline tracking error")
    ax.set_xlabel("raceline arc length s [m]")
    ax.set_ylabel("lateral distance to raceline [m]")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(TRACKING_ERROR_PLOT_PATH, dpi=180)
    plt.close(fig)


def save_first_5s_csv(samples, cones, raceline) -> None:
    FIRST_5S_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIRST_5S_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "t",
            "x",
            "y",
            "v_actual",
            "steering_cmd",
            "nearest_cone_distance",
            "raceline_target_x",
            "raceline_target_y",
            "lookahead_used",
        ])
        for sample in samples:
            if sample.time_s > 5.0 + 1e-9:
                break
            target = raceline.target_at_index(sample.target_index)
            writer.writerow([
                f"{sample.time_s:.2f}",
                f"{sample.state.x:.6f}",
                f"{sample.state.y:.6f}",
                f"{sample.state.speed:.6f}",
                f"{sample.steering_rad:.6f}",
                f"{nearest_cone_distance(sample.state.x, sample.state.y, cones):.6f}",
                f"{target.x_m:.6f}",
                f"{target.y_m:.6f}",
                f"{sample.lookahead_used_m:.6f}",
            ])


def nearest_cone_distance(x_m: float, y_m: float, cones) -> float:
    return min(math.hypot(x_m - cone.x, y_m - cone.y) for cone in cones)


def print_incident_tuples(events, samples, raceline) -> None:
    print("Incident tuples: kind, t_s, segment, s_m, v_actual, v_target_seen, v_optimizer_at_this_s")
    for event in events:
        sample_index = min(range(len(samples)), key=lambda i: abs(samples[i].time_s - event.time_s))
        sample = samples[sample_index]
        raceline_index = sample.nearest_index
        s_m = raceline.s_m[raceline_index]
        v_optimizer = raceline.v_target_mps[raceline_index]
        print(
            f"{event.kind}, "
            f"t={event.time_s:.2f}, "
            f"segment={event.segment_index}, "
            f"s={s_m:.2f}, "
            f"v_actual={sample.state.speed:.2f}, "
            f"v_target={sample.target_speed_mps:.2f}, "
            f"v_optimizer_at_this_s={v_optimizer:.2f}"
        )


def setup_matplotlib() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")


if __name__ == "__main__":
    main()
