from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_track_baseline import run_baseline  # noqa: E402
from fs_controller.raceline import load_raceline  # noqa: E402


TRACK_NAME = "hockenheim_fsg"
RACELINE_PATH = PROJECT_ROOT / "inputs" / "racelines" / "hockenheim_fsg_benjamin24.csv"
SPEED_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_speed_tracking.png"
LATERAL_PLOT_PATH = PROJECT_ROOT / "outputs" / "diag_lateral_accel.png"
OPTIMIZER_MAX_LAT_ACCEL_MPS2 = 13.27


def main() -> None:
    raceline = load_raceline(RACELINE_PATH)
    result = run_baseline(TRACK_NAME, RACELINE_PATH, speed_cap_mps=None)
    samples = result.samples

    if not samples:
        raise RuntimeError("No samples were produced by the baseline run.")

    nearest_indices = np.array([sample.nearest_index for sample in samples], dtype=int)
    sample_s = raceline.s_m[nearest_indices]
    target_speed_seen = np.array([sample.target_speed_mps for sample in samples])
    actual_speed = np.array([sample.state.speed for sample in samples])
    optimizer_speed_at_sample = raceline.v_target_mps[nearest_indices]

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

    print(f"Speed tracking plot: {SPEED_PLOT_PATH}")
    print(f"Lateral acceleration plot: {LATERAL_PLOT_PATH}")
    print_incident_tuples(result.events, samples, raceline)


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
        label="optimizer max 13.27 m/s^2",
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
