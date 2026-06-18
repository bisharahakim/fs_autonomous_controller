from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from autocross_track import TrackPoint, build_cone_boundaries
except ModuleNotFoundError:
    from visualization.autocross_track import TrackPoint, build_cone_boundaries

from fs_controller import (
    MAX_SPEED_30_KMH_MPS,
    TireAwareSpeedPlannerConfig,
    max_lateral_accel_equal_load,
    tire_aware_centerline_speeds,
)


SOLID_FAC = 0.322
TRACK_WIDTH_M = 3.0
CONE_SPACING_M = 2.5
DEFAULT_DS_M = 0.5
SCALED_ENDURANCE_TARGET_M = 22000.0 * SOLID_FAC
LONGEST_STRAIGHT_INDEX = 40
TIGHTEST_HAIRPIN_INDEX = 8

STRAIGHT_LINE = [
    94.95, 0, 0, 0, 0, 94.58, 0, 67.85, 0, 123.119, 0, 0, 23.68, 0,
    22.68, 60.15, 0, 119.42, 0, 0, 0, 0, 0, 0, 0, 0, 14.72, 0, 0, 0, 30.48, 25.84,
    0, 0, 27.36, 0, 41.8, 0, 0, 0, 158.15, 0, 0, 0, 0, 0, 20.8, 106.29, 14.81,
    36.7908, 43.32, 0, 0, 75.87, 0, 29.057, 65.723, 0, 0
]

R = [
    39.26, 87.94, 58.02, 40.6, 58.59, 65.79, 35.2, 83.14, 9.67, 73.59, 35.45,
    42.09, 25.243, 20.41, 28.66, 31.74, 54.83, 28.36, 43.65, 38.65, 44.13, 247.07,
    23.53, 65.13, 26.9, 26.9, 78.76, 29.26, 14.96, 47.21, 52.05, 35.86, 20.25,
    97.44, 20.08, 41.62, 59.58, 32.04, 52.5, 52.5, 22.17, 34.96, 75.11, 105.98,
    58.87, 41.15, 42.06, 41.95, 191.91, 36.497, 30.58, 30.58, 27.67, 16.354,
    16.354, 38.961, 18.961, 51.375, 107.61
]

ALPHA = [
    0.1263, 0.0536, -0.0796, -0.1361, 0.09445, 0.08319, -0.0753, 0.1229,
    0.3817, -0.0883, 0.16938, -0.1948, 0.20819, -0.25, -0.25, -0.2345, 0.1333,
    0.2859, 0.1586, 0.1854, -0.2312, -0.0115, -0.14269, 0.07327, -0.34963,
    -0.00764, 0.0305, 0.07328, -0.1713, 0.125, -0.2805, 0.1451, 0.2516, 0.17216,
    -0.3039, -0.2026, -0.1421, 0.4417, 0.08688, 0.1024, 0.20825, -0.105972,
    0.06288, 0.08356, 0.07425, -0.14816, 0.07755, 0.15188, -0.0501, -0.1741,
    0.34822, 0.07588, -0.17244, -0.15513, -0.17094, 0.08827, 0.2078, 0.17597,
    0.05272
]


def build_fsg_hockenheim_endurance_track(ds_m: float = DEFAULT_DS_M) -> list[TrackPoint]:
    """Build the FSG Hockenheim endurance centerline from MATLAB segment data."""

    if ds_m <= 0.0:
        raise ValueError("ds_m must be positive")
    if not (len(STRAIGHT_LINE) == len(R) == len(ALPHA)):
        raise ValueError("FSG Hockenheim arrays must have the same length")

    points = _build_raw_centerline(ds_m)
    centerline = _close_loop_pose(points)
    _assign_tire_aware_speeds(centerline)
    return centerline


def build_fsg_hockenheim_raw_centerline(ds_m: float = DEFAULT_DS_M) -> list[TrackPoint]:
    """Build the uncorrected centerline from the pasted MATLAB segment arrays."""

    if ds_m <= 0.0:
        raise ValueError("ds_m must be positive")
    if not (len(STRAIGHT_LINE) == len(R) == len(ALPHA)):
        raise ValueError("FSG Hockenheim arrays must have the same length")

    return _build_raw_centerline(ds_m)


def build_fsg_hockenheim_segment_indices(ds_m: float = DEFAULT_DS_M) -> list[int]:
    """Return the source segment index aligned with the generated centerline."""

    if ds_m <= 0.0:
        raise ValueError("ds_m must be positive")
    if not (len(STRAIGHT_LINE) == len(R) == len(ALPHA)):
        raise ValueError("FSG Hockenheim arrays must have the same length")

    _, segment_indices, _ = _build_raw_centerline_with_metadata(ds_m)
    return segment_indices


def _build_raw_centerline(ds_m: float) -> list[TrackPoint]:
    points, _, _ = _build_raw_centerline_with_metadata(ds_m)
    return points


def _build_raw_centerline_with_features(
    ds_m: float,
) -> tuple[list[TrackPoint], dict[str, tuple[TrackPoint, float] | float]]:
    points, _, features = _build_raw_centerline_with_metadata(ds_m)
    return points, features


def _build_raw_centerline_with_metadata(
    ds_m: float,
) -> tuple[list[TrackPoint], list[int], dict[str, tuple[TrackPoint, float] | float]]:
    x = 0.0
    y = 0.0
    theta = 0.0
    points = [TrackPoint(x=x, y=y, yaw=theta, speed=25.0)]
    segment_indices = [0]
    distance = 0.0
    features: dict[str, tuple[TrackPoint, float] | float] = {}

    for segment_index, (straight_m, radius_m, alpha) in enumerate(zip(STRAIGHT_LINE, R, ALPHA)):
        straight_length = straight_m * SOLID_FAC
        if segment_index == LONGEST_STRAIGHT_INDEX:
            features["longest_straight_start"] = (TrackPoint(x=x, y=y, yaw=theta, speed=25.0), distance)
        n_steps = max(1, int(straight_length / ds_m))
        step = straight_length / n_steps
        for _ in range(n_steps):
            x += step * math.cos(theta)
            y += step * math.sin(theta)
            distance += step
            points.append(TrackPoint(x=x, y=y, yaw=theta, speed=25.0))
            segment_indices.append(segment_index)
        if segment_index == LONGEST_STRAIGHT_INDEX:
            features["longest_straight_end"] = (TrackPoint(x=x, y=y, yaw=theta, speed=25.0), distance)

        radius = radius_m * SOLID_FAC
        sweep = 2.0 * math.pi * alpha
        arc_length = abs(sweep) * radius
        n_steps = max(1, int(arc_length / ds_m))
        dtheta = sweep / n_steps
        step = arc_length / n_steps
        target_speed = _corner_speed(radius)
        turn_direction = 1.0 if sweep >= 0.0 else -1.0
        if segment_index == TIGHTEST_HAIRPIN_INDEX:
            center_x = x - turn_direction * radius * math.sin(theta)
            center_y = y + turn_direction * radius * math.cos(theta)
            features["tightest_hairpin_center"] = (
                TrackPoint(x=center_x, y=center_y, yaw=theta, speed=target_speed),
                distance + 0.5 * arc_length,
            )
            features["tightest_hairpin_radius"] = radius
        for _ in range(n_steps):
            next_theta = theta + dtheta
            x += radius * turn_direction * (math.sin(next_theta) - math.sin(theta))
            y -= radius * turn_direction * (math.cos(next_theta) - math.cos(theta))
            theta = next_theta
            distance += step
            points.append(TrackPoint(x=x, y=y, yaw=theta, speed=target_speed))
            segment_indices.append(segment_index)

    features["raw_final_point"] = (points[-1], distance)
    return points, segment_indices, features


def build_fsg_hockenheim_cones(
    ds_m: float = DEFAULT_DS_M,
    track_width_m: float = TRACK_WIDTH_M,
    cone_spacing_m: float = CONE_SPACING_M,
) -> tuple[list[TrackPoint], list[TrackPoint]]:
    centerline = build_fsg_hockenheim_endurance_track(ds_m)
    return build_cone_boundaries(centerline, track_width_m, cone_spacing_m)


def fsg_hockenheim_track_length_m() -> float:
    return sum(_segment_lengths_m())


def save_sanity_plot(output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centerline = build_fsg_hockenheim_endurance_track()
    left_cones, right_cones = build_cone_boundaries(
        centerline,
        track_width_m=TRACK_WIDTH_M,
        cone_spacing_m=CONE_SPACING_M,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(
        [point.x for point in centerline],
        [point.y for point in centerline],
        color="#111827",
        linewidth=1.4,
        label="centerline",
    )
    ax.scatter(
        [point.x for point in left_cones],
        [point.y for point in left_cones],
        s=8,
        color="#f4a23a",
        label="left cones",
    )
    ax.scatter(
        [point.x for point in right_cones],
        [point.y for point in right_cones],
        s=8,
        color="#3aa0f4",
        label="right cones",
    )
    ax.scatter(
        [centerline[0].x],
        [centerline[0].y],
        s=42,
        color="#16a34a",
        marker="s",
        label="start / finish",
        zorder=4,
    )
    ax.set_title("FSG Hockenheim Endurance Centerline and Cones")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_labeled_sanity_plot(output_path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "fs_autonomous_controller_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_centerline, features = _build_raw_centerline_with_features(DEFAULT_DS_M)
    centerline = _close_loop_pose(raw_centerline)
    left_cones, right_cones = build_cone_boundaries(
        centerline,
        track_width_m=TRACK_WIDTH_M,
        cone_spacing_m=CONE_SPACING_M,
    )
    total_length = features["raw_final_point"][1]
    end = raw_centerline[-1]
    end_heading = _wrap_to_pi(end.yaw - raw_centerline[0].yaw)

    def corrected_feature(name: str) -> TrackPoint:
        point, distance = features[name]
        return _apply_closure_transform(point, distance, total_length, end, end_heading)

    straight_start = corrected_feature("longest_straight_start")
    straight_end = corrected_feature("longest_straight_end")
    hairpin_center = corrected_feature("tightest_hairpin_center")
    hairpin_radius = float(features["tightest_hairpin_radius"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.plot(
        [point.x for point in centerline],
        [point.y for point in centerline],
        color="#111827",
        linewidth=1.3,
        label="centerline",
    )
    ax.scatter(
        [point.x for point in left_cones],
        [point.y for point in left_cones],
        s=8,
        color="#f4a23a",
        label="left cones",
    )
    ax.scatter(
        [point.x for point in right_cones],
        [point.y for point in right_cones],
        s=8,
        color="#3aa0f4",
        label="right cones",
    )
    ax.scatter(
        [centerline[0].x],
        [centerline[0].y],
        s=65,
        color="#16a34a",
        marker="o",
        label="start / finish (0, 0)",
        zorder=5,
    )
    ax.scatter(
        [straight_start.x, straight_end.x],
        [straight_start.y, straight_end.y],
        s=70,
        color="#7c3aed",
        marker="D",
        label="straight_line[40] endpoints",
        zorder=5,
    )
    ax.plot(
        [straight_start.x, straight_end.x],
        [straight_start.y, straight_end.y],
        color="#7c3aed",
        linewidth=2.0,
        zorder=4,
    )
    ax.text(
        straight_start.x,
        straight_start.y - 4.0,
        "straight_line[40] start",
        color="#5b21b6",
        fontsize=9,
        ha="right",
    )
    ax.text(
        straight_end.x,
        straight_end.y + 4.0,
        "straight_line[40] end\n158.15 * 0.322 = 50.9 m",
        color="#5b21b6",
        fontsize=9,
        ha="left",
    )
    hairpin_circle = plt.Circle(
        (hairpin_center.x, hairpin_center.y),
        hairpin_radius,
        fill=False,
        color="#dc2626",
        linewidth=2.0,
        label="R[8] tightest hairpin",
        zorder=5,
    )
    ax.add_patch(hairpin_circle)
    ax.scatter(
        [hairpin_center.x],
        [hairpin_center.y],
        s=36,
        color="#dc2626",
        marker="+",
        zorder=6,
    )
    ax.text(
        hairpin_center.x + 4.0,
        hairpin_center.y,
        "R[8] = 9.67 * 0.322 = 3.11 m",
        color="#991b1b",
        fontsize=9,
        va="center",
    )
    ax.text(
        centerline[0].x + 3.0,
        centerline[0].y - 5.0,
        "start / finish (0, 0)",
        color="#166534",
        fontsize=9,
    )
    ax.set_title("FSG Hockenheim Endurance Labeled Sanity Check")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _segment_lengths_m() -> list[float]:
    lengths: list[float] = []
    for straight_m, radius_m, alpha in zip(STRAIGHT_LINE, R, ALPHA):
        lengths.append(straight_m * SOLID_FAC)
        lengths.append(abs(2.0 * math.pi * alpha) * radius_m * SOLID_FAC)
    return lengths


def _close_loop_pose(points: list[TrackPoint]) -> list[TrackPoint]:
    """Distribute the small MATLAB-copy closure error over the full sampled lap."""

    if len(points) < 2:
        return points[:]

    cumulative_lengths = [0.0]
    for previous, point in zip(points, points[1:]):
        cumulative_lengths.append(
            cumulative_lengths[-1] + math.hypot(point.x - previous.x, point.y - previous.y)
        )

    total_length = cumulative_lengths[-1]
    if total_length <= 0.0:
        return points[:]

    end = points[-1]
    end_heading = _wrap_to_pi(end.yaw - points[0].yaw)
    closed: list[TrackPoint] = []

    for point, distance in zip(points, cumulative_lengths):
        progress = distance / total_length
        closed.append(_apply_closure_transform(point, distance, total_length, end, end_heading))

    closed[-1].x = closed[0].x
    closed[-1].y = closed[0].y
    closed[-1].yaw = closed[0].yaw
    _assign_yaw_from_geometry(closed)
    return closed


def _apply_closure_transform(
    point: TrackPoint,
    distance: float,
    total_length: float,
    end: TrackPoint,
    end_heading: float,
) -> TrackPoint:
    progress = distance / total_length
    correction_angle = -progress * end_heading
    translated_x = point.x - progress * end.x
    translated_y = point.y - progress * end.y
    cos_a = math.cos(correction_angle)
    sin_a = math.sin(correction_angle)
    return TrackPoint(
        x=cos_a * translated_x - sin_a * translated_y,
        y=sin_a * translated_x + cos_a * translated_y,
        yaw=point.yaw + correction_angle,
        speed=point.speed,
    )


def _assign_yaw_from_geometry(points: list[TrackPoint]) -> None:
    if len(points) < 2:
        return

    is_closed = math.hypot(points[-1].x - points[0].x, points[-1].y - points[0].y) < 1e-6
    last_unique_index = len(points) - 2 if is_closed and len(points) > 2 else len(points) - 1

    for i, point in enumerate(points):
        if is_closed and i == len(points) - 1:
            point.yaw = points[0].yaw
            continue

        previous_index = i - 1
        next_index = i + 1
        if is_closed:
            if i == 0:
                previous_index = last_unique_index
            elif i == last_unique_index:
                next_index = 0
        else:
            previous_index = max(0, previous_index)
            next_index = min(len(points) - 1, next_index)

        previous = points[previous_index]
        next_point = points[next_index]
        point.yaw = math.atan2(next_point.y - previous.y, next_point.x - previous.x)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _corner_speed(radius_m: float) -> float:
    if radius_m <= 0.0:
        return 3.0
    speed = 0.75 * math.sqrt(max_lateral_accel_equal_load(320.0) * radius_m)
    return max(3.0, min(MAX_SPEED_30_KMH_MPS, speed))


def _assign_tire_aware_speeds(points: list[TrackPoint]) -> None:
    speeds = tire_aware_centerline_speeds(
        points,
        TireAwareSpeedPlannerConfig(
            mass_kg=320.0,
            safety_factor=0.75,
            max_speed_mps=MAX_SPEED_30_KMH_MPS,
            min_speed_mps=3.0,
            curvature_sample_offset=4,
        ),
        closed=True,
    )
    for point, speed in zip(points, speeds):
        point.speed = speed


def main() -> None:
    centerline = build_fsg_hockenheim_endurance_track()
    raw_centerline = build_fsg_hockenheim_raw_centerline()
    left_cones, right_cones = build_cone_boundaries(
        centerline,
        track_width_m=TRACK_WIDTH_M,
        cone_spacing_m=CONE_SPACING_M,
    )
    length_m = fsg_hockenheim_track_length_m()
    output_path = Path(__file__).resolve().parents[1] / "outputs" / "hockenheim_fsg_sanity.png"
    labeled_output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / "hockenheim_fsg_sanity_labeled.png"
    )
    save_sanity_plot(output_path)
    save_labeled_sanity_plot(labeled_output_path)

    print(f"FSG Hockenheim centerline points: {len(centerline)}")
    print(f"Left cones: {len(left_cones)} | Right cones: {len(right_cones)}")
    print(
        "Raw final centerline point before closure: "
        f"({raw_centerline[-1].x:.3f}, {raw_centerline[-1].y:.3f})"
    )
    print(
        "Raw start/end gap: "
        f"{math.hypot(raw_centerline[-1].x - raw_centerline[0].x, raw_centerline[-1].y - raw_centerline[0].y):.2f} m"
    )
    print(
        "Corrected start/end gap: "
        f"{math.hypot(centerline[-1].x - centerline[0].x, centerline[-1].y - centerline[0].y):.2f} m"
    )
    print(f"Reconstructed one-lap track length: {length_m:.2f} m")
    print(f"Scaled 22 km endurance target: {SCALED_ENDURANCE_TARGET_M:.2f} m")
    print(f"Target / reconstructed lap length: {SCALED_ENDURANCE_TARGET_M / length_m:.2f} laps")
    print("Length check: pasted arrays reconstruct one scaled lap; 7084 m is the scaled endurance distance.")
    print(f"Saved sanity plot to {output_path}")
    print(f"Saved labeled sanity plot to {labeled_output_path}")


if __name__ == "__main__":
    main()
