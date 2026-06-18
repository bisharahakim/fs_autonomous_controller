from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from fs_controller import MAX_SPEED_30_KMH_MPS, TireAwareSpeedPlannerConfig, tire_aware_centerline_speeds
except ModuleNotFoundError:
    MAX_SPEED_30_KMH_MPS = 30.0 / 3.6
    TireAwareSpeedPlannerConfig = None
    tire_aware_centerline_speeds = None


@dataclass
class TrackPoint:
    x: float
    y: float
    yaw: float = 0.0
    speed: float = 0.0


def build_fsd_autocross_track() -> list[TrackPoint]:
    """Build a Formula Student Driverless autocross-style centerline.

    This is a synthetic course, not an official layout. It combines common
    autocross features: a launch straight, slalom, sweeper, hairpin, offset
    gates, chicane, and a return straight. Target speed is assigned by a simple
    curvature-based speed planner.
    """

    control_points = [
        (8.0, 5.0),
        (22.0, 0.0),
        (36.0, 5.0),
        (50.0, -2.5),
        (64.0, 5.5),
        (78.0, -1.5),
        (92.0, 5.0),
        (107.0, 15.0),
        (116.0, 30.0),
        (115.0, 45.0),
        (105.0, 58.0),
        (88.0, 66.0),
        (70.0, 63.0),
        (58.0, 72.0),
        (42.0, 68.0),
        (31.0, 57.0),
        (22.0, 62.0),
        (11.0, 54.0),
        (7.0, 40.0),
        (13.0, 28.0),
        (5.0, 18.0),
        (8.0, 5.0),
    ]

    points = _sample_catmull_rom(control_points, samples_per_segment=18)
    _assign_yaw(points)
    _assign_curvature_speed(points)
    return points


def build_cone_boundaries(
    centerline: list[TrackPoint],
    track_width_m: float,
    cone_spacing_m: float,
) -> tuple[list[TrackPoint], list[TrackPoint]]:
    left: list[TrackPoint] = []
    right: list[TrackPoint] = []
    distance_since_cone = 0.0
    previous = centerline[0]

    for i, point in enumerate(centerline):
        distance_since_cone += math.hypot(point.x - previous.x, point.y - previous.y)
        previous = point
        if distance_since_cone < cone_spacing_m and i != 0:
            continue
        distance_since_cone = 0.0

        normal_x = -math.sin(point.yaw)
        normal_y = math.cos(point.yaw)
        left.append(
            TrackPoint(
                x=point.x + normal_x * track_width_m / 2.0,
                y=point.y + normal_y * track_width_m / 2.0,
            )
        )
        right.append(
            TrackPoint(
                x=point.x - normal_x * track_width_m / 2.0,
                y=point.y - normal_y * track_width_m / 2.0,
            )
        )

    return left, right


def _sample_catmull_rom(
    control_points: list[tuple[float, float]],
    samples_per_segment: int,
) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    is_closed = control_points[0] == control_points[-1]

    if is_closed:
        unique_points = control_points[:-1]
        for i in range(len(unique_points)):
            p0 = unique_points[(i - 1) % len(unique_points)]
            p1 = unique_points[i]
            p2 = unique_points[(i + 1) % len(unique_points)]
            p3 = unique_points[(i + 2) % len(unique_points)]
            for j in range(samples_per_segment):
                t = j / samples_per_segment
                x, y = _catmull_rom_point(p0, p1, p2, p3, t)
                if points and math.hypot(points[-1].x - x, points[-1].y - y) < 0.2:
                    continue
                points.append(TrackPoint(x=x, y=y))
        points.append(TrackPoint(points[0].x, points[0].y))
        return points

    extended = [control_points[0], *control_points, control_points[-1]]

    for i in range(1, len(extended) - 2):
        p0 = extended[i - 1]
        p1 = extended[i]
        p2 = extended[i + 1]
        p3 = extended[i + 2]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            x, y = _catmull_rom_point(p0, p1, p2, p3, t)
            if points and math.hypot(points[-1].x - x, points[-1].y - y) < 0.2:
                continue
            points.append(TrackPoint(x=x, y=y))

    points.append(TrackPoint(*control_points[-1]))
    return points


def _catmull_rom_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        2.0 * p1[0]
        + (-p0[0] + p2[0]) * t
        + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        2.0 * p1[1]
        + (-p0[1] + p2[1]) * t
        + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
    )
    return x, y


def _assign_yaw(points: list[TrackPoint]) -> None:
    for i, point in enumerate(points):
        prev = points[max(0, i - 1)]
        nxt = points[min(len(points) - 1, i + 1)]
        point.yaw = math.atan2(nxt.y - prev.y, nxt.x - prev.x)


def _assign_curvature_speed(points: list[TrackPoint]) -> None:
    if tire_aware_centerline_speeds is None or TireAwareSpeedPlannerConfig is None:
        raw_speeds = [12.0 for _ in points]
    else:
        raw_speeds = tire_aware_centerline_speeds(
            points,
            TireAwareSpeedPlannerConfig(
                mass_kg=320.0,
                safety_factor=0.75,
                max_speed_mps=MAX_SPEED_30_KMH_MPS,
                min_speed_mps=3.0,
                curvature_sample_offset=6,
            ),
            closed=True,
        )

    smoothed = raw_speeds[:]
    for _ in range(3):
        next_speeds = smoothed[:]
        for i in range(1, len(smoothed) - 1):
            next_speeds[i] = 0.25 * smoothed[i - 1] + 0.5 * smoothed[i] + 0.25 * smoothed[i + 1]
        smoothed = next_speeds

    for point, speed in zip(points, smoothed):
        point.speed = speed


def _curvature(a: TrackPoint, b: TrackPoint, c: TrackPoint) -> float:
    ab = math.hypot(b.x - a.x, b.y - a.y)
    bc = math.hypot(c.x - b.x, c.y - b.y)
    ca = math.hypot(a.x - c.x, a.y - c.y)
    if ab * bc * ca <= 1e-6:
        return 0.0

    twice_area = abs(
        (b.x - a.x) * (c.y - a.y)
        - (b.y - a.y) * (c.x - a.x)
    )
    return 2.0 * twice_area / (ab * bc * ca)
