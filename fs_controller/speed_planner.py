from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from typing import Protocol, Sequence

from .tire_grip import max_lateral_accel_equal_load

MAX_SPEED_30_KMH_MPS = 30.0 / 3.6


class XYPoint(Protocol):
    x: float
    y: float


@dataclass(frozen=True)
class TireAwareSpeedPlannerConfig:
    mass_kg: float = 320.0
    safety_factor: float = 0.75
    max_speed_mps: float = MAX_SPEED_30_KMH_MPS
    min_speed_mps: float = 3.0
    curvature_sample_offset: int = 4
    straight_curvature_threshold: float = 1e-4


def tire_aware_centerline_speeds(
    points: Sequence[XYPoint],
    config: TireAwareSpeedPlannerConfig | None = None,
    closed: bool = True,
) -> list[float]:
    cfg = config or TireAwareSpeedPlannerConfig()
    if not points:
        return []

    lateral_accel_limit = max_lateral_accel_equal_load(cfg.mass_kg)
    speeds: list[float] = []
    for index in range(len(points)):
        curvature = centerline_curvature(
            points,
            index,
            sample_offset=cfg.curvature_sample_offset,
            closed=closed,
        )
        if curvature < cfg.straight_curvature_threshold:
            speed = cfg.max_speed_mps
        else:
            speed = cfg.safety_factor * sqrt(lateral_accel_limit / curvature)
        speeds.append(max(cfg.min_speed_mps, min(cfg.max_speed_mps, speed)))
    return speeds


def centerline_curvature(
    points: Sequence[XYPoint],
    index: int,
    sample_offset: int,
    closed: bool,
) -> float:
    if len(points) < 3:
        return 0.0

    offset = max(1, sample_offset)
    if closed:
        previous = points[(index - offset) % len(points)]
        current = points[index]
        next_point = points[(index + offset) % len(points)]
    else:
        previous = points[max(0, index - offset)]
        current = points[index]
        next_point = points[min(len(points) - 1, index + offset)]

    return _curvature(previous, current, next_point)


def _curvature(a: XYPoint, b: XYPoint, c: XYPoint) -> float:
    ab = hypot(b.x - a.x, b.y - a.y)
    bc = hypot(c.x - b.x, c.y - b.y)
    ca = hypot(a.x - c.x, a.y - c.y)
    if ab * bc * ca <= 1e-6:
        return 0.0

    twice_area = abs(
        (b.x - a.x) * (c.y - a.y)
        - (b.y - a.y) * (c.x - a.x)
    )
    return 2.0 * twice_area / (ab * bc * ca)
