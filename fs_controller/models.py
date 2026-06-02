from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    """Vehicle pose and speed in the map frame."""

    x: float
    y: float
    yaw: float
    speed: float


@dataclass(frozen=True)
class PathPoint:
    """A reference path point.

    speed is optional per point; the high-level controller falls back to its
    configured cruise speed when speed is None.
    """

    x: float
    y: float
    speed: float | None = None
