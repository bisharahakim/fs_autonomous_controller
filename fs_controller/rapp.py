from __future__ import annotations

from dataclasses import dataclass

from .raceline import Raceline


@dataclass(frozen=True)
class RAPPConfig:
    speed_gain: float = 0.45
    lookahead_min: float = 2.0
    lookahead_max: float = 6.0
    horizon_ahead: float = 10.0
    horizon_behind: float = 2.0
    brake_decel: float = 8.0
    brake_dist_min: float = 3.0
    wheelbase_m: float = 1.55
    max_steer_rad: float = 0.5


@dataclass(frozen=True)
class RAPPCommand:
    steering: float
    target_speed: float
    target_accel: float
    target_index: int
    nearest_index: int
    lookahead_used: float
    s_now: float
    s_lookup: float
    kappa_window: float
    lookahead_speed: float
    lookahead_curv: float


class RAPP:
    """Adaptive Regulated Pure Pursuit controller for raceline tracking."""

    def __init__(self, raceline: Raceline, config: RAPPConfig | None = None) -> None:
        self.raceline = raceline
        self.config = config or RAPPConfig()

    def compute_control(self, x: float, y: float, yaw: float, v: float) -> RAPPCommand:
        raise NotImplementedError


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
