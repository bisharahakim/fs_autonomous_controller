from __future__ import annotations

from dataclasses import dataclass
import math

from .raceline import Raceline, RacelineTarget


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
    target_x: float
    target_y: float
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
        nearest = self.raceline.nearest_index(x, y)
        s_now = float(self.raceline.s_m[nearest])
        lookahead, lookahead_speed, lookahead_curv, kappa_window = self.compute_lookahead(v, s_now)
        target = self.raceline.target_at_s(s_now + lookahead)
        speed_target, s_lookup = self.speed_target(s_now, v)

        dx = target.x_m - x
        dy = target.y_m - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
        curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
        steering = clamp(
            math.atan(self.config.wheelbase_m * curvature),
            -self.config.max_steer_rad,
            self.config.max_steer_rad,
        )

        return RAPPCommand(
            steering=steering,
            target_speed=speed_target.v_target_mps,
            target_accel=speed_target.a_target_mps2,
            target_index=target.index,
            target_x=target.x_m,
            target_y=target.y_m,
            nearest_index=nearest,
            lookahead_used=lookahead,
            s_now=s_now,
            s_lookup=s_lookup,
            kappa_window=kappa_window,
            lookahead_speed=lookahead_speed,
            lookahead_curv=lookahead_curv,
        )

    def compute_lookahead(self, v: float, s_now: float) -> tuple[float, float, float, float]:
        config = self.config
        sample_count = 20
        window_start = s_now - config.horizon_behind
        window_length = config.horizon_behind + config.horizon_ahead
        s_samples = [
            (window_start + window_length * i / (sample_count - 1)) % self.raceline.total_s
            for i in range(sample_count)
        ]
        kappa_window = max(abs(self.raceline.kappa_at(s)) for s in s_samples)
        lookahead_speed = v * config.speed_gain
        lookahead_curv = 1.0 / max(kappa_window, 1e-3)
        lookahead = clamp(
            min(lookahead_speed, lookahead_curv),
            config.lookahead_min,
            config.lookahead_max,
        )
        return lookahead, lookahead_speed, lookahead_curv, kappa_window

    def speed_target(self, s_now: float, v: float) -> tuple[RacelineTarget, float]:
        brake_distance = max(
            self.config.brake_dist_min,
            v * v / (2.0 * self.config.brake_decel),
        )
        s_lookup = s_now + brake_distance
        return self.raceline.target_at_s(s_lookup), s_lookup


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
