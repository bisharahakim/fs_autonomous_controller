"""Lateral (steering) controllers behind one common interface.

All three controllers answer the same question -- "given the vehicle pose and
speed, what front-wheel steering angle do we command?" -- so the benchmark
runner can swap them while everything else (vehicle, raceline, speed
reference, velocity PID, timestep) stays identical.

Frame convention (same as fs_controller/rapp.py):
    x, y in metres, yaw in radians, counter-clockwise positive.
    steering > 0 turns LEFT.
    cross-track error > 0 means the vehicle is LEFT of the raceline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .raceline import Raceline
from .rapp import RAPP, RAPPConfig


@dataclass(frozen=True)
class LateralCommand:
    steering: float
    nearest_index: int
    target_index: int
    target_x: float
    target_y: float
    lookahead_m: float
    cross_track_m: float
    lookahead_mode: str = "fixed"   # which rule set the lookahead (for display only)


def signed_cross_track(raceline: Raceline, x: float, y: float, nearest: int) -> float:
    """Signed distance from (x, y) to the raceline polyline near `nearest`.

    Uses the two segments touching the nearest vertex, so the value does not
    jump by half a point spacing the way a nearest-vertex distance does.
    Positive = vehicle is left of the path (path direction = increasing s).
    """
    n = len(raceline.x_m) - 1  # last CSV row repeats the first point
    best_dist = float("inf")
    best_signed = 0.0
    for i in ((nearest - 1) % n, nearest % n):
        j = (i + 1) % n
        ax, ay = float(raceline.x_m[i]), float(raceline.y_m[i])
        tx, ty = float(raceline.x_m[j]) - ax, float(raceline.y_m[j]) - ay
        seg_sq = tx * tx + ty * ty
        if seg_sq < 1e-12:
            continue
        px, py = x - ax, y - ay
        u = min(1.0, max(0.0, (px * tx + py * ty) / seg_sq))
        dx, dy = px - u * tx, py - u * ty
        dist = math.hypot(dx, dy)
        if dist < best_dist:
            best_dist = dist
            cross = tx * py - ty * px  # > 0 when the point is left of the segment
            best_signed = math.copysign(dist, cross) if dist > 0.0 else 0.0
    return best_signed


def _pure_pursuit_steering(
    x: float,
    y: float,
    yaw: float,
    target_x: float,
    target_y: float,
    wheelbase_m: float,
    max_steer_rad: float,
) -> float:
    """Identical geometry to RAPP.compute_control, kept here so standard Pure
    Pursuit and RAPP differ ONLY in how the lookahead distance is chosen."""
    dx = target_x - x
    dy = target_y - y
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
    curvature = 0.0 if local_x <= 0.0 else 2.0 * local_y / distance_sq
    return _clamp(math.atan(wheelbase_m * curvature), -max_steer_rad, max_steer_rad)


class RAPPLateral:
    """Adaptive Regulated Pure Pursuit. Thin wrapper: rapp.py is used unchanged."""

    name = "rapp"

    def __init__(self, raceline: Raceline, config: RAPPConfig) -> None:
        self.raceline = raceline
        self.config = config
        self._rapp = RAPP(raceline, config)

    def reset(self) -> None:
        pass  # RAPP is stateless

    def compute(self, x: float, y: float, yaw: float, v: float, dt_s: float) -> LateralCommand:
        cmd = self._rapp.compute_control(x, y, yaw, v)
        return LateralCommand(
            steering=cmd.steering,
            nearest_index=cmd.nearest_index,
            target_index=cmd.target_index,
            target_x=cmd.target_x,
            target_y=cmd.target_y,
            lookahead_m=cmd.lookahead_used,
            cross_track_m=signed_cross_track(self.raceline, x, y, cmd.nearest_index),
            lookahead_mode=self._mode(cmd),
        )

    def _mode(self, cmd) -> str:
        term = "curvature" if cmd.lookahead_curv < cmd.lookahead_speed else "speed"
        raw = min(cmd.lookahead_curv, cmd.lookahead_speed)
        if raw < self.config.lookahead_min:
            return f"{term}, at floor"
        if raw > self.config.lookahead_max:
            return f"{term}, at ceiling"
        return term

    def params(self) -> dict:
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}


@dataclass(frozen=True)
class PurePursuitLateralConfig:
    wheelbase_m: float = 2.51
    max_steer_rad: float = 0.5
    lookahead_m: float = 4.0       # fixed lookahead (the classic formulation)
    lookahead_gain_s: float = 0.0  # optional speed term: Ld = lookahead_m + gain * v
    lookahead_min: float = 2.0
    lookahead_max: float = 10.0


class PurePursuitLateral:
    """Standard Pure Pursuit on the closed raceline.

    Same nearest-point search, same arc-length target lookup and same steering
    geometry as RAPP. The only difference is the lookahead law: fixed (or
    speed-proportional) here, curvature-regulated in RAPP.
    """

    name = "pp"

    def __init__(self, raceline: Raceline, config: PurePursuitLateralConfig) -> None:
        self.raceline = raceline
        self.config = config

    def reset(self) -> None:
        pass  # stateless

    def compute(self, x: float, y: float, yaw: float, v: float, dt_s: float) -> LateralCommand:
        c = self.config
        nearest = self.raceline.nearest_index(x, y)
        s_now = float(self.raceline.s_m[nearest])
        lookahead = _clamp(c.lookahead_m + c.lookahead_gain_s * v, c.lookahead_min, c.lookahead_max)
        target = self.raceline.target_at_s(s_now + lookahead)
        steering = _pure_pursuit_steering(
            x, y, yaw, target.x_m, target.y_m, c.wheelbase_m, c.max_steer_rad
        )
        return LateralCommand(
            steering=steering,
            nearest_index=nearest,
            target_index=target.index,
            target_x=target.x_m,
            target_y=target.y_m,
            lookahead_m=lookahead,
            cross_track_m=signed_cross_track(self.raceline, x, y, nearest),
        )

    def params(self) -> dict:
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}


@dataclass(frozen=True)
class CrossTrackPIDConfig:
    kp: float = 0.50                 # rad of steering per metre of error
    ki: float = 0.02                 # rad per (metre * second)
    kd: float = 0.15                 # rad per (metre / second)
    max_steer_rad: float = 0.5
    integral_limit_rad: float = 0.15  # largest steering the I term may add
    derivative_tau_s: float = 0.10    # low-pass on the D term


class CrossTrackPID:
    """Steering PID on the signed cross-track error only.

    Deliberately the simple reactive baseline: no lookahead, no heading term,
    no curvature feed-forward. It reacts to error after it appears.
    """

    name = "pid"

    def __init__(self, raceline: Raceline, config: CrossTrackPIDConfig) -> None:
        self.raceline = raceline
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None
        self._d_filtered = 0.0

    def compute(self, x: float, y: float, yaw: float, v: float, dt_s: float) -> LateralCommand:
        c = self.config
        nearest = self.raceline.nearest_index(x, y)
        error = signed_cross_track(self.raceline, x, y, nearest)

        if self._prev_error is None or dt_s <= 0.0:
            d_raw = 0.0
        else:
            d_raw = (error - self._prev_error) / dt_s
        self._prev_error = error
        alpha = dt_s / (c.derivative_tau_s + dt_s) if dt_s > 0.0 else 1.0
        self._d_filtered += alpha * (d_raw - self._d_filtered)

        # Left of the path (error > 0) must steer right (negative), hence the minus.
        unsaturated = -(c.kp * error + c.ki * self._integral + c.kd * self._d_filtered)
        steering = _clamp(unsaturated, -c.max_steer_rad, c.max_steer_rad)

        # Anti-windup: stop integrating while saturated in the direction that
        # would push further into saturation; also clamp the I contribution.
        saturated = unsaturated != steering
        pushing_further = (unsaturated > 0.0 and error < 0.0) or (unsaturated < 0.0 and error > 0.0)
        if c.ki > 0.0 and dt_s > 0.0 and not (saturated and pushing_further):
            limit = c.integral_limit_rad / c.ki
            self._integral = _clamp(self._integral + error * dt_s, -limit, limit)

        target = self.raceline.target_at_index(nearest)
        return LateralCommand(
            steering=steering,
            nearest_index=nearest,
            target_index=target.index,
            target_x=target.x_m,
            target_y=target.y_m,
            lookahead_m=0.0,
            cross_track_m=error,
        )

    def params(self) -> dict:
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
