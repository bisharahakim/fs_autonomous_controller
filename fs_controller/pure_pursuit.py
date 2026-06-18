from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, hypot, sin
from typing import Sequence

try:
    from .models import PathPoint, VehicleState
except ImportError:
    from models import PathPoint, VehicleState


@dataclass(frozen=True)
class PurePursuitConfig:
    wheelbase_m: float = 2.0
    min_lookahead_m: float = 2.5
    lookahead_gain_s: float = 0.18
    max_lookahead_m: float = 10.0
    max_steer_rad: float = 0.3
    default_target_speed_mps: float = 6.0
    finish_tolerance_m: float = 1.0


@dataclass(frozen=True)
class PurePursuitOutput:
    steering_rad: float
    target_speed_mps: float
    target_index: int
    lookahead_m: float
    finished: bool


class PurePursuitController:
    """Geometric path follower for a bicycle-model vehicle.

    The controller selects a lookahead point on the path, transforms it into
    the vehicle frame, then computes steering curvature toward that point.
    """

    def __init__(self, config: PurePursuitConfig) -> None:
        if config.wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be positive")
        if config.min_lookahead_m <= 0.0:
            raise ValueError("min_lookahead_m must be positive")
        if config.max_lookahead_m < config.min_lookahead_m:
            raise ValueError("max_lookahead_m must be >= min_lookahead_m")
        self.config = config
        self._last_target_index = 0

    def reset(self) -> None:
        self._last_target_index = 0

    def compute(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
    ) -> PurePursuitOutput:
        if not path:
            return PurePursuitOutput(
                steering_rad=0.0,
                target_speed_mps=0.0,
                target_index=0,
                lookahead_m=self.config.min_lookahead_m,
                finished=True,
            )

        nearest_index = self._nearest_index(state, path)
        lookahead = self._lookahead_distance(state.speed)
        target_index = self._target_index_from(nearest_index, state, path, lookahead)
        self._last_target_index = target_index

        target = path[target_index]
        # Express the target point in the vehicle frame (forward x, left y).
        local_x, local_y = self._to_vehicle_frame(state, target)

        if local_x <= 0.0:
            # Do not steer toward points behind the vehicle.
            steering = 0.0
        else:
            # Pure pursuit curvature: kappa = 2*y / Ld^2, then bicycle-model steer.
            distance_sq = max(local_x * local_x + local_y * local_y, 1e-6)
            curvature = 2.0 * local_y / distance_sq
            steering = atan(self.config.wheelbase_m * curvature)

        # Respect steering actuator limits.
        steering = _clamp(
            steering,
            -self.config.max_steer_rad,
            self.config.max_steer_rad,
        )
        # Use per-point speed if provided, otherwise fall back to cruise speed.
        target_speed = (
            target.speed
            if target.speed is not None
            else self.config.default_target_speed_mps
        )
        # Finish only when at final path index and physically close to endpoint.
        finished = (
            target_index == len(path) - 1
            and hypot(state.x - path[-1].x, state.y - path[-1].y)
            <= self.config.finish_tolerance_m
        )

        return PurePursuitOutput(
            steering_rad=steering,
            target_speed_mps=max(0.0, target_speed),
            target_index=target_index,
            lookahead_m=lookahead,
            finished=finished,
        )

    def _nearest_index(self, state: VehicleState, path: Sequence[PathPoint]) -> int:
        start = min(self._last_target_index, len(path) - 1)
        best_index = start
        best_dist = float("inf")

        for index in range(start, len(path)):
            point = path[index]
            dist = (state.x - point.x) ** 2 + (state.y - point.y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_index = index

        return best_index

    def _target_index_from(
        self,
        nearest_index: int,
        state: VehicleState,
        path: Sequence[PathPoint],
        lookahead: float,
    ) -> int:
        target_index = nearest_index
        for index in range(nearest_index, len(path)):
            point = path[index]
            if hypot(point.x - state.x, point.y - state.y) >= lookahead:
                target_index = index
                break
        else:
            target_index = len(path) - 1
        return target_index

    def _lookahead_distance(self, speed_mps: float) -> float:
        return _clamp(
            self.config.min_lookahead_m + self.config.lookahead_gain_s * speed_mps,
            self.config.min_lookahead_m,
            self.config.max_lookahead_m,
        )

    @staticmethod
    def _to_vehicle_frame(
        state: VehicleState,
        point: PathPoint,
    ) -> tuple[float, float]:
        dx = point.x - state.x
        dy = point.y - state.y
        c = cos(state.yaw)
        s = sin(state.yaw)
        return c * dx + s * dy, -s * dx + c * dy


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
