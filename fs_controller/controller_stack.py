from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .low_level import PIConfig, SpeedPIController
from .pure_pursuit import PurePursuitConfig, PurePursuitController
from .models import PathPoint, VehicleState


@dataclass(frozen=True)
class ControllerStackConfig:
    pure_pursuit: PurePursuitConfig
    speed_pi: PIConfig
    brake_gain: float = 1.0


@dataclass(frozen=True)
class ControlCommand:
    steering_rad: float
    throttle: float
    brake: float
    target_speed_mps: float
    target_index: int
    finished: bool


class ControllerStack:
    """High-level pure pursuit plus low-level PI speed controller."""

    def __init__(self, config: ControllerStackConfig) -> None:
        self.path_follower = PurePursuitController(config.pure_pursuit)
        self.speed_controller = SpeedPIController(config.speed_pi, config.brake_gain)

    def reset(self) -> None:
        self.path_follower.reset()
        self.speed_controller.reset()

    def update(
        self,
        state: VehicleState,
        path: Sequence[PathPoint],
        dt_s: float,
    ) -> ControlCommand:
        high_level = self.path_follower.compute(state, path)
        low_level = self.speed_controller.update(
            target_speed_mps=high_level.target_speed_mps,
            measured_speed_mps=state.speed,
            dt_s=dt_s,
        )

        return ControlCommand(
            steering_rad=high_level.steering_rad,
            throttle=low_level.throttle,
            brake=low_level.brake,
            target_speed_mps=high_level.target_speed_mps,
            target_index=high_level.target_index,
            finished=high_level.finished,
        )
