from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PIConfig:
    kp: float
    ki: float
    output_min: float
    output_max: float
    integrator_min: float
    integrator_max: float


class PIController:
    """PI controller with integrator clamping anti-windup."""

    def __init__(self, config: PIConfig) -> None:
        if config.output_max < config.output_min:
            raise ValueError("output_max must be >= output_min")
        if config.integrator_max < config.integrator_min:
            raise ValueError("integrator_max must be >= integrator_min")
        self.config = config
        self.integrator = 0.0

    def reset(self) -> None:
        self.integrator = 0.0

    def update(self, error: float, dt_s: float) -> float:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        self.integrator += error * dt_s
        self.integrator = _clamp(
            self.integrator,
            self.config.integrator_min,
            self.config.integrator_max,
        )
        output = self.config.kp * error + self.config.ki * self.integrator
        return _clamp(output, self.config.output_min, self.config.output_max)


@dataclass(frozen=True)
class SpeedActuatorCommand:
    throttle: float
    brake: float
    raw_accel_command: float


class SpeedPIController:
    """Low-level speed controller.

    Positive PI output maps to throttle. Negative output maps to brake.
    The values are normalized to [0, 1] so a platform-specific layer can turn
    them into inverter torque, hydraulic brake pressure, or CAN messages.
    """

    def __init__(self, config: PIConfig, brake_gain: float = 1.0) -> None:
        if brake_gain <= 0.0:
            raise ValueError("brake_gain must be positive")
        self.pi = PIController(config)
        self.brake_gain = brake_gain

    def reset(self) -> None:
        self.pi.reset()

    def update(
        self,
        target_speed_mps: float,
        measured_speed_mps: float,
        dt_s: float,
    ) -> SpeedActuatorCommand:
        speed_error = target_speed_mps - measured_speed_mps
        accel_command = self.pi.update(speed_error, dt_s)

        throttle = _clamp(accel_command, 0.0, 1.0)
        brake = _clamp(-accel_command * self.brake_gain, 0.0, 1.0)

        return SpeedActuatorCommand(
            throttle=throttle,
            brake=brake,
            raw_accel_command=accel_command,
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
