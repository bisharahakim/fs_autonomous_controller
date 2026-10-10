"""Shared longitudinal controller for the lateral-controller benchmark.

One implementation, one set of gains, used by every lateral controller, so a
difference in lap time or tracking error can be attributed to steering alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityPIDConfig:
    # Defaults reproduce the September CARLA tuning (commit d3e1319):
    #   throttle = 0.50 * err + 0.04 * max(a_ref, 0), capped at 0.75
    #   brake    = 0.06 * |err|,                      capped at 0.80
    # plus a small, clamped integral term. Set ki=0 for the exact old mapping.
    kp: float = 0.50
    ki: float = 0.05
    kd: float = 0.0
    kff_accel: float = 0.04
    integral_limit: float = 0.15   # largest actuator fraction the I term may add
    brake_gain: float = 0.12       # brake = -u * brake_gain  (0.50 * 0.12 = 0.06)
    max_throttle: float = 0.75
    max_brake: float = 0.80


@dataclass(frozen=True)
class LongitudinalCommand:
    throttle: float
    brake: float
    raw: float
    speed_error: float


class VelocityPID:
    def __init__(self, config: VelocityPIDConfig | None = None) -> None:
        self.config = config or VelocityPIDConfig()
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error: float | None = None

    def update(
        self,
        target_speed: float,
        target_accel: float,
        speed: float,
        dt_s: float,
    ) -> LongitudinalCommand:
        c = self.config
        error = target_speed - speed

        derivative = 0.0
        if self._prev_error is not None and dt_s > 0.0:
            derivative = (error - self._prev_error) / dt_s
        self._prev_error = error

        raw = (
            c.kp * error
            + c.ki * self._integral
            + c.kd * derivative
            + c.kff_accel * max(target_accel, 0.0)
        )

        if raw >= 0.0:
            throttle = min(raw, c.max_throttle)
            brake = 0.0
            saturated = raw > c.max_throttle
        else:
            throttle = 0.0
            brake = min(-raw * c.brake_gain, c.max_brake)
            saturated = -raw * c.brake_gain > c.max_brake

        # Anti-windup: the car is power-limited for long stretches (throttle
        # pinned at max), so only integrate when the actuator can still respond.
        pushing_further = (raw > 0.0 and error > 0.0) or (raw < 0.0 and error < 0.0)
        if c.ki > 0.0 and dt_s > 0.0 and not (saturated and pushing_further):
            limit = c.integral_limit / c.ki
            self._integral = min(max(self._integral + error * dt_s, -limit), limit)

        return LongitudinalCommand(throttle=throttle, brake=brake, raw=raw, speed_error=error)

    def params(self) -> dict:
        return {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}
