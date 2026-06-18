from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class PowertrainConfig:
    mass_kg: float = 250.0
    wheel_radius_m: float = 0.198
    gear_ratio: float = 3.6
    motor_torque_max_nm: float = 135.0
    motor_omega_max_radps: float = 4290.0 * pi / 30.0
    motor_efficiency: float = 0.92
    battery_power_max_w: float = 80_000.0 * 0.92
    rho_air_kgpm3: float = 1.225
    cd0: float = 0.1
    cl: float = 3.0
    frontal_area_m2: float = 1.0
    brake_decel_limit_mps2: float = 1.5 * 9.81

    @property
    def cd(self) -> float:
        return self.cd0 + 0.1 * self.cl**2


class PowertrainModel:
    def __init__(self, config: PowertrainConfig | None = None) -> None:
        self.config = config or PowertrainConfig()

    def actual_acceleration(self, speed_mps: float, requested_accel_mps2: float) -> float:
        cfg = self.config
        speed = max(0.0, speed_mps)
        a_req = max(requested_accel_mps2, -cfg.brake_decel_limit_mps2)

        omega_wheel = speed / cfg.wheel_radius_m
        omega_motor = omega_wheel * cfg.gear_ratio

        if omega_motor < 1e-3:
            motor_torque = cfg.motor_torque_max_nm
        elif omega_motor > cfg.motor_omega_max_radps:
            motor_torque = 0.0
        else:
            motor_torque = min(cfg.motor_torque_max_nm, cfg.battery_power_max_w / omega_motor)

        drive_force_max = motor_torque * cfg.gear_ratio / cfg.wheel_radius_m
        drag_force = 0.5 * cfg.rho_air_kgpm3 * cfg.cd * cfg.frontal_area_m2 * speed**2
        requested_force = cfg.mass_kg * a_req

        if a_req >= 0.0:
            drive_force = min(requested_force + drag_force, drive_force_max)
        else:
            drive_force = requested_force + drag_force

        return (drive_force - drag_force) / cfg.mass_kg


def requested_acceleration_from_actuators(
    throttle: float,
    brake: float,
    throttle_accel_mps2: float = 5.0,
    brake_decel_mps2: float = 6.0,
) -> float:
    return throttle_accel_mps2 * throttle - brake_decel_mps2 * brake
