from __future__ import annotations

import sys
from pathlib import Path
from math import cos, pi, sin, tan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fs_controller import (
    ControllerStack,
    ControllerStackConfig,
    MAX_SPEED_30_KMH_MPS,
    PIConfig,
    PathPoint,
    PowertrainConfig,
    PowertrainModel,
    PurePursuitConfig,
    VehicleState,
    apply_friction_circle,
    requested_acceleration_from_actuators,
)


def make_track() -> list[PathPoint]:
    points: list[PathPoint] = []
    radius = 20.0
    for i in range(220):
        theta = 1.55 * pi * i / 219
        x = radius * sin(theta)
        y = radius * (1.0 - cos(theta))
        speed = 7.0 if i < 120 else 4.5
        points.append(PathPoint(x=x, y=y, speed=speed))
    return points


def main() -> None:
    wheelbase = 1.55
    controller = ControllerStack(
        ControllerStackConfig(
            pure_pursuit=PurePursuitConfig(
                wheelbase_m=wheelbase,
                min_lookahead_m=2.5,
                lookahead_gain_s=0.35,
                max_steer_rad=0.5,
                default_target_speed_mps=5.0,
            ),
            speed_pi=PIConfig(
                kp=0.35,
                ki=0.12,
                output_min=-1.0,
                output_max=1.0,
                integrator_min=-3.0,
                integrator_max=3.0,
            ),
            brake_gain=1.3,
        )
    )

    path = make_track()
    state = VehicleState(x=0.0, y=-1.0, yaw=0.0, speed=0.0)
    powertrain = PowertrainModel()
    vehicle_config = PowertrainConfig()
    dt = 0.02

    for step in range(900):
        command = controller.update(state, path, dt)

        requested_accel = requested_acceleration_from_actuators(
            command.throttle,
            command.brake,
            throttle_accel_mps2=3.0,
            brake_decel_mps2=5.0,
        )
        powertrain_accel = powertrain.actual_acceleration(state.speed, requested_accel)
        speed_before_lateral = max(0.0, state.speed + powertrain_accel * dt)
        commanded_lateral_accel = speed_before_lateral * speed_before_lateral * tan(command.steering_rad) / wheelbase
        actual_long_force, actual_lateral_force, _ = apply_friction_circle(
            vehicle_config.mass_kg * powertrain_accel,
            vehicle_config.mass_kg * commanded_lateral_accel,
            vehicle_config.mass_kg,
        )
        actual_accel = actual_long_force / vehicle_config.mass_kg
        achievable_lateral_accel = actual_lateral_force / vehicle_config.mass_kg
        speed = min(MAX_SPEED_30_KMH_MPS, max(0.0, state.speed + actual_accel * dt))
        yaw_rate = achievable_lateral_accel / max(speed, 1e-3)
        yaw = state.yaw + yaw_rate * dt
        x = state.x + speed * cos(yaw) * dt
        y = state.y + speed * sin(yaw) * dt
        state = VehicleState(x=x, y=y, yaw=yaw, speed=speed)

        if step % 50 == 0 or command.finished:
            print(
                f"t={step * dt:5.2f}s "
                f"x={state.x:6.2f} y={state.y:6.2f} "
                f"v={state.speed:4.2f} "
                f"steer={command.steering_rad:5.2f} "
                f"thr={command.throttle:4.2f} brk={command.brake:4.2f} "
                f"target={command.target_index:03d}"
            )

        if command.finished:
            print("Finished path.")
            break


if __name__ == "__main__":
    main()
