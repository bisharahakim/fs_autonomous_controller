from __future__ import annotations

import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import carla
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fs_controller.raceline import Raceline
from fs_controller.rapp import RAPP, RAPPConfig


RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_raceline(path: Path) -> Raceline:
    data = np.loadtxt(path, delimiter=";", comments="#")

    return Raceline(
        s_m=data[:, 0],
        x_m=data[:, 1],
        y_m=data[:, 2],
        psi_rad=data[:, 3],
        kappa_radpm=data[:, 4],
        v_target_mps=data[:, 5],
        a_target_mps2=data[:, 6],
    )

def speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def carla_yaw_deg_from_rapp_yaw_rad(psi_rad: float) -> float:
    # We mirror the track in CARLA with y -> -y.
    # Original conversion was: CARLA yaw = psi + 90 deg.
    # Mirroring Y flips the yaw sign.
    return -math.degrees(psi_rad + math.pi / 2.0)


def rapp_yaw_rad_from_carla_yaw_deg(yaw_deg: float) -> float:
    # RAPP.compute_control expects the vehicle heading angle in track coordinates.
    # Since CARLA is mirrored with y -> -y, track heading = -CARLA heading.
    return -math.radians(yaw_deg)

def get_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    bps = list(world.get_blueprint_library().filter("vehicle.*"))

    preferred = [
        "vehicle.ue4.audi.tt",
        "vehicle.mini.cooper",
        "vehicle.taxi.ford",
    ]

    for wanted in preferred:
        for bp in bps:
            if bp.id == wanted:
                return bp

    if not bps:
        raise RuntimeError("No vehicle blueprints found.")

    return bps[0]


def destroy_old_ego(world: carla.World) -> None:
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "ego":
            print("Destroying old ego vehicle:", actor.id)
            actor.destroy()
            time.sleep(0.2)


def spawn_ego_at_raceline_start(world: carla.World, raceline: Raceline) -> carla.Vehicle:
    start_idx = 0

    # RAPP track coordinates
    track_x = float(raceline.x_m[start_idx])
    track_y = float(raceline.y_m[start_idx])
    psi = float(raceline.psi_rad[start_idx])

    # CARLA coordinates: mirror Y
    x = track_x
    y = -track_y
    yaw_deg = carla_yaw_deg_from_rapp_yaw_rad(psi)

    destroy_old_ego(world)

    bp = get_vehicle_blueprint(world)
    bp.set_attribute("role_name", "ego")

    spawn_transform = carla.Transform(
        carla.Location(x=x, y=y, z=1.0),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )

    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    print("Spawned ego vehicle:", bp.id, "actor id:", vehicle.id)
    print(f"Spawn CARLA: x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.2f} deg")
    print(f"Spawn RAPP:  x={track_x:.3f}, y={track_y:.3f}, psi={psi:.3f} rad")

    # Let the car settle, then force it flat.
    time.sleep(1.0)
    flat_transform = vehicle.get_transform()
    flat_transform.rotation.pitch = 0.0
    flat_transform.rotation.roll = 0.0
    flat_transform.rotation.yaw = yaw_deg
    flat_transform.location.z += 0.05
    vehicle.set_transform(flat_transform)
    time.sleep(0.3)

    return vehicle

def set_chase_camera(world: carla.World, vehicle: carla.Vehicle) -> None:
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)

    # Camera behind and above the vehicle
    cam_x = transform.location.x - 8.0 * math.cos(yaw)
    cam_y = transform.location.y - 8.0 * math.sin(yaw)

    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=5.0),
            carla.Rotation(
                pitch=-18.0,
                yaw=transform.rotation.yaw,
                roll=0.0,
            ),
        )
    )


def translate_rapp_to_carla_control(
    cmd,
    current_speed: float,
    max_steer_rad: float,
    max_throttle: float,
) -> carla.VehicleControl:
    # RAPP steering is radians. CARLA steer is normalized [-1, 1].
    steer_norm = clamp(-cmd.steering / max_steer_rad, -1.0, 1.0)

    # Simple first longitudinal controller for M3.
    # We keep throttle limited so the first smoke test is not violent.
    speed_error = cmd.target_speed - current_speed

    if speed_error >= 0.0:
        throttle = clamp(0.06 * speed_error + 0.04 * max(cmd.target_accel, 0.0), 0.0, max_throttle)
        brake = 0.0
    else:
        throttle = 0.0
        brake = clamp(-0.15 * speed_error, 0.0, 0.8)

    return carla.VehicleControl(
        throttle=throttle,
        steer=steer_norm,
        brake=brake,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
        gear=1,
    )


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print("Connected to CARLA")
    print("Client version:", client.get_client_version())
    print("Server version:", client.get_server_version())

    raceline = load_raceline(RACELINE_PATH)
    print("Loaded raceline points:", len(raceline.s_m))

    config = RAPPConfig()
    rapp = RAPP(raceline, config)

    vehicle = spawn_ego_at_raceline_start(world, raceline)

    out_dir = PROJECT_ROOT / "outputs" / "carla" / f"rapp_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.csv"

    print("Logging to:", log_path)

    max_time_s = 30.0
    dt_s = 0.05
    max_throttle = 0.45

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_s",
            "x_m",
            "y_m",
            "yaw_carla_deg",
            "yaw_rapp_rad",
            "v_mps",
            "steering_rad",
            "steer_norm",
            "target_speed_mps",
            "target_accel_mps2",
            "throttle",
            "brake",
            "nearest_index",
            "target_index",
            "target_x",
            "target_y",
        ])

        start_time = time.time()
        step = 0

        while True:
            now = time.time()
            t = now - start_time
            if t > max_time_s:
                break

            transform = vehicle.get_transform()

            carla_x = float(transform.location.x)
            carla_y = float(transform.location.y)
            yaw_carla_deg = float(transform.rotation.yaw)

            # Convert CARLA mirrored coordinates back to RAPP track coordinates.
            x = carla_x
            y = -carla_y

            yaw_rapp_rad = rapp_yaw_rad_from_carla_yaw_deg(yaw_carla_deg)
            v = speed_mps(vehicle)

            cmd = rapp.compute_control(x, y, yaw_rapp_rad, v)
            control = translate_rapp_to_carla_control(
                cmd=cmd,
                current_speed=v,
                max_steer_rad=config.max_steer_rad,
                max_throttle=max_throttle,
            )

            vehicle.apply_control(control)

            writer.writerow([
                f"{t:.3f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{yaw_carla_deg:.6f}",
                f"{yaw_rapp_rad:.6f}",
                f"{v:.6f}",
                f"{cmd.steering:.6f}",
                f"{control.steer:.6f}",
                f"{cmd.target_speed:.6f}",
                f"{cmd.target_accel:.6f}",
                f"{control.throttle:.6f}",
                f"{control.brake:.6f}",
                cmd.nearest_index,
                cmd.target_index,
                f"{cmd.target_x:.6f}",
                f"{cmd.target_y:.6f}",
            ])

            if step % 10 == 0:
                print(
                    f"t={t:5.2f}s | "
                    f"x={x:7.2f}, y={y:7.2f}, "
                    f"v={v:5.2f}, "
                    f"steer={cmd.steering:+.3f} rad, "
                    f"steer_norm={control.steer:+.3f}, "
                    f"thr={control.throttle:.2f}, brk={control.brake:.2f}, "
                    f"target_idx={cmd.target_index}"
                )
                # set_chase_camera(world, vehicle)  # disabled: custom level spectator may throw

            step += 1
            time.sleep(dt_s)

    vehicle.apply_control(
        carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    )

    time.sleep(1.0)

    end = vehicle.get_transform()
    print("Done.")
    print(f"Final position: x={end.location.x:.3f}, y={end.location.y:.3f}, yaw={end.rotation.yaw:.2f}")
    print("Log saved to:", log_path)


if __name__ == "__main__":
    main()
