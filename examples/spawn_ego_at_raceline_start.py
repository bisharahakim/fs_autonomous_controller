from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import carla
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fs_controller.raceline import Raceline


RACELINE_PATH = PROJECT_ROOT / "inputs/racelines/hockenheim_fsg_benjamin24_safe.csv"


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


def set_camera(world: carla.World, x: float, y: float, yaw_deg: float) -> None:
    spectator = world.get_spectator()

    spectator.set_transform(
        carla.Transform(
            carla.Location(x=x - 8.0, y=y + 5.0, z=4.0),
            carla.Rotation(pitch=-18.0, yaw=yaw_deg + 25.0, roll=0.0),
        )
    )


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    print("Connected to CARLA")
    print("Client version:", client.get_client_version())
    print("Server version:", client.get_server_version())

    raceline = load_raceline(RACELINE_PATH)

    start_idx = 0

    x = float(raceline.x_m[start_idx])
    y = float(raceline.y_m[start_idx])
    # Use raceline yaw only for direction.
    psi = float(raceline.psi_rad[start_idx])

    # Raceline psi is approximately CARLA/world heading - pi/2.
    yaw_deg = math.degrees(psi + math.pi / 2.0)

    print("Raceline start:")
    print(f"  x = {x:.3f}")
    print(f"  y = {y:.3f}")
    print(f"  psi = {psi:.3f} rad")
    print(f"  CARLA yaw = {yaw_deg:.2f} deg")

    destroy_old_ego(world)

    bp = get_vehicle_blueprint(world)
    bp.set_attribute("role_name", "ego")

    spawn_transform = carla.Transform(
        carla.Location(x=x, y=y, z=1.0),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )

    vehicle = world.try_spawn_actor(bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle. Try increasing z or clearing the area.")

    print("Spawned ego vehicle:", bp.id, "actor id:", vehicle.id)

    # Let the car settle onto the plane.
    # In the custom empty Unreal level, world.get_settings() may throw,
    # so keep this simple for now.
    time.sleep(1.0)


    # Force the vehicle flat after settling.
    flat_transform = vehicle.get_transform()
    flat_transform.rotation.pitch = 0.0
    flat_transform.rotation.roll = 0.0
    flat_transform.rotation.yaw = yaw_deg
    flat_transform.location.z += 0.05
    vehicle.set_transform(flat_transform)
    time.sleep(0.3)

    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    print("Vehicle transform after settling:")
    print(f"  x = {transform.location.x:.3f}")
    print(f"  y = {transform.location.y:.3f}")
    print(f"  z = {transform.location.z:.3f}")
    print(f"  yaw = {transform.rotation.yaw:.2f} deg")
    print(f"  speed = {speed:.3f} m/s")

    set_camera(world, x, y, yaw_deg)

    print("Ego vehicle left in world. Stop Play or rerun script to remove/replace it.")


if __name__ == "__main__":
    main()
