from __future__ import annotations

import math
import time

import carla


def find_ego_vehicle(world: carla.World) -> carla.Vehicle:
    vehicles = list(world.get_actors().filter("vehicle.*"))

    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "ego":
            return vehicle

    if len(vehicles) == 1:
        return vehicles[0]

    raise RuntimeError(f"Could not find ego vehicle. Found {len(vehicles)} vehicles.")


def speed_mps(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicle = find_ego_vehicle(world)

    print("Found ego vehicle:", vehicle.id, vehicle.type_id)

    for i in range(50):
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.45,
                steer=0.50,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1,
            )
        )

        t = vehicle.get_transform()
        print(
            f"t={i*0.1:.1f}s | "
            f"x={t.location.x:.3f}, "
            f"y={t.location.y:.3f}, "
            f"yaw={t.rotation.yaw:.2f}, "
            f"v={speed_mps(vehicle):.2f}"
        )

        time.sleep(0.1)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
    time.sleep(1.0)


if __name__ == "__main__":
    main()
