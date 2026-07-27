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

    start = vehicle.get_transform()
    print("Start:")
    print(f"  x = {start.location.x:.3f}")
    print(f"  y = {start.location.y:.3f}")
    print(f"  yaw = {start.rotation.yaw:.2f}")
    print(f"  speed = {speed_mps(vehicle):.3f} m/s")

    # Strong forward test
    for i in range(50):  # about 5 seconds
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=1.00,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1,
            )
        )

        transform = vehicle.get_transform()

        print(
            f"t={0.1 * i:.1f}s | "
            f"x={transform.location.x:.3f}, "
            f"y={transform.location.y:.3f}, "
            f"yaw={transform.rotation.yaw:.2f}, "
            f"v={speed_mps(vehicle):.3f} m/s"
        )

        time.sleep(0.1)

    # Stop the car
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            reverse=False,
        )
    )

    time.sleep(1.0)

    end = vehicle.get_transform()
    print("End:")
    print(f"  x = {end.location.x:.3f}")
    print(f"  y = {end.location.y:.3f}")
    print(f"  yaw = {end.rotation.yaw:.2f}")
    print(f"  speed = {speed_mps(vehicle):.3f} m/s")


if __name__ == "__main__":
    main()
