import argparse
import math
import time
import carla


def find_car(world):
    cars = list(world.get_actors().filter("vehicle.*"))

    for car in cars:
        if car.attributes.get("role_name") == "ego":
            return car

    if len(cars) == 1:
        return cars[0]

    raise RuntimeError(f"Found {len(cars)} cars, cannot choose ego")


def smooth_angle(current, target, alpha):
    diff = (target - current + 180.0) % 360.0 - 180.0
    return current + alpha * diff


def target_camera(car, mode):
    tr = car.get_transform()
    yaw = math.radians(tr.rotation.yaw)

    fx = math.cos(yaw)
    fy = math.sin(yaw)

    if mode == "hood":
        distance = 2.0
        height = 1.0
        pitch = -8.0
        x = tr.location.x + distance * fx
        y = tr.location.y + distance * fy

    elif mode == "cockpit":
        distance = 1.2
        height = 1.3
        pitch = -5.0
        x = tr.location.x + distance * fx
        y = tr.location.y + distance * fy

    else:
        distance = 8.0
        height = 5.0
        pitch = -18.0
        x = tr.location.x - distance * fx
        y = tr.location.y - distance * fy

    return carla.Transform(
        carla.Location(x=x, y=y, z=tr.location.z + height),
        carla.Rotation(pitch=pitch, yaw=tr.rotation.yaw, roll=0.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chase", "hood", "cockpit"], default="chase")
    parser.add_argument("--alpha", type=float, default=0.30)
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()

    print("Camera running")
    print("mode =", args.mode)
    print("alpha =", args.alpha)
    print("Ctrl+C to stop")

    while True:
        try:
            car = find_car(world)
            target = target_camera(car, args.mode)
            current = spectator.get_transform()

            new_location = carla.Location(
                x=current.location.x + args.alpha * (target.location.x - current.location.x),
                y=current.location.y + args.alpha * (target.location.y - current.location.y),
                z=current.location.z + args.alpha * (target.location.z - current.location.z),
            )

            new_rotation = carla.Rotation(
                pitch=current.rotation.pitch + args.alpha * (target.rotation.pitch - current.rotation.pitch),
                yaw=smooth_angle(current.rotation.yaw, target.rotation.yaw, args.alpha),
                roll=0.0,
            )

            spectator.set_transform(carla.Transform(new_location, new_rotation))

        except RuntimeError as e:
            print("Waiting for car:", e)

        time.sleep(0.03)


if __name__ == "__main__":
    main()
