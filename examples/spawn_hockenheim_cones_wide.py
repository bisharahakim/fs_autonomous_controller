from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import carla


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from visualization.track_registry import get_track


def widen_cones(left, right, desired_width_m: float):
    widened_left = []
    widened_right = []

    for l, r in zip(left, right):
        mid_x = 0.5 * (l.x + r.x)
        mid_y = 0.5 * (l.y + r.y)

        vec_l_x = l.x - mid_x
        vec_l_y = l.y - mid_y
        vec_r_x = r.x - mid_x
        vec_r_y = r.y - mid_y

        current_half_width = math.sqrt(vec_l_x**2 + vec_l_y**2)
        if current_half_width < 1e-6:
            scale = 1.0
        else:
            scale = (desired_width_m / 2.0) / current_half_width

        widened_left.append(
            replace(
                l,
                x=mid_x + scale * vec_l_x,
                y=mid_y + scale * vec_l_y,
            )
        )

        widened_right.append(
            replace(
                r,
                x=mid_x + scale * vec_r_x,
                y=mid_y + scale * vec_r_y,
            )
        )

    return widened_left, widened_right


def get_blueprint(world: carla.World, bp_id: str):
    bps = world.get_blueprint_library().filter(bp_id)
    if len(bps) == 0:
        raise RuntimeError(f"Could not find blueprint: {bp_id}")
    return bps[0]


def point_to_location(point, z: float, flip_y: bool) -> carla.Location:
    y = -point.y if flip_y else point.y
    return carla.Location(x=float(point.x), y=float(y), z=float(z))


def spawn_cones(world, points, bp, z: float, flip_y: bool):
    actors = []

    for p in points:
        transform = carla.Transform(
            point_to_location(p, z=z, flip_y=flip_y),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            actors.append(actor)

    return actors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=float, default=5.0, help="Desired total track width in meters")
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--track", default="hockenheim_fsg",
                        help="Track to lay out: hockenheim_fsg, or a name from inputs/tracks/ (e.g. brandshatch).")
    parser.add_argument("--clear", action="store_true", help="Remove all cones already in the world first.")
    args = parser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    if args.clear:
        old = [a for a in world.get_actors() if "wemas" in a.type_id or "cone" in a.type_id]
        for actor in old:
            actor.destroy()
        print(f"Removed {len(old)} existing cones.")
        time.sleep(0.5)

    left, right = get_track(args.track).build_cones()
    left, right = widen_cones(left, right, desired_width_m=args.width)

    left_bp = get_blueprint(world, "static.prop.wemas_small_blue")
    right_bp = get_blueprint(world, "static.prop.wemas_small_yellow")

    print(f"Spawning widened {args.track} cones")
    print(f"Desired width: {args.width:.2f} m")
    print(f"Left cones: {len(left)}")
    print(f"Right cones: {len(right)}")
    print(f"flip_y: {args.flip_y}")

    actors = []
    actors += spawn_cones(world, left, left_bp, z=args.z, flip_y=args.flip_y)
    actors += spawn_cones(world, right, right_bp, z=args.z, flip_y=args.flip_y)

    print(f"Actually spawned actors: {len(actors)}")

    if args.keep:
        print("Keeping cones in the world.")
        return

    input("Press Enter to destroy cones...")
    for actor in actors:
        actor.destroy()
    time.sleep(0.5)
    print("Destroyed cones.")


if __name__ == "__main__":
    main()
